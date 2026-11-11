//! YUV → premultiplied BGRA (video is opaque, so it's plain BGRA with A=255).
//!
//! Shared by all CPU paths: dav1d/rav1d (I420/I422/I444/I400, 8–12 bit),
//! openh264 (I420), libvpx (I420/I444), NVDEC and VA-API with a copy to system
//! memory (NV12/P016). Fixed-point integer math (14 bit).
//!
//! A decoder keeps one [`Converter`] for its frames:
//! - samples above 8 bit are first cut down to 8 bit, so one set of kernels does the rest;
//! - on x86-64 with AVX2 the rows are converted 8 pixels at a time, with the same integer
//!   math as the scalar code (the output is identical byte for byte);
//! - a frame larger than it is shown is halved first (2x2 averages), as often as it still
//!   covers the size asked for: fewer pixels to convert and to upload as a texture;
//! - output buffers come from a small pool instead of a fresh allocation per frame.

use std::sync::Arc;

use crate::frame::CpuFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020,
}

impl Matrix {
    /// Heuristic when the stream doesn't say: HD and up is 709, everything else 601.
    pub fn guess(height: u32) -> Self {
        if height >= 720 {
            Matrix::Bt709
        } else {
            Matrix::Bt601
        }
    }

    /// Coefficients (Kr, Kb).
    fn kr_kb(self) -> (f64, f64) {
        match self {
            Matrix::Bt601 => (0.299, 0.114),
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt2020 => (0.2627, 0.0593),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chroma {
    /// 4:2:0: one chroma sample per 2x2 pixels
    Sub420,
    /// 4:2:2: one chroma sample per 2x1
    Sub422,
    /// 4:4:4
    Full,
    /// luma only
    Mono,
}

impl Chroma {
    fn shift(self) -> (u32, u32) {
        match self {
            Chroma::Sub420 => (1, 1),
            Chroma::Sub422 => (1, 0),
            Chroma::Full | Chroma::Mono => (0, 0),
        }
    }
}

/// Plane: bytes + row stride in bytes.
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
}

/// Chroma plane layout.
#[derive(Clone, Copy)]
pub enum ChromaPlanes<'a> {
    /// separate U and V planes (I420/I422/I444)
    Planar {
        u: Plane<'a>,
        v: Plane<'a>,
    },
    /// interleaved UV (NV12)
    Interleaved(Plane<'a>),
    None,
}

pub struct YuvImage<'a> {
    pub width: u32,
    pub height: u32,
    /// 8, 10 or 12. Above 8, samples are u16 little-endian.
    pub bit_depth: u32,
    pub chroma: Chroma,
    pub matrix: Matrix,
    pub full_range: bool,
    pub y: Plane<'a>,
    pub uv: ChromaPlanes<'a>,
}

const FIX: i32 = 14;

/// Frame size from which the scalar conversion goes multithreaded (~1 MP: 1280x800 and up).
const PARALLEL_FROM_PIXELS: usize = 1_000_000;
/// With AVX2 one thread converts 4K in about 2 ms: only frames beyond 5K are split.
const PARALLEL_FROM_PIXELS_SIMD: usize = 16_000_000;
/// No more threads than this: the compositor and decoder need CPU too.
const MAX_THREADS: usize = 4;
/// Output buffers a converter keeps for reuse: the frame on screen, the one waiting to be
/// picked up, the ones a decoder returned in one go.
const POOL_KEEP: usize = 6;
/// A frame is halved at most this many times.
const MAX_HALVINGS: u32 = 6;

/// Fixed-point coefficients for 8-bit values.
struct Coefs {
    y_off: i32,
    y_mul: i32,
    rv: i32,
    gu: i32,
    gv: i32,
    bu: i32,
}

impl Coefs {
    fn new(matrix: Matrix, full: bool) -> Self {
        let (kr, kb) = matrix.kr_kb();
        let kg = 1.0 - kr - kb;
        // luma and chroma scale for limited range
        let (ys, cs, yo) = if full {
            (1.0, 1.0, 0)
        } else {
            (255.0 / 219.0, 255.0 / 224.0, 16)
        };
        let one = f64::from(1 << FIX);
        let r = |v: f64| (v * one).round() as i32;
        Self {
            y_off: yo,
            y_mul: r(ys),
            rv: r(2.0 * (1.0 - kr) * cs),
            gu: r(2.0 * (1.0 - kb) * kb / kg * cs),
            gv: r(2.0 * (1.0 - kr) * kr / kg * cs),
            bu: r(2.0 * (1.0 - kb) * cs),
        }
    }

    #[inline(always)]
    fn px(&self, y: i32, u: i32, v: i32) -> [u8; 4] {
        let y = (y - self.y_off) * self.y_mul;
        let u = u - 128;
        let v = v - 128;
        let half = 1 << (FIX - 1);
        let clamp = |x: i32| ((x + half) >> FIX).clamp(0, 255) as u8;
        let r = clamp(y + self.rv * v);
        let g = clamp(y - self.gu * u - self.gv * v);
        let b = clamp(y + self.bu * u);
        [b, g, r, 255]
    }
}

/// Checks that the planes actually fit the image (so we return None instead of panicking).
fn plane_ok(p: &Plane<'_>, w: usize, h: usize, bytes_per: usize) -> bool {
    h == 0 || (p.stride >= w * bytes_per && p.data.len() >= p.stride * (h - 1) + w * bytes_per)
}

/// The chroma samples of one row, 8 bit.
#[derive(Clone, Copy)]
enum RowChroma<'a> {
    Planar(&'a [u8], &'a [u8]),
    /// U and V interleaved
    Interleaved(&'a [u8]),
    None,
}

/// A picture with 8-bit samples, as the kernels take it. `sub_x` / `sub_y`: one chroma
/// sample stands for two pixels in that direction.
struct Image8<'a> {
    w: usize,
    h: usize,
    y: Plane<'a>,
    uv: ChromaPlanes<'a>,
    sub_x: bool,
    sub_y: bool,
}

impl Image8<'_> {
    fn luma_row(&self, row: usize) -> &[u8] {
        &self.y.data[row * self.y.stride..][..self.w]
    }

    fn chroma_row(&self, row: usize) -> RowChroma<'_> {
        let cw = if self.sub_x {
            self.w.div_ceil(2)
        } else {
            self.w
        };
        let cy = if self.sub_y { row >> 1 } else { row };
        match self.uv {
            ChromaPlanes::Planar { u, v } => RowChroma::Planar(
                &u.data[cy * u.stride..][..cw],
                &v.data[cy * v.stride..][..cw],
            ),
            ChromaPlanes::Interleaved(p) => {
                RowChroma::Interleaved(&p.data[cy * p.stride..][..cw * 2])
            }
            ChromaPlanes::None => RowChroma::None,
        }
    }
}

/// Pixels `from..` of one row: the whole row without AVX2, what is left after its 8-pixel
/// blocks with it (so `from` is a multiple of 8: a chroma sample starts there too).
fn row_scalar(c: &Coefs, y: &[u8], uv: RowChroma<'_>, sub_x: bool, dst: &mut [u8], from: usize) {
    let px = |y: &u8, u: &u8, v: &u8| c.px(i32::from(*y), i32::from(*u), i32::from(*v));
    let dst = &mut dst.as_chunks_mut::<4>().0[from..];
    let y = &y[from..];
    match (uv, sub_x) {
        // one chroma sample for each pair of pixels
        (RowChroma::Planar(u, v), true) => {
            let uv = u[from / 2..].iter().zip(&v[from / 2..]);
            for ((d, y), (u, v)) in dst.chunks_mut(2).zip(y.chunks(2)).zip(uv) {
                for (d, y) in d.iter_mut().zip(y) {
                    *d = px(y, u, v);
                }
            }
        }
        (RowChroma::Interleaved(p), true) => {
            let uv = p[from..].as_chunks::<2>().0;
            for ((d, y), [u, v]) in dst.chunks_mut(2).zip(y.chunks(2)).zip(uv) {
                for (d, y) in d.iter_mut().zip(y) {
                    *d = px(y, u, v);
                }
            }
        }
        (RowChroma::Planar(u, v), false) => {
            let uv = u[from..].iter().zip(&v[from..]);
            for ((d, y), (u, v)) in dst.iter_mut().zip(y).zip(uv) {
                *d = px(y, u, v);
            }
        }
        (RowChroma::Interleaved(p), false) => {
            let uv = p[from * 2..].as_chunks::<2>().0;
            for ((d, y), [u, v]) in dst.iter_mut().zip(y).zip(uv) {
                *d = px(y, u, v);
            }
        }
        (RowChroma::None, _) => {
            for (d, y) in dst.iter_mut().zip(y) {
                *d = px(y, &128, &128);
            }
        }
    }
}

fn rows_scalar(img: &Image8<'_>, c: &Coefs, first_row: usize, band: &mut [u8]) {
    for (i, dst) in band.chunks_exact_mut(img.w * 4).enumerate() {
        let row = first_row + i;
        row_scalar(c, img.luma_row(row), img.chroma_row(row), img.sub_x, dst, 0);
    }
}

/// The same conversion 8 pixels at a time. Every product is one `pmaddwd`: a 32-bit lane
/// holds a sample (or a [u, v] pair) as 16-bit halves, the coefficients sit in the other
/// operand the same way, so the sums are the scalar code's 32-bit sums exactly.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{Coefs, FIX, Image8, RowChroma, row_scalar};

    struct Kernel {
        y_off: __m256i,
        y_mul: __m256i,
        half: __m256i,
        r: __m256i,
        g: __m256i,
        b1: __m256i,
        b2: __m256i,
        bias: __m256i,
        alpha: __m256i,
        order: __m256i,
        twice: __m256i,
    }

    /// The 16-bit pair (lo, hi) in every 32-bit lane.
    #[target_feature(enable = "avx2")]
    fn pair(lo: i32, hi: i32) -> __m256i {
        _mm256_set1_epi32((lo & 0xffff) | (hi << 16))
    }

    impl Kernel {
        #[target_feature(enable = "avx2")]
        fn new(c: &Coefs) -> Self {
            Self {
                y_off: _mm256_set1_epi32(c.y_off),
                y_mul: pair(c.y_mul, 0),
                half: _mm256_set1_epi32(1 << (FIX - 1)),
                r: pair(0, c.rv),
                g: pair(-c.gu, -c.gv),
                // bu is above 32767 for every matrix, its two halves are not
                b1: pair(c.bu / 2, 0),
                b2: pair(c.bu - c.bu / 2, 0),
                bias: _mm256_set1_epi16(128),
                alpha: _mm256_set1_epi32(255),
                // b0..3 r0..3 g0..3 a0..3 -> b g r a per pixel, in each half
                order: _mm256_setr_epi8(
                    0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15, 0, 8, 4, 12, 1, 9, 5, 13,
                    2, 10, 6, 14, 3, 11, 7, 15,
                ),
                twice: _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3),
            }
        }

        /// 8 pixels. `y`: their luma samples, one per 32-bit lane; `uv`: [u, v] per lane.
        #[target_feature(enable = "avx2")]
        fn px8(&self, y: __m256i, uv: __m256i) -> __m256i {
            let y = _mm256_madd_epi16(_mm256_sub_epi32(y, self.y_off), self.y_mul);
            let y = _mm256_add_epi32(y, self.half);
            let uv = _mm256_sub_epi16(uv, self.bias);
            let r = _mm256_add_epi32(y, _mm256_madd_epi16(uv, self.r));
            let g = _mm256_add_epi32(y, _mm256_madd_epi16(uv, self.g));
            let b = _mm256_add_epi32(y, _mm256_madd_epi16(uv, self.b1));
            let b = _mm256_add_epi32(b, _mm256_madd_epi16(uv, self.b2));
            // the two saturating packs are the scalar clamp to 0..=255
            let br = _mm256_packs_epi32(_mm256_srai_epi32(b, FIX), _mm256_srai_epi32(r, FIX));
            let ga = _mm256_packs_epi32(_mm256_srai_epi32(g, FIX), self.alpha);
            _mm256_shuffle_epi8(_mm256_packus_epi16(br, ga), self.order)
        }

        /// Each of 4 [u, v] lanes for two pixels.
        #[target_feature(enable = "avx2")]
        fn twice(&self, uv: __m128i) -> __m256i {
            _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(uv), self.twice)
        }
    }

    /// 4 samples, one per 32-bit lane.
    #[target_feature(enable = "avx2")]
    fn wide4(s: &[u8; 4]) -> __m128i {
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(*s)))
    }

    /// 8 samples, one per 32-bit lane.
    #[target_feature(enable = "avx2")]
    fn wide8(s: &[u8; 8]) -> __m256i {
        _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(i64::from_le_bytes(*s)))
    }

    #[target_feature(enable = "avx2")]
    fn store(dst: &mut [u8; 32], px: __m256i) {
        // SAFETY: an unaligned store of 32 bytes into 32 bytes.
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), px) };
    }

    /// One row of `halve` for a single-channel plane, 32 output pixels at a time. Returns
    /// how many it wrote; the rest of the row is the caller's.
    #[target_feature(enable = "avx2")]
    pub(super) fn halve_row(top: &[u8], bottom: &[u8], out: &mut [u8]) -> usize {
        let ones = _mm256_set1_epi8(1);
        let two = _mm256_set1_epi16(2);
        // the sum of each pair of neighbours in the two rows, plus 2, over 4
        let avg = |t: __m256i, b: __m256i| {
            let sum =
                _mm256_add_epi16(_mm256_maddubs_epi16(t, ones), _mm256_maddubs_epi16(b, ones));
            _mm256_srli_epi16(_mm256_add_epi16(sum, two), 2)
        };
        let rows = top
            .as_chunks::<64>()
            .0
            .iter()
            .zip(bottom.as_chunks::<64>().0);
        let blocks = out.as_chunks_mut::<32>().0;
        let done = blocks.len().min(top.len() / 64).min(bottom.len() / 64) * 32;
        for (o, (t, b)) in blocks.iter_mut().zip(rows) {
            // SAFETY: unaligned loads of 32 bytes from each half of 64 bytes.
            let (t0, t1, b0, b1) = unsafe {
                (
                    _mm256_loadu_si256(t.as_ptr().cast()),
                    _mm256_loadu_si256(t.as_ptr().add(32).cast()),
                    _mm256_loadu_si256(b.as_ptr().cast()),
                    _mm256_loadu_si256(b.as_ptr().add(32).cast()),
                )
            };
            // the pack works on each 128-bit half: put the four quarters back in order
            let px = _mm256_packus_epi16(avg(t0, b0), avg(t1, b1));
            store(o, _mm256_permute4x64_epi64(px, 0b11_01_10_00));
        }
        done
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn rows(img: &Image8<'_>, c: &Coefs, first_row: usize, band: &mut [u8]) {
        let k = Kernel::new(c);
        for (i, dst) in band.chunks_exact_mut(img.w * 4).enumerate() {
            let row = first_row + i;
            let y = img.luma_row(row);
            let uv = img.chroma_row(row);
            // every zip below ends with the luma blocks: the chroma rows hold at least as
            // many blocks (see chroma_row)
            let blocks = dst
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(y.as_chunks::<8>().0);
            match (uv, img.sub_x) {
                (RowChroma::Planar(u, v), true) => {
                    let uv = u.as_chunks::<4>().0.iter().zip(v.as_chunks::<4>().0);
                    for ((dst, y), (u, v)) in blocks.zip(uv) {
                        let uv = _mm_or_si128(wide4(u), _mm_slli_epi32(wide4(v), 16));
                        store(dst, k.px8(wide8(y), k.twice(uv)));
                    }
                }
                (RowChroma::Planar(u, v), false) => {
                    let uv = u.as_chunks::<8>().0.iter().zip(v.as_chunks::<8>().0);
                    for ((dst, y), (u, v)) in blocks.zip(uv) {
                        let uv = _mm256_or_si256(wide8(u), _mm256_slli_epi32(wide8(v), 16));
                        store(dst, k.px8(wide8(y), uv));
                    }
                }
                (RowChroma::Interleaved(p), true) => {
                    for ((dst, y), p) in blocks.zip(p.as_chunks::<8>().0) {
                        let uv = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(i64::from_le_bytes(*p)));
                        store(dst, k.px8(wide8(y), k.twice(uv)));
                    }
                }
                (RowChroma::Interleaved(p), false) => {
                    for ((dst, y), p) in blocks.zip(p.as_chunks::<16>().0) {
                        // SAFETY: an unaligned load of 16 bytes from 16 bytes.
                        let p = unsafe { _mm_loadu_si128(p.as_ptr().cast()) };
                        store(dst, k.px8(wide8(y), _mm256_cvtepu8_epi16(p)));
                    }
                }
                (RowChroma::None, _) => {
                    for (dst, y) in blocks {
                        store(dst, k.px8(wide8(y), k.bias));
                    }
                }
            }
            row_scalar(c, y, uv, img.sub_x, dst, y.len() / 8 * 8);
        }
    }
}

fn has_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn rows(img: &Image8<'_>, c: &Coefs, first_row: usize, band: &mut [u8], simd: bool) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only set after AVX2 was detected (has_simd).
        unsafe { avx2::rows(img, c, first_row, band) };
        return;
    }
    let _ = simd;
    rows_scalar(img, c, first_row, band);
}

/// `out`: `w * h * 4` bytes.
fn convert8(img: &Image8<'_>, c: &Coefs, out: &mut [u8], simd: bool) {
    let from = if simd {
        PARALLEL_FROM_PIXELS_SIMD
    } else {
        PARALLEL_FROM_PIXELS
    };
    // Rows are independent: a large frame is split into bands across threads, otherwise
    // the conversion alone eats the whole frame budget.
    let threads = if img.w * img.h >= from {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(MAX_THREADS))
    } else {
        1
    };
    if threads <= 1 {
        rows(img, c, 0, out, simd);
        return;
    }
    let rows_per = img.h.div_ceil(threads);
    std::thread::scope(|scope| {
        for (i, band) in out.chunks_mut(rows_per * img.w * 4).enumerate() {
            scope.spawn(move || rows(img, c, i * rows_per, band, simd));
        }
    });
}

/// `rows` rows of `n` 16-bit samples -> 8 bit, rows packed.
fn reduce(src: &Plane<'_>, n: usize, rows: usize, shift: u32, dst: &mut Vec<u8>) {
    dst.resize(n * rows, 0);
    if n == 0 {
        return;
    }
    for (r, out) in dst.chunks_exact_mut(n).enumerate() {
        let row = &src.data[r * src.stride..][..n * 2];
        for (o, s) in out.iter_mut().zip(row.as_chunks::<2>().0) {
            *o = (u16::from_le_bytes(*s) >> shift).min(255) as u8;
        }
    }
}

#[inline(always)]
fn avg4(a: u8, b: u8, c: u8, d: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + u16::from(c) + u16::from(d) + 2) >> 2) as u8
}

/// Half the size: every output pixel is the average of 2x2. `ch`: interleaved channels
/// per pixel (1, or 2 for a UV plane). An odd last column or row is dropped.
fn halve(src: &[u8], stride: usize, w: usize, h: usize, ch: usize, dst: &mut Vec<u8>, simd: bool) {
    let (ow, oh) = (w / 2, h / 2);
    dst.resize(ow * oh * ch, 0);
    if ow == 0 || oh == 0 {
        return;
    }
    for (r, out) in dst.chunks_exact_mut(ow * ch).enumerate() {
        let top = &src[2 * r * stride..][..ow * 2 * ch];
        let bottom = &src[(2 * r + 1) * stride..][..ow * 2 * ch];
        if ch == 1 {
            #[cfg(target_arch = "x86_64")]
            let done = if simd {
                // SAFETY: `simd` is only set after AVX2 was detected (has_simd).
                unsafe { avx2::halve_row(top, bottom, out) }
            } else {
                0
            };
            #[cfg(not(target_arch = "x86_64"))]
            let done = {
                let _ = simd;
                0
            };
            let pairs = top[done * 2..]
                .as_chunks::<2>()
                .0
                .iter()
                .zip(bottom[done * 2..].as_chunks::<2>().0);
            for (o, (t, b)) in out[done..].iter_mut().zip(pairs) {
                *o = avg4(t[0], t[1], b[0], b[1]);
            }
        } else {
            let quads = top.as_chunks::<4>().0.iter().zip(bottom.as_chunks::<4>().0);
            for (o, (t, b)) in out.as_chunks_mut::<2>().0.iter_mut().zip(quads) {
                *o = [avg4(t[0], t[2], b[0], b[2]), avg4(t[1], t[3], b[1], b[3])];
            }
        }
    }
}

/// `src` halved `n >= 1` times, back and forth between the two buffers. Returns the one
/// that holds the result: rows of `w >> n` pixels, packed.
fn halve_n(
    src: &Plane<'_>,
    (mut w, mut h): (usize, usize),
    ch: usize,
    n: u32,
    bufs: &mut [Vec<u8>; 2],
    simd: bool,
) -> usize {
    let [a, b] = bufs;
    halve(src.data, src.stride, w, h, ch, a, simd);
    let mut cur = 0;
    for _ in 1..n {
        (w, h) = (w / 2, h / 2);
        if cur == 0 {
            halve(a, w * ch, w, h, ch, b, simd);
        } else {
            halve(b, w * ch, w, h, ch, a, simd);
        }
        cur ^= 1;
    }
    cur
}

/// Output buffers handed out again once nobody else holds them. A frame's pixels are an
/// `Arc`, so "nobody else" is a reference count of one.
#[derive(Default)]
struct Pool {
    bufs: Vec<Arc<[u8]>>,
}

impl Pool {
    fn take(&mut self, len: usize) -> Arc<[u8]> {
        // another frame size: the old buffers are of no use any more
        self.bufs.retain(|b| b.len() == len);
        match self.bufs.iter().position(|b| Arc::strong_count(b) == 1) {
            Some(i) => self.bufs.swap_remove(i),
            None => std::iter::repeat_n(0, len).collect(),
        }
    }

    fn keep(&mut self, buf: &Arc<[u8]>) {
        if self.bufs.len() < POOL_KEEP {
            self.bufs.push(Arc::clone(buf));
        }
    }

    /// Drop the buffers nobody uses right now.
    fn trim(&mut self) {
        self.bufs.retain(|b| Arc::strong_count(b) > 1);
    }
}

/// Converts the frames of one video. Keeps what the next frame can use again: the output
/// buffers and the planes it works in.
#[derive(Default)]
pub struct Converter {
    pool: Pool,
    /// the size the frames are shown at; `None`: unknown, frames keep their size
    target: Option<(u32, u32)>,
    /// planes cut down to 8 bit: Y, U (or UV), V
    deep: [Vec<u8>; 3],
    half_y: [Vec<u8>; 2],
    half_u: [Vec<u8>; 2],
    half_v: [Vec<u8>; 2],
}

impl Converter {
    /// The size the frames are drawn at, in pixels (0 = unknown). Frames larger than
    /// that come out halved, as often as they still cover it: the texture is never
    /// stretched, only shrunk by less than two.
    pub fn set_target(&mut self, width: u32, height: u32) {
        self.target = (width > 0 && height > 0).then_some((width, height));
    }

    /// Nobody is watching: give the memory back.
    pub fn idle(&mut self) {
        self.pool.trim();
        self.deep = Default::default();
        self.half_y = Default::default();
        self.half_u = Default::default();
        self.half_v = Default::default();
    }

    fn halvings(&self, w: usize, h: usize) -> u32 {
        let Some((tw, th)) = self.target else {
            return 0;
        };
        let (tw, th) = (tw as usize, th as usize);
        let mut n = 0;
        while n < MAX_HALVINGS && (w >> (n + 1)) >= tw && (h >> (n + 1)) >= th {
            n += 1;
        }
        n
    }

    /// Conversion. `None` means invalid plane sizes.
    pub fn convert(&mut self, img: &YuvImage<'_>) -> Option<CpuFrame> {
        self.convert_with(img, has_simd())
    }

    fn convert_with(&mut self, img: &YuvImage<'_>, simd: bool) -> Option<CpuFrame> {
        let (w, h) = (img.width as usize, img.height as usize);
        if w == 0 || h == 0 || !(8..=16).contains(&img.bit_depth) {
            return None;
        }
        let bpp = if img.bit_depth > 8 { 2 } else { 1 };
        let down = img.bit_depth.saturating_sub(8);
        let (sx, sy) = img.chroma.shift();
        let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));

        if !plane_ok(&img.y, w, h, bpp) {
            return None;
        }
        let mut uv = match img.uv {
            _ if img.chroma == Chroma::Mono => ChromaPlanes::None,
            ChromaPlanes::Planar { u, v } => {
                if !plane_ok(&u, cw, ch, bpp) || !plane_ok(&v, cw, ch, bpp) {
                    return None;
                }
                img.uv
            }
            ChromaPlanes::Interleaved(p) => {
                if !plane_ok(&p, cw * 2, ch, bpp) {
                    return None;
                }
                img.uv
            }
            ChromaPlanes::None => return None,
        };
        let mut y = img.y;
        let n = if img.chroma == Chroma::Sub420 {
            self.halvings(w, h)
        } else {
            0
        };
        let Self {
            pool,
            deep,
            half_y,
            half_u,
            half_v,
            ..
        } = self;

        // samples above 8 bit: only their top 8 bits are used
        if bpp == 2 {
            let [dy, du, dv] = deep;
            reduce(&y, w, h, down, dy);
            y = Plane {
                data: dy,
                stride: w,
            };
            uv = match uv {
                ChromaPlanes::Planar { u, v } => {
                    reduce(&u, cw, ch, down, du);
                    reduce(&v, cw, ch, down, dv);
                    ChromaPlanes::Planar {
                        u: Plane {
                            data: du,
                            stride: cw,
                        },
                        v: Plane {
                            data: dv,
                            stride: cw,
                        },
                    }
                }
                ChromaPlanes::Interleaved(p) => {
                    reduce(&p, cw * 2, ch, down, du);
                    ChromaPlanes::Interleaved(Plane {
                        data: du,
                        stride: cw * 2,
                    })
                }
                ChromaPlanes::None => ChromaPlanes::None,
            };
        }

        // Shown smaller than it is: halve the luma n times. 4:2:0 chroma is half the size
        // already, so it needs one halving less and then has a sample for every pixel.
        let (mut ow, mut oh, mut sub) = (w, h, (sx == 1, sy == 1));
        if n > 0 {
            let i = halve_n(&y, (w, h), 1, n, half_y, simd);
            (ow, oh, sub) = (w >> n, h >> n, (false, false));
            y = Plane {
                data: &half_y[i],
                stride: ow,
            };
            if n > 1 {
                let cs = cw >> (n - 1);
                uv = match uv {
                    ChromaPlanes::Planar { u, v } => {
                        let iu = halve_n(&u, (cw, ch), 1, n - 1, half_u, simd);
                        let iv = halve_n(&v, (cw, ch), 1, n - 1, half_v, simd);
                        ChromaPlanes::Planar {
                            u: Plane {
                                data: &half_u[iu],
                                stride: cs,
                            },
                            v: Plane {
                                data: &half_v[iv],
                                stride: cs,
                            },
                        }
                    }
                    ChromaPlanes::Interleaved(p) => {
                        let i = halve_n(&p, (cw, ch), 2, n - 1, half_u, simd);
                        ChromaPlanes::Interleaved(Plane {
                            data: &half_u[i],
                            stride: cs * 2,
                        })
                    }
                    ChromaPlanes::None => ChromaPlanes::None,
                };
            }
        }

        let img8 = Image8 {
            w: ow,
            h: oh,
            y,
            uv,
            sub_x: sub.0,
            sub_y: sub.1,
        };
        let c = Coefs::new(img.matrix, img.full_range);
        let mut buf = pool.take(ow * oh * 4);
        // the pool only hands out buffers nobody else holds
        convert8(&img8, &c, Arc::get_mut(&mut buf)?, simd);
        pool.keep(&buf);
        Some(CpuFrame::from_bgra(
            u32::try_from(ow).ok()?,
            u32::try_from(oh).ok()?,
            buf,
        ))
    }
}

/// One frame on its own. A decoder keeps a [`Converter`] instead.
pub fn to_bgra(img: &YuvImage<'_>) -> Option<CpuFrame> {
    Converter::default().convert(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid420(y: u8, u: u8, v: u8, w: usize, h: usize, matrix: Matrix, full: bool) -> [u8; 4] {
        let yp = vec![y; w * h];
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let up = vec![u; cw * ch];
        let vp = vec![v; cw * ch];
        let img = YuvImage {
            width: w as u32,
            height: h as u32,
            bit_depth: 8,
            chroma: Chroma::Sub420,
            matrix,
            full_range: full,
            y: Plane {
                data: &yp,
                stride: w,
            },
            uv: ChromaPlanes::Planar {
                u: Plane {
                    data: &up,
                    stride: cw,
                },
                v: Plane {
                    data: &vp,
                    stride: cw,
                },
            },
        };
        let f = to_bgra(&img).unwrap();
        f.pixels[..4].try_into().unwrap()
    }

    fn close(a: [u8; 4], b: [u8; 4]) -> bool {
        a.iter()
            .zip(b)
            .all(|(x, y)| (i16::from(*x) - i16::from(y)).abs() <= 2)
    }

    #[test]
    fn limited_range_black_white() {
        assert_eq!(
            solid420(16, 128, 128, 4, 4, Matrix::Bt601, false),
            [0, 0, 0, 255]
        );
        assert_eq!(
            solid420(235, 128, 128, 4, 4, Matrix::Bt601, false),
            [255, 255, 255, 255]
        );
        assert_eq!(
            solid420(255, 128, 128, 4, 4, Matrix::Bt601, true),
            [255, 255, 255, 255]
        );
    }

    #[test]
    fn bt601_and_bt709_red() {
        // pure red in limited range
        assert!(close(
            solid420(81, 90, 240, 4, 4, Matrix::Bt601, false),
            [0, 0, 255, 255]
        ));
        assert!(close(
            solid420(63, 102, 240, 4, 4, Matrix::Bt709, false),
            [0, 0, 255, 255]
        ));
    }

    #[test]
    fn odd_sizes_and_nv12() {
        // 3x3, NV12 with stride larger than width
        let yp = vec![235u8; 4 * 3];
        let uvp = vec![128u8; 4 * 2];
        let img = YuvImage {
            width: 3,
            height: 3,
            bit_depth: 8,
            chroma: Chroma::Sub420,
            matrix: Matrix::Bt709,
            full_range: false,
            y: Plane {
                data: &yp,
                stride: 4,
            },
            uv: ChromaPlanes::Interleaved(Plane {
                data: &uvp,
                stride: 4,
            }),
        };
        let f = to_bgra(&img).unwrap();
        assert_eq!((f.width, f.height, f.pixels.len()), (3, 3, 36));
        assert!(f.pixels.chunks(4).all(|p| p == [255, 255, 255, 255]));
    }

    #[test]
    fn ten_bit_and_mono() {
        let yp: Vec<u8> = std::iter::repeat_n(940u16.to_le_bytes(), 4)
            .flatten()
            .collect();
        let img = YuvImage {
            width: 2,
            height: 2,
            bit_depth: 10,
            chroma: Chroma::Mono,
            matrix: Matrix::Bt709,
            full_range: false,
            y: Plane {
                data: &yp,
                stride: 4,
            },
            uv: ChromaPlanes::None,
        };
        let f = to_bgra(&img).unwrap();
        assert_eq!(&f.pixels[..4], &[255, 255, 255, 255]);
    }

    #[test]
    fn short_planes_are_rejected() {
        let yp = vec![0u8; 3];
        let img = YuvImage {
            width: 4,
            height: 4,
            bit_depth: 8,
            chroma: Chroma::Mono,
            matrix: Matrix::Bt601,
            full_range: false,
            y: Plane {
                data: &yp,
                stride: 4,
            },
            uv: ChromaPlanes::None,
        };
        assert!(to_bgra(&img).is_none());
    }

    #[test]
    fn parallel_matches_single_thread() {
        // 1600x900 > threshold: threaded bands must match the row-by-row result
        let (w, h) = (1600usize, 900usize);
        let yp: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
        let (cw, ch) = (w / 2, h / 2);
        let up: Vec<u8> = (0..cw * ch).map(|i| (i * 3 % 241) as u8).collect();
        let vp: Vec<u8> = (0..cw * ch).map(|i| (i * 5 % 239) as u8).collect();
        let img = YuvImage {
            width: w as u32,
            height: h as u32,
            bit_depth: 8,
            chroma: Chroma::Sub420,
            matrix: Matrix::Bt709,
            full_range: false,
            y: Plane {
                data: &yp,
                stride: w,
            },
            uv: ChromaPlanes::Planar {
                u: Plane {
                    data: &up,
                    stride: cw,
                },
                v: Plane {
                    data: &vp,
                    stride: cw,
                },
            },
        };
        let f = to_bgra(&img).unwrap();
        let c = Coefs::new(Matrix::Bt709, false);
        for &(x, y) in &[
            (0usize, 0usize),
            (1599, 0),
            (800, 450),
            (0, 899),
            (1599, 899),
            (777, 333),
        ] {
            let want = c.px(
                i32::from(yp[y * w + x]),
                i32::from(up[(y / 2) * cw + x / 2]),
                i32::from(vp[(y / 2) * cw + x / 2]),
            );
            assert_eq!(&f.pixels[(y * w + x) * 4..][..4], &want, "({x},{y})");
        }
    }

    /// Deterministic noise.
    fn noise(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// The conversion spelled out per pixel, for any layout.
    fn reference(img: &YuvImage<'_>) -> Vec<u8> {
        let c = Coefs::new(img.matrix, img.full_range);
        let (sx, sy) = img.chroma.shift();
        let down = img.bit_depth.saturating_sub(8);
        let at = |p: &Plane<'_>, x: usize, y: usize| -> i32 {
            if img.bit_depth == 8 {
                i32::from(p.data[y * p.stride + x])
            } else {
                let i = y * p.stride + x * 2;
                (i32::from(u16::from_le_bytes([p.data[i], p.data[i + 1]])) >> down).min(255)
            }
        };
        let mut out = Vec::new();
        for y in 0..img.height as usize {
            for x in 0..img.width as usize {
                let (cx, cy) = (x >> sx, y >> sy);
                let (u, v) = match img.uv {
                    _ if img.chroma == Chroma::Mono => (128, 128),
                    ChromaPlanes::Planar { u, v } => (at(&u, cx, cy), at(&v, cx, cy)),
                    ChromaPlanes::Interleaved(p) => (at(&p, cx * 2, cy), at(&p, cx * 2 + 1, cy)),
                    ChromaPlanes::None => (128, 128),
                };
                out.extend_from_slice(&c.px(at(&img.y, x, y), u, v));
            }
        }
        out
    }

    /// Every layout the decoders produce, at sizes with and without a tail after the
    /// 8-pixel blocks, against the per-pixel reference: scalar rows and AVX2 rows.
    #[test]
    fn every_layout_matches_the_reference() {
        let sizes = [
            (1, 1),
            (2, 2),
            (7, 3),
            (8, 2),
            (13, 5),
            (16, 16),
            (64, 36),
            (131, 17),
        ];
        let layouts = [
            (Chroma::Sub420, false),
            (Chroma::Sub420, true),
            (Chroma::Sub422, false),
            (Chroma::Full, false),
            (Chroma::Full, true),
            (Chroma::Mono, false),
        ];
        let mut seed = 1;
        for (w, h) in sizes {
            for (chroma, interleaved) in layouts {
                for depth in [8, 10, 16] {
                    for (matrix, full) in [
                        (Matrix::Bt601, false),
                        (Matrix::Bt709, true),
                        (Matrix::Bt2020, false),
                    ] {
                        seed += 1;
                        let bpp = if depth > 8 { 2 } else { 1 };
                        let (sx, sy) = chroma.shift();
                        let (cw, ch) = (usize::div_ceil(w, 1 << sx), usize::div_ceil(h, 1 << sy));
                        // strides wider than the rows, as decoders have them
                        let (ys, cs) = (w * bpp + 3, cw * bpp + 5);
                        let mut plane = |stride: usize, rows: usize| {
                            let mut d = noise(stride * rows, seed);
                            seed += 1;
                            if depth == 10 {
                                // 10-bit samples in 16-bit words (rows at an odd offset keep
                                // larger ones: those are cut to 255)
                                for hi in d.iter_mut().skip(1).step_by(2) {
                                    *hi &= 3;
                                }
                            }
                            d
                        };
                        let yp = plane(ys, h);
                        let up = plane(cs, ch);
                        let vp = plane(cs, ch);
                        let uvp = plane(cs * 2, ch);
                        let uv = match (chroma, interleaved) {
                            (Chroma::Mono, _) => ChromaPlanes::None,
                            (_, true) => ChromaPlanes::Interleaved(Plane {
                                data: &uvp,
                                stride: cs * 2,
                            }),
                            (_, false) => ChromaPlanes::Planar {
                                u: Plane {
                                    data: &up,
                                    stride: cs,
                                },
                                v: Plane {
                                    data: &vp,
                                    stride: cs,
                                },
                            },
                        };
                        let img = YuvImage {
                            width: w as u32,
                            height: h as u32,
                            bit_depth: depth,
                            chroma,
                            matrix,
                            full_range: full,
                            y: Plane {
                                data: &yp,
                                stride: ys,
                            },
                            uv,
                        };
                        let what =
                            format!("{w}x{h} {chroma:?} interleaved={interleaved} {depth} bit");
                        let want = reference(&img);
                        let scalar = Converter::default().convert_with(&img, false).unwrap();
                        assert_eq!(&*scalar.pixels, &want[..], "scalar, {what}");
                        if has_simd() {
                            let simd = Converter::default().convert_with(&img, true).unwrap();
                            assert_eq!(&*simd.pixels, &want[..], "AVX2, {what}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn coefficients_fit_the_simd_kernel() {
        for matrix in [Matrix::Bt601, Matrix::Bt709, Matrix::Bt2020] {
            for full in [false, true] {
                let c = Coefs::new(matrix, full);
                for v in [c.y_mul, c.rv, c.gu, c.gv, c.bu - c.bu / 2] {
                    assert!((0..32768).contains(&v), "{matrix:?} full={full}: {v}");
                }
            }
        }
    }

    /// 4:2:0 picture whose luma is `luma(x, y)` and whose chroma is flat.
    fn picture(w: usize, h: usize, nv12: bool, luma: impl Fn(usize, usize) -> u8) -> Vec<Vec<u8>> {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y = (0..w * h).map(|i| luma(i % w, i / w)).collect();
        if nv12 {
            vec![y, [90, 200].repeat(cw * ch)]
        } else {
            vec![y, vec![90; cw * ch], vec![200; cw * ch]]
        }
    }

    fn view(planes: &[Vec<u8>], w: usize, h: usize) -> YuvImage<'_> {
        let cw = w.div_ceil(2);
        YuvImage {
            width: w as u32,
            height: h as u32,
            bit_depth: 8,
            chroma: Chroma::Sub420,
            matrix: Matrix::Bt709,
            full_range: false,
            y: Plane {
                data: &planes[0],
                stride: w,
            },
            uv: if planes.len() == 2 {
                ChromaPlanes::Interleaved(Plane {
                    data: &planes[1],
                    stride: cw * 2,
                })
            } else {
                ChromaPlanes::Planar {
                    u: Plane {
                        data: &planes[1],
                        stride: cw,
                    },
                    v: Plane {
                        data: &planes[2],
                        stride: cw,
                    },
                }
            },
        }
    }

    #[test]
    fn a_frame_shown_smaller_is_halved() {
        let c = Coefs::new(Matrix::Bt709, false);
        for nv12 in [false, true] {
            // luma in 4x4 blocks of 40, 80, 120, 160 (rows) + 0, 8 (columns): averages are exact
            let luma = |x: usize, y: usize| (40 + 40 * (y % 4) + 8 * (x % 2)) as u8;
            let planes = picture(64, 48, nv12, luma);
            let img = view(&planes, 64, 48);
            let mut conv = Converter::default();

            conv.set_target(64, 48);
            let f = conv.convert(&img).unwrap();
            assert_eq!((f.width, f.height), (64, 48), "shown at its size: as it is");

            // 32x24 still covers 20x20, 16x12 would not
            conv.set_target(20, 20);
            let f = conv.convert(&img).unwrap();
            assert_eq!((f.width, f.height, f.stride), (32, 24, 128));
            assert_eq!(f.pixels.len(), 32 * 24 * 4);
            // rows 0+1 average to 60 + 4, rows 2+3 to 140 + 4
            assert_eq!(&f.pixels[..4], &c.px(64, 90, 200), "nv12={nv12}");
            assert_eq!(&f.pixels[32 * 4..][..4], &c.px(144, 90, 200), "nv12={nv12}");

            // twice: 16x12, every pixel the average of a 4x4 block
            conv.set_target(10, 10);
            let f = conv.convert(&img).unwrap();
            assert_eq!((f.width, f.height), (16, 12));
            assert!(
                f.pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| *p == c.px(104, 90, 200)),
                "nv12={nv12}"
            );

            conv.set_target(0, 0);
            let f = conv.convert(&img).unwrap();
            assert_eq!((f.width, f.height), (64, 48), "size unknown: as it is");
            conv.set_target(500, 500);
            let f = conv.convert(&img).unwrap();
            assert_eq!((f.width, f.height), (64, 48), "never enlarged");
        }
    }

    #[test]
    fn halving_is_the_same_with_and_without_simd() {
        if !has_simd() {
            return;
        }
        // wide enough for the 64-pixel blocks, with a tail after them
        for (w, h) in [(200, 90), (333, 71), (64, 2)] {
            for nv12 in [false, true] {
                let mut planes = picture(w, h, nv12, |_, _| 0);
                for (i, p) in planes.iter_mut().enumerate() {
                    *p = noise(p.len(), (w * 31 + i) as u32);
                }
                for target in [(w as u32 / 3, 1), (w as u32 / 5, 1)] {
                    let img = view(&planes, w, h);
                    let (mut a, mut b) = (Converter::default(), Converter::default());
                    a.set_target(target.0, target.1);
                    b.set_target(target.0, target.1);
                    let (a, b) = (a.convert_with(&img, false), b.convert_with(&img, true));
                    let (a, b) = (a.unwrap(), b.unwrap());
                    assert!(a.width < w as u32, "{w}x{h}: it is halved");
                    assert_eq!((a.width, a.height), (b.width, b.height));
                    assert_eq!(&*a.pixels, &*b.pixels, "{w}x{h} nv12={nv12} to {target:?}");
                }
            }
        }
    }

    /// P016, as NVDEC hands out 10-bit video: 16-bit words, chroma interleaved.
    #[test]
    fn deep_samples_are_cut_to_8_bit_before_halving() {
        let (w, h) = (64usize, 48usize);
        let words = |v: u16, n: usize| -> Vec<u8> { v.to_le_bytes().repeat(n) };
        let y = words(180 << 8, w * h);
        let uv: Vec<u8> = [90u16 << 8, 200 << 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
            .repeat(w / 2 * h / 2);
        let img = YuvImage {
            width: w as u32,
            height: h as u32,
            bit_depth: 16,
            chroma: Chroma::Sub420,
            matrix: Matrix::Bt709,
            full_range: false,
            y: Plane {
                data: &y,
                stride: w * 2,
            },
            uv: ChromaPlanes::Interleaved(Plane {
                data: &uv,
                stride: w * 2,
            }),
        };
        let want = Coefs::new(Matrix::Bt709, false).px(180, 90, 200);
        let mut conv = Converter::default();
        for (target, size) in [((0, 0), (64, 48)), ((30, 20), (32, 24)), ((9, 9), (16, 12))] {
            conv.set_target(target.0, target.1);
            let f = conv.convert(&img).unwrap();
            assert_eq!((f.width, f.height), size);
            assert!(
                f.pixels.as_chunks::<4>().0.iter().all(|p| *p == want),
                "shown at {target:?}"
            );
        }
    }

    #[test]
    fn odd_sizes_are_halved_without_reading_past_the_planes() {
        for (w, h) in [(13, 7), (33, 9), (9, 33)] {
            for nv12 in [false, true] {
                let planes = picture(w, h, nv12, |_, _| 235);
                let mut conv = Converter::default();
                conv.set_target(1, 1);
                let f = conv.convert(&view(&planes, w, h)).unwrap();
                assert!(f.width >= 1 && f.height >= 1);
                assert_eq!(f.pixels.len(), (f.width * f.height * 4) as usize);
            }
        }
    }

    #[test]
    fn output_buffers_are_reused() {
        let planes = picture(64, 48, false, |_, _| 100);
        let img = view(&planes, 64, 48);
        let mut conv = Converter::default();
        let a = conv.convert(&img).unwrap();
        let b = conv.convert(&img).unwrap();
        let (pa, pb) = (a.pixels.as_ptr(), b.pixels.as_ptr());
        assert_ne!(pa, pb, "a frame in use keeps its buffer");
        drop(a);
        let c = conv.convert(&img).unwrap();
        assert_eq!(
            c.pixels.as_ptr(),
            pa,
            "a dropped frame's buffer is used again"
        );
        assert_eq!(&*c.pixels, &*b.pixels);
        drop((b, c));
        conv.idle();
        assert!(conv.pool.bufs.is_empty(), "idle gives the buffers back");
    }
}
