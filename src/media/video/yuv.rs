//! YUV → premultiplied BGRA (video is opaque, so it's plain BGRA with A=255).
//!
//! Shared by all CPU paths: dav1d/rav1d (I420/I422/I444/I400, 8–12 bit),
//! openh264 (I420), libvpx (I420/I444), NVDEC with a copy to system memory (NV12).
//! Fixed-point integer math (14 bit), with a fast path for
//! the most common case: 8-bit 4:2:0.

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

/// Frame size at which conversion goes multithreaded (~1 MP: 1280x800 and up).
const PARALLEL_FROM_PIXELS: usize = 1_000_000;
/// No more threads than this: the compositor and decoder need CPU too.
const MAX_THREADS: usize = 4;

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

#[inline(always)]
fn sample(p: &Plane<'_>, x: usize, y: usize, shift: u32) -> i32 {
    if shift == 0 {
        i32::from(p.data[y * p.stride + x])
    } else {
        let i = y * p.stride + x * 2;
        (i32::from(u16::from_le_bytes([p.data[i], p.data[i + 1]])) >> shift).min(255)
    }
}

/// Checks that the planes actually fit the image (so we return None instead of panicking).
fn plane_ok(p: &Plane<'_>, w: usize, h: usize, bytes_per: usize) -> bool {
    h == 0 || (p.stride >= w * bytes_per && p.data.len() >= p.stride * (h - 1) + w * bytes_per)
}

/// Conversion. `None` means invalid plane sizes.
pub fn to_bgra(img: &YuvImage<'_>) -> Option<CpuFrame> {
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
    match img.uv {
        ChromaPlanes::Planar { u, v } if img.chroma != Chroma::Mono => {
            if !plane_ok(&u, cw, ch, bpp) || !plane_ok(&v, cw, ch, bpp) {
                return None;
            }
        }
        ChromaPlanes::Interleaved(uv) if img.chroma != Chroma::Mono => {
            if !plane_ok(&uv, cw * 2, ch, bpp) {
                return None;
            }
        }
        _ if img.chroma == Chroma::Mono => {}
        _ => return None,
    }

    let c = Coefs::new(img.matrix, img.full_range);
    let mut out = vec![0u8; w * h * 4];

    // Rows are independent: split a large frame (4K etc.) into bands across threads,
    // otherwise the conversion alone eats the whole frame budget.
    let convert_band = |first_row: usize, band: &mut [u8]| {
        // fast path: 8-bit, 4:2:0, separate planes
        if let (8, Chroma::Sub420, ChromaPlanes::Planar { u, v }) =
            (img.bit_depth, img.chroma, img.uv)
        {
            for (i, dst) in band.chunks_exact_mut(w * 4).enumerate() {
                let row = first_row + i;
                let yr = &img.y.data[row * img.y.stride..][..w];
                let ur = &u.data[(row >> 1) * u.stride..][..cw];
                let vr = &v.data[(row >> 1) * v.stride..][..cw];
                for (x, px) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    *px = c.px(
                        i32::from(yr[x]),
                        i32::from(ur[x >> 1]),
                        i32::from(vr[x >> 1]),
                    );
                }
            }
        } else {
            for (i, dst) in band.chunks_exact_mut(w * 4).enumerate() {
                let row = first_row + i;
                let cy = row >> sy;
                for (x, px) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let yv = sample(&img.y, x, row, down);
                    let cx = x >> sx;
                    let (uv, vv) = match img.uv {
                        _ if img.chroma == Chroma::Mono => (128, 128),
                        ChromaPlanes::Planar { u, v } => {
                            (sample(&u, cx, cy, down), sample(&v, cx, cy, down))
                        }
                        ChromaPlanes::Interleaved(p) => (
                            sample(&p, cx * 2, cy, down),
                            sample(&p, cx * 2 + 1, cy, down),
                        ),
                        ChromaPlanes::None => (128, 128),
                    };
                    *px = c.px(yv, uv, vv);
                }
            }
        }
    };

    let threads = if w * h >= PARALLEL_FROM_PIXELS {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(MAX_THREADS))
    } else {
        1
    };
    if threads <= 1 {
        convert_band(0, &mut out);
    } else {
        let rows_per = h.div_ceil(threads);
        std::thread::scope(|scope| {
            for (i, band) in out.chunks_mut(rows_per * w * 4).enumerate() {
                let convert_band = &convert_band;
                scope.spawn(move || convert_band(i * rows_per, band));
            }
        });
    }

    Some(CpuFrame::from_bgra(img.width, img.height, Arc::from(out)))
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
}
