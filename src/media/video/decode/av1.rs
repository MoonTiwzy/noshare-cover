//! AV1 on CPU: rav1d (dav1d rewritten in Rust, with the same C API and asm kernels).
//!
//! Nothing is loaded from the system: the decoder is built into the plugin. Frames come out as
//! YUV and are converted to BGRA via [`yuv`](super::super::yuv).

use std::ptr::NonNull;
use std::time::Duration;

use rav1d::Dav1dResult;
use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::{
    DAV1D_MC_BT709, DAV1D_MC_BT2020_NCL, DAV1D_MC_IDENTITY, DAV1D_PIXEL_LAYOUT_I400,
    DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I422, DAV1D_PIXEL_LAYOUT_I444,
};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_flush,
    dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

use crate::frame::FrameData;
use crate::media::video::pipeline::{DecodedFrame, Decoder, Packet, PipeResult};
use crate::media::video::yuv::{Chroma, ChromaPlanes, Converter, Matrix, Plane, YuvImage};

const EAGAIN: i32 = -libc::EAGAIN;

pub struct Av1Decoder {
    ctx: Option<Dav1dContext>,
    conv: Converter,
}

// The dav1d context lives on the single decoder thread; it is thread-safe itself
// (it runs its own worker threads), and the pointer never escapes.
unsafe impl Send for Av1Decoder {}

impl Av1Decoder {
    pub fn new() -> PipeResult<Self> {
        let mut s = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
        let mut ctx: Option<Dav1dContext> = None;
        // SAFETY: default_settings fully initializes the struct; open writes to ctx.
        let r = unsafe {
            dav1d_default_settings(NonNull::new_unchecked(s.as_mut_ptr()));
            let s = s.as_mut_ptr();
            // The cover is a background image: a couple of threads is enough, the compositor needs the CPU more.
            (*s).n_threads = threads();
            (*s).max_frame_delay = 1;
            dav1d_open(Some(NonNull::from(&mut ctx)), NonNull::new(s))
        };
        if r.0 < 0 || ctx.is_none() {
            return Err(format!("dav1d_open: {}", r.0));
        }
        Ok(Self {
            ctx,
            conv: Converter::default(),
        })
    }

    fn drain(&mut self, out: &mut Vec<DecodedFrame>) -> PipeResult<()> {
        loop {
            let mut pic = Dav1dPicture::default();
            // SAFETY: ctx is open, pic is a valid empty struct.
            let r = unsafe { dav1d_get_picture(self.ctx, Some(NonNull::from(&mut pic))) };
            if r.0 == EAGAIN {
                return Ok(());
            }
            if r.0 < 0 {
                return Err(format!("dav1d_get_picture: {}", r.0));
            }
            let frame = convert(&mut self.conv, &pic);
            // SAFETY: the picture came from get_picture; release it exactly once.
            unsafe { dav1d_picture_unref(Some(NonNull::from(&mut pic))) };
            if let Some(f) = frame {
                out.push(f);
            }
        }
    }
}

/// Hands one packet to the decoder: `send` offers it and says how much of it is left (all of
/// it on EAGAIN: the decoder still holds earlier data and wants its pictures taken first),
/// `drain` takes out the pictures that are ready.
///
/// A drain that ends in EAGAIN has used up whatever the decoder held (rav1d's get_picture
/// parses its input until a picture is ready or nothing is left), so the next send takes the
/// packet. A second EAGAIN in a row means the decoder is stuck: the packet is given up
/// instead of spinning the decoder thread, which unloading the plugin waits for.
fn feed(
    mut send: impl FnMut() -> (i32, usize),
    mut drain: impl FnMut() -> PipeResult<()>,
) -> PipeResult<()> {
    let mut waited = false;
    loop {
        let (r, left) = send();
        if r < 0 && r != EAGAIN {
            return Err(format!("dav1d_send_data: {r}"));
        }
        drain()?;
        if left == 0 {
            return Ok(());
        }
        if std::mem::replace(&mut waited, true) {
            return Err("dav1d_send_data: the decoder takes no more data".into());
        }
    }
}

fn threads() -> i32 {
    std::thread::available_parallelism().map_or(2, |n| n.get().clamp(1, 4) as i32)
}

fn convert(conv: &mut Converter, pic: &Dav1dPicture) -> Option<DecodedFrame> {
    let (w, h) = (u32::try_from(pic.p.w).ok()?, u32::try_from(pic.p.h).ok()?);
    // the plane lengths below count from the last row
    if w == 0 || h == 0 {
        return None;
    }
    let bpc = u32::try_from(pic.p.bpc).ok()?;
    let chroma = match pic.p.layout {
        DAV1D_PIXEL_LAYOUT_I400 => Chroma::Mono,
        DAV1D_PIXEL_LAYOUT_I420 => Chroma::Sub420,
        DAV1D_PIXEL_LAYOUT_I422 => Chroma::Sub422,
        DAV1D_PIXEL_LAYOUT_I444 => Chroma::Full,
        _ => return None,
    };
    let (matrix, full_range) = match pic.seq_hdr {
        // SAFETY: seq_hdr lives as long as the picture.
        Some(sh) => unsafe {
            let sh = sh.as_ref();
            let m = match sh.mtrx {
                DAV1D_MC_BT709 => Matrix::Bt709,
                DAV1D_MC_BT2020_NCL => Matrix::Bt2020,
                // RGB: the planes are G, B, R. Only 4:4:4 can be RGB (rav1d lets identity
                // through on other layouts unless told to be strict): a monochrome picture
                // has no B and R, and ffmpeg's "gbr" mislabel lands on plain YUV.
                DAV1D_MC_IDENTITY if chroma == Chroma::Full => Matrix::Identity,
                2 => Matrix::guess(h), // unspecified
                _ => Matrix::Bt601,
            };
            (m, sh.color_range != 0)
        },
        None => (Matrix::guess(h), false),
    };
    let bpp = if bpc > 8 { 2 } else { 1 };
    let (sx, sy) = match chroma {
        Chroma::Sub420 => (1, 1),
        Chroma::Sub422 => (1, 0),
        _ => (0, 0),
    };
    let ch = (h as usize).div_ceil(1 << sy);
    let cw = (w as usize).div_ceil(1 << sx);
    let ys = usize::try_from(pic.stride[0]).ok()?;
    let cs = usize::try_from(pic.stride[1]).ok()?;
    // SAFETY: the planes belong to the picture and hold at least stride*(rows) bytes; rows
    // is at least 1 (checked above).
    let plane = |i: usize, stride: usize, width: usize, rows: usize| -> Option<Plane<'_>> {
        let p = pic.data[i]?;
        let len = stride * (rows - 1) + width * bpp;
        Some(Plane {
            data: unsafe { std::slice::from_raw_parts(p.as_ptr().cast::<u8>(), len) },
            stride,
        })
    };
    let y = plane(0, ys, w as usize, h as usize)?;
    let uv = if chroma == Chroma::Mono {
        ChromaPlanes::None
    } else {
        ChromaPlanes::Planar {
            u: plane(1, cs, cw, ch)?,
            v: plane(2, cs, cw, ch)?,
        }
    };
    let img = YuvImage {
        width: w,
        height: h,
        bit_depth: bpc,
        chroma,
        matrix,
        full_range,
        y,
        uv,
    };
    let cpu = conv.convert(&img)?;
    Some(DecodedFrame {
        pts: Duration::from_micros(u64::try_from(pic.m.timestamp).unwrap_or(0)),
        data: FrameData::Cpu(cpu),
    })
}

impl Decoder for Av1Decoder {
    fn name(&self) -> &'static str {
        "rav1d (CPU)"
    }

    fn decode(&mut self, packet: &Packet) -> PipeResult<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        if packet.data.is_empty() {
            return Ok(out);
        }
        let mut data = Dav1dData::default();
        // SAFETY: data_create allocates exactly len bytes; we copy the packet into it.
        unsafe {
            let buf = dav1d_data_create(Some(NonNull::from(&mut data)), packet.data.len());
            if buf.is_null() {
                return Err("dav1d_data_create: out of memory".into());
            }
            std::ptr::copy_nonoverlapping(packet.data.as_ptr(), buf, packet.data.len());
        }
        data.m.timestamp = i64::try_from(packet.pts.as_micros()).unwrap_or(i64::MAX);

        let ctx = self.ctx;
        let res = feed(
            || {
                // SAFETY: data was created above; send_data takes all of it (sz becomes 0),
                // or none of it on EAGAIN or an error.
                let r: Dav1dResult =
                    unsafe { dav1d_send_data(ctx, Some(NonNull::from(&mut data))) };
                (r.0, data.sz)
            },
            || self.drain(&mut out),
        );
        // SAFETY: if anything is left (on error), release it; unref ignores empty data.
        if data.sz > 0 {
            unsafe { dav1d_data_unref(Some(NonNull::from(&mut data))) };
        }
        res.map(|()| out)
    }

    fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        self.drain(&mut out)?;
        Ok(out)
    }

    fn reset(&mut self) {
        if let Some(c) = self.ctx {
            // SAFETY: ctx is open.
            unsafe { dav1d_flush(c) };
        }
    }

    fn set_target(&mut self, width: u32, height: u32) {
        self.conv.set_target(width, height);
    }

    fn idle(&mut self) {
        self.conv.idle();
    }
}

impl Drop for Av1Decoder {
    fn drop(&mut self) {
        // SAFETY: closed exactly once; close zeroes ctx.
        unsafe { dav1d_close(Some(NonNull::from(&mut self.ctx))) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::video::decode::testing;
    use crate::media::video::demux;

    /// A picture without rows has no last row to measure the planes by: nothing comes out,
    /// and nothing is read.
    #[test]
    fn an_empty_picture_gives_no_frame() {
        let mut buf = [0u8; 64];
        let p = NonNull::new(buf.as_mut_ptr().cast::<std::ffi::c_void>());
        for (w, h) in [(4, 0), (0, 4), (0, 0)] {
            let mut pic = Dav1dPicture::default();
            pic.p.w = w;
            pic.p.h = h;
            pic.p.bpc = 8;
            pic.p.layout = DAV1D_PIXEL_LAYOUT_I420;
            pic.data = [p, p, p];
            pic.stride = [8, 4];
            assert!(
                convert(&mut Converter::default(), &pic).is_none(),
                "{w}x{h}"
            );
        }
    }

    /// The decoder still holds data, a drain empties it, and the packet goes in; a decoder
    /// that keeps saying EAGAIN is stuck, and the packet is given up instead of spinning.
    #[test]
    fn a_decoder_that_takes_nothing_does_not_spin() {
        let (mut sends, mut drains) = (0, 0);
        let r = feed(
            || {
                sends += 1;
                if sends == 1 { (EAGAIN, 100) } else { (0, 0) }
            },
            || {
                drains += 1;
                Ok(())
            },
        );
        assert_eq!((r, sends, drains), (Ok(()), 2, 2));

        let mut sends = 0;
        let r = feed(
            || {
                sends += 1;
                (EAGAIN, 100)
            },
            || Ok(()),
        );
        assert!(r.is_err(), "{r:?}");
        assert_eq!(sends, 2);

        // an error from either side ends the packet
        let e = feed(|| (-libc::EINVAL, 100), || Ok(())).unwrap_err();
        assert!(e.contains(&(-libc::EINVAL).to_string()), "{e}");
        assert_eq!(feed(|| (0, 0), || Err("drain".into())), Err("drain".into()));
    }

    /// The first picture of a clip ffmpeg makes from `source`, as BGRA; `None` (the test is
    /// skipped) without ffmpeg or the encoder.
    fn first_pixel(name: &str, source: &str, args: &[&str]) -> Option<[u8; 4]> {
        let clip = testing::ffmpeg(name, source, args)?;
        let mut d = demux::open(clip.path()).unwrap();
        let mut dec = Av1Decoder::new().unwrap();
        let mut frames = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            frames.extend(dec.decode(&p).unwrap());
        }
        frames.extend(dec.flush().unwrap());
        #[allow(irrefutable_let_patterns)]
        let FrameData::Cpu(c) = &frames.first().expect("a picture").data else {
            panic!("a CPU frame")
        };
        Some(c.pixels[..4].try_into().unwrap())
    }

    fn near(px: [u8; 4], bgr: [u8; 3], by: u8) -> bool {
        px[..3].iter().zip(bgr).all(|(a, b)| a.abs_diff(b) <= by) && px[3] == 255
    }

    /// AV1 can carry RGB as matrix 0 (identity): the planes are G, B and R.
    #[test]
    fn identity_keeps_its_colour() {
        let Some(px) = first_pixel(
            "identity.mp4",
            "color=c=0x3080c0:s=32x32:r=10:d=0.3",
            &[
                "-pix_fmt",
                "gbrp",
                "-c:v",
                "libaom-av1",
                "-cpu-used",
                "8",
                "-aom-params",
                "lossless=1",
            ],
        ) else {
            return;
        };
        assert!(near(px, [0xc0, 0x80, 0x30], 2), "{px:?}");
    }

    /// Matrix 0 on subsampled chroma is no RGB (identity needs 4:4:4; ffmpeg's "gbr" mislabel
    /// writes it on plain YUV): the planes are still Y, U and V.
    #[test]
    fn identity_on_subsampled_chroma_is_yuv() {
        let Some(px) = first_pixel(
            "identity420.mp4",
            "color=c=0x3080c0:s=32x32:r=10:d=0.3",
            &[
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libaom-av1",
                "-cpu-used",
                "8",
                "-bsf:v",
                "av1_metadata=matrix_coefficients=0",
            ],
        ) else {
            return;
        };
        assert!(near(px, [0xc0, 0x80, 0x30], 4), "{px:?}");
    }

    /// A monochrome picture tagged with matrix 0 has no blue or red planes: it stays grey.
    #[test]
    fn identity_on_monochrome_stays_grey() {
        let Some(px) = first_pixel(
            "identity400.mp4",
            "color=c=0x404040:s=32x32:r=10:d=0.3",
            &[
                "-pix_fmt",
                "gray",
                "-c:v",
                "libaom-av1",
                "-cpu-used",
                "8",
                "-bsf:v",
                "av1_metadata=color_primaries=1:transfer_characteristics=1:matrix_coefficients=0",
            ],
        ) else {
            return;
        };
        assert!(near(px, [0x40, 0x40, 0x40], 2), "{px:?}");
    }
}
