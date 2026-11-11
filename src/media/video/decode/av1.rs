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
    DAV1D_MC_BT709, DAV1D_MC_BT2020_NCL, DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420,
    DAV1D_PIXEL_LAYOUT_I422, DAV1D_PIXEL_LAYOUT_I444,
};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_flush,
    dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

use crate::frame::FrameData;
use crate::media::video::pipeline::{DecodedFrame, Decoder, Packet, PipeResult};
use crate::media::video::yuv::{self, Chroma, ChromaPlanes, Matrix, Plane, YuvImage};

const EAGAIN: i32 = -libc::EAGAIN;

pub struct Av1Decoder {
    ctx: Option<Dav1dContext>,
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
        Ok(Self { ctx })
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
            let frame = convert(&pic);
            // SAFETY: the picture came from get_picture; release it exactly once.
            unsafe { dav1d_picture_unref(Some(NonNull::from(&mut pic))) };
            if let Some(f) = frame {
                out.push(f);
            }
        }
    }
}

fn threads() -> i32 {
    std::thread::available_parallelism().map_or(2, |n| n.get().clamp(1, 4) as i32)
}

fn convert(pic: &Dav1dPicture) -> Option<DecodedFrame> {
    let (w, h) = (u32::try_from(pic.p.w).ok()?, u32::try_from(pic.p.h).ok()?);
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
    // SAFETY: the planes belong to the picture and hold at least stride*(rows) bytes.
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
    let cpu = yuv::to_bgra(&img)?;
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

        let res = loop {
            // SAFETY: data was created above; send_data advances data/sz as it consumes input.
            let r: Dav1dResult =
                unsafe { dav1d_send_data(self.ctx, Some(NonNull::from(&mut data))) };
            if r.0 < 0 && r.0 != EAGAIN {
                break Err(format!("dav1d_send_data: {}", r.0));
            }
            if let Err(e) = self.drain(&mut out) {
                break Err(e);
            }
            if data.sz == 0 {
                break Ok(());
            }
        };
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
}

impl Drop for Av1Decoder {
    fn drop(&mut self) {
        // SAFETY: closed exactly once; close zeroes ctx.
        unsafe { dav1d_close(Some(NonNull::from(&mut self.ctx))) };
    }
}
