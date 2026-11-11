//! VP8 and VP9 on CPU: system libvpx via dlopen.
//!
//! Our own minimal FFI based on the libvpx headers (vpx_decoder.h, vpx_image.h):
//! six functions and two structs. The libvpx decoder ABI hasn't changed since 1.8
//! (VPX_DECODER_ABI_VERSION = 12), but the soname keeps bumping, so we try them all.
//! pts travels via `user_priv`: VP8/VP9 don't reorder frames, but hidden
//! (altref) frames produce no picture, and the timestamp stays with its own frame.

use std::ffi::{CStr, c_char, c_int, c_long, c_uint, c_void};
use std::ptr::{null, null_mut};
use std::time::Duration;

use libloading::Library;

use crate::frame::FrameData;
use crate::media::video::pipeline::{Codec, DecodedFrame, Decoder, Packet, PipeResult};
use crate::media::video::yuv::{self, Chroma, ChromaPlanes, Matrix, Plane, YuvImage};

const CANDIDATES: &[&str] = &[
    "libvpx.so.11",
    "libvpx.so.10",
    "libvpx.so.9",
    "libvpx.so.8",
    "libvpx.so.7",
    "libvpx.so.6",
    "libvpx.so",
];
const ENV_PATH: &str = "NOSHARE_COVER_LIBVPX";

const VPX_IMAGE_ABI_VERSION: c_int = 5;
const VPX_CODEC_ABI_VERSION: c_int = 4 + VPX_IMAGE_ABI_VERSION;
const VPX_DECODER_ABI_VERSION: c_int = 3 + VPX_CODEC_ABI_VERSION;

const FMT_PLANAR: c_int = 0x100;
const FMT_UV_FLIP: c_int = 0x200;
const FMT_HIGHBITDEPTH: c_int = 0x800;
const FMT_I420: c_int = FMT_PLANAR | 2;
const FMT_YV12: c_int = FMT_PLANAR | FMT_UV_FLIP | 1;
const FMT_I422: c_int = FMT_PLANAR | 5;
const FMT_I444: c_int = FMT_PLANAR | 6;

const CS_BT_709: c_int = 2;
const CS_BT_2020: c_int = 5;

#[repr(C)]
struct VpxCodecCtx {
    name: *const c_char,
    iface: *const c_void,
    err: c_int,
    err_detail: *const c_char,
    init_flags: c_long,
    config: *const c_void,
    priv_: *mut c_void,
}

#[repr(C)]
struct VpxDecCfg {
    threads: c_uint,
    w: c_uint,
    h: c_uint,
}

#[repr(C)]
struct VpxImage {
    fmt: c_int,
    cs: c_int,
    range: c_int,
    w: c_uint,
    h: c_uint,
    bit_depth: c_uint,
    d_w: c_uint,
    d_h: c_uint,
    r_w: c_uint,
    r_h: c_uint,
    x_chroma_shift: c_uint,
    y_chroma_shift: c_uint,
    planes: [*mut u8; 4],
    stride: [c_int; 4],
    bps: c_int,
    user_priv: *mut c_void,
    img_data: *mut u8,
    img_data_owner: c_int,
    self_allocd: c_int,
    fb_priv: *mut c_void,
}

type FnIface = unsafe extern "C" fn() -> *const c_void;
type FnInit =
    unsafe extern "C" fn(*mut VpxCodecCtx, *const c_void, *const VpxDecCfg, c_long, c_int) -> c_int;
type FnDecode =
    unsafe extern "C" fn(*mut VpxCodecCtx, *const u8, c_uint, *mut c_void, c_long) -> c_int;
type FnGetFrame = unsafe extern "C" fn(*mut VpxCodecCtx, *mut *const c_void) -> *const VpxImage;
type FnDestroy = unsafe extern "C" fn(*mut VpxCodecCtx) -> c_int;
type FnErrStr = unsafe extern "C" fn(c_int) -> *const c_char;

struct Api {
    decode: FnDecode,
    get_frame: FnGetFrame,
    destroy: FnDestroy,
    err_str: FnErrStr,
    // keep the library loaded while pointers to its functions are alive
    _lib: Library,
}

pub struct VpxDecoder {
    api: Api,
    // Box: libvpx keeps a pointer to the context internally, so the address must not change
    ctx: Box<VpxCodecCtx>,
    vp9: bool,
}

// The context is only used from the decoder thread.
unsafe impl Send for VpxDecoder {}

fn open_lib() -> Result<Library, String> {
    let mut tried = Vec::new();
    let explicit = std::env::var(ENV_PATH).ok();
    // NSC_LIB_VPX: absolute path baked in at build time (Nix: library from the store;
    // on NixOS dlopen by soname won't find it)
    for name in explicit
        .iter()
        .map(String::as_str)
        .chain(option_env!("NSC_LIB_VPX"))
        .chain(CANDIDATES.iter().copied())
    {
        // SAFETY: libvpx has no constructors with side effects.
        match unsafe { Library::new(name) } {
            Ok(l) => return Ok(l),
            Err(e) => tried.push(format!("{name}: {e}")),
        }
    }
    Err(format!(
        "libvpx not found (install the libvpx package or set {ENV_PATH}); tried: {}",
        tried.join("; ")
    ))
}

impl VpxDecoder {
    pub fn new(codec: &Codec) -> PipeResult<Self> {
        let vp9 = match codec {
            Codec::Vp8 => false,
            Codec::Vp9 => true,
            other => return Err(format!("libvpx cannot decode {other:?}")),
        };
        let lib = open_lib()?;
        // SAFETY: signatures are from vpx_decoder.h / vp8dx.h.
        let (iface, init, api) = unsafe {
            let sym = |n: &[u8]| -> Result<*const c_void, String> {
                lib.get::<*const c_void>(n).map(|s| *s).map_err(|e| {
                    format!(
                        "libvpx: missing {}: {e}",
                        String::from_utf8_lossy(&n[..n.len() - 1])
                    )
                })
            };
            let iface_fn: FnIface = std::mem::transmute(sym(if vp9 {
                b"vpx_codec_vp9_dx\0"
            } else {
                b"vpx_codec_vp8_dx\0"
            })?);
            let init: FnInit = std::mem::transmute(sym(b"vpx_codec_dec_init_ver\0")?);
            let api = Api {
                decode: std::mem::transmute::<*const c_void, FnDecode>(sym(b"vpx_codec_decode\0")?),
                get_frame: std::mem::transmute::<*const c_void, FnGetFrame>(sym(
                    b"vpx_codec_get_frame\0",
                )?),
                destroy: std::mem::transmute::<*const c_void, FnDestroy>(sym(
                    b"vpx_codec_destroy\0",
                )?),
                err_str: std::mem::transmute::<*const c_void, FnErrStr>(sym(
                    b"vpx_codec_err_to_string\0",
                )?),
                _lib: lib,
            };
            (iface_fn(), init, api)
        };
        if iface.is_null() {
            return Err("libvpx was built without a decoder for this codec".into());
        }
        let mut ctx = Box::new(VpxCodecCtx {
            name: null(),
            iface: null(),
            err: 0,
            err_detail: null(),
            init_flags: 0,
            config: null(),
            priv_: null_mut(),
        });
        let cfg = VpxDecCfg {
            threads: std::thread::available_parallelism()
                .map_or(2, |n| n.get().clamp(1, 4) as c_uint),
            w: 0,
            h: 0,
        };
        // SAFETY: ctx is our buffer of the right size; cfg outlives the call.
        let r = unsafe {
            init(
                &raw mut *ctx,
                iface,
                &raw const cfg,
                0,
                VPX_DECODER_ABI_VERSION,
            )
        };
        if r != 0 {
            return Err(format!("vpx_codec_dec_init: {}", err_text(&api, r)));
        }
        Ok(Self { api, ctx, vp9 })
    }

    fn drain(&mut self, out: &mut Vec<DecodedFrame>) {
        let mut iter: *const c_void = null();
        loop {
            // SAFETY: the context is initialized; the image lives until the next decode.
            let img = unsafe { (self.api.get_frame)(&raw mut *self.ctx, &raw mut iter) };
            let Some(img) = (unsafe { img.as_ref() }) else {
                break;
            };
            out.extend(convert(img));
        }
    }
}

fn err_text(api: &Api, code: c_int) -> String {
    // SAFETY: err_to_string returns a static string.
    let p = unsafe { (api.err_str)(code) };
    if p.is_null() {
        return format!("code {code}");
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn convert(img: &VpxImage) -> Option<DecodedFrame> {
    let hbd = img.fmt & FMT_HIGHBITDEPTH != 0;
    let (chroma, flip) = match img.fmt & !FMT_HIGHBITDEPTH {
        FMT_I420 => (Chroma::Sub420, false),
        FMT_YV12 => (Chroma::Sub420, true),
        FMT_I422 => (Chroma::Sub422, false),
        FMT_I444 => (Chroma::Full, false),
        _ => return None,
    };
    let (w, h) = (img.d_w as usize, img.d_h as usize);
    if w == 0 || h == 0 || img.planes[..3].iter().any(|p| p.is_null()) {
        return None;
    }
    let bit_depth = if hbd { img.bit_depth.max(9) } else { 8 };
    let bpp = if hbd { 2 } else { 1 };
    let (sx, sy) = (img.x_chroma_shift, img.y_chroma_shift);
    let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
    let plane = |i: usize, width: usize, rows: usize| -> Option<Plane<'_>> {
        let stride = usize::try_from(img.stride[i]).ok()?;
        // SAFETY: libvpx allocates stride*rows bytes per plane.
        Some(Plane {
            data: unsafe {
                std::slice::from_raw_parts(img.planes[i], stride * (rows - 1) + width * bpp)
            },
            stride,
        })
    };
    let (ui, vi) = if flip { (2, 1) } else { (1, 2) };
    let matrix = match img.cs {
        CS_BT_709 => Matrix::Bt709,
        CS_BT_2020 => Matrix::Bt2020,
        0 => Matrix::guess(img.d_h),
        _ => Matrix::Bt601,
    };
    let yuv = YuvImage {
        width: img.d_w,
        height: img.d_h,
        bit_depth,
        chroma,
        matrix,
        full_range: img.range == 1,
        y: plane(0, w, h)?,
        uv: ChromaPlanes::Planar {
            u: plane(ui, cw, ch)?,
            v: plane(vi, cw, ch)?,
        },
    };
    Some(DecodedFrame {
        pts: Duration::from_micros(img.user_priv as u64),
        data: FrameData::Cpu(yuv::to_bgra(&yuv)?),
    })
}

impl Decoder for VpxDecoder {
    fn name(&self) -> &'static str {
        if self.vp9 {
            "libvpx VP9 (CPU)"
        } else {
            "libvpx VP8 (CPU)"
        }
    }

    fn decode(&mut self, packet: &Packet) -> PipeResult<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        let Ok(len) = c_uint::try_from(packet.data.len()) else {
            return Ok(out);
        };
        // pts in microseconds stored in the pointer itself: libvpx never dereferences it
        let tag = usize::try_from(packet.pts.as_micros()).unwrap_or(usize::MAX) as *mut c_void;
        // SAFETY: the packet outlives the call; deadline 0 = best quality.
        let r = unsafe { (self.api.decode)(&raw mut *self.ctx, packet.data.as_ptr(), len, tag, 0) };
        if r == 0 {
            self.drain(&mut out);
        }
        // skip a broken frame: the next keyframe fixes everything
        Ok(out)
    }

    fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        // NULL data = end of stream; the decoder returns whatever it was holding
        // SAFETY: this is how vpx_decoder.h documents flush.
        let r = unsafe { (self.api.decode)(&raw mut *self.ctx, null(), 0, null_mut(), 0) };
        if r == 0 {
            self.drain(&mut out);
        }
        Ok(out)
    }

    fn reset(&mut self) {
        // VP8/VP9 start with a keyframe, which resets all references.
        let _ = self.flush();
    }
}

impl Drop for VpxDecoder {
    fn drop(&mut self) {
        // SAFETY: the context is initialized in new and destroyed exactly once.
        unsafe { (self.api.destroy)(&raw mut *self.ctx) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_libvpx_on_lp64() {
        if cfg!(target_pointer_width = "64") {
            assert_eq!(std::mem::size_of::<VpxCodecCtx>(), 56);
        }
        assert_eq!(std::mem::offset_of!(VpxImage, planes), 48);
        assert_eq!(std::mem::offset_of!(VpxImage, stride), 80);
        assert_eq!(std::mem::offset_of!(VpxImage, user_priv), 104);
    }
}
