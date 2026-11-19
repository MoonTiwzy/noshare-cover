//! VP8 and VP9 on CPU: system libvpx via dlopen.
//!
//! Our own minimal FFI based on the libvpx headers (vpx_decoder.h, vpx_image.h):
//! six functions and two structs. The libvpx decoder ABI hasn't changed since 1.8
//! (VPX_DECODER_ABI_VERSION = 12), but the soname keeps bumping, so we try them all.
//! pts travels via `user_priv`: VP8/VP9 don't reorder frames, but hidden
//! (altref) frames produce no picture, and the timestamp stays with its own frame.
//!
//! A packet libvpx refuses is skipped: after a broken frame it refuses every inter frame
//! up to the next keyframe, which fixes everything. A pass that has shown nothing yet is
//! reported as undecodable, with libvpx's reason, when keyframes keep being refused, no
//! keyframe comes at all, or the pass ends. Once a picture came out, the next keyframe or
//! the loop is waited for, up to just short of the pipeline's own guard, and the reason
//! goes to the NOSHARE_COVER_DEBUG trace.

use std::ffi::{CStr, c_char, c_int, c_long, c_uint, c_void};
use std::io::Write;
use std::path::PathBuf;
use std::ptr::{null, null_mut};
use std::time::Duration;

use libloading::Library;

use super::Refusals;
use crate::frame::FrameData;
use crate::media::video::pipeline::{Codec, DecodedFrame, Decoder, Packet, PipeResult};
use crate::media::video::yuv::{Chroma, ChromaPlanes, Converter, Matrix, Plane, YuvImage};

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
/// RGB (VP9 profile 1 and 3): the planes are G, B and R, always full range.
const CS_SRGB: c_int = 7;

/// Keyframes refused in a row (with nothing decoded in between) that end a video whose
/// pass has shown nothing yet.
const MAX_REFUSED_KEYFRAMES: u32 = 2;
/// Packets refused in a row that end it when no keyframe comes along: longer than any GOP
/// a cover is likely to have.
const MAX_REFUSED: u32 = 600;
/// Packets refused in a row that end a video whose pass has shown pictures. Just short of
/// the pipeline's guard (4096 packets without a frame), which would otherwise end it as a
/// decoder that "produces no frames" and lose libvpx's reason.
const MAX_REFUSED_PLAYED: u32 = 4000;
/// The step-by-step trace file the shim writes to (see shim/main.cpp).
const TRACE_ENV: &str = "NOSHARE_COVER_DEBUG";

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
struct VpxStreamInfo {
    sz: c_uint,
    w: c_uint,
    h: c_uint,
    is_kf: c_uint,
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
type FnErrDetail = unsafe extern "C" fn(*const VpxCodecCtx) -> *const c_char;
type FnPeek = unsafe extern "C" fn(*const c_void, *const u8, c_uint, *mut VpxStreamInfo) -> c_int;

struct Api {
    decode: FnDecode,
    get_frame: FnGetFrame,
    destroy: FnDestroy,
    err_str: FnErrStr,
    err_detail: FnErrDetail,
    peek: FnPeek,
    // keep the library loaded while pointers to its functions are alive
    _lib: Library,
}

pub struct VpxDecoder {
    api: Api,
    // Box: libvpx keeps a pointer to the context internally, so the address must not change
    ctx: Box<VpxCodecCtx>,
    /// the codec's interface, a static object in the library
    iface: *const c_void,
    vp9: bool,
    refusals: Refusals,
    /// why libvpx refused the packets of the current run
    why: String,
    /// the trace file, if NOSHARE_COVER_DEBUG names one
    trace: Option<PathBuf>,
    /// the reason last written there: a loop that breaks at the same packet every pass
    /// writes it once
    traced: String,
    conv: Converter,
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
                err_detail: std::mem::transmute::<*const c_void, FnErrDetail>(sym(
                    b"vpx_codec_error_detail\0",
                )?),
                peek: std::mem::transmute::<*const c_void, FnPeek>(sym(
                    b"vpx_codec_peek_stream_info\0",
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
        Ok(Self {
            api,
            ctx,
            iface,
            vp9,
            refusals: Refusals::default(),
            why: String::new(),
            trace: std::env::var_os(TRACE_ENV)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            traced: String::new(),
            conv: Converter::default(),
        })
    }

    /// A video that played starts refusing packets: no error yet (the next keyframe or the
    /// loop may fix it), but a clip that ends this way leaves a line in the trace, once
    /// per reason.
    fn trace_refusal(&mut self) {
        let Some(path) = &self.trace else {
            return;
        };
        if self.traced == self.why {
            return;
        }
        self.traced.clone_from(&self.why);
        let line = format!(
            "[noshare-cover] {}: refused a packet, skipping to the next keyframe: {}\n",
            self.name(),
            self.why
        );
        // one write in append mode, as the shim's own lines
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
        {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// Whatever libvpx still holds (NULL data = end of stream).
    fn drain_end(&mut self) -> Vec<DecodedFrame> {
        let mut out = Vec::new();
        // SAFETY: this is how vpx_decoder.h documents flush.
        let r = unsafe { (self.api.decode)(&raw mut *self.ctx, null(), 0, null_mut(), 0) };
        if r == 0 {
            self.drain(&mut out);
        }
        out
    }

    /// Whether a refused packet was a keyframe. The container says so, unless libvpx reads
    /// an inter frame in it: an MP4 without a sync sample table calls every packet a keyframe.
    fn keyframe(&self, packet: &Packet, len: c_uint) -> bool {
        if !packet.keyframe {
            return false;
        }
        let mut si = VpxStreamInfo {
            sz: std::mem::size_of::<VpxStreamInfo>() as c_uint,
            w: 0,
            h: 0,
            is_kf: 0,
        };
        // SAFETY: iface is the decoder's; the packet and si outlive the call.
        let r = unsafe { (self.api.peek)(self.iface, packet.data.as_ptr(), len, &raw mut si) };
        // VP9 sets is_kf only once it has read the frame type; VP8 sets it from the frame
        // tag and turns an inter frame down with an error
        if self.vp9 && r != 0 {
            return true;
        }
        si.is_kf != 0
    }

    /// libvpx's words for the error `code` it just returned.
    fn error(&self, code: c_int) -> String {
        let what = err_text(&self.api, code);
        // SAFETY: the context is initialized; the detail lives until the next libvpx call.
        let detail = unsafe { (self.api.err_detail)(&raw const *self.ctx) };
        if detail.is_null() {
            return what;
        }
        let detail = unsafe { CStr::from_ptr(detail) }.to_string_lossy();
        format!("{what} ({detail})")
    }

    fn drain(&mut self, out: &mut Vec<DecodedFrame>) {
        let mut iter: *const c_void = null();
        loop {
            // SAFETY: the context is initialized; the image lives until the next decode.
            let img = unsafe { (self.api.get_frame)(&raw mut *self.ctx, &raw mut iter) };
            let Some(img) = (unsafe { img.as_ref() }) else {
                break;
            };
            out.extend(convert(&mut self.conv, img));
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

fn convert(conv: &mut Converter, img: &VpxImage) -> Option<DecodedFrame> {
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
        // planes 0, 1, 2 are G, B, R: y, u, v to the converter
        CS_SRGB => Matrix::Identity,
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
        data: FrameData::Cpu(conv.convert(&yuv)?),
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
        // libvpx takes an empty buffer for a broken call, and there is nothing to decode
        if packet.data.is_empty() {
            return Ok(out);
        }
        let Ok(len) = c_uint::try_from(packet.data.len()) else {
            return Ok(out);
        };
        // pts in microseconds stored in the pointer itself: libvpx never dereferences it
        let tag = usize::try_from(packet.pts.as_micros()).unwrap_or(usize::MAX) as *mut c_void;
        // SAFETY: the packet outlives the call; deadline 0 = best quality.
        let r = unsafe { (self.api.decode)(&raw mut *self.ctx, packet.data.as_ptr(), len, tag, 0) };
        if r == 0 {
            self.drain(&mut out);
            self.refusals.note(false, false, !out.is_empty());
            return Ok(out);
        }
        // skip a broken frame: the next keyframe fixes everything
        let why = self.error(r);
        let keyframe = self.keyframe(packet, len);
        if keyframe || self.refusals.run() == 0 {
            self.why = why;
        }
        self.refusals.note(true, keyframe, false);
        // once the pass has shown pictures the video decodes: a later keyframe or the loop
        // picks it up again, unless the run reaches the pipeline's own guard
        let undecodable = if self.refusals.shown() {
            if self.refusals.run() == 1 {
                self.trace_refusal();
            }
            self.refusals.run() >= MAX_REFUSED_PLAYED
        } else {
            self.refusals.keyframes() >= MAX_REFUSED_KEYFRAMES || self.refusals.run() >= MAX_REFUSED
        };
        if undecodable {
            return Err(format!("libvpx can't decode this video: {}", self.why));
        }
        Ok(out)
    }

    fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
        let out = self.drain_end();
        self.refusals.flushed(!out.is_empty());
        // without a loop no other keyframe comes, and a loop would refuse it all again
        if self.refusals.blank_pass() {
            return Err(format!("libvpx can't decode this video: {}", self.why));
        }
        Ok(out)
    }

    fn reset(&mut self) {
        // VP8/VP9 start with a keyframe, which resets all references.
        let _ = self.drain_end();
        self.refusals.reset();
    }

    fn set_target(&mut self, width: u32, height: u32) {
        self.conv.set_target(width, height);
    }

    fn idle(&mut self) {
        self.conv.idle();
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
    use crate::media::video::decode::testing;
    use crate::media::video::demux;

    #[test]
    fn struct_layouts_match_libvpx_on_lp64() {
        if cfg!(target_pointer_width = "64") {
            assert_eq!(std::mem::size_of::<VpxCodecCtx>(), 56);
        }
        assert_eq!(std::mem::offset_of!(VpxImage, planes), 48);
        assert_eq!(std::mem::offset_of!(VpxImage, stride), 80);
        assert_eq!(std::mem::offset_of!(VpxImage, user_priv), 104);
        assert_eq!(std::mem::size_of::<VpxStreamInfo>(), 16);
    }

    /// A 4x2 picture in libvpx's I444 layout, all one colour.
    fn image(planes: &mut [Vec<u8>; 3], cs: c_int, range: c_int) -> VpxImage {
        VpxImage {
            fmt: FMT_I444,
            cs,
            range,
            w: 4,
            h: 2,
            bit_depth: 8,
            d_w: 4,
            d_h: 2,
            r_w: 4,
            r_h: 2,
            x_chroma_shift: 0,
            y_chroma_shift: 0,
            planes: [
                planes[0].as_mut_ptr(),
                planes[1].as_mut_ptr(),
                planes[2].as_mut_ptr(),
                null_mut(),
            ],
            stride: [4, 4, 4, 0],
            bps: 24,
            user_priv: null_mut(),
            img_data: null_mut(),
            img_data_owner: 0,
            self_allocd: 0,
            fb_priv: null_mut(),
        }
    }

    fn first_pixel(f: &DecodedFrame) -> [u8; 4] {
        #[allow(irrefutable_let_patterns)]
        let FrameData::Cpu(c) = &f.data else {
            panic!("a CPU frame")
        };
        c.pixels[..4].try_into().unwrap()
    }

    /// VP9 profile 1 can carry RGB: VPX_CS_SRGB, full range, the planes G, B and R.
    #[test]
    fn srgb_planes_are_green_blue_red() {
        let mut planes = [vec![128; 8], vec![192; 8], vec![48; 8]];
        let img = image(&mut planes, CS_SRGB, 1);
        let f = convert(&mut Converter::default(), &img).unwrap();
        assert_eq!(first_pixel(&f), [192, 128, 48, 255]);
    }

    /// Every packet of a clip, in order.
    fn packets(clip: &testing::Clip) -> (Codec, Vec<Packet>) {
        let mut d = demux::open(clip.path()).unwrap();
        let codec = d.info().codec.clone();
        let mut all = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            all.push(p);
        }
        (codec, all)
    }

    fn decoder(codec: &Codec) -> Option<VpxDecoder> {
        VpxDecoder::new(codec)
            .inspect_err(|e| eprintln!("no libvpx ({e}), skipping"))
            .ok()
    }

    /// The same through libvpx itself: an RGB VP9 made by ffmpeg keeps its colour.
    #[test]
    fn srgb_vp9_keeps_its_colour() {
        let Some(clip) = testing::ffmpeg(
            "srgb.webm",
            "color=c=0x3080c0:s=32x32:r=10:d=0.3",
            &["-pix_fmt", "gbrp", "-c:v", "libvpx-vp9", "-lossless", "1"],
        ) else {
            return;
        };
        let (codec, packets) = packets(&clip);
        let Some(mut dec) = decoder(&codec) else {
            return;
        };
        let f = dec.decode(&packets[0]).unwrap().pop().expect("a picture");
        let [b, g, r, a] = first_pixel(&f);
        assert!(
            b.abs_diff(0xc0) <= 2 && g.abs_diff(0x80) <= 2 && r.abs_diff(0x30) <= 2 && a == 255,
            "{:?}",
            [b, g, r, a]
        );
    }

    /// A VP9 clip of `frames` frames at 30 fps with a keyframe every `gop` frames.
    fn vp9_clip(name: &str, gop: u32, frames: u32) -> Option<testing::Clip> {
        vpx_clip(name, "libvpx-vp9", gop, frames)
    }

    /// The same made by `encoder`: libvpx (VP8) or libvpx-vp9.
    fn vpx_clip(name: &str, encoder: &str, gop: u32, frames: u32) -> Option<testing::Clip> {
        testing::ffmpeg(
            name,
            "testsrc=size=64x64:rate=30",
            &[
                "-frames:v",
                &frames.to_string(),
                "-c:v",
                encoder,
                "-g",
                &gop.to_string(),
                "-auto-alt-ref",
                "0",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
            ],
        )
    }

    /// Where the keyframes are, by the container.
    fn keyframes(packets: &[Packet]) -> Vec<usize> {
        let keys: Vec<usize> = (0..packets.len())
            .filter(|&i| packets[i].keyframe)
            .collect();
        assert_eq!(keys.first(), Some(&0), "a clip starts with a keyframe");
        keys
    }

    /// A broken inter frame makes libvpx refuse every frame up to the next keyframe; that
    /// is no reason to give the video up, however long the wait.
    #[test]
    fn one_broken_packet_waits_for_the_next_keyframe() {
        let Some(clip) = vp9_clip("gop100.webm", 100, 120) else {
            return;
        };
        let (codec, mut packets) = packets(&clip);
        if decoder(&codec).is_none() {
            return;
        }
        let keys = keyframes(&packets);
        assert!(keys.len() >= 2 && keys[1] > 60, "{keys:?}");
        let next = keys[1];
        packets[10].data[0] = 0;
        let shown = shown(&mut decoder(&codec).unwrap(), &packets);
        assert!(
            shown.contains(&9) && !shown.contains(&(next - 1)),
            "refused from the broken packet to the keyframe: {shown:?}"
        );
        assert!(
            shown.contains(&next) && shown.contains(&(packets.len() - 1)),
            "{shown:?}"
        );
    }

    /// A broken first keyframe in a container that calls every packet a keyframe (an MP4
    /// without a sync sample table): the inter frames libvpx refuses after it are no
    /// keyframes, so they don't count towards the two-keyframe rule, and the video starts
    /// at the next real keyframe. VP9 reads the frame type in the frame header, VP8 in the
    /// frame tag.
    #[test]
    fn inter_frames_called_keyframes_by_the_container_are_not_counted_as_keyframes() {
        // VP9: no frame marker; VP8: no start code, the tag still says keyframe
        for (encoder, broken_byte) in [("libvpx-vp9", 0), ("libvpx", 3)] {
            let Some(clip) = vpx_clip(&format!("allkey-{encoder}.webm"), encoder, 10, 40) else {
                continue;
            };
            let (codec, mut packets) = packets(&clip);
            let Some(mut dec) = decoder(&codec) else {
                return;
            };
            let keys = keyframes(&packets);
            assert!(keys.len() >= 2 && keys[1] >= 3, "{encoder}: {keys:?}");
            packets[0].data[broken_byte] = 0;
            for p in &mut packets {
                p.keyframe = true;
            }
            let shown = shown(&mut dec, &packets);
            assert_eq!(
                shown.first(),
                Some(&keys[1]),
                "{encoder}: from the next real keyframe on: {shown:?}"
            );
            assert!(shown.contains(&(packets.len() - 1)), "{encoder}: {shown:?}");
        }
    }

    /// Decodes every packet; the packets that gave a picture.
    fn shown(dec: &mut VpxDecoder, packets: &[Packet]) -> Vec<usize> {
        let mut shown = Vec::new();
        for (i, p) in packets.iter().enumerate() {
            let frames = dec
                .decode(p)
                .unwrap_or_else(|e| panic!("packet {i} after {} pictures: {e}", shown.len()));
            if !frames.is_empty() {
                shown.push(i);
            }
        }
        dec.flush().unwrap();
        shown
    }

    /// The same with a GOP longer than any limit on refusals: a video that played waits for
    /// its next keyframe, however far away.
    #[test]
    fn a_video_that_played_waits_out_a_long_gop() {
        let Some(clip) = vp9_clip("gop650.webm", 650, 700) else {
            return;
        };
        let (codec, mut packets) = packets(&clip);
        let Some(mut dec) = decoder(&codec) else {
            return;
        };
        let keys = keyframes(&packets);
        assert!(
            keys.len() >= 2 && keys[1] > 10 + MAX_REFUSED as usize,
            "{keys:?}"
        );
        packets[10].data[0] = 0;
        let shown = shown(&mut dec, &packets);
        assert!(
            shown.contains(&9) && !shown.contains(&(keys[1] - 1)),
            "{shown:?}"
        );
        assert!(
            shown.contains(&keys[1]) && shown.contains(&(packets.len() - 1)),
            "{shown:?}"
        );
    }

    /// Two broken keyframes in a row in a video that played: playback picks up at the
    /// keyframe after them.
    #[test]
    fn a_video_that_played_outlasts_two_broken_keyframes() {
        let Some(clip) = vp9_clip("twokeys.webm", 10, 60) else {
            return;
        };
        let (codec, mut packets) = packets(&clip);
        let Some(mut dec) = decoder(&codec) else {
            return;
        };
        let keys = keyframes(&packets);
        assert!(keys.len() >= 4, "{keys:?}");
        packets[keys[1]].data[0] = 0;
        packets[keys[2]].data[0] = 0;
        let shown = shown(&mut dec, &packets);
        assert!(
            shown.contains(&(keys[1] - 1)) && !shown.contains(&(keys[3] - 1)),
            "{shown:?}"
        );
        assert!(
            shown.contains(&keys[3]) && shown.contains(&(packets.len() - 1)),
            "{shown:?}"
        );
    }

    /// A packet libvpx refuses whatever came before: no frame marker.
    fn refused(pts: Duration) -> Packet {
        Packet {
            pts,
            keyframe: false,
            data: vec![0; 16],
        }
    }

    /// A video that played waits for a keyframe up to just short of the pipeline's guard,
    /// then ends with libvpx's reason. The trace gets the reason once, at the first refusal.
    #[test]
    fn a_video_that_played_ends_with_the_reason_short_of_the_pipeline_guard() {
        let Some(clip) = vp9_clip("played.webm", 10, 1) else {
            return;
        };
        let (codec, packets) = packets(&clip);
        let Some(mut dec) = decoder(&codec) else {
            return;
        };
        let trace = clip.path().with_file_name("trace.log");
        dec.trace = Some(trace.clone());
        let key = &packets[0];
        let bad = refused(key.pts);
        let refuse = |dec: &mut VpxDecoder, n: u32| {
            for i in 0..n {
                let frames = dec
                    .decode(&bad)
                    .unwrap_or_else(|e| panic!("refusal {} of {n}: {e}", i + 1));
                assert!(frames.is_empty());
            }
        };
        assert!(!dec.decode(key).unwrap().is_empty(), "the keyframe shows");
        refuse(&mut dec, MAX_REFUSED_PLAYED - 1);
        assert!(
            !dec.decode(key).unwrap().is_empty(),
            "a keyframe in time picks the video up again"
        );
        refuse(&mut dec, MAX_REFUSED_PLAYED - 1);
        let e = dec.decode(&bad).unwrap_err();
        let why = e
            .strip_prefix("libvpx can't decode this video: ")
            .unwrap_or_else(|| panic!("{e}"));
        let trace = std::fs::read_to_string(&trace).unwrap();
        assert_eq!(
            trace,
            format!(
                "[noshare-cover] libvpx VP9 (CPU): refused a packet, skipping to the next keyframe: {why}\n"
            ),
            "one line for both runs: the same reason"
        );
    }

    /// Through the pipeline: such a video ends with libvpx's reason, not as a decoder that
    /// produces no frames.
    #[test]
    fn the_pipeline_ends_a_video_that_played_with_libvpxs_reason() {
        use crate::media::video::pipeline::{Demuxer, Next, Pipeline, VideoInfo};

        /// The clip's keyframe, then refused packets for ever.
        struct Endless {
            info: VideoInfo,
            key: Packet,
            n: u64,
        }
        impl Demuxer for Endless {
            fn info(&self) -> &VideoInfo {
                &self.info
            }
            fn next_packet(&mut self) -> PipeResult<Option<Packet>> {
                self.n += 1;
                Ok(Some(match self.n {
                    1 => self.key.clone(),
                    n => refused(Duration::from_millis(33 * n)),
                }))
            }
            fn rewind(&mut self) -> PipeResult<()> {
                self.n = 0;
                Ok(())
            }
        }

        let Some(clip) = vp9_clip("endless.webm", 10, 1) else {
            return;
        };
        let info = demux::open(clip.path()).unwrap().info().clone();
        let (codec, packets) = packets(&clip);
        let Some(dec) = decoder(&codec) else {
            return;
        };
        let demux = Endless {
            info,
            key: packets[0].clone(),
            n: 0,
        };
        let mut pipe = Pipeline::new(Box::new(demux), Box::new(dec), false);
        assert!(matches!(pipe.next_frame(), Ok(Next::Frame(_))));
        match pipe.next_frame() {
            Err(e) => assert!(e.starts_with("libvpx can't decode this video: "), "{e}"),
            Ok(_) => panic!("a frame or an end out of refused packets"),
        }
    }

    /// A stream libvpx can't decode at all ends with libvpx's own reason, as soon as two
    /// keyframes in a row are refused, instead of hundreds of empty packets later.
    #[test]
    fn a_stream_libvpx_refuses_ends_with_its_reason() {
        let Some(clip) = vp9_clip("gop10.webm", 10, 120) else {
            return;
        };
        let (codec, mut packets) = packets(&clip);
        let Some(mut dec) = decoder(&codec) else {
            return;
        };
        let keys = keyframes(&packets);
        assert!(keys.len() >= 2, "{keys:?}");
        for p in &mut packets {
            p.data[0] = 0; // no frame marker
        }
        let failed = packets.iter().position(|p| dec.decode(p).is_err());
        assert_eq!(failed, Some(keys[1]), "the second keyframe");
        let e = dec.decode(&packets[keys[1] + 1]).unwrap_err();
        assert!(e.starts_with("libvpx can't decode this video: "), "{e}");
        // nothing to decode is not a refusal
        let mut dec = decoder(&codec).unwrap();
        let empty = Packet {
            data: Vec::new(),
            ..packets[0].clone()
        };
        for _ in 0..1000 {
            assert!(dec.decode(&empty).unwrap().is_empty());
        }
    }

    /// A short clip with one keyframe that libvpx refuses: the pass shows nothing, so its
    /// end ends the video with the reason, played once or looped.
    #[test]
    fn a_pass_that_shows_nothing_ends_with_the_reason() {
        let Some(clip) = testing::ffmpeg(
            "short.webm",
            "testsrc=size=64x64:rate=30:duration=1",
            &[
                "-c:v",
                "libvpx-vp9",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
            ],
        ) else {
            return;
        };
        let (codec, mut packets) = packets(&clip);
        let Some(mut dec) = decoder(&codec) else {
            return;
        };
        assert_eq!(keyframes(&packets), [0]);
        for p in &mut packets {
            p.data[0] = 0;
        }
        for p in &packets {
            assert!(dec.decode(p).is_ok(), "one refused keyframe is no verdict");
        }
        let e = dec.flush().unwrap_err();
        assert!(e.starts_with("libvpx can't decode this video: "), "{e}");
    }
}
