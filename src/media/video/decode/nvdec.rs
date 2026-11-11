//! NVIDIA: NVDEC directly, via the driver's libcuda + libnvcuvid (dlopen).
//!
//! Bitstream parsing is done by the driver's built-in parser (cuvidCreateVideoParser):
//! it manages the DPB, display order and timestamps itself. We just feed it
//! packets, create a decoder for the stream format and collect finished frames.
//! A frame is copied from video memory as NV12/P016 with a single cuMemcpy2D and
//! then goes through the common YUV → BGRA path. For a window-sized cover this is
//! cheaper and more reliable than CUDA↔EGL interop inside the compositor.
//!
//! Nothing is linked: without the NVIDIA driver `new()` returns an error and
//! backend.rs falls back to VA-API or CPU.

use std::ffi::{CStr, c_char, c_int, c_longlong, c_short, c_uint, c_ulong, c_void};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::time::Duration;

use libloading::Library;

use crate::frame::FrameData;
use crate::media::video::pipeline::{Codec, DecodedFrame, Decoder, Packet, PipeResult};
use crate::media::video::yuv::{self, Chroma, ChromaPlanes, Matrix, Plane, YuvImage};

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuDevicePtr = u64;
type CuVideoParser = *mut c_void;
type CuVideoDecoder = *mut c_void;
type CuVideoCtxLock = *mut c_void;

// cudaVideoCodec
const CODEC_H264: c_int = 4;
const CODEC_HEVC: c_int = 8;
const CODEC_VP8: c_int = 9;
const CODEC_VP9: c_int = 10;
const CODEC_AV1: c_int = 11;
// cudaVideoChromaFormat
const CHROMA_MONO: c_int = 0;
const CHROMA_420: c_int = 1;
// cudaVideoSurfaceFormat
const SURFACE_NV12: c_int = 0;
const SURFACE_P016: c_int = 1;
// cudaVideoCreateFlags / DeinterlaceMode
const CREATE_PREFER_CUVID: c_ulong = 0x04;
const DEINTERLACE_WEAVE: c_int = 0;
const DEINTERLACE_ADAPTIVE: c_int = 2;
// CUvideopacketflags
const PKT_ENDOFSTREAM: c_ulong = 0x01;
const PKT_TIMESTAMP: c_ulong = 0x02;
// CUmemorytype
const MEM_HOST: c_int = 1;
const MEM_DEVICE: c_int = 2;

/// Timestamps in microseconds, same as our Packet.
const CLOCK_RATE: c_uint = 1_000_000;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Rect16 {
    left: c_short,
    top: c_short,
    right: c_short,
    bottom: c_short,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Rect32 {
    left: c_int,
    top: c_int,
    right: c_int,
    bottom: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VideoFormat {
    codec: c_int,
    frame_rate: [c_uint; 2],
    progressive_sequence: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    min_num_decode_surfaces: u8,
    coded_width: c_uint,
    coded_height: c_uint,
    display_area: Rect32,
    chroma_format: c_int,
    bitrate: c_uint,
    display_aspect_ratio: [c_int; 2],
    /// video_format:3, video_full_range_flag:1, reserved:4; primaries; transfer; matrix
    video_signal_description: [u8; 4],
    seqhdr_data_length: c_uint,
}

#[repr(C)]
struct SourceDataPacket {
    flags: c_ulong,
    payload_size: c_ulong,
    payload: *const u8,
    timestamp: c_longlong,
}

#[repr(C)]
struct ParserDispInfo {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    repeat_first_field: c_int,
    timestamp: c_longlong,
}

type SeqCb = unsafe extern "C" fn(*mut c_void, *mut VideoFormat) -> c_int;
type PicCb = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type DispCb = unsafe extern "C" fn(*mut c_void, *mut ParserDispInfo) -> c_int;

#[repr(C)]
struct ParserParams {
    codec_type: c_int,
    max_num_decode_surfaces: c_uint,
    clock_rate: c_uint,
    error_threshold: c_uint,
    max_display_delay: c_uint,
    /// bAnnexb:1, bMemoryOptimize:1, reserved:30
    flags: c_uint,
    reserved1: [c_uint; 4],
    user_data: *mut c_void,
    sequence_cb: Option<SeqCb>,
    decode_cb: Option<PicCb>,
    display_cb: Option<DispCb>,
    op_point_cb: *const c_void,
    sei_cb: *const c_void,
    reserved2: [*mut c_void; 5],
    ext_video_info: *mut c_void,
}

#[repr(C)]
struct DecodeCreateInfo {
    width: c_ulong,
    height: c_ulong,
    num_decode_surfaces: c_ulong,
    codec_type: c_int,
    chroma_format: c_int,
    creation_flags: c_ulong,
    bit_depth_minus8: c_ulong,
    intra_decode_only: c_ulong,
    max_width: c_ulong,
    max_height: c_ulong,
    reserved1: c_ulong,
    display_area: Rect16,
    output_format: c_int,
    deinterlace_mode: c_int,
    target_width: c_ulong,
    target_height: c_ulong,
    num_output_surfaces: c_ulong,
    vid_lock: CuVideoCtxLock,
    target_rect: Rect16,
    enable_histogram: c_ulong,
    enable_decode_features: c_ulong,
    reserved2: [c_ulong; 3],
}

#[repr(C)]
struct ProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: c_uint,
    reserved_zero: c_uint,
    raw_input_dptr: u64,
    raw_input_pitch: c_uint,
    raw_input_format: c_uint,
    raw_output_dptr: u64,
    raw_output_pitch: c_uint,
    reserved1: c_uint,
    output_stream: *mut c_void,
    reserved: [c_uint; 46],
    histogram_dptr: *mut u64,
    proc_ext: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct DecodeCaps {
    codec_type: c_int,
    chroma_format: c_int,
    bit_depth_minus8: c_uint,
    reserved1: [c_uint; 3],
    is_supported: u8,
    num_nvdecs: u8,
    output_format_mask: u16,
    max_width: c_uint,
    max_height: c_uint,
    max_mb_count: c_uint,
    min_width: u16,
    min_height: u16,
    histogram_supported: u8,
    counter_bit_depth: u8,
    max_histogram_bins: u16,
    decode_stats_supported: u8,
    reserved4: [u8; 3],
    reserved3: [c_uint; 9],
}

#[repr(C)]
struct Memcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: c_int,
    src_host: *const c_void,
    src_device: CuDevicePtr,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: c_int,
    dst_host: *mut c_void,
    dst_device: CuDevicePtr,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

struct Cuda {
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    device_get_pci_bus_id: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    ctx_create: unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult,
    ctx_destroy: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_push: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_pop: unsafe extern "C" fn(*mut CuContext) -> CuResult,
    memcpy2d: unsafe extern "C" fn(*const Memcpy2D) -> CuResult,
    get_error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    _lib: Library,
}

struct Cuvid {
    get_decoder_caps: unsafe extern "C" fn(*mut DecodeCaps) -> CuResult,
    create_decoder: unsafe extern "C" fn(*mut CuVideoDecoder, *mut DecodeCreateInfo) -> CuResult,
    destroy_decoder: unsafe extern "C" fn(CuVideoDecoder) -> CuResult,
    decode_picture: unsafe extern "C" fn(CuVideoDecoder, *mut c_void) -> CuResult,
    map_frame: unsafe extern "C" fn(
        CuVideoDecoder,
        c_int,
        *mut u64,
        *mut c_uint,
        *mut ProcParams,
    ) -> CuResult,
    unmap_frame: unsafe extern "C" fn(CuVideoDecoder, u64) -> CuResult,
    create_parser: unsafe extern "C" fn(*mut CuVideoParser, *mut ParserParams) -> CuResult,
    parse: unsafe extern "C" fn(CuVideoParser, *mut SourceDataPacket) -> CuResult,
    destroy_parser: unsafe extern "C" fn(CuVideoParser) -> CuResult,
    ctx_lock_create: unsafe extern "C" fn(*mut CuVideoCtxLock, CuContext) -> CuResult,
    ctx_lock_destroy: unsafe extern "C" fn(CuVideoCtxLock) -> CuResult,
    _lib: Library,
}

unsafe fn sym<T: Copy>(lib: &Library, name: &str) -> Result<T, String> {
    let cname = format!("{name}\0");
    // SAFETY: the caller picks T according to the NVIDIA header.
    unsafe { lib.get::<T>(cname.as_bytes()) }
        .map(|s| *s)
        .map_err(|e| format!("{name}: {e}"))
}

fn open_any(names: &[&str]) -> Result<Library, String> {
    let mut errs = Vec::new();
    for n in names {
        // SAFETY: NVIDIA driver libraries; constructors have no side effects.
        match unsafe { Library::new(n) } {
            Ok(l) => return Ok(l),
            Err(e) => errs.push(format!("{n}: {e}")),
        }
    }
    Err(errs.join("; "))
}

impl Cuda {
    fn load() -> Result<Self, String> {
        let lib = open_any(&[
            "libcuda.so.1",
            "libcuda.so",
            "/run/opengl-driver/lib/libcuda.so.1",
        ])
        .map_err(|e| format!("NVIDIA driver not found ({e})"))?;
        // SAFETY: names and signatures come from dynlink_cuda.h (nv-codec-headers).
        unsafe {
            Ok(Self {
                init: sym(&lib, "cuInit")?,
                device_get_count: sym(&lib, "cuDeviceGetCount")?,
                device_get: sym(&lib, "cuDeviceGet")?,
                device_get_pci_bus_id: sym(&lib, "cuDeviceGetPCIBusId")?,
                ctx_create: sym(&lib, "cuCtxCreate_v2")?,
                ctx_destroy: sym(&lib, "cuCtxDestroy_v2")?,
                ctx_push: sym(&lib, "cuCtxPushCurrent_v2")?,
                ctx_pop: sym(&lib, "cuCtxPopCurrent_v2")?,
                memcpy2d: sym(&lib, "cuMemcpy2D_v2")?,
                get_error_string: sym(&lib, "cuGetErrorString")?,
                _lib: lib,
            })
        }
    }

    fn check(&self, what: &str, r: CuResult) -> Result<(), String> {
        if r == 0 {
            return Ok(());
        }
        let mut s: *const c_char = null();
        // SAFETY: cuGetErrorString writes a pointer to a static string.
        let text = unsafe {
            if (self.get_error_string)(r, &raw mut s) == 0 && !s.is_null() {
                CStr::from_ptr(s).to_string_lossy().into_owned()
            } else {
                format!("code {r}")
            }
        };
        Err(format!("{what}: {text}"))
    }
}

impl Cuvid {
    fn load() -> Result<Self, String> {
        let lib = open_any(&[
            "libnvcuvid.so.1",
            "libnvcuvid.so",
            "/run/opengl-driver/lib/libnvcuvid.so.1",
        ])
        .map_err(|e| format!("libnvcuvid not found ({e})"))?;
        // SAFETY: names and signatures come from dynlink_cuviddec.h / dynlink_nvcuvid.h.
        unsafe {
            Ok(Self {
                get_decoder_caps: sym(&lib, "cuvidGetDecoderCaps")?,
                create_decoder: sym(&lib, "cuvidCreateDecoder")?,
                destroy_decoder: sym(&lib, "cuvidDestroyDecoder")?,
                decode_picture: sym(&lib, "cuvidDecodePicture")?,
                map_frame: sym(&lib, "cuvidMapVideoFrame64")?,
                unmap_frame: sym(&lib, "cuvidUnmapVideoFrame64")?,
                create_parser: sym(&lib, "cuvidCreateVideoParser")?,
                parse: sym(&lib, "cuvidParseVideoData")?,
                destroy_parser: sym(&lib, "cuvidDestroyVideoParser")?,
                ctx_lock_create: sym(&lib, "cuvidCtxLockCreate")?,
                ctx_lock_destroy: sym(&lib, "cuvidCtxLockDestroy")?,
                _lib: lib,
            })
        }
    }
}

/// Everything the parser callbacks touch. Lives in a Box: the parser holds a raw pointer.
struct State {
    cuda: Cuda,
    cuvid: Cuvid,
    ctx: CuContext,
    lock: CuVideoCtxLock,
    decoder: CuVideoDecoder,
    /// format the decoder was created for
    fmt: Option<VideoFormat>,
    out_w: u32,
    out_h: u32,
    surface_h: u32,
    high_depth: bool,
    ready: Vec<DecodedFrame>,
    error: Option<String>,
    /// buffer for NV12/P016 from video memory, reused across frames
    host: Vec<u8>,
}

pub struct NvdecDecoder {
    state: Box<State>,
    parser: CuVideoParser,
    codec: c_int,
}

// Everything is used only from the decoder thread; the CUDA context is
// explicitly made current for the duration of each call.
unsafe impl Send for NvdecDecoder {}

fn cuda_codec(codec: &Codec) -> Option<c_int> {
    Some(match codec {
        Codec::H264 => CODEC_H264,
        Codec::Hevc => CODEC_HEVC,
        Codec::Vp8 => CODEC_VP8,
        Codec::Vp9 => CODEC_VP9,
        Codec::Av1 => CODEC_AV1,
        Codec::Other(_) => return None,
    })
}

/// PCI address of the device behind a render node: /sys/class/drm/renderD128/device → 0000:01:00.0
fn pci_of_node(node: &Path) -> Option<String> {
    let name = node.file_name()?;
    let dev = std::fs::read_link(Path::new("/sys/class/drm").join(name).join("device")).ok()?;
    Some(dev.file_name()?.to_string_lossy().to_ascii_lowercase())
}

/// Whether the render node belongs to an NVIDIA device (vendor 0x10de).
pub fn node_is_nvidia(node: &Path) -> bool {
    node.file_name()
        .and_then(|n| {
            std::fs::read_to_string(Path::new("/sys/class/drm").join(n).join("device/vendor")).ok()
        })
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("0x10de"))
}

impl NvdecDecoder {
    pub fn new(codec: &Codec, node: Option<&Path>) -> PipeResult<Self> {
        let cc = cuda_codec(codec).ok_or_else(|| format!("NVDEC doesn't support {codec:?}"))?;
        let cuda = Cuda::load()?;
        let cuvid = Cuvid::load()?;
        // SAFETY: initialization sequence from the NVDEC Programming Guide.
        unsafe {
            cuda.check("cuInit", (cuda.init)(0))?;
            let dev = pick_device(&cuda, node)?;
            let mut ctx: CuContext = null_mut();
            cuda.check("cuCtxCreate", (cuda.ctx_create)(&raw mut ctx, 0, dev))?;
            // cuCtxCreate makes the context current; pop it right away and use push/pop from here on
            let mut prev: CuContext = null_mut();
            (cuda.ctx_pop)(&raw mut prev);

            let mut state = Box::new(State {
                cuda,
                cuvid,
                ctx,
                lock: null_mut(),
                decoder: null_mut(),
                fmt: None,
                out_w: 0,
                out_h: 0,
                surface_h: 0,
                high_depth: false,
                ready: Vec::new(),
                error: None,
                host: Vec::new(),
            });
            let supported = state.with_ctx(|s| {
                let mut caps = DecodeCaps {
                    codec_type: cc,
                    chroma_format: CHROMA_420,
                    ..Default::default()
                };
                s.cuda.check(
                    "cuvidGetDecoderCaps",
                    (s.cuvid.get_decoder_caps)(&raw mut caps),
                )?;
                Ok(caps.is_supported != 0)
            });
            match supported {
                Ok(true) => {}
                Ok(false) => {
                    state.destroy();
                    return Err(format!("this GPU can't decode {codec:?} via NVDEC"));
                }
                Err(e) => {
                    state.destroy();
                    return Err(e);
                }
            }
            let r = (state.cuvid.ctx_lock_create)(&raw mut state.lock, state.ctx);
            if let Err(e) = state.cuda.check("cuvidCtxLockCreate", r) {
                state.destroy();
                return Err(e);
            }
            let mut me = Self {
                state,
                parser: null_mut(),
                codec: cc,
            };
            me.new_parser()?;
            Ok(me)
        }
    }

    fn new_parser(&mut self) -> PipeResult<()> {
        let mut p = ParserParams {
            codec_type: self.codec,
            max_num_decode_surfaces: 1, // the sequence callback will adjust it
            clock_rate: CLOCK_RATE,
            error_threshold: 100,
            max_display_delay: 0,
            flags: 0,
            reserved1: [0; 4],
            user_data: (&raw mut *self.state).cast(),
            sequence_cb: Some(on_sequence),
            decode_cb: Some(on_decode),
            display_cb: Some(on_display),
            op_point_cb: null(),
            sei_cb: null(),
            reserved2: [null_mut(); 5],
            ext_video_info: null_mut(),
        };
        // SAFETY: p outlives the call; user_data is a Box, so its address is stable.
        let r = unsafe { (self.state.cuvid.create_parser)(&raw mut self.parser, &raw mut p) };
        self.state.cuda.check("cuvidCreateVideoParser", r)
    }

    fn drop_parser(&mut self) {
        if !self.parser.is_null() {
            // SAFETY: we created the parser and destroy it exactly once.
            unsafe { (self.state.cuvid.destroy_parser)(self.parser) };
            self.parser = null_mut();
        }
    }

    fn feed(
        &mut self,
        data: &[u8],
        flags: c_ulong,
        pts: Duration,
    ) -> PipeResult<Vec<DecodedFrame>> {
        let mut pkt = SourceDataPacket {
            flags,
            payload_size: data.len() as c_ulong,
            payload: if data.is_empty() {
                null()
            } else {
                data.as_ptr()
            },
            timestamp: c_longlong::try_from(pts.as_micros()).unwrap_or(c_longlong::MAX),
        };
        let parser = self.parser;
        // callbacks are invoked synchronously inside parse and write into state
        let r = self.state.with_ctx(|s| {
            // SAFETY: the parser is alive; the packet outlives the call.
            s.cuda.check("cuvidParseVideoData", unsafe {
                (s.cuvid.parse)(parser, &raw mut pkt)
            })
        });
        if let Some(e) = self.state.error.take() {
            return Err(e);
        }
        r?;
        Ok(std::mem::take(&mut self.state.ready))
    }
}

fn pick_device(cuda: &Cuda, node: Option<&Path>) -> Result<CuDevice, String> {
    let mut count = 0;
    // SAFETY: cuInit has already been called.
    cuda.check("cuDeviceGetCount", unsafe {
        (cuda.device_get_count)(&raw mut count)
    })?;
    if count <= 0 {
        return Err("CUDA sees no GPUs".into());
    }
    let want = node.and_then(pci_of_node);
    for i in 0..count {
        let mut dev = 0;
        // SAFETY: the index is within count.
        cuda.check("cuDeviceGet", unsafe { (cuda.device_get)(&raw mut dev, i) })?;
        let Some(want) = &want else { return Ok(dev) };
        let mut buf = [0 as c_char; 32];
        // SAFETY: 32-byte buffer; the function writes a C string no longer than len.
        if unsafe { (cuda.device_get_pci_bus_id)(buf.as_mut_ptr(), buf.len() as c_int, dev) } == 0 {
            let id = unsafe { CStr::from_ptr(buf.as_ptr()) }
                .to_string_lossy()
                .to_ascii_lowercase();
            // CUDA writes the domain with 8 digits (00000000:01:00.0), sysfs with 4
            if id.ends_with(want.as_str()) || want.ends_with(id.trim_start_matches('0')) {
                return Ok(dev);
            }
        }
    }
    Err(format!(
        "{} is not an NVIDIA/CUDA device",
        node.map_or_else(String::new, |n| n.display().to_string())
    ))
}

impl State {
    /// Make our CUDA context current for the duration of `f`.
    fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R, String>) -> Result<R, String> {
        // SAFETY: we created ctx and it is still alive.
        self.cuda.check("cuCtxPushCurrent", unsafe {
            (self.cuda.ctx_push)(self.ctx)
        })?;
        let r = f(self);
        let mut prev: CuContext = null_mut();
        // SAFETY: pops exactly what we pushed.
        unsafe { (self.cuda.ctx_pop)(&raw mut prev) };
        r
    }

    fn destroy_decoder(&mut self) {
        if !self.decoder.is_null() {
            let dec = self.decoder;
            // SAFETY: we created the decoder.
            let _ = self.with_ctx(|s| {
                unsafe { (s.cuvid.destroy_decoder)(dec) };
                Ok(())
            });
            self.decoder = null_mut();
            self.fmt = None;
        }
    }

    fn destroy(&mut self) {
        self.destroy_decoder();
        // SAFETY: released in reverse creation order, each exactly once.
        unsafe {
            if !self.lock.is_null() {
                (self.cuvid.ctx_lock_destroy)(self.lock);
                self.lock = null_mut();
            }
            if !self.ctx.is_null() {
                (self.cuda.ctx_destroy)(self.ctx);
                self.ctx = null_mut();
            }
        }
    }

    fn sequence(&mut self, f: &VideoFormat) -> Result<c_int, String> {
        let surfaces = c_int::from(f.min_num_decode_surfaces.max(1)) + 2;
        if let Some(old) = &self.fmt
            && old.coded_width == f.coded_width
            && old.coded_height == f.coded_height
            && old.chroma_format == f.chroma_format
            && old.bit_depth_luma_minus8 == f.bit_depth_luma_minus8
        {
            return Ok(surfaces);
        }
        if f.chroma_format != CHROMA_420 && f.chroma_format != CHROMA_MONO {
            return Err("the plugin's NVDEC path supports 4:2:0 only".into());
        }
        self.destroy_decoder();

        let area = f.display_area;
        let w = (area.right - area.left).max(1) as u32;
        let h = (area.bottom - area.top).max(1) as u32;
        let high = f.bit_depth_luma_minus8 > 0;
        let mut info = DecodeCreateInfo {
            width: c_ulong::from(f.coded_width),
            height: c_ulong::from(f.coded_height),
            num_decode_surfaces: surfaces as c_ulong,
            codec_type: f.codec,
            chroma_format: f.chroma_format,
            creation_flags: CREATE_PREFER_CUVID,
            bit_depth_minus8: c_ulong::from(f.bit_depth_luma_minus8),
            intra_decode_only: 0,
            max_width: c_ulong::from(f.coded_width),
            max_height: c_ulong::from(f.coded_height),
            reserved1: 0,
            display_area: Rect16 {
                left: area.left as c_short,
                top: area.top as c_short,
                right: area.right as c_short,
                bottom: area.bottom as c_short,
            },
            output_format: if high { SURFACE_P016 } else { SURFACE_NV12 },
            deinterlace_mode: if f.progressive_sequence != 0 {
                DEINTERLACE_WEAVE
            } else {
                DEINTERLACE_ADAPTIVE
            },
            target_width: c_ulong::from(w),
            target_height: c_ulong::from(h),
            num_output_surfaces: 2,
            vid_lock: self.lock,
            target_rect: Rect16::default(),
            enable_histogram: 0,
            enable_decode_features: 0,
            reserved2: [0; 3],
        };
        let mut dec: CuVideoDecoder = null_mut();
        // SAFETY: the callback runs inside parse; the context is already current.
        self.cuda.check("cuvidCreateDecoder", unsafe {
            (self.cuvid.create_decoder)(&raw mut dec, &raw mut info)
        })?;
        self.decoder = dec;
        self.fmt = Some(*f);
        self.out_w = w;
        self.out_h = h;
        self.surface_h = h;
        self.high_depth = high;
        Ok(surfaces)
    }

    fn display(&mut self, d: &ParserDispInfo) -> Result<(), String> {
        let mut pp = ProcParams {
            progressive_frame: d.progressive_frame,
            second_field: 0,
            top_field_first: d.top_field_first,
            unpaired_field: c_int::from(d.repeat_first_field < 0),
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            reserved1: 0,
            output_stream: null_mut(),
            reserved: [0; 46],
            histogram_dptr: null_mut(),
            proc_ext: null_mut(),
        };
        let mut dptr: u64 = 0;
        let mut pitch: c_uint = 0;
        // SAFETY: the picture index comes from the parser; the context is current.
        self.cuda.check("cuvidMapVideoFrame", unsafe {
            (self.cuvid.map_frame)(
                self.decoder,
                d.picture_index,
                &raw mut dptr,
                &raw mut pitch,
                &raw mut pp,
            )
        })?;

        let bpp = if self.high_depth { 2 } else { 1 };
        let (w, h) = (self.out_w as usize, self.out_h as usize);
        let row = w * bpp;
        let ch = h.div_ceil(2);
        self.host.resize(row * (h + ch), 0);
        // Y, then UV: in video memory UV immediately follows surface_h luma rows
        let chroma_off = u64::from(pitch) * u64::from((self.surface_h + 1) & !1);
        let copies = [(dptr, 0usize, h), (dptr + chroma_off, row * h, ch)];
        let mut res = Ok(());
        for (src, off, rows) in copies {
            let m = Memcpy2D {
                src_x_in_bytes: 0,
                src_y: 0,
                src_memory_type: MEM_DEVICE,
                src_host: null(),
                src_device: src,
                src_array: null_mut(),
                src_pitch: pitch as usize,
                dst_x_in_bytes: 0,
                dst_y: 0,
                dst_memory_type: MEM_HOST,
                // SAFETY: host holds row*(h+ch) bytes.
                dst_host: unsafe { self.host.as_mut_ptr().add(off) }.cast(),
                dst_device: 0,
                dst_array: null_mut(),
                dst_pitch: row,
                width_in_bytes: row,
                height: rows,
            };
            // SAFETY: m is fully initialized; the source is the mapped frame.
            res = self
                .cuda
                .check("cuMemcpy2D", unsafe { (self.cuda.memcpy2d)(&raw const m) });
            if res.is_err() {
                break;
            }
        }
        // SAFETY: mapped above; unmapped right after the copy.
        unsafe { (self.cuvid.unmap_frame)(self.decoder, dptr) };
        res?;

        let fmt = self.fmt.as_ref().ok_or("frame without a format")?;
        let vsd = fmt.video_signal_description;
        let matrix = match vsd[3] {
            1 => Matrix::Bt709,
            9 | 10 => Matrix::Bt2020,
            5 | 6 => Matrix::Bt601,
            _ => Matrix::guess(self.out_h),
        };
        let (y, uv) = self.host.split_at(row * h);
        let img = YuvImage {
            width: self.out_w,
            height: self.out_h,
            // P016: 10/12-bit samples sit in the high bits of a 16-bit word
            bit_depth: if self.high_depth { 16 } else { 8 },
            chroma: if fmt.chroma_format == CHROMA_MONO {
                Chroma::Mono
            } else {
                Chroma::Sub420
            },
            matrix,
            full_range: vsd[0] & 0b1000 != 0,
            y: Plane {
                data: y,
                stride: row,
            },
            uv: ChromaPlanes::Interleaved(Plane {
                data: uv,
                stride: row,
            }),
        };
        if let Some(cpu) = yuv::to_bgra(&img) {
            self.ready.push(DecodedFrame {
                pts: Duration::from_micros(u64::try_from(d.timestamp).unwrap_or(0)),
                data: FrameData::Cpu(cpu),
            });
        }
        Ok(())
    }
}

// Parser callbacks: errors go into state.error and we return 0, so the parser
// aborts cuvidParseVideoData and feed() passes the error text up.
// A panic must not cross the C boundary, so everything runs inside catch_unwind.
fn callback<T>(
    user: *mut c_void,
    arg: *mut T,
    f: impl FnOnce(&mut State, &T) -> Result<c_int, String>,
) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: user is our Box<State>; arg is a parser struct valid for the duration of the call.
        let (Some(state), Some(arg)) = (unsafe { user.cast::<State>().as_mut() }, unsafe {
            arg.as_ref()
        }) else {
            return 0;
        };
        match f(state, arg) {
            Ok(v) => v,
            Err(e) => {
                state.error = Some(e);
                0
            }
        }
    }))
    .unwrap_or(0)
}

unsafe extern "C" fn on_sequence(user: *mut c_void, f: *mut VideoFormat) -> c_int {
    callback(user, f, |s, f| s.sequence(f))
}

unsafe extern "C" fn on_decode(user: *mut c_void, pic: *mut c_void) -> c_int {
    std::panic::catch_unwind(|| {
        // SAFETY: user is our State; pic is passed to the driver as is.
        let Some(s) = (unsafe { user.cast::<State>().as_mut() }) else {
            return 0;
        };
        if s.decoder.is_null() {
            return 0;
        }
        match s.cuda.check("cuvidDecodePicture", unsafe {
            (s.cuvid.decode_picture)(s.decoder, pic)
        }) {
            Ok(()) => 1,
            Err(e) => {
                s.error = Some(e);
                0
            }
        }
    })
    .unwrap_or(0)
}

unsafe extern "C" fn on_display(user: *mut c_void, d: *mut ParserDispInfo) -> c_int {
    if d.is_null() {
        return 1; // end of stream (CUVID_PKT_NOTIFY_EOS)
    }
    callback(user, d, |s, d| s.display(d).map(|()| 1))
}

impl Decoder for NvdecDecoder {
    fn name(&self) -> &'static str {
        "NVDEC (GPU)"
    }

    fn decode(&mut self, packet: &Packet) -> PipeResult<Vec<DecodedFrame>> {
        if packet.data.is_empty() {
            return Ok(Vec::new());
        }
        self.feed(&packet.data, PKT_TIMESTAMP, packet.pts)
    }

    fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
        let out = self.feed(&[], PKT_ENDOFSTREAM, Duration::ZERO);
        // after end of stream the parser accepts no new data, so we need a fresh one
        self.drop_parser();
        self.new_parser()?;
        out
    }

    fn reset(&mut self) {
        self.drop_parser();
        self.state.ready.clear();
        self.state.error = None;
        if let Err(e) = self.new_parser() {
            self.state.error = Some(e);
        }
    }
}

impl Drop for NvdecDecoder {
    fn drop(&mut self) {
        self.drop_parser();
        self.state.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Struct layouts checked against nv-codec-headers on Linux x86_64 (LP64).
    #[test]
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    fn layouts_match_nv_codec_headers() {
        assert_eq!(size_of::<VideoFormat>(), 64);
        assert_eq!(size_of::<SourceDataPacket>(), 32);
        assert_eq!(size_of::<ParserDispInfo>(), 24);
        assert_eq!(offset_of!(ParserParams, user_data), 40);
        assert_eq!(size_of::<ParserParams>(), 136);
        assert_eq!(offset_of!(DecodeCreateInfo, display_area), 80);
        assert_eq!(offset_of!(DecodeCreateInfo, vid_lock), 120);
        assert_eq!(size_of::<DecodeCreateInfo>(), 176);
        assert_eq!(offset_of!(ProcParams, output_stream), 56);
        assert_eq!(size_of::<ProcParams>(), 264);
        assert_eq!(size_of::<DecodeCaps>(), 88);
        assert_eq!(size_of::<Memcpy2D>(), 128);
    }

    /// Whatever doesn't depend on the size of long is checked everywhere.
    #[test]
    fn portable_layouts() {
        assert_eq!(size_of::<VideoFormat>(), 64);
        assert_eq!(size_of::<ParserDispInfo>(), 24);
        assert_eq!(size_of::<DecodeCaps>(), 88);
        assert_eq!(offset_of!(DecodeCaps, is_supported), 24);
        assert_eq!(offset_of!(DecodeCaps, max_width), 28);
        assert_eq!(offset_of!(DecodeCaps, reserved3), 52);
        if cfg!(target_pointer_width = "64") {
            assert_eq!(size_of::<Memcpy2D>(), 128);
        }
    }

    #[test]
    fn codecs_map() {
        assert_eq!(cuda_codec(&Codec::H264), Some(4));
        assert_eq!(cuda_codec(&Codec::Av1), Some(11));
        assert_eq!(cuda_codec(&Codec::Other("x".into())), None);
    }

    #[test]
    fn no_driver_is_an_error_not_a_crash() {
        // on a machine without NVIDIA (CI, Windows): a clear error, not a panic
        if open_any(&["libcuda.so.1"]).is_ok() {
            return;
        }
        let e = NvdecDecoder::new(&Codec::H264, None).err().unwrap();
        assert!(
            e.contains("NVIDIA") || e.contains("nvcuvid") || e.contains("CUDA"),
            "{e}"
        );
    }
}
