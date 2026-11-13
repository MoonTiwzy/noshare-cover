//! VA-API decoder for noshare-cover: a thin C wrapper over the cros-codecs
//! stateless decoders. The main plugin loads it via dlopen (from a memfd), so
//! the whole interface is a few `extern "C"` functions and one frame struct.
//!
//! Frames are decoded into linear NV12 dma-bufs (see Allocator). From there a frame
//! goes one of two ways:
//! - copied out of video memory and passed to the callback as NV12 planes, valid only
//!   for the duration of the callback (the plugin converts them on the CPU);
//! - with `nsc_vaapi_export`: converted to RGB by the GPU's video processor, scaled to
//!   the size it is shown at, and passed as a dma-buf the compositor imports as a
//!   texture. Nothing is copied to system memory. The picture stays untouched until
//!   the plugin gives its token back (`nsc_vaapi_release`).
//!
//! cros-codecs 0.0.6 limitation: the VA-API path is 8-bit only (NV12).

#![cfg_attr(not(target_os = "linux"), allow(unused))]

use std::ffi::{c_char, c_void};

/// ABI version between the helper and the plugin.
pub const ABI: u32 = 2;

/// A decoded frame, in one of two forms.
/// In memory (`y` is not NULL): NV12 planes, valid during the callback.
/// In video memory (`y` is NULL): one plane of RGB in a dma-buf. `fd` is valid during the
/// callback (dup it to keep it); the picture is valid until `token` is released.
#[repr(C)]
pub struct NscVaFrame {
    pub width: u32,
    pub height: u32,
    pub y: *const u8,
    pub y_stride: usize,
    pub uv: *const u8,
    pub uv_stride: usize,
    pub pts_ns: u64,
    pub fd: i32,
    /// DRM fourcc and modifier of the dma-buf
    pub drm_format: u32,
    pub modifier: u64,
    pub offset: u32,
    pub pitch: u32,
    /// The same number for the same memory: frames take turns in a few buffers, and
    /// what was imported for one of them shows every later frame in it.
    pub buffer: u64,
    pub token: u64,
}

pub type FrameCb = unsafe extern "C" fn(user: *mut c_void, frame: *const NscVaFrame);

// Codec IDs; must match decode/vaapi.rs in the main plugin.
pub const CODEC_H264: u32 = 1;
pub const CODEC_HEVC: u32 = 2;
pub const CODEC_VP8: u32 = 3;
pub const CODEC_VP9: u32 = 4;
pub const CODEC_AV1: u32 = 5;

#[unsafe(no_mangle)]
pub extern "C" fn nsc_vaapi_abi() -> u32 {
    ABI
}

fn write_err(buf: *mut c_char, len: usize, msg: &str) {
    if buf.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(len - 1);
    // SAFETY: the caller provided a buffer of len bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.cast::<u8>(), n);
        *buf.add(n) = 0;
    }
}

/// A panic must not cross the C boundary.
fn guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(fallback)
}

/// A caught panic as an error message: cros-codecs panics instead of returning some errors.
fn panic_error(p: &(dyn std::any::Any + Send)) -> String {
    let text = p
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| p.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown");
    format!("VA-API helper panicked: {text}")
}

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::{CStr, CString, c_int};
    use std::fs::File;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::rc::Rc;
    use std::sync::Arc;

    use cros_codecs::decoder::stateless::av1::Av1;
    use cros_codecs::decoder::stateless::h264::H264;
    use cros_codecs::decoder::stateless::h265::H265;
    use cros_codecs::decoder::stateless::vp8::Vp8;
    use cros_codecs::decoder::stateless::vp9::Vp9;
    use cros_codecs::decoder::stateless::{
        DecodeError, DynStatelessVideoDecoder, StatelessDecoder, StatelessVideoDecoder,
    };
    use cros_codecs::decoder::{DecodedHandle, DecoderEvent};
    use cros_codecs::libva::{self as va, Display, Surface, UsageHint};
    use cros_codecs::video_frame::frame_pool::{FramePool, PooledVideoFrame};
    use cros_codecs::video_frame::generic_dma_video_frame::GenericDmaVideoFrame;
    use cros_codecs::video_frame::{ReadMapping, VideoFrame, WriteMapping};
    use cros_codecs::{BlockingMode, Fourcc, FrameLayout, PlaneLayout, Resolution};

    type Frame = PooledVideoFrame<Nv12Frame>;

    pub struct Dec {
        // field order = drop order: the decoder releases frames and context before the pool and display
        dec: DynStatelessVideoDecoder<Frame>,
        pool: FramePool<Nv12Frame>,
        /// set while frames are handed out in video memory
        vpp: Option<Vpp>,
        /// why they are not (any more), for the plugin's trace
        pub note: String,
        display: Rc<Display>,
        // the last frame's planes, copied out of its buffer
        y: Vec<u8>,
        uv: Vec<u8>,
        // H.264 and HEVC output frames in another order than they come in
        reorders: bool,
        pub err: String,
    }

    mod gbm {
        use std::ffi::c_int;

        #[repr(C)]
        pub struct Device {
            _p: [u8; 0],
        }
        #[repr(C)]
        pub struct Bo {
            _p: [u8; 0],
        }
        pub const FORMAT_R8: u32 = u32::from_le_bytes(*b"R8  ");
        pub const BO_USE_LINEAR: u32 = 1 << 4;

        #[link(name = "gbm")]
        unsafe extern "C" {
            pub fn gbm_create_device(fd: c_int) -> *mut Device;
            pub fn gbm_device_destroy(dev: *mut Device);
            pub fn gbm_bo_create(
                dev: *mut Device,
                width: u32,
                height: u32,
                format: u32,
                flags: u32,
            ) -> *mut Bo;
            pub fn gbm_bo_get_fd(bo: *mut Bo) -> c_int;
            pub fn gbm_bo_get_stride(bo: *mut Bo) -> u32;
            pub fn gbm_bo_destroy(bo: *mut Bo);
        }
    }

    /// Decode targets: linear NV12 in one dma-buf. cros-codecs' own GBM frames ask for NV12
    /// with a minigbm-only usage flag, and Mesa's GBM can't allocate NV12 at all on some
    /// drivers (radeonsi), so no decoder could get a frame there. A linear R8 buffer 1.5
    /// times the height holds the same picture and every driver allocates it.
    struct Allocator {
        dev: *mut gbm::Device,
        _node: File,
    }

    // The GBM device is only used to allocate, from the decoder's thread.
    unsafe impl Send for Allocator {}
    unsafe impl Sync for Allocator {}

    impl Allocator {
        fn open(node: &str) -> Result<Self, String> {
            let file = File::options()
                .read(true)
                .write(true)
                .open(node)
                .map_err(|e| format!("GBM on {node}: {e}"))?;
            // SAFETY: a valid fd, kept open as long as the device.
            let dev = unsafe { gbm::gbm_create_device(file.as_raw_fd()) };
            if dev.is_null() {
                return Err(format!("GBM on {node}: no device"));
            }
            Ok(Self { dev, _node: file })
        }

        fn frame(&self, coded: Resolution) -> Result<Nv12Frame, String> {
            let (width, rows) = (
                coded.width.next_multiple_of(2),
                coded.height.next_multiple_of(2),
            );
            // SAFETY: a live device; the buffer is released right after its dma-buf is taken
            // (the dma-buf keeps the memory).
            let (fd, stride) = unsafe {
                let bo = gbm::gbm_bo_create(
                    self.dev,
                    width,
                    rows / 2 * 3,
                    gbm::FORMAT_R8,
                    gbm::BO_USE_LINEAR,
                );
                if bo.is_null() {
                    return Err(format!("GBM: can't allocate a {width}x{rows} frame"));
                }
                let fd: c_int = gbm::gbm_bo_get_fd(bo);
                let stride = gbm::gbm_bo_get_stride(bo) as usize;
                gbm::gbm_bo_destroy(bo);
                (fd, stride)
            };
            if fd < 0 {
                return Err("GBM: no dma-buf for a frame".into());
            }
            // SAFETY: a new fd we own.
            let file = unsafe { File::from_raw_fd(fd) };
            let sync = file.try_clone().map_err(|e| format!("dma-buf: {e}"))?;
            let offsets = [0, stride * rows as usize];
            let dma = GenericDmaVideoFrame::new(
                vec![file],
                FrameLayout {
                    format: (Fourcc::from(b"NV12"), 0), // DRM_FORMAT_MOD_LINEAR
                    size: coded,
                    planes: offsets
                        .iter()
                        .map(|&offset| PlaneLayout {
                            buffer_index: 0,
                            offset,
                            stride,
                        })
                        .collect(),
                },
            )?;
            Nv12Frame::new(dma, sync, offsets)
        }
    }

    impl Drop for Allocator {
        fn drop(&mut self) {
            // SAFETY: created in open, destroyed once.
            unsafe { gbm::gbm_device_destroy(self.dev) };
        }
    }

    /// A decode target, mapped for reading once. Its memory is uncached for the CPU: it is
    /// read only with one sequential copy per plane (see Dec::drain). cros-codecs' own
    /// mapping maps it for every frame and flushes every byte from the cache on unmap (an
    /// Intel workaround), 30 ms per 640x360 frame.
    #[derive(Debug)]
    pub struct Nv12Frame {
        dma: GenericDmaVideoFrame,
        sync: File, // the same dma-buf
        mem: *const u8,
        len: usize,
        offsets: [usize; 2],
    }

    // The mapping is read only and lives as long as the frame.
    unsafe impl Send for Nv12Frame {}
    unsafe impl Sync for Nv12Frame {}

    impl Nv12Frame {
        fn new(dma: GenericDmaVideoFrame, sync: File, offsets: [usize; 2]) -> Result<Self, String> {
            let len = usize::try_from(sync.metadata().map_err(|e| format!("dma-buf: {e}"))?.len())
                .map_err(|e| e.to_string())?;
            // SAFETY: a read-only shared mapping of the whole dma-buf, unmapped on drop.
            let mem = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    sync.as_raw_fd(),
                    0,
                )
            };
            if mem == libc::MAP_FAILED {
                return Err(format!("dma-buf mmap: {}", std::io::Error::last_os_error()));
            }
            Ok(Self {
                dma,
                sync,
                mem: mem.cast(),
                len,
                offsets,
            })
        }

        fn sync(&self, flags: u64) {
            #[repr(C)]
            struct DmaBufSync {
                flags: u64,
            }
            const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x4008_6200; // _IOW('b', 0, struct dma_buf_sync)
            // SAFETY: a dma-buf fd and the struct the ioctl takes; on failure the read still works,
            // only without the driver's cache handling.
            unsafe {
                libc::ioctl(
                    self.sync.as_raw_fd(),
                    DMA_BUF_IOCTL_SYNC,
                    &DmaBufSync { flags },
                )
            };
        }
    }

    impl Drop for Nv12Frame {
        fn drop(&mut self) {
            // SAFETY: mapped in new, unmapped once.
            unsafe { libc::munmap(self.mem.cast_mut().cast(), self.len) };
        }
    }

    const DMA_BUF_SYNC_READ: u64 = 1;
    const DMA_BUF_SYNC_END: u64 = 4;

    /// The planes of an Nv12Frame, between the dma-buf's begin and end of a CPU read.
    struct Nv12Read<'a>(&'a Nv12Frame);

    impl<'a> ReadMapping<'a> for Nv12Read<'a> {
        fn get(&self) -> Vec<&[u8]> {
            let f = self.0;
            // SAFETY: the mapping is len bytes and lives as long as the frame.
            let mem = unsafe { std::slice::from_raw_parts(f.mem, f.len) };
            f.offsets
                .iter()
                .zip(f.get_plane_size())
                .map(|(&start, size)| &mem[start.min(f.len)..(start + size).min(f.len)])
                .collect()
        }
    }

    impl Drop for Nv12Read<'_> {
        fn drop(&mut self) {
            self.0.sync(DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
        }
    }

    impl VideoFrame for Nv12Frame {
        type MemDescriptor = <GenericDmaVideoFrame as VideoFrame>::MemDescriptor;
        type NativeHandle = <GenericDmaVideoFrame as VideoFrame>::NativeHandle;

        fn fourcc(&self) -> Fourcc {
            self.dma.fourcc()
        }

        fn resolution(&self) -> Resolution {
            self.dma.resolution()
        }

        fn get_plane_size(&self) -> Vec<usize> {
            self.dma.get_plane_size()
        }

        fn get_plane_pitch(&self) -> Vec<usize> {
            self.dma.get_plane_pitch()
        }

        fn map<'a>(&'a self) -> Result<Box<dyn ReadMapping<'a> + 'a>, String> {
            self.sync(DMA_BUF_SYNC_READ); // START is 0
            Ok(Box::new(Nv12Read(self)))
        }

        fn map_mut<'a>(&'a mut self) -> Result<Box<dyn WriteMapping<'a> + 'a>, String> {
            Err("decode frames are read only".into())
        }

        fn to_native_handle(&self, display: &Rc<Display>) -> Result<Self::NativeHandle, String> {
            self.dma.to_native_handle(display)
        }
    }

    /// Extra frames to keep in the pool above the minimum, so the decoder doesn't
    /// hit NotEnoughOutputBuffers while we copy a frame out.
    const POOL_HEADROOM: usize = 2;

    /// The VA drivers of a render node's GPU, from its kernel driver: libva's own table
    /// (va_drm_utils.c), which it uses only when LIBVA_DRIVER_NAME isn't set.
    fn node_drivers(node: &str) -> &'static [&'static str] {
        let driver = std::fs::canonicalize(node).ok().and_then(|p| {
            let name = p.file_name()?.to_str()?.to_owned();
            let link = std::fs::read_link(format!("/sys/class/drm/{name}/device/driver")).ok()?;
            Some(link.file_name()?.to_str()?.to_owned())
        });
        match driver.as_deref() {
            Some("amdgpu") => &["radeonsi"],
            Some("radeon") => &["r600", "radeonsi"],
            Some("i915") => &["iHD", "i965"],
            Some("xe") => &["iHD"],
            Some("nouveau") => &["nouveau"],
            Some("nvidia") => &["nvidia"],
            _ => &[],
        }
    }

    /// LIBVA_DRIVER_NAME applies to every device: with an NVIDIA card next to an AMD or Intel
    /// one and LIBVA_DRIVER_NAME=nvidia (usual on such setups), libva loads the NVIDIA driver
    /// for the other GPU's render node too, and it fails there. When it names another GPU's
    /// driver, the node's own is loaded instead.
    fn driver_for(node: &str) -> Option<CString> {
        let wanted = std::env::var("LIBVA_DRIVER_NAME").ok()?;
        let own = node_drivers(node);
        if own.is_empty() || own.contains(&wanted.as_str()) {
            return None;
        }
        CString::new(own[0]).ok()
    }

    /// `VAProcColorProperties` (va_vpp.h).
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct ColorProperties {
        chroma_sample_location: u8,
        color_range: u8,
        colour_primaries: u8,
        transfer_characteristics: u8,
        matrix_coefficients: u8,
        reserved: [u8; 3],
    }

    /// `VAProcPipelineParameterBuffer` (va_vpp.h, LP64): what the video processor is asked
    /// to do with one picture. The vendored bindings leave it out.
    #[repr(C)]
    struct ProcPipeline {
        surface: va::VASurfaceID,
        surface_region: *const va::VARectangle,
        surface_color_standard: u32,
        output_region: *const va::VARectangle,
        output_background_color: u32,
        output_color_standard: u32,
        pipeline_flags: u32,
        filter_flags: u32,
        filters: *mut va::VABufferID,
        num_filters: u32,
        forward_references: *mut va::VASurfaceID,
        num_forward_references: u32,
        backward_references: *mut va::VASurfaceID,
        num_backward_references: u32,
        rotation_state: u32,
        blend_state: *const c_void,
        mirror_state: u32,
        additional_outputs: *mut va::VASurfaceID,
        num_additional_outputs: u32,
        input_surface_flag: u32,
        output_surface_flag: u32,
        input_color_properties: ColorProperties,
        output_color_properties: ColorProperties,
        processing_mode: u32,
        output_hdr_metadata: *mut c_void,
        va_reserved: [u32; 16],
    }

    // VAProcColorStandardType: the colours are spelled out in ColorProperties
    const COLOR_EXPLICIT: u32 = 13;
    // ISO/IEC 23001-8 (H.273) code points
    const PRIMARIES_BT709: u8 = 1;
    const TRANSFER_BT709: u8 = 1;
    const MATRIX_RGB: u8 = 0;
    const MATRIX_BT709: u8 = 1;
    const MATRIX_BT601: u8 = 6;

    fn check(what: &str, status: va::VAStatus) -> Result<(), String> {
        if status as u32 == va::VA_STATUS_SUCCESS {
            return Ok(());
        }
        // SAFETY: vaErrorStr returns a static string for any status.
        let text = unsafe { CStr::from_ptr(va::vaErrorStr(status)) }.to_string_lossy();
        Err(format!("{what}: {text}"))
    }

    /// How a picture in video memory is reached from outside: its dma-buf and layout.
    struct Exported {
        fd: OwnedFd,
        drm_format: u32,
        modifier: u64,
        offset: u32,
        pitch: u32,
    }

    /// An RGB picture in video memory: the video processor draws a frame into it, the
    /// compositor shows it.
    struct OutBuf {
        id: u64,
        size: (u32, u32),
        surface: Surface<()>,
        /// Taken after the first frame was drawn: a driver may still replace the memory
        /// of a fresh surface when it is first used.
        exported: Option<Exported>,
    }

    /// At most this many pictures at a time: the one on screen, the one waiting to be
    /// picked up, and what a decoder returns in one go at the end of a stream.
    const MAX_OUT: usize = 12;
    /// The video processor isn't asked for anything smaller.
    const MIN_OUT: u32 = 16;

    /// NV12 frames to RGB pictures the compositor can import, on the GPU.
    struct Vpp {
        display: Rc<Display>,
        config: va::VAConfigID,
        /// for pictures of `size`
        context: Option<va::VAContextID>,
        size: (u32, u32),
        /// the size the pictures are shown at (0 = unknown)
        target: (u32, u32),
        /// pictures of `size` nobody holds
        free: Vec<OutBuf>,
        /// by token: with the plugin until it gives the token back
        held: HashMap<u64, OutBuf>,
        next_id: u64,
        next_token: u64,
    }

    impl Vpp {
        fn new(display: &Rc<Display>) -> Result<Self, String> {
            let mut config: va::VAConfigID = 0;
            // SAFETY: a live display; no attributes; config is written on success.
            check("video processor", unsafe {
                va::vaCreateConfig(
                    display.handle(),
                    va::VAProfile::VAProfileNone,
                    va::VAEntrypoint::VAEntrypointVideoProc,
                    std::ptr::null_mut(),
                    0,
                    &mut config,
                )
            })?;
            Ok(Self {
                display: Rc::clone(display),
                config,
                context: None,
                size: (0, 0),
                target: (0, 0),
                free: Vec::new(),
                held: HashMap::new(),
                next_id: 1,
                next_token: 1,
            })
        }

        /// The size a `res` frame comes out at: halved as long as it still covers the
        /// size it is shown at, like the plugin's CPU path does.
        fn out_size(&self, res: (u32, u32)) -> (u32, u32) {
            let (mut w, mut h) = res;
            let (tw, th) = self.target;
            if tw > 0 && th > 0 {
                while w / 2 >= tw.max(MIN_OUT) && h / 2 >= th.max(MIN_OUT) {
                    (w, h) = (w / 2, h / 2);
                }
            }
            (w, h)
        }

        fn drop_context(&mut self) {
            if let Some(ctx) = self.context.take() {
                // SAFETY: created by us, destroyed once.
                unsafe { va::vaDestroyContext(self.display.handle(), ctx) };
            }
        }

        /// Pictures are `size` from now on. The ones still out stay as they are and are
        /// dropped when they come back.
        fn resize(&mut self, size: (u32, u32)) -> Result<(), String> {
            self.free.clear();
            self.drop_context();
            self.size = size;
            let mut ctx: va::VAContextID = 0;
            // SAFETY: a live display and config; no render targets are named (as ffmpeg does).
            check("video processor context", unsafe {
                va::vaCreateContext(
                    self.display.handle(),
                    self.config,
                    size.0 as c_int,
                    size.1 as c_int,
                    va::VA_PROGRESSIVE as c_int,
                    std::ptr::null_mut(),
                    0,
                    &mut ctx,
                )
            })?;
            self.context = Some(ctx);
            Ok(())
        }

        fn new_out(&mut self) -> Result<OutBuf, String> {
            let hint = UsageHint::USAGE_HINT_VPP_WRITE | UsageHint::USAGE_HINT_EXPORT;
            // X, not A: the compositor must not blend by whatever the driver leaves in alpha
            let mut surfaces = [va::VA_FOURCC_BGRX, va::VA_FOURCC_BGRA]
                .into_iter()
                .find_map(|fourcc| {
                    self.display
                        .create_surfaces(
                            va::VA_RT_FORMAT_RGB32,
                            Some(fourcc),
                            self.size.0,
                            self.size.1,
                            Some(hint),
                            vec![()],
                        )
                        .ok()
                })
                .ok_or("the GPU has no RGB surface for the video processor")?;
            let surface = surfaces.pop().ok_or("no RGB surface")?;
            let id = self.next_id;
            self.next_id += 1;
            Ok(OutBuf {
                id,
                size: self.size,
                surface,
                exported: None,
            })
        }

        /// `input` (NV12, `res` of it is the picture) drawn into `out`, scaled to its size.
        fn draw(
            &self,
            input: va::VASurfaceID,
            res: (u32, u32),
            out: &OutBuf,
        ) -> Result<(), String> {
            let ctx = self.context.ok_or("no video processor context")?;
            let dpy = self.display.handle();
            let region = va::VARectangle {
                x: 0,
                y: 0,
                width: u16::try_from(res.0).map_err(|_| "frame too wide")?,
                height: u16::try_from(res.1).map_err(|_| "frame too tall")?,
            };
            // SAFETY: all-zero is valid for this C struct (no filters, no references).
            let mut p: ProcPipeline = unsafe { std::mem::zeroed() };
            p.surface = input;
            p.surface_region = &region;
            // Only the YCbCr matrix and the range are undone, exactly what the plugin's CPU
            // path does. With a named standard a driver may go further (Mesa also maps the
            // primaries and the transfer curve of BT.601 to those of the output), and the
            // same video would look different depending on where it was decoded. So both
            // sides are spelled out with the same primaries and transfer.
            // The stream's own matrix isn't known here: HD and up is 709, like the CPU path guesses.
            p.surface_color_standard = COLOR_EXPLICIT;
            p.input_color_properties = ColorProperties {
                color_range: va::VA_SOURCE_RANGE_REDUCED as u8,
                colour_primaries: PRIMARIES_BT709,
                transfer_characteristics: TRANSFER_BT709,
                matrix_coefficients: if res.1 >= 720 {
                    MATRIX_BT709
                } else {
                    MATRIX_BT601
                },
                ..Default::default()
            };
            p.output_color_standard = COLOR_EXPLICIT;
            p.output_color_properties = ColorProperties {
                color_range: va::VA_SOURCE_RANGE_FULL as u8,
                colour_primaries: PRIMARIES_BT709,
                transfer_characteristics: TRANSFER_BT709,
                matrix_coefficients: MATRIX_RGB,
                ..Default::default()
            };
            let mut buf: va::VABufferID = 0;
            // SAFETY: the driver copies `p` (and what it points to is alive) within the calls;
            // the buffer is destroyed whether rendering worked or not.
            unsafe {
                check(
                    "video processor buffer",
                    va::vaCreateBuffer(
                        dpy,
                        ctx,
                        va::VABufferType::VAProcPipelineParameterBufferType,
                        size_of::<ProcPipeline>() as u32,
                        1,
                        (&raw mut p).cast(),
                        &mut buf,
                    ),
                )?;
                let drawn = check(
                    "vaBeginPicture",
                    va::vaBeginPicture(dpy, ctx, out.surface.id()),
                )
                .and_then(|()| {
                    check(
                        "vaRenderPicture",
                        va::vaRenderPicture(dpy, ctx, &mut buf, 1),
                    )
                })
                .and_then(|()| check("vaEndPicture", va::vaEndPicture(dpy, ctx)));
                va::vaDestroyBuffer(dpy, buf);
                drawn?;
                check("vaSyncSurface", va::vaSyncSurface(dpy, out.surface.id()))
            }
        }

        fn export(surface: &Surface<()>) -> Result<Exported, String> {
            let mut desc = surface
                .export_prime()
                .map_err(|e| format!("exporting the picture: {e}"))?;
            let layer = desc.layers.first().ok_or("exported picture has no layer")?;
            if layer.num_planes != 1 || desc.layers.len() != 1 {
                return Err("exported picture isn't a single RGB plane".into());
            }
            // the alpha byte carries nothing: tell the compositor so
            const ARGB: u32 = u32::from_le_bytes(*b"AR24");
            const ABGR: u32 = u32::from_le_bytes(*b"AB24");
            let drm_format = match layer.drm_format {
                ARGB => u32::from_le_bytes(*b"XR24"),
                ABGR => u32::from_le_bytes(*b"XB24"),
                other => other,
            };
            let (offset, pitch) = (layer.offset[0], layer.pitch[0]);
            let index = usize::from(layer.object_index[0]);
            if index >= desc.objects.len() {
                return Err("exported picture names no dma-buf".into());
            }
            let object = desc.objects.swap_remove(index);
            Ok(Exported {
                fd: object.fd,
                drm_format,
                modifier: object.drm_format_modifier,
                offset,
                pitch,
            })
        }

        /// One decoded frame as an RGB picture. `Ok(None)`: every picture is out right
        /// now, this frame has to go the other way. `Err`: the GPU can't do it at all.
        fn frame(
            &mut self,
            vf: &Frame,
            res: Resolution,
            pts_ns: u64,
        ) -> Result<Option<NscVaFrame>, String> {
            let res = (res.width, res.height);
            if res.0 < MIN_OUT || res.1 < MIN_OUT {
                return Err("the video is too small for the video processor".into());
            }
            let size = self.out_size(res);
            if size != self.size || self.context.is_none() {
                self.resize(size)?;
            }
            let mut out = match self.free.pop() {
                Some(b) => b,
                None if self.held.len() < MAX_OUT => self.new_out()?,
                None => return Ok(None),
            };
            // the decoded frame as a surface of its own, only for this one drawing
            let input = vf.to_native_handle(&self.display)?;
            self.draw(input.id(), res, &out)?;
            if out.exported.is_none() {
                out.exported = Some(Self::export(&out.surface)?);
            }
            let e = out.exported.as_ref().ok_or("picture not exported")?;
            let token = self.next_token;
            self.next_token += 1;
            let frame = NscVaFrame {
                width: size.0,
                height: size.1,
                y: std::ptr::null(),
                y_stride: 0,
                uv: std::ptr::null(),
                uv_stride: 0,
                pts_ns,
                fd: e.fd.as_raw_fd(),
                drm_format: e.drm_format,
                modifier: e.modifier,
                offset: e.offset,
                pitch: e.pitch,
                buffer: out.id,
                token,
            };
            self.held.insert(token, out);
            Ok(Some(frame))
        }

        /// The pixels of a picture that is out, for tests: rows of 4 bytes per pixel, as its
        /// DRM format has them.
        fn read(&self, buffer: u64, out: &mut [u8]) -> Result<(), String> {
            let buf = self
                .held
                .values()
                .find(|b| b.id == buffer)
                .ok_or("no such picture")?;
            let (w, h) = (buf.size.0 as usize, buf.size.1 as usize);
            if out.len() < w * h * 4 {
                return Err("buffer too small for the picture".into());
            }
            let image = va::Image::derive_from(&buf.surface, buf.size)
                .map_err(|e| format!("reading the picture: {e}"))?;
            let (offset, pitch) = (
                image.image().offsets[0] as usize,
                image.image().pitches[0] as usize,
            );
            let data: &[u8] = image.as_ref();
            for (y, row) in out[..w * h * 4].chunks_exact_mut(w * 4).enumerate() {
                let from = offset + y * pitch;
                row.copy_from_slice(data.get(from..from + w * 4).ok_or("picture is cut short")?);
            }
            Ok(())
        }

        /// The plugin is done with the picture behind `token`: it can take the next frame.
        fn release(&mut self, token: u64) {
            if let Some(out) = self.held.remove(&token)
                && out.size == self.size
            {
                self.free.push(out);
            }
        }
    }

    impl Drop for Vpp {
        fn drop(&mut self) {
            self.drop_context();
            // SAFETY: created in new, destroyed once; the pictures go with the fields.
            unsafe { va::vaDestroyConfig(self.display.handle(), self.config) };
        }
    }

    pub fn open(node: &str, codec: u32) -> Result<Dec, String> {
        let driver = driver_for(node);
        let display =
            Display::open_drm_display_with_driver(node, driver.as_deref()).map_err(|e| {
                match &driver {
                    Some(d) => format!("VA-API on {node} ({}): {e}", d.to_string_lossy()),
                    None => format!("VA-API on {node}: {e}"),
                }
            })?;
        let alloc = Arc::new(Allocator::open(node)?);
        let pool =
            FramePool::new(move |si| alloc.frame(si.coded_resolution).expect("decode frame"));
        let bm = BlockingMode::Blocking;
        let d = Rc::clone(&display);
        let dec: DynStatelessVideoDecoder<Frame> = match codec {
            CODEC_H264 => {
                StatelessDecoder::<H264, _>::new_vaapi(d, bm).map(|x| x.into_trait_object())
            }
            CODEC_HEVC => {
                StatelessDecoder::<H265, _>::new_vaapi(d, bm).map(|x| x.into_trait_object())
            }
            CODEC_VP8 => {
                StatelessDecoder::<Vp8, _>::new_vaapi(d, bm).map(|x| x.into_trait_object())
            }
            CODEC_VP9 => {
                StatelessDecoder::<Vp9, _>::new_vaapi(d, bm).map(|x| x.into_trait_object())
            }
            CODEC_AV1 => {
                StatelessDecoder::<Av1, _>::new_vaapi(d, bm).map(|x| x.into_trait_object())
            }
            other => return Err(format!("unknown codec {other}")),
        }
        .map_err(|e| format!("VA-API can't decode this codec on {node}: {e}"))?;
        Ok(Dec {
            dec,
            pool,
            vpp: None,
            note: String::new(),
            display,
            y: Vec::new(),
            uv: Vec::new(),
            reorders: matches!(codec, CODEC_H264 | CODEC_HEVC),
            err: String::new(),
        })
    }

    impl Dec {
        /// Drain decoder events. Returns `true` if anything made progress. packet_pts: the
        /// packet being decoded (none when flushing).
        fn drain(
            &mut self,
            cb: FrameCb,
            user: *mut c_void,
            packet_pts: Option<u64>,
        ) -> Result<bool, String> {
            let mut progress = false;
            while let Some(ev) = self.dec.next_event() {
                progress = true;
                match ev {
                    DecoderEvent::FormatChanged => {
                        let mut si = self
                            .dec
                            .stream_info()
                            .ok_or("format change without stream_info")?
                            .clone();
                        si.min_num_frames += POOL_HEADROOM;
                        self.pool.resize(&si);
                    }
                    DecoderEvent::FrameReady(h) => {
                        h.sync().map_err(|e| format!("frame sync: {e}"))?;
                        // AV1 and VP9 show an earlier hidden frame again later (show_existing_frame),
                        // and cros-codecs hands out its handle with the time of the packet it was
                        // decoded in. They don't reorder: a frame is shown at the packet showing it.
                        let pts_ns = match packet_pts {
                            Some(pts) if !self.reorders => pts,
                            _ => h.timestamp(),
                        };
                        let vf = h.video_frame();
                        // The frame is the coded size, and what is shown must lie inside it:
                        // an AV1 header can name a render size larger than the frame.
                        let (shown, coded) = (h.display_resolution(), vf.resolution());
                        let mut res = shown;
                        res.width = shown.width.min(coded.width);
                        res.height = shown.height.min(coded.height);
                        // in video memory if asked for and possible; otherwise, and from
                        // the first failure on, through system memory as below
                        let exported = match self.vpp.as_mut().map(|v| v.frame(&vf, res, pts_ns)) {
                            Some(Ok(f)) => f,
                            Some(Err(e)) => {
                                self.note = e;
                                self.vpp = None;
                                None
                            }
                            None => None,
                        };
                        if let Some(f) = exported {
                            // SAFETY: the plugin dups the fd within the call.
                            unsafe { cb(user, &f) };
                            continue;
                        }
                        let pitch = vf.get_plane_pitch();
                        {
                            let map = vf.map().map_err(|e| format!("frame mapping: {e}"))?;
                            let planes = map.get();
                            if planes.len() < 2 || pitch.len() < 2 {
                                return Err("frame lacks the two NV12 planes".into());
                            }
                            // one sequential read each: the frame's memory is uncached
                            self.y.clear();
                            self.y.extend_from_slice(planes[0]);
                            self.uv.clear();
                            self.uv.extend_from_slice(planes[1]);
                        }
                        // the plugin reads width x height out of these two planes: never
                        // more than was copied into them
                        let fit_w = pitch[0].min(pitch[1]);
                        let fit_h = (self.y.len() / pitch[0].max(1))
                            .min(self.uv.len() / pitch[1].max(1) * 2);
                        let f = NscVaFrame {
                            width: res.width.min(u32::try_from(fit_w).unwrap_or(u32::MAX)),
                            height: res.height.min(u32::try_from(fit_h).unwrap_or(u32::MAX)),
                            y: self.y.as_ptr(),
                            y_stride: pitch[0],
                            uv: self.uv.as_ptr(),
                            uv_stride: pitch[1],
                            pts_ns,
                            fd: -1,
                            drm_format: 0,
                            modifier: 0,
                            offset: 0,
                            pitch: 0,
                            buffer: 0,
                            token: 0,
                        };
                        // SAFETY: the plugin callback copies the data within the call.
                        unsafe { cb(user, &f) };
                    }
                }
            }
            Ok(progress)
        }

        pub fn decode(
            &mut self,
            data: &[u8],
            pts_ns: u64,
            cb: FrameCb,
            user: *mut c_void,
        ) -> Result<(), String> {
            let mut off = 0;
            let mut stalls = 0;
            while off < data.len() {
                let pool = &mut self.pool;
                match self.dec.decode(pts_ns, &data[off..], &mut || pool.alloc()) {
                    Ok(0) => break, // decoder consumed nothing; don't spin forever
                    Ok(n) => {
                        off += n;
                        stalls = 0;
                    }
                    Err(DecodeError::CheckEvents | DecodeError::NotEnoughOutputBuffers(_)) => {
                        if !self.drain(cb, user, Some(pts_ns))? {
                            stalls += 1;
                            if stalls > 3 {
                                return Err("VA-API: decoder is out of output frames".into());
                            }
                        }
                    }
                    Err(e) => return Err(format!("VA-API: {e}")),
                }
            }
            self.drain(cb, user, Some(pts_ns)).map(|_| ())
        }

        pub fn flush(&mut self, cb: FrameCb, user: *mut c_void) -> Result<(), String> {
            self.dec.flush().map_err(|e| format!("VA-API flush: {e}"))?;
            self.drain(cb, user, None).map(|_| ())
        }

        pub fn reset(&mut self) {
            // after flush the decoder waits for a keyframe, which is exactly what the next loop starts with
            let _ = self.dec.flush();
            while self.dec.next_event().is_some() {}
        }

        /// Hand frames out in video memory from now on (or stop doing so).
        pub fn export(&mut self, on: bool) -> Result<(), String> {
            self.vpp = None;
            if on {
                self.vpp = Some(Vpp::new(&self.display)?);
            }
            Ok(())
        }

        pub fn set_target(&mut self, width: u32, height: u32) {
            if let Some(v) = &mut self.vpp {
                v.target = (width, height);
            }
        }

        pub fn release(&mut self, token: u64) {
            if let Some(v) = &mut self.vpp {
                v.release(token);
            }
        }

        pub fn exporting(&self) -> bool {
            self.vpp.is_some()
        }

        pub fn read(&self, buffer: u64, out: &mut [u8]) -> Result<(), String> {
            self.vpp
                .as_ref()
                .ok_or("frames are not in video memory")?
                .read(buffer, out)
        }
    }

    pub fn cstr(p: *const c_char) -> Option<String> {
        if p.is_null() {
            return None;
        }
        // SAFETY: the caller passes a C string.
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    pub fn with_dec(h: *mut c_void, f: impl FnOnce(&mut Dec) -> Result<(), String>) -> i32 {
        // SAFETY: h comes from nsc_vaapi_open and hasn't been closed yet.
        let Some(d) = (unsafe { h.cast::<Dec>().as_mut() }) else {
            return -1;
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(d))) {
            Ok(Ok(())) => 0,
            Ok(Err(e)) => {
                d.err = e;
                -1
            }
            Err(p) => {
                d.err = panic_error(&*p);
                -1
            }
        }
    }
}

/// Open a decoder. Returns NULL on error, with the message in `err`.
///
/// # Safety
/// `node` is a C string; `err` is a buffer of `err_len` bytes or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_open(
    node: *const c_char,
    codec: u32,
    err: *mut c_char,
    err_len: usize,
) -> *mut c_void {
    #[cfg(target_os = "linux")]
    {
        let Some(node) = imp::cstr(node) else {
            write_err(err, err_len, "no render node");
            return std::ptr::null_mut();
        };
        match std::panic::catch_unwind(|| imp::open(&node, codec)) {
            Ok(Ok(d)) => Box::into_raw(Box::new(d)).cast(),
            Ok(Err(e)) => {
                write_err(err, err_len, &e);
                std::ptr::null_mut()
            }
            Err(p) => {
                write_err(err, err_len, &panic_error(&*p));
                std::ptr::null_mut()
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (node, codec);
        write_err(err, err_len, "VA-API is Linux-only");
        std::ptr::null_mut()
    }
}

/// # Safety
/// `h` comes from `nsc_vaapi_open`; `data` is `len` bytes; `cb` is called synchronously.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_decode(
    h: *mut c_void,
    data: *const u8,
    len: usize,
    pts_ns: u64,
    cb: FrameCb,
    user: *mut c_void,
) -> i32 {
    #[cfg(target_os = "linux")]
    {
        if data.is_null() {
            return 0;
        }
        // SAFETY: see the function contract.
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        imp::with_dec(h, |d| d.decode(bytes, pts_ns, cb, user))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (h, data, len, pts_ns, cb, user);
        -1
    }
}

/// # Safety
/// Same as `nsc_vaapi_decode`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_flush(h: *mut c_void, cb: FrameCb, user: *mut c_void) -> i32 {
    #[cfg(target_os = "linux")]
    {
        imp::with_dec(h, |d| d.flush(cb, user))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (h, cb, user);
        -1
    }
}

/// # Safety
/// `h` comes from `nsc_vaapi_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_reset(h: *mut c_void) {
    #[cfg(target_os = "linux")]
    imp::with_dec(h, |d| {
        d.reset();
        Ok(())
    });
    #[cfg(not(target_os = "linux"))]
    let _ = h;
}

/// Hand frames out in video memory (as RGB dma-bufs) instead of NV12 planes in system
/// memory. Returns 0 if the GPU's video processor is there; a frame it can't do later
/// still arrives the other way (see `nsc_vaapi_note`).
///
/// # Safety
/// `h` comes from `nsc_vaapi_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_export(h: *mut c_void, on: bool) -> i32 {
    #[cfg(target_os = "linux")]
    {
        imp::with_dec(h, |d| d.export(on))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (h, on);
        -1
    }
}

/// The size frames are shown at, in pixels (0 = unknown): frames in video memory are
/// scaled down towards it.
///
/// # Safety
/// `h` comes from `nsc_vaapi_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_set_target(h: *mut c_void, width: u32, height: u32) {
    #[cfg(target_os = "linux")]
    imp::with_dec(h, |d| {
        d.set_target(width, height);
        Ok(())
    });
    #[cfg(not(target_os = "linux"))]
    let _ = (h, width, height);
}

/// The frame with this token isn't shown any more: its memory may take another frame.
///
/// # Safety
/// `h` comes from `nsc_vaapi_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_release(h: *mut c_void, token: u64) {
    #[cfg(target_os = "linux")]
    imp::with_dec(h, |d| {
        d.release(token);
        Ok(())
    });
    #[cfg(not(target_os = "linux"))]
    let _ = (h, token);
}

/// Copies the pixels of a picture that is still out (by its `buffer` number) into `out`:
/// width * height * 4 bytes, in the byte order of its DRM format. For tests.
///
/// # Safety
/// `h` comes from `nsc_vaapi_open`; `out` is `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_read(
    h: *mut c_void,
    buffer: u64,
    out: *mut u8,
    cap: usize,
) -> i32 {
    #[cfg(target_os = "linux")]
    {
        if out.is_null() {
            return -1;
        }
        // SAFETY: see the function contract.
        let out = unsafe { std::slice::from_raw_parts_mut(out, cap) };
        imp::with_dec(h, |d| d.read(buffer, out))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (h, buffer, out, cap);
        -1
    }
}

/// Whether frames come in video memory right now, and if not, why (`buf`, as a C string).
///
/// # Safety
/// `h` comes from `nsc_vaapi_open`; `buf` is `len` bytes or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_note(h: *mut c_void, buf: *mut c_char, len: usize) -> bool {
    guard(false, || {
        // SAFETY: see the contract.
        #[cfg(target_os = "linux")]
        if let Some(d) = unsafe { h.cast::<imp::Dec>().as_ref() } {
            write_err(buf, len, &d.note);
            return d.exporting();
        }
        let _ = (h, buf, len);
        false
    })
}

/// Text of the last error. Returns its length without the NUL terminator.
///
/// # Safety
/// `h` comes from `nsc_vaapi_open`; `buf` is `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_error(h: *mut c_void, buf: *mut c_char, len: usize) -> usize {
    guard(0, || {
        // SAFETY: see the contract.
        #[cfg(target_os = "linux")]
        if let Some(d) = unsafe { h.cast::<imp::Dec>().as_ref() } {
            write_err(buf, len, &d.err);
            return d.err.len();
        }
        let _ = (h, buf, len);
        0
    })
}

/// # Safety
/// `h` comes from `nsc_vaapi_open` and is closed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_vaapi_close(h: *mut c_void) {
    guard((), || {
        #[cfg(target_os = "linux")]
        if !h.is_null() {
            // SAFETY: the Box from nsc_vaapi_open.
            drop(unsafe { Box::from_raw(h.cast::<imp::Dec>()) });
        }
        let _ = h;
    });
}
