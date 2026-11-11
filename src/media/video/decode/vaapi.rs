//! VA-API (Intel, AMD; NVIDIA via nvidia-vaapi-driver).
//!
//! The actual decoding lives in a separate library, vaapi-helper (cros-codecs on
//! top of libva). It is embedded in the plugin as bytes and loaded from a memfd
//! via dlopen on the first GPU video. That keeps libva/libgbm dependencies of the
//! helper only: without libva, VA-API is unavailable, but the plugin still loads
//! and keeps working on the CPU. The helper is shared by all decoders and is
//! unloaded together with the last one.
//!
//! A frame arrives one of two ways. As NV12 planes copied out of video memory, which
//! are converted to BGRA here. Or, when the compositor draws with the same GPU, as an
//! RGB picture that never leaves video memory: the helper has the GPU's video
//! processor convert and scale it, and the compositor imports its dma-buf as a
//! texture. Such a picture belongs to this frame until the frame is dropped; then its
//! token goes back to the helper (on the decoder thread, with the next packet).

use std::ffi::{CString, c_char, c_int, c_void};
use std::os::fd::{FromRawFd, OwnedFd};
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use libloading::Library;

use crate::frame::{DmaBufFrame, DmaBufPlane, FrameData};
use crate::media::video::pipeline::{Codec, DecodedFrame, Decoder, Packet, PipeResult};
use crate::media::video::yuv::{Chroma, ChromaPlanes, Converter, Matrix, Plane, YuvImage};

#[cfg(nsc_vaapi_embedded)]
const HELPER_ABI: u32 = 2;

#[cfg(nsc_vaapi_embedded)]
static HELPER_SO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vaapi-helper.so"));

/// See vaapi-helper: in memory (`y` is not NULL) or in video memory.
#[repr(C)]
struct NscVaFrame {
    width: u32,
    height: u32,
    y: *const u8,
    y_stride: usize,
    uv: *const u8,
    uv_stride: usize,
    pts_ns: u64,
    fd: i32,
    drm_format: u32,
    modifier: u64,
    offset: u32,
    pitch: u32,
    buffer: u64,
    token: u64,
}

type FrameCb = unsafe extern "C" fn(*mut c_void, *const NscVaFrame);

struct Helper {
    open: unsafe extern "C" fn(*const c_char, u32, *mut c_char, usize) -> *mut c_void,
    decode: unsafe extern "C" fn(*mut c_void, *const u8, usize, u64, FrameCb, *mut c_void) -> c_int,
    flush: unsafe extern "C" fn(*mut c_void, FrameCb, *mut c_void) -> c_int,
    reset: unsafe extern "C" fn(*mut c_void),
    export: unsafe extern "C" fn(*mut c_void, bool) -> c_int,
    set_target: unsafe extern "C" fn(*mut c_void, u32, u32),
    release: unsafe extern "C" fn(*mut c_void, u64),
    note: unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> bool,
    #[allow(dead_code)] // tests read pictures back
    read: unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> c_int,
    error: unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> usize,
    close: unsafe extern "C" fn(*mut c_void),
    _lib: Library,
}

/// One shared helper, alive while at least one decoder is.
static HELPER: Mutex<Weak<Helper>> = Mutex::new(Weak::new());

fn helper() -> Result<Arc<Helper>, String> {
    let mut slot = HELPER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = slot.upgrade() {
        return Ok(h);
    }
    let h = Arc::new(load()?);
    *slot = Arc::downgrade(&h);
    Ok(h)
}

#[cfg(not(nsc_vaapi_embedded))]
fn load() -> Result<Helper, String> {
    Err("plugin built without VA-API (make without vaapi-helper)".into())
}

#[cfg(nsc_vaapi_embedded)]
fn load() -> Result<Helper, String> {
    use std::io::Write;
    use std::os::fd::FromRawFd;

    // SAFETY: memfd_create with a valid C string; the fd is immediately owned by File.
    let fd = unsafe { libc::memfd_create(c"noshare-cover-vaapi".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(format!("memfd_create: {}", std::io::Error::last_os_error()));
    }
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.write_all(HELPER_SO)
        .map_err(|e| format!("memfd: {e}"))?;
    let path = format!("/proc/self/fd/{fd}");
    // SAFETY: our own helper; no constructors with side effects.
    // After dlopen the mapping stays alive on its own, so the fd can be closed.
    let lib = unsafe { Library::new(&path) }.map_err(|e| {
        let e = e.to_string();
        if e.contains("libva") || e.contains("libgbm") {
            format!("libva/libgbm not found for VA-API (install libva): {e}")
        } else {
            format!("failed to load VA-API helper: {e}")
        }
    })?;
    drop(file);

    // SAFETY: signatures come from vaapi-helper/src/lib.rs; the ABI is checked below.
    unsafe {
        let abi: unsafe extern "C" fn() -> u32 =
            *lib.get(b"nsc_vaapi_abi\0").map_err(|e| e.to_string())?;
        if abi() != HELPER_ABI {
            return Err("VA-API helper version mismatch".into());
        }
        Ok(Helper {
            open: *lib.get(b"nsc_vaapi_open\0").map_err(|e| e.to_string())?,
            decode: *lib.get(b"nsc_vaapi_decode\0").map_err(|e| e.to_string())?,
            flush: *lib.get(b"nsc_vaapi_flush\0").map_err(|e| e.to_string())?,
            reset: *lib.get(b"nsc_vaapi_reset\0").map_err(|e| e.to_string())?,
            export: *lib.get(b"nsc_vaapi_export\0").map_err(|e| e.to_string())?,
            set_target: *lib
                .get(b"nsc_vaapi_set_target\0")
                .map_err(|e| e.to_string())?,
            release: *lib.get(b"nsc_vaapi_release\0").map_err(|e| e.to_string())?,
            note: *lib.get(b"nsc_vaapi_note\0").map_err(|e| e.to_string())?,
            read: *lib.get(b"nsc_vaapi_read\0").map_err(|e| e.to_string())?,
            error: *lib.get(b"nsc_vaapi_error\0").map_err(|e| e.to_string())?,
            close: *lib.get(b"nsc_vaapi_close\0").map_err(|e| e.to_string())?,
            _lib: lib,
        })
    }
}

fn helper_codec(codec: &Codec) -> Option<u32> {
    Some(match codec {
        Codec::H264 => 1,
        Codec::Hevc => 2,
        Codec::Vp8 => 3,
        Codec::Vp9 => 4,
        Codec::Av1 => 5,
        Codec::Other(_) => return None,
    })
}

/// Tokens of frames in video memory that nobody shows any more, on their way back to
/// the helper. A frame is dropped on whatever thread held it last; the helper is only
/// ever called from the decoder thread.
type Released = Arc<Mutex<Vec<u64>>>;

pub struct VaapiDecoder {
    h: Arc<Helper>,
    handle: *mut c_void,
    conv: Converter,
    released: Released,
}

/// What the helper's frame callback gets: where the frames go, what converts the ones
/// that come as NV12, and where the tokens of the others go back.
struct Sink<'a> {
    frames: Vec<DecodedFrame>,
    conv: &'a mut Converter,
    released: &'a Released,
}

// The VA context is used only from the decoder thread: opened, used and closed
// by a single owner; the pointer never escapes.
unsafe impl Send for VaapiDecoder {}

impl VaapiDecoder {
    /// `shared`: the compositor draws with this GPU. Frames are then handed out in video
    /// memory, if the GPU has a video processor for it; if not, as everywhere else.
    pub fn new(codec: &Codec, node: &Path, shared: bool) -> PipeResult<Self> {
        let c = helper_codec(codec).ok_or_else(|| format!("VA-API doesn't support {codec:?}"))?;
        let h = helper()?;
        let node_c = CString::new(node.as_os_str().as_encoded_bytes())
            .map_err(|_| "render node path contains a NUL byte".to_string())?;
        let mut err = [0 as c_char; 512];
        // SAFETY: the C string and error buffer outlive the call.
        let handle = unsafe { (h.open)(node_c.as_ptr(), c, err.as_mut_ptr(), err.len()) };
        if handle.is_null() {
            // SAFETY: the helper writes a C string into err.
            let msg = unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(if msg.is_empty() {
                "VA-API: failed to open".into()
            } else {
                msg
            });
        }
        // SAFETY: handle is alive. A refusal only means frames come the other way.
        if shared {
            unsafe { (h.export)(handle, true) };
        }
        Ok(Self {
            h,
            handle,
            conv: Converter::default(),
            released: Released::default(),
        })
    }

    /// Whether frames stay in video memory, and if not, the helper's reason.
    pub fn in_video_memory(&self) -> (bool, String) {
        let mut buf = [0 as c_char; 512];
        // SAFETY: handle is alive; the buffer is 512 bytes.
        let on = unsafe { (self.h.note)(self.handle, buf.as_mut_ptr(), buf.len()) };
        let why = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        (on, why)
    }

    /// Hand the pictures nobody shows any more back to the helper.
    fn give_back(&mut self) {
        let tokens = std::mem::take(&mut *self.released.lock().unwrap_or_else(|e| e.into_inner()));
        for token in tokens {
            // SAFETY: handle is alive; we are on the decoder thread.
            unsafe { (self.h.release)(self.handle, token) };
        }
    }

    fn last_error(&self) -> String {
        let mut buf = [0 as c_char; 512];
        // SAFETY: handle is alive; the buffer is 512 bytes.
        unsafe { (self.h.error)(self.handle, buf.as_mut_ptr(), buf.len()) };
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

/// A picture in video memory as a frame: its dma-buf (dup-ed, the helper's fd is only
/// good during the callback) and the token that goes back when the frame is dropped.
fn gpu_frame(f: &NscVaFrame, released: &Released) -> Option<DecodedFrame> {
    let (token, back) = (f.token, Arc::clone(released));
    let release = move || back.lock().unwrap_or_else(|e| e.into_inner()).push(token);
    // SAFETY: a dup of an fd that is open for the duration of the callback.
    let fd = unsafe { libc::fcntl(f.fd, libc::F_DUPFD_CLOEXEC, 0) };
    if fd < 0 {
        release();
        return None;
    }
    Some(DecodedFrame {
        pts: Duration::from_nanos(f.pts_ns),
        data: FrameData::DmaBuf(Arc::new(DmaBufFrame {
            width: f.width,
            height: f.height,
            fourcc: f.drm_format,
            modifier: f.modifier,
            planes: vec![DmaBufPlane {
                // SAFETY: a new fd we own.
                fd: unsafe { OwnedFd::from_raw_fd(fd) },
                offset: f.offset,
                stride: f.pitch,
            }],
            buffer: f.buffer,
            release: Some(Box::new(release)),
        })),
    })
}

/// Helper callback: a picture in video memory is passed on as it is; NV12 from the
/// mapped GBM buffer -> BGRA, copied immediately.
unsafe extern "C" fn on_frame(user: *mut c_void, f: *const NscVaFrame) {
    let _ = std::panic::catch_unwind(|| {
        // SAFETY: user is our Sink; f is a frame valid for the duration of the call.
        let (Some(sink), Some(f)) = (unsafe { user.cast::<Sink<'_>>().as_mut() }, unsafe {
            f.as_ref()
        }) else {
            return;
        };
        let (w, h) = (f.width as usize, f.height as usize);
        if w == 0 || h == 0 {
            return;
        }
        if f.y.is_null() && f.fd >= 0 {
            sink.frames.extend(gpu_frame(f, sink.released));
            return;
        }
        if f.y.is_null() || f.uv.is_null() {
            return;
        }
        // SAFETY: the helper guarantees stride*rows bytes per plane.
        let y = unsafe { std::slice::from_raw_parts(f.y, f.y_stride * (h - 1) + w) };
        let ch = h.div_ceil(2);
        let uv =
            unsafe { std::slice::from_raw_parts(f.uv, f.uv_stride * (ch - 1) + w.div_ceil(2) * 2) };
        let img = YuvImage {
            width: f.width,
            height: f.height,
            bit_depth: 8,
            chroma: Chroma::Sub420,
            matrix: Matrix::guess(f.height),
            full_range: false,
            y: Plane {
                data: y,
                stride: f.y_stride,
            },
            uv: ChromaPlanes::Interleaved(Plane {
                data: uv,
                stride: f.uv_stride,
            }),
        };
        if let Some(cpu) = sink.conv.convert(&img) {
            sink.frames.push(DecodedFrame {
                pts: Duration::from_nanos(f.pts_ns),
                data: FrameData::Cpu(cpu),
            });
        }
    });
}

impl Decoder for VaapiDecoder {
    fn name(&self) -> &'static str {
        "VA-API (GPU)"
    }

    fn decode(&mut self, packet: &Packet) -> PipeResult<Vec<DecodedFrame>> {
        self.give_back();
        let mut sink = Sink {
            frames: Vec::new(),
            conv: &mut self.conv,
            released: &self.released,
        };
        let pts = u64::try_from(packet.pts.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the packet and sink outlive the call; the callback is synchronous.
        let r = unsafe {
            (self.h.decode)(
                self.handle,
                packet.data.as_ptr(),
                packet.data.len(),
                pts,
                on_frame,
                (&raw mut sink).cast(),
            )
        };
        let frames = sink.frames;
        if r != 0 {
            return Err(self.last_error());
        }
        Ok(frames)
    }

    fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
        self.give_back();
        let mut sink = Sink {
            frames: Vec::new(),
            conv: &mut self.conv,
            released: &self.released,
        };
        // SAFETY: same as in decode.
        let r = unsafe { (self.h.flush)(self.handle, on_frame, (&raw mut sink).cast()) };
        let frames = sink.frames;
        if r != 0 {
            return Err(self.last_error());
        }
        Ok(frames)
    }

    fn reset(&mut self) {
        // SAFETY: handle is alive.
        unsafe { (self.h.reset)(self.handle) };
    }

    fn set_target(&mut self, width: u32, height: u32) {
        self.conv.set_target(width, height);
        // SAFETY: handle is alive.
        unsafe { (self.h.set_target)(self.handle, width, height) };
    }

    fn idle(&mut self) {
        self.conv.idle();
        self.give_back();
    }

    fn refuse_video_memory(&mut self) {
        // SAFETY: handle is alive. The pictures still out stay valid until dropped.
        unsafe { (self.h.export)(self.handle, false) };
    }
}

impl Drop for VaapiDecoder {
    fn drop(&mut self) {
        // SAFETY: closed exactly once; the helper is unloaded with the last Arc.
        unsafe { (self.h.close)(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_match_helper() {
        assert_eq!(helper_codec(&Codec::H264), Some(1));
        assert_eq!(helper_codec(&Codec::Av1), Some(5));
        assert_eq!(helper_codec(&Codec::Other("x".into())), None);
    }

    /// The pixels of a frame in video memory, as BGRA.
    #[cfg(all(nsc_vaapi_embedded, feature = "cpu-h264"))]
    fn read_back(dec: &VaapiDecoder, d: &DmaBufFrame) -> Vec<u8> {
        let mut px = vec![0u8; (d.width * d.height * 4) as usize];
        // SAFETY: handle is alive; the buffer is as large as said.
        let r = unsafe { (dec.h.read)(dec.handle, d.buffer, px.as_mut_ptr(), px.len()) };
        assert_eq!(r, 0, "{}", dec.last_error());
        if &d.fourcc.to_le_bytes() == b"XB24" {
            for p in px.as_chunks_mut::<4>().0 {
                p.swap(0, 2);
            }
        }
        px
    }

    /// Mean absolute difference of the colour channels of two BGRA pictures.
    #[cfg(all(nsc_vaapi_embedded, feature = "cpu-h264"))]
    fn mean_diff(a: &[u8], b: &[u8]) -> f64 {
        let sum: u64 = a
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.as_chunks::<4>().0)
            .map(|(p, q)| (0..3).map(|i| u64::from(p[i].abs_diff(q[i]))).sum::<u64>())
            .sum();
        sum as f64 / (a.len() / 4 * 3) as f64
    }

    /// A frame converted by the GPU's video processor is the picture the CPU path makes
    /// of it: same matrix, same range, no other colour mapping. On the render node in
    /// NSC_TEST_GPU with the clips in NSC_TEST_MEDIA; skipped without them.
    #[cfg(all(nsc_vaapi_embedded, feature = "cpu-h264"))]
    #[test]
    fn a_frame_in_video_memory_looks_like_the_cpu_one() {
        use crate::media::video::decode::h264::H264Decoder;
        use crate::media::video::demux;
        let (Some(node), Some(dir)) = (
            std::env::var_os("NSC_TEST_GPU").map(std::path::PathBuf::from),
            std::env::var_os("NSC_TEST_MEDIA").map(std::path::PathBuf::from),
        ) else {
            return;
        };
        let path = dir.join("h264.mp4");
        if !path.exists() {
            return;
        }
        let mut gpu = VaapiDecoder::new(&Codec::H264, &node, true).unwrap();
        let (on, why) = gpu.in_video_memory();
        assert!(on, "no video processor on {}: {why}", node.display());
        let Ok(mut cpu) = H264Decoder::new() else {
            eprintln!("no openh264 — skipping");
            return;
        };

        let mut d = demux::open(&path).unwrap();
        // The `nth` frame the GPU hands out from here on, and the CPU's frame of the same
        // moment. Only that one GPU frame is kept: each holds a picture in video memory.
        let mut same_moment = |gpu: &mut VaapiDecoder, cpu: &mut H264Decoder, nth: usize| {
            let (mut seen, mut from_cpu) = (0, Vec::new());
            let mut kept: Option<DecodedFrame> = None;
            loop {
                if let Some(g) = &kept
                    && let Some(c) = from_cpu.iter().position(|c: &DecodedFrame| {
                        // the two decoders round the packet's time differently
                        c.pts.abs_diff(g.pts) < Duration::from_millis(1)
                    })
                {
                    return (kept.take().unwrap(), from_cpu.swap_remove(c));
                }
                let p = d.next_packet().unwrap().expect("the clip is long enough");
                from_cpu.extend(cpu.decode(&p).unwrap());
                for f in gpu.decode(&p).unwrap() {
                    if seen == nth {
                        kept = Some(f);
                    }
                    seen += 1;
                }
            }
        };
        let pictures = |gpu: &VaapiDecoder, g: &DecodedFrame, c: &DecodedFrame| {
            let (FrameData::DmaBuf(g), FrameData::Cpu(c)) = (&g.data, &c.data) else {
                panic!("the frame left video memory: {}", gpu.in_video_memory().1)
            };
            assert_eq!((g.width, g.height), (c.width, c.height));
            (
                (g.width, g.height),
                mean_diff(&read_back(gpu, g), &c.pixels),
            )
        };

        let (g, c) = same_moment(&mut gpu, &mut cpu, 30);
        let (size, diff) = pictures(&gpu, &g, &c);
        assert_eq!(size, (640, 360));
        eprintln!("GPU vs CPU picture: {diff:.2} levels apart on average");
        // edges differ (the GPU interpolates chroma), flat areas must not
        assert!(
            diff < 2.5,
            "the GPU's picture is off by {diff:.2} levels on average"
        );
        drop((g, c));

        // shown at half the size: the GPU scales, the CPU halves
        gpu.set_target(320, 180);
        cpu.set_target(320, 180);
        let (g, c) = same_moment(&mut gpu, &mut cpu, 30);
        let (size, diff) = pictures(&gpu, &g, &c);
        assert_eq!(size, (320, 180));
        eprintln!("at half the size: {diff:.2} levels apart on average");
        assert!(diff < 4.0, "the scaled picture is off by {diff:.2} levels");
    }

    #[test]
    fn missing_helper_or_device_is_an_error() {
        let e = VaapiDecoder::new(&Codec::H264, Path::new("/nonexistent/renderD999"), false)
            .err()
            .unwrap();
        eprintln!("VA-API without a device: {e}");
        assert!(!e.is_empty());
        // with the embedded helper the error comes from the helper itself (memfd + dlopen + ABI succeeded)
        #[cfg(nsc_vaapi_embedded)]
        assert!(e.contains("VA-API on") || e.contains("libva"), "{e}");
    }
}
