//! Picks the demuxer and decoder for a file and settings.
//!
//! Order: `backend = gpu`: GPU only, error if unavailable; `cpu`: software only;
//! `auto`: a GPU if one is present and supports the codec, otherwise software. The
//! GPUs are tried in the order of `gpu::render_node_order`: the compositor's own
//! first. Every rejection comes with a reason that is shown to the user instead of
//! being silently dropped.
//!
//! Demuxers live in demux.rs (mp4/mov, webm/mkv), decoders in decode/: VA-API
//! (vaapi-helper), NVDEC, and the CPU fallbacks rav1d (AV1), openh264 (H.264)
//! and libvpx (VP8/VP9). The clock, decode thread and lifecycle are in
//! pipeline.rs and mod.rs.

use std::path::Path;

use super::pipeline::{Codec, Decoder, Demuxer, Pipeline};
use crate::config::{Backend, Settings};
use crate::gpu;

pub fn open_pipeline(path: &Path, looped: bool, settings: &Settings) -> Result<Pipeline, String> {
    let demux = open_demuxer(path)?;
    let decoder = open_decoder(&demux.info().codec, settings)?;
    Ok(Pipeline::new(demux, decoder, looped))
}

fn open_demuxer(path: &Path) -> Result<Box<dyn Demuxer>, String> {
    super::demux::open(path)
}

fn open_decoder(codec: &Codec, settings: &Settings) -> Result<Box<dyn Decoder>, String> {
    let mut reasons = Vec::new();

    if matches!(settings.backend, Backend::Auto | Backend::Gpu) {
        let compositor = settings.compositor.as_deref();
        match gpu::render_node_order(settings.gpu_device.as_deref(), compositor) {
            Ok(nodes) => {
                for node in nodes {
                    // decoded on the compositor's own GPU, a frame can stay in video memory
                    let shared =
                        compositor.is_some() && gpu::device_of_node(&node).as_deref() == compositor;
                    match open_hw(codec, &node, shared) {
                        Ok(d) => return Ok(d),
                        Err(e) => reasons.push(format!("GPU {}: {e}", node.display())),
                    }
                }
            }
            Err(e) => reasons.push(format!("GPU: {e}")),
        }
        if settings.backend == Backend::Gpu {
            return Err(reasons.join("; "));
        }
    }

    match open_sw(codec) {
        Ok(d) => Ok(d),
        Err(e) => {
            reasons.push(format!("CPU: {e}"));
            Err(reasons.join("; "))
        }
    }
}

/// GPU decoder for a render node. NVIDIA: NVDEC directly first (the driver
/// parses the stream itself), then VA-API (nvidia-vaapi-driver); Intel/AMD: VA-API.
/// `shared`: the compositor draws with this GPU, so VA-API hands its frames out in
/// video memory when the GPU can do that.
#[cfg_attr(
    not(all(feature = "vaapi", target_os = "linux")),
    allow(unused_variables)
)]
fn open_hw(codec: &Codec, node: &Path, shared: bool) -> Result<Box<dyn Decoder>, String> {
    let mut reasons: Vec<String> = Vec::new();

    #[cfg(feature = "nvdec")]
    if super::decode::nvdec::node_is_nvidia(node) {
        match super::decode::nvdec::NvdecDecoder::new(codec, Some(node)) {
            Ok(d) => return Ok(Box::new(d)),
            Err(e) => reasons.push(format!("NVDEC: {e}")),
        }
    }

    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    match super::decode::vaapi::VaapiDecoder::new(codec, node, shared) {
        Ok(d) => return Ok(Box::new(d)),
        Err(e) => reasons.push(format!("VA-API: {e}")),
    }
    #[cfg(not(all(feature = "vaapi", target_os = "linux")))]
    reasons.push(format!(
        "VA-API is not available in this build ({codec:?}, {})",
        node.display()
    ));

    Err(reasons.join("; "))
}

fn open_sw(codec: &Codec) -> Result<Box<dyn Decoder>, String> {
    // YUV frames -> premultiplied BGRA in a single pass (yuv.rs).
    match codec {
        #[cfg(feature = "cpu-av1")]
        Codec::Av1 => Ok(Box::new(super::decode::av1::Av1Decoder::new()?)),
        #[cfg(feature = "cpu-h264")]
        Codec::H264 => Ok(Box::new(super::decode::h264::H264Decoder::new()?)),
        #[cfg(feature = "cpu-vpx")]
        Codec::Vp8 | Codec::Vp9 => Ok(Box::new(super::decode::vpx::VpxDecoder::new(codec)?)),
        _ => Err(format!(
            "software decoder for {codec:?} is not wired up yet"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_only_reports_why() {
        let s = Settings {
            backend: Backend::Gpu,
            gpu_device: Some("/nonexistent/renderD999".into()),
            ..Settings::default()
        };
        let err = open_decoder(&Codec::Av1, &s).err().unwrap();
        assert!(err.contains("GPU"), "{err}");
        assert!(!err.contains("CPU"), "gpu-only must not fall back: {err}");
    }

    #[test]
    fn auto_tries_both() {
        let s = Settings {
            backend: Backend::Auto,
            gpu_device: Some("/nonexistent/renderD999".into()),
            ..Settings::default()
        };
        let err = open_decoder(&Codec::Other("V_TEST".into()), &s)
            .err()
            .unwrap();
        assert!(err.contains("GPU") && err.contains("CPU"), "{err}");
    }
}

#[cfg(test)]
mod real {
    use super::*;
    use crate::frame::FrameData;
    use crate::media::video::pipeline::Next;

    fn run_cpu(file: &str, decoder: &str) {
        run(
            file,
            decoder,
            Settings {
                backend: Backend::Cpu,
                ..Settings::default()
            },
        );
    }

    /// GPU decode on the render node in NSC_TEST_GPU (`/dev/dri/renderD129`); without it
    /// the test is skipped.
    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    fn run_gpu(file: &str, decoder: &str) {
        let Some(node) = std::env::var_os("NSC_TEST_GPU") else {
            return;
        };
        run(
            file,
            decoder,
            Settings {
                backend: Backend::Gpu,
                gpu_device: Some(node.into()),
                ..Settings::default()
            },
        );
    }

    fn run(file: &str, decoder: &str, s: Settings) {
        let Some(dir) = std::env::var_os("NSC_TEST_MEDIA").map(std::path::PathBuf::from) else {
            return;
        };
        let path = dir.join(file);
        if !path.exists() {
            return;
        }
        let mut p = match open_pipeline(&path, true, &s) {
            Ok(p) => p,
            Err(e) if e.contains("not found") => {
                eprintln!("{file}: {e} — skipping");
                return;
            }
            Err(e) => panic!("{file}: {e}"),
        };
        assert_eq!(p.decoder_name(), decoder);
        let mut last = None;
        for n in 0..400 {
            let Next::Frame(f) = p.next_frame().unwrap() else {
                panic!("loop must not end")
            };
            #[allow(irrefutable_let_patterns)]
            let FrameData::Cpu(c) = &f.data else {
                panic!("expected a CPU frame")
            };
            assert_eq!((c.width, c.height), (640, 360));
            // NSC_DUMP=dir: save frame 60 as PNG to check the colors by eye
            if let (60, Some(d)) = (n, std::env::var_os("NSC_DUMP")) {
                let rgba: Vec<u8> = c
                    .pixels
                    .chunks(4)
                    .flat_map(|p| [p[2], p[1], p[0], p[3]])
                    .collect();
                crate::media::still::write_png(
                    &std::path::Path::new(&d).join(format!("{file}.png")),
                    c.width,
                    c.height,
                    &rgba,
                );
            }
            if let Some(prev) = last {
                assert!(
                    f.pts > prev,
                    "{file}: pts keep increasing across the loop: {prev:?} -> {:?}",
                    f.pts
                );
            }
            last = Some(f.pts);
        }
        assert!(last.unwrap() > std::time::Duration::from_secs(10));
    }

    #[cfg(feature = "cpu-h264")]
    #[test]
    fn h264_mp4_decodes_on_cpu() {
        run_cpu("h264.mp4", "openh264 (CPU)");
    }

    #[cfg(feature = "cpu-vpx")]
    #[test]
    fn vp9_webm_decodes_on_cpu() {
        run_cpu("vp9.webm", "libvpx VP9 (CPU)");
    }

    #[cfg(feature = "cpu-av1")]
    #[test]
    fn av1_mp4_decodes_on_cpu() {
        run_cpu("av1.mp4", "rav1d (CPU)");
    }

    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    #[test]
    fn h264_mp4_decodes_on_gpu() {
        run_gpu("h264.mp4", "VA-API (GPU)");
    }

    /// With the compositor on the decoder's GPU the frames never leave video memory: RGB
    /// dma-bufs, a few buffers taking turns, scaled towards the size they are shown at.
    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    #[test]
    fn frames_stay_in_video_memory_on_the_compositors_gpu() {
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
        let s = Settings {
            backend: Backend::Gpu,
            compositor: gpu::device_of_node(&node),
            gpu_device: Some(node),
            ..Settings::default()
        };
        assert!(s.compositor.is_some(), "the node has a device");
        let mut p = open_pipeline(&path, true, &s).unwrap();
        let mut buffers = std::collections::HashSet::new();
        for n in 0..300 {
            if n == 150 {
                p.set_target(300, 100); // half the size still covers it
            }
            let Next::Frame(f) = p.next_frame().unwrap() else {
                panic!("loop must not end")
            };
            let FrameData::DmaBuf(d) = &f.data else {
                panic!("frame {n} came through system memory")
            };
            match n {
                0..150 => assert_eq!((d.width, d.height), (640, 360), "frame {n}"),
                // what was decoded before the size was known is still on its way
                150..170 => {}
                _ => assert_eq!((d.width, d.height), (320, 180), "frame {n}"),
            }
            assert_eq!(d.planes.len(), 1, "one RGB plane");
            assert!(d.planes[0].stride >= d.width * 4);
            let fourcc = d.fourcc.to_le_bytes();
            assert!(
                matches!(&fourcc, b"XR24" | b"XB24"),
                "an opaque RGB format, got {}",
                String::from_utf8_lossy(&fourcc)
            );
            assert_ne!(d.buffer, 0);
            buffers.insert(d.buffer);
        }
        assert!(
            buffers.len() <= 16,
            "{} buffers for 300 frames: they must take turns",
            buffers.len()
        );
    }

    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    #[test]
    fn vp9_webm_decodes_on_gpu() {
        run_gpu("vp9.webm", "VA-API (GPU)");
    }

    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    #[test]
    fn av1_mp4_decodes_on_gpu() {
        run_gpu("av1.mp4", "VA-API (GPU)");
    }
}
