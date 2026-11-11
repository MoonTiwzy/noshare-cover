//! Demuxers: MP4/MOV (re_mp4) and WebM/MKV (matroska-demuxer), both pure Rust.
//!
//! The file is never read into memory whole: from MP4 we take only the index (moov);
//! samples are read from disk by offset. Output packets are already in the form
//! decoders expect: H.264/HEVC as Annex-B with parameter sets before keyframes,
//! AV1/VP8/VP9 as stored in the container.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

use super::bitstream::{self, NalConfig};
use super::pipeline::{Codec, Demuxer, Packet, PipeResult, VideoInfo};

/// How to turn a container sample into a decoder packet.
enum Framing {
    /// AVCC/HVCC → Annex-B
    Nal(NalConfig),
    /// as is (AV1, VP8, VP9)
    Raw,
}

impl Framing {
    fn for_codec(codec: &Codec, config: Option<&[u8]>) -> PipeResult<Self> {
        match codec {
            Codec::H264 => config
                .and_then(bitstream::parse_avcc)
                .map(Framing::Nal)
                .ok_or_else(|| "H.264 avcC missing or broken".into()),
            Codec::Hevc => config
                .and_then(bitstream::parse_hvcc)
                .map(Framing::Nal)
                .ok_or_else(|| "HEVC hvcC missing or broken".into()),
            _ => Ok(Framing::Raw),
        }
    }

    fn packet(&self, data: Vec<u8>, pts: Duration, keyframe: bool) -> Option<Packet> {
        let data = match self {
            Framing::Nal(cfg) => bitstream::to_annexb(&data, cfg, keyframe)?,
            Framing::Raw => data,
        };
        Some(Packet {
            pts,
            keyframe,
            data,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    Mp4,
    Matroska,
}

/// Detect the container from the first bytes: extensions can't be trusted (an mp4 named .mkv
/// is common). If unrecognized, fall back to the extension.
fn sniff(path: &Path) -> PipeResult<Container> {
    let mut head = [0u8; 12];
    let n = File::open(path)
        .and_then(|mut f| f.read(&mut head))
        .map_err(|e| e.to_string())?;
    let head = &head[..n];
    if head.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return Ok(Container::Matroska);
    }
    if head.len() >= 8
        && matches!(
            &head[4..8],
            b"ftyp" | b"moov" | b"mdat" | b"wide" | b"free" | b"skip"
        )
    {
        return Ok(Container::Mp4);
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "mp4" | "m4v" | "mov" => Ok(Container::Mp4),
        "webm" | "mkv" => Ok(Container::Matroska),
        other => Err(format!("container .{other} is not supported")),
    }
}

pub fn open(path: &Path) -> PipeResult<Box<dyn Demuxer>> {
    Ok(match sniff(path)? {
        Container::Mp4 => Box::new(Mp4Demux::open(path)?),
        Container::Matroska => Box::new(MkvDemux::open(path)?),
    })
}

// ------------------------------------------------------------------ MP4 / MOV

pub struct Mp4Demux {
    file: BufReader<File>,
    info: VideoInfo,
    framing: Framing,
    samples: Vec<re_mp4::Sample>,
    next: usize,
}

impl Mp4Demux {
    pub fn open(path: &Path) -> PipeResult<Self> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        let mut reader = BufReader::new(file);
        let mp4 = re_mp4::Mp4::read(&mut reader, size).map_err(|e| format!("MP4: {e}"))?;

        let track = mp4
            .tracks()
            .values()
            .find(|t| t.kind == Some(re_mp4::TrackKind::Video))
            .ok_or("no video track in MP4")?;

        let codec = codec_from_mp4(track, &mp4);
        let config = track.raw_codec_config(&mp4);
        let framing = Framing::for_codec(&codec, config.as_deref())?;
        let duration = (track.timescale > 0 && track.duration > 0)
            .then(|| Duration::from_secs_f64(track.duration as f64 / track.timescale as f64));

        Ok(Self {
            info: VideoInfo {
                width: u32::from(track.width),
                height: u32::from(track.height),
                duration,
                codec,
            },
            framing,
            samples: track.samples.clone(),
            file: reader,
            next: 0,
        })
    }
}

fn codec_from_mp4(track: &re_mp4::Track, mp4: &re_mp4::Mp4) -> Codec {
    use re_mp4::StsdBoxContent as S;
    match &track.trak(mp4).mdia.minf.stbl.stsd.contents {
        S::Av01(_) => Codec::Av1,
        S::Avc1(_) => Codec::H264,
        S::Hev1(_) | S::Hvc1(_) => Codec::Hevc,
        S::Vp08(_) => Codec::Vp8,
        S::Vp09(_) => Codec::Vp9,
        other => Codec::Other(format!("{other:?}").chars().take(16).collect()),
    }
}

impl Demuxer for Mp4Demux {
    fn info(&self) -> &VideoInfo {
        &self.info
    }

    fn next_packet(&mut self) -> PipeResult<Option<Packet>> {
        // skip broken samples, but not indefinitely
        while let Some(s) = self.samples.get(self.next).copied() {
            self.next += 1;
            if s.size == 0 || s.size > 256 * 1024 * 1024 {
                continue;
            }
            self.file
                .seek(SeekFrom::Start(s.offset))
                .map_err(|e| e.to_string())?;
            let mut data = vec![0u8; s.size as usize];
            self.file
                .read_exact(&mut data)
                .map_err(|e| format!("MP4 sample {}: {e}", s.id))?;
            let pts_units = s.composition_timestamp.max(0) as f64;
            let pts = if s.timescale > 0 {
                Duration::from_secs_f64(pts_units / s.timescale as f64)
            } else {
                Duration::ZERO
            };
            if let Some(p) = self.framing.packet(data, pts, s.is_sync) {
                return Ok(Some(p));
            }
        }
        Ok(None)
    }

    fn rewind(&mut self) -> PipeResult<()> {
        self.next = 0;
        Ok(())
    }
}

// ------------------------------------------------------------------ WebM / MKV

pub struct MkvDemux {
    mkv: matroska_demuxer::MatroskaFile<BufReader<File>>,
    track: u64,
    /// nanoseconds per Matroska time unit
    scale_ns: u64,
    info: VideoInfo,
    framing: Framing,
    frame: matroska_demuxer::Frame,
}

impl MkvDemux {
    pub fn open(path: &Path) -> PipeResult<Self> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mkv = matroska_demuxer::MatroskaFile::open(BufReader::new(file))
            .map_err(|e| format!("MKV: {e}"))?;

        let entry = mkv
            .tracks()
            .iter()
            .find(|t| t.track_type() == matroska_demuxer::TrackType::Video)
            .ok_or("no video track in MKV")?;

        let codec = codec_from_mkv(entry.codec_id());
        let framing = Framing::for_codec(&codec, entry.codec_private())?;
        let (width, height) = entry
            .video()
            .map(|v| (v.pixel_width().get() as u32, v.pixel_height().get() as u32))
            .unwrap_or((0, 0));
        let scale_ns = mkv.info().timestamp_scale().get();
        let duration = mkv
            .info()
            .duration()
            .filter(|d| *d > 0.0)
            .map(|d| Duration::from_secs_f64(d * scale_ns as f64 / 1e9));
        let track = entry.track_number().get();

        Ok(Self {
            info: VideoInfo {
                width,
                height,
                duration,
                codec,
            },
            framing,
            track,
            scale_ns,
            frame: Default::default(),
            mkv,
        })
    }
}

fn codec_from_mkv(id: &str) -> Codec {
    match id {
        "V_AV1" => Codec::Av1,
        "V_MPEG4/ISO/AVC" => Codec::H264,
        "V_MPEGH/ISO/HEVC" => Codec::Hevc,
        "V_VP8" => Codec::Vp8,
        "V_VP9" => Codec::Vp9,
        other => Codec::Other(other.to_owned()),
    }
}

impl Demuxer for MkvDemux {
    fn info(&self) -> &VideoInfo {
        &self.info
    }

    fn next_packet(&mut self) -> PipeResult<Option<Packet>> {
        loop {
            let more = self
                .mkv
                .next_frame(&mut self.frame)
                .map_err(|e| format!("MKV: {e}"))?;
            if !more {
                return Ok(None);
            }
            if self.frame.track != self.track || self.frame.is_invisible {
                continue;
            }
            let pts = Duration::from_nanos(self.frame.timestamp.saturating_mul(self.scale_ns));
            // is_keyframe is only known for SimpleBlock; otherwise treat the first frame as a keyframe
            let key = self.frame.is_keyframe.unwrap_or(false);
            let data = std::mem::take(&mut self.frame.data);
            if let Some(p) = self.framing.packet(data, pts, key) {
                return Ok(Some(p));
            }
        }
    }

    fn rewind(&mut self) -> PipeResult<()> {
        self.mkv.seek(0).map_err(|e| format!("MKV seek: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_ids() {
        assert_eq!(codec_from_mkv("V_AV1"), Codec::Av1);
        assert_eq!(codec_from_mkv("V_MPEG4/ISO/AVC"), Codec::H264);
        assert_eq!(codec_from_mkv("V_VP9"), Codec::Vp9);
        assert!(matches!(codec_from_mkv("V_THEORA"), Codec::Other(_)));
    }

    #[test]
    fn missing_file_is_error() {
        assert!(open(Path::new("/nonexistent/x.mp4")).is_err());
        assert!(open(Path::new("/nonexistent/x.avi")).is_err());
    }

    /// Real clips from `NSC_TEST_MEDIA` (10 s, 640x360). Without the directory the test is skipped.
    #[test]
    fn real_files() {
        let Some(dir) = std::env::var_os("NSC_TEST_MEDIA").map(std::path::PathBuf::from) else {
            eprintln!("NSC_TEST_MEDIA not set — skipping");
            return;
        };
        for (name, codec) in [
            ("h264.mp4", Codec::H264),
            ("av1.mp4", Codec::Av1),
            ("vp9.webm", Codec::Vp9),
            ("h264.mkv", Codec::H264),
        ] {
            let path = dir.join(name);
            if !path.exists() {
                eprintln!("{name}: file missing — skipping");
                continue;
            }
            let mut d = open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
            let info = d.info().clone();
            assert_eq!(info.codec, codec, "{name}");
            assert_eq!((info.width, info.height), (640, 360), "{name}");
            let dur = info.duration.expect("duration").as_secs_f64();
            assert!((9.0..11.0).contains(&dur), "{name}: duration {dur}");

            let mut count = 0;
            let mut max_pts = Duration::ZERO;
            let mut first_key = None;
            while let Some(p) = d.next_packet().unwrap() {
                if count == 0 {
                    first_key = Some(p.keyframe);
                    if codec == Codec::H264 {
                        assert_eq!(&p.data[..4], &[0, 0, 0, 1], "{name}: Annex-B");
                        assert_eq!(p.data[4] & 0x1f, 7, "{name}: SPS before keyframe");
                    }
                }
                assert!(!p.data.is_empty());
                max_pts = max_pts.max(p.pts);
                count += 1;
            }
            eprintln!("{name}: {count} packets, max pts {max_pts:?}, {dur:.2} s");
            assert!(count > 200, "{name}: {count} packets");
            assert!(max_pts.as_secs_f64() > 9.0, "{name}: pts {max_pts:?}");
            if name.ends_with(".mp4") {
                assert_eq!(first_key, Some(true), "{name}: first packet must be key");
            }

            d.rewind().unwrap();
            let again = d.next_packet().unwrap().expect("packet after rewind");
            assert!(
                again.pts < Duration::from_millis(100),
                "{name}: rewind pts {:?}",
                again.pts
            );
        }
    }

    /// Official Matroska test file (test1.mkv: MPEG-4 Part 2 in MKV).
    /// We don't support the codec, so the demuxer must honestly report Other and
    /// still read the whole file (clusters, BlockGroup, lacing).
    #[test]
    fn real_matroska_file() {
        let Some(path) = std::env::var_os("NSC_TEST_MEDIA")
            .map(|d| std::path::PathBuf::from(d).join("real_test1.mkv"))
        else {
            return;
        };
        if !path.exists() {
            return;
        }
        let mut d = open(&path).unwrap();
        assert!(
            matches!(d.info().codec, Codec::Other(ref id) if id.starts_with("V_MS")),
            "{:?}",
            d.info().codec
        );
        assert!(d.info().width > 0);
        let first = d.next_packet().unwrap().unwrap();
        let mut n = 1;
        let mut keys = usize::from(first.keyframe);
        while let Some(p) = d.next_packet().unwrap() {
            keys += usize::from(p.keyframe);
            n += 1;
        }
        eprintln!("test1.mkv: {n} packets, {keys} keyframes, {:?}", d.info());
        assert!(n > 100 && keys > 0);
    }

    #[test]
    fn h264_without_avcc_is_error() {
        assert!(Framing::for_codec(&Codec::H264, None).is_err());
        assert!(Framing::for_codec(&Codec::Av1, None).is_ok());
    }
}
