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
        // only the video's samples: an audio track's (QuickTime counts one per PCM frame
        // in some, more than the file has bytes) aren't needed
        let video = |trak: &re_mp4::TrakBox| {
            trak.mdia.minf.stbl.stsd.kind() == Some(re_mp4::TrackKind::Video)
        };
        let mp4 =
            re_mp4::Mp4::read_tracks(&mut reader, size, video).map_err(|e| format!("MP4: {e}"))?;

        let track = mp4
            .tracks()
            .values()
            .find(|t| t.kind == Some(re_mp4::TrackKind::Video))
            .ok_or("no video track in MP4")?;
        let id = track.track_id;

        let codec = codec_from_mp4(track, &mp4);
        let config = track.raw_codec_config(&mp4);
        let framing = Framing::for_codec(&codec, config.as_deref())?;
        // a duration longer than a Duration holds is as good as none
        let duration = (track.timescale > 0 && track.duration > 0)
            .then(|| track.duration as f64 / track.timescale as f64)
            .and_then(|secs| Duration::try_from_secs_f64(secs).ok());

        let info = VideoInfo {
            width: u32::from(track.width),
            height: u32::from(track.height),
            duration,
            codec,
        };
        // taken, not copied: there may be millions
        let samples = mp4
            .into_tracks()
            .remove(&id)
            .map(|track| track.samples)
            .unwrap_or_default();

        Ok(Self {
            info,
            framing,
            samples,
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
    /// The open file, read again from its start on a rewind: the same file even if the path
    /// is another one by then.
    file: File,
    track: u64,
    /// nanoseconds per Matroska time unit
    scale_ns: u64,
    info: VideoInfo,
    framing: Framing,
    /// bytes the muxer stripped from the front of every frame (header stripping)
    stripped: Vec<u8>,
    frame: matroska_demuxer::Frame,
    /// nothing read yet since the start of the file
    first: bool,
    /// the video track's DefaultDuration, in nanoseconds
    default_ns: Option<u64>,
    /// see `Demuxer::frames_end`
    frames_end: Option<Duration>,
    /// the file has tracks besides the video
    others: bool,
}

impl MkvDemux {
    pub fn open(path: &Path) -> PipeResult<Self> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let mkv = Self::read(&file)?;

        let entry = mkv
            .tracks()
            .iter()
            .find(|t| t.track_type() == matroska_demuxer::TrackType::Video)
            .ok_or("no video track in MKV")?;

        let codec = codec_from_mkv(entry.codec_id());
        let stripped = Stripped::from_track(entry)?;
        let private = entry
            .codec_private()
            .map(|private| [&stripped.private[..], private].concat());
        let framing = Framing::for_codec(&codec, private.as_deref())?;
        let (width, height) = entry
            .video()
            .map(|v| (v.pixel_width().get() as u32, v.pixel_height().get() as u32))
            .unwrap_or((0, 0));
        let scale_ns = mkv.info().timestamp_scale().get();
        // an infinite duration, or one longer than a Duration holds, is as good as none
        let duration = mkv
            .info()
            .duration()
            .filter(|d| *d > 0.0)
            .and_then(|d| Duration::try_from_secs_f64(d * scale_ns as f64 / 1e9).ok());
        let track = entry.track_number().get();
        let default_ns = entry.default_duration().map(|d| d.get());

        Ok(Self {
            info: VideoInfo {
                width,
                height,
                duration,
                codec,
            },
            framing,
            stripped: stripped.frames,
            track,
            scale_ns,
            frame: Default::default(),
            first: true,
            default_ns,
            frames_end: None,
            others: mkv.tracks().len() > 1,
            mkv,
            file,
        })
    }

    /// Reads the file from its start, through a handle of its own. The handles share the
    /// file's position, and only the reader in use moves it.
    fn read(file: &File) -> PipeResult<matroska_demuxer::MatroskaFile<BufReader<File>>> {
        let mut file = file.try_clone().map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        matroska_demuxer::MatroskaFile::open(BufReader::new(file)).map_err(|e| format!("MKV: {e}"))
    }
}

/// What a track's ContentEncodings (RFC 9559, 5.1.4.1.31) did to its frames and its
/// CodecPrivate: header stripping (mkvmerge's "header removal") takes the same bytes off
/// the front of each, and putting them back gives what the codec wrote. A track compressed
/// otherwise (zlib, bzlib, lzo) or encrypted is refused: its frames can't be decoded as
/// they are.
#[derive(Default)]
struct Stripped {
    frames: Vec<u8>,
    private: Vec<u8>,
}

impl Stripped {
    /// ContentEncodingScope: the frames, CodecPrivate (4, the next ContentEncoding, is
    /// neither)
    const FRAMES: u64 = 1;
    const PRIVATE: u64 = 2;

    fn from_track(entry: &matroska_demuxer::TrackEntry) -> PipeResult<Self> {
        use matroska_demuxer::{ContentCompAlgo, ContentEncAlgo, ContentEncodingValue};

        let mut applied = entry
            .content_encodings()
            .unwrap_or_default()
            .iter()
            .filter(|e| e.scope() & (Self::FRAMES | Self::PRIVATE) != 0);
        let Some(encoding) = applied.next() else {
            return Ok(Self::default());
        };
        if applied.next().is_some() {
            return Err("MKV: the video track has several content encodings, not supported".into());
        }
        let bytes = match encoding.encoding() {
            ContentEncodingValue::Compression(c) if c.algo() == ContentCompAlgo::Stripping => {
                c.settings().unwrap_or_default().to_vec()
            }
            ContentEncodingValue::Compression(c) => {
                return Err(format!(
                    "MKV: the video track is compressed ({:?}), not supported: remux it without compression",
                    c.algo()
                ));
            }
            // "not encrypted" (ContentEncAlgo 0): the frames are as they were
            ContentEncodingValue::Encryption(e) if e.algo() == ContentEncAlgo::NotEncrypted => {
                Vec::new()
            }
            ContentEncodingValue::Encryption(_) => {
                return Err("MKV: the video track is encrypted".into());
            }
            ContentEncodingValue::Unknown => {
                return Err(
                    "MKV: the video track has a content encoding of an unknown type".into(),
                );
            }
        };
        let scope = |bit| {
            if encoding.scope() & bit != 0 {
                bytes.clone()
            } else {
                Vec::new()
            }
        };
        Ok(Self {
            frames: scope(Self::FRAMES),
            private: scope(Self::PRIVATE),
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
            // an invisible frame (a VP8 alt-ref) is decoded like any other, it only has no
            // picture of its own: left out, every frame that refers to it comes out broken
            if self.frame.track != self.track {
                continue;
            }
            let pts = Duration::from_nanos(self.frame.timestamp.saturating_mul(self.scale_ns));
            // a BlockDuration is in the file's time units, a DefaultDuration in nanoseconds
            let own = self
                .frame
                .duration
                .map(|d| d.saturating_mul(self.scale_ns))
                .or(self.default_ns)
                .map(|ns| pts.saturating_add(Duration::from_nanos(ns)));
            self.frames_end = self.frames_end.max(own);
            // is_keyframe is only known for SimpleBlock; otherwise treat the first frame as a keyframe
            let key = self.frame.is_keyframe.unwrap_or(self.first);
            self.first = false;
            let mut data = std::mem::take(&mut self.frame.data);
            if !self.stripped.is_empty() {
                // before the framing: what was stripped is usually the start of a NAL length
                data.splice(0..0, self.stripped.iter().copied());
            }
            if let Some(p) = self.framing.packet(data, pts, key) {
                return Ok(Some(p));
            }
        }
    }

    fn frames_end(&self) -> Option<Duration> {
        self.frames_end
    }

    /// The Segment's duration is the file's: with other tracks it may be where the
    /// longest one ends.
    fn duration_may_be_another_tracks(&self) -> bool {
        self.others
    }

    fn rewind(&mut self) -> PipeResult<()> {
        self.first = true;
        self.frames_end = None;
        // Not a seek: without Cues it goes by the clusters' timestamps and finds nothing when
        // the one cluster starts before the time asked for (the video starts after another
        // track's blocks) or after it (a file cut with its timestamps kept starts minutes in,
        // not at zero). Opened again, the file is read from its first cluster, as at first.
        self.mkv = Self::read(&self.file)?;
        Ok(())
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

    /// A block of a test WebM: its track and its time in ms after its cluster's.
    type Block = (u8, u16);

    /// A Matroska element with an 8-byte size, which any EBML reader takes.
    fn el(id: &[u8], body: &[u8]) -> Vec<u8> {
        let mut v = id.to_vec();
        v.push(0x01);
        v.extend_from_slice(&(body.len() as u64).to_be_bytes()[1..]);
        v.extend_from_slice(body);
        v
    }

    fn uint(id: &[u8], n: u64) -> Vec<u8> {
        el(id, &n.to_be_bytes())
    }

    /// The video track of a test Matroska file: its codec, CodecPrivate and
    /// ContentEncodings element.
    struct VideoTrack<'a> {
        codec: &'a str,
        private: &'a [u8],
        encodings: Vec<u8>,
    }

    /// A block of a test Matroska file: its track, its time in ms after its cluster's,
    /// whether it is a keyframe, and its frame.
    struct MkvBlock {
        track: u8,
        at: u16,
        key: bool,
        data: Vec<u8>,
    }

    /// A WebM in one cluster at `cluster` ms and without Cues, as one written to a pipe:
    /// VP9 frames on track 1 and Opus packets on track 2, its duration written as ffmpeg
    /// writes it (where the file ends).
    fn write_webm(path: &Path, cluster: u64, blocks: &[Block], end_ms: u64) {
        let vp9 = VideoTrack {
            codec: "V_VP9",
            private: &[],
            encodings: Vec::new(),
        };
        let blocks: Vec<_> = blocks
            .iter()
            .enumerate()
            .map(|(n, &(track, at))| MkvBlock {
                track,
                at,
                key: true,
                data: vec![n as u8],
            })
            .collect();
        write_mkv(path, &vp9, end_ms as f64, cluster, &blocks);
    }

    /// A Matroska file in one cluster at `cluster` ms and without Cues: `video` on track 1,
    /// Opus on track 2, and `duration` (in ms) in its Info.
    fn write_mkv(
        path: &Path,
        video: &VideoTrack,
        duration: f64,
        cluster: u64,
        blocks: &[MkvBlock],
    ) {
        let header = [
            uint(&[0x42, 0x86], 1), // EBMLVersion
            uint(&[0x42, 0xF7], 1), // EBMLReadVersion
            uint(&[0x42, 0xF2], 4), // EBMLMaxIDLength
            uint(&[0x42, 0xF3], 8), // EBMLMaxSizeLength
            el(&[0x42, 0x82], b"webm"),
            uint(&[0x42, 0x87], 4), // DocTypeVersion
            uint(&[0x42, 0x85], 2), // DocTypeReadVersion
        ]
        .concat();
        let info = [
            uint(&[0x2A, 0xD7, 0xB1], 1_000_000), // TimestampScale: milliseconds
            el(&[0x44, 0x89], &duration.to_be_bytes()), // Duration
            el(&[0x4D, 0x80], b"test"),           // MuxingApp
            el(&[0x57, 0x41], b"test"),           // WritingApp
        ]
        .concat();
        let size = [uint(&[0xB0], 2), uint(&[0xBA], 2)].concat(); // PixelWidth, PixelHeight
        let mut entry = [
            uint(&[0xD7], 1),       // TrackNumber
            uint(&[0x73, 0xC5], 1), // TrackUID
            uint(&[0x83], 1),       // TrackType: video
            el(&[0x86], video.codec.as_bytes()),
            el(&[0xE0], &size),
        ]
        .concat();
        if !video.private.is_empty() {
            entry.extend(el(&[0x63, 0xA2], video.private));
        }
        entry.extend_from_slice(&video.encodings);
        let tracks = [
            entry,
            [
                uint(&[0xD7], 2),
                uint(&[0x73, 0xC5], 2),
                uint(&[0x83], 2), // audio
                el(&[0x86], b"A_OPUS"),
            ]
            .concat(),
        ]
        .map(|t| el(&[0xAE], &t))
        .concat();
        let mut body = uint(&[0xE7], cluster); // Timestamp
        for b in blocks {
            // a SimpleBlock: the track, the timestamp relative to the cluster, the flags
            let [hi, lo] = b.at.to_be_bytes();
            let flags = if b.key { 0x80 } else { 0 };
            body.extend(el(
                &[0xA3],
                &[&[0x80 | b.track, hi, lo, flags][..], &b.data].concat(),
            ));
        }
        let segment = [
            el(&[0x15, 0x49, 0xA9, 0x66], &info),
            el(&[0x16, 0x54, 0xAE, 0x6B], &tracks),
            el(&[0x1F, 0x43, 0xB6, 0x75], &body),
        ]
        .concat();
        let file = [
            el(&[0x1A, 0x45, 0xDF, 0xA3], &header),
            el(&[0x18, 0x53, 0x80, 0x67], &segment),
        ]
        .concat();
        std::fs::write(path, file).unwrap();
    }

    /// A rewind reads the file again from its first block, wherever that is: a seek finds
    /// nothing in a file without Cues whose one cluster starts before or after the time it
    /// is asked for.
    #[test]
    fn a_webm_is_read_again_from_its_start_however_it_begins() {
        let dir = std::env::temp_dir().join(format!("nsc-demux-rewind-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ms = Duration::from_millis;
        let (v, a) = (1, 2);
        // the cluster's timestamp and its blocks; the video always has three 40 ms frames
        let files: [(&str, u64, &[Block]); 5] = [
            // cut with its timestamps kept (-copyts): it starts 600 s in
            ("late", 600_000, &[(v, 0), (v, 40), (v, 80)]),
            // the video starts after the cluster does
            ("offset", 0, &[(v, 40), (v, 80), (v, 120)]),
            // another track's blocks come first
            (
                "audio first",
                0,
                &[(a, 0), (a, 20), (v, 40), (a, 40), (v, 80), (v, 120)],
            ),
            (
                "late, audio first",
                600_000,
                &[(a, 0), (v, 40), (a, 40), (v, 80), (v, 120)],
            ),
            ("late, offset", 600_000, &[(v, 40), (v, 80), (v, 120)]),
        ];
        for (name, cluster, blocks) in files {
            let path = dir.join(format!("{name}.webm"));
            let video: Vec<u64> = blocks
                .iter()
                .filter(|b| b.0 == v)
                .map(|b| cluster + u64::from(b.1))
                .collect();
            let end = video[2] + 40;
            write_webm(&path, cluster, blocks, end);

            let mut d = open(&path).unwrap();
            assert_eq!(d.info().duration, Some(ms(end)), "{name}: where it ends");
            let pass = |d: &mut Box<dyn Demuxer>| {
                std::iter::from_fn(|| d.next_packet().unwrap())
                    .map(|p| (p.pts, p.data))
                    .collect::<Vec<_>>()
            };
            let first = pass(&mut d);
            let pts: Vec<_> = first.iter().map(|p| p.0).collect();
            assert_eq!(
                pts,
                video.iter().map(|&t| ms(t)).collect::<Vec<_>>(),
                "{name}"
            );
            d.rewind().unwrap();
            assert_eq!(
                pass(&mut d),
                first,
                "{name}: read again from its first frame"
            );
            d.rewind().unwrap();
            assert_eq!(pass(&mut d), first, "{name}: and again");

            // and a loop goes on without a pause (its duration is where it ends, not a length)
            use crate::media::video::pipeline::{Next, Pipeline, testing::FakeDecoder};
            d.rewind().unwrap();
            let mut p = Pipeline::new(d, Box::new(FakeDecoder::default()), true);
            let got: Vec<_> = (0..6)
                .map(|_| match p.next_frame() {
                    Ok(Next::Frame(f)) => f.pts - ms(video[0]),
                    other => panic!("{name}: it loops, got {:?}", other.err()),
                })
                .collect();
            assert_eq!(got, [0, 40, 80, 120, 160, 200].map(ms), "{name}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Audio that runs on for two seconds after three frames of video, in a file that
    /// doesn't give its frames durations (SimpleBlocks, no DefaultDuration): the file's
    /// duration is the audio's, and a loop goes on where the frames end, not there.
    #[test]
    fn a_webm_without_frame_durations_loops_where_its_video_ends() {
        let dir = std::env::temp_dir().join(format!("nsc-demux-tracks-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("long audio.webm");
        let (v, a) = (1, 2);
        let mut blocks = vec![(v, 0), (v, 40), (v, 80)];
        blocks.extend((0..=100).map(|i| (a, i * 20)));
        write_webm(&path, 0, &blocks, 2020);

        let ms = Duration::from_millis;
        let d = open(&path).unwrap();
        assert_eq!(d.info().duration, Some(ms(2020)));
        assert!(d.duration_may_be_another_tracks());
        use crate::media::video::pipeline::{Next, Pipeline, testing::FakeDecoder};
        let mut p = Pipeline::new(d, Box::new(FakeDecoder::default()), true);
        let got: Vec<_> = (0..6)
            .map(|_| match p.next_frame() {
                Ok(Next::Frame(f)) => f.pts,
                other => panic!("it loops, got {:?}", other.err()),
            })
            .collect();
        assert_eq!(got, [0, 40, 80, 120, 160, 200].map(ms));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Real clips from `NSC_TEST_MEDIA` (10 s, 640x360). Without the directory the test is skipped.
    #[test]
    fn real_files() {
        let Some(dir) = std::env::var_os("NSC_TEST_MEDIA").map(std::path::PathBuf::from) else {
            eprintln!("NSC_TEST_MEDIA not set - skipping");
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
                eprintln!("{name}: file missing - skipping");
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

    /// A packet as the tests compare it: its time in ms, whether it is a keyframe, its bytes.
    type Got = (u64, bool, Vec<u8>);

    /// A file for a test in the temp directory, its name unique to the test and the run.
    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("nsc-demux-{}-{name}", std::process::id()))
    }

    /// Writes a test file, opens it and reads it through.
    fn demux_bytes(name: &str, bytes: &[u8]) -> PipeResult<(VideoInfo, Vec<Got>)> {
        let path = scratch(name);
        std::fs::write(&path, bytes).unwrap();
        let read = || -> PipeResult<_> {
            let mut d = open(&path)?;
            let info = d.info().clone();
            let mut got = Vec::new();
            while let Some(p) = d.next_packet()? {
                got.push((
                    (p.pts.as_secs_f64() * 1e3).round() as u64,
                    p.keyframe,
                    p.data,
                ));
            }
            Ok((info, got))
        };
        let result = read();
        std::fs::remove_file(&path).ok();
        result
    }

    // MP4 test files, box by box (ISO/IEC 14496-12)

    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        [&(8 + body.len() as u32).to_be_bytes()[..], kind, body].concat()
    }

    /// A box with a version and flags before its body.
    fn full_box(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
        let mut head = flags.to_be_bytes();
        head[0] = version;
        bx(kind, &[&head[..], body].concat())
    }

    fn be32(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_be_bytes()).collect()
    }

    /// A sample table (`stts`, `ctts`, `stss`, `stsc`, `stco`): its entries, each a row.
    fn table(kind: &[u8; 4], rows: &[&[u32]]) -> Vec<u8> {
        let body = [be32(&[rows.len() as u32]), be32(&rows.concat())].concat();
        full_box(kind, 0, 0, &body)
    }

    /// `stsz`: one size for every sample, or (`size` 0) a size each.
    fn stsz(size: u32, count: u32, sizes: &[u32]) -> Vec<u8> {
        full_box(b"stsz", 0, 0, &[be32(&[size, count]), be32(sizes)].concat())
    }

    fn mdhd(version: u8, timescale: u32, duration: u64) -> Vec<u8> {
        let body = if version == 1 {
            [
                &[0; 16][..],
                &timescale.to_be_bytes(),
                &duration.to_be_bytes(),
                &[0; 4],
            ]
            .concat()
        } else {
            [&[0; 8][..], &be32(&[timescale, duration as u32]), &[0; 4]].concat()
        };
        full_box(b"mdhd", version, 0, &body)
    }

    const SPS: [u8; 4] = [0x67, 0x64, 0x00, 0x1f];
    const PPS: [u8; 2] = [0x68, 0xee];

    /// An H.264 sample of one NAL: an IDR slice or not, with a byte to tell it by.
    fn nal(idr: bool, n: u8) -> Vec<u8> {
        vec![0, 0, 0, 2, if idr { 0x65 } else { 0x41 }, n]
    }

    /// The packet a decoder gets for `nal(idr, n)`: Annex-B, with the parameter sets in
    /// front of a keyframe.
    fn annexb(idr: bool, n: u8) -> Vec<u8> {
        let mut v = Vec::new();
        if idr {
            for ps in [&SPS[..], &PPS[..]] {
                v.extend([0, 0, 0, 1]);
                v.extend(ps);
            }
        }
        v.extend([0, 0, 0, 1, if idr { 0x65 } else { 0x41 }, n]);
        v
    }

    fn ftyp() -> Vec<u8> {
        bx(b"ftyp", b"isom\0\0\0\0isomavc1")
    }

    /// `avcC` with the test SPS and PPS and 4-byte NAL lengths.
    fn avcc() -> Vec<u8> {
        [
            &[1, 0x64, 0, 0x1f, 0xff, 0xe1, 0, 4][..],
            &SPS,
            &[1, 0, 2],
            &PPS,
        ]
        .concat()
    }

    /// `trex` default sample flags: not a sync sample, depends on others.
    const NON_SYNC: u32 = 0x0101_0000;
    /// a sync sample that depends on no other
    const SYNC: u32 = 0x0200_0000;

    /// The `moov` of an MP4 with one H.264 track (320x180), these boxes in its `stbl`
    /// after `stsd`, and `mvex` when it is fragmented.
    fn moov(mdhd: &[u8], tables: &[&[u8]], fragmented: bool) -> Vec<u8> {
        moov_of(&[trak(1, true, mdhd, tables)], fragmented)
    }

    /// The `moov` of an MP4 with these `trak`s, numbered from 1, and `mvex` when it is
    /// fragmented.
    fn moov_of(traks: &[Vec<u8>], fragmented: bool) -> Vec<u8> {
        let mut mvhd = [0u8; 96];
        mvhd[8..12].copy_from_slice(&be32(&[1000])); // timescale
        let mut moov = [full_box(b"mvhd", 0, 0, &mvhd), traks.concat()].concat();
        if fragmented {
            let trexs = (1..=traks.len() as u32)
                .map(|id| full_box(b"trex", 0, 0, &be32(&[id, 1, 0, 0, NON_SYNC])))
                .collect::<Vec<_>>();
            moov.extend(bx(b"mvex", &trexs.concat()));
        }
        bx(b"moov", &moov)
    }

    /// A `trak` of a test MP4: track `id`, H.264 (320x180) or else IMA4 sound (a sample
    /// entry re_mp4 knows no kind of), these boxes in its `stbl` after `stsd`.
    fn trak(id: u32, video: bool, mdhd: &[u8], tables: &[&[u8]]) -> Vec<u8> {
        let entry = if video {
            let mut avc1 = vec![0u8; 78];
            avc1[7] = 1; // data_reference_index
            avc1[24..28].copy_from_slice(&[0x01, 0x40, 0x00, 0xb4]);
            avc1.extend(bx(b"avcC", &avcc()));
            bx(b"avc1", &avc1)
        } else {
            let mut ima4 = vec![0u8; 28];
            ima4[7] = 1; // data_reference_index
            ima4[16..18].copy_from_slice(&[0, 1]); // one channel
            bx(b"ima4", &ima4)
        };
        let stsd = full_box(b"stsd", 0, 0, &[be32(&[1]), entry].concat());
        let stbl = bx(b"stbl", &[&stsd[..], &tables.concat()].concat());
        let dinf = bx(b"dinf", &full_box(b"dref", 0, 0, &be32(&[0])));
        let (header, handler) = if video {
            (full_box(b"vmhd", 0, 1, &[0; 8]), b"vide")
        } else {
            (full_box(b"smhd", 0, 0, &[0; 4]), b"soun")
        };
        let minf = bx(b"minf", &[header, dinf, stbl].concat());
        let hdlr = full_box(b"hdlr", 0, 0, &[&[0; 4][..], handler, &[0; 13]].concat());
        let mdia = bx(b"mdia", &[mdhd, &hdlr, &minf].concat());
        let mut tkhd = [0u8; 80];
        tkhd[8..12].copy_from_slice(&be32(&[id])); // track_ID
        if video {
            tkhd[72..80].copy_from_slice(&be32(&[320 << 16, 180 << 16]));
        }
        bx(b"trak", &[full_box(b"tkhd", 0, 3, &tkhd), mdia].concat())
    }

    /// An MP4 indexed by its `moov`: `ftyp`, `mdat` with `MP4_SAMPLES` one after another
    /// from byte 32, then `moov` with other boxes (`extra`) before the tracks, these boxes
    /// of the video track and the `trak` of a sound track after it, then the fragments (a
    /// `moof` and its `mdat`; with them the `moov` has an `mvex`). A test swaps one for a
    /// broken one; an empty one is left out.
    struct Mp4File {
        mdhd: Vec<u8>,
        stts: Vec<u8>,
        ctts: Vec<u8>,
        stss: Vec<u8>,
        stsc: Vec<u8>,
        stsz: Vec<u8>,
        stco: Vec<u8>,
        sound: Vec<u8>,
        extra: Vec<u8>,
        fragments: Vec<u8>,
    }

    /// An IDR, a P and a B picture in decode order, 40 ms each (timescale 12800), each in
    /// a chunk of its own; shown at 0, 80 and 40 ms.
    const MP4_SAMPLES: [(bool, u8); 3] = [(true, 0), (false, 1), (false, 2)];

    impl Default for Mp4File {
        fn default() -> Self {
            Self {
                mdhd: mdhd(0, 12800, 3 * 512),
                stts: table(b"stts", &[&[3, 512]]),
                ctts: table(b"ctts", &[&[1, 512], &[1, 1024], &[1, 0]]),
                stss: table(b"stss", &[&[1]]),
                stsc: table(b"stsc", &[&[1, 1, 1]]),
                stsz: stsz(0, 3, &[6, 6, 6]),
                stco: table(b"stco", &[&[32], &[38], &[44]]),
                sound: Vec::new(),
                extra: Vec::new(),
                fragments: Vec::new(),
            }
        }
    }

    impl Mp4File {
        fn bytes(&self) -> Vec<u8> {
            let mdat = bx(b"mdat", &MP4_SAMPLES.map(|(idr, n)| nal(idr, n)).concat());
            let tables = [
                &self.stts[..],
                &self.ctts,
                &self.stss,
                &self.stsc,
                &self.stsz,
                &self.stco,
            ];
            let video = trak(1, true, &self.mdhd, &tables);
            let boxes = [self.extra.clone(), video, self.sound.clone()];
            let moov = moov_of(&boxes, !self.fragments.is_empty());
            [ftyp(), mdat, moov, self.fragments.clone()].concat()
        }
    }

    /// What `Mp4File::default()` reads as.
    fn mp4_packets() -> Vec<Got> {
        vec![
            (0, true, annexb(true, 0)),
            (80, false, annexb(false, 1)),
            (40, false, annexb(false, 2)),
        ]
    }

    #[test]
    fn an_mp4_reads_as_its_sample_tables_say() {
        let (info, got) = demux_bytes("plain.mp4", &Mp4File::default().bytes()).unwrap();
        assert_eq!(info.codec, Codec::H264);
        assert_eq!((info.width, info.height), (320, 180));
        assert_eq!(info.duration, Some(Duration::from_millis(120)));
        assert_eq!(got, mp4_packets());
    }

    /// Sample tables that don't add up are read as far as they make sense, or are an
    /// error; never a panic, nor gigabytes of samples.
    #[test]
    fn broken_mp4_sample_tables_are_read_or_refused() {
        let plain = Mp4File::default;
        let cases = [
            (
                "a chunk past the end of stco",
                Mp4File {
                    stco: table(b"stco", &[&[32], &[38]]),
                    ..plain()
                },
                Err("more chunks than stco/co64 has"),
            ),
            (
                "no stsc entries",
                Mp4File {
                    stsc: table(b"stsc", &[]),
                    ..plain()
                },
                Err("stsc has no entries"),
            ),
            (
                "stsc chunks numbered from 0",
                Mp4File {
                    stsc: table(b"stsc", &[&[0, 1, 1], &[0, 1, 1]]),
                    ..plain()
                },
                Ok(mp4_packets()),
            ),
            (
                "stts two samples short",
                Mp4File {
                    stts: table(b"stts", &[&[1, 512]]),
                    ..plain()
                },
                Err("stts describes fewer samples than stsz"),
            ),
            (
                // the last sample lasts until the media ends
                "stts one sample short",
                Mp4File {
                    stts: table(b"stts", &[&[2, 512]]),
                    ..plain()
                },
                Ok(mp4_packets()),
            ),
            (
                // past its end the samples have no composition offset
                "ctts short",
                Mp4File {
                    ctts: table(b"ctts", &[&[1, 512]]),
                    ..plain()
                },
                Ok(vec![
                    (0, true, annexb(true, 0)),
                    (0, false, annexb(false, 1)),
                    (40, false, annexb(false, 2)),
                ]),
            ),
            (
                "stss with sample 0",
                Mp4File {
                    stss: table(b"stss", &[&[0], &[1]]),
                    ..plain()
                },
                Ok(mp4_packets()),
            ),
            (
                "mdhd shorter than the samples",
                Mp4File {
                    mdhd: mdhd(0, 12800, 1),
                    ..plain()
                },
                Ok(mp4_packets()),
            ),
            (
                "thousands of samples in a few hundred bytes",
                Mp4File {
                    stts: table(b"stts", &[&[5000, 512]]),
                    ctts: Vec::new(),
                    stsc: table(b"stsc", &[&[1, 5000, 1]]),
                    stsz: stsz(1, 5000, &[]),
                    stco: table(b"stco", &[&[32]]),
                    ..plain()
                },
                Err("more samples than the file has bytes"),
            ),
            (
                // a few bytes of tables for 4 billion samples
                "more samples than the file has bytes",
                Mp4File {
                    stts: table(b"stts", &[&[u32::MAX, 512]]),
                    ctts: Vec::new(),
                    stsc: table(b"stsc", &[&[1, u32::MAX, 1]]),
                    stsz: stsz(1, u32::MAX, &[]),
                    stco: table(b"stco", &[&[32]]),
                    ..plain()
                },
                Err("more samples than the file has bytes"),
            ),
            (
                "neither stco nor co64",
                Mp4File {
                    stco: Vec::new(),
                    ..plain()
                },
                Err("stco and co64 not found"),
            ),
        ];
        for (name, file, want) in cases {
            let got = demux_bytes("broken.mp4", &file.bytes()).map(|(_, packets)| packets);
            match (got, want) {
                (Ok(got), Ok(want)) => assert_eq!(got, want, "{name}"),
                (Err(e), Err(want)) => assert!(e.contains(want), "{name}: {e}"),
                (got, want) => panic!("{name}: got {got:?}, want {want:?}"),
            }
        }
    }

    /// Only the video track's samples are read. QuickTime's legacy sound layout counts one
    /// sample per PCM frame (stts delta 1, stsz size 1), more than the file has bytes: read
    /// with the rest, it gets the file refused, and the video is as good as ever.
    #[test]
    fn an_mp4_sound_track_with_more_samples_than_bytes_leaves_the_video_readable() {
        // 10 s at 48 kHz in one chunk
        let sound = trak(
            2,
            false,
            &mdhd(0, 48000, 480_000),
            &[
                &table(b"stts", &[&[480_000, 1]]),
                &table(b"stsc", &[&[1, 480_000, 1]]),
                &stsz(1, 480_000, &[]),
                &table(b"stco", &[&[32]]),
            ],
        );
        let file = Mp4File {
            sound,
            ..Mp4File::default()
        }
        .bytes();
        let e = re_mp4::Mp4::read_bytes(&file).unwrap_err().to_string();
        assert!(e.contains("more samples than the file has bytes"), "{e}");

        let (info, got) = demux_bytes("legacy-sound.mov", &file).unwrap();
        assert_eq!(info.duration, Some(Duration::from_millis(120)));
        assert_eq!(got, mp4_packets());
    }

    /// Past the file's size, the samples are limited to `Mp4::MAX_SAMPLES` in all: a file of
    /// gigabytes can't make the reader allocate tens of gigabytes for them.
    #[test]
    fn an_mp4_with_more_samples_than_are_read_is_refused() {
        let over = re_mp4::Mp4::MAX_SAMPLES as u32 + 1;
        let file = Mp4File {
            stts: table(b"stts", &[&[over, 512]]),
            ctts: Vec::new(),
            stsc: table(b"stsc", &[&[1, over, 1]]),
            stsz: stsz(1, over, &[]),
            stco: table(b"stco", &[&[32]]),
            ..Mp4File::default()
        };
        let path = scratch("max-samples.mp4");
        std::fs::write(&path, file.bytes()).unwrap();
        // zeros after the moov (where the boxes end), and a byte for every sample
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(2 * u64::from(over))
            .unwrap();
        let e = open(&path).err();
        std::fs::remove_file(&path).ok();
        let e = e.expect("refused");
        assert!(
            e.contains(&format!("more than {} samples", re_mp4::Mp4::MAX_SAMPLES)),
            "{e}"
        );
    }

    /// A track fragment that isn't read still places the next one: the video's, without a
    /// base of its own, starts where the sound's before it end. Those end after runs of
    /// default-sized samples, listed sizes, and 4 billion samples of no bytes, counted
    /// without walking them.
    #[test]
    fn a_video_fragment_starts_where_the_sound_fragments_before_it_end() {
        let empty = [
            table(b"stts", &[]),
            table(b"stsc", &[]),
            stsz(0, 0, &[]),
            table(b"stco", &[]),
        ];
        let empty = empty.each_ref().map(|t| &t[..]);
        let moov = moov_of(
            &[
                trak(1, true, &mdhd(0, 12800, 0), &empty),
                trak(2, false, &mdhd(0, 48000, 0), &empty),
            ],
            true,
        );
        // the sound's 13 bytes, then the video's
        let mdat = [vec![0xaa; 13], nal(true, 0), nal(false, 1)].concat();
        // the first run's data offset is from the start of the moof
        let moof = |data: u32| {
            let sound = [
                [
                    // default-base-is-moof, a default sample duration and size
                    full_box(b"tfhd", 0, 0x02_0018, &be32(&[2, 1, 4])),
                    // a data offset: 3 samples of 4 bytes
                    full_box(b"trun", 0, 0x01, &be32(&[3, data])),
                ]
                .concat(),
                [
                    // no base of its own, a default sample size of 0
                    full_box(b"tfhd", 0, 0x10, &be32(&[2, 0])),
                    // sample sizes: 1 byte
                    full_box(b"trun", 0, 0x0200, &be32(&[1, 1])),
                    // 4 billion samples of the default size
                    full_box(b"trun", 0, 0, &be32(&[u32::MAX])),
                ]
                .concat(),
            ];
            let video = [
                // no base of its own, a default sample duration
                full_box(b"tfhd", 0, 0x08, &be32(&[1, 512])),
                full_box(b"tfdt", 1, 0, &0u64.to_be_bytes()),
                // first sample flags, sample sizes
                full_box(b"trun", 0, 0x0204, &be32(&[2, SYNC, 6, 6])),
            ]
            .concat();
            let trafs = [&sound[0], &sound[1], &video].map(|t| bx(b"traf", t));
            bx(
                b"moof",
                &[full_box(b"mfhd", 0, 0, &be32(&[1])), trafs.concat()].concat(),
            )
        };
        let data = moof(0).len() as u32 + 8;
        let file = [ftyp(), moov, moof(data), bx(b"mdat", &mdat)].concat();
        let (_, got) = demux_bytes("sound-first.mp4", &file).unwrap();
        assert_eq!(
            got,
            vec![(0, true, annexb(true, 0)), (40, false, annexb(false, 1))]
        );
    }

    /// The duration in `mdhd`: all ones is unknown (ISO/IEC 14496-12, 8.4.2), and the
    /// samples tell instead; one too long for a Duration is none.
    #[test]
    fn an_mp4_duration_unknown_or_too_long_is_not_a_panic() {
        // at timescale 1 the samples (512 units each) make 1024 s
        let samples = Some(Duration::from_secs(1024));
        for (mdhd, want) in [
            (mdhd(1, 1, u64::MAX), samples),
            (mdhd(0, 1, u64::from(u32::MAX)), samples),
            (mdhd(1, 1, u64::MAX - 1), None),
            (mdhd(1, 1, 3 * 512), Some(Duration::from_secs(3 * 512))),
        ] {
            let file = Mp4File {
                mdhd,
                ..Mp4File::default()
            };
            let (info, got) = demux_bytes("duration.mp4", &file.bytes()).unwrap();
            assert_eq!(info.duration, want);
            assert_eq!(got.len(), 3);
        }
    }

    /// A run of samples in a fragmented test MP4: its `trun` version, whether its first
    /// sample is flagged a sync sample (the others take `trex`'s flags: not one), the
    /// bytes of another track before it in the `mdat`, and its samples: IDR or not, a
    /// byte to tell them by, the composition offset.
    struct Run {
        version: u8,
        first_is_sync: bool,
        gap: usize,
        samples: Vec<(bool, u8, i32)>,
    }

    /// A fragmented MP4 as ffmpeg writes one: a `moov` without samples, then one `moof`
    /// (based at itself, 40 ms samples from 0) and its `mdat` with the runs.
    fn fragmented(runs: &[Run]) -> Vec<u8> {
        let empty = [
            table(b"stts", &[]),
            table(b"stsc", &[]),
            stsz(0, 0, &[]),
            table(b"stco", &[]),
        ];
        let moov = moov(&mdhd(0, 12800, 0), &empty.each_ref().map(|t| &t[..]), true);
        [ftyp(), moov, fragment(Some(0), runs)].concat()
    }

    /// A `moof` of track 1, based at itself, with the runs (40 ms samples) from the decode
    /// time in its `tfdt` (none: no `tfdt`), and the `mdat` after it.
    fn fragment(tfdt: Option<u64>, runs: &[Run]) -> Vec<u8> {
        let mut mdat = Vec::new();
        let mut starts = Vec::new();
        for run in runs {
            mdat.resize(mdat.len() + run.gap, 0xaa);
            starts.push(mdat.len() as u32);
            for &(idr, n, _) in &run.samples {
                mdat.extend(nal(idr, n));
            }
        }
        // the runs' data offsets are from the start of the moof
        let moof = |data: u32| {
            // default-base-is-moof, a default sample duration
            let mut traf = full_box(b"tfhd", 0, 0x02_0008, &be32(&[1, 512]));
            if let Some(time) = tfdt {
                traf.extend(full_box(b"tfdt", 1, 0, &time.to_be_bytes()));
            }
            for (run, start) in runs.iter().zip(&starts) {
                // data offset, sample sizes, composition offsets, first sample flags
                let mut flags = 0x0a01;
                let mut body = be32(&[run.samples.len() as u32, data + start]);
                if run.first_is_sync {
                    flags |= 0x04;
                    body.extend(be32(&[SYNC]));
                }
                for &(_, _, cts) in &run.samples {
                    body.extend(be32(&[6, cts as u32]));
                }
                traf.extend(full_box(b"trun", run.version, flags, &body));
            }
            bx(
                b"moof",
                &[full_box(b"mfhd", 0, 0, &be32(&[1])), bx(b"traf", &traf)].concat(),
            )
        };
        let data = moof(0).len() as u32 + 8;
        [moof(data), bx(b"mdat", &mdat)].concat()
    }

    /// In a fragment a sample is a sync sample when `sample_is_non_sync_sample` (bit 16 of
    /// its flags, ISO/IEC 14496-12 8.8.3) is clear: the keyframe gets the parameter sets.
    #[test]
    fn fragmented_mp4_keyframes_are_the_samples_flagged_sync() {
        let run = Run {
            version: 0,
            first_is_sync: true,
            gap: 0,
            samples: vec![
                (true, 0, 1024),
                (false, 1, 2048),
                (false, 2, 512),
                (false, 3, 512),
            ],
        };
        let (_, got) = demux_bytes("fragmented.mp4", &fragmented(&[run])).unwrap();
        assert_eq!(
            got,
            vec![
                (80, true, annexb(true, 0)),
                (200, false, annexb(false, 1)),
                (120, false, annexb(false, 2)),
                (160, false, annexb(false, 3)),
            ]
        );
    }

    /// A version 1 `trun` has signed composition offsets (ffmpeg's negative_cts_offsets).
    #[test]
    fn negative_composition_offsets_of_a_version_1_trun() {
        let run = Run {
            version: 1,
            first_is_sync: true,
            gap: 0,
            samples: vec![
                (true, 0, 0),
                (false, 1, 1024),
                (false, 2, -512),
                (false, 3, -512),
            ],
        };
        let (_, got) = demux_bytes("negative-cts.mp4", &fragmented(&[run])).unwrap();
        let pts: Vec<_> = got.iter().map(|g| g.0).collect();
        assert_eq!(pts, [0, 120, 40, 80]);
    }

    /// Each `trun` with a data offset starts there, not where the previous one ended
    /// (ffmpeg's -frag_interleave puts other tracks' samples between them).
    #[test]
    fn every_trun_starts_at_its_own_data_offset() {
        let runs = [
            Run {
                version: 0,
                first_is_sync: true,
                gap: 0,
                samples: vec![(true, 0, 0)],
            },
            Run {
                version: 0,
                first_is_sync: false,
                gap: 5,
                samples: vec![(false, 1, 0), (false, 2, 0)],
            },
        ];
        let (_, got) = demux_bytes("interleaved.mp4", &fragmented(&runs)).unwrap();
        assert_eq!(
            got,
            vec![
                (0, true, annexb(true, 0)),
                (40, false, annexb(false, 1)),
                (80, false, annexb(false, 2)),
            ]
        );
    }

    /// A fragment decode time out of the timestamps' range, or one they overflow from, is
    /// an error, not a wrapped or panicking sum.
    #[test]
    fn fragment_decode_times_out_of_range_are_errors() {
        let run = Run {
            version: 0,
            first_is_sync: true,
            gap: 0,
            samples: vec![(true, 0, 0), (false, 1, 0)],
        };
        let file = fragmented(&[run]);
        // tfdt's decode time: after its type, version and flags
        let at = file.windows(4).position(|w| w == b"tfdt").unwrap() + 8;
        for (time, want) in [
            (u64::MAX, "tfdt decode time out of range"),
            (i64::MAX as u64, "sample timestamps overflow"),
        ] {
            let mut file = file.clone();
            file[at..at + 8].copy_from_slice(&time.to_be_bytes());
            let e = demux_bytes("tfdt.mp4", &file).unwrap_err();
            assert!(e.contains(want), "{e}");
        }
    }

    /// Samples in the `moov` and more in fragments after it (ffmpeg's frag_keyframe without
    /// empty_moov): the fragments' times are moved as the `moov`'s are, by the smallest
    /// composition time (and their decode times by the largest negative composition offset
    /// too), so the pictures go on at 40 ms steps. A fragment without a `tfdt` starts where
    /// the samples in the `moov` end.
    #[test]
    fn fragments_after_samples_in_the_moov_go_on_from_them() {
        // decode times 1536, 2048, 2560 in the media, after the moov's three samples
        let run = |version, offsets: [i32; 3]| Run {
            version,
            first_is_sync: true,
            gap: 0,
            samples: vec![
                (true, 3, offsets[0]),
                (false, 4, offsets[1]),
                (false, 5, offsets[2]),
            ],
        };
        let cases = [
            (
                "smallest composition time 512",
                Mp4File::default(),
                Some(1536),
                run(0, [512, 1024, 0]),
            ),
            (
                "negative composition offsets",
                Mp4File {
                    ctts: table(b"ctts", &[&[1, 0], &[1, 512], &[1, -512i32 as u32]]),
                    ..Mp4File::default()
                },
                Some(1536),
                run(1, [0, 512, -512]),
            ),
            ("no tfdt", Mp4File::default(), None, run(0, [512, 1024, 0])),
        ];
        for (name, moov, tfdt, run) in cases {
            let file = Mp4File {
                fragments: fragment(tfdt, &[run]),
                ..moov
            };
            let (_, got) = demux_bytes("moov-and-moof.mp4", &file.bytes()).unwrap();
            let shown: Vec<_> = got.iter().map(|g| (g.0, g.1)).collect();
            assert_eq!(
                shown,
                [
                    (0, true),
                    (80, false),
                    (40, false),
                    (120, true),
                    (200, false),
                    (160, false)
                ],
                "{name}"
            );
        }
    }

    /// `demux_bytes` on a thread of its own, given 10 s: a reader caught in a loop fails
    /// the test instead of hanging it.
    fn demux_bytes_in_time(name: &str, bytes: Vec<u8>) -> PipeResult<Vec<Got>> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(demux_bytes("in-time.mp4", &bytes).map(|(_, got)| got));
        });
        rx.recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|e| panic!("{name}: no result: {e}"))
    }

    /// The first `kind` box in an MP4 file: its size set, and its type changed.
    fn patch_box(mut file: Vec<u8>, kind: &[u8; 4], size: u32, rename: &[u8; 4]) -> Vec<u8> {
        let at = file.windows(4).position(|w| w == kind).unwrap();
        file[at - 4..at].copy_from_slice(&size.to_be_bytes());
        file[at..at + 4].copy_from_slice(rename);
        file
    }

    /// Box sizes too small for a box header, or larger than the box they are in, and box
    /// fields that run past the end of their box are errors: never a loop reading the same
    /// box forever (a box inside another of size 0 seeks back to its own header), an
    /// allocation of the size a broken header claims, or a panic. A QuickTime user data
    /// list may end in a 32-bit 0, which is no box.
    #[test]
    fn broken_mp4_boxes_are_errors_not_hangs() {
        let plain = Mp4File::default;
        let empty = [
            table(b"stts", &[]),
            table(b"stsc", &[]),
            stsz(0, 0, &[]),
            table(b"stco", &[]),
        ];
        let empty = empty.each_ref().map(|t| &t[..]);
        let mdir = full_box(b"hdlr", 0, 0, &[&[0; 4][..], b"mdir", &[0; 13]].concat());
        let other = full_box(b"hdlr", 0, 0, &[&[0; 4][..], b"abcd", &[0; 13]].concat());
        // `avcC` content: its 8-byte header, 8 bytes of fields, the SPS, the PPS count, then
        // the PPS length
        let pps_len = 8 + 8 + SPS.len() + 1;
        let emsg = |file: Vec<u8>| {
            // a scheme without its terminating 0 in the box: the strings and fields go on
            // into the moov after it
            let at = file.windows(4).position(|w| w == b"moov").unwrap() - 4;
            [&file[..at], &full_box(b"emsg", 0, 0, b"abcd"), &file[at..]].concat()
        };
        let cases = [
            (
                "a udta of size 0",
                Mp4File {
                    extra: [&[0; 4][..], b"udta", &bx(b"free", &[])].concat(),
                    ..plain()
                }
                .bytes(),
                Err("a box inside another box has a size of 0"),
            ),
            (
                "a tkhd of size 0 in the sound track, which isn't read",
                Mp4File {
                    sound: patch_box(
                        trak(2, false, &mdhd(0, 48000, 0), &empty),
                        b"tkhd",
                        0,
                        b"tkhd",
                    ),
                    ..plain()
                }
                .bytes(),
                Err("a box inside another box has a size of 0"),
            ),
            (
                "a box of size 0 before avcC",
                patch_box(plain().bytes(), b"avcC", 0, b"free"),
                Err("a box inside another box has a size of 0"),
            ),
            (
                "a box of 64-bit size 0",
                Mp4File {
                    extra: [&be32(&[1])[..], b"free", &0u64.to_be_bytes()].concat(),
                    ..plain()
                }
                .bytes(),
                Err("a box inside another box has a size of 0"),
            ),
            (
                "a box of size 4",
                Mp4File {
                    extra: [&be32(&[4])[..], b"free"].concat(),
                    ..plain()
                }
                .bytes(),
                Err("box size smaller than its header"),
            ),
            (
                "a box in meta of 1 TB",
                Mp4File {
                    extra: full_box(
                        b"meta",
                        0,
                        0,
                        &[
                            &other[..],
                            &be32(&[1]),
                            b"free",
                            &(1u64 << 40).to_be_bytes(),
                        ]
                        .concat(),
                    ),
                    ..plain()
                }
                .bytes(),
                Err("meta box contains a box with a larger size than it"),
            ),
            (
                "a udta ending in a 32-bit 0",
                Mp4File {
                    extra: bx(b"udta", &[0; 4]),
                    ..plain()
                }
                .bytes(),
                Ok(mp4_packets()),
            ),
            (
                "a PPS past the end of avcC",
                {
                    let mut file = plain().bytes();
                    let at = file.windows(4).position(|w| w == b"avcC").unwrap() - 4 + pps_len;
                    file[at..at + 2].copy_from_slice(&[0, 32]);
                    file
                },
                Err("avcC parameter sets run past its end"),
            ),
            (
                "a metadata data box shorter than its fields",
                Mp4File {
                    extra: bx(
                        b"udta",
                        &full_box(
                            b"meta",
                            0,
                            0,
                            &[
                                &mdir[..],
                                &bx(b"ilst", &bx(b"\xa9nam", &bx(b"data", &be32(&[1])))),
                            ]
                            .concat(),
                        ),
                    ),
                    ..plain()
                }
                .bytes(),
                Err("data box too small"),
            ),
            (
                "an emsg whose fields run past its end",
                emsg(plain().bytes()),
                Err("emsg fields run past its end"),
            ),
        ];
        for (name, file, want) in cases {
            match (demux_bytes_in_time(name, file), want) {
                (Ok(got), Ok(want)) => assert_eq!(got, want, "{name}"),
                (Err(e), Err(want)) => assert!(e.contains(want), "{name}: {e}"),
                (got, want) => panic!("{name}: got {got:?}, want {want:?}"),
            }
        }
    }

    /// A Duration in a Matroska Info that is infinite or too long for a Duration is none.
    #[test]
    fn a_matroska_duration_too_long_is_none() {
        let vp9 = VideoTrack {
            codec: "V_VP9",
            private: &[],
            encodings: Vec::new(),
        };
        let block = MkvBlock {
            track: 1,
            at: 0,
            key: true,
            data: vec![7],
        };
        let path = scratch("duration.webm");
        for (duration, want) in [
            (f64::INFINITY, None),
            (1e300, None),
            (40.0, Some(Duration::from_millis(40))),
        ] {
            write_mkv(&path, &vp9, duration, 0, std::slice::from_ref(&block));
            let mut d = open(&path).unwrap();
            assert_eq!(d.info().duration, want, "{duration}");
            assert_eq!(d.next_packet().unwrap().unwrap().data, [7]);
        }
        std::fs::remove_file(&path).ok();
    }

    /// ContentEncodings with one ContentEncoding (RFC 9559, 5.1.4.1.31): its scope, its
    /// type (0 compression, 1 encryption) and the algorithm; for compression, its settings.
    fn content_encodings(scope: u64, kind: u64, algo: u64, settings: &[u8]) -> Vec<u8> {
        let how = if kind == 0 {
            let compression = [uint(&[0x42, 0x54], algo), el(&[0x42, 0x55], settings)];
            el(&[0x50, 0x34], &compression.concat())
        } else {
            el(&[0x50, 0x35], &uint(&[0x47, 0xE1], algo))
        };
        let encoding = [
            uint(&[0x50, 0x31], 0), // ContentEncodingOrder
            uint(&[0x50, 0x32], scope),
            uint(&[0x50, 0x33], kind),
            how,
        ];
        el(&[0x6D, 0x80], &el(&[0x62, 0x40], &encoding.concat()))
    }

    /// Header stripping (mkvmerge's header removal) takes the same bytes off the front of
    /// every frame, or of CodecPrivate: they are put back before the frames are framed.
    #[test]
    fn matroska_header_stripping_is_undone() {
        let blocks = |strip: usize| {
            [(true, 0), (false, 1)].map(|(idr, n)| MkvBlock {
                track: 1,
                at: u16::from(n) * 40,
                key: idr,
                data: nal(idr, n)[strip..].to_vec(),
            })
        };
        let path = scratch("stripped.mkv");
        let avcc = avcc();
        // the frames' NAL lengths start with two zero bytes, avcC with its version, 1
        for (scope, settings, private, strip) in
            [(1, &[0, 0][..], &avcc[..], 2), (2, &[1][..], &avcc[1..], 0)]
        {
            let track = VideoTrack {
                codec: "V_MPEG4/ISO/AVC",
                private,
                encodings: content_encodings(scope, 0, 3, settings),
            };
            write_mkv(&path, &track, 80.0, 0, &blocks(strip));
            let mut d = open(&path).unwrap_or_else(|e| panic!("scope {scope}: {e}"));
            let got: Vec<_> = std::iter::from_fn(|| d.next_packet().unwrap())
                .map(|p| (p.keyframe, p.data))
                .collect();
            assert_eq!(
                got,
                [(true, annexb(true, 0)), (false, annexb(false, 1))],
                "scope {scope}"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    /// Frames compressed with a codec or encrypted can't be decoded as they are: refused,
    /// and said why.
    #[test]
    fn matroska_compressed_or_encrypted_frames_are_refused() {
        let path = scratch("encoded.webm");
        let block = MkvBlock {
            track: 1,
            at: 0,
            key: true,
            data: vec![7],
        };
        for (encodings, want) in [
            (content_encodings(1, 0, 0, &[]), Some("compressed (Zlib)")),
            (content_encodings(1, 0, 2, &[]), Some("compressed (Lzo1x)")),
            (content_encodings(1, 1, 5, &[]), Some("encrypted")),
            // not encrypted, and an encoding of something else than the frames
            (content_encodings(1, 1, 0, &[]), None),
            (content_encodings(4, 0, 0, &[]), None),
        ] {
            let track = VideoTrack {
                codec: "V_VP9",
                private: &[],
                encodings,
            };
            write_mkv(&path, &track, 40.0, 0, std::slice::from_ref(&block));
            match (open(&path), want) {
                (Err(e), Some(want)) => assert!(e.contains(want), "{e}"),
                (Ok(mut d), None) => assert_eq!(d.next_packet().unwrap().unwrap().data, [7]),
                (Ok(_), Some(want)) => panic!("opened, want {want}"),
                (Err(e), None) => panic!("{e}"),
            }
        }
        std::fs::remove_file(&path).ok();
    }

    /// WebMs as ffmpeg writes them (CPU encoders; skipped without them): a second of video
    /// with three of audio, a GIF whose last frame stays up for two seconds, and a single
    /// picture with three seconds of audio. The file's duration is the longer track's;
    /// where the video ends comes from its frames' own durations, and a loop goes on from
    /// there (a single picture's once per file).
    ///
    /// What ffmpeg writes depends on its version (its GIF reader, its encoders, its
    /// muxer): ffprobe, another reader of the same file, tells whether a file is the one
    /// a case is about, and one that isn't is skipped rather than failed.
    #[test]
    fn where_the_video_of_a_webm_ends() {
        let dir = std::env::temp_dir().join(format!("nsc-demux-end-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ffmpeg = |args: &[&str], out: &Path| {
            std::process::Command::new("ffmpeg")
                .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y"])
                .args(args)
                .arg(out)
                .status()
                .is_ok_and(|s| s.success())
        };
        let ms = Duration::from_millis;
        // the video's packets as ffprobe reads them, and where they end by their durations
        // if every one has one
        let probe = |path: &Path| -> Option<(u32, Option<Duration>)> {
            let out = std::process::Command::new("ffprobe")
                .args(["-v", "error", "-select_streams", "v:0"])
                .args([
                    "-show_entries",
                    "packet=pts_time,duration_time",
                    "-of",
                    "csv=p=0",
                ])
                .arg(path)
                .output()
                .ok()
                .filter(|o| o.status.success())?;
            let time = |s: &str| {
                let secs = s.trim().parse::<f64>().ok().filter(|s| *s >= 0.0)?;
                Some(ms((secs * 1000.0).round() as u64))
            };
            let mut packets = 0;
            let mut end = Some(Duration::ZERO);
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let Some((pts, duration)) = line.split_once(',') else {
                    continue;
                };
                packets += 1;
                end = end
                    .zip(time(pts).zip(time(duration)))
                    .map(|(e, (p, d))| e.max(p + d));
            }
            Some((packets, end))
        };
        let gif = dir.join("pause.gif");
        {
            let mut f = File::create(&gif).unwrap();
            let palette = [255, 0, 0, 0, 255, 0, 0, 0, 255];
            let mut enc = gif::Encoder::new(&mut f, 16, 16, &palette).unwrap();
            for (i, delay) in [10u16, 10, 200].into_iter().enumerate() {
                let frame = gif::Frame {
                    width: 16,
                    height: 16,
                    delay,
                    buffer: std::borrow::Cow::Owned(vec![i as u8; 256]),
                    ..gif::Frame::default()
                };
                enc.write_frame(&frame).unwrap();
            }
        }
        let (av, pause, still) = (
            dir.join("av.webm"),
            dir.join("pause.webm"),
            dir.join("still.webm"),
        );
        let vp9 = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "50k"];
        let opus = ["-c:a", "libopus"];
        let audio = "sine=duration=3";
        let with_audio = |video: &str, out: &Path| {
            let inputs = ["-f", "lavfi", "-i", video, "-f", "lavfi", "-i", audio];
            ffmpeg(&[inputs.as_slice(), &vp9, &opus].concat(), out)
        };
        let made = with_audio("testsrc=size=64x64:rate=10:duration=1", &av)
            && ffmpeg(
                &[["-i", gif.to_str().unwrap()].as_slice(), &vp9].concat(),
                &pause,
            )
            && with_audio("color=red:size=64x64:rate=25:duration=0.04", &still);
        if !made {
            eprintln!("no ffmpeg with libvpx-vp9 and libopus, skipping");
            std::fs::remove_dir_all(&dir).ok();
            return;
        }
        // (the file, its video's frames, where they end, and whether that end is where a
        // frame rate would put it: a reader may make a duration up from the frame rate
        // where the file has none, the GIF's pause it can't)
        let cases = [
            (&av, 10, ms(1000), true),
            (&pause, 3, ms(2200), false),
            (&still, 1, ms(40), true),
        ];
        for (path, frames, end, by_rate) in cases {
            let name = path.file_name().unwrap().to_string_lossy();
            match probe(path) {
                Some(seen) if seen == (frames, Some(end)) => {}
                seen => {
                    eprintln!(
                        "{name}: ffprobe reads {seen:?}, not {frames} frames that end at \
                         {end:?}: skipping it"
                    );
                    continue;
                }
            }
            let mut d = open(path).unwrap();
            assert_eq!(d.frames_end(), None, "{name}: nothing read yet");
            let mut n = 0;
            while d.next_packet().unwrap().is_some() {
                n += 1;
            }
            assert_eq!(n, frames, "{name}");
            if by_rate && d.frames_end().is_none() {
                eprintln!("{name}: the frames have no durations in the file, skipping it");
                continue;
            }
            assert_eq!(d.frames_end(), Some(end), "{name}");
            let file = d.info().duration.expect("duration");
            if path != &pause {
                assert!(file > ms(2900), "{name}: the file's duration is {file:?}");
            }

            use crate::media::video::pipeline::{Next, Pipeline, testing::FakeDecoder};
            d.rewind().unwrap();
            assert_eq!(d.frames_end(), None, "{name}: read again from the start");
            let mut p = Pipeline::new(d, Box::new(FakeDecoder::default()), true);
            let next_pass = (0..=frames)
                .map(|_| match p.next_frame() {
                    Ok(Next::Frame(f)) => f.pts,
                    other => panic!("{name}: it loops, got {:?}", other.err()),
                })
                .last();
            let want = if frames == 1 { file } else { end };
            assert_eq!(next_pass, Some(want), "{name}: the next pass starts there");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
