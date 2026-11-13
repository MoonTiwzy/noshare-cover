//! Container → packets → decoder → frames. Concrete demuxers and decoders
//! plug in via traits; looping and seeking live here, once.

use std::time::Duration;

use crate::frame::FrameData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
    Vp8,
    Vp9,
    Av1,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub duration: Option<Duration>,
    pub codec: Codec,
}

#[derive(Debug, Clone)]
pub struct Packet {
    pub pts: Duration,
    pub keyframe: bool,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct DecodedFrame {
    pub pts: Duration,
    pub data: FrameData,
}

pub type PipeResult<T> = Result<T, String>;

pub trait Demuxer: Send {
    fn info(&self) -> &VideoInfo;
    /// `None` means end of file.
    fn next_packet(&mut self) -> PipeResult<Option<Packet>>;
    /// Rewind to the start (for looping).
    fn rewind(&mut self) -> PipeResult<()>;
}

pub trait Decoder: Send {
    fn name(&self) -> &'static str;
    /// Feed a packet and collect ready frames (zero or more).
    fn decode(&mut self, packet: &Packet) -> PipeResult<Vec<DecodedFrame>>;
    /// End of stream: return everything still buffered.
    fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>>;
    /// Reset state after a seek.
    fn reset(&mut self);
    /// The size the frames are drawn at, in pixels (0 = unknown). A decoder that hands
    /// out pixels may then make its frames smaller than the video.
    fn set_target(&mut self, _width: u32, _height: u32) {}
    /// Nobody is watching: release what is only kept between frames.
    fn idle(&mut self) {}
    /// Frames in video memory can't be shown: hand out pixels from now on.
    fn refuse_video_memory(&mut self) {}
}

/// Demuxer + decoder + queue of ready frames.
pub struct Pipeline {
    demux: Box<dyn Demuxer>,
    decoder: Box<dyn Decoder>,
    ready: std::collections::VecDeque<DecodedFrame>,
    flushed: bool,
    looped: bool,
    /// pts offset added after each loop pass so time stays monotonic.
    loop_offset: Duration,
    /// This pass: packets read, the last frame's pts, the gap between the last two
    /// frames, and where the pass ends by its frames (the last pts plus that gap).
    packets: u32,
    last_pts: Option<Duration>,
    gap: Duration,
    seen_end: Duration,
    /// A decoder to go on with if this one turns out not to take the stream (see
    /// `with_spare`), until the first frame.
    spare: Option<Spare>,
    /// a frame came out at least once
    produced: bool,
    /// the size the decoder was last told
    target: (u32, u32),
}

/// Opens another decoder for the codec.
pub type Spare = fn(&Codec) -> PipeResult<Box<dyn Decoder>>;

/// A pass as long as nothing: one frame, no duration in the container.
const SHORTEST_PASS: Duration = Duration::from_millis(40);

pub enum Next {
    Frame(DecodedFrame),
    /// Video ended and looping is off.
    End,
}

impl Pipeline {
    pub fn new(demux: Box<dyn Demuxer>, decoder: Box<dyn Decoder>, looped: bool) -> Self {
        Self {
            demux,
            decoder,
            ready: Default::default(),
            flushed: false,
            looped,
            loop_offset: Duration::ZERO,
            packets: 0,
            last_pts: None,
            gap: Duration::ZERO,
            seen_end: Duration::ZERO,
            spare: None,
            produced: false,
            target: (0, 0),
        }
    }

    /// A GPU decoder opens for a codec, not for a stream: that this one is 10-bit or
    /// 4:4:4 and the GPU can't do it only shows at the first packets. With a spare
    /// (the CPU decoder, for `backend = auto`) the file is then started over on it.
    pub fn with_spare(mut self, spare: Spare) -> Self {
        self.spare = Some(spare);
        self
    }

    /// The decoder failed with `reason`: start over on the spare one, if there is one
    /// and nothing has been shown yet.
    fn fall_back(&mut self, reason: String) -> PipeResult<()> {
        let spare = match self.spare.take() {
            Some(s) if !self.produced => s,
            _ => return Err(reason),
        };
        let codec = self.demux.info().codec.clone();
        self.decoder = spare(&codec).map_err(|e| format!("{reason}; CPU: {e}"))?;
        self.decoder.set_target(self.target.0, self.target.1);
        self.demux.rewind()?;
        self.ready.clear();
        self.flushed = false;
        (self.packets, self.last_pts, self.seen_end) = (0, None, Duration::ZERO);
        Ok(())
    }

    pub fn info(&self) -> &VideoInfo {
        self.demux.info()
    }

    pub fn decoder_name(&self) -> &'static str {
        self.decoder.name()
    }

    pub fn set_target(&mut self, width: u32, height: u32) {
        self.target = (width, height);
        self.decoder.set_target(width, height);
    }

    pub fn idle(&mut self) {
        self.decoder.idle();
    }

    pub fn refuse_video_memory(&mut self) {
        self.decoder.refuse_video_memory();
        // what is already decoded that way is of no use either
        self.ready.retain(|f| matches!(f.data, FrameData::Cpu(_)));
    }

    pub fn next_frame(&mut self) -> PipeResult<Next> {
        loop {
            match self.next_frame_once() {
                Err(e) => self.fall_back(e)?,
                next => return next,
            }
        }
    }

    fn next_frame_once(&mut self) -> PipeResult<Next> {
        // guard: a broken file must not spin the thread forever
        for _ in 0..4096 {
            if let Some(mut f) = self.ready.pop_front() {
                self.produced = true;
                if let Some(gap) = self.last_pts.and_then(|p| f.pts.checked_sub(p))
                    && !gap.is_zero()
                {
                    self.gap = gap;
                }
                self.last_pts = Some(f.pts);
                self.seen_end = self.seen_end.max(f.pts + self.gap);
                f.pts += self.loop_offset;
                return Ok(Next::Frame(f));
            }
            if self.flushed {
                if !self.looped {
                    return Ok(Next::End);
                }
                // The container's duration may be missing (a WebM written to a pipe) or
                // shorter than the frames reach: the next pass must start after the last
                // frame of this one, or its frames are all in the past and get dropped.
                let length = self.demux.info().duration.unwrap_or(Duration::ZERO);
                self.loop_offset += length.max(self.seen_end).max(SHORTEST_PASS);
                (self.packets, self.last_pts, self.seen_end) = (0, None, Duration::ZERO);
                self.demux.rewind()?;
                self.decoder.reset();
                self.flushed = false;
                continue;
            }
            let packet = match self.demux.next_packet() {
                // a file cut short (a recording that crashed) ends where it can't be read
                // any further, as long as it had something before that
                Err(_) if self.packets > 0 => None,
                other => other?,
            };
            match packet {
                Some(p) => {
                    self.packets += 1;
                    self.ready.extend(self.decoder.decode(&p)?);
                }
                None => {
                    self.ready.extend(self.decoder.flush()?);
                    self.flushed = true;
                }
            }
        }
        Err(format!("{} produces no frames", self.decoder.name()))
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! Synthetic demuxer and decoder: N frames spaced by `step`, each a 1x1 BGRA
    //! holding the frame number. Enough to test looping, the clock and the thread.
    use super::*;
    use crate::frame::CpuFrame;

    pub struct FakeDemux {
        info: VideoInfo,
        pub frames: u32,
        pub step: Duration,
        pos: u32,
    }

    impl FakeDemux {
        pub fn new(frames: u32, step: Duration) -> Self {
            let info = VideoInfo {
                width: 1,
                height: 1,
                duration: Some(step * frames),
                codec: Codec::Av1,
            };
            Self {
                info,
                frames,
                step,
                pos: 0,
            }
        }
    }

    impl Demuxer for FakeDemux {
        fn info(&self) -> &VideoInfo {
            &self.info
        }
        fn next_packet(&mut self) -> PipeResult<Option<Packet>> {
            if self.pos >= self.frames {
                return Ok(None);
            }
            let p = Packet {
                pts: self.step * self.pos,
                keyframe: self.pos == 0,
                data: vec![self.pos as u8],
            };
            self.pos += 1;
            Ok(Some(p))
        }
        fn rewind(&mut self) -> PipeResult<()> {
            self.pos = 0;
            Ok(())
        }
    }

    /// Holds one frame back (like real decoders with latency) to exercise flush.
    /// A frame's alpha is the target width it was decoded for (255 without one).
    #[derive(Default)]
    pub struct FakeDecoder {
        held: Option<Packet>,
        target: Option<u8>,
        /// told to hand out pixels only: from then on the frames' red is 0
        refused: bool,
    }

    fn out(p: Packet, target: Option<u8>, refused: bool) -> DecodedFrame {
        let px = p.data[0];
        let a = target.unwrap_or(255);
        let r = if refused { 0 } else { px };
        DecodedFrame {
            pts: p.pts,
            data: FrameData::Cpu(CpuFrame::from_bgra(1, 1, vec![px, px, r, a].into())),
        }
    }

    impl Decoder for FakeDecoder {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn decode(&mut self, packet: &Packet) -> PipeResult<Vec<DecodedFrame>> {
            let (target, refused) = (self.target, self.refused);
            Ok(self
                .held
                .replace(packet.clone())
                .map(|p| out(p, target, refused))
                .into_iter()
                .collect())
        }
        fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
            let (target, refused) = (self.target, self.refused);
            Ok(self
                .held
                .take()
                .map(|p| out(p, target, refused))
                .into_iter()
                .collect())
        }
        fn reset(&mut self) {
            self.held = None;
        }
        fn set_target(&mut self, width: u32, _height: u32) {
            self.target = u8::try_from(width).ok().filter(|w| *w > 0);
        }
        fn refuse_video_memory(&mut self) {
            self.refused = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    fn pts(n: Next) -> Option<Duration> {
        match n {
            Next::Frame(f) => Some(f.pts),
            Next::End => None,
        }
    }

    /// A demuxer without a duration whose file can't be read past `good` packets.
    struct Cut {
        inner: FakeDemux,
        info: VideoInfo,
        good: u32,
        read: u32,
    }

    impl Cut {
        fn new(frames: u32, good: u32) -> Self {
            let inner = FakeDemux::new(frames, Duration::from_millis(40));
            let info = VideoInfo {
                duration: None,
                ..inner.info().clone()
            };
            Self {
                inner,
                info,
                good,
                read: 0,
            }
        }
    }

    impl Demuxer for Cut {
        fn info(&self) -> &VideoInfo {
            &self.info
        }
        fn next_packet(&mut self) -> PipeResult<Option<Packet>> {
            if self.read >= self.good {
                return Err("read past the end".into());
            }
            self.read += 1;
            self.inner.next_packet()
        }
        fn rewind(&mut self) -> PipeResult<()> {
            self.read = 0;
            self.inner.rewind()
        }
    }

    #[test]
    fn loops_by_its_frames_when_the_container_has_no_duration() {
        let ms = Duration::from_millis;
        let mut p = Pipeline::new(
            Box::new(Cut::new(3, 100)),
            Box::new(FakeDecoder::default()),
            true,
        );
        let got: Vec<_> = (0..7)
            .map(|_| pts(p.next_frame().unwrap()).unwrap())
            .collect();
        let want = [0, 40, 80, 120, 160, 200, 240].map(ms);
        assert_eq!(got, want, "each pass starts a frame after the last one");
    }

    #[test]
    fn a_file_cut_short_ends_where_it_stops() {
        let ms = Duration::from_millis;
        let mut p = Pipeline::new(
            Box::new(Cut::new(10, 2)),
            Box::new(FakeDecoder::default()),
            true,
        );
        let got: Vec<_> = (0..4)
            .map(|_| pts(p.next_frame().unwrap()).unwrap())
            .collect();
        assert_eq!(got, [0, 40, 80, 120].map(ms), "two frames, then again");

        // nothing readable at all is still an error
        let mut p = Pipeline::new(
            Box::new(Cut::new(10, 0)),
            Box::new(FakeDecoder::default()),
            true,
        );
        assert!(p.next_frame().is_err());
    }

    /// A decoder that opened fine and then can't take the stream.
    struct Refuses;
    impl Decoder for Refuses {
        fn name(&self) -> &'static str {
            "refuses"
        }
        fn decode(&mut self, _: &Packet) -> PipeResult<Vec<DecodedFrame>> {
            Err("10-bit".into())
        }
        fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
            Ok(Vec::new())
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn starts_over_on_the_spare_decoder_when_the_first_one_fails() {
        let demux = || Box::new(FakeDemux::new(3, Duration::from_millis(40)));
        let mut p = Pipeline::new(demux(), Box::new(Refuses), false)
            .with_spare(|_| Ok(Box::new(FakeDecoder::default())));
        p.set_target(7, 7);
        match p.next_frame().unwrap() {
            Next::Frame(f) => {
                assert_eq!(f.pts, Duration::ZERO, "from the first frame");
                let FrameData::Cpu(cpu) = f.data else {
                    panic!("pixels expected")
                };
                assert_eq!(cpu.pixels[3], 7, "the spare knows the size too");
            }
            Next::End => panic!("a frame expected"),
        }
        assert_eq!(p.decoder_name(), "fake");

        // without a spare the error is the decoder's
        let mut p = Pipeline::new(demux(), Box::new(Refuses), false);
        assert_eq!(p.next_frame().err().as_deref(), Some("10-bit"));
        // and when the spare can't be opened either, both reasons are given
        let mut p = Pipeline::new(demux(), Box::new(Refuses), false)
            .with_spare(|_| Err("no libvpx".into()));
        assert_eq!(
            p.next_frame().err().as_deref(),
            Some("10-bit; CPU: no libvpx")
        );
    }

    #[test]
    fn plays_all_frames_then_ends() {
        let step = Duration::from_millis(40);
        let mut p = Pipeline::new(
            Box::new(FakeDemux::new(3, step)),
            Box::new(FakeDecoder::default()),
            false,
        );
        let got: Vec<_> = std::iter::from_fn(|| pts(p.next_frame().unwrap())).collect();
        assert_eq!(got, vec![Duration::ZERO, step, step * 2]);
    }

    #[test]
    fn loop_keeps_time_monotonic() {
        let step = Duration::from_millis(40);
        let mut p = Pipeline::new(
            Box::new(FakeDemux::new(2, step)),
            Box::new(FakeDecoder::default()),
            true,
        );
        let got: Vec<_> = (0..5)
            .map(|_| pts(p.next_frame().unwrap()).unwrap())
            .collect();
        assert_eq!(
            got,
            vec![Duration::ZERO, step, step * 2, step * 3, step * 4]
        );
    }
}
