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
    /// Where the video's frames read since the last rewind end by the durations the
    /// container gives them, the last one's included; `None` if it gives none. Unlike
    /// `VideoInfo::duration`, which can be the whole file's (a Matroska file's audio may
    /// run on after the video), this is the video track's alone.
    fn frames_end(&self) -> Option<Duration> {
        None
    }
    /// `VideoInfo::duration` may be where another track ends: the container gives the
    /// whole file's, and the file has more than the video (a Matroska file's audio may run
    /// on after the video).
    fn duration_may_be_another_tracks(&self) -> bool {
        false
    }
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
    /// This pass: packets read, frames out, the first and the last frame's pts, the gap
    /// between the last two frames, and where the pass ends by its frames (the last pts
    /// plus that gap).
    packets: u32,
    frames: u32,
    first_pts: Option<Duration>,
    last_pts: Option<Duration>,
    gap: Duration,
    seen_end: Duration,
    /// This pass ended where the file couldn't be read any further, for this reason.
    cut: Option<String>,
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
            frames: 0,
            first_pts: None,
            last_pts: None,
            gap: Duration::ZERO,
            seen_end: Duration::ZERO,
            cut: None,
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
        self.new_pass();
        Ok(())
    }

    /// The file is read from its start again.
    fn new_pass(&mut self) {
        (self.packets, self.frames) = (0, 0);
        (self.first_pts, self.last_pts) = (None, None);
        self.seen_end = Duration::ZERO;
        self.cut = None;
    }

    pub fn info(&self) -> &VideoInfo {
        self.demux.info()
    }

    pub fn looped(&self) -> bool {
        self.looped
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
                self.frames = self.frames.saturating_add(1);
                if let Some(gap) = self.last_pts.and_then(|p| f.pts.checked_sub(p))
                    && !gap.is_zero()
                {
                    self.gap = gap;
                }
                self.first_pts = Some(self.first_pts.map_or(f.pts, |p| p.min(f.pts)));
                self.last_pts = Some(f.pts);
                // timestamps come from the file: adding them up must not panic on any
                self.seen_end = self.seen_end.max(f.pts.saturating_add(self.gap));
                f.pts = f.pts.saturating_add(self.loop_offset);
                return Ok(Next::Frame(f));
            }
            if self.flushed {
                // Not a single frame out of what could be read: what stopped the reading is
                // the error, the next pass would only stop at the same place.
                if self.first_pts.is_none()
                    && let Some(reason) = self.cut.take()
                {
                    return Err(reason);
                }
                if !self.looped {
                    return Ok(Next::End);
                }
                // The container's duration may be missing (a WebM written to a pipe) or
                // shorter than the frames reach: the next pass must start after the last
                // frame of this one, or its frames are all in the past and get dropped.
                // Both are where the pass ends, not how long it is: a file cut with its
                // timestamps kept starts minutes in, and ffmpeg writes a Matroska file's
                // duration as its last timestamp (600 s..610 s says 610 s). It may also
                // reach past the video: the end of a Matroska file's longer audio, of a file
                // cut short. The video would stand still there on every pass, so where the
                // video's frames end by their own durations wins, if the container has them
                // (they also keep a last frame meant to stay up longer, a GIF's final pause).
                // Without them the frames alone tell where the pass ends when the duration
                // may be another track's or the file was cut short.
                // A pass of a single frame is a still: the next one shows the same picture,
                // so how long it is shows nowhere, only what it costs (the whole file read
                // again, the frame decoded and handed out again). It lasts as long as the
                // file says then, whatever track that is of, the way a picture with a song
                // under it plays once per song.
                let file = self.demux.info().duration;
                let frames_end = self.demux.frames_end();
                let end = if self.frames == 1 {
                    frames_end.max(file)
                } else {
                    frames_end.or(file.filter(|_| {
                        self.cut.is_none() && !self.demux.duration_may_be_another_tracks()
                    }))
                }
                .unwrap_or(Duration::ZERO);
                let length = end
                    .max(self.seen_end)
                    .saturating_sub(self.first_pts.unwrap_or_default());
                self.loop_offset = self.loop_offset.saturating_add(length.max(SHORTEST_PASS));
                self.new_pass();
                self.demux.rewind()?;
                self.decoder.reset();
                self.flushed = false;
                continue;
            }
            let packet = match self.demux.next_packet() {
                // a file cut short (a recording that crashed) ends where it can't be read
                // any further, as long as it had something before that
                Err(e) if self.packets > 0 => {
                    self.cut = Some(e);
                    None
                }
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

    /// The video of a file that may have more tracks: `frames` spaced by `step`, the
    /// container's `duration` (the file's), where the frames `end` by their own durations
    /// if the container gives them, and whether there are `others`.
    pub struct Tracks {
        inner: FakeDemux,
        info: VideoInfo,
        end: Option<Duration>,
        others: bool,
    }

    impl Tracks {
        pub fn new(
            frames: u32,
            step: Duration,
            duration: Duration,
            end: Option<Duration>,
            others: bool,
        ) -> Self {
            let inner = FakeDemux::new(frames, step);
            let info = VideoInfo {
                duration: Some(duration),
                ..inner.info.clone()
            };
            Self {
                inner,
                info,
                end,
                others,
            }
        }
    }

    impl Demuxer for Tracks {
        fn info(&self) -> &VideoInfo {
            &self.info
        }
        fn next_packet(&mut self) -> PipeResult<Option<Packet>> {
            self.inner.next_packet()
        }
        fn rewind(&mut self) -> PipeResult<()> {
            self.inner.rewind()
        }
        fn frames_end(&self) -> Option<Duration> {
            self.end
        }
        fn duration_may_be_another_tracks(&self) -> bool {
            self.others
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

    /// Timestamps from `start` on, as in a WebM cut with its timestamps kept (-copyts).
    struct Late {
        inner: FakeDemux,
        info: VideoInfo,
        start: Duration,
    }

    impl Demuxer for Late {
        fn info(&self) -> &VideoInfo {
            &self.info
        }
        fn next_packet(&mut self) -> PipeResult<Option<Packet>> {
            let start = self.start;
            Ok(self.inner.next_packet()?.map(|p| Packet {
                pts: p.pts + start,
                ..p
            }))
        }
        fn rewind(&mut self) -> PipeResult<()> {
            self.inner.rewind()
        }
    }

    #[test]
    fn a_loop_of_a_video_that_starts_late_has_no_pause() {
        let ms = Duration::from_millis;
        let start = Duration::from_secs(600);
        // the container's duration as ffmpeg writes it (where the file ends), as a length,
        // and none at all
        for duration in [Some(start + ms(80)), Some(ms(80)), None] {
            let inner = FakeDemux::new(2, ms(40));
            let info = VideoInfo {
                duration,
                ..inner.info().clone()
            };
            let mut p = Pipeline::new(
                Box::new(Late { inner, info, start }),
                Box::new(FakeDecoder::default()),
                true,
            );
            let got: Vec<_> = (0..5)
                .map(|_| pts(p.next_frame().unwrap()).unwrap() - start)
                .collect();
            assert_eq!(got, [0, 40, 80, 120, 160].map(ms), "duration {duration:?}");
        }
    }

    #[test]
    fn a_file_cut_short_ends_where_it_stops() {
        let ms = Duration::from_millis;
        // the container doesn't say how long it is, or says how long it was before the cut
        // (its header is whole, 400 ms of frames): either way the loop goes on right after
        // the frames that are left, not after a pause where the rest used to be
        for duration in [None, Some(ms(400))] {
            let mut cut = Cut::new(10, 2);
            cut.info.duration = duration;
            let mut p = Pipeline::new(Box::new(cut), Box::new(FakeDecoder::default()), true);
            let got: Vec<_> = (0..4)
                .map(|_| pts(p.next_frame().unwrap()).unwrap())
                .collect();
            assert_eq!(
                got,
                [0, 40, 80, 120].map(ms),
                "two frames, then again ({duration:?})"
            );
        }

        // nothing readable at all is still an error
        let mut p = Pipeline::new(
            Box::new(Cut::new(10, 0)),
            Box::new(FakeDecoder::default()),
            true,
        );
        assert!(p.next_frame().is_err());
    }

    /// Takes packets and never has a picture.
    struct Swallows;
    impl Decoder for Swallows {
        fn name(&self) -> &'static str {
            "swallows"
        }
        fn decode(&mut self, _: &Packet) -> PipeResult<Vec<DecodedFrame>> {
            Ok(Vec::new())
        }
        fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
            Ok(Vec::new())
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn a_file_cut_short_before_its_first_frame_says_so() {
        // what could be read gave no picture: the error is the file's, not a decoder
        // that "produces no frames" after reading the file over and over
        for looped in [true, false] {
            let mut p = Pipeline::new(Box::new(Cut::new(10, 1)), Box::new(Swallows), looped);
            assert_eq!(
                p.next_frame().err().as_deref(),
                Some("read past the end"),
                "looped: {looped}"
            );
        }
    }

    #[test]
    fn a_loop_goes_on_where_the_video_ends() {
        let ms = Duration::from_millis;
        // (frames 100 ms apart, the container's duration, where the video's frames end by
        // their own durations, other tracks, where the next pass starts)
        let files = [
            // ten frames, and audio that runs on for two more seconds: no pause for it
            (10, ms(3008), Some(ms(1000)), true, ms(1000)),
            // and so without the frames' durations: the frames alone say where they end
            (10, ms(3008), None, true, ms(1000)),
            // three frames, the last one staying up for two seconds (a GIF's final pause,
            // as ffmpeg writes it into a WebM): kept
            (3, ms(2200), Some(ms(2200)), false, ms(2200)),
            // and so without the frames' durations: the duration is the video's alone
            (3, ms(2200), None, false, ms(2200)),
            // a single picture with a minute of a song under it: the same picture again
            // every frame would only cost, it comes again once per song
            (1, ms(60008), Some(ms(40)), true, ms(60008)),
            (1, ms(60008), None, true, ms(60008)),
            (1, ms(60008), Some(ms(40)), false, ms(60008)),
            // or its own duration is longer than the file says
            (1, ms(30), Some(ms(40)), true, ms(40)),
        ];
        for (frames, duration, end, others, next) in files {
            let mut p = Pipeline::new(
                Box::new(Tracks::new(frames, ms(100), duration, end, others)),
                Box::new(FakeDecoder::default()),
                true,
            );
            let got: Vec<_> = (0..=frames)
                .map(|_| pts(p.next_frame().unwrap()).unwrap())
                .collect();
            assert_eq!(
                got[frames as usize], next,
                "{frames} frames, {end:?}, other tracks: {others}: {got:?}"
            );
        }
    }

    #[test]
    fn timestamps_at_the_end_of_time_do_not_panic() {
        let ms = Duration::from_millis;
        let inner = FakeDemux::new(2, ms(40));
        let info = inner.info().clone();
        // the second frame's end is past what a Duration holds, and so is every pass after
        let start = Duration::MAX - ms(50);
        let mut p = Pipeline::new(
            Box::new(Late { inner, info, start }),
            Box::new(FakeDecoder::default()),
            true,
        );
        for _ in 0..6 {
            assert!(pts(p.next_frame().unwrap()).is_some());
        }
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
