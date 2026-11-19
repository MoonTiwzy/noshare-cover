//! Video: decoding runs on its own thread; only the latest ready frame is exposed.
//!
//! The thread sleeps while nobody requests frames (no active screen capture),
//! same as the original plugin (400 ms threshold). After waking, the clock
//! restarts from the current frame instead of catching up on missed time.
//! A source opened ahead of its first capture stops at its first frame until it
//! is shown, so the video starts from its beginning there.
//! Dropping the source stops and joins the thread, so nothing is left
//! running after the plugin is unloaded.

pub mod backend;
pub mod bitstream;
pub mod decode;
pub mod demux;
pub mod pipeline;
pub mod yuv;

use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{Kind, MediaError, Source};
use crate::config::Settings;
use crate::frame::{Frame, FrameData};
use pipeline::{DecodedFrame, Next, Pipeline};

/// How long frames can go unrequested before the thread goes to sleep.
const IDLE_AFTER: Duration = Duration::from_millis(400);
/// A frame later than this is dropped instead of shown.
const DROP_LATE: Duration = Duration::from_millis(80);
/// How many late frames are dropped one after another before playback gives up catching up.
const LATE_IN_A_ROW: u32 = 3;
/// "Any moment now": the next frame is still being decoded, or its time has come and the
/// thread is about to hand it over.
const SOON: Duration = Duration::from_millis(2);

#[derive(Default)]
struct Shared {
    latest: Option<Frame>,
    generation: u64,
    watched: Option<Instant>,
    /// When the cover first reached the screen (see `Source::shown`).
    shown: Option<Instant>,
    quit: bool,
    /// Fatal thread error (shown to the user). The thread is gone, the video stopped there.
    error: Option<String>,
    ended: bool,
    /// The size the frames are drawn at (0 = unknown), see `Decoder::set_target`.
    target: (u32, u32),
    /// Frames in video memory can't be shown, see `Decoder::refuse_video_memory`.
    pixels_only: bool,
    /// When the next frame is expected: the time of the one that is waiting for it, or a
    /// guess from the gap between frames while it is still being decoded.
    next_due: Option<Instant>,
}

struct Sync {
    state: Mutex<Shared>,
    cv: Condvar,
}

impl Sync {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct VideoSource {
    sync: Arc<Sync>,
    worker: Option<JoinHandle<()>>,
    sent: u64,
    path: String,
}

impl VideoSource {
    pub fn open(
        path: &Path,
        speed: f64,
        looped: bool,
        settings: &Settings,
    ) -> Result<Self, MediaError> {
        let shown = path.display().to_string();
        let pipeline =
            backend::open_pipeline(path, looped, settings).map_err(|reason| MediaError::Video {
                path: shown.clone(),
                reason,
            })?;
        Ok(Self::spawn(pipeline, speed, shown))
    }

    fn spawn(pipeline: Pipeline, speed: f64, path: String) -> Self {
        let sync = Arc::new(Sync {
            state: Mutex::new(Shared::default()),
            cv: Condvar::new(),
        });
        let s2 = Arc::clone(&sync);
        let speed = crate::config::sanitize_speed(speed);
        let worker = std::thread::Builder::new()
            .name("noshare-video".into())
            .spawn(move || {
                // a panic in the decoder must not take down the compositor
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(pipeline, speed, &s2)
                }));
                if r.is_err() {
                    let mut st = s2.lock();
                    st.error.get_or_insert_with(|| "decoder crashed".into());
                    s2.cv.notify_all();
                }
            })
            .expect("spawn video thread");
        Self {
            sync,
            worker: Some(worker),
            sent: 0,
            path,
        }
    }
}

impl Source for VideoSource {
    fn kind(&self) -> Kind {
        Kind::Video
    }

    fn poll(&mut self, now: Instant) -> Result<Option<Frame>, MediaError> {
        let mut st = self.sync.lock();
        self.watch(&mut st, now);
        // also after frames were shown: the picture would stand still for good, unexplained
        if let Some(e) = &st.error {
            return Err(MediaError::Video {
                path: self.path.clone(),
                reason: e.clone(),
            });
        }
        if st.generation == self.sent {
            return Ok(None);
        }
        self.sent = st.generation;
        Ok(st.latest.take())
    }

    /// Until now the thread has stopped at the first frame, the one the loader took (see
    /// `wanted`): the video goes on from there.
    fn shown(&mut self) {
        let mut st = self.sync.lock();
        if st.shown.is_none() {
            st.shown = Some(Instant::now());
            self.sync.cv.notify_all();
        }
    }

    fn set_target(&mut self, width: u32, height: u32) {
        self.sync.lock().target = (width, height);
    }

    fn refuse_video_memory(&mut self) {
        let mut st = self.sync.lock();
        st.pixels_only = true;
        // the frame that is waiting came the same way
        if st
            .latest
            .as_ref()
            .is_some_and(|f| !matches!(f.data, FrameData::Cpu(_)))
        {
            st.latest = None;
        }
    }

    fn next_frame_in(&mut self, now: Instant) -> Option<Duration> {
        let mut st = self.sync.lock();
        self.watch(&mut st, now);
        // a new frame, or the error the next poll reports
        if st.generation != self.sent || st.error.is_some() {
            return Some(Duration::ZERO);
        }
        if st.ended {
            return None;
        }
        Some(
            st.next_due
                .map_or(SOON, |due| due.saturating_duration_since(now).max(SOON)),
        )
    }
}

impl VideoSource {
    /// Somebody wants frames: wake the thread if it went to sleep.
    fn watch(&self, st: &mut Shared, now: Instant) {
        let asleep = !wanted(st);
        st.watched = Some(now);
        if asleep {
            self.sync.cv.notify_all();
        }
    }
}

impl Drop for VideoSource {
    fn drop(&mut self) {
        {
            let mut st = self.sync.lock();
            st.quit = true;
        }
        self.sync.cv.notify_all();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn is_watched(st: &Shared) -> bool {
    st.watched.is_some_and(|t| t.elapsed() < IDLE_AFTER)
}

/// The thread has something to do: somebody watches, and the video is on screen, or
/// its first frame isn't there yet. The loader takes the first frame of a cover opened
/// ahead of its first capture, possibly long before: the video must not play on unseen
/// until then, a short one that doesn't loop would be over.
fn wanted(st: &Shared) -> bool {
    is_watched(st) && (st.shown.is_some() || st.generation == 0)
}

/// `d` of the video's time in wall time at `speed`, or as much as a `Duration` holds.
fn scaled(d: Duration, speed: f64) -> Duration {
    Duration::try_from_secs_f64(d.as_secs_f64() / speed).unwrap_or(Duration::MAX)
}

/// `d` after `t`. Timestamps come from the file: one further off than a century is due no
/// sooner than in a century, instead of past what an `Instant` holds.
fn later(t: Instant, d: Duration) -> Instant {
    const CENTURY: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);
    t.checked_add(d.min(CENTURY)).unwrap_or(t)
}

fn run(mut pipe: Pipeline, speed: f64, sync: &Sync) {
    // (wall time, pts): the clock's reference point
    let mut origin: Option<(Instant, Duration)> = None;
    let mut target = (0, 0);
    let mut pixels_only = false;
    let mut last_pts: Option<Duration> = None;
    let mut late = 0u32; // frames dropped in a row
    // the frame that was waiting for its time when watching stopped: shown first after that,
    // if it is kept (see the sleep)
    let mut pending: Option<DecodedFrame> = None;
    // when the last frame was handed out, and its pts
    let mut handed: Option<(Instant, Duration)> = None;
    // the cover has been on screen (see `wanted`)
    let mut on_screen = false;

    loop {
        // nobody is watching: give back what is only kept between frames (outside the
        // lock: the render thread takes it for every poll), then sleep
        let idle = {
            let st = sync.lock();
            !st.quit && !wanted(&st)
        };
        if idle {
            // A frame that was waiting for its time in video memory holds one of the
            // decoder's buffers for as long as this sleeps, possibly for good: let it go
            // first, so it is given back too. The clock starts over after the sleep anyway,
            // a looped video goes on from the frame after it. One that doesn't loop keeps
            // it: it may be the last one, which stays up when the video has ended.
            if pipe.looped() {
                pending = pending.filter(|f| matches!(f.data, FrameData::Cpu(_)));
            }
            pipe.idle();
        }
        let (size, refused) = {
            let mut st = sync.lock();
            let mut slept = false;
            while !st.quit && !wanted(&st) {
                slept = true;
                st = sync
                    .cv
                    .wait_timeout(st, IDLE_AFTER)
                    .map(|(g, _)| g)
                    .unwrap_or_else(|e| e.into_inner().0);
            }
            if st.quit {
                return;
            }
            if let (false, Some(at)) = (on_screen, st.shown) {
                on_screen = true;
                // the first frame is on screen since it was shown (or since it came, if that
                // was later): the clock starts there, the next frame is a frame's time away
                origin = handed.map(|(t, pts)| (t.max(at), pts));
            } else if slept {
                origin = None; // after sleeping, resume from the current position instead of catching up
            }
            (st.target, st.pixels_only)
        };
        // told to the decoder outside the lock, like everything that can take a moment
        if size != target {
            target = size;
            pipe.set_target(target.0, target.1);
        }
        if refused && !pixels_only {
            pixels_only = true;
            pipe.refuse_video_memory();
            pending = pending.filter(|f| matches!(f.data, FrameData::Cpu(_)));
        }

        let frame = match pending.take() {
            Some(f) => f,
            None => match pipe.next_frame() {
                Ok(Next::Frame(f)) => f,
                Ok(Next::End) => {
                    let mut st = sync.lock();
                    st.ended = true;
                    st.next_due = None;
                    // video ended without looping: the last frame stays on screen
                    while !st.quit {
                        st = sync.cv.wait(st).unwrap_or_else(|e| e.into_inner());
                    }
                    return;
                }
                Err(e) => {
                    let mut st = sync.lock();
                    st.error = Some(e);
                    sync.cv.notify_all();
                    return;
                }
            },
        };

        let (t0, pts0) = *origin.get_or_insert((Instant::now(), frame.pts));
        let mut due = later(t0, scaled(frame.pts.saturating_sub(pts0), speed));
        let now = Instant::now();
        // Late by more than a frame can be: drop it so we don't fall further behind. But
        // a decoder that is slower than the video falls behind anyway, and a dropped frame
        // costs it as much as a shown one: then nothing would ever be shown again. After a
        // few in a row, go on from this frame, slower than the video.
        if now.saturating_duration_since(due) > DROP_LATE {
            late += 1;
            if late <= LATE_IN_A_ROW {
                last_pts = Some(frame.pts);
                continue;
            }
            origin = Some((now, frame.pts));
            due = now;
        }
        late = 0;
        // the frame after this one, if the gap stays the same
        let after = last_pts
            .map(|p| scaled(frame.pts.saturating_sub(p), speed))
            .filter(|gap| !gap.is_zero())
            .map(|gap| later(due, gap));

        if due > now {
            // Wait until display time, but wake up on quit. Watching stops without a word
            // (no more polls): look at it now and then, and when it has stopped, sleep and
            // give back what the decoder keeps between frames, as between two frames, even
            // when this one is a long way off.
            let mut st = sync.lock();
            st.next_due = Some(due);
            while !st.quit && wanted(&st) {
                let left = due.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                st = sync
                    .cv
                    .wait_timeout(st, left.min(IDLE_AFTER))
                    .map(|(g, _)| g)
                    .unwrap_or_else(|e| e.into_inner().0);
            }
            if st.quit {
                return;
            }
            if !wanted(&st) {
                // asking wakes the thread, which then hands this frame out right away
                st.next_due = None;
                pending = Some(frame);
                continue;
            }
        }
        last_pts = Some(frame.pts);
        handed = Some((Instant::now(), frame.pts));

        let mut st = sync.lock();
        st.generation += 1;
        let generation = st.generation;
        st.latest = Some(Frame {
            generation,
            data: frame.data,
        });
        st.next_due = after;
        drop(st);
        sync.cv.notify_all(); // the first poll may be waiting for this frame
    }
}

#[cfg(test)]
mod tests {
    use super::pipeline::Demuxer;
    use super::pipeline::testing::{FakeDecoder, FakeDemux, Tracks};
    use super::*;

    /// A source on screen from the start.
    fn source(frames: u32, step_ms: u64, looped: bool) -> VideoSource {
        let pipe = Pipeline::new(
            Box::new(FakeDemux::new(frames, Duration::from_millis(step_ms))),
            Box::new(FakeDecoder::default()),
            looped,
        );
        shown(pipe, "fake")
    }

    fn shown(pipe: Pipeline, name: &str) -> VideoSource {
        let mut s = VideoSource::spawn(pipe, 1.0, name.into());
        s.shown();
        s
    }

    /// The number of the fake decoder's frame (see `InVideoMemory` for one in video memory).
    fn number(f: &Frame) -> u8 {
        match &f.data {
            FrameData::Cpu(c) => c.pixels[0],
            #[cfg(target_os = "linux")]
            FrameData::DmaBuf(d) => d.buffer as u8,
        }
    }

    fn wait_frame(s: &mut VideoSource, timeout: Duration) -> Option<Frame> {
        let end = Instant::now() + timeout;
        while Instant::now() < end {
            if let Some(f) = s.poll(Instant::now()).unwrap() {
                return Some(f);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        None
    }

    /// Takes three times a frame's time to decode it.
    struct SlowDecoder(FakeDecoder);
    impl pipeline::Decoder for SlowDecoder {
        fn name(&self) -> &'static str {
            "slow"
        }
        fn decode(
            &mut self,
            p: &pipeline::Packet,
        ) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
            std::thread::sleep(Duration::from_millis(60));
            self.0.decode(p)
        }
        fn flush(&mut self) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
            self.0.flush()
        }
        fn reset(&mut self) {
            self.0.reset();
        }
    }

    #[test]
    fn a_decoder_slower_than_the_video_still_shows_frames() {
        let pipe = Pipeline::new(
            Box::new(FakeDemux::new(1000, Duration::from_millis(20))),
            Box::new(SlowDecoder(FakeDecoder::default())),
            true,
        );
        let mut s = shown(pipe, "slow");
        // It used to stop for good after the first three or four. Now it shows one of every
        // few frames it decodes, however slowly that goes on a busy machine: wait for well
        // more than four, with a deadline only for when they never come.
        const WANTED: u32 = 12;
        let end = Instant::now() + Duration::from_secs(20);
        let mut shown = 0;
        while shown < WANTED && Instant::now() < end {
            if s.poll(Instant::now()).unwrap().is_some() {
                shown += 1;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(shown, WANTED, "only {shown} frames");
    }

    #[test]
    fn produces_frames_while_watched() {
        let mut s = source(50, 10, true);
        let a = wait_frame(&mut s, Duration::from_secs(2)).expect("first frame");
        let b = wait_frame(&mut s, Duration::from_secs(2)).expect("next frame");
        assert!(b.generation > a.generation);
    }

    #[test]
    fn sleeps_when_not_watched() {
        let mut s = source(1000, 5, true);
        wait_frame(&mut s, Duration::from_secs(2)).expect("frame");
        std::thread::sleep(IDLE_AFTER + Duration::from_millis(200));
        let before = s.sync.lock().generation;
        std::thread::sleep(Duration::from_millis(300));
        let after = s.sync.lock().generation;
        assert_eq!(before, after, "no decoding without viewers");
    }

    #[test]
    fn asking_for_the_next_frame_keeps_it_awake() {
        let mut s = source(1000, 5, true);
        wait_frame(&mut s, Duration::from_secs(2)).expect("frame");
        // no poll for longer than IDLE_AFTER, only the question
        let end = Instant::now() + IDLE_AFTER + Duration::from_millis(200);
        while Instant::now() < end {
            assert!(s.next_frame_in(Instant::now()).is_some());
            std::thread::sleep(Duration::from_millis(20));
        }
        let before = s.sync.lock().generation;
        let end = Instant::now() + Duration::from_millis(300);
        while Instant::now() < end {
            s.next_frame_in(Instant::now());
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(s.sync.lock().generation > before, "still decoding");
    }

    #[test]
    fn next_frame_in_follows_the_frame_times() {
        let step = Duration::from_millis(150);
        let mut s = source(50, 150, true);
        wait_frame(&mut s, Duration::from_secs(2)).expect("first frame");
        let wait = s
            .next_frame_in(Instant::now())
            .expect("more frames to come");
        assert!(
            wait > Duration::ZERO && wait <= step,
            "the next frame is {wait:?} away"
        );
        std::thread::sleep(wait + Duration::from_millis(40));
        assert_eq!(
            s.next_frame_in(Instant::now()),
            Some(Duration::ZERO),
            "it is there now"
        );
        assert!(s.poll(Instant::now()).unwrap().is_some());

        // played out: nothing more to wait for
        let mut s = source(2, 5, false);
        let end = Instant::now() + Duration::from_secs(2);
        while Instant::now() < end && !s.sync.lock().ended {
            let _ = s.poll(Instant::now());
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = s.poll(Instant::now()); // the last frame, if it wasn't taken yet
        assert_eq!(s.next_frame_in(Instant::now()), None);
    }

    #[test]
    fn the_target_size_reaches_the_decoder() {
        let mut s = source(1000, 5, true);
        let alpha = |f: &Frame| match &f.data {
            crate::frame::FrameData::Cpu(c) => c.pixels[3],
            #[allow(unreachable_patterns)]
            _ => 0,
        };
        let first = wait_frame(&mut s, Duration::from_secs(2)).expect("frame");
        assert_eq!(alpha(&first), 255, "no target yet");
        s.set_target(7, 9);
        let end = Instant::now() + Duration::from_secs(2);
        let mut seen = 255;
        while Instant::now() < end && seen != 7 {
            if let Some(f) = wait_frame(&mut s, Duration::from_millis(200)) {
                seen = alpha(&f);
            }
        }
        assert_eq!(seen, 7, "frames decoded after set_target carry it");
    }

    #[test]
    fn a_refusal_of_video_memory_reaches_the_decoder() {
        let mut s = source(1000, 5, true);
        let red = |f: &Frame| match &f.data {
            FrameData::Cpu(c) => c.pixels[2],
            #[allow(unreachable_patterns)]
            _ => 255,
        };
        // the fake decoder's frames are grey until it is told, then their red is 0;
        // skip the first frames, whose grey is still near black
        let end = Instant::now() + Duration::from_secs(2);
        let mut seen = 0;
        while Instant::now() < end && seen == 0 {
            seen = wait_frame(&mut s, Duration::from_millis(200)).map_or(0, |f| red(&f));
        }
        assert_ne!(seen, 0, "frames as the decoder makes them");
        s.refuse_video_memory();
        let end = Instant::now() + Duration::from_secs(2);
        while Instant::now() < end && seen != 0 {
            seen = wait_frame(&mut s, Duration::from_millis(200)).map_or(seen, |f| red(&f));
        }
        assert_eq!(
            seen, 0,
            "frames decoded after the refusal come the other way"
        );
    }

    #[test]
    fn a_video_opened_ahead_of_time_starts_from_its_first_frame_when_shown() {
        let step = Duration::from_millis(50);
        let pipe = Pipeline::new(
            Box::new(FakeDemux::new(50, step)),
            Box::new(FakeDecoder::default()),
            false,
        );
        let mut s = VideoSource::spawn(pipe, 1.0, "early".into());
        // the loader takes the first frame; the first capture comes a while later
        let first = wait_frame(&mut s, Duration::from_secs(2)).expect("first frame");
        assert_eq!(number(&first), 0);
        std::thread::sleep(IDLE_AFTER + step * 4);
        s.shown();
        let shown = Instant::now();
        assert!(
            s.poll(shown).unwrap().is_none(),
            "the video didn't play on unseen: still its first frame"
        );
        let next = wait_frame(&mut s, Duration::from_secs(2)).expect("it plays once shown");
        assert_eq!(number(&next), 1, "the frame after the first one");
        assert!(
            shown.elapsed() + Duration::from_millis(5) >= step,
            "the first frame stays up for its time, not {:?}",
            shown.elapsed()
        );
    }

    #[test]
    fn a_still_with_a_longer_track_is_not_handed_out_again_every_frame() {
        let ms = Duration::from_millis;
        // a single picture with a minute of a song under it, the picture 40 ms long
        let pipe = Pipeline::new(
            Box::new(Tracks::new(1, ms(40), ms(60008), Some(ms(40)), true)),
            Box::new(FakeDecoder::default()),
            true,
        );
        let mut s = shown(pipe, "still");
        wait_frame(&mut s, Duration::from_secs(2)).expect("the picture");
        // the next pass is read as soon as the picture is handed out: it is due in a minute
        let far = Duration::from_secs(50);
        let end = Instant::now() + Duration::from_secs(2);
        while Instant::now() < end && s.next_frame_in(Instant::now()).is_some_and(|d| d < far) {
            std::thread::sleep(ms(2));
        }
        let wait = s.next_frame_in(Instant::now());
        assert!(
            wait.is_some_and(|d| d >= far),
            "the picture comes again in {wait:?}"
        );
        let before = s.sync.lock().generation;
        let end = Instant::now() + ms(300);
        while Instant::now() < end {
            assert!(
                s.poll(Instant::now()).unwrap().is_none(),
                "handed out again"
            );
            std::thread::sleep(ms(10));
        }
        assert_eq!(s.sync.lock().generation, before);
    }

    /// Decodes `left` packets, then fails.
    struct Breaks {
        inner: FakeDecoder,
        left: u32,
    }
    impl pipeline::Decoder for Breaks {
        fn name(&self) -> &'static str {
            "breaks"
        }
        fn decode(
            &mut self,
            p: &pipeline::Packet,
        ) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
            if self.left == 0 {
                return Err("broke".into());
            }
            self.left -= 1;
            self.inner.decode(p)
        }
        fn flush(&mut self) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
            self.inner.flush()
        }
        fn reset(&mut self) {
            self.inner.reset();
        }
    }

    #[test]
    fn an_error_after_the_first_frames_is_reported() {
        let pipe = Pipeline::new(
            Box::new(FakeDemux::new(100, Duration::from_millis(5))),
            Box::new(Breaks {
                inner: FakeDecoder::default(),
                left: 4,
            }),
            true,
        );
        let mut s = shown(pipe, "breaks");
        let end = Instant::now() + Duration::from_secs(2);
        while Instant::now() < end && s.sync.lock().error.is_none() {
            s.next_frame_in(Instant::now()); // keeps it decoding
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(s.sync.lock().generation > 0, "frames came first");
        assert!(s.sync.lock().error.is_some(), "then the decoder broke");
        // Every frame there was has been taken: the picture would stand still for good. The
        // shim is told to repaint, and the repaint's poll reports the error.
        s.sent = s.sync.lock().generation;
        assert_eq!(s.next_frame_in(Instant::now()), Some(Duration::ZERO));
        let err = s.poll(Instant::now()).expect_err("the error");
        assert!(err.to_string().contains("broke"), "{err}");
    }

    /// Counts how often it is told nobody watches.
    struct CountsIdle(FakeDecoder, Arc<std::sync::atomic::AtomicUsize>);
    impl pipeline::Decoder for CountsIdle {
        fn name(&self) -> &'static str {
            "counts idle"
        }
        fn decode(
            &mut self,
            p: &pipeline::Packet,
        ) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
            self.0.decode(p)
        }
        fn flush(&mut self) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
            self.0.flush()
        }
        fn reset(&mut self) {
            self.0.reset();
        }
        fn idle(&mut self) {
            self.1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn waiting_for_a_frame_a_long_way_off_stops_when_nobody_watches() {
        use std::sync::atomic::Ordering::SeqCst;
        let idles = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pipe = Pipeline::new(
            Box::new(FakeDemux::new(2, Duration::from_secs(10))),
            Box::new(CountsIdle(FakeDecoder::default(), Arc::clone(&idles))),
            false,
        );
        let mut s = shown(pipe, "far");
        wait_frame(&mut s, Duration::from_secs(2)).expect("first frame");
        let before = idles.load(SeqCst);
        // nobody asks any more, long before the next frame is due
        std::thread::sleep(IDLE_AFTER * 4);
        assert!(idles.load(SeqCst) > before, "the decoder was told");
        // asked again: the frame waiting for its time comes now, the clock goes on from it
        let next = wait_frame(&mut s, Duration::from_secs(5)).expect("the next frame");
        assert_eq!(number(&next), 1);
    }

    /// The fake decoder's frames in video memory: no memory and no fds, the number as the
    /// buffer, and a log of what is let go and when it is told nobody watches.
    #[cfg(target_os = "linux")]
    struct InVideoMemory(FakeDecoder, Arc<Mutex<Vec<String>>>);
    #[cfg(target_os = "linux")]
    impl InVideoMemory {
        fn moved(&self, frames: Vec<DecodedFrame>) -> Vec<DecodedFrame> {
            let to_gpu = |f: DecodedFrame| {
                let FrameData::Cpu(c) = f.data else {
                    return f;
                };
                let (n, log) = (c.pixels[0], Arc::clone(&self.1));
                let release = move || log.lock().unwrap().push(format!("let go of {n}"));
                let data = FrameData::DmaBuf(Arc::new(crate::frame::DmaBufFrame {
                    width: c.width,
                    height: c.height,
                    fourcc: 0,
                    modifier: 0,
                    planes: Vec::new(),
                    buffer: n.into(),
                    release: Some(Box::new(release)),
                }));
                DecodedFrame { pts: f.pts, data }
            };
            frames.into_iter().map(to_gpu).collect()
        }
    }
    #[cfg(target_os = "linux")]
    impl pipeline::Decoder for InVideoMemory {
        fn name(&self) -> &'static str {
            "in video memory"
        }
        fn decode(
            &mut self,
            p: &pipeline::Packet,
        ) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
            let frames = self.0.decode(p)?;
            Ok(self.moved(frames))
        }
        fn flush(&mut self) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
            let frames = self.0.flush()?;
            Ok(self.moved(frames))
        }
        fn reset(&mut self) {
            self.0.reset();
        }
        fn idle(&mut self) {
            self.1.lock().unwrap().push("idle".into());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_frame_in_video_memory_is_let_go_before_a_long_sleep() {
        // (looped, frames 10 s apart, the frame shown when watching comes back)
        for (looped, frames, next) in [(true, 3, 2), (false, 2, 1)] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let pipe = Pipeline::new(
                Box::new(FakeDemux::new(frames, Duration::from_secs(10))),
                Box::new(InVideoMemory(FakeDecoder::default(), Arc::clone(&log))),
                looped,
            );
            let mut s = shown(pipe, "far, in video memory");
            let first = wait_frame(&mut s, Duration::from_secs(2)).expect("first frame");
            assert_eq!(number(&first), 0);
            // nobody asks any more while frame 1 waits for its time
            std::thread::sleep(IDLE_AFTER * 4);
            let seen = log.lock().unwrap().clone();
            let at = |what: &str| seen.iter().rposition(|e| e == what);
            let (let_go, idle) = (at("let go of 1"), at("idle"));
            assert!(idle.is_some(), "the decoder was told: {seen:?}");
            if looped {
                // let go of first, so the decoder can take its buffer back
                assert!(let_go.is_some() && idle > let_go, "{seen:?}");
            } else {
                // kept: it is the last frame, which stays up once the video has ended
                assert_eq!(let_go, None, "{seen:?}");
            }
            // asked again
            let next_one = wait_frame(&mut s, Duration::from_secs(5)).expect("the next frame");
            assert_eq!(number(&next_one), next, "looped: {looped}");
        }
    }

    /// A frame, then one at the end of time.
    struct Jumps {
        info: pipeline::VideoInfo,
        read: u8,
    }
    impl pipeline::Demuxer for Jumps {
        fn info(&self) -> &pipeline::VideoInfo {
            &self.info
        }
        fn next_packet(&mut self) -> pipeline::PipeResult<Option<pipeline::Packet>> {
            let pts = match self.read {
                0 => Duration::ZERO,
                1 => Duration::MAX,
                _ => return Ok(None),
            };
            self.read += 1;
            Ok(Some(pipeline::Packet {
                pts,
                keyframe: pts.is_zero(),
                data: vec![self.read - 1],
            }))
        }
        fn rewind(&mut self) -> pipeline::PipeResult<()> {
            self.read = 0;
            Ok(())
        }
    }

    #[test]
    fn timestamps_out_of_range_do_not_break_the_video() {
        let info = FakeDemux::new(1, Duration::from_millis(40)).info().clone();
        let pipe = Pipeline::new(
            Box::new(Jumps { info, read: 0 }),
            Box::new(FakeDecoder::default()),
            true,
        );
        // slowed down, the time between the two frames is twice as far out of range
        let mut s = VideoSource::spawn(pipe, 0.5, "jumps".into());
        s.shown();
        wait_frame(&mut s, Duration::from_secs(2)).expect("first frame");
        let end = Instant::now() + Duration::from_millis(300);
        while Instant::now() < end {
            if let Err(e) = s.poll(Instant::now()) {
                panic!("{e}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(s.sync.lock().error, None, "the thread is still there");
        let t = Instant::now();
        drop(s);
        assert!(
            t.elapsed() < Duration::from_millis(500),
            "still waiting for it"
        );
    }

    #[test]
    fn drop_joins_thread_quickly() {
        let mut s = source(1000, 1000, true); // long gaps between frames
        wait_frame(&mut s, Duration::from_secs(2));
        let t = Instant::now();
        drop(s);
        assert!(
            t.elapsed() < Duration::from_millis(500),
            "drop must not wait for next frame"
        );
    }

    #[test]
    fn no_loop_keeps_last_frame_and_stops() {
        let mut s = source(3, 5, false);
        let mut last = None;
        let end = Instant::now() + Duration::from_secs(2);
        while Instant::now() < end && !s.sync.lock().ended {
            if let Some(f) = s.poll(Instant::now()).unwrap() {
                last = Some(f.generation);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(s.sync.lock().ended);
        // on a busy machine it may have played out between two polls: the last frame is
        // still there to take
        if let Some(f) = s.poll(Instant::now()).unwrap() {
            last = Some(f.generation);
        }
        assert!(last.is_some());
    }
}
