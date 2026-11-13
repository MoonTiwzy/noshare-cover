//! Video: decoding runs on its own thread; only the latest ready frame is exposed.
//!
//! The thread sleeps while nobody requests frames (no active screen capture),
//! same as the original plugin (400 ms threshold). After waking, the clock
//! restarts from the current frame instead of catching up on missed time.
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
use pipeline::{Next, Pipeline};

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
    quit: bool,
    /// Fatal thread error (shown to the user).
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
        if let Some(e) = &st.error
            && st.generation == 0
        {
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
        if st.generation != self.sent {
            return Some(Duration::ZERO);
        }
        if st.ended || st.error.is_some() {
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
        let asleep = !is_watched(st);
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

fn run(mut pipe: Pipeline, speed: f64, sync: &Sync) {
    // (wall time, pts): the clock's reference point
    let mut origin: Option<(Instant, Duration)> = None;
    let mut target = (0, 0);
    let mut pixels_only = false;
    let mut last_pts: Option<Duration> = None;
    let mut late = 0u32; // frames dropped in a row

    loop {
        // nobody is watching: give back what is only kept between frames (outside the
        // lock: the render thread takes it for every poll), then sleep
        let idle = {
            let st = sync.lock();
            !st.quit && !is_watched(&st)
        };
        if idle {
            pipe.idle();
        }
        let (wanted, refused) = {
            let mut st = sync.lock();
            let mut slept = false;
            while !st.quit && !is_watched(&st) {
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
            if slept {
                origin = None; // after sleeping, resume from the current position instead of catching up
            }
            (st.target, st.pixels_only)
        };
        // told to the decoder outside the lock, like everything that can take a moment
        if wanted != target {
            target = wanted;
            pipe.set_target(target.0, target.1);
        }
        if refused && !pixels_only {
            pixels_only = true;
            pipe.refuse_video_memory();
        }

        let frame = match pipe.next_frame() {
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
        };

        let (t0, pts0) = *origin.get_or_insert((Instant::now(), frame.pts));
        let mut due = t0 + frame.pts.saturating_sub(pts0).div_f64(speed);
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
            .map(|p| frame.pts.saturating_sub(p).div_f64(speed))
            .filter(|gap| !gap.is_zero())
            .map(|gap| due + gap);
        last_pts = Some(frame.pts);

        if due > now {
            // wait until display time, but wake up on quit
            let mut st = sync.lock();
            st.next_due = Some(due);
            let deadline = due;
            while !st.quit {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                st = sync
                    .cv
                    .wait_timeout(st, left.min(Duration::from_secs(1)))
                    .map(|(g, _)| g)
                    .unwrap_or_else(|e| e.into_inner().0);
            }
            if st.quit {
                return;
            }
        }

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
    use super::pipeline::testing::{FakeDecoder, FakeDemux};
    use super::*;

    fn source(frames: u32, step_ms: u64, looped: bool) -> VideoSource {
        let pipe = Pipeline::new(
            Box::new(FakeDemux::new(frames, Duration::from_millis(step_ms))),
            Box::new(FakeDecoder::default()),
            looped,
        );
        VideoSource::spawn(pipe, 1.0, "fake".into())
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
        fn decode(&mut self, p: &pipeline::Packet) -> pipeline::PipeResult<Vec<pipeline::DecodedFrame>> {
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
        let mut s = VideoSource::spawn(pipe, 1.0, "slow".into());
        // long enough to fall behind by more than DROP_LATE and then some
        let end = Instant::now() + Duration::from_millis(1500);
        let mut shown = 0;
        while Instant::now() < end {
            if s.poll(Instant::now()).unwrap().is_some() {
                shown += 1;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // it used to stop for good after the first three or four
        assert!(shown >= 8, "only {shown} frames in 1.5 s");
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
        assert!(last.is_some());
    }
}
