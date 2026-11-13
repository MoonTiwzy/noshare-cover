//! Cover registry: one source per unique (file, speed, loop), no matter how
//! many windows show it.
//!
//! Lifecycle:
//! - global settings change: everything is reset and the epoch is bumped (the
//!   shim drops its textures on a new epoch);
//! - file missing: retry once per second, no stat() on every frame;
//! - failed to open: one notification, no retries until the file or the config changes;
//! - the file changed on disk (looked at once per second, while the cover is shown):
//!   it is opened again once it has stopped changing;
//! - cover not shown for a while: the source is closed (for video this stops
//!   the decode thread) instead of living until the plugin is unloaded.
//!
//! Every `resolve` says how large the cover is drawn. The source gets the largest
//! size asked for lately, so a video shown in a small window is decoded into small
//! frames (see `Want`).
//!
//! Opening never happens on the render path: `resolve` is called from the
//! compositor's render hook, and opening a file (decoding a PNG, reading a GIF,
//! setting up a video decoder and waiting for its first frame) can take tens
//! of milliseconds. A new cover is opened on a loader thread; until it is ready
//! `resolve` returns `None` and the shim keeps drawing the black box, and
//! `animating` stays true so the shim keeps asking for frames and picks the
//! cover up as soon as it lands.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use crate::config::{ConfigError, PlayParams, Settings};
use crate::frame::Frame;
use crate::media::{self, MediaError, Source};
use crate::notify::Notifier;
use crate::worker::Worker;

/// Opens a source for the given parameters (replaced in tests).
pub type Opener = fn(&PlayParams, &Settings) -> Result<Box<dyn Source>, MediaError>;

const MISSING_RETRY: Duration = Duration::from_secs(1);
const EVICT_AFTER: Duration = Duration::from_secs(30);
/// How long the loader keeps polling a freshly opened source for its first
/// frame before handing it over anyway (the render path then just polls it).
const FIRST_FRAME_WAIT: Duration = Duration::from_secs(2);
const FIRST_FRAME_STEP: Duration = Duration::from_millis(4);
/// How soon to look again at a cover that is still being opened.
const OPENING_STEP: Duration = Duration::from_millis(8);

/// What tells one version of a file from the next: its modification time and length.
type Stamp = (std::time::SystemTime, u64);

fn stamp_of(path: &std::path::Path) -> Option<Stamp> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

type Loaded = Result<(Box<dyn Source>, Option<Frame>), MediaError>;

/// Opens a source and prepares its first frame on a separate thread.
struct Loader {
    rx: Receiver<Loaded>,
    cancel: Arc<AtomicBool>,
    worker: Option<Worker>,
}

impl Loader {
    fn spawn(opener: Opener, play: PlayParams, settings: Settings) -> Self {
        let (tx, rx) = mpsc::sync_channel(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&cancel);
        let job = move || {
            // a panic in a decoder must not take down the compositor
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                load(opener, &play, &settings, &stop)
            }))
            .unwrap_or_else(|_| {
                Err(MediaError::Open {
                    path: play.path.display().to_string(),
                    reason: "loader crashed".into(),
                })
            });
            let _ = tx.send(r);
        };
        // started from Hyprland's thread: see worker.rs
        let worker = Worker::spawn(c"noshare-open", job).ok();
        Self { rx, cancel, worker }
    }
}

impl Loader {
    /// The thread is done (result sent, or it never started): dropping the
    /// loader now doesn't wait.
    fn finished(&self) -> bool {
        self.worker.as_ref().is_none_or(|w| w.is_finished())
    }

    /// Nobody needs the result any more. Stops the first-frame wait; the open
    /// itself (decoding a whole GIF, setting up a decoder) can't be interrupted,
    /// so the registry parks the loader instead of dropping it on the render path.
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        // The thread must be gone before the plugin can be unloaded. The registry
        // only drops loaders that are finished, or on its own drop (unload).
        self.cancel();
        self.worker.take(); // joins
    }
}

fn load(opener: Opener, play: &PlayParams, settings: &Settings, cancel: &AtomicBool) -> Loaded {
    let mut src = opener(play, settings)?;
    let deadline = Instant::now() + FIRST_FRAME_WAIT;
    loop {
        if let Some(frame) = src.poll(Instant::now())? {
            return Ok((src, Some(frame)));
        }
        if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
            return Ok((src, None));
        }
        std::thread::sleep(FIRST_FRAME_STEP);
    }
}

enum State {
    /// Being opened on a loader thread. `retry` is set when this is a reopen of
    /// a file that was missing (its absence was already reported).
    Opening {
        loader: Loader,
        retry: bool,
    },
    Ready(Box<dyn Source>),
    Missing {
        retry_at: Instant,
    },
    Failed,
}

/// How large a cover is drawn: the largest size asked for in this stretch of time and
/// the one before it. One cover can be on several monitors and in several windows, each
/// capture frame asking for its own size; with two stretches the answer doesn't flip
/// between them, and it still shrinks a few seconds after the largest window is gone.
#[derive(Default)]
struct Want {
    now: (u32, u32),
    before: (u32, u32),
    since: Option<Instant>,
}

/// Stands for "as large as the source is" inside `Want`.
const FULL: (u32, u32) = (u32::MAX, u32::MAX);

impl Want {
    const STRETCH: Duration = Duration::from_secs(2);

    /// `size` in pixels; a zero side means "at its own size".
    fn add(&mut self, size: (u32, u32), at: Instant) {
        let size = if size.0 == 0 || size.1 == 0 {
            FULL
        } else {
            size
        };
        let age = self.since.map(|t| at.saturating_duration_since(t));
        match age {
            Some(a) if a < Self::STRETCH => {}
            Some(a) if a < Self::STRETCH * 2 => {
                self.before = std::mem::take(&mut self.now);
                self.since = Some(at);
            }
            _ => {
                (self.now, self.before) = ((0, 0), (0, 0));
                self.since = Some(at);
            }
        }
        self.now = (self.now.0.max(size.0), self.now.1.max(size.1));
    }

    /// What the source is told: the size in pixels, (0, 0) for its own size.
    fn target(&self) -> (u32, u32) {
        let w = self.now.0.max(self.before.0);
        let h = self.now.1.max(self.before.1);
        if w == u32::MAX || h == u32::MAX {
            (0, 0)
        } else {
            (w, h)
        }
    }
}

struct Cover {
    id: u64,
    state: State,
    current: Option<Frame>,
    polled_in: u64,
    last_used: Instant,
    want: Want,
    /// what the source was last told (see `Want::target`)
    target: (u32, u32),
    /// the file as it was when it was opened
    stamp: Option<Stamp>,
    /// a newer file seen at the last look; opened once it is the same at the next one
    /// (a file that is still being written keeps changing)
    changed: Option<Stamp>,
    look_at: Instant,
}

/// What the shim draws: a stable cover id (texture cache key) and the frame.
pub struct CoverView<'a> {
    pub id: u64,
    pub frame: &'a Frame,
}

pub struct Registry {
    settings: Option<Settings>,
    epoch: u64,
    covers: HashMap<PlayParams, Cover>,
    next_id: u64,
    frame_no: u64,
    notifier: Notifier,
    opener: Opener,
    /// Loaders of covers dropped while still opening (settings change, eviction).
    /// Joining them right there could stall the render hook for as long as the
    /// open takes, so they are parked here and reaped once finished. What is left
    /// is joined when the registry itself goes away, on unload.
    retired: Vec<Loader>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::with_opener(media::open)
    }
}

impl Registry {
    pub fn with_opener(opener: Opener) -> Self {
        Self {
            settings: None,
            epoch: 0,
            covers: HashMap::new(),
            next_id: 1,
            frame_no: 0,
            notifier: Notifier::default(),
            opener,
            retired: Vec::new(),
        }
    }

    /// Park the loader of a cover that is going away (see `retired`).
    fn retire(&mut self, cover: Cover) {
        // a finished loader is simply dropped with the cover: joining it is instant
        if let State::Opening { loader, .. } = cover.state
            && !loader.finished()
        {
            loader.cancel();
            self.retired.push(loader);
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn settings(&self) -> Settings {
        self.settings.clone().unwrap_or_default()
    }

    /// The default cover's parameters (no window rules applied).
    pub fn default_play(&self) -> PlayParams {
        match &self.settings {
            Some(s) => s.default_play(),
            None => Settings::default().default_play(),
        }
    }

    pub fn notifier(&mut self) -> &mut Notifier {
        &mut self.notifier
    }

    /// New global settings. If they changed, reset everything.
    pub fn set_settings(&mut self, settings: Settings, error: Option<ConfigError>) {
        if self.settings.as_ref() == Some(&settings) {
            return;
        }
        // dropping sources stops the video threads; covers still opening are parked
        for (_, cover) in std::mem::take(&mut self.covers) {
            self.retire(cover);
        }
        self.settings = Some(settings);
        self.epoch += 1;
        self.notifier.reset();
        if let Some(e) = error {
            self.notifier.push(e.to_string());
        }
        self.prewarm(Instant::now());
    }

    /// Open the default cover up front so the first screen capture already has
    /// a frame ready. The video then sleeps (nobody is watching) but keeps its
    /// last frame; eviction never touches it.
    fn prewarm(&mut self, now: Instant) {
        let play = self.default_play();
        if play.path.as_os_str().is_empty() {
            return;
        }
        self.warm(&play, now);
    }

    pub fn begin_frame(&mut self) {
        self.frame_no += 1;
        // joining a finished thread is instant
        self.retired.retain(|l| !l.finished());
    }

    /// Cover for a window. `None` means nothing to draw (no file, error, no frame yet).
    /// `size`: how large it is drawn, in pixels; a zero side means at its own size.
    pub fn resolve(
        &mut self,
        play: &PlayParams,
        size: (u32, u32),
        now: Instant,
    ) -> Option<CoverView<'_>> {
        // no default cover and no rule for this window: nothing to draw, not an error
        if play.path.as_os_str().is_empty() {
            return None;
        }
        let frame_no = self.frame_no;

        if !self.covers.contains_key(play) {
            self.start(play, now);
        }

        let cover = self.covers.get_mut(play)?;
        cover.last_used = now;
        cover.want.add(size, now);

        if let State::Missing { retry_at } = cover.state {
            if now >= retry_at && play.path.exists() {
                cover.stamp = stamp_of(&play.path);
                cover.state = State::Opening {
                    loader: Loader::spawn(
                        self.opener,
                        play.clone(),
                        self.settings.clone().unwrap_or_default(),
                    ),
                    retry: true,
                };
            } else if now >= retry_at {
                cover.state = State::Missing {
                    retry_at: now + MISSING_RETRY,
                };
            }
        }

        // A file replaced in place: open the new one. Also after a failed open, which may
        // have read a file that was only half there.
        if matches!(cover.state, State::Ready(_) | State::Failed) && now >= cover.look_at {
            cover.look_at = now + MISSING_RETRY;
            let seen = stamp_of(&play.path);
            if seen.is_none() || seen == cover.stamp {
                cover.changed = None;
            } else if seen != cover.changed {
                cover.changed = seen;
            } else {
                cover.changed = None;
                cover.stamp = seen;
                // A new id: the shim keeps textures by id and frame generation, and the new
                // source counts its generations (and its video memory buffers) from the start.
                cover.id = self.next_id;
                self.next_id += 1;
                // the old picture stays up until the new one is there, if it is plain pixels
                let pixels = |f: &Frame| matches!(f.data, crate::frame::FrameData::Cpu(_));
                if !cover.current.as_ref().is_some_and(pixels) {
                    cover.current = None;
                }
                cover.target = (0, 0);
                cover.state = State::Opening {
                    loader: Loader::spawn(
                        self.opener,
                        play.clone(),
                        self.settings.clone().unwrap_or_default(),
                    ),
                    retry: false,
                };
            }
        }

        if let State::Opening { loader, retry } = &cover.state {
            let retry = *retry;
            let next = match loader.rx.try_recv() {
                Err(TryRecvError::Empty) => None,
                Ok(Ok((mut src, frame))) => {
                    // first time on screen: restart its timeline and poll it
                    // right below, so the clock starts with this frame
                    src.shown();
                    if let Some(f) = frame {
                        cover.current = Some(f);
                    }
                    Some(State::Ready(src))
                }
                Ok(Err(MediaError::Missing(p))) => {
                    if !retry {
                        self.notifier
                            .push(format!("noshare-cover: file not found: {p}"));
                    }
                    Some(State::Missing {
                        retry_at: now + MISSING_RETRY,
                    })
                }
                Ok(Err(e)) => {
                    self.notifier.push(format!("noshare-cover: {e}"));
                    cover.current = None; // of the file's earlier version
                    Some(State::Failed)
                }
                Err(TryRecvError::Disconnected) => {
                    self.notifier.push(String::from(
                        "noshare-cover: failed to start the cover loader",
                    ));
                    cover.current = None;
                    Some(State::Failed)
                }
            };
            if let Some(state) = next {
                cover.state = state; // drops the loader, its thread has finished
            }
        }

        if let State::Ready(src) = &mut cover.state {
            let target = cover.want.target();
            if target != cover.target {
                cover.target = target;
                src.set_target(target.0, target.1);
            }
            // a cover shown in several windows is polled once per frame
            if cover.polled_in != frame_no {
                cover.polled_in = frame_no;
                match src.poll(now) {
                    Ok(Some(f)) => cover.current = Some(f),
                    Ok(None) => {}
                    Err(e) => {
                        self.notifier.push(format!("noshare-cover: {e}"));
                        cover.state = State::Failed;
                        cover.current = None;
                    }
                }
            }
        }

        let cover = self.covers.get(play)?;
        cover.current.as_ref().map(|frame| CoverView {
            id: cover.id,
            frame,
        })
    }

    /// Start opening a cover ahead of time: a window just got a rule with it,
    /// so by the first capture (a single screenshot has no second frame to
    /// catch up in) the frame is already there. No-op if the cover exists.
    pub fn warm(&mut self, play: &PlayParams, now: Instant) {
        if play.path.as_os_str().is_empty() || self.covers.contains_key(play) {
            return;
        }
        self.start(play, now);
    }

    fn start(&mut self, play: &PlayParams, now: Instant) {
        let id = self.next_id;
        self.next_id += 1;
        let settings = self.settings.clone().unwrap_or_default();
        let state = State::Opening {
            loader: Loader::spawn(self.opener, play.clone(), settings),
            retry: false,
        };
        self.covers.insert(
            play.clone(),
            Cover {
                id,
                state,
                current: None,
                polled_in: 0,
                last_used: now,
                want: Want::default(),
                target: (0, 0),
                stamp: stamp_of(&play.path),
                changed: None,
                look_at: now + MISSING_RETRY,
            },
        );
    }

    /// The id of the cover with these parameters, also while it has no frame yet.
    pub fn id_of(&self, play: &PlayParams) -> Option<u64> {
        self.covers.get(play).map(|c| c.id)
    }

    /// The shim couldn't make a texture of cover `id`'s frame, which came in video
    /// memory: drop it and have the source hand out pixels from now on.
    pub fn refuse_video_memory(&mut self, id: u64) {
        let Some(cover) = self.covers.values_mut().find(|c| c.id == id) else {
            return;
        };
        if let State::Ready(src) = &mut cover.state {
            src.refuse_video_memory();
        }
        let pixels = |f: &Frame| matches!(f.data, crate::frame::FrameData::Cpu(_));
        if !cover.current.as_ref().is_some_and(pixels) {
            cover.current = None;
        }
    }

    /// When cover `id` will have a new frame to show: zero if it has one now, `None`
    /// if its picture won't change (a still, a video that ended, an error, no such
    /// cover). The shim repaints by this instead of at a fixed rate. A missing file is
    /// looked for again by the next resolve, so that one is due when the retry is.
    pub fn next_frame_in(&mut self, id: u64, now: Instant) -> Option<Duration> {
        let cover = self.covers.values_mut().find(|c| c.id == id)?;
        match &mut cover.state {
            // picked up by the next resolve; until then look again shortly
            State::Opening { loader, .. } => Some(if loader.finished() {
                Duration::ZERO
            } else {
                OPENING_STEP
            }),
            State::Ready(src) => src.next_frame_in(now),
            State::Missing { retry_at } => Some(retry_at.saturating_duration_since(now)),
            State::Failed => None,
        }
    }

    /// End of frame: close covers nobody has shown for a while.
    pub fn end_frame(&mut self, now: Instant) {
        let old = |c: &Cover| now.saturating_duration_since(c.last_used) >= EVICT_AFTER;
        if !self.covers.values().any(old) {
            return;
        }
        let default = self.default_play();
        let stale: Vec<PlayParams> = self
            .covers
            .iter()
            .filter(|(play, c)| **play != default && old(c))
            .map(|(play, _)| play.clone())
            .collect();
        for play in stale {
            if let Some(cover) = self.covers.remove(&play) {
                self.retire(cover);
            }
        }
    }

    /// Whether any live animated cover (video, GIF) exists. The shim uses this
    /// to decide whether to nudge Hyprland into new capture frames. Covers not
    /// shown for a while get evicted after EVICT_AFTER anyway.
    ///
    /// A cover that is still being opened counts too: its first frame only
    /// shows up on the next rendered frame, and on a static screen there
    /// wouldn't be one.
    pub fn animating(&self, now: Instant) -> bool {
        self.covers.values().any(|c| {
            let live = match &c.state {
                // a missing file may appear: on a still screen nothing else would look
                State::Opening { .. } | State::Missing { .. } => true,
                State::Ready(src) => src.kind() != crate::media::Kind::Still,
                _ => false,
            };
            live && now.saturating_duration_since(c.last_used) < EVICT_AFTER
        })
    }

    /// Live cover ids (the shim frees textures for ids not listed here).
    pub fn live_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.covers.values().map(|c| c.id)
    }

    pub fn len(&self) -> usize {
        self.covers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.covers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{CpuFrame, FrameData};
    use crate::media::Kind;
    use std::sync::Mutex;

    // Opens per path: the opener runs on loader threads and tests run in
    // parallel, so each test counts only its own paths.
    static OPENS: Mutex<Vec<String>> = Mutex::new(Vec::new());

    fn opens(path: &str) -> usize {
        OPENS.lock().unwrap().iter().filter(|p| *p == path).count()
    }

    struct Solid(u64);
    impl Source for Solid {
        fn kind(&self) -> Kind {
            Kind::Still
        }
        fn poll(&mut self, _: Instant) -> Result<Option<Frame>, MediaError> {
            self.0 += 1;
            Ok(Some(Frame {
                generation: self.0,
                data: FrameData::Cpu(CpuFrame::from_bgra(1, 1, vec![0; 4].into())),
            }))
        }
    }

    static TARGETS: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());
    static REFUSED: AtomicBool = AtomicBool::new(false);

    /// Records the sizes it is told; its next frame is always 40 ms away.
    struct Sized;
    impl Source for Sized {
        fn kind(&self) -> Kind {
            Kind::Video
        }
        fn poll(&mut self, _: Instant) -> Result<Option<Frame>, MediaError> {
            Ok(Some(Frame {
                generation: 1,
                data: FrameData::Cpu(CpuFrame::from_bgra(1, 1, vec![0; 4].into())),
            }))
        }
        fn set_target(&mut self, width: u32, height: u32) {
            TARGETS.lock().unwrap().push((width, height));
        }
        fn next_frame_in(&mut self, _: Instant) -> Option<Duration> {
            Some(Duration::from_millis(40))
        }
        fn refuse_video_memory(&mut self) {
            REFUSED.store(true, Ordering::SeqCst);
        }
    }

    /// Slow to open and to produce its first frame, like a big video.
    struct Slow {
        ready_at: Instant,
    }
    impl Source for Slow {
        fn kind(&self) -> Kind {
            Kind::Video
        }
        fn poll(&mut self, now: Instant) -> Result<Option<Frame>, MediaError> {
            if now < self.ready_at {
                return Ok(None);
            }
            Ok(Some(Frame {
                generation: 1,
                data: FrameData::Cpu(CpuFrame::from_bgra(1, 1, vec![0; 4].into())),
            }))
        }
    }

    fn opener(p: &PlayParams, _: &Settings) -> Result<Box<dyn Source>, MediaError> {
        let path = p.path.to_str().unwrap();
        OPENS.lock().unwrap().push(path.to_string());
        match path {
            "/missing" | "/missing-backoff" => Err(MediaError::Missing(path.into())),
            "/bad" => Err(MediaError::UnknownFormat(".xyz".into())),
            "/sized" => Ok(Box::new(Sized)),
            "/slow" => {
                std::thread::sleep(Duration::from_millis(150));
                Ok(Box::new(Slow {
                    ready_at: Instant::now() + Duration::from_millis(150),
                }))
            }
            _ => Ok(Box::new(Solid(0))),
        }
    }

    /// Resolve until the loader thread has delivered (or the state settled).
    fn settle(r: &mut Registry, p: &PlayParams) -> Option<u64> {
        let end = Instant::now() + Duration::from_secs(5);
        loop {
            r.begin_frame();
            let now = Instant::now();
            if let Some(v) = r.resolve(p, (0, 0), now) {
                return Some(v.frame.generation);
            }
            let opening = matches!(
                r.covers.get(p).map(|c| &c.state),
                Some(State::Opening { .. })
            );
            if !opening || Instant::now() > end {
                return None;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn play(path: &str) -> PlayParams {
        PlayParams {
            path: path.into(),
            speed: 1.0,
            looped: true,
        }
    }

    fn defaults() -> Settings {
        Settings {
            path_cover: "/default".into(),
            ..Settings::default()
        }
    }

    /// Registry with settings; the default cover (/default) is already prewarmed.
    fn reg() -> Registry {
        let mut r = Registry::with_opener(opener);
        r.set_settings(defaults(), None);
        r
    }

    #[test]
    fn a_file_replaced_in_place_is_opened_again_once_it_settles() {
        let dir = std::env::temp_dir().join(format!("nsc-reg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("replaced.png");
        std::fs::write(&file, b"one").unwrap();
        let path = file.to_str().unwrap();
        let p = play(path);

        let mut r = reg();
        settle(&mut r, &p).expect("cover loaded");
        let id = r.id_of(&p).unwrap();
        let t = Instant::now();

        // a look a second later: the same file, nothing happens
        r.begin_frame();
        r.resolve(&p, (0, 0), t + MISSING_RETRY);
        assert_eq!(opens(path), 1);

        // the file changes: not opened at the first look (it may still be written)
        std::fs::write(&file, b"another one").unwrap();
        r.begin_frame();
        assert!(r.resolve(&p, (0, 0), t + MISSING_RETRY * 2).is_some());
        assert_eq!(opens(path), 1);
        // changed again in between: still waiting
        std::fs::write(&file, b"another one, longer").unwrap();
        r.begin_frame();
        r.resolve(&p, (0, 0), t + MISSING_RETRY * 3);
        assert_eq!(opens(path), 1);
        // the same at two looks: opened again, under a new id, the old picture still up
        r.begin_frame();
        assert!(
            r.resolve(&p, (0, 0), t + MISSING_RETRY * 4).is_some(),
            "the old picture stays until the new one is ready"
        );
        settle(&mut r, &p).expect("cover loaded again");
        assert_eq!(opens(path), 2);
        assert_ne!(r.id_of(&p), Some(id), "the shim's textures of the old file");
        assert_eq!(r.len(), 2, "the default cover and this one");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn want_is_the_largest_size_asked_for_lately() {
        let t = Instant::now();
        let mut w = Want::default();
        assert_eq!(w.target(), (0, 0), "nothing asked: its own size");
        w.add((300, 200), t);
        w.add((800, 100), t);
        assert_eq!(w.target(), (800, 200));
        // the next stretch has the small window only, the large one still counts
        w.add((300, 200), t + Want::STRETCH);
        assert_eq!(w.target(), (800, 200));
        // one more and it is forgotten
        w.add((300, 200), t + Want::STRETCH * 2);
        assert_eq!(w.target(), (300, 200));
        // drawn at its own size somewhere
        w.add((0, 0), t + Want::STRETCH * 2);
        assert_eq!(w.target(), (0, 0));
        // unused for long: starts over
        w.add((10, 10), t + Want::STRETCH * 10);
        assert_eq!(w.target(), (10, 10));
    }

    #[test]
    fn source_is_told_the_size_and_asked_for_its_next_frame() {
        let mut r = reg();
        let p = play("/sized");
        settle(&mut r, &p).expect("cover loaded"); // at its own size
        assert!(
            TARGETS.lock().unwrap().is_empty(),
            "own size is the default"
        );

        // later, when the first resolve no longer counts
        let now = Instant::now() + Want::STRETCH * 2;
        r.begin_frame();
        let id = r.resolve(&p, (640, 360), now).unwrap().id;
        r.resolve(&p, (320, 180), now); // a second, smaller window in the same frame
        r.begin_frame();
        r.resolve(&p, (640, 360), now);
        assert_eq!(*TARGETS.lock().unwrap(), [(640, 360)], "told once");
        r.resolve(&p, (1280, 200), now);
        r.resolve(&p, (0, 0), now);
        assert_eq!(
            *TARGETS.lock().unwrap(),
            [(640, 360), (1280, 360), (0, 0)],
            "grows right away, a window at the cover's own size lifts the limit"
        );

        assert_eq!(r.next_frame_in(id, now), Some(Duration::from_millis(40)));
        assert_eq!(r.next_frame_in(id + 1000, now), None, "no such cover");
        assert_eq!(r.id_of(&p), Some(id));

        // a frame in video memory the compositor can't show: the source is told, by id
        r.refuse_video_memory(id + 1000);
        assert!(!REFUSED.load(Ordering::SeqCst));
        r.refuse_video_memory(id);
        assert!(REFUSED.load(Ordering::SeqCst));
        assert!(
            r.resolve(&p, (0, 0), now).is_some(),
            "a frame of pixels stays"
        );

        // a cover that is still being opened is looked at again shortly
        let slow = play("/slow");
        r.begin_frame();
        assert!(r.resolve(&slow, (0, 0), Instant::now()).is_none());
        let opening = r.id_of(&slow).expect("it exists already");
        assert_eq!(r.next_frame_in(opening, Instant::now()), Some(OPENING_STEP));
        // the default cover is a still
        let still = r.id_of(&play("/default")).unwrap();
        assert!(loaded(&r, &play("/default")));
        r.begin_frame();
        r.resolve(&play("/default"), (0, 0), Instant::now());
        assert_eq!(r.next_frame_in(still, Instant::now()), None);
    }

    #[test]
    fn default_cover_is_prewarmed_and_kept() {
        let r = reg();
        assert_eq!(r.len(), 1, "default cover is opened right away");
        let mut r = r;
        r.end_frame(Instant::now() + EVICT_AFTER * 3);
        assert_eq!(
            r.len(),
            1,
            "and never evicted, even if unused for a long time"
        );
    }

    #[test]
    fn shared_cover_polled_once_per_frame() {
        let mut r = reg();
        let now = Instant::now();
        let g1 = settle(&mut r, &play("/a")).expect("cover loaded");
        let g2 = r
            .resolve(&play("/a"), (0, 0), now)
            .unwrap()
            .frame
            .generation;
        assert_eq!(g1, g2, "second window in same frame reuses the frame");
        r.begin_frame();
        assert_eq!(
            r.resolve(&play("/a"), (0, 0), now)
                .unwrap()
                .frame
                .generation,
            g1 + 1
        );
        assert_eq!(r.len(), 2, "/default + /a");
    }

    #[test]
    fn opening_does_not_block_the_render_path() {
        let mut r = reg();
        let p = play("/slow");
        r.begin_frame();
        let t = Instant::now();
        assert!(
            r.resolve(&p, (0, 0), Instant::now()).is_none(),
            "nothing to draw yet"
        );
        assert!(
            t.elapsed() < Duration::from_millis(50),
            "resolve returned in {:?}, it must not wait for the open",
            t.elapsed()
        );
        assert!(
            r.animating(Instant::now()),
            "a cover being opened keeps the shim asking for frames"
        );
        assert_eq!(settle(&mut r, &p), Some(1), "first frame arrives later");
    }

    #[test]
    fn warmed_cover_is_ready_on_first_resolve() {
        let mut r = reg();
        let p = play("/warm");
        r.warm(&p, Instant::now());
        r.warm(&p, Instant::now());
        assert!(loaded(&r, &p), "loader finished");
        assert_eq!(opens("/warm"), 1, "warming twice opens once");
        r.begin_frame();
        assert!(
            r.resolve(&p, (0, 0), Instant::now()).is_some(),
            "the first capture after warming already has a frame"
        );
    }

    /// Wait until the cover's loader thread is done (its result is ready to be
    /// picked up by the next resolve), with a timeout instead of a fixed sleep.
    fn loaded(r: &Registry, p: &PlayParams) -> bool {
        let end = Instant::now() + Duration::from_secs(5);
        loop {
            let done = match r.covers.get(p).map(|c| &c.state) {
                Some(State::Opening { loader, .. }) => loader.finished(),
                Some(_) => true,
                None => false,
            };
            if done || Instant::now() > end {
                return done;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn dropping_an_opening_cover_does_not_block() {
        let mut r = reg();
        assert!(
            loaded(&r, &play("/default")),
            "default cover out of the way"
        );
        r.begin_frame();
        assert!(r.resolve(&play("/slow"), (0, 0), Instant::now()).is_none());
        // a settings change drops every cover, including the one being opened
        // (the slow opener takes 150 ms): that must not wait for it
        let t = Instant::now();
        r.set_settings(
            Settings {
                speed: 3.0,
                ..defaults()
            },
            None,
        );
        assert!(
            t.elapsed() < Duration::from_millis(50),
            "set_settings took {:?}",
            t.elapsed()
        );
        assert_eq!(r.retired.len(), 1, "the loader is parked");

        // eviction the same way
        r.begin_frame();
        let now = Instant::now();
        assert!(r.resolve(&play("/slow"), (0, 0), now).is_none());
        let t = Instant::now();
        r.end_frame(now + EVICT_AFTER * 2);
        assert!(t.elapsed() < Duration::from_millis(50));
        assert_eq!(r.retired.len(), 2);

        // finished loaders are reaped at the start of a frame
        let end = Instant::now() + Duration::from_secs(5);
        while !r.retired.iter().all(Loader::finished) && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(2));
        }
        r.begin_frame();
        assert!(r.retired.is_empty());
    }

    #[test]
    fn unload_joins_parked_loaders() {
        let mut r = reg();
        assert!(
            loaded(&r, &play("/default")),
            "default cover out of the way"
        );
        r.begin_frame();
        assert!(r.resolve(&play("/slow"), (0, 0), Instant::now()).is_none());
        r.set_settings(
            Settings {
                speed: 4.0,
                ..defaults()
            },
            None,
        );
        assert_eq!(r.retired.len(), 1);
        let t = Instant::now();
        drop(r); // plugin unload: the parked loader is joined, not leaked
        assert!(
            t.elapsed() >= Duration::from_millis(50),
            "drop waited for the loader ({:?})",
            t.elapsed()
        );
    }

    #[test]
    fn empty_path_is_silent() {
        let mut r = reg();
        let now = Instant::now();
        r.begin_frame();
        assert!(r.resolve(&play(""), (0, 0), now).is_none());
        assert!(r.notifier().pop().is_none());
    }

    #[test]
    fn errors_notify_once() {
        let mut r = reg();
        assert!(settle(&mut r, &play("/bad")).is_none());
        let now = Instant::now();
        for _ in 0..3 {
            r.begin_frame();
            assert!(r.resolve(&play("/bad"), (0, 0), now).is_none());
        }
        assert!(r.notifier().pop().is_some());
        assert!(r.notifier().pop().is_none());
    }

    #[test]
    fn settings_change_resets_everything() {
        let mut r = reg();
        let now = Instant::now();
        settle(&mut r, &play("/a")).expect("cover loaded");
        let id = r.resolve(&play("/a"), (0, 0), now).unwrap().id;
        let epoch = r.epoch();
        r.set_settings(
            Settings {
                speed: 2.0,
                ..defaults()
            },
            None,
        );
        assert_eq!(r.epoch(), epoch + 1);
        assert_eq!(
            r.len(),
            1,
            "everything reset, only the default cover is prewarmed again"
        );
        settle(&mut r, &play("/a")).expect("cover loaded");
        assert_ne!(
            r.resolve(&play("/a"), (0, 0), now).unwrap().id,
            id,
            "new cover id after reset"
        );
    }

    #[test]
    fn same_settings_do_not_reset() {
        let mut r = reg();
        let e = r.epoch();
        r.set_settings(defaults(), None);
        assert_eq!(r.epoch(), e);
    }

    #[test]
    fn unused_covers_are_evicted() {
        let mut r = reg();
        let now = Instant::now();
        r.begin_frame();
        r.resolve(&play("/a"), (0, 0), now);
        r.end_frame(now + Duration::from_secs(5));
        assert_eq!(r.len(), 2);
        r.end_frame(now + EVICT_AFTER + Duration::from_secs(1));
        assert_eq!(r.len(), 1, "/a evicted, /default kept");
    }

    #[test]
    fn missing_file_is_retried_with_backoff() {
        let mut r = reg();
        let p = play("/missing-backoff");
        assert!(settle(&mut r, &p).is_none());
        let now = Instant::now();
        for i in 0..10 {
            r.begin_frame();
            r.resolve(&p, (0, 0), now + Duration::from_millis(i * 10));
        }
        // first open only; retries happen after the backoff and only if the file appears
        assert_eq!(opens("/missing-backoff"), 1);
        assert!(r.notifier().pop().is_some(), "missing file reported once");
        assert!(r.notifier().pop().is_none());
    }
}
