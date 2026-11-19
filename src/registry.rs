//! Cover registry: one source per unique (file, speed, loop), no matter how
//! many windows show it.
//!
//! Lifecycle:
//! - global settings change: everything is reset and the epoch is bumped (the
//!   shim drops its textures on a new epoch);
//! - while a cover is shown its file is looked at once per second, no stat() on every
//!   frame (see `look`): a missing file is opened once it is there, a file replaced on
//!   disk once it has stopped changing;
//! - failed to open: one notification, no retries until the file or the config changes;
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
//! `next_frame_in` has the shim look again shortly, so it picks the cover up as
//! soon as it lands. Closing doesn't happen there either (see `Retired`).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use crate::config::{ConfigError, PlayParams, Settings};
use crate::frame::{Frame, FrameData};
use crate::media::{self, Kind, MediaError, Source};
use crate::notify::Notifier;
use crate::worker::Worker;

/// Opens a source for the given parameters (replaced in tests).
pub type Opener = fn(&PlayParams, &Settings) -> Result<Box<dyn Source>, MediaError>;

/// How often the file of a shown cover is looked at (see `Registry::look`).
const LOOK_EVERY: Duration = Duration::from_secs(1);
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

/// Plain pixels, as opposed to a frame in the decoder's video memory.
fn in_memory(f: &Frame) -> bool {
    matches!(f.data, FrameData::Cpu(_))
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
    Missing,
    Failed,
}

/// What goes away with a cover or its source, and can take a while: a loader nobody
/// waits for any more (the open can't be interrupted: decoding a whole GIF, setting up a
/// decoder), a video source (dropping it joins its decode thread, which may be in the
/// middle of a frame), a GIF's frames. Waiting for that in the render hook would stall
/// the compositor, so loaders are parked here and sources are closed on a thread of
/// their own; all of it is reaped once finished, and what is left is joined when the
/// registry itself goes away, on unload.
#[derive(Default)]
struct Retired {
    loaders: Vec<Loader>,
    closing: Vec<Worker>,
}

impl Retired {
    /// What a cover's state holds.
    fn state(&mut self, state: State) {
        match state {
            State::Opening { loader, .. } => self.loader(loader),
            State::Ready(src) => self.source(src),
            State::Missing | State::Failed => {}
        }
    }

    /// Parked until its thread has finished; what it opened is closed like any source (it
    /// stays in the loader's channel until taken).
    fn loader(&mut self, loader: Loader) {
        loader.cancel();
        if !loader.finished() {
            self.loaders.push(loader);
            return;
        }
        // sent before its thread finished, so it is there if it was sent at all
        if let Ok(Ok((src, _))) = loader.rx.try_recv() {
            self.source(src);
        }
    }

    fn source(&mut self, src: Box<dyn Source>) {
        // a still only has its picture left: nothing to wait for
        if src.kind() == Kind::Still {
            return;
        }
        // started from Hyprland's thread: see worker.rs. A thread that can't be started
        // drops the job, and the source with it, right here.
        if let Ok(w) = Worker::spawn(c"noshare-close", move || drop(src)) {
            self.closing.push(w);
        }
    }

    /// Lets go of what has finished: joining it doesn't wait.
    fn reap(&mut self) {
        for loader in std::mem::take(&mut self.loaders) {
            self.loader(loader);
        }
        self.closing.retain(|w| !w.is_finished());
    }
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
    /// A source of this cover was opened before: the shim may still hold textures of it
    /// under `id`, so the next one gets a new id (see `resolve`).
    had_source: bool,
    /// The id before that: the shim asks by the id it last drew the cover with, and on
    /// another monitor that frame may have come before the new id. A box asked about by it
    /// still shows the earlier picture: it is due now (see `next_frame_in`).
    earlier_id: Option<u64>,
    /// the file as it was when it was opened
    stamp: Option<Stamp>,
    /// a newer file seen at the last look; opened once it is the same at the next one
    /// (a file that is still being written keeps changing)
    changed: Option<Stamp>,
    /// when the file is looked at next
    look_at: Instant,
}

impl Cover {
    fn look_due(&self, now: Instant) -> bool {
        !matches!(self.state, State::Opening { .. }) && now >= self.look_at
    }
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
    retired: Retired,
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
            retired: Retired::default(),
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
        // sources are closed and loaders parked off the render path (see `Retired`)
        for (_, cover) in std::mem::take(&mut self.covers) {
            self.retired.state(cover.state);
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
        self.retired.reap();
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
        self.look(play, now);

        let cover = self.covers.get_mut(play)?;
        cover.last_used = now;
        cover.want.add(size, now);

        if let State::Opening { loader, retry } = &cover.state {
            let retry = *retry;
            let next = match loader.rx.try_recv() {
                Err(TryRecvError::Empty) => None,
                Ok(Ok((mut src, frame))) => {
                    // first time on screen: restart its timeline and poll it
                    // right below, so the clock starts with this frame
                    src.shown();
                    // The shim keeps textures by id and frame generation, and a new source
                    // counts its generations (and its video memory buffers) from the start:
                    // after an earlier one it gets a new id. The id changes only here, with
                    // the picture: the shim asks about the cover by the id it last drew.
                    if cover.had_source {
                        cover.earlier_id = Some(cover.id);
                        cover.id = self.next_id;
                        self.next_id += 1;
                        // the earlier picture, until the first frame is there, under a
                        // generation no source hands out
                        if let Some(f) = &mut cover.current {
                            f.generation = 0;
                        }
                    }
                    cover.had_source = true;
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
                    cover.look_at = now + LOOK_EVERY;
                    Some(State::Missing)
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
                        let old = std::mem::replace(&mut cover.state, State::Failed);
                        self.retired.state(old);
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

    /// Looks at the file of a shown cover, at most once per `LOOK_EVERY`: from a capture
    /// frame (`resolve`), and on a still screen, where none comes, from the shim's question
    /// when the cover changes next (`next_frame_in`). A missing file that is there now is
    /// opened; so is a file replaced on disk once it is the same at two looks in a row, also
    /// after a failed open, which may have read a file that was only half there. The cover
    /// keeps its id and its picture until the new source is there (see `resolve`).
    fn look(&mut self, play: &PlayParams, now: Instant) {
        let Some(cover) = self.covers.get_mut(play) else {
            return;
        };
        if !cover.look_due(now) {
            return;
        }
        cover.look_at = now + LOOK_EVERY;
        let retry = if matches!(cover.state, State::Missing) {
            if !play.path.exists() {
                return;
            }
            cover.stamp = stamp_of(&play.path);
            true
        } else {
            let seen = stamp_of(&play.path);
            if seen.is_none() || seen == cover.stamp {
                cover.changed = None;
                return;
            }
            if seen != cover.changed {
                cover.changed = seen;
                return;
            }
            (cover.changed, cover.stamp) = (None, seen);
            // the old picture stays up until the new one is there, if it is plain pixels
            if !cover.current.as_ref().is_some_and(in_memory) {
                cover.current = None;
            }
            cover.target = (0, 0);
            false
        };
        let loader = Loader::spawn(
            self.opener,
            play.clone(),
            self.settings.clone().unwrap_or_default(),
        );
        let old = std::mem::replace(&mut cover.state, State::Opening { loader, retry });
        self.retired.state(old);
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
                had_source: false,
                earlier_id: None,
                stamp: stamp_of(&play.path),
                changed: None,
                look_at: now + LOOK_EVERY,
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
        if !cover.current.as_ref().is_some_and(in_memory) {
            cover.current = None;
        }
    }

    /// When cover `id` will have something new to show: zero if it has it now. The shim
    /// asks this about every cover on a captured monitor and repaints by it instead of at
    /// a fixed rate. A cover whose picture doesn't change by itself (a still, a video that
    /// ended, a missing or broken file) is due when its file is looked at next: on a still
    /// screen this question is what looks at it, and only a new picture makes it zero.
    /// Asked by the id the cover had before its new source, it is zero at once: that box is
    /// on a monitor not drawn since the source came, so it shows the earlier picture, and
    /// its repaint draws the new one under the new id. `None`: no such cover.
    pub fn next_frame_in(&mut self, id: u64, now: Instant) -> Option<Duration> {
        let asked = |c: &Cover| c.id == id || c.earlier_id == Some(id);
        let (play, cover) = self.covers.iter().find(|(_, c)| asked(c))?;
        if cover.look_due(now) {
            let play = play.clone();
            self.look(&play, now); // keeps the id (see `resolve`)
        }
        let cover = self.covers.values_mut().find(|c| asked(c))?;
        // the shim asks about what is on a captured monitor: shown, on a still one too
        cover.last_used = now;
        if cover.id != id {
            return Some(Duration::ZERO);
        }
        let look = cover.look_at.saturating_duration_since(now);
        Some(match &mut cover.state {
            // picked up by the next resolve; until then look again shortly
            State::Opening { loader, .. } => {
                if loader.finished() {
                    Duration::ZERO
                } else {
                    OPENING_STEP
                }
            }
            State::Ready(src) => src.next_frame_in(now).map_or(look, |next| next.min(look)),
            State::Missing | State::Failed => look,
        })
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
                self.retired.state(cover.state);
            }
        }
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
    use crate::frame::CpuFrame;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    // Opens per path: the opener runs on loader threads and tests run in
    // parallel, so each test counts only its own paths.
    static OPENS: Mutex<Vec<String>> = Mutex::new(Vec::new());

    fn opens(path: &str) -> usize {
        OPENS.lock().unwrap().iter().filter(|p| *p == path).count()
    }

    /// A 1x1 frame whose pixel is `value`.
    fn pixel(generation: u64, value: u8) -> Frame {
        Frame {
            generation,
            data: FrameData::Cpu(CpuFrame::from_bgra(
                1,
                1,
                vec![value, value, value, 255].into(),
            )),
        }
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

    /// A real file opened as a picture, the way `StillSource` is: its one frame is handed
    /// out once, as generation 1. The pixel is the file's first byte.
    struct Picture(Option<Frame>);
    impl Source for Picture {
        fn kind(&self) -> Kind {
            Kind::Still
        }
        fn poll(&mut self, _: Instant) -> Result<Option<Frame>, MediaError> {
            Ok(self.0.take())
        }
    }

    static CLOSED: AtomicUsize = AtomicUsize::new(0);
    const STICKY: Duration = Duration::from_millis(200);

    /// Takes a while to close, like a video joining its decode thread. With `fails` its
    /// polls after the first one (the loader's) fail.
    struct Sticky {
        fails: bool,
        polls: u64,
    }
    impl Source for Sticky {
        fn kind(&self) -> Kind {
            Kind::Video
        }
        fn poll(&mut self, _: Instant) -> Result<Option<Frame>, MediaError> {
            self.polls += 1;
            if self.fails && self.polls > 1 {
                return Err(MediaError::Video {
                    path: "sticky".into(),
                    reason: "broke".into(),
                });
            }
            Ok(Some(pixel(self.polls, 0)))
        }
    }
    impl Drop for Sticky {
        fn drop(&mut self) {
            std::thread::sleep(STICKY);
            CLOSED.fetch_add(1, Ordering::SeqCst);
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
            "/sticky" | "/sticky-again" => Ok(Box::new(Sticky {
                fails: false,
                polls: 0,
            })),
            "/sticky-fails" => Ok(Box::new(Sticky {
                fails: true,
                polls: 0,
            })),
            "/slow" => {
                std::thread::sleep(Duration::from_millis(150));
                Ok(Box::new(Slow {
                    ready_at: Instant::now() + Duration::from_millis(150),
                }))
            }
            // files in a `scratch` directory, opened as media::open would: a file that isn't
            // there is missing, one that starts with `!` is broken, a clip is sticky
            _ if p.path.starts_with(std::env::temp_dir()) => match std::fs::read(&p.path) {
                Err(_) => Err(MediaError::Missing(path.into())),
                Ok(b) if b.first() == Some(&b'!') => Err(MediaError::Open {
                    path: path.into(),
                    reason: "broken".into(),
                }),
                Ok(_) if path.ends_with(".mp4") => Ok(Box::new(Sticky {
                    fails: false,
                    polls: 0,
                })),
                Ok(b) => Ok(Box::new(Picture(Some(pixel(1, b[0]))))),
            },
            _ => Ok(Box::new(Solid(0))),
        }
    }

    /// An empty directory of this test's own.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("nsc-reg-{}-{name}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// What the shim shows of the covers: it keeps a texture per cover id and makes it
    /// again only when a frame's generation differs from the one it has (`textureFor`).
    #[derive(Default)]
    struct Textures(HashMap<u64, (u64, u8)>);

    impl Textures {
        /// The pixel the shim draws for what `resolve` returned.
        fn show(&mut self, view: Option<CoverView<'_>>) -> Option<u8> {
            let view = view?;
            let FrameData::Cpu(f) = &view.frame.data else {
                panic!("pixels expected");
            };
            let texture = self.0.entry(view.id).or_insert((u64::MAX, 0));
            if texture.0 != view.frame.generation {
                *texture = (view.frame.generation, f.pixels[0]);
            }
            Some(texture.1)
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
        let dir = scratch("replaced");
        let file = dir.join("replaced.png");
        std::fs::write(&file, b"one").unwrap();
        let path = file.to_str().unwrap();
        let p = play(path);
        let mut shown = Textures::default();

        let mut r = reg();
        settle(&mut r, &p).expect("cover loaded");
        let id = r.id_of(&p).unwrap();
        let t = Instant::now();
        let mut frame = |r: &mut Registry, at: Instant| {
            r.begin_frame();
            shown.show(r.resolve(&p, (0, 0), at))
        };

        // a look a second later: the same file, nothing happens
        assert_eq!(frame(&mut r, t + LOOK_EVERY), Some(b'o'));
        assert_eq!(opens(path), 1);

        // the file changes: not opened at the first look (it may still be written)
        std::fs::write(&file, b"another one").unwrap();
        assert_eq!(frame(&mut r, t + LOOK_EVERY * 2), Some(b'o'));
        assert_eq!(opens(path), 1);
        // changed again in between: still waiting
        std::fs::write(&file, b"the last one").unwrap();
        frame(&mut r, t + LOOK_EVERY * 3);
        assert_eq!(opens(path), 1);
        // the same at two looks: opened again, the old picture stays until the new one is
        // there (or it is already)
        assert!(frame(&mut r, t + LOOK_EVERY * 4).is_some());
        assert!(loaded(&r, &p));
        // the new picture, under a new id: the shim's texture of the old file, which had
        // the same generation, is not shown for it
        assert_eq!(frame(&mut r, t + LOOK_EVERY * 4), Some(b't'));
        assert_eq!(opens(path), 2);
        assert_ne!(r.id_of(&p), Some(id));
        assert_eq!(r.len(), 2, "the default cover and this one");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_picture_replaced_on_a_still_screen_shows_up() {
        let dir = scratch("still-screen");
        let file = dir.join("cover.png");
        std::fs::write(&file, b"red").unwrap();
        let path = file.to_str().unwrap();
        let p = play(path);
        let mut shown = Textures::default();

        let mut r = reg();
        settle(&mut r, &p).expect("cover loaded");
        let t = Instant::now();
        r.begin_frame();
        assert_eq!(shown.show(r.resolve(&p, (0, 0), t)), Some(b'r'));
        let id = r.id_of(&p).unwrap();

        // From here on nothing renders: the shim only asks when the cover changes next,
        // and that is never now while the picture is the same.
        std::fs::write(&file, b"green").unwrap();
        let first = r.next_frame_in(id, t + LOOK_EVERY).expect("looked at");
        assert!(first > Duration::ZERO && first <= LOOK_EVERY, "{first:?}");
        assert_eq!(
            opens(path),
            1,
            "a file that just changed may still be written"
        );
        let opening = r.next_frame_in(id, t + LOOK_EVERY * 2).expect("looked at");
        assert!(opening <= OPENING_STEP, "{opening:?}: being opened again");
        assert_eq!(r.id_of(&p), Some(id), "the id the shim asks by stays");
        assert!(loaded(&r, &p));
        assert_eq!(opens(path), 2);
        assert_eq!(
            r.next_frame_in(id, t + LOOK_EVERY * 2),
            Some(Duration::ZERO),
            "the new picture is there: repaint"
        );

        // the repaint
        r.begin_frame();
        assert_eq!(
            shown.show(r.resolve(&p, (0, 0), t + LOOK_EVERY * 2)),
            Some(b'g')
        );
        let new = r.id_of(&p).unwrap();
        assert_ne!(new, id);
        let wait = r.next_frame_in(new, t + LOOK_EVERY * 2).unwrap();
        assert!(wait > Duration::ZERO, "and then nothing to repaint");
        assert_eq!(
            r.next_frame_in(id, t + LOOK_EVERY * 2),
            Some(Duration::ZERO),
            "asked by the old id (the cover's box on a monitor not drawn since): repaint it"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_replaced_picture_shows_up_on_every_captured_monitor() {
        let dir = scratch("two-monitors");
        let file = dir.join("cover.png");
        std::fs::write(&file, b"red").unwrap();
        let p = play(file.to_str().unwrap());
        // the shim's textures are per cover id; each monitor draws its box by its own frame
        let (mut a, mut b) = (Textures::default(), Textures::default());

        let mut r = reg();
        settle(&mut r, &p).expect("cover loaded");
        let t = Instant::now();
        for shown in [&mut a, &mut b] {
            r.begin_frame();
            assert_eq!(shown.show(r.resolve(&p, (0, 0), t)), Some(b'r'));
        }
        let id = r.id_of(&p).unwrap(); // both monitors' boxes have it

        std::fs::write(&file, b"green").unwrap();
        r.next_frame_in(id, t + LOOK_EVERY);
        r.next_frame_in(id, t + LOOK_EVERY * 2);
        assert!(loaded(&r, &p));
        // monitor A has a capture frame of its own first and takes the new picture
        r.begin_frame();
        assert_eq!(
            a.show(r.resolve(&p, (0, 0), t + LOOK_EVERY * 2)),
            Some(b'g')
        );
        let new = r.id_of(&p).unwrap();
        assert_ne!(new, id);
        // B's box, still asked about by the old id, shows the old picture: repaint it
        let soon = t + LOOK_EVERY * 2 + OPENING_STEP;
        assert_eq!(r.next_frame_in(id, soon), Some(Duration::ZERO));
        r.begin_frame();
        assert_eq!(b.show(r.resolve(&p, (0, 0), soon)), Some(b'g'));
        // from then on both boxes have the new id, and the picture is the same
        assert!(
            r.next_frame_in(new, soon)
                .is_some_and(|d| d > Duration::ZERO)
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_broken_file_fixed_on_a_still_screen_shows_up() {
        let dir = scratch("fixed");
        let file = dir.join("cover.png");
        std::fs::write(&file, b"!half").unwrap();
        let p = play(file.to_str().unwrap());
        let mut shown = Textures::default();

        let mut r = reg();
        assert!(settle(&mut r, &p).is_none());
        assert!(r.notifier().pop().is_some(), "reported");
        let id = r.id_of(&p).unwrap();
        let t = Instant::now();
        assert!(r.next_frame_in(id, t).is_some_and(|d| d > Duration::ZERO));

        std::fs::write(&file, b"whole picture").unwrap();
        r.next_frame_in(id, t + LOOK_EVERY);
        r.next_frame_in(id, t + LOOK_EVERY * 2);
        assert!(loaded(&r, &p));
        assert_eq!(
            r.next_frame_in(id, t + LOOK_EVERY * 2),
            Some(Duration::ZERO)
        );
        r.begin_frame();
        assert_eq!(
            shown.show(r.resolve(&p, (0, 0), t + LOOK_EVERY * 2)),
            Some(b'w')
        );
        assert!(r.notifier().pop().is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_file_is_looked_for_without_repainting() {
        let dir = scratch("appears");
        let file = dir.join("later.png");
        let path = file.to_str().unwrap();
        let p = play(path);
        let mut shown = Textures::default();

        let mut r = reg();
        assert!(settle(&mut r, &p).is_none());
        let id = r.id_of(&p).unwrap();
        let t = Instant::now();
        // looked for once a second, and never "repaint now" for it
        for i in 1..=3 {
            let wait = r.next_frame_in(id, t + LOOK_EVERY * i).unwrap();
            assert!(wait > Duration::ZERO, "look {i}: {wait:?}");
        }
        assert_eq!(opens(path), 1);

        // it appears: the next look opens it, and the repaint comes once it is open
        std::fs::write(&file, b"here").unwrap();
        r.next_frame_in(id, t + LOOK_EVERY * 4);
        assert!(loaded(&r, &p));
        assert_eq!(opens(path), 2);
        assert_eq!(
            r.next_frame_in(id, t + LOOK_EVERY * 4),
            Some(Duration::ZERO)
        );
        r.begin_frame();
        assert_eq!(
            shown.show(r.resolve(&p, (0, 0), t + LOOK_EVERY * 4)),
            Some(b'h')
        );
        assert!(r.notifier().pop().is_some(), "missing, reported once");
        assert!(r.notifier().pop().is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Runs `f` and says whether it was quick: well under a sticky source's close.
    fn quick(f: impl FnOnce()) -> bool {
        let t = Instant::now();
        f();
        t.elapsed() < STICKY / 4
    }

    #[test]
    fn sources_are_closed_off_the_render_path() {
        let dir = scratch("sticky");
        let file = dir.join("clip.mp4");
        std::fs::write(&file, b"one").unwrap();
        let clip = play(file.to_str().unwrap());
        let mut r = reg();

        // a source that fails: dropped by the resolve that finds out
        let fails = play("/sticky-fails");
        r.warm(&fails, Instant::now());
        assert!(loaded(&r, &fails));
        r.begin_frame();
        assert!(quick(|| assert!(
            r.resolve(&fails, (0, 0), Instant::now()).is_none()
        )));

        // a file replaced on disk: the old source goes when the new one is being opened
        settle(&mut r, &clip).expect("clip loaded");
        std::fs::write(&file, b"two!").unwrap();
        let t = Instant::now();
        for i in 1..=2 {
            r.begin_frame();
            assert!(quick(|| {
                r.resolve(&clip, (0, 0), t + LOOK_EVERY * i);
            }));
        }
        assert!(loaded(&r, &clip));
        assert_eq!(opens(file.to_str().unwrap()), 2);

        // a settings change
        settle(&mut r, &play("/sticky")).expect("loaded");
        assert!(quick(|| r.set_settings(
            Settings {
                speed: 5.0,
                ..defaults()
            },
            None
        )));

        // eviction
        settle(&mut r, &play("/sticky-again")).expect("loaded");
        assert!(quick(|| r.end_frame(Instant::now() + EVICT_AFTER * 2)));
        r.begin_frame(); // reaps, without waiting either
        assert_eq!(r.len(), 1, "only the default cover is left");

        // unload: everything is closed by the time the registry is gone (the failed one,
        // the clip before and after, /sticky and /sticky-again)
        drop(r);
        assert_eq!(CLOSED.load(Ordering::SeqCst), 5);

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
        // the default cover is a still: nothing new until its file is looked at again
        let still = r.id_of(&play("/default")).unwrap();
        assert!(loaded(&r, &play("/default")));
        r.begin_frame();
        let now = Instant::now();
        r.resolve(&play("/default"), (0, 0), now);
        let wait = r.next_frame_in(still, now).unwrap();
        assert!(wait > Duration::ZERO && wait <= LOOK_EVERY, "{wait:?}");
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
        assert_eq!(
            r.next_frame_in(r.id_of(&p).unwrap(), Instant::now()),
            Some(OPENING_STEP),
            "a cover being opened has the shim look again shortly"
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
        assert_eq!(r.retired.loaders.len(), 1, "the loader is parked");

        // eviction the same way
        r.begin_frame();
        let now = Instant::now();
        assert!(r.resolve(&play("/slow"), (0, 0), now).is_none());
        let t = Instant::now();
        r.end_frame(now + EVICT_AFTER * 2);
        assert!(t.elapsed() < Duration::from_millis(50));
        assert_eq!(r.retired.loaders.len(), 2);

        // finished loaders are reaped at the start of a frame
        let end = Instant::now() + Duration::from_secs(5);
        while !r.retired.loaders.iter().all(Loader::finished) && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(2));
        }
        r.begin_frame();
        assert!(r.retired.loaders.is_empty());
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
        assert_eq!(r.retired.loaders.len(), 1);
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
        r.resolve(&play("/b"), (0, 0), now);
        r.end_frame(now + Duration::from_secs(5));
        assert_eq!(r.len(), 3);
        // /b is on a captured monitor that doesn't repaint: the shim keeps asking about it
        let b = r.id_of(&play("/b")).unwrap();
        r.next_frame_in(b, now + EVICT_AFTER);
        r.end_frame(now + EVICT_AFTER + Duration::from_secs(1));
        assert_eq!(r.len(), 2, "/a evicted, /b and /default kept");
        assert!(r.id_of(&play("/b")).is_some());
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
