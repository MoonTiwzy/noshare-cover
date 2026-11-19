//! Decoders. Software ones are either built in (rav1d) or loaded from the system
//! at runtime (openh264, libvpx); hardware ones (VA-API and NVDEC) also go through
//! dlopen, so the plugin has no hard dependencies and doesn't crash when a library is missing.

#[cfg(feature = "cpu-av1")]
pub mod av1;
#[cfg(feature = "cpu-h264")]
pub mod h264;
#[cfg(feature = "nvdec")]
pub mod nvdec;
#[cfg(all(feature = "vaapi", target_os = "linux"))]
pub mod vaapi;
#[cfg(feature = "cpu-vpx")]
pub mod vpx;

/// Packets a CPU decoder refused, one after another, with no picture in between. A decoder
/// skips a broken packet (the next keyframe fixes everything) and reports the stream as
/// undecodable once the run gets too long (libvpx: a much longer run once the pass has
/// shown pictures, see [`Refusals::shown`]).
///
/// A loop doesn't end the run when the pass showed nothing: a clip shorter than the limit
/// that never decodes must still reach it. A pass that did show pictures starts over:
/// what its tail refused says nothing about the next pass, which begins at a keyframe.
#[cfg_attr(not(any(feature = "cpu-h264", feature = "cpu-vpx")), allow(dead_code))]
#[derive(Debug, Default)]
pub(crate) struct Refusals {
    /// packets refused in a row
    run: u32,
    /// keyframes among them
    keyframes: u32,
    /// a picture came out since the last reset
    shown: bool,
}

#[cfg_attr(not(any(feature = "cpu-h264", feature = "cpu-vpx")), allow(dead_code))]
impl Refusals {
    /// The decoder took a packet (`refused == false`) or not, and gave out a picture or not.
    pub(crate) fn note(&mut self, refused: bool, keyframe: bool, picture: bool) {
        self.shown |= picture;
        if refused && !picture {
            self.run += 1;
            self.keyframes += u32::from(keyframe);
        } else {
            (self.run, self.keyframes) = (0, 0);
        }
    }

    /// End of stream: the decoder gave out what it held. A picture ends the run; nothing
    /// says nothing about it (a clip that never decodes ends this way every time).
    pub(crate) fn flushed(&mut self, picture: bool) {
        if picture {
            self.note(false, false, true);
        }
    }

    /// The pass showed no picture and ends refusing: there is nothing to wait for, a loop
    /// would only refuse it all again.
    #[cfg_attr(not(feature = "cpu-vpx"), allow(dead_code))]
    pub(crate) fn blank_pass(&self) -> bool {
        self.run > 0 && !self.shown
    }

    /// The decoder was reset to go through the file again.
    pub(crate) fn reset(&mut self) {
        if std::mem::take(&mut self.shown) {
            (self.run, self.keyframes) = (0, 0);
        }
    }

    pub(crate) fn run(&self) -> u32 {
        self.run
    }

    /// A picture came out in this pass.
    #[cfg_attr(not(feature = "cpu-vpx"), allow(dead_code))]
    pub(crate) fn shown(&self) -> bool {
        self.shown
    }

    #[cfg_attr(not(feature = "cpu-vpx"), allow(dead_code))]
    pub(crate) fn keyframes(&self) -> u32 {
        self.keyframes
    }
}

#[cfg(all(
    test,
    any(feature = "cpu-av1", feature = "cpu-h264", feature = "cpu-vpx")
))]
pub(crate) mod testing {
    //! Small clips made by ffmpeg on the CPU, for tests of the CPU decoders.
    use std::path::{Path, PathBuf};

    /// A file made by ffmpeg in a directory of its own, removed when dropped.
    pub struct Clip(PathBuf);

    impl Clip {
        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Clip {
        fn drop(&mut self) {
            if let Some(dir) = self.0.parent() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    /// `ffmpeg -f lavfi -i <source> <args> <name>` in the temporary directory. `None`, so
    /// the test is skipped, when ffmpeg or the encoder isn't there.
    pub fn ffmpeg(name: &str, source: &str, args: &[&str]) -> Option<Clip> {
        let dir = std::env::temp_dir().join(format!("nsc-decode-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).ok()?;
        let clip = Clip(dir.join(name));
        let out = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
            ])
            .arg(source)
            .args(args)
            .arg(clip.path())
            .output();
        match out {
            Ok(o) if o.status.success() => Some(clip),
            Ok(o) => {
                eprintln!(
                    "{name}: ffmpeg failed, skipping: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
                None
            }
            Err(e) => {
                eprintln!("{name}: no ffmpeg ({e}), skipping");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Refusals;

    #[test]
    fn a_run_of_refusals_ends_with_a_picture_or_an_accepted_packet() {
        let mut r = Refusals::default();
        r.note(true, true, false);
        r.note(true, false, false);
        assert_eq!((r.run(), r.keyframes()), (2, 1));
        // taken without a picture (a hidden frame): the run is over all the same
        r.note(false, false, false);
        assert_eq!((r.run(), r.keyframes()), (0, 0));
        r.note(true, false, false);
        // refused, yet a picture came out (one the decoder held back)
        r.note(true, false, true);
        assert_eq!(r.run(), 0);
    }

    #[test]
    fn a_pass_that_showed_pictures_starts_the_next_one_clean() {
        let mut r = Refusals::default();
        assert!(!r.shown());
        r.note(false, true, true);
        assert!(r.shown());
        // a broken tail, then a loop into a head that takes a while to decode again
        for _ in 0..50 {
            r.note(true, false, false);
        }
        r.reset();
        assert!(!r.shown(), "a new pass has shown nothing yet");
        for _ in 0..50 {
            r.note(true, false, false);
        }
        assert_eq!(
            r.run(),
            50,
            "the tail of the pass that played doesn't count"
        );
    }

    #[test]
    fn a_clip_that_never_decodes_keeps_counting_across_loops() {
        let mut r = Refusals::default();
        for _ in 0..3 {
            r.note(true, true, false);
            for _ in 0..29 {
                r.note(true, false, false);
            }
            r.flushed(false);
            r.reset();
        }
        assert_eq!((r.run(), r.keyframes()), (90, 3));
        // once it played, the next loop starts over
        r.note(false, true, true);
        r.note(true, false, false);
        r.reset();
        assert_eq!(r.run(), 0);
        // so does a picture the end of the stream gave out
        r.note(true, false, false);
        r.flushed(true);
        assert_eq!(r.run(), 0);
    }

    #[test]
    fn a_blank_pass_is_one_that_showed_nothing_and_ends_refusing() {
        let mut r = Refusals::default();
        assert!(!r.blank_pass(), "nothing refused");
        r.note(true, true, false);
        r.flushed(false);
        assert!(r.blank_pass());
        // a picture before the refusals
        let mut r = Refusals::default();
        r.note(false, true, true);
        r.note(true, false, false);
        r.flushed(false);
        assert!(!r.blank_pass());
        // and in the next pass, which shows nothing
        r.reset();
        r.note(true, true, false);
        assert!(r.blank_pass());
        // refusals, then a picture the end of the stream gave out
        let mut r = Refusals::default();
        r.note(true, true, false);
        r.flushed(true);
        assert!(!r.blank_pass());
    }
}
