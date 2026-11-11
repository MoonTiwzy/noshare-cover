//! Playback clock: position in the clip, accounting for speed and looping.
//!
//! Time comes from the caller's monotonic clock (compositor frames), so all
//! functions are pure and testable without sleep.

use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct PlaybackClock {
    started: Option<Instant>,
    speed: f64,
    looped: bool,
}

impl PlaybackClock {
    pub fn new(speed: f64, looped: bool) -> Self {
        Self {
            started: None,
            speed: crate::config::sanitize_speed(speed),
            looped,
        }
    }

    /// Position at `now`. The first call starts the clock.
    ///
    /// `duration == None` means unknown length (live stream): just elapsed time
    /// scaled by speed. Without looping, the position clamps to the clip's last moment.
    pub fn position(&mut self, now: Instant, duration: Option<Duration>) -> Duration {
        let started = *self.started.get_or_insert(now);
        let raw = now.saturating_duration_since(started).mul_f64(self.speed);
        match duration {
            Some(d) if !d.is_zero() => {
                if self.looped {
                    Duration::from_nanos((raw.as_nanos() % d.as_nanos()) as u64)
                } else {
                    raw.min(d.saturating_sub(Duration::from_nanos(1)))
                }
            }
            _ => raw,
        }
    }

    pub fn restart(&mut self) {
        self.started = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    #[test]
    fn loops_over_duration() {
        let t0 = Instant::now();
        let mut c = PlaybackClock::new(1.0, true);
        assert_eq!(c.position(t0, Some(ms(1000))), ms(0));
        assert_eq!(c.position(t0 + ms(1500), Some(ms(1000))), ms(500));
    }

    #[test]
    fn speed_scales_time() {
        let t0 = Instant::now();
        let mut c = PlaybackClock::new(2.0, true);
        c.position(t0, None);
        assert_eq!(c.position(t0 + ms(300), None), ms(600));
    }

    #[test]
    fn no_loop_clamps_to_last_moment() {
        let t0 = Instant::now();
        let mut c = PlaybackClock::new(1.0, false);
        c.position(t0, Some(ms(1000)));
        assert_eq!(
            c.position(t0 + ms(5000), Some(ms(1000))),
            ms(1000) - Duration::from_nanos(1)
        );
    }

    #[test]
    fn restart_resets_origin() {
        let t0 = Instant::now();
        let mut c = PlaybackClock::new(1.0, true);
        c.position(t0, None);
        c.restart();
        assert_eq!(c.position(t0 + ms(700), None), ms(0));
    }
}
