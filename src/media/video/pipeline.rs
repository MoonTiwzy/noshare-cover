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
}

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
        }
    }

    pub fn info(&self) -> &VideoInfo {
        self.demux.info()
    }

    pub fn decoder_name(&self) -> &'static str {
        self.decoder.name()
    }

    pub fn next_frame(&mut self) -> PipeResult<Next> {
        // guard: a broken file must not spin the thread forever
        for _ in 0..4096 {
            if let Some(mut f) = self.ready.pop_front() {
                f.pts += self.loop_offset;
                return Ok(Next::Frame(f));
            }
            if self.flushed {
                if !self.looped {
                    return Ok(Next::End);
                }
                self.loop_offset += self.demux.info().duration.unwrap_or(Duration::ZERO);
                self.demux.rewind()?;
                self.decoder.reset();
                self.flushed = false;
                continue;
            }
            match self.demux.next_packet()? {
                Some(p) => self.ready.extend(self.decoder.decode(&p)?),
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
    #[derive(Default)]
    pub struct FakeDecoder {
        held: Option<Packet>,
    }

    fn out(p: Packet) -> DecodedFrame {
        let px = p.data[0];
        DecodedFrame {
            pts: p.pts,
            data: FrameData::Cpu(CpuFrame::from_bgra(1, 1, vec![px, px, px, 255].into())),
        }
    }

    impl Decoder for FakeDecoder {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn decode(&mut self, packet: &Packet) -> PipeResult<Vec<DecodedFrame>> {
            Ok(self
                .held
                .replace(packet.clone())
                .map(out)
                .into_iter()
                .collect())
        }
        fn flush(&mut self) -> PipeResult<Vec<DecodedFrame>> {
            Ok(self.held.take().map(out).into_iter().collect())
        }
        fn reset(&mut self) {
            self.held = None;
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
