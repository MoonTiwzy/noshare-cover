//! Converts H.264/HEVC from container format to Annex-B.
//!
//! MP4 and Matroska store H.264/HEVC as "length + NAL" (AVCC/HVCC format), with
//! parameter sets (SPS/PPS/VPS) kept separately in avcC/hvcC. Decoders (openh264, VA-API
//! parsers) expect Annex-B: `00 00 00 01` + NAL, with parameter sets in-stream before
//! each keyframe. The avcC/hvcC layouts are the same in both containers, so
//! we parse them here ourselves from raw bytes.

const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Parsed codec config: length field size and parameter sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NalConfig {
    /// Size of the NAL length prefix in a sample, in bytes (1, 2 or 4).
    pub length_size: usize,
    /// VPS/SPS/PPS in the order they must be fed to the decoder.
    pub parameter_sets: Vec<Vec<u8>>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn u8(&mut self) -> Option<u8> {
        let v = *self.buf.get(self.pos)?;
        self.pos += 1;
        Some(v)
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_be_bytes([self.u8()?, self.u8()?]))
    }
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.buf.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }
    fn skip(&mut self, n: usize) -> Option<()> {
        self.bytes(n).map(|_| ())
    }
}

/// `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15, 5.3.3.1).
pub fn parse_avcc(raw: &[u8]) -> Option<NalConfig> {
    let mut r = Reader::new(raw);
    if r.u8()? != 1 {
        return None; // configurationVersion
    }
    r.skip(3)?; // profile, compat, level
    let length_size = usize::from(r.u8()? & 0b11) + 1;
    let mut sets = Vec::new();
    let sps = r.u8()? & 0b1_1111;
    for _ in 0..sps {
        let n = usize::from(r.u16()?);
        sets.push(r.bytes(n)?.to_vec());
    }
    let pps = r.u8()?;
    for _ in 0..pps {
        let n = usize::from(r.u16()?);
        sets.push(r.bytes(n)?.to_vec());
    }
    (length_size != 3).then_some(NalConfig {
        length_size,
        parameter_sets: sets,
    })
}

/// `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15, 8.3.3.1).
pub fn parse_hvcc(raw: &[u8]) -> Option<NalConfig> {
    let mut r = Reader::new(raw);
    if r.u8()? != 1 {
        return None;
    }
    // profile/tier/level etc. up to lengthSizeMinusOne: 20 bytes
    r.skip(20)?;
    let length_size = usize::from(r.u8()? & 0b11) + 1;
    let arrays = r.u8()?;
    let mut sets = Vec::new();
    for _ in 0..arrays {
        r.u8()?; // completeness + nal type
        let count = r.u16()?;
        for _ in 0..count {
            let n = usize::from(r.u16()?);
            sets.push(r.bytes(n)?.to_vec());
        }
    }
    (length_size != 3).then_some(NalConfig {
        length_size,
        parameter_sets: sets,
    })
}

/// "length + NAL" sample → Annex-B. Keyframes get the parameter sets prepended.
/// `None`: the sample is broken (a length runs past the end); drop that frame.
pub fn to_annexb(sample: &[u8], cfg: &NalConfig, keyframe: bool) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(sample.len() + 64);
    if keyframe {
        for ps in &cfg.parameter_sets {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(ps);
        }
    }
    let mut r = Reader::new(sample);
    while r.pos < sample.len() {
        let len_bytes = r.bytes(cfg.length_size)?;
        let n = len_bytes
            .iter()
            .fold(0usize, |acc, b| (acc << 8) | usize::from(*b));
        if n == 0 {
            continue;
        }
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(r.bytes(n)?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avcc(sps: &[u8], pps: &[u8], len_minus_one: u8) -> Vec<u8> {
        let mut v = vec![1, 0x64, 0, 0x1f, 0xfc | len_minus_one, 0xe1];
        v.extend((sps.len() as u16).to_be_bytes());
        v.extend(sps);
        v.push(1);
        v.extend((pps.len() as u16).to_be_bytes());
        v.extend(pps);
        v
    }

    #[test]
    fn avcc_roundtrip_to_annexb() {
        let cfg = parse_avcc(&avcc(&[0x67, 1, 2], &[0x68, 3], 3)).unwrap();
        assert_eq!(cfg.length_size, 4);
        assert_eq!(cfg.parameter_sets, vec![vec![0x67, 1, 2], vec![0x68, 3]]);

        // two NALs in one sample
        let sample = [0, 0, 0, 2, 0x65, 9, 0, 0, 0, 1, 0x06];
        let key = to_annexb(&sample, &cfg, true).unwrap();
        assert_eq!(
            key,
            [
                0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 9, 0, 0, 0, 1, 0x06
            ]
        );
        let delta = to_annexb(&sample, &cfg, false).unwrap();
        assert_eq!(delta, [0, 0, 0, 1, 0x65, 9, 0, 0, 0, 1, 0x06]);
    }

    #[test]
    fn short_length_fields() {
        let cfg = parse_avcc(&avcc(&[0x67], &[0x68], 1)).unwrap();
        assert_eq!(cfg.length_size, 2);
        assert_eq!(
            to_annexb(&[0, 1, 0x41], &cfg, false).unwrap(),
            [0, 0, 0, 1, 0x41]
        );
    }

    #[test]
    fn truncated_is_none_not_panic() {
        let cfg = NalConfig {
            length_size: 4,
            parameter_sets: vec![],
        };
        assert!(to_annexb(&[0, 0, 0, 9, 1, 2], &cfg, false).is_none());
        assert!(parse_avcc(&[1, 2]).is_none());
        assert!(
            parse_avcc(&[2, 0, 0, 0, 0xff, 0]).is_none(),
            "wrong version"
        );
        assert!(parse_hvcc(&[1; 10]).is_none());
    }

    #[test]
    fn hvcc_arrays() {
        let mut v = vec![1u8];
        v.extend([0u8; 20]);
        v.push(0xf3); // lengthSizeMinusOne = 3
        v.push(2); // two arrays
        for (ty, nal) in [(32u8, vec![0x40, 1]), (33, vec![0x42, 2, 3])] {
            v.push(0x80 | ty);
            v.extend(1u16.to_be_bytes());
            v.extend((nal.len() as u16).to_be_bytes());
            v.extend(&nal);
        }
        let cfg = parse_hvcc(&v).unwrap();
        assert_eq!(cfg.length_size, 4);
        assert_eq!(cfg.parameter_sets, vec![vec![0x40, 1], vec![0x42, 2, 3]]);
    }
}
