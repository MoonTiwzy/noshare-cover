use std::collections::BTreeMap;
use std::io::{Read, Seek};

use crate::{
    skip_box, BoxHeader, BoxType, EmsgBox, Error, FtypBox, MdhdBox, MoofBox, MoovBox, ReadBox,
    Result, StsdBoxContent, TfhdBox, TrackId, TrackKind, TrakBox, TrunBox,
};

#[derive(Debug)]
pub struct Mp4 {
    pub ftyp: FtypBox,
    pub moov: MoovBox,
    pub moofs: Vec<MoofBox>,
    pub emsgs: Vec<EmsgBox>,
    tracks: BTreeMap<TrackId, Track>,
}

impl Mp4 {
    /// Parses the contents of a byte slice as MP4 data.
    ///
    /// Sample ranges returned by the resulting [`Mp4`] should be used with the same input buffer.
    pub fn read_bytes(bytes: &[u8]) -> Result<Self> {
        let mp4 = Self::read(std::io::Cursor::new(bytes), bytes.len() as u64)?;
        Ok(mp4)
    }

    /// Reads the contents of a file as MP4 data, and returns both the parsed MP4 and its raw data.
    ///
    /// Sample ranges returned by the resulting [`Mp4`] should be used with the same input buffer.
    pub fn read_file(file_path: impl AsRef<std::path::Path>) -> Result<(Self, Vec<u8>)> {
        let bytes = std::fs::read(file_path)?;
        Ok((Self::read_bytes(&bytes)?, bytes))
    }

    pub fn read<R: Read + Seek>(reader: R, size: u64) -> Result<Self> {
        Self::read_tracks(reader, size, |_| true)
    }

    /// The most samples read from a file, in all its tracks read together: 24 hours at
    /// 90 frames a second fit. At 56 bytes a [`Sample`], they take up to 470 MB; sample
    /// tables that claim more are refused before that is allocated.
    pub const MAX_SAMPLES: u64 = 1 << 23;

    /// Like [`Self::read`], with the samples of only the tracks `wanted` picks. The others
    /// are listed without samples, and their sample tables are not walked: a track the
    /// caller has no use for can't make the file unreadable, nor take memory.
    pub fn read_tracks<R: Read + Seek>(
        mut reader: R,
        size: u64,
        wanted: impl Fn(&TrakBox) -> bool,
    ) -> Result<Self> {
        let start = reader.stream_position()?;

        let mut ftyp = None;
        let mut moov = None;
        let mut moofs = Vec::new();
        let mut moof_offsets = Vec::new();
        let mut emsgs = Vec::new();

        let mut current = start;
        while current < size {
            // Get box header.
            let header = BoxHeader::read_top_level(&mut reader)?;
            let BoxHeader { name, size: s } = header;
            if s > size {
                return Err(Error::InvalidData(
                    "file contains a box with a larger size than it",
                ));
            }

            // Break if size zero BoxHeader, which can result in dead-loop.
            if s == 0 {
                break;
            }

            // Match and parse the atom boxes.
            match name {
                BoxType::FtypBox => {
                    ftyp = Some(FtypBox::read_box(&mut reader, s)?);
                }
                BoxType::FreeBox => {
                    skip_box(&mut reader, s)?;
                }
                BoxType::MdatBox => {
                    skip_box(&mut reader, s)?;
                }
                BoxType::MoovBox => {
                    moov = Some(MoovBox::read_box(&mut reader, s)?);
                }
                BoxType::MoofBox => {
                    let moof_offset = reader.stream_position()? - 8;
                    let moof = MoofBox::read_box(&mut reader, s)?;
                    moofs.push(moof);
                    moof_offsets.push(moof_offset);
                }
                BoxType::EmsgBox => {
                    let emsg = EmsgBox::read_box(&mut reader, s)?;
                    emsgs.push(emsg);
                }
                _ => {
                    // XXX warn!()
                    skip_box(&mut reader, s)?;
                }
            }
            current = reader.stream_position()?;
        }

        let Some(ftyp) = ftyp else {
            return Err(Error::BoxNotFound(BoxType::FtypBox));
        };
        let Some(moov) = moov else {
            return Err(Error::BoxNotFound(BoxType::MoovBox));
        };

        let mut this = Self {
            ftyp,
            moov,
            moofs,
            emsgs,
            tracks: Default::default(),
        };

        let mut budget = SampleBudget::new(size);
        let mut tracks = this.build_tracks(&wanted, &mut budget)?;
        this.update_sample_list(&mut tracks, &wanted, &mut budget)?;
        this.tracks = tracks;
        this.update_tracks();

        Ok(this)
    }

    pub fn tracks(&self) -> &BTreeMap<TrackId, Track> {
        &self.tracks
    }

    /// The tracks, taken out of the file: their samples without a copy.
    pub fn into_tracks(self) -> BTreeMap<TrackId, Track> {
        self.tracks
    }

    /// Process each `trak` box to obtain a list of samples for each track `wanted` picks;
    /// the others are listed without samples.
    ///
    /// Note that the list will be incomplete if the file is fragmented.
    fn build_tracks(
        &mut self,
        wanted: &dyn Fn(&TrakBox) -> bool,
        budget: &mut SampleBudget,
    ) -> Result<BTreeMap<TrackId, Track>> {
        let mut tracks = BTreeMap::new();

        // load samples from traks
        for trak in &self.moov.traks {
            let (samples, shift) = if wanted(trak) {
                Self::trak_samples(trak, budget)?
            } else {
                Default::default()
            };

            tracks.insert(
                trak.tkhd.track_id,
                Track {
                    track_id: trak.tkhd.track_id,
                    width: trak.tkhd.width.value(),
                    height: trak.tkhd.height.value(),
                    first_traf_merged: false,
                    shift,
                    timescale: trak.mdia.mdhd.timescale as u64,
                    duration: media_duration(&trak.mdia.mdhd),
                    kind: trak.mdia.minf.stbl.stsd.kind(),
                    samples,
                },
            );
        }

        Ok(tracks)
    }

    /// The samples of a `trak` box, from its sample tables, and what was taken off their
    /// timestamps.
    fn trak_samples(trak: &TrakBox, budget: &mut SampleBudget) -> Result<(Vec<Sample>, TimeShift)> {
        let stbl = &trak.mdia.minf.stbl;
        let stsc = &stbl.stsc;
        let stsz = &stbl.stsz;
        let timescale = trak.mdia.mdhd.timescale as u64;
        let media_duration = media_duration(&trak.mdia.mdhd);

        let sample_count = if stsz.sample_size == 0 {
            stsz.sample_sizes.len()
        } else {
            stsz.sample_count as usize
        };
        budget.take(sample_count as u64)?;

        // The file offset of a chunk, counted from 1.
        let chunk_offset = |chunk: usize| -> Option<u64> {
            let index = chunk.checked_sub(1)?;
            if let Some(stco) = &stbl.stco {
                stco.entries.get(index).map(|&offset| u64::from(offset))
            } else {
                stbl.co64.as_ref()?.entries.get(index).copied()
            }
        };

        // The chunk the next sample is in (counted from 1), the stsc entry for that
        // chunk, how many samples the chunk has left and where the next one starts.
        let mut chunk = 0usize;
        let mut chunk_run = 0usize;
        let mut left_in_chunk = 0u64;
        let mut offset_in_file = 0u64;

        let mut stts_runs = Runs::new(&stbl.stts.entries);
        let mut ctts_runs = stbl.ctts.as_ref().map(|ctts| Runs::new(&ctts.entries));
        let mut next_stss_index = 0;
        let mut next_decode_timestamp = 0i64;
        let mut dts_shift = 0;

        // The smallest presentation timestamp observed in this stream.
        //
        // This is typically 0, but in the presence of sample reordering (caused by AVC/HVC b-frames), it may be non-zero.
        // In fact, many formats don't require this to be zero, but video players typically
        // normalize the shown time to start at zero.
        // This is roughly equivalent to FFmpeg's internal `min_corrected_pts`
        // https://github.com/FFmpeg/FFmpeg/blob/4047b887fc44b110bccb1da09bcb79d6e454b88b/libavformat/isom.h#L202
        // To learn more about this I recommend reading the patch that introduced this in FFmpeg:
        // https://patchwork.ffmpeg.org/project/ffmpeg/patch/20170606181601.25187-1-isasi@google.com/#12592
        let mut min_composition_timestamp = i64::MAX;

        // Grown as the samples are walked, not reserved for all the tables claim: tables
        // that break off after claiming millions of samples take no more than they gave.
        let mut samples = Vec::<Sample>::new();

        for sample_n in 0..sample_count {
            // compute offset: move on to the next chunk with samples once this one is
            // used up (an stsc entry may give its chunks no samples)
            while left_in_chunk == 0 {
                chunk += 1;
                while stsc
                    .entries
                    .get(chunk_run + 1)
                    .is_some_and(|next| next.first_chunk as usize <= chunk)
                {
                    chunk_run += 1;
                }
                let Some(run) = stsc.entries.get(chunk_run) else {
                    return Err(Error::InvalidData("stsc has no entries for the samples"));
                };
                offset_in_file = chunk_offset(chunk).ok_or(Error::InvalidData(
                    "stsc places samples in more chunks than stco/co64 has",
                ))?;
                left_in_chunk = u64::from(run.samples_per_chunk);
            }
            left_in_chunk -= 1;

            let size = if stsz.sample_size == 0 {
                stsz.sample_sizes
                    .get(sample_n)
                    .map_or(0, |&size| u64::from(size))
            } else {
                u64::from(stsz.sample_size)
            };
            let offset = offset_in_file;
            offset_in_file = offset_in_file.saturating_add(size);

            // compute timestamp, duration, is_sync
            let decode_timestamp = next_decode_timestamp;
            let duration = match stts_runs.next_sample(|entry| entry.sample_count) {
                Some(entry) => u64::from(entry.sample_delta),
                // an stts one sample short: the last sample lasts until the media ends
                None if sample_n + 1 == sample_count => {
                    media_duration.saturating_sub(decode_timestamp.unsigned_abs())
                }
                None => return Err(Error::InvalidData("stts describes fewer samples than stsz")),
            };
            next_decode_timestamp = decode_timestamp
                .checked_add_unsigned(duration)
                .ok_or(Error::InvalidData("sample timestamps overflow"))?;

            let composition_timestamp = if let Some(ctts_runs) = &mut ctts_runs {
                // past the end of a short ctts: no offset
                let offset = ctts_runs
                    .next_sample(|entry| entry.sample_count)
                    .map_or(0, |entry| i64::from(entry.sample_offset));

                // dts shift is determined by the smallest negative sample offset:
                // https://github.com/FFmpeg/FFmpeg/blob/455db6fe109cf905fe518ea2690495948937438f/libavformat/mov.c#L3671
                if offset < 0 {
                    dts_shift = dts_shift.max(-offset);
                }

                decode_timestamp
                    .checked_add(offset)
                    .ok_or(Error::InvalidData("sample timestamps overflow"))?
            } else {
                decode_timestamp
            };
            min_composition_timestamp = min_composition_timestamp.min(composition_timestamp);

            let is_sync = if let Some(stss) = &stbl.stss {
                // stss numbers the sync samples from 1, in increasing order
                let number = sample_n as u64 + 1;
                while stss
                    .entries
                    .get(next_stss_index)
                    .is_some_and(|&n| u64::from(n) < number)
                {
                    next_stss_index += 1;
                }
                stss.entries
                    .get(next_stss_index)
                    .is_some_and(|&n| u64::from(n) == number)
            } else {
                true
            };

            let id = samples.len() as u32;
            push_sample(
                &mut samples,
                Sample {
                    id,
                    timescale,
                    size,
                    offset,
                    decode_timestamp,
                    composition_timestamp,
                    is_sync,
                    duration,
                },
            )?;
        }

        // Fixup all DTS by the dts shift if there's one.
        // https://github.com/FFmpeg/FFmpeg/blob/455db6fe109cf905fe518ea2690495948937438f/libavformat/mov.c#L4271
        if dts_shift > 0 {
            for sample in &mut samples {
                sample.decode_timestamp -= dts_shift;
            }
        }

        // Shift both DTS & CTS by the smallest CTS.
        // For details, see declaration of `min_composition_timestamp` above.
        if min_composition_timestamp != 0 {
            for sample in &mut samples {
                sample.decode_timestamp = sample
                    .decode_timestamp
                    .saturating_sub(min_composition_timestamp);
                sample.composition_timestamp = sample
                    .composition_timestamp
                    .saturating_sub(min_composition_timestamp);
            }
        }

        let shift = TimeShift {
            decode: dts_shift,
            // with no samples there is no smallest one
            both: if samples.is_empty() {
                0
            } else {
                min_composition_timestamp
            },
        };
        Ok((samples, shift))
    }

    /// In case the input file is fragmented, it will contain one or more `moof` boxes,
    /// which must be processed to obtain the full list of samples for each track.
    fn update_sample_list(
        &mut self,
        tracks: &mut BTreeMap<TrackId, Track>,
        wanted: &dyn Fn(&TrakBox) -> bool,
        budget: &mut SampleBudget,
    ) -> Result<()> {
        for moof in &self.moofs {
            // Where the data of the previous track fragment ends: the base data offset of a
            // track fragment that has none of its own and isn't based at the moof. For the
            // first one that is the moof (ISO/IEC 14496-12, 8.8.7, tfhd).
            let mut previous_traf_end = moof.start;

            // process moof to update sample list
            for traf in &moof.trafs {
                let track_id = traf.tfhd.track_id;
                let track = tracks
                    .get_mut(&track_id)
                    .ok_or(Error::TrakNotFound(track_id))?;
                let trak = self
                    .moov
                    .traks
                    .iter()
                    .find(|trak| trak.tkhd.track_id == track_id)
                    .ok_or(Error::TrakNotFound(track_id))?;
                let read = wanted(trak);
                let trex = if let Some(mvex) = &self.moov.mvex {
                    mvex.trexs
                        .iter()
                        .find(|trex| trex.track_id == track_id)
                        .ok_or(Error::BoxInTrafNotFound(track_id, BoxType::TrexBox))?
                        .clone()
                } else {
                    Default::default()
                };

                let default_sample_duration = traf
                    .tfhd
                    .default_sample_duration
                    .unwrap_or(trex.default_sample_duration);
                let default_sample_size = traf
                    .tfhd
                    .default_sample_size
                    .unwrap_or(trex.default_sample_size);
                let default_sample_flags = traf
                    .tfhd
                    .default_sample_flags
                    .unwrap_or(trex.default_sample_flags);

                let base_data_offset = if traf.tfhd.flags & TfhdBox::FLAG_BASE_DATA_OFFSET != 0 {
                    traf.tfhd.base_data_offset.unwrap_or(moof.start)
                } else if traf.tfhd.flags & TfhdBox::FLAG_DEFAULT_BASE_IS_MOOF != 0 {
                    moof.start
                } else {
                    previous_traf_end
                };
                // Where a run without a data offset of its own starts: right after the
                // previous run, or at the base data offset for the first (8.8.8, trun).
                let mut run_start = base_data_offset;

                for trun in &traf.truns {
                    // Sample offset in bytes. (Must be positive, otherwise this would be outside of the file.)
                    let mut offset_in_file = match trun.data_offset {
                        Some(data_offset) => {
                            base_data_offset.saturating_add_signed(i64::from(data_offset))
                        }
                        None => run_start,
                    };

                    if !read {
                        // A track not read: only where its run ends matters, as the next
                        // run, or the next track fragment, may start there.
                        run_start =
                            offset_in_file.saturating_add(run_size(trun, default_sample_size));
                        continue;
                    }
                    budget.take(u64::from(trun.sample_count))?;

                    for sample_n in 0..trun.sample_count as usize {
                        let mut sample_flags = default_sample_flags;
                        if trun.flags & TrunBox::FLAG_SAMPLE_FLAGS != 0 {
                            sample_flags = trun
                                .sample_flags
                                .get(sample_n)
                                .copied()
                                .unwrap_or(sample_flags);
                        } else if sample_n == 0
                            && (trun.flags & TrunBox::FLAG_FIRST_SAMPLE_FLAGS != 0)
                        {
                            sample_flags = trun.first_sample_flags.unwrap_or(sample_flags);
                        }

                        let decode_timestamp = match (&traf.tfdt, track.samples.last()) {
                            // the first sample of the track's first fragment: at the decode
                            // time it gives, moved as the samples in the moov before it were
                            (Some(tfdt), _) if !track.first_traf_merged && sample_n == 0 => {
                                let Ok(time) = i64::try_from(tfdt.base_media_decode_time) else {
                                    return Err(Error::InvalidData(
                                        "tfdt decode time out of range",
                                    ));
                                };
                                track.shift.decode_timestamp(time)?
                            }
                            // after the sample before, from the moov or a fragment: without a
                            // tfdt, a fragment starts where the samples before it end
                            // (ISO/IEC 14496-12, 8.8.12)
                            (_, Some(prev)) => prev
                                .decode_timestamp
                                .checked_add_unsigned(prev.duration)
                                .ok_or(Error::InvalidData("sample timestamps overflow"))?,
                            (_, None) => 0,
                        };
                        track.first_traf_merged = true;

                        let composition_offset = if trun.flags & TrunBox::FLAG_SAMPLE_CTS != 0 {
                            trun.sample_cts.get(sample_n).copied().unwrap_or(0)
                        } else {
                            0
                        };
                        let composition_timestamp = track
                            .shift
                            .composition_timestamp(decode_timestamp, composition_offset)?;

                        let duration = trun
                            .sample_durations
                            .get(sample_n)
                            .copied()
                            .unwrap_or(default_sample_duration)
                            as u64;

                        let sample_size =
                            trun.sample_sizes
                                .get(sample_n)
                                .copied()
                                .unwrap_or(default_sample_size) as u64;

                        let offset = offset_in_file;
                        offset_in_file = offset_in_file.saturating_add(sample_size);

                        let id = track.samples.len() as u32;
                        push_sample(
                            &mut track.samples,
                            Sample {
                                id,
                                is_sync: is_sync_sample(sample_flags),
                                size: sample_size,
                                offset,
                                timescale: trak.mdia.mdhd.timescale as u64,
                                decode_timestamp,
                                composition_timestamp,
                                duration,
                            },
                        )?;
                    }

                    run_start = offset_in_file;
                }

                previous_traf_end = run_start;
            }
        }

        Ok(())
    }

    /// Update track metadata after all samples have been read
    fn update_tracks(&mut self) {
        for track in self.tracks.values_mut() {
            if track.duration == 0 {
                track.duration = track
                    .samples
                    .last()
                    .map(|v| v.duration.saturating_add_signed(v.composition_timestamp))
                    .unwrap_or_default();
            }
        }
    }
}

/// How many more samples the tracks read may have: one per byte of the file, and
/// [`Mp4::MAX_SAMPLES`] in all. A few bytes of sample tables can claim billions of samples;
/// tables that claim more than the budget are refused before their samples are walked.
struct SampleBudget {
    left: u64,
    /// why the tables are refused when they claim more
    over: &'static str,
}

impl SampleBudget {
    fn new(file_size: u64) -> Self {
        if file_size <= Mp4::MAX_SAMPLES {
            Self {
                left: file_size,
                over: "the sample tables describe more samples than the file has bytes",
            }
        } else {
            Self {
                left: Mp4::MAX_SAMPLES,
                over: "the sample tables describe more than 8388608 samples, the most read",
            }
        }
    }

    fn take(&mut self, samples: u64) -> Result<()> {
        self.left = self
            .left
            .checked_sub(samples)
            .ok_or(Error::InvalidData(self.over))?;
        Ok(())
    }
}

/// Adds a sample to a track's list. Memory the list can't get is an error, not an abort:
/// the reader may run in a process that must not go down with a broken file.
fn push_sample(samples: &mut Vec<Sample>, sample: Sample) -> Result<()> {
    samples
        .try_reserve(1)
        .map_err(|_| Error::InvalidData("no memory for the samples"))?;
    samples.push(sample);
    Ok(())
}

/// The bytes a `trun`'s samples take: the sizes it lists, the default size for the samples
/// it has none for. Counted without walking them: a run can claim 4 billion samples with
/// no bytes of its own.
fn run_size(trun: &TrunBox, default_sample_size: u32) -> u64 {
    let listed = trun.sample_sizes.len().min(trun.sample_count as usize);
    let sizes = trun.sample_sizes[..listed]
        .iter()
        .fold(0u64, |sum, &size| sum.saturating_add(u64::from(size)));
    let unlisted = u64::from(trun.sample_count) - listed as u64;
    sizes.saturating_add(unlisted.saturating_mul(u64::from(default_sample_size)))
}

/// Walks a run-length table (`stts`, `ctts`) one sample at a time.
struct Runs<'a, T> {
    entries: std::slice::Iter<'a, T>,
    entry: Option<&'a T>,
    left: u32,
}

impl<'a, T> Runs<'a, T> {
    fn new(entries: &'a [T]) -> Self {
        Self {
            entries: entries.iter(),
            entry: None,
            left: 0,
        }
    }

    /// The entry the next sample falls under; `None` past the end of the table. Entries
    /// for no samples are skipped.
    fn next_sample(&mut self, sample_count: impl Fn(&T) -> u32) -> Option<&'a T> {
        while self.left == 0 {
            let entry = self.entries.next()?;
            self.entry = Some(entry);
            self.left = sample_count(entry);
        }
        self.left -= 1;
        self.entry
    }
}

/// The duration of a track's media from its `mdhd`, 0 when it is unknown: all ones in
/// either version of the box (ISO/IEC 14496-12, 8.4.2).
fn media_duration(mdhd: &MdhdBox) -> u64 {
    let unknown = if mdhd.version == 1 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    if mdhd.duration == unknown {
        0
    } else {
        mdhd.duration
    }
}

/// What the samples from a `trak` box had taken off their timestamps (see
/// [`Mp4::trak_samples`]): the decode timestamps the largest negative composition offset,
/// and both the smallest composition timestamp. The samples of the track's fragments after
/// them (ffmpeg's `frag_keyframe` without `empty_moov` puts samples in both) are moved
/// the same, so their timeline runs on from those.
#[derive(Default, Clone, Copy)]
struct TimeShift {
    decode: i64,
    both: i64,
}

impl TimeShift {
    /// The decode timestamp of a sample at `time` in the track's media.
    fn decode_timestamp(self, time: i64) -> Result<i64> {
        time.checked_sub(self.decode)
            .and_then(|time| time.checked_sub(self.both))
            .ok_or(Error::InvalidData("sample timestamps overflow"))
    }

    /// The composition timestamp of a sample with this (moved) decode timestamp and
    /// composition offset: the offset, and what only the decode timestamps had taken off.
    fn composition_timestamp(self, decode_timestamp: i64, offset: i64) -> Result<i64> {
        decode_timestamp
            .checked_add(offset)
            .and_then(|time| time.checked_add(self.decode))
            .ok_or(Error::InvalidData("sample timestamps overflow"))
    }
}

/// `sample_is_non_sync_sample` in the sample flags of `trex`, `tfhd` and `trun`
/// (ISO/IEC 14496-12, 8.8.3): set for a sample that is not a sync sample.
const SAMPLE_IS_NON_SYNC_SAMPLE: u32 = 1 << 16;

fn is_sync_sample(sample_flags: u32) -> bool {
    sample_flags & SAMPLE_IS_NON_SYNC_SAMPLE == 0
}

pub struct Track {
    /// Internal field used when decoding a fragmented MP4 file.
    first_traf_merged: bool,

    /// What the samples from the `trak` box had taken off their timestamps; the samples
    /// of the track's fragments after them are moved the same.
    shift: TimeShift,

    pub width: u16,
    pub height: u16,

    pub track_id: u32,

    /// Timescale of the sample.
    ///
    /// One time unit is equal to `1.0 / timescale` seconds.
    pub timescale: u64,

    /// Duration of the track in time units.
    pub duration: u64,

    pub kind: Option<TrackKind>,

    /// List of samples in the track.
    pub samples: Vec<Sample>,
}

impl Track {
    pub fn trak<'a>(&self, mp4: &'a Mp4) -> &'a TrakBox {
        let Some(trak) = mp4
            .moov
            .traks
            .iter()
            .find(|trak| trak.tkhd.track_id == self.track_id)
        else {
            // `Track` structs are only constructed when we have `trak` boxes,
            // so unless the user removes the `trak` box from the `Mp4`, it
            // will always be present.
            unreachable!("track with id \"{}\" not found", self.track_id);
        };

        trak
    }

    pub fn raw_codec_config(&self, mp4: &Mp4) -> Option<Vec<u8>> {
        let sample_description = &self.trak(mp4).mdia.minf.stbl.stsd;

        match &sample_description.contents {
            StsdBoxContent::Av01(content) => Some(content.av1c.raw.clone()),
            StsdBoxContent::Avc1(content) => Some(content.avcc.raw.clone()),
            StsdBoxContent::Hev1(content) | StsdBoxContent::Hvc1(content) => {
                Some(content.hvcc.raw.clone())
            }
            StsdBoxContent::Vp08(content) => Some(content.vpcc.raw.clone()),
            StsdBoxContent::Vp09(content) => Some(content.vpcc.raw.clone()),
            StsdBoxContent::Mp4a(_) | StsdBoxContent::Tx3g(_) | StsdBoxContent::Unknown(_) => None,
        }
    }

    pub fn codec_string(&self, mp4: &Mp4) -> Option<String> {
        self.trak(mp4).mdia.minf.stbl.stsd.contents.codec_string()
    }
}

#[derive(Default, Clone, Copy)]
pub struct Sample {
    /// Sample number.
    pub id: u32,

    /// Whether or not an entire frame can be decoded from this one sample,
    /// or if it needs the context of other samples.
    pub is_sync: bool,

    /// Size of the sample in bytes.
    pub size: u64,

    /// Offset of the sample in bytes from the start of the MP4 file.
    pub offset: u64,

    /// Timescale of the sample.
    ///
    /// One time unit is equal to `1.0 / timescale` seconds.
    pub timescale: u64,

    /// Timestamp of the sample at which it should be decoded,
    /// in time units.
    ///
    /// This is offsetted:
    /// * with decode timestamp shift determined from negative sample offsets
    /// * such that the first [`Self::composition_timestamp`] is zero.
    pub decode_timestamp: i64,

    /// Timestamp of the sample at which the sample should be displayed,
    /// in time units.
    ///
    /// This is offsetted such that the first composition timestamp is zero.
    pub composition_timestamp: i64,

    /// Duration of the sample in time units.
    pub duration: u64,
}

impl Sample {
    /// Returns the range of bytes in the input data that this sample covers.
    pub fn byte_range(&self) -> std::ops::Range<usize> {
        self.offset as usize..(self.offset + self.size) as usize
    }
}

impl std::fmt::Debug for Track {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Track")
            .field("first_traf_merged", &self.first_traf_merged)
            .field("kind", &self.kind)
            .field("timescale", &self.timescale)
            .field("duration", &self.duration)
            .finish()
    }
}

impl std::fmt::Debug for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sample")
            .field("is_sync", &self.is_sync)
            .field("size", &self.size)
            .field("offset", &self.offset)
            .field("decode_timestamp", &self.decode_timestamp)
            .field("composition_timestamp", &self.composition_timestamp)
            .field("duration", &self.duration)
            .finish()
    }
}
