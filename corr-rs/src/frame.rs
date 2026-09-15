//! Frame arithmetic and missing-frame bookkeeping.
//!
//! This is the part of the correlator where a silent mistake turns into
//! plausible-looking but wrong spectra, so it is kept pure, free of sockets and
//! device handles, and covered by unit tests.
//!
//! The load-bearing facts, all measured rather than assumed:
//!
//! * The packet counter is a **global** packet counter shared by all three
//!   RFSoC boards. A 120 s run with four stations showed no drift beyond the
//!   +/-20 packet sampling jitter, so `counter` identifies a sample instant
//!   across stations and no per-station offset has to be estimated.
//! * Each packet carries exactly 4096 int8 samples, so a frame of `2 * n_freq`
//!   samples is `P = n_freq / 2048` whole packets.
//!
//! Therefore
//!
//! ```text
//!     frame_id        = counter / P        (which FFT frame)
//!     slot_in_frame   = counter % P        (which packet inside that frame)
//!     frame_index     = frame_id % n_cum   (row inside the staging block)
//! ```
//!
//! A lost packet shows up as a jump in `counter`, so the frames that contain it
//! are known to be incomplete the moment the next packet arrives. No timeout is
//! needed for that judgement; the timeout in the configuration only exists to
//! stop a *dead station* from holding up the cross-station watermark.

use crate::config::SAMPLES_PER_PACKET;

/// How a newly observed counter relates to the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arrival {
    /// Consecutive with the previous packet.
    InOrder,
    /// Larger than expected: `lost` counts the counters that never arrived,
    /// covering `[previous + 1, counter)`.
    Gap { lost: u64 },
    /// Same counter seen twice.
    Duplicate,
    /// Smaller than the previous counter. Measured to never happen on this
    /// link, so it is counted and tolerated rather than treated as fatal.
    OutOfOrder,
    /// The very first packet of the stream.
    First,
}

/// Converts packet counters into frame coordinates.
#[derive(Debug, Clone, Copy)]
pub struct FrameMath {
    packets_per_frame: u64,
}

impl FrameMath {
    /// `packets_per_frame` must be >= 1 and <= 8, because [`FrameArrival`]
    /// stores the per-frame arrival state in a single byte.
    pub fn new(packets_per_frame: u64) -> Self {
        assert!(
            (1..=8).contains(&packets_per_frame),
            "packets_per_frame={packets_per_frame} outside 1..=8; n_freq must be a \
             multiple of 2048 and at most 16384"
        );
        FrameMath {
            packets_per_frame,
        }
    }

    pub fn packets_per_frame(&self) -> u64 {
        self.packets_per_frame
    }

    /// Which FFT frame a packet belongs to.
    #[inline]
    pub fn frame_id(&self, counter: u64) -> u64 {
        counter / self.packets_per_frame
    }

    /// Which packet of its frame this is, `0 .. P-1`.
    #[inline]
    pub fn slot_in_frame(&self, counter: u64) -> u64 {
        counter % self.packets_per_frame
    }

    /// Byte offset of this packet's samples inside its frame. The 8-byte packet
    /// counter is stripped by the caller; `samples` is the 4096-byte payload.
    #[inline]
    pub fn sample_offset_in_frame(&self, counter: u64) -> usize {
        (self.slot_in_frame(counter) * SAMPLES_PER_PACKET) as usize
    }

    /// Absolute sample index of this packet's first sample, counting from
    /// counter 0. Useful for metadata and for reasoning about time.
    #[inline]
    pub fn first_sample_index(&self, counter: u64) -> u64 {
        counter * SAMPLES_PER_PACKET
    }
}

/// Per-station stream statistics and gap detection.
#[derive(Debug, Default, Clone)]
pub struct CounterTracker {
    last: Option<u64>,
    pub packets: u64,
    pub gaps: u64,
    pub lost_packets: u64,
    pub duplicates: u64,
    pub out_of_order: u64,
    /// Highest counter seen, i.e. this station's progress.
    pub max_counter: u64,
}

impl CounterTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one packet counter; returns how it related to the previous one.
    pub fn observe(&mut self, counter: u64) -> Arrival {
        self.packets += 1;
        let a = match self.last {
            None => Arrival::First,
            Some(prev) if counter == prev + 1 => Arrival::InOrder,
            Some(prev) if counter == prev => {
                self.duplicates += 1;
                Arrival::Duplicate
            }
            Some(prev) if counter > prev => {
                let lost = counter - prev - 1;
                self.gaps += 1;
                self.lost_packets += lost;
                Arrival::Gap { lost }
            }
            Some(_) => {
                self.out_of_order += 1;
                Arrival::OutOfOrder
            }
        };
        // max_counter deliberately ignores out-of-order packets: it is the
        // station's progress watermark, and letting a stray old packet pull it
        // backwards would stall the block scheduler.
        if counter > self.max_counter {
            self.max_counter = counter;
            self.last = Some(counter);
        }
        a
    }

    /// Frame id this station has definitely moved past, or `None` before the
    /// first packet. A frame is not "past" until all of its packets could have
    /// arrived, hence the `+ 1`.
    pub fn completed_frames(&self, fm: &FrameMath) -> Option<u64> {
        self.last.map(|c| fm.frame_id(c) + 1)
    }
}

/// Which packets of each frame inside the block currently being filled have
/// arrived.
///
/// `P` is small (1, 2, 4 or 8), so the state of one frame fits in a byte and the
/// whole block needs one byte per frame. A frame is complete only when every one
/// of its packets has been seen; anything else must be zero filled, because a
/// half-filled frame is not a valid FFT input.
#[derive(Debug, Clone)]
pub struct FrameArrival {
    packets_per_frame: u8,
    full_mask: u8,
    bits: Vec<u8>,
    complete: usize,
}

impl FrameArrival {
    pub fn new(n_cum: usize, packets_per_frame: u64) -> Self {
        assert!((1..=8).contains(&packets_per_frame));
        let p = packets_per_frame as u8;
        FrameArrival {
            packets_per_frame: p,
            full_mask: if p == 8 { 0xff } else { (1u8 << p) - 1 },
            bits: vec![0; n_cum],
            complete: 0,
        }
    }

    /// Forget everything, for reuse on the next block.
    pub fn reset(&mut self) {
        self.bits.iter_mut().for_each(|b| *b = 0);
        self.complete = 0;
    }

    pub fn n_cum(&self) -> usize {
        self.bits.len()
    }

    pub fn packets_per_frame(&self) -> u8 {
        self.packets_per_frame
    }

    /// Record that packet `slot` of `frame_index` arrived.
    ///
    /// Returns `Ok(true)` when this packet completed the frame, `Ok(false)` when
    /// the frame is still incomplete. An out-of-range `frame_index` or `slot`
    /// returns `Err` rather than being silently dropped: it would mean the block
    /// scheduler and the packet stream disagree, which is exactly the class of
    /// bug that must not be papered over.
    pub fn observe(&mut self, frame_index: u64, slot: u64) -> Result<bool, ArrivalError> {
        let idx = frame_index as usize;
        if idx >= self.bits.len() {
            return Err(ArrivalError::FrameOutOfBlock {
                frame_index,
                n_cum: self.bits.len(),
            });
        }
        if slot >= self.packets_per_frame as u64 {
            return Err(ArrivalError::SlotOutOfRange {
                slot,
                packets_per_frame: self.packets_per_frame,
            });
        }
        let before = self.bits[idx];
        let after = before | (1u8 << slot);
        self.bits[idx] = after;
        if after == self.full_mask && before != self.full_mask {
            self.complete += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn is_complete(&self, frame_index: usize) -> bool {
        self.bits[frame_index] == self.full_mask
    }

    /// True once at least one packet of the frame arrived but it is not complete.
    pub fn is_partial(&self, frame_index: usize) -> bool {
        let b = self.bits[frame_index];
        b != 0 && b != self.full_mask
    }

    /// How many packets of the frame arrived, for diagnostics.
    pub fn packets_seen(&self, frame_index: usize) -> u32 {
        self.bits[frame_index].count_ones()
    }

    /// Number of frames that received every packet.
    pub fn complete_frames(&self) -> usize {
        self.complete
    }

    /// Frames that must be zero filled: everything that did not receive all of
    /// its packets, whether it saw none or only some.
    ///
    /// Returned in ascending order so the caller can zero contiguous runs.
    pub fn incomplete_frames(&self) -> Vec<usize> {
        self.bits
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if *b == self.full_mask { None } else { Some(i) })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrivalError {
    FrameOutOfBlock { frame_index: u64, n_cum: usize },
    SlotOutOfRange { slot: u64, packets_per_frame: u8 },
}

impl std::fmt::Display for ArrivalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrivalError::FrameOutOfBlock { frame_index, n_cum } => write!(
                f,
                "frame {frame_index} does not belong to the block being filled (n_cum={n_cum})"
            ),
            ArrivalError::SlotOutOfRange {
                slot,
                packets_per_frame,
            } => write!(
                f,
                "packet slot {slot} is outside the frame (P={packets_per_frame})"
            ),
        }
    }
}

impl std::error::Error for ArrivalError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_coordinates_for_n_freq_4096() {
        let fm = FrameMath::new(2);
        // counter 100 is the first packet of frame 50, counter 101 the second.
        assert_eq!(fm.frame_id(100), 50);
        assert_eq!(fm.slot_in_frame(100), 0);
        assert_eq!(fm.sample_offset_in_frame(100), 0);
        assert_eq!(fm.frame_id(101), 50);
        assert_eq!(fm.slot_in_frame(101), 1);
        assert_eq!(fm.sample_offset_in_frame(101), 4096);
        assert_eq!(fm.frame_id(102), 51);
        assert_eq!(fm.slot_in_frame(102), 0);
        assert_eq!(fm.first_sample_index(101), 101 * 4096);
    }

    #[test]
    fn frame_coordinates_for_other_n_freq() {
        // n_freq = 2048 -> one packet per frame, frame id equals counter.
        let fm = FrameMath::new(1);
        assert_eq!(fm.frame_id(12345), 12345);
        assert_eq!(fm.slot_in_frame(12345), 0);

        // n_freq = 8192 -> four packets per frame.
        let fm = FrameMath::new(4);
        assert_eq!(fm.frame_id(1000), 250);
        assert_eq!(fm.slot_in_frame(1000), 0);
        assert_eq!(fm.sample_offset_in_frame(1002), 2 * 4096);
        assert_eq!(fm.frame_id(1003), 250);
        assert_eq!(fm.slot_in_frame(1003), 3);
        assert_eq!(fm.frame_id(1004), 251);
    }

    #[test]
    fn a_counter_gap_immediately_reveals_the_lost_range() {
        let mut t = CounterTracker::new();
        assert_eq!(t.observe(1000), Arrival::First);
        assert_eq!(t.observe(1001), Arrival::InOrder);
        // 1002..1004 never arrived.
        assert_eq!(t.observe(1005), Arrival::Gap { lost: 3 });
        assert_eq!(t.lost_packets, 3);
        assert_eq!(t.gaps, 1);
        assert_eq!(t.packets, 3, "the counter reports packets, not expected packets");
        assert_eq!(t.duplicates, 0);
        assert_eq!(t.out_of_order, 0);
    }

    #[test]
    fn a_gap_is_never_mistaken_for_a_duplicate_or_reordering() {
        let mut t = CounterTracker::new();
        t.observe(10);
        assert_eq!(t.observe(10), Arrival::Duplicate);
        assert_eq!(t.observe(9), Arrival::OutOfOrder);
        // A stray old packet must not drag the progress watermark backwards.
        assert_eq!(t.max_counter, 10);
        assert_eq!(t.duplicates, 1);
        assert_eq!(t.out_of_order, 1);
        // ...and the next in-order packet is still measured against 10.
        assert_eq!(t.observe(11), Arrival::InOrder);
    }

    #[test]
    fn completed_frames_counts_the_frame_actually_in_progress() {
        let fm = FrameMath::new(2);
        let mut t = CounterTracker::new();
        assert_eq!(t.completed_frames(&fm), None);
        t.observe(100); // frame 50 first packet
        assert_eq!(t.completed_frames(&fm), Some(51));
        t.observe(101); // frame 50 complete
        assert_eq!(t.completed_frames(&fm), Some(51));
        t.observe(102); // frame 51 first packet
        assert_eq!(t.completed_frames(&fm), Some(52));
    }

    #[test]
    fn frame_arrival_needs_every_packet() {
        let mut a = FrameArrival::new(4, 2);
        assert!(!a.observe(0, 0).unwrap(), "half a frame is not a frame");
        assert!(!a.is_complete(0));
        assert!(a.is_partial(0));
        assert!(a.observe(0, 1).unwrap(), "second packet completes it");
        assert!(a.is_complete(0));
        assert!(!a.is_partial(0));
        assert_eq!(a.complete_frames(), 1);

        // A repeated packet must not double count.
        assert!(!a.observe(0, 1).unwrap());
        assert_eq!(a.complete_frames(), 1);
    }

    #[test]
    fn incomplete_frames_lists_both_untouched_and_partial_frames() {
        let mut a = FrameArrival::new(6, 2);
        a.observe(1, 0).unwrap(); // frame 1 partial
        a.observe(1, 1).unwrap(); // frame 1 complete
        a.observe(4, 1).unwrap(); // frame 4 partial, packets arriving out of order
        assert_eq!(a.incomplete_frames(), vec![0, 2, 3, 4, 5]);
        assert_eq!(a.packets_seen(4), 1);
        assert_eq!(a.complete_frames(), 1);
        assert!(!a.is_partial(0), "a frame that saw nothing is not partial");
        assert!(!a.is_partial(1));
        assert!(a.is_partial(4));
    }

    #[test]
    fn a_four_packet_frame_needs_all_four() {
        let mut a = FrameArrival::new(2, 4);
        assert!(!a.observe(0, 0).unwrap());
        assert!(!a.observe(0, 1).unwrap());
        assert!(!a.observe(0, 3).unwrap());
        assert!(!a.is_complete(0));
        assert!(a.observe(0, 2).unwrap());
        assert!(a.is_complete(0));
        assert_eq!(a.incomplete_frames(), vec![1]);
    }

    #[test]
    fn a_single_packet_frame_is_complete_on_arrival() {
        let mut a = FrameArrival::new(3, 1);
        assert!(a.observe(0, 0).unwrap());
        assert!(a.observe(2, 0).unwrap());
        assert_eq!(a.complete_frames(), 2);
        assert_eq!(a.incomplete_frames(), vec![1]);
    }

    #[test]
    fn reset_makes_the_blocks_independent() {
        let mut a = FrameArrival::new(4, 2);
        a.observe(0, 0).unwrap();
        a.observe(0, 1).unwrap();
        a.observe(3, 0).unwrap();
        a.reset();
        assert_eq!(a.complete_frames(), 0);
        assert_eq!(a.incomplete_frames(), vec![0, 1, 2, 3]);
        assert_eq!(a.packets_seen(0), 0);
    }

    #[test]
    fn out_of_block_indices_are_errors_not_silent_drops() {
        let mut a = FrameArrival::new(4, 2);
        assert_eq!(
            a.observe(4, 0),
            Err(ArrivalError::FrameOutOfBlock {
                frame_index: 4,
                n_cum: 4
            })
        );
        assert_eq!(
            a.observe(0, 2),
            Err(ArrivalError::SlotOutOfRange {
                slot: 2,
                packets_per_frame: 2
            })
        );
    }

    #[test]
    #[should_panic(expected = "outside 1..=8")]
    fn rejects_a_frame_longer_than_a_byte_of_state() {
        FrameMath::new(9);
    }
}
