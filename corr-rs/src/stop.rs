//! Why a run stopped, in one place.
//!
//! A run can stop for four reasons: the operator interrupted it, the requested
//! duration elapsed, a station fabricated more silence than the cap allows, or a
//! receive thread hit an error the whole run has to answer. They used to be three
//! mechanisms: a shared `AtomicBool` for "stopped", a `Mutex<Option<String>>` for
//! the first fatal message, and a local `Option<String>` in the pipeline for the
//! zero-fill cap. Reading a finished run meant reading all three and deciding
//! which one was the cause.
//!
//! Now there is one value with one rule: **the first reason wins**. A run that is
//! interrupted while a station is also fabricating silence should say it was
//! interrupted, not report the symptom it happened to notice later.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// What stopped a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The operator interrupted it (SIGINT/SIGTERM).
    Signal,
    /// The requested duration elapsed: the normal end of a bounded run.
    Duration,
    /// A station exceeded the zero-fill cap, so the rows it was in are no longer
    /// describing the sky.
    ZeroFillCap {
        /// The station's configured name: what an operator recognises, and what
        /// the message printed before this type existed. The index alone is a
        /// regression in wording, which is how it was nearly shipped.
        name: String,
        station: usize,
        fabricated: u64,
        limit: u64,
    },
    /// The watermark stayed lifted for longer than the configuration allows.
    ///
    /// Earlier than a zero-fill cap, and for a different reason: `lift > 0` means the
    /// scheduler has already stopped waiting for a station, so its frames are being
    /// recorded as missing. A socket backlog can drain when the machine quiets down;
    /// this cannot -- every station advances at the same rate, so the gap is conserved
    /// and only a restart clears it.
    LiftCap {
        seconds: u64,
        /// Slope in thousandths of a slot: `f64` would cost `Eq` on this type, and the
        /// number is only ever formatted for a person.
        slots_milli: u64,
        /// Station index. The name is not available where this is raised -- the pipeline
        /// holds the plan, not the station table -- and the per-station panel shows the
        /// index beside the name.
        station: usize,
    },
    /// Every station is receiving packets, but the run's own frame numbering has
    /// not moved for `seconds`.
    ///
    /// This is the "the stream and the run disagree about which frame it is" state,
    /// and it is dangerous precisely because nothing looks wrong: the sockets are
    /// read and drained, the ring is untouched, no packet is counted as lost, no
    /// frame is counted as zero filled, and the run writes no rows for as long as
    /// it is left alone. Two ways in, both seen for real on 2026-09-20:
    ///
    /// * the front end's counter restarted (its epoch changed) mid-run, so every
    ///   later packet belonged to a frame the assembler had already finalized --
    ///   a healthy run producing 1.00 rows/s froze at row 132 and stayed there;
    /// * a restarted front end plus `daily_files` resuming the day's set: the resume
    ///   floor pinned the start frame at the set's continuation (1.0e10) while the
    ///   stream was at 4.0e7, so every packet arrived "before the start" and a run
    ///   that reported `0 slots, 0 rows` sat at 12 cores of CPU forever.
    StreamNotAdvancing {
        /// The frame the run's rows are numbered from.
        start_frame: u64,
        /// The frame every station's finalized progress is stuck at.
        progress: u64,
        seconds: u64,
        n_station: usize,
    },
    /// No station has taken a packet off its socket for `seconds`.
    ///
    /// The front end stopped, or the path to it did. Distinct from
    /// [`StopReason::StreamNotAdvancing`], which is the opposite situation --
    /// packets arriving that cannot be placed -- and the reason the two have to be
    /// told apart is on the record: on the night of 2026-09-20/21 a front end that
    /// went quiet for the rest of the night was reported as a frame-numbering
    /// mismatch, and a run that had produced 16404 good rows was stopped with a
    /// paragraph about the wrong cause.
    ///
    /// Stopping here is a choice, not a necessity: left alone the run would wait
    /// for the front end and, if it did not come back, spend its whole zero-fill
    /// budget writing rows in which every frame is fabricated (marked as such in
    /// blocks.csv, which is what that budget is for). Seconds-scale gaps are
    /// absorbed without this firing at all.
    StreamQuiet {
        seconds: u64,
        start_frame: u64,
        progress: u64,
        n_station: usize,
    },
    /// A receive thread hit an error it could not fix.
    Fatal { station: usize, message: String },
    /// The output file set for a new local day could not be opened. The run stops
    /// rather than keeping the day's rows in the previous day's files: a set whose
    /// name says one day and whose rows are another is silent wrong data for
    /// anyone reading by day (docs/DESIGN-daily-files.md §3).
    RolloverFailed { prefix: String, detail: String },
}

impl StopReason {
    /// A one-line explanation, phrased for a person reading a run log.
    pub fn describe(&self) -> String {
        match self {
            StopReason::Signal => "interrupted".to_string(),
            StopReason::Duration => "duration elapsed".to_string(),
            StopReason::ZeroFillCap {
                name,
                station,
                fabricated,
                limit,
            } => format!(
                "station {name} (index {station}) exceeded the zero-fill cap: {fabricated} \
                 frames requested, limit {limit}. The rows written so far record how many \
                 frames each one is missing, in blocks.csv"
            ),
            StopReason::LiftCap {
                seconds,
                slots_milli,
                station,
            } => format!(
                "the watermark stayed lifted for {seconds}s: station index {station} has been \
                 left behind by {:.2} slot(s), so its frames are recorded as missing. A \
                 lifted watermark does not clear itself -- every station advances at the same \
                 rate, so the gap only closes on a restart. Stopping now keeps the run from \
                 writing hours of data that is short on those frames; blocks.csv carries the \
                 per-row count for the rows already written",
                *slots_milli as f64 / 1000.0
            ),
            StopReason::StreamNotAdvancing {
                start_frame,
                progress,
                seconds,
                n_station,
            } => format!(
                "the run's frame numbering has not advanced for {seconds}s while packets are \
                 still arriving at all {n_station} stations: the rows are numbered from frame \
                 {start_frame} and every station is stuck at {progress}. Packets that arrive \
                 below the start frame, or behind a frame already finalized, are dropped \
                 without being counted as lost or zero filled -- so a run in this state reads \
                 its sockets, loses nothing on paper and writes no rows. Packets are arriving \
                 and being discarded, which is what rules out a quiet front end (that is \
                 reported as StreamQuiet instead). The usual cause is the front end's counter \
                 restarting (a new epoch) part way through, or a resume of the day's set at a \
                 frame the stream will never reach again. Start a new set \
                 (--allow-second-set, or NEW_SET=1 for tools/run-24-nf16384.sh) after \
                 checking the front end, and keep this set's rows: they stop at the epoch \
                 change"
            ),
            StopReason::StreamQuiet {
                seconds,
                start_frame,
                progress,
                n_station,
            } => format!(
                "no packet has arrived at any of the {n_station} stations for {seconds}s: the \
                 front end stopped, or the path to it did (the run's rows are numbered from \
                 frame {start_frame} and every station is parked at {progress}, waiting). \
                 Stopping rather than waiting it out, because a front end that does not come \
                 back would otherwise spend the whole zero-fill budget on rows where every \
                 frame is fabricated. Check the front end, then start the run again: this \
                 set's rows are complete up to the gap, and blocks.csv shows where it is. \
                 CORR_STREAM_QUIET_S changes how long a quiet front end is tolerated"
            ),
            StopReason::Fatal { station, message } => format!("station {station}: {message}"),
            StopReason::RolloverFailed { prefix, detail } => format!(
                "the output file set for the new local day ({prefix}) could not be opened: \
                 {detail}. The rows of the new day are not in any file: nothing was written \
                 into the previous day's set after the boundary"
            ),
        }
    }
}

/// The run's stop flag and the first reason for it.
pub struct RunStop {
    stopped: AtomicBool,
    reason: Mutex<Option<StopReason>>,
}

impl RunStop {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            stopped: AtomicBool::new(false),
            reason: Mutex::new(None),
        })
    }

    /// Record a reason and set the flag. The first caller wins; later ones only
    /// keep the flag set.
    pub fn stop(&self, reason: StopReason) {
        if let Ok(mut slot) = self.reason.lock() {
            if slot.is_none() {
                *slot = Some(reason);
            }
        }
        self.stopped.store(true, Ordering::Relaxed);
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }

    pub fn reason(&self) -> Option<StopReason> {
        self.reason.lock().ok().and_then(|r| r.clone())
    }

    /// The fatal-error wording the run report has always printed, or `None` if the
    /// run did not stop that way.
    pub fn fatal_message(&self) -> Option<String> {
        match self.reason() {
            Some(r @ StopReason::Fatal { .. }) => Some(r.describe()),
            _ => None,
        }
    }

    /// The zero-fill-cap wording the run report has always printed.
    /// The lift-cap explanation, when that is what stopped the run.
    pub fn lift_message(&self) -> Option<String> {
        match self.reason() {
            Some(r @ StopReason::LiftCap { .. }) => Some(r.describe()),
            _ => None,
        }
    }

    /// The stream/run frame mismatch, reported on the same "STOPPED:" line as the
    /// other reasons a run ends without being asked to: a reader of the report has
    /// one place to look, and this one in particular must not be silent -- a run
    /// stopped here wrote no rows at all, or stopped writing them part way.
    pub fn stream_message(&self) -> Option<String> {
        match self.reason() {
            Some(r @ StopReason::StreamNotAdvancing { .. }) => Some(r.describe()),
            _ => None,
        }
    }

    /// The front end going quiet, on the same `STOPPED:` line as the rest.
    pub fn stream_quiet_message(&self) -> Option<String> {
        match self.reason() {
            Some(r @ StopReason::StreamQuiet { .. }) => Some(r.describe()),
            _ => None,
        }
    }

    pub fn zero_fill_message(&self) -> Option<String> {
        match self.reason() {
            Some(r @ StopReason::ZeroFillCap { .. }) => Some(r.describe()),
            _ => None,
        }
    }

    /// A file set that could not be opened for the new local day.
    ///
    /// Reported through the same "STOPPED:" line as the other fatals, because a
    /// reader of the report needs one place to look for why the run ended; what
    /// distinguishes it is the sentence, and that nothing of the new day is in any
    /// file (docs/DESIGN-daily-files.md §3).
    pub fn rollover_message(&self) -> Option<String> {
        match self.reason() {
            Some(r @ StopReason::RolloverFailed { .. }) => Some(r.describe()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_reason_wins() {
        let stop = RunStop::new();
        assert!(!stop.is_stopped() && stop.reason().is_none());
        stop.stop(StopReason::Signal);
        stop.stop(StopReason::Fatal {
            station: 3,
            message: "too late".into(),
        });
        assert!(stop.is_stopped());
        assert_eq!(stop.reason(), Some(StopReason::Signal));
        assert!(stop.fatal_message().is_none(), "the later reason is not reported");
    }

    #[test]
    fn the_report_wording_matches_the_reason() {
        let stop = RunStop::new();
        stop.stop(StopReason::ZeroFillCap {
            name: "W17".into(),
            station: 7,
            fabricated: 100,
            limit: 50,
        });
        let msg = stop.zero_fill_message().expect("a zero-fill stop");
        // The name is what an operator keys on; the index is the fallback.
        assert!(msg.contains("W17") && msg.contains("index 7"));
        assert!(msg.contains("100") && msg.contains("50"));
        assert!(stop.fatal_message().is_none());
    }
}
