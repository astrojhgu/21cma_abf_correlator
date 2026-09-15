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
    /// A receive thread hit an error it could not fix.
    Fatal { station: usize, message: String },
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
            StopReason::Fatal { station, message } => format!("station {station}: {message}"),
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
    pub fn zero_fill_message(&self) -> Option<String> {
        match self.reason() {
            Some(r @ StopReason::ZeroFillCap { .. }) => Some(r.describe()),
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
