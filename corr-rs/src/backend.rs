//! The seam between the run and *how* packets are received.
//!
//! Everything after the sockets -- the staging ring, the row scheduler, the GPU
//! hand-off, the writer and the report -- does not care whether one station has a
//! thread of its own or shares an event loop with twenty others. It only needs
//! three things from the receive side: that frames land in the staging ring, that
//! each station's progress is published, and that the loop can be stopped and
//! joined.
//!
//! That is what this trait is. It exists so that a second receiver (an async,
//! readiness-driven one that multiplexes many streams over few threads -- see
//! `corr-rs-async`) can be compared against the thread-per-station one without
//! duplicating the orchestration, the statistics or the report: two receivers
//! behind the same contract differ in exactly one variable, which is what makes a
//! measurement between them mean anything.
//!
//! What a backend may NOT do: read a packet without publishing the station's
//! progress first (see `docs/ARCHITECTURE.md` section 3.1). The orchestrator hands
//! the same `Arc<StationProgress>` the scheduler reads, so that rule is unchanged
//! by which backend is in use.

use std::sync::Arc;

use anyhow::Result;

use crate::assemble::SlotRing;
use crate::config::{Config, RunPlan};
use crate::net::MulticastSocket;
use crate::receiver::RecvOutcome;
use crate::run::StagingPtr;
use crate::sched::StationProgress;
use crate::stop::RunStop;

/// Everything a receive backend is handed once the sockets are open, the counters
/// have been sniffed and the starting frame is known.
///
/// Owned values are moved in: the sockets and the staging pointers are the
/// backend's for the life of the run. The `Arc`s are the same objects the
/// scheduler and the report hold.
pub(crate) struct RecvSetup<'a> {
    /// The run's configuration, for the receive-side parameters
    /// (`integration.zero_fill_limit_frames`, the receiver section).
    pub(crate) cfg: &'a Config,
    /// The compiled plan: station table, geometry, per-station receive cores.
    pub(crate) plan: &'a RunPlan,
    /// One socket per enabled station, in station order.
    pub(crate) sockets: Vec<MulticastSocket>,
    /// Where each station's region of the pinned staging starts, in station order.
    pub(crate) staging: Vec<StagingPtr>,
    /// The shared staging ring: a slot may not be written before it is free.
    pub(crate) ring: Arc<SlotRing>,
    /// One progress record per station, in station order.
    pub(crate) progress: Vec<Arc<StationProgress>>,
    /// The run's stop flag and reason.
    pub(crate) stop: Arc<RunStop>,
    /// First frame id of the run: everything before it is discarded, and it is
    /// what the ring is primed against.
    pub(crate) start_frame: u64,
    /// Bytes per frame of one station (`fft_len`).
    pub(crate) frame_stride: usize,
    /// Packets that make up one frame.
    pub(crate) packets_per_frame: u64,
    /// Frames per staging slot.
    pub(crate) frames_per_slot: u64,
    /// Offset of the validity mask inside a station's staging region, and its
    /// size per slot; zero disables the mask (a dry run has no device to read it).
    pub(crate) validity_offset: usize,
    pub(crate) validity_bytes_per_slot: usize,
}

/// A way of getting packets from sockets into the staging ring.
///
/// `start` must not return until every station is being received, and `finish`
/// must not return until every receiver has stopped: the orchestrator frees the
/// staging right after it, so a receiver that outlives its join is a
/// use-after-free, not a slow shutdown. `finish` is called after
/// [`SlotRing::abort`], which is what releases anyone still parked on the ring.
pub(crate) trait ReceiveBackend: Send {
    /// Short name for the run log (`threads`, `async`).
    fn name(&self) -> &'static str;

    /// Take ownership of the sockets and start receiving.
    fn start(&mut self, setup: RecvSetup<'_>) -> Result<()>;

    /// Stop and join every receiver, returning one outcome per station **in
    /// station order**.
    fn finish(self: Box<Self>) -> Vec<RecvOutcome>;
}
