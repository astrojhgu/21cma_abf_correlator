//! Row scheduling: deciding when a staging slot is full enough to hand to the
//! GPU, and when an integration row is complete.
//!
//! The scheduler owns no bytes. It watches the per-station progress counters that
//! the receive threads publish and turns them into three decisions:
//!
//! * **which slot to submit** -- a slot is ready when every station has finalized
//!   all of its frames;
//! * **when to close a row** -- when every station has finalized the row's last
//!   frame;
//! * **what to do about a station that stopped** -- ask it to zero fill up to the
//!   point the row needs, rather than waiting for ever.
//!
//! The watermark is `min` over the stations, not the average and not the first
//! station's: a row is only comparable across baselines if every station covers
//! the same frames, so the slowest station is the only safe limit.
//!
//! Time is a parameter rather than a call to `Instant::now`, so the timeout path
//! can be tested by advancing a fake clock instead of sleeping.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Monotonic clock shared by every thread in a run, so that "when did we last
/// hear from this station" can be published as a plain atomic.
fn epoch() -> &'static Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now)
}

/// Nanoseconds since the run's epoch, from the monotonic clock.
///
/// Wall-clock time would be wrong here: it can jump backwards when the system
/// clock is stepped, and a step would look like a station that has been silent
/// for aeons or one that just spoke.
pub fn now_ns() -> u64 {
    epoch().elapsed().as_nanos() as u64
}

/// Nanoseconds between the run's epoch and an `Instant`.
pub fn ns_of(t: Instant) -> u64 {
    t.saturating_duration_since(*epoch()).as_nanos() as u64
}

/// Wall clock in Unix milliseconds. Used only for the output timestamps: those
/// have to line up with other instruments' logs, so they are system time, not
/// the monotonic clock the scheduler reasons with.
pub fn system_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Shared between one receive thread and the scheduler.
#[derive(Debug)]
pub struct StationProgress {
    /// First frame id that is not yet final. Everything below it is either fully
    /// written or zero filled, and may be handed to the GPU.
    progress: AtomicU64,
    /// Frame id the scheduler is asking this station to finalize up to.
    ///
    /// Only used when the station stops sending: the receive thread owns the
    /// staging bytes, so the scheduler cannot zero fill them itself without
    /// racing the next packet. The receive thread honours this whenever it finds
    /// itself with no packets to process.
    advance_to: AtomicU64,
    /// Nanoseconds (via [`now_ns`]) of the last non-empty receive batch.
    ///
    /// This, not the progress counter, is what the stall detector watches.
    /// Progress only moves once per completed staging slot, so a perfectly
    /// healthy station is "silent" for a whole slot at a time, and deciding
    /// liveness from it needs the data rate to know how long a slot takes --
    /// which means trusting a configured sample rate that may not match reality.
    /// Packets arrive continuously (every 5 us at the real rate, every 50 us in a
    /// slow replay), so an arrival-time test is both finer grained and completely
    /// independent of the rate.
    last_packet_ns: AtomicU64,
    /// Cumulative frames this station had to zero fill, published by the receive
    /// thread. The per-row figure is the difference across a row, which is what
    /// `blocks.csv` records.
    zero_filled_frames: AtomicU64,
    /// Cumulative packets the stream never delivered: the counter jumped. Only
    /// the receive thread can know this, and the timeline of it is what tells a
    /// burst loss apart from a steady shortfall.
    lost_packets: AtomicU64,
    /// Cumulative nanoseconds this station spent waiting for the pipeline to
    /// release a staging slot, published by whichever receiver owns it: the
    /// thread path out of `wait_free`, the async path out of its park.
    ///
    /// The live monitor differences it, because "how much of this second was
    /// spent not reading the socket" is the receive side's own measure of how
    /// much jitter margin is left. Waiting is not loss; a run that waits a lot is
    /// one whose socket buffers are the only thing between it and loss.
    slot_wait_ns: AtomicU64,
}

impl StationProgress {
    pub fn new(first_frame_id: u64) -> Self {
        StationProgress {
            progress: AtomicU64::new(first_frame_id),
            advance_to: AtomicU64::new(first_frame_id),
            last_packet_ns: AtomicU64::new(now_ns()),
            zero_filled_frames: AtomicU64::new(0),
            lost_packets: AtomicU64::new(0),
            slot_wait_ns: AtomicU64::new(0),
        }
    }

    /// Publish the assembler's cumulative lost-packet count.
    pub fn set_lost_packets(&self, n: u64) {
        self.lost_packets.store(n, Ordering::Release);
    }

    /// Publish the cumulative time spent waiting for a staging slot.
    pub fn set_slot_wait_ns(&self, ns: u64) {
        self.slot_wait_ns.store(ns, Ordering::Release);
    }

    pub fn slot_wait_ns(&self) -> u64 {
        self.slot_wait_ns.load(Ordering::Acquire)
    }

    pub fn lost_packets(&self) -> u64 {
        self.lost_packets.load(Ordering::Acquire)
    }

    /// Publish the assembler's cumulative zero-filled frame count.
    pub fn set_zero_filled_frames(&self, n: u64) {
        self.zero_filled_frames.store(n, Ordering::Release);
    }

    pub fn zero_filled_frames(&self) -> u64 {
        self.zero_filled_frames.load(Ordering::Acquire)
    }

    /// Called by the receive thread after every non-empty batch.
    pub fn mark_packet(&self) {
        self.last_packet_ns.store(now_ns(), Ordering::Release);
    }

    /// Nanoseconds since the last packet arrived, as of `now_ns`.
    pub fn silence_ns(&self, now_ns: u64) -> u64 {
        now_ns.saturating_sub(self.last_packet_ns.load(Ordering::Acquire))
    }

    /// Test and diagnostic hook: pretend the last packet arrived at `ns`.
    pub fn set_last_packet_ns(&self, ns: u64) {
        self.last_packet_ns.store(ns, Ordering::Release);
    }

    pub fn get(&self) -> u64 {
        self.progress.load(Ordering::Acquire)
    }

    pub fn set(&self, frame_id: u64) {
        self.progress.store(frame_id, Ordering::Release);
    }

    /// Requested finalization target, or `None` if there is no outstanding ask.
    pub fn requested(&self) -> u64 {
        self.advance_to.load(Ordering::Acquire)
    }

    /// Ask the station to finalize up to `frame_id`. Never moves the request
    /// backwards.
    pub fn request_advance_to(&self, frame_id: u64) {
        let mut cur = self.advance_to.load(Ordering::Relaxed);
        while cur < frame_id {
            match self.advance_to.compare_exchange_weak(
                cur,
                frame_id,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Clear an advance request once the receive thread has honoured it.
    pub fn clear_request(&self, up_to: u64) {
        let _ = self
            .advance_to
            .compare_exchange(up_to, up_to, Ordering::Release, Ordering::Relaxed);
    }
}

/// Set `CORR_SCHED_DEBUG=1` to have every stall recovery explain itself. Used
/// when a run misbehaves and the numbers alone do not say which condition fired.
fn debug_enabled() -> bool {
    std::env::var_os("CORR_SCHED_DEBUG").is_some()
}

/// Reported when the scheduler has asked for more zero filling than the
/// configuration allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroFillExceeded {
    pub station: usize,
    pub frames: u64,
    pub limit: u64,
}

impl std::fmt::Display for ZeroFillExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "station {} has been asked to zero fill {} frames, past the limit of {}; \
             it has stopped producing data",
            self.station, self.frames, self.limit
        )
    }
}

/// What the pipeline thread should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Hand the staging slot holding `slot_seq` to the GPU. The physical slot
    /// index is the ring's business, not the scheduler's, so the caller derives
    /// it as `slot_seq % block_slots`.
    ///
    /// `frames` is the portion of the slot that belongs to the current row, which
    /// is the whole slot except for the row's last one.
    Submit {
        slot_seq: u64,
        frames: i32,
        /// Index of the row this slot belongs to, for logging.
        row: u64,
    },
    /// Close the current integration row, dividing it by `frames`.
    ///
    /// `open_ms` is the wall clock, in Unix milliseconds, of the moment the row's
    /// first frame became final -- the acquisition time of the row rather than
    /// the time it happened to be written, so a backed-up writer cannot shift it.
    CloseRow {
        frames: u64,
        row: u64,
        open_ms: i64,
    },
    /// Nothing is ready; the caller should wait briefly and step again.
    Idle,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SchedStats {
    pub slots_submitted: u64,
    pub rows_closed: u64,
    /// Distinct recovery *episodes*: one per station per stall timeout, not one
    /// per scheduler step. A station that stays silent for a second should count
    /// as a handful of recoveries, not thousands.
    pub stall_recoveries: u64,
    /// Frames the scheduler asked stations to zero fill, summed over stations.
    /// This is the number that has to stay under the configured cap.
    pub zero_fill_requested_frames: u64,
    /// Frames the scheduler counted as belonging to the run.
    pub frames_scheduled: u64,
    /// Recoveries per station, in station order.
    pub recoveries_per_station: Vec<u64>,
    /// Frames requested per station, in station order.
    pub requested_per_station: Vec<u64>,
}

pub struct RowScheduler {
    progress: Vec<Arc<StationProgress>>,
    frames_per_slot: u64,
    n_cum: u64,
    rows_per_fetch: u32,

    /// First frame not yet handed to the GPU.
    ///
    /// The run starts here, so it is aligned to `frames_per_slot` as well as to
    /// the row grid: `frame_id % frames_per_slot` is the frame's position inside
    /// the staging slot, and the device side always reads a slot from its first
    /// frame. A start that was only row aligned would leave the first slot
    /// covering frames that were never assembled.
    next_frame: u64,
    /// First frame of the row currently being accumulated.
    row_start_frame: u64,
    /// Wall clock when this row's first frame became final, for the output
    /// timestamp. `None` until the watermark reaches the row's first frame.
    row_open_ms: Option<i64>,
    /// Rows closed but not yet fetched.
    rows_pending: u32,

    /// How long a station may deliver no packets at all before it is treated as
    /// having stopped. Purely physical time: no dependence on the data rate.
    stall_timeout: Duration,
    /// See [`RowSchedulerParams::max_spread_frames`].
    max_spread_frames: Option<u64>,
    /// When the watermark was first lifted above the slowest station, or `None` while it
    /// is not lifted.
    ///
    /// Only a clock can answer "how long have we been leaving that station behind".
    /// Cleared rather than paused when the lift goes away, so a station that recovers
    /// and falls behind again gets a fresh interval instead of inheriting the old one.
    lift_since: Option<Instant>,

    /// Per station: last progress value seen, and when it changed.
    ///
    /// The stall detector is per station rather than global on purpose. A global
    /// "the watermark has not moved" test cannot tell a station that died from a
    /// front end that is simply running slowly -- and firing on the latter
    /// finalizes frames whose packets are still in flight, which zero fills good
    /// data. That is not hypothetical: at a fifth of the real packet rate with a
    /// 30 ms timeout it fabricated thousands of zero-filled frames on a link with
    /// zero packet loss.
    last_progress: Vec<u64>,
    last_moved: Vec<Instant>,
    /// Last time each station was asked to advance, so one long silence counts as
    /// repeated asks spaced by the timeout rather than one per scheduler step.
    last_recovery: Vec<Instant>,
    /// Frames already requested per station, and the cap on them.
    requested_frames: Vec<u64>,
    block_slots: u32,
    zero_fill_limit: u64,
    /// Set when the zero-fill cap is exceeded; the caller must stop cleanly.
    zero_fill_exceeded: Option<ZeroFillExceeded>,

    stats: SchedStats,
}

/// The fixed geometry and policy a scheduler runs with.
#[derive(Debug, Clone, Copy)]
pub struct RowSchedulerParams {
    pub frames_per_slot: u64,
    /// Frames per integration row. Must be a multiple of `frames_per_slot`.
    pub n_cum: u64,
    pub rows_per_fetch: u32,
    /// Must be aligned to `n_cum`, which `Config::validate` enforces, so that
    /// every row covers the same frames on every run.
    pub start_frame_id: u64,
    /// How long a station may deliver no packets before it counts as stopped.
    /// Physical time, so it does not depend on the data rate.
    pub stall_timeout: Duration,
    /// Cap on zero filling one station. 0 disables it.
    pub zero_fill_limit_frames: u64,
    /// Staging slots in the receive ring.
    ///
    /// Needed to tell a station that has *stopped* from one that is merely
    /// blocked: a receive thread whose ring is full cannot refresh its liveness
    /// stamp, so it looks silent. A blocked station sits at the ring ceiling,
    /// which is `block_slots` slots ahead of `next_frame`, whereas a dead one
    /// sits at `next_frame`. Without this, one station stalling makes the
    /// scheduler zero fill every other station too -- measured, 28 spurious
    /// recoveries and 3593 good frames discarded during a 1.5 s stall.
    pub block_slots: u32,
    /// How far behind the slowest station the watermark may sit, in frames. `None`
    /// means "wait for ever", which is what this did before the cap existed.
    ///
    /// See `integration.max_spread_slots` in the configuration: without a cap, a
    /// station that falls behind can never be caught up with, because every station
    /// advances at the same rate -- so the gap between them is a ratchet.
    pub max_spread_frames: Option<u64>,
}

impl RowScheduler {
    pub fn new(
        progress: Vec<Arc<StationProgress>>,
        p: RowSchedulerParams,
        now: Instant,
    ) -> Self {
        let RowSchedulerParams {
            frames_per_slot,
            n_cum,
            rows_per_fetch,
            start_frame_id,
            stall_timeout,
            zero_fill_limit_frames: zero_fill_limit,
            block_slots,
            max_spread_frames,
        } = p;
        let n = progress.len();
        assert!(!progress.is_empty(), "a scheduler with no stations");
        assert!(frames_per_slot >= 1 && n_cum >= 1);
        assert!(rows_per_fetch >= 1);
        assert_eq!(
            n_cum % frames_per_slot,
            0,
            "one row must be a whole number of staging slots. The device side always \
             reads a slot from its first frame, so a row that starts in the middle of \
             a slot would have its earliest frames addressed at the wrong offset. \
             Config::validate enforces this, so reaching here means a caller built a \
             scheduler directly"
        );
        debug_assert_eq!(
            start_frame_id % frames_per_slot,
            0,
            "a slot-aligned row length makes a row-aligned start slot aligned too"
        );
        RowScheduler {
            progress,
            frames_per_slot,
            n_cum,
            rows_per_fetch,
            next_frame: start_frame_id,
            row_start_frame: start_frame_id,
            row_open_ms: None,
            rows_pending: 0,
            stall_timeout,
            last_progress: vec![start_frame_id; n],
            last_moved: vec![now; n],
            // "Long ago", so that the first recovery is not delayed by the rate
            // limiter for a station that has never delivered anything.
            last_recovery: vec![now.checked_sub(stall_timeout).unwrap_or(now); n],
            requested_frames: vec![0; n],
            block_slots,
            zero_fill_limit,
            max_spread_frames,
            lift_since: None,
            zero_fill_exceeded: None,
            // Sized up front: a scheduler indexes these by station on a hot path
            // and a default empty Vec would panic the first time it fires.
            stats: SchedStats {
                recoveries_per_station: vec![0; n],
                requested_per_station: vec![0; n],
                ..SchedStats::default()
            },
        }
    }

    pub fn stats(&self) -> SchedStats {
        self.stats.clone()
    }

    /// How long a station may be silent before the scheduler treats it as gone.
    pub fn stall_timeout(&self) -> Duration {
        self.stall_timeout
    }

    /// Lowest station progress: the only frame id all stations have finalized.
    pub fn watermark(&self) -> u64 {
        let (min, max) = self.progress_range();
        match self.max_spread_frames {
            // The cap: a station further behind than this stops being waited for. Its
            // frames become missing frames -- the same outcome `stall_timeout`
            // reaches, arrived at by distance instead of by silence.
            Some(cap) => min.max(max.saturating_sub(cap)),
            None => min,
        }
    }

    /// How far the watermark was lifted above the slowest station, in frames.
    ///
    /// Zero unless the cap is doing something, and non-zero is worth reporting: it
    /// says a station is being left behind *on purpose*, so its rows carry more zero
    /// fill than the others and a reader of `blocks.csv` should know why.
    pub fn watermark_lift(&self) -> u64 {
        if self.max_spread_frames.is_none() {
            return 0;
        }
        let (min, _) = self.progress_range();
        self.watermark().saturating_sub(min)
    }

    /// Frames in one staging slot, for callers that report a gap in slots.
    pub fn frames_per_slot(&self) -> u64 {
        self.frames_per_slot
    }

    /// Note whether the watermark is lifted as of `now`, and report how long it has been.
    ///
    /// Called about once a second by the pipeline, which is the thread that has both a
    /// clock and spare time. Deliberately not done in `step`: that runs per staging slot
    /// on the hot path, and a stop condition measured in seconds does not need that rate.
    ///
    /// Zero means "not lifted right now" -- either it never was, or it recovered, in
    /// which case the clock is cleared rather than left running.
    pub fn note_lift(&mut self, now: Instant) -> Duration {
        if self.max_spread_frames.is_none() {
            return Duration::ZERO;
        }
        if self.watermark_lift() > 0 {
            let since = *self.lift_since.get_or_insert(now);
            now.saturating_duration_since(since)
        } else {
            self.lift_since = None;
            Duration::ZERO
        }
    }

    /// Every station's progress relative to the watermark, in frames and in station
    /// order: positive is ahead of the watermark, negative is behind it.
    ///
    /// A station behind the watermark is the one whose frames are being recorded as
    /// missing, so this is also the answer to "which station is holding the run
    /// back" -- which `spread_slots` alone cannot say, because it is a distance
    /// without a direction or a name.
    pub fn progress_offsets(&self) -> Vec<i64> {
        let wm = self.watermark();
        self.progress
            .iter()
            .map(|p| p.get() as i64 - wm as i64)
            .collect()
    }

    /// The configured cap in frames, for callers that report in staging slots.
    pub fn max_spread_frames(&self) -> Option<u64> {
        self.max_spread_frames
    }

    /// Slowest and fastest station progress in one pass: this is on the hot path.
    fn progress_range(&self) -> (u64, u64) {
        let mut min = u64::MAX;
        let mut max = 0;
        for p in &self.progress {
            let v = p.get();
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        (min, max)
    }

    /// Which station currently holds the watermark, for diagnostics.
    pub fn watermark_station(&self) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;
        for (i, p) in self.progress.iter().enumerate() {
            let v = p.get();
            if best.map(|(_, bv)| v < bv).unwrap_or(true) {
                best = Some((i, v));
            }
        }
        best.map(|(i, _)| i)
    }

    pub fn highest_progress(&self) -> u64 {
        self.progress.iter().map(|p| p.get()).max().unwrap()
    }

    pub fn row_start_frame(&self) -> u64 {
        self.row_start_frame
    }

    /// How long station `i` has delivered no packets, in milliseconds.
    pub fn silence_ms(&self, i: usize, now: Instant) -> u64 {
        self.progress[i].silence_ns(ns_of(now)) / 1_000_000
    }

    /// What station `i` has been asked to finalize up to, if anything.
    pub fn requested_of(&self, i: usize) -> u64 {
        self.progress[i].requested()
    }

    /// Cumulative packets this station's stream lost, as published by its thread.
    pub fn lost_of(&self, i: usize) -> u64 {
        self.progress[i].lost_packets()
    }

    /// Cumulative frames this station zero filled.
    pub fn zero_filled_of(&self, i: usize) -> u64 {
        self.progress[i].zero_filled_frames()
    }

    /// This station's published progress.
    pub fn progress_of(&self, i: usize) -> u64 {
        self.progress[i].get()
    }

    /// Tell the caller that a row has been handed off and is no longer pending
    /// here; the caller tracks its own fetch accounting.
    pub fn note_row_fetched(&mut self, rows: u32) {
        self.rows_pending = self.rows_pending.saturating_sub(rows);
    }

    pub fn rows_pending(&self) -> u32 {
        self.rows_pending
    }

    /// Decide what to do next.
    ///
    /// `now` drives the stall detector. When the watermark has not moved for
    /// `stall_timeout`, every station behind the next slot boundary is asked to
    /// zero fill up to it, which lets the row finish instead of hanging on a
    /// station that has gone quiet.
    pub fn step(&mut self, now: Instant) -> Action {
        // Remember when each station last moved.
        for (i, p) in self.progress.iter().enumerate() {
            let v = p.get();
            if v > self.last_progress[i] {
                self.last_progress[i] = v;
                self.last_moved[i] = now;
            }
        }
        self.recover_from_stall(now, ns_of(now));

        let watermark = self.watermark();
        let row_end = self.row_start_frame + self.n_cum;

        // Stamp the row's acquisition time the first moment its first frame is
        // final. Doing it at close time instead would add a whole row period to
        // every timestamp.
        if self.row_open_ms.is_none() && watermark >= self.row_start_frame {
            self.row_open_ms = Some(system_ms());
        }

        // Everything this row needed has been submitted. Close it once the
        // watermark confirms every station finalized it.
        if self.next_frame >= row_end {
            if watermark >= row_end {
                return self.close_row();
            }
            return Action::Idle;
        }

        // Submit the slot that `next_frame` lives in. Its usable length is capped
        // by the end of the row, so a row whose length is not a multiple of the
        // slot size ends with a partial slot -- and never overruns into the next
        // row's frames.
        let slot_seq = self.next_frame / self.frames_per_slot;
        // n_cum being a multiple of frames_per_slot is what makes this always the
        // whole slot: a row never starts or ends part way through one.
        let slot_end = ((slot_seq + 1) * self.frames_per_slot).min(row_end);
        if watermark >= slot_end {
            let frames = (slot_end - self.next_frame) as i32;
            self.next_frame = slot_end;
            self.stats.slots_submitted += 1;
            self.stats.frames_scheduled += frames as u64;
            return Action::Submit {
                slot_seq,
                frames,
                row: self.row_start_frame / self.n_cum,
            };
        }

        Action::Idle
    }

    fn close_row(&mut self) -> Action {
        debug_assert!(
            self.next_frame >= self.row_start_frame + self.n_cum,
            "closing a row whose frames were not all submitted would silently drop them"
        );
        let frames = self.n_cum;
        let row = self.row_start_frame / self.n_cum;
        let open_ms = self.row_open_ms.take().unwrap_or_else(system_ms);
        self.row_start_frame += self.n_cum;
        self.rows_pending += 1;
        self.stats.rows_closed += 1;
        Action::CloseRow { frames, row, open_ms }
    }

    /// Ask a genuinely silent station to zero fill, so the run can continue.
    ///
    /// Two conditions, both necessary:
    ///
    /// * the station has made **no progress** for `stall_timeout`, and
    /// * another station has moved at least one slot further along.
    ///
    /// The first is what distinguishes a dead station from a slow one. It is not
    /// enough on its own: a station whose staging ring has filled up makes no
    /// progress either, and that is backpressure from the pipeline, not a fault.
    /// The second condition is what separates the two cases -- a blocked station
    /// is blocked because the watermark has not reached it, so nothing is ahead
    /// of it, whereas a dead station falls behind everything.
    ///
    /// Requests are rate limited to one per station per timeout, so a station
    /// that stays silent for a second is a handful of episodes rather than one
    /// per scheduler iteration.
    fn recover_from_stall(&mut self, now: Instant, now_ns: u64) {
        let slot_seq = self.next_frame / self.frames_per_slot;
        let row_end = self.row_start_frame + self.n_cum;
        // Never past the end of the row: a lagging station must not be advanced
        // into frames this row does not own.
        let target = ((slot_seq + 1) * self.frames_per_slot).min(row_end);
        if target <= self.next_frame && self.next_frame >= row_end {
            return;
        }

        let highest = self.highest_progress();
        let mut asked_any = false;
        for i in 0..self.progress.len() {
            let cur = self.progress[i].get();
            if cur >= target {
                continue;
            }
            // Liveness is judged from when the station last *delivered a packet*,
            // not from when its progress counter last moved. Packets arrive
            // continuously, so this needs no knowledge of the data rate.
            if self.progress[i].silence_ns(now_ns) < self.stall_timeout.as_nanos() as u64 {
                continue;
            }
            if highest.saturating_sub(cur) < self.frames_per_slot {
                // Nobody is ahead of this station, so it is the pipeline holding
                // it back, not a dead front end.
                continue;
            }
            // A station at the staging ring ceiling is blocked, not dead: its
            // receive thread is inside the slot wait and cannot refresh its
            // liveness stamp, while its socket is still filling up with data.
            // Recovering it would throw that data away.
            let ring_ceiling = self.next_frame
                + (self.block_slots as u64 - 1) * self.frames_per_slot;
            if cur >= ring_ceiling {
                continue;
            }
            if now.saturating_duration_since(self.last_recovery[i]) < self.stall_timeout {
                continue;
            }
            if debug_enabled() {
                eprintln!(
                    "sched: recover st={i} progress={cur} target={target} highest={highest} \
                     silent_ms={} ahead_of_watermark={}",
                    self.progress[i].silence_ns(now_ns) / 1_000_000,
                    highest.saturating_sub(cur),
                );
            }
            self.last_recovery[i] = now;
            self.progress[i].request_advance_to(target);
            let frames = target - cur;
            self.requested_frames[i] += frames;
            self.stats.recoveries_per_station[i] += 1;
            self.stats.requested_per_station[i] += frames;
            self.stats.zero_fill_requested_frames += frames;
            self.stats.stall_recoveries += 1;
            asked_any = true;

            if self.zero_fill_limit > 0 && self.requested_frames[i] > self.zero_fill_limit {
                self.zero_fill_exceeded = Some(ZeroFillExceeded {
                    station: i,
                    frames: self.requested_frames[i],
                    limit: self.zero_fill_limit,
                });
            }
        }
        let _ = asked_any;
    }

    /// Set once a station has been asked to zero fill more frames than the
    /// configured cap allows. The caller must stop cleanly: past this point the
    /// output would be rows of silence dressed up as observations, and the
    /// receive-side buffers would keep growing to hold data nobody will use.
    pub fn zero_fill_exceeded(&self) -> Option<ZeroFillExceeded> {
        self.zero_fill_exceeded
    }

    /// True if a row could be closed right now.
    pub fn row_ready(&self) -> bool {
        self.next_frame >= self.row_start_frame + self.n_cum
            && self.watermark() >= self.row_start_frame + self.n_cum
    }

    /// First frame not yet handed to the GPU.
    pub fn next_frame(&self) -> u64 {
        self.next_frame
    }

    /// True if all `rows_per_fetch` slots of the device row stack are full.
    pub fn fetch_ready(&self) -> bool {
        self.rows_pending >= self.rows_per_fetch
    }
}

/// Number of staging slots a row spans, for logging and for sizing expectations.
pub fn slots_per_row(n_cum: u64, frames_per_slot: u64) -> u64 {
    n_cum.div_ceil(frames_per_slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FPS: u64 = 256;
    const N_CUM: u64 = 1024; // 4 whole slots; Config::validate enforces the multiple
    const START: u64 = 1_024_000; // a multiple of both N_CUM and FPS
    const TIMEOUT_MS: u64 = 200;

    struct Fixture {
        stations: Vec<Arc<StationProgress>>,
        sched: RowScheduler,
        t: Instant,
    }

    impl Fixture {
        fn new(n_station: usize, rows_per_fetch: u32) -> Self {
            let stations: Vec<_> = (0..n_station)
                .map(|_| Arc::new(StationProgress::new(START)))
                .collect();
            let t = Instant::now();
            let sched = RowScheduler::new(
                stations.clone(),
                RowSchedulerParams {
                    frames_per_slot: FPS,
                    n_cum: N_CUM,
                    rows_per_fetch,
                    start_frame_id: START,
                    stall_timeout: Duration::from_millis(TIMEOUT_MS),
                    zero_fill_limit_frames: 0,
                    block_slots: 3,
                    max_spread_frames: None,
                },
                t,
            );
            Fixture { stations, sched, t }
        }

        /// Progress moving means packets arrived, so the arrival stamp moves with
        /// it. Keeping the two in step is what the production receive thread does.
        fn advance_one(&mut self, idx: usize, frames: u64) {
            let p = &self.stations[idx];
            p.set(p.get() + frames);
            p.set_last_packet_ns(ns_of(self.t));
        }

        fn advance_all(&mut self, frames: u64) {
            for i in 0..self.stations.len() {
                self.advance_one(i, frames);
            }
        }

        /// Record a packet arrival at the fixture's fake clock, which is what the
        /// scheduler reads. `mark_packet()` would use the real clock and the two
        /// would disagree the moment the fake clock advances.
        fn mark(&mut self, idx: usize) {
            self.stations[idx].set_last_packet_ns(ns_of(self.t));
        }

        /// Stop delivering packets for this station, without changing its
        /// progress: exactly what a dead front end looks like. The caller then
        /// advances the clock past the timeout to make the silence visible.
        fn stop_delivering(&mut self, idx: usize) {
            self.mark(idx);
        }

        /// Step without advancing the clock, collecting everything ready.
        fn drain(&mut self) -> Vec<Action> {
            let mut v = Vec::new();
            loop {
                match self.sched.step(self.t) {
                    Action::Idle => break,
                    a => v.push(a),
                }
            }
            v
        }

        fn tick(&mut self, ms: u64) -> Action {
            self.t += Duration::from_millis(ms);
            self.sched.step(self.t)
        }

        fn next_frame(&self) -> u64 {
            self.sched.next_frame()
        }
    }

    fn submits(actions: &[Action]) -> Vec<(u64, i32, u64)> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Submit { slot_seq, frames, row } => Some((*slot_seq, *frames, *row)),
                _ => None,
            })
            .collect()
    }

    fn closes(actions: &[Action]) -> Vec<(u64, u64)> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::CloseRow { frames, row, .. } => Some((*frames, *row)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_lift_clock_starts_clears_and_restarts() {
        let mut f = Fixture::new(2, 1);
        f.advance_all(10 * N_CUM);
        f.sched.max_spread_frames = Some(FPS); // one slot

        // Level: nothing is being skipped, so there is nothing to time.
        assert_eq!(f.sched.note_lift(f.t), Duration::ZERO);

        // One station pulls a whole slot ahead: the clock starts at zero, not at the
        // fixture's start time.
        f.advance_one(0, 2 * FPS);
        assert!(f.sched.watermark_lift() > 0);
        assert_eq!(f.sched.note_lift(f.t), Duration::ZERO);

        let after_30 = f.t + Duration::from_secs(30);
        assert_eq!(f.sched.note_lift(after_30), Duration::from_secs(30));

        // Catching up clears it...
        f.advance_one(1, 2 * FPS);
        assert_eq!(f.sched.watermark_lift(), 0);
        assert_eq!(f.sched.note_lift(after_30), Duration::ZERO);

        // ...and falling behind again starts a fresh interval rather than resuming the
        // old one, which would trip the stop immediately.
        f.advance_one(0, 2 * FPS);
        assert_eq!(f.sched.note_lift(after_30), Duration::ZERO);
        let after_35 = after_30 + Duration::from_secs(5);
        assert_eq!(f.sched.note_lift(after_35), Duration::from_secs(5));
    }

    #[test]
    fn progress_offsets_name_the_station_that_is_behind() {
        let mut f = Fixture::new(3, 1);
        f.advance_all(10 * N_CUM);
        // One station pulls ahead, one drops back: the offsets must carry both the
        // sign and the identity, which `spread_slots` cannot do on its own.
        f.advance_one(1, 3 * FPS);
        let back = f.stations[2].get() - 2 * FPS;
        f.stations[2].set(back);

        let offs = f.sched.progress_offsets();
        assert_eq!(offs.len(), 3);
        // Station 2 is the slowest, so it *is* the watermark and reads zero; the
        // others read how far ahead of it they are. A station that had fallen behind
        // the watermark would read negative, and that is the case the page colours.
        assert_eq!(offs[2], 0, "the watermark is the zero of this scale");
        assert_eq!(offs[0], 2 * FPS as i64);
        assert_eq!(offs[1], 5 * FPS as i64);
        assert_eq!(f.sched.watermark_station(), Some(2));
        assert_eq!(f.sched.max_spread_frames(), None);
    }

    #[test]
    fn a_capped_watermark_moves_the_scale_under_every_station() {
        let mut f = Fixture::new(2, 1);
        f.advance_all(10 * N_CUM);
        f.advance_one(0, 5 * FPS);
        f.sched.max_spread_frames = Some(2 * FPS);

        // The watermark sits at `fastest - cap`, so the fast station reads +cap and
        // the slow one reads exactly `-lift`: the two are the same fact seen from
        // either end.
        let offs = f.sched.progress_offsets();
        assert_eq!(offs[0], 2 * FPS as i64);
        assert_eq!(offs[1], -(3 * FPS as i64));
        assert_eq!(f.sched.watermark_lift(), 3 * FPS);
        assert_eq!(-offs[1], f.sched.watermark_lift() as i64);
    }

    /// Every station advances at the same rate, so the gap between them never closes
    /// by itself. Without a cap, a station that falls behind holds the watermark for
    /// ever -- and the ring's writable slots with it.
    #[test]
    fn the_watermark_stops_waiting_for_a_station_that_falls_too_far_behind() {
        let mut f = Fixture::new(3, 1);
        f.stations[0].progress.store(START + 10 * FPS, Ordering::Release);
        f.stations[1].progress.store(START + 10 * FPS, Ordering::Release);
        // No cap: the station still at START holds the watermark where it is.
        assert_eq!(f.sched.watermark(), START);
        assert_eq!(f.sched.watermark_lift(), 0, "nothing is being skipped");

        // With a two-slot cap the scheduler stops waiting for it.
        f.sched.max_spread_frames = Some(2 * FPS);
        assert_eq!(f.sched.watermark(), START + 8 * FPS);
        assert_eq!(f.sched.watermark_lift(), 8 * FPS);

        // A cap wider than the actual spread changes nothing.
        f.sched.max_spread_frames = Some(64 * FPS);
        assert_eq!(f.sched.watermark(), START);
        assert_eq!(f.sched.watermark_lift(), 0);
    }

    #[test]
    fn a_row_spans_the_expected_number_of_slots() {
        assert_eq!(slots_per_row(1024, 256), 4);
        assert_eq!(slots_per_row(97792, 256), 382, "the shipped configuration");
        assert_eq!(slots_per_row(97792, 256) * 256, 97792, "no partial slot");
    }

    #[test]
    fn the_watermark_is_the_minimum_not_the_first() {
        let mut f = Fixture::new(3, 1);
        f.advance_one(0, 10 * FPS);
        f.advance_one(1, 3 * FPS);
        f.advance_one(2, 7 * FPS);
        assert_eq!(f.sched.watermark(), START + 3 * FPS);
        assert_eq!(f.sched.highest_progress(), START + 10 * FPS);
        f.advance_one(0, 100 * FPS);
        assert_eq!(f.sched.watermark(), START + 3 * FPS);
    }

    #[test]
    fn slots_are_submitted_only_when_every_station_has_finalized_them() {
        let mut f = Fixture::new(4, 1);
        for i in 1..4 {
            f.advance_one(i, 3 * FPS);
        }
        // Station 0 is one frame short of the first slot, but it is still
        // delivering packets the whole time.
        f.advance_one(0, FPS - 1);
        assert!(f.drain().is_empty(), "the lagging station must hold the watermark");

        f.advance_one(0, 1);
        assert_eq!(submits(&f.drain()), vec![(START / FPS, FPS as i32, START / N_CUM)]);
    }

    #[test]
    fn a_run_start_that_is_not_slot_aligned_is_refused() {
        let stations: Vec<_> = (0..2)
            .map(|_| Arc::new(StationProgress::new(START + 1)))
            .collect();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RowScheduler::new(
                stations,
                RowSchedulerParams {
                    frames_per_slot: FPS,
                    n_cum: N_CUM + 1,
                    rows_per_fetch: 1,
                    start_frame_id: START + 1,
                    stall_timeout: Duration::from_millis(TIMEOUT_MS),
                    zero_fill_limit_frames: 0,
                    block_slots: 3,
                    max_spread_frames: None,
                },
                Instant::now(),
            )
        }));
        assert!(r.is_err(), "a row length that is not a whole number of slots must be refused");
    }

    #[test]
    fn a_row_is_closed_only_after_all_of_its_frames_were_submitted() {
        let mut f = Fixture::new(2, 1);
        let row = START / N_CUM;
        f.advance_all(N_CUM - 1);
        let a = f.drain();
        assert_eq!(submits(&a).len(), 3);
        assert!(closes(&a).is_empty(), "a row must not close early");
        assert_eq!(f.sched.watermark(), START + N_CUM - 1);

        f.advance_all(1);
        let a = f.drain();
        assert_eq!(submits(&a), vec![(START / FPS + 3, FPS as i32, row)]);
        assert_eq!(closes(&a), vec![(N_CUM, row)]);
        assert_eq!(f.sched.row_start_frame(), START + N_CUM);
        assert_eq!(f.next_frame(), START + N_CUM, "no frame is submitted twice");
    }

    #[test]
    fn a_rows_slots_never_overrun_into_the_next_row() {
        let mut f = Fixture::new(2, 1);
        f.advance_all(3 * N_CUM);
        let a = f.drain();
        let subs = submits(&a);
        let cls = closes(&a);
        assert_eq!(cls.len(), 3, "three rows should close");
        assert_eq!(cls.iter().map(|(fr, _)| *fr).sum::<u64>(), 3 * N_CUM);
        for (row, chunk) in subs.chunk_by(|x, y| x.2 == y.2).map(|c| (c[0].2, c)) {
            let total: i64 = chunk.iter().map(|(_, fr, _)| *fr as i64).sum();
            assert_eq!(total, N_CUM as i64, "row {row} submitted {total} frames");
        }
        for (i, (_, frames, _)) in subs.iter().enumerate() {
            assert_eq!(*frames as u64, FPS, "slot {i} should be full");
        }
    }

    #[test]
    fn a_watermark_jump_never_skips_a_slot() {
        let mut f = Fixture::new(2, 1);
        f.advance_all(5 * N_CUM);
        let a = f.drain();
        let subs = submits(&a);
        assert_eq!(subs.len(), 5 * 4, "5 rows x 4 whole slots");
        assert_eq!(closes(&a).len(), 5);
        for (i, (slot_seq, _, _)) in subs.iter().enumerate() {
            assert_eq!(*slot_seq, START / FPS + i as u64, "slot {i} is out of order");
        }
        let mut seen = std::collections::BTreeMap::new();
        for (idx, act) in a.iter().enumerate() {
            match act {
                Action::Submit { row, .. } => {
                    seen.insert(*row, idx);
                }
                Action::CloseRow { row, .. } => {
                    assert!(
                        seen.get(row).map(|i| *i < idx).unwrap_or(false),
                        "row {row} closed before all of its frames were submitted"
                    );
                }
                Action::Idle => {}
            }
        }
    }

    #[test]
    fn liveness_is_judged_by_packet_arrival_not_by_progress() {
        // The regression this guards: progress only moves once per completed
        // staging slot, so a healthy station is "silent" for a whole slot at a
        // time. Deciding liveness from that needs the data rate, and any mismatch
        // between the configured rate and the real one turns every healthy
        // station into a dead one -- measured, that zero filled thousands of good
        // frames on a link that lost nothing.
        let mut f = Fixture::new(2, 1);
        f.advance_one(1, 3 * FPS);

        // Station 0's progress advances only once every 300 ms -- slower than the
        // 200 ms timeout -- but packets keep arriving throughout, which is what a
        // real receive thread looks like.
        for _ in 0..10 {
            for _ in 0..6 {
                f.tick(50); // 50 ms between arrivals, as a real stream has
                f.mark(0);
                f.mark(1);
                f.sched.step(f.t);
            }
            f.advance_one(0, FPS);
        }
        assert_eq!(
            f.sched.stats().stall_recoveries,
            0,
            "a slow but live station must not be treated as stopped"
        );

        // Now it really stops: no packets at all from here on.
        f.stop_delivering(0);
        let _ = f.tick(TIMEOUT_MS + 50);
        assert!(
            f.sched.stats().stall_recoveries > 0,
            "a station that has delivered nothing for the timeout is stopped"
        );
    }

    #[test]
    fn a_dead_station_is_asked_to_zero_fill_rather_than_stalling_the_run() {
        let mut f = Fixture::new(3, 1);
        f.advance_all(FPS);
        // Stations 1 and 2 keep going; station 0 goes quiet.
        for _ in 0..4 {
            f.advance_one(1, FPS);
            f.advance_one(2, FPS);
            f.tick(10);
            f.drain();
        }
        assert_eq!(f.sched.stats().stall_recoveries, 0, "nothing is silent yet");

        f.stop_delivering(0);
        f.tick(TIMEOUT_MS + 10);
        assert_eq!(f.sched.stats().stall_recoveries, 1);
        // next_frame is already at a slot boundary, so the slot waiting to be
        // submitted is [next_frame, next_frame + FPS) and the request has to reach
        // its *end*.
        assert_eq!(
            f.stations[0].requested(),
            START + 2 * FPS,
            "the silent station is asked to cover exactly the next slot"
        );
        assert_eq!(f.sched.stats().zero_fill_requested_frames, FPS);

        // Once it honours the request the watermark moves and work resumes.
        f.stations[0].set(START + 2 * FPS);
        f.mark(0);
        assert!(!f.drain().is_empty(), "work resumes once it catches up");
    }

    #[test]
    fn one_silence_is_a_handful_of_recoveries_not_one_per_step() {
        // Before rate limiting, a station that stayed silent for a second counted
        // as one recovery per scheduler iteration -- 632 of them in a four second
        // run -- which makes the number useless as a signal.
        let mut f = Fixture::new(2, 1);
        f.advance_one(1, 4 * FPS);
        f.advance_one(0, FPS); // it was alive and then stopped
        f.stop_delivering(0);

        let mut now = f.t;
        for _ in 0..2000 {
            now += Duration::from_micros(200); // the real scheduler's step interval
            f.sched.step(now);
        }
        let r = f.sched.stats().stall_recoveries;
        assert!(r <= 3, "2000 steps of 200 us is 0.4 s, so a few recoveries at most, got {r}");
        assert!(r >= 1);
    }

    #[test]
    fn the_zero_fill_cap_is_reported_once_it_is_exceeded() {
        let stations: Vec<_> = (0..2)
            .map(|_| Arc::new(StationProgress::new(START)))
            .collect();
        let t = Instant::now();
        let mut sched = RowScheduler::new(
            stations.clone(),
            RowSchedulerParams {
                frames_per_slot: FPS,
                n_cum: N_CUM,
                rows_per_fetch: 1,
                start_frame_id: START,
                stall_timeout: Duration::from_millis(10),
                zero_fill_limit_frames: N_CUM, // one row's worth of zero filling
                block_slots: 3,
                max_spread_frames: None,
            },
            t,
        );
        stations[0].set(START + 100 * FPS);
        stations[0].set_last_packet_ns(ns_of(t));
        let mut now = t;
        for _ in 0..500 {
            now += Duration::from_millis(5);
            stations[1].set_last_packet_ns(ns_of(t)); // never delivers anything
            sched.step(now);
            if sched.zero_fill_exceeded().is_some() {
                break;
            }
        }
        let ex = sched.zero_fill_exceeded().expect("the cap must be reached");
        assert_eq!(ex.station, 1);
        assert!(ex.frames > ex.limit);
        assert!(format!("{ex}").contains("stopped producing data"));
    }

    #[test]
    fn a_stall_request_is_capped_at_the_row_end_and_never_moves_backwards() {
        let mut f = Fixture::new(2, 1);
        f.advance_one(0, 3 * N_CUM + 10 * FPS);
        f.tick(500);
        let asked = f.stations[1].requested();
        assert!(asked <= START + N_CUM, "a request must not run past the row end: {asked}");
        f.tick(500);
        assert!(f.stations[1].requested() >= asked, "requests only move forward");
    }

    #[test]
    fn rows_pending_tracks_fetch_readiness() {
        let mut f = Fixture::new(1, 3);
        for _ in 0..3 {
            f.advance_all(N_CUM);
            assert_eq!(closes(&f.drain()).len(), 1);
        }
        assert!(f.sched.fetch_ready(), "3 rows fill a rows_per_fetch of 3");
        f.sched.note_row_fetched(3);
        assert!(!f.sched.fetch_ready());
        assert_eq!(f.sched.rows_pending(), 0);
    }
}
