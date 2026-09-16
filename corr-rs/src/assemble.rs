//! Frame assembly: turning a station's packet stream into staging bytes.
//!
//! One [`StationAssembler`] per receive thread. It owns exactly one station's
//! region of the pinned staging ring, so the only shared state it touches is the
//! [`SlotRing`], which tells it when the GPU has finished reading a slot.
//!
//! Everything here is pure with respect to the network: `on_packet` takes a
//! counter and 4096 bytes of samples. That is deliberate. The measured link is
//! clean enough (120 s, zero gaps, zero reordering) that the branches worth
//! worrying about -- a lost packet, a station that stops, a counter jump -- never
//! occur naturally, so they are driven directly by tests instead.
//!
//! Where frames land:
//!
//! ```text
//! slot_seq      = frame_id / frames_per_slot
//! slot          = slot_seq % block_slots
//! frame_in_slot = frame_id % frames_per_slot
//!
//! address = base + slot*slot_stride + frame_in_slot*frame_stride + packet*4096
//! ```
//!
//! The slot index comes from `frames_per_slot`, **not** from the integration row:
//! with a 1 second integration (97656 frames) and 256-frame slots, one row spans
//! 382 slots, so a per-row index would be constant for the whole row and the
//! whole ring would collapse onto one buffer.
//!
//! Finality:
//!
//! A frame is *final* once it can no longer change -- either every one of its
//! `P` packets arrived, or the counter has jumped past it, which on this link
//! means the missing packets are gone for good (measured reordering: zero). Every
//! frame below [`StationAssembler::progress`] is final, and the scheduler's
//! watermark is the minimum of the stations' progress. That is why a lost packet
//! is detected the instant the next one arrives, with no timeout involved; the
//! timeout in the configuration exists only for a station that stops sending
//! altogether, which is a different failure.
//!
//! What finality does *not* mean is that the bytes are there. A final frame whose
//! packets were lost is declared missing in this station's validity mask (one bit
//! per frame of a slot, inside the staging region; see `corr_cuda.h`) and zeroed
//! on the device. The receive thread used to write those zeros itself, which cost
//! it the socket time it needed to catch up, and cost a whole slot period of
//! waiting for the pipeline at every step of a long gap -- the loop that made one
//! station's loss everyone's loss (docs/DESIGN-zero-fill-decoupling.md).
//!
//! Writing the mask is the stronger statement of the two, so it is also the
//! cheaper one: it is a bit set per *complete* frame, and nothing at all for the
//! missing ones. Two things follow, and both are load bearing:
//!
//! * `progress` may run ahead of the data, so a jump over a gap is now a handful
//!   of host-side bookkeeping and one publish, not a slot-by-slot crawl.
//! * the assembler never clears a mask bit: an all-zero mask is what a slot holds
//!   when nothing arrived, and `corr_row_slot_free` restores that state as the
//!   pipeline hands each slot back. The receive thread may only write a slot after
//!   the ring grants it, so a bit it sets belongs to the sequence it is filling.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// How long a receive thread spins before it starts blocking on the ring.
///
/// Spinning has the lowest latency -- the pipeline's release is picked up in
/// nanoseconds -- but a thread that spins while the pipeline is the bottleneck
/// burns a whole core, and at 24 stations that was 21.5 cores of a 32-core
/// machine shared with other people. Blocking costs a wakeup (~1 us, plus a 1 ms
/// poll as a backstop) and takes the process to 6.1 cores in the same test.
///
/// Neither extreme is right on its own: the spin matters when the ring frees up
/// almost immediately, which is the normal case, and the blocking matters when it
/// does not. So: spin briefly, then block. `CORR_RING_SPIN=1` forces the old
/// pure-spin behaviour, which is how the two were compared.
const SPINS_BEFORE_BLOCK: u64 = 20_000;

use crate::config::SAMPLES_PER_PACKET;
use crate::frame::{Arrival, CounterTracker, FrameArrival, FrameMath};
use crate::sched::StationProgress;

/// Shared bookkeeping for the staging ring.
///
/// The only cross-thread rule in the whole receive path: a slot's bytes may not
/// be written until the GPU has finished copying that slot out. `freed[s]` holds
/// the highest slot sequence for slot `s` that the pipeline has released, or -1
/// if it has never been used.
#[derive(Debug)]
pub struct SlotRing {
    slots: usize,
    /// i64 rather than u64 so that "never used" is representable without an
    /// awkward sentinel.
    freed: Vec<AtomicI64>,
    /// Sequence the window currently starts at: the writable sequences are
    /// `base + 1 ..= base + slots`.
    ///
    /// [`SlotRing::is_free`] reads `freed` directly and does not need this; it
    /// exists so that [`SlotRing::reprime`] can tell a forward move from a no-op.
    base: AtomicI64,
    /// True until the pipeline takes its first slot.
    ///
    /// While it is true nothing has been read out of the staging, so the window
    /// may still be moved to wherever the first packet actually is -- see
    /// [`SlotRing::reprime`]. The pipeline closes it, not the receive side: only
    /// the pipeline knows when a byte stops being the assembler's.
    window_open: AtomicBool,
    /// How many startup moves the window made, for the run report.
    reprimes: AtomicU64,
    /// Where blocked waiters sleep. The generation counter is bumped on every
    /// release and on abort, so a waiter that missed the change re-checks it.
    ///
    /// Waiting by spinning is what this replaced: a receive thread whose ring is
    /// full used to burn a whole core in `wait_free`, and at 24 stations that was
    /// ~20 of 32 cores -- CPU taken from the packet path the thread belongs to,
    /// on a machine that is often shared. Blocking costs a wakeup of latency,
    /// which the socket buffer (0.5 s of data here) absorbs without a trace.
    wait: Mutex<u64>,
    cv: Condvar,
    /// Set when the run is shutting down.
    ///
    /// Without it, `wait_free` is an unbounded spin with no way out: a receive
    /// thread waiting for a slot that the pipeline will never release, because
    /// the pipeline has already stopped, never sees the stop flag and never
    /// returns -- and the join that follows hangs for ever. That is not
    /// theoretical: it left a correlator burning two cores for thirteen minutes
    /// after `timeout` had already sent SIGTERM.
    aborted: AtomicBool,
    /// Diagnostics for the one question a stalled run cannot answer from outside:
    /// which sequence a receive thread is waiting for, what release it needs to
    /// see, and what that physical slot actually holds. A frozen watermark looks
    /// identical whether the ring is refused a slot it should own, whether the
    /// pipeline never released one, or whether nobody is sending at all -- and
    /// those call for opposite fixes.
    ///
    /// Written once per blocking episode, not per spin, so a thread that waits a
    /// second writes these a few hundred times rather than a few billion.
    last_wait: AtomicI64,
    last_need: AtomicI64,
    last_have: AtomicI64,
    blocked: AtomicI64,
    block_events: AtomicU64,
}

/// True when `CORR_RING_SPIN` is set: never block, spin instead. Only useful for
/// comparing the two behaviours on a quiet machine.
fn spin_only() -> bool {
    static SPIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SPIN.get_or_init(|| std::env::var_os("CORR_RING_SPIN").is_some())
}

/// How many times to re-read the ring before parking on the condition variable.
///
/// This is a *fixed* cost per slot acquisition, and a slot is acquired once per
/// 256 frames, so it is paid 380 times a second per station whether or not the
/// release is close. The original value was chosen as "spin a little, then
/// block", which is the right shape for a lock held for microseconds; a staging
/// slot is held for a whole slot period (2.8 ms at 24 stations), so spinning
/// 20000 times is 1.5 ms of pure waste with the pipeline's cache line bouncing
/// between sockets. Measured: 19140 spins per slot, 14.1 s of a 20 s run per
/// station inside `wait_free`, and a receive rate of half the link's -- the
/// threads were slower than the ring cadence as a direct result.
///
/// `CORR_RING_SPINS` overrides it, which is how the default was chosen.
fn spins_before_block() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CORR_RING_SPINS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(SPINS_BEFORE_BLOCK)
    })
}

impl SlotRing {
    pub fn new(slots: usize) -> Self {
        Self::primed(slots, 0)
    }

    /// A ring whose first `slots` sequences are already free.
    ///
    /// This is not an optimisation, it is required for correctness. The run
    /// starts at whatever frame id the front end's counter happens to be at --
    /// around 1e10 in practice, so the first slot sequence is around 2.5e9. A
    /// ring initialised to "never released" would have the very first packet wait
    /// for a release of sequence `first - slots`, which no pipeline will ever
    /// issue, and the receive thread would spin for ever.
    pub fn primed(slots: usize, first_slot_seq: u64) -> Self {
        assert!(slots >= 2, "a ring of {slots} slots cannot overlap anything");
        // freed[i] = first - 1 makes exactly the sequences
        // first .. first+slots-1 writable and no more.
        let init = first_slot_seq as i64 - 1;
        SlotRing {
            slots,
            freed: (0..slots).map(|_| AtomicI64::new(init)).collect(),
            base: AtomicI64::new(init),
            window_open: AtomicBool::new(true),
            reprimes: AtomicU64::new(0),
            wait: Mutex::new(0),
            cv: Condvar::new(),
            aborted: AtomicBool::new(false),
            last_wait: AtomicI64::new(-1),
            last_need: AtomicI64::new(-1),
            last_have: AtomicI64::new(-1),
            blocked: AtomicI64::new(0),
            block_events: AtomicU64::new(0),
        }
    }

    /// How many times a thread had to stop and wait, and how many are waiting
    /// now. Unlike the spin counter these are small, meaningful numbers.
    pub fn block_events(&self) -> u64 {
        self.block_events.load(Ordering::Relaxed)
    }

    pub fn blocked_now(&self) -> i64 {
        self.blocked.load(Ordering::Relaxed)
    }

    /// What this ring has been refusing, for the pipeline's stall diagnostics.
    pub fn wait_snapshot(&self) -> String {
        let freed: Vec<i64> = self.freed.iter().map(|f| f.load(Ordering::Relaxed)).collect();
        format!(
            "ring: slots={} base={} freed={:?} blocked={} blocks={} last_wait={} need>={} have={}",
            self.slots,
            self.base.load(Ordering::Relaxed),
            freed,
            self.blocked.load(Ordering::Relaxed),
            self.block_events.load(Ordering::Relaxed),
            self.last_wait.load(Ordering::Relaxed),
            self.last_need.load(Ordering::Relaxed),
            self.last_have.load(Ordering::Relaxed),
        )
    }

    /// Freeze the window: the pipeline has taken its first slot, so every byte
    /// behind it now belongs to the consumer and the window may not move again.
    pub fn close_window(&self) {
        self.window_open.store(false, Ordering::Release);
    }

    pub fn is_window_open(&self) -> bool {
        self.window_open.load(Ordering::Acquire)
    }

    /// How many times a waiting station moved the window forward at startup.
    pub fn reprimes(&self) -> u64 {
        self.reprimes.load(Ordering::Acquire)
    }

    /// Move the window forward so that `slot_seq` and the `slots - 1` sequences
    /// after it become writable. Returns true when the window actually moved.
    ///
    /// This exists because the run's start frame is a *prediction*. It is computed
    /// from sniffed counters before any receive thread exists, and slot
    /// writability is derived from it. If a station's stream has moved further
    /// than the ring is deep by the time its first packet is placed -- because its
    /// socket filled up while the sockets were still being opened, or while the
    /// GPU context and the output files were being created -- then that first
    /// packet waits for the release of a sequence no pipeline will ever issue, and
    /// the run hangs with zero slots submitted and no error at all. Measured: a
    /// produce run with 24 stations sat like that indefinitely while a dry run of
    /// the same configuration received every packet of every stream.
    ///
    /// Moving the window is only safe while the pipeline has taken nothing out of
    /// the staging, which is exactly what `window_open` records: before the first
    /// slot is handed over there is no copy in flight that a re-declared slot
    /// could clobber. Only forward moves are allowed, and concurrent callers
    /// converge on the largest request because every entry is updated with
    /// `fetch_max`.
    pub fn reprime(&self, slot_seq: u64) -> bool {
        if !self.window_open.load(Ordering::Acquire) {
            return false;
        }
        let new_base = slot_seq as i64 - 1;
        if new_base <= self.base.load(Ordering::Acquire) {
            return false;
        }
        for f in &self.freed {
            f.fetch_max(new_base, Ordering::AcqRel);
        }
        self.base.fetch_max(new_base, Ordering::AcqRel);
        self.reprimes.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Release every waiter and refuse to wait again. Called on shutdown, before
    /// joining the receive threads.
    pub fn abort(&self) {
        self.aborted.store(true, Ordering::Release);
        self.wake_waiters();
    }

    /// Bump the generation and wake every waiter, so one that missed the change
    /// re-checks the slot and the abort flag.
    fn wake_waiters(&self) {
        if let Ok(mut generation) = self.wait.lock() {
            *generation = generation.wrapping_add(1);
            self.cv.notify_all();
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn slot_of(&self, slot_seq: u64) -> usize {
        (slot_seq % self.slots as u64) as usize
    }

    /// True once slot `slot_seq` may be written.
    ///
    /// A slot is writable if the pipeline has released the *previous* use of the
    /// same physical slot, i.e. sequence `slot_seq - slots`.
    pub fn is_free(&self, slot_seq: u64) -> bool {
        let prev = slot_seq as i64 - self.slots as i64;
        if prev < 0 {
            return true;
        }
        self.freed[self.slot_of(slot_seq)].load(Ordering::Acquire) >= prev
    }

    /// Wait until `slot_seq` may be written.
    ///
    /// Returns `(free, spins)`. `free` is false only when the ring was aborted,
    /// in which case the caller must NOT write the slot and must unwind.
    ///
    /// While the window is still open -- nothing has been handed to the pipeline
    /// yet -- a wait that cannot be satisfied moves the window instead of spinning
    /// for ever: see [`SlotRing::reprime`]. Once the pipeline has taken its first
    /// slot the window is frozen and this is a genuine wait for the consumer.
    ///
    /// `alive` is the waiting station's own progress record, and it is refreshed
    /// while we spin. A receive thread blocked on a full ring is *not* a station
    /// that has gone silent: the front end is still sending, the packets are
    /// piling up in the socket, and the thread simply cannot read them until the
    /// pipeline catches up. Without this, one station stalling makes the
    /// scheduler judge every other station dead as well and declare their data missing:
    /// measured, 28 spurious recoveries and 3606 good frames discarded during a
    /// single 1.5 s stall.
    ///
    /// The trade is deliberate and bounded: a station that dies *while* its ring
    /// is full will not be noticed until the ring drains. The zero-fill cap is
    /// what keeps that from running away.
    pub fn wait_free(&self, slot_seq: u64, alive: Option<&StationProgress>) -> (bool, u64) {
        let prev = slot_seq as i64 - self.slots as i64;
        if prev < 0 {
            return (true, 0);
        }
        let idx = self.slot_of(slot_seq);
        let mut waits = 0u64;
        let mut blocked_here = false;
        loop {
            if self.freed[idx].load(Ordering::Acquire) >= prev {
                if blocked_here {
                    self.blocked.fetch_sub(1, Ordering::Relaxed);
                }
                return (true, waits);
            }
            // Shutdown first: an aborted ring must unwind its waiters, and that
            // takes precedence over any decision about the window.
            if self.aborted.load(Ordering::Acquire) {
                if blocked_here {
                    self.blocked.fetch_sub(1, Ordering::Relaxed);
                }
                return (false, waits);
            }
            // Startup: the pipeline has taken nothing yet, so the window can still
            // be moved to where the first packet actually is.
            if self.reprime(slot_seq) {
                continue;
            }
            waits += 1;
            if waits <= spins_before_block() && !spin_only() {
                std::hint::spin_loop();
                continue;
            }
            if waits == spins_before_block() + 1 {
                // The moment we commit to waiting instead of spinning: this is
                // the state that a stalled run has to be explained by.
                self.last_wait.store(slot_seq as i64, Ordering::Relaxed);
                self.last_need.store(prev, Ordering::Relaxed);
                self.last_have
                    .store(self.freed[idx].load(Ordering::Relaxed), Ordering::Relaxed);
                self.blocked.fetch_add(1, Ordering::Relaxed);
                self.block_events.fetch_add(1, Ordering::Relaxed);
                blocked_here = true;
            }
            // Blocked on the ring is not the same as silent: the front end is
            // still sending and the packets are queueing in the socket, so this
            // counts as being alive for the stall detector.
            if let Some(p) = alive {
                p.mark_packet();
            }
            // Sleep until the pipeline releases something, re-checking every
            // millisecond so a missed wakeup -- or a shutdown -- cannot strand
            // this thread.
            if let Ok(guard) = self.wait.lock() {
                let generation = *guard;
                let _ = self.cv.wait_timeout_while(guard, Duration::from_millis(1), |g| {
                    *g == generation
                        && !self.aborted.load(Ordering::Acquire)
                        && self.freed[idx].load(Ordering::Acquire) < prev
                });
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Called by the pipeline after `corr_row_slot_free` returns.
    pub fn release(&self, slot_seq: u64) {
        self.release_inner(slot_seq);
        self.wake_waiters();
    }

    fn release_inner(&self, slot_seq: u64) {
        let idx = self.slot_of(slot_seq);
        let seq = slot_seq as i64;
        // Only ever move forward: a release for an older sequence must not undo a
        // newer one.
        let mut cur = self.freed[idx].load(Ordering::Relaxed);
        while cur < seq {
            match self.freed[idx].compare_exchange_weak(
                cur,
                seq,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Highest sequence released for a given slot, for diagnostics. -1 if never.
    pub fn released(&self, slot: usize) -> i64 {
        self.freed[slot].load(Ordering::Acquire)
    }
}

/// Counters a receive thread keeps about its own stream.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AssembleStats {
    pub packets: u64,
    pub gaps: u64,
    pub lost_packets: u64,
    pub duplicates: u64,
    pub out_of_order: u64,
    /// Frames that will be zeroed on the device because at least one packet was
    /// lost. The count is the assembler's own tally of the frames it declared
    /// missing in the validity mask, which is what the GPU acts on; it no longer
    /// counts bytes this thread wrote.
    pub zero_filled_frames: u64,
    pub finalized_frames: u64,
    /// How many times this thread had to wait for the pipeline to release a
    /// staging slot (each wait blocks, it does not spin). A large number here
    /// means the pipeline is the bottleneck, not the network.
    pub slot_wait_spins: u64,
    /// Wall time spent inside `wait_free`, in nanoseconds. The spin count alone
    /// cannot say whether the ring is expensive: it depends entirely on what one
    /// probe costs, which on this machine is a cache line owned by a thread on
    /// another NUMA node.
    pub ns_slot_wait: u64,
    /// Packets dropped because they belonged to a frame that was already
    /// finalized -- a reordered or duplicated packet arriving after its frame was
    /// closed, or one whose place was taken by a zero fill.
    pub late_packets: u64,
    /// Packets dropped because they arrived before the run's aligned start frame.
    ///
    /// Kept separate from `late_packets`: at startup the scheduler discards
    /// everything before the first row boundary, which at the real frame rate is
    /// a second's worth of perfectly good packets. Counting them as "late" makes a
    /// clean startup look like a reordering problem.
    pub before_start_packets: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssembleError {
    /// The packet belongs to a frame the assembler cannot place: either far
    /// behind the finalized watermark, or so far ahead that its slot cannot be
    /// addressed. Reported rather than silently dropped, because it means the
    /// counter and the geometry disagree.
    CounterOutOfWindow { frame_id: u64, progress: u64 },
    /// Slot index out of range, which can only be a geometry bug.
    BadSlot { slot: u64, slots: usize },
    /// The run is shutting down and this packet belongs to a slot the pipeline
    /// will never release. Not an error to report, just a reason to unwind.
    Aborted,
    /// The first packet this station produced is so far past the run's start frame
    /// that the frames in between cannot be declared missing without lapping the
    /// staging ring. See [`AssembleError::is_fatal`].
    StartupGap { gap: u64, window: u64 },
    /// This station has fabricated more frames of silence than the configuration
    /// allows. See [`AssembleError::is_fatal`].
    ZeroFillCapExceeded { fabricated: u64, limit: u64 },
    /// A packet shorter than one sample payload.
    ShortPacket { len: usize },
}

impl AssembleError {
    /// True when the stream cannot be salvaged and the run must stop.
    ///
    /// A per-packet fault (a short packet, a geometry mismatch) is reported and
    /// the stream continues, because one malformed datagram is not a reason to
    /// throw away an observation. These are different: the run is over, either
    /// because it is shutting down, because its staging geometry and the front
    /// end's counter never agreed, or because a station has produced nothing but
    /// silence for longer than the cap allows. Continuing would either write wrong
    /// data or grind out rows of zeros for ever.
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            AssembleError::Aborted
                | AssembleError::StartupGap { .. }
                | AssembleError::ZeroFillCapExceeded { .. }
        )
    }
}

impl std::fmt::Display for AssembleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssembleError::CounterOutOfWindow { frame_id, progress } => write!(
                f,
                "frame {frame_id} is outside the assembler window (finalized up to {progress})"
            ),
            AssembleError::BadSlot { slot, slots } => {
                write!(f, "slot {slot} outside a ring of {slots}")
            }
            AssembleError::ShortPacket { len } => {
                write!(f, "packet payload is {len} bytes, need at least 4096")
            }
            AssembleError::StartupGap { gap, window } => write!(
                f,
                "this station's first packet is {gap} frames past the frame the run \
                 started at, and the staging ring only holds {window} frames, so zero \
                 filling the difference would overwrite the very frames it is filling \
                 (the range laps the ring, and the GPU would read whichever write \
                 landed in a slot last). The run did not start on time: the sockets \
                 filled up before the receive threads were draining them. Nothing has \
                 been written for this run; start it again, and check what delayed the \
                 first packet -- a slow output directory, a loaded machine, or a \
                 station whose stream begins later than the others"
            ),
            AssembleError::ZeroFillCapExceeded { fabricated, limit } => write!(
                f,
                "this station has fabricated {fabricated} frames of silence, past the \
                 configured limit of {limit}; it has stopped producing data. The rows \
                 written so far record how many frames each one is missing, in \
                 blocks.csv"
            ),
            AssembleError::Aborted => write!(f, "shutting down"),
        }
    }
}

impl std::error::Error for AssembleError {}

/// Assembles one station's packets into its region of the pinned staging ring.
///
/// Not `Sync`: one thread per station, by construction. The raw pointer is the
/// staging region handed out by `corr_staging_info`, which the GPU side owns.
pub struct StationAssembler {
    fm: FrameMath,
    frames_per_slot: u64,
    frame_stride: usize,
    slot_stride: usize,
    ring: Arc<SlotRing>,

    /// Start of this station's staging region.
    base: *mut u8,

    /// Per physical slot, the arrival bitmap and which slot sequence it describes.
    arrival: Vec<FrameArrival>,
    arrival_seq: Vec<u64>,

    /// The sequence this thread has already been granted, per physical slot.
    ///
    /// Asking the ring once per packet was 74% of a receive thread's run time
    /// and cost more than the link did: a slot holds 512 packets at 24 stations'
    /// geometry, so 511 of every 512 ring probes asked a question whose answer
    /// could not have changed. Measured on the affected run: 3.2e9 spin
    /// iterations over 4.0M packets, 790 per packet, with the assembly path
    /// taking 14.8 s of a 20 s run per station while the pipeline thread sat 97%
    /// idle -- the threads were spinning on a cache line the pipeline owned, not
    /// waiting for data.
    ///
    /// Caching a *grant* is safe where caching a refusal would not be: a release
    /// only ever widens the writable window ([`SlotRing::wait_free`] compares
    /// `freed` against a threshold that never decreases), so a slot that was
    /// writable when it was granted stays writable for that use. A stale entry
    /// can therefore only make this thread wait slightly longer than it must,
    /// never write a slot the pipeline is still reading.
    granted: Vec<u64>,

    /// First frame id that is not yet final. Every frame below it is final.
    progress: u64,
    /// The frame id the run started at, so earlier packets can be told apart from
    /// merely late ones.
    first_frame_id: u64,
    /// True once a packet has actually been written into the staging.
    ///
    /// The startup gap check only applies before that: once this station is
    /// contributing data, a jump is ordinary packet loss and is declared missing as
    /// before, because by then the pipeline is demonstrably consuming slots and a
    /// release is on its way.
    placed_any: bool,

    tracker: CounterTracker,
    stats: AssembleStats,

    /// This station's liveness record, refreshed while the ring is full. Set by
    /// the receive thread; `None` in tests that do not care.
    alive: Option<Arc<StationProgress>>,

    /// Where this station's validity mask lives inside its staging region, and
    /// how many bytes it takes per slot. Both come from the C side's layout,
    /// which `--check` validates against this side's arithmetic; `0` disables the
    /// mask, which is what a unit test or a dry run passes.
    ///
    /// The mask is one bit per frame of a slot, **LSB first** (`byte f/8`, bit
    /// `f%8`), set when that frame's samples in the staging are real. It is what
    /// moves the zeroing of a missing frame to the GPU (see
    /// docs/DESIGN-zero-fill-decoupling.md): the device reads it with the samples
    /// and zeroes every frame whose bit is clear. This side therefore sets a bit
    /// per frame that arrived whole and writes nothing else -- in particular it
    /// never clears one, because clearing a mask the pipeline is still copying
    /// would turn real frames into silence, which is the one failure mode of this
    /// design that produces no error at all.
    validity_offset: usize,
    validity_bytes_per_slot: usize,

    /// Set once `finalize_upto` has been called with a frame id, so a stream that
    /// stops is still closed out.
    finalized_to: Option<u64>,

    /// Cap on the frames this station may fabricate as silence. 0 disables it.
    ///
    /// The scheduler has its own cap, on the frames it *asks* for; this one also
    /// covers the frames a station declares missing because its own counter jumped.
    /// Both exist for the same reason: a station that is not producing data must
    /// not be able to make the run grind out rows of silence for ever.
    zero_fill_limit: u64,
}

// SAFETY: the assembler is owned by exactly one receive thread, and the raw
// pointer it holds refers to pinned memory that outlives it. It is never shared.
unsafe impl Send for StationAssembler {}

impl StationAssembler {
    /// `base` must be the station's staging base from `corr_staging_info`.
    #[allow(clippy::too_many_arguments)] // staging layout + ring + mask layout
    pub fn new(
        packets_per_frame: u64,
        frames_per_slot: u64,
        frame_stride: usize,
        slot_stride: usize,
        ring: Arc<SlotRing>,
        base: *mut u8,
        first_frame_id: u64,
        validity_offset: usize,
        validity_bytes_per_slot: usize,
    ) -> Self {
        let slots = ring.slots();
        StationAssembler {
            fm: FrameMath::new(packets_per_frame),
            frames_per_slot,
            frame_stride,
            slot_stride,
            ring,
            base,
            validity_offset,
            validity_bytes_per_slot,
            arrival: (0..slots).map(|_| FrameArrival::new(frames_per_slot as usize, packets_per_frame)).collect(),
            arrival_seq: vec![u64::MAX; slots],
            granted: vec![u64::MAX; slots],
            progress: first_frame_id,
            first_frame_id,
            placed_any: false,
            tracker: CounterTracker::new(),
            stats: AssembleStats::default(),
            alive: None,
            finalized_to: None,
            zero_fill_limit: 0,
        }
    }

    /// Cap the frames this station may fabricate as silence; 0 means no limit.
    ///
    /// Set from the configuration's `integration.zero_fill_limit_frames`, the
    /// same number the scheduler's recovery path is capped by.
    pub fn set_zero_fill_limit(&mut self, limit: u64) {
        self.zero_fill_limit = limit;
    }

    pub fn stats(&self) -> AssembleStats {
        self.stats
    }

    /// Give the assembler this station's liveness record, so that waiting for a
    /// slot counts as being alive rather than as silence.
    pub fn set_liveness(&mut self, alive: Arc<StationProgress>) {
        self.alive = Some(alive);
    }

    /// Publish the station's progress to the scheduler.
    ///
    /// Every path that moves `progress` forward must go through this, and it must
    /// happen *before* anything that can block. The scheduler's permission to
    /// release the staging slot this thread is about to wait for is derived from
    /// the published value, so a publication that happens after the wait is a
    /// publication that never happens: the two threads wait for each other for
    /// ever, and the symptom is a run that submits two slots, produces no rows and
    /// reports no error at all. That is why this is a method rather than three
    /// copies of two lines (HANDOFF 6.13.1, docs/ARCHITECTURE.md 3.1).
    fn publish(&self) {
        if let Some(p) = &self.alive {
            p.set(self.progress);
        }
    }

    /// Move `progress` forward to `frame` and publish it.
    fn advance_to(&mut self, frame: u64) {
        self.progress = frame;
        self.publish();
    }

    /// First frame id that is not yet final.
    pub fn progress(&self) -> u64 {
        self.progress
    }

    pub fn packets_per_frame(&self) -> u64 {
        self.fm.packets_per_frame()
    }

    /// Address of one frame's samples inside the staging.
    fn frame_ptr(&self, frame_id: u64) -> (*mut u8, usize) {
        let slot_seq = frame_id / self.frames_per_slot;
        let slot = (slot_seq % self.ring.slots() as u64) as usize;
        let frame_in_slot = frame_id % self.frames_per_slot;
        let off = slot * self.slot_stride + frame_in_slot as usize * self.frame_stride;
        // SAFETY-adjacent: the caller has already established that the slot is
        // writable and that `off` is inside this station's region.
        (unsafe { self.base.add(off) }, off)
    }

    /// Wait (once per use) for the physical slot holding `frame_id`.
    ///
    /// See [`Self::granted`] for why this is not simply a `wait_free` call at
    /// every packet.
    fn ensure_writable(&mut self, slot_seq: u64) -> Result<(), AssembleError> {
        let slot = (slot_seq % self.ring.slots() as u64) as usize;
        if self.granted[slot] == slot_seq {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let (free, spins) = self.ring.wait_free(slot_seq, self.alive.as_deref());
        self.stats.ns_slot_wait += t0.elapsed().as_nanos() as u64;
        self.stats.slot_wait_spins += spins;
        if !free {
            return Err(AssembleError::Aborted);
        }
        self.granted[slot] = slot_seq;
        Ok(())
    }

    /// This station's mask bytes for one physical slot, or `None` when disabled.
    fn mask_slot(&self, slot_seq: u64) -> Option<*mut u8> {
        if self.validity_bytes_per_slot == 0 {
            return None;
        }
        let slot = (slot_seq % self.ring.slots() as u64) as usize;
        let off = self.validity_offset + slot * self.validity_bytes_per_slot;
        // SAFETY: the caller owns this station's staging region, and the offset is
        // inside it by the layout the ABI reported and `--check` verified.
        Some(unsafe { self.base.add(off) })
    }

    /// Mark this frame's samples as real.
    ///
    /// Only ever sets bits. A frame that is missing keeps whatever the slot had,
    /// which is nothing: the mask starts at zero when the staging is allocated and
    /// `corr_row_slot_free` puts a slot back to zero as the pipeline hands it over,
    /// so "no bit" is the default the assembler inherits rather than something it
    /// has to write. That is the whole reason a gap no longer costs this thread a
    /// slot-by-slot walk.
    fn mask_set_frame(&mut self, frame_id: u64) {
        if let Some(p) = self.mask_slot(frame_id / self.frames_per_slot) {
            let f = (frame_id % self.frames_per_slot) as usize;
            // SAFETY: `f/8 < validity_bytes_per_slot` because the mask has one bit
            // per frame of a slot and `frames_per_slot` is what sized it. The slot
            // was granted by `ensure_writable` before any packet was written into
            // it, so no copy of a previous sequence can still be reading this.
            unsafe { *p.add(f / 8) |= 1u8 << (f % 8) };
        }
    }

    /// Borrow the arrival bitmap for the slot holding `frame_id`, resetting it if
    /// the physical slot has been recycled since it was last used.
    ///
    /// The reset is this thread's own memory only. The staging's mask for that
    /// slot is deliberately *not* touched here: this is called from paths that may
    /// not own the slot (a jump finalizes frames of slots it never wrote into), and
    /// clearing a mask the pipeline is still copying would turn real frames into
    /// silence -- a wrong answer that reads as a clean run.
    fn arrival_for(&mut self, frame_id: u64) -> &mut FrameArrival {
        let slot_seq = frame_id / self.frames_per_slot;
        let slot = (slot_seq % self.ring.slots() as u64) as usize;
        if self.arrival_seq[slot] != slot_seq {
            self.arrival[slot].reset();
            self.arrival_seq[slot] = slot_seq;
        }
        &mut self.arrival[slot]
    }

    /// Feed one packet. `payload` is the sample bytes, with the 8-byte counter
    /// already stripped by the caller.
    pub fn on_packet(&mut self, counter: u64, payload: &[u8]) -> Result<(), AssembleError> {
        if payload.len() < SAMPLES_PER_PACKET as usize {
            return Err(AssembleError::ShortPacket { len: payload.len() });
        }
        self.stats.packets += 1;

        match self.tracker.observe(counter) {
            Arrival::First | Arrival::InOrder | Arrival::Gap { .. } => {}
            Arrival::Duplicate => {
                self.stats.duplicates += 1;
                // A duplicate of a still-open frame is harmless to re-write; a
                // duplicate of a finalized frame is simply late.
            }
            Arrival::OutOfOrder => {
                self.stats.out_of_order += 1;
            }
        }
        self.stats.gaps = self.tracker.gaps;
        self.stats.lost_packets = self.tracker.lost_packets;

        let frame_id = self.fm.frame_id(counter);

        // Before the aligned start is the startup discard, not a fault.
        if frame_id < self.first_frame_id {
            self.stats.before_start_packets += 1;
            return Ok(());
        }
        // A packet for a frame we have already finalized is late: its slot may
        // even belong to a different row by now, so writing it would corrupt
        // data that has already been handed to the GPU.
        if frame_id < self.progress {
            self.stats.late_packets += 1;
            return Ok(());
        }

        // Anything strictly between the watermark and this frame can never
        // change again: the counter passed it, so the missing packets are gone.
        if frame_id > self.progress {
            // Before this station has contributed anything, the ring has not been
            // handed over at all, so the declaration below runs with the window
            // still open and may move it. That is only safe while the whole range
            // fits inside the ring, i.e. while no physical slot is written twice:
            // a range of a lap or more would lap the staging, and the last write
            // into a physical slot is what the pipeline would read for whichever
            // sequence it happened to be working on. Beyond that, stop and say so
            // rather than write rows that mix frames from different times.
            if !self.placed_any {
                let window = self.frames_per_slot.saturating_mul(self.ring.slots() as u64);
                let gap = frame_id - self.progress;
                if gap >= window {
                    return Err(AssembleError::StartupGap { gap, window });
                }
            }
            self.finalize_range(self.progress, frame_id)?;
            // Published here, before the slot wait below can block: see
            // `publish` for why the order matters more than the value.
            self.advance_to(frame_id);
        }

        // Make sure the physical slot is not still being read by the GPU.
        self.ensure_writable(frame_id / self.frames_per_slot)?;

        let slot_in_frame = self.fm.slot_in_frame(counter);
        let off_in_frame = self.fm.sample_offset_in_frame(counter);
        if off_in_frame + SAMPLES_PER_PACKET as usize > self.frame_stride {
            return Err(AssembleError::BadSlot {
                slot: slot_in_frame,
                slots: self.fm.packets_per_frame() as usize,
            });
        }
        let (p, _) = self.frame_ptr(frame_id);
        // SAFETY: the slot is writable (acquired above) and the offset was just
        // bounds checked against the frame stride.
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                p.add(off_in_frame),
                SAMPLES_PER_PACKET as usize,
            )
        };
        // From here on a forward jump is ordinary loss, not a stale start frame.
        self.placed_any = true;

        let frame_in_slot = (frame_id % self.frames_per_slot) as usize;
        let complete = self
            .arrival_for(frame_id)
            .observe(frame_in_slot as u64, slot_in_frame)
            .map_err(|_| AssembleError::BadSlot {
                slot: slot_in_frame,
                slots: self.fm.packets_per_frame() as usize,
            })?;

        if complete {
            // The frame's samples are all real: say so, for whoever zeroes the
            // missing ones later.
            self.mask_set_frame(frame_id);
        }

        // An in-order stream finalizes frames as soon as all their packets land.
        if complete && frame_id == self.progress {
            self.progress += 1;
            self.stats.finalized_frames += 1;
            // Publish on every completed slot, from here rather than from the
            // receive loop's batch boundary.
            //
            // This is the fix for a deadlock that only shows up with more than
            // one slot in flight, and it is worth spelling out because the
            // symptom is so far from the cause. The pipeline may submit the next
            // slot only once the watermark says this station finished the current
            // one, and it releases staging slots as it submits. So the packet that
            // *completes* a slot is the packet that unlocks the release this thread
            // is about to wait for -- but the wait is on the next packet, which
            // needs the next slot, and the published progress was only updated at
            // the end of the receive batch. Measured: 24 stations parked at 752 to
            // 768 frames of a 768-frame window, one batch short of the boundary,
            // every thread blocked in wait_free with the ring holding exactly the
            // allocation it needed freed.
            if self.progress % self.frames_per_slot == 0 {
                self.publish();
            }
        }
        Ok(())
    }

    /// Declare every frame in `[from, to)` final, counting the ones that never
    /// arrived whole.
    ///
    /// This used to zero fill them, one staging slot at a time, waiting for the
    /// ring before each step. Both halves of that are gone:
    ///
    /// * **The zeros are the device's job now.** A frame that did not arrive whole
    ///   is simply a clear bit in the validity mask, which the GPU reads and acts
    ///   on. Nothing here writes samples, so nothing here has to own the slot.
    /// * **The waiting was the bug.** `wait_free` per step meant a 150 ms gap took
    ///   150 ms to walk, during which this station read no packets and every other
    ///   station's slots were held -- one station's loss, everyone's loss. With the
    ///   walk reduced to host bookkeeping, `progress` crosses the whole gap before
    ///   the next packet is even addressed.
    ///
    /// The cost that remains is honest and bounded by the configuration: the walk
    /// is one pass over the frames the counter skipped, and the frame count is what
    /// [`Self::set_zero_fill_limit`] caps, so a station that stops sending cannot
    /// make this loop unbounded.
    ///
    /// The arrival bitmap still has to be reset slot by slot (`arrival_for` does
    /// that as it goes): for a slot this station never wrote, "no frames arrived" is
    /// exactly what the reset says, and it is what the missing-frame count is made
    /// of.
    fn finalize_range(&mut self, from: u64, to: u64) -> Result<(), AssembleError> {
        let mut frame = from;
        while frame < to {
            let slot_seq = frame / self.frames_per_slot;
            let step_end = ((slot_seq + 1) * self.frames_per_slot)
                .min(to)
                .max(frame + 1);

            for f in frame..step_end {
                let frame_in_slot = (f % self.frames_per_slot) as usize;
                // A recycled physical slot has had its arrival bitmap reset by
                // `arrival_for`, so every frame of a jumped-over slot reads as
                // incomplete here -- which is exactly what it is. Its mask bit is
                // clear for the same reason (nothing has set it this sequence), so
                // the device will see silence, not the previous lap's samples.
                let complete = self.arrival_for(f).is_complete(frame_in_slot);
                if !complete {
                    self.stats.zero_filled_frames += 1;
                }
                self.stats.finalized_frames += 1;
            }
            frame = step_end;

            if self.zero_fill_limit > 0 && self.stats.zero_filled_frames > self.zero_fill_limit {
                return Err(AssembleError::ZeroFillCapExceeded {
                    fabricated: self.stats.zero_filled_frames,
                    limit: self.zero_fill_limit,
                });
            }
        }
        Ok(())
    }

    /// Finalize everything below `frame_id`, counting whatever is missing.
    ///
    /// Used when a station stops sending: the scheduler's watermark could
    /// otherwise never advance past frames this station will never produce. The
    /// receive thread is still the one to do it, because these frames are declared
    /// missing in *its* station's mask and it owns that decision; it no longer has
    /// to own the bytes, which is what makes the walk cheap.
    pub fn finalize_upto(&mut self, frame_id: u64) -> Result<(), AssembleError> {
        if frame_id <= self.progress {
            return Ok(());
        }
        self.finalize_range(self.progress, frame_id)?;
        self.advance_to(frame_id);
        self.finalized_to = Some(frame_id);
        Ok(())
    }

    /// The frame id this assembler was last explicitly finalized to, if any.
    pub fn finalized_to(&self) -> Option<u64> {
        self.finalized_to
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 2; // n_freq = 4096
    const FPS: u64 = 4; // tiny slots, so the ring turns over quickly
    const SLOTS: usize = 3;
    const FRAME_STRIDE: usize = 8192;
    const SLOT_STRIDE: usize = FRAME_STRIDE * FPS as usize;
    const STATION_BYTES: usize = SLOT_STRIDE * SLOTS;
    const MASK_BYTES: usize = (FPS as usize).div_ceil(8);
    const STATION_TOTAL: usize = STATION_BYTES + MASK_BYTES * SLOTS;

    /// What the staging holds before anything is written into it. A frame the
    /// assembler did not fill must still hold this, because filling missing frames
    /// became the device's job: if this thread wrote a zero there, the test that
    /// says "the mask alone decides" would pass for the wrong reason.
    const UNTOUCHED: u8 = 0xA5;

    struct Fixture {
        buf: Vec<u8>,
        asm: StationAssembler,
        ring: Arc<SlotRing>,
        first: u64,
    }

    impl Fixture {
        fn new(first_frame_id: u64) -> Self {
            // Samples prefilled with a sentinel, mask zeroed: the state
            // `corr_create` leaves behind, except that the samples are non-zero so
            // that a missing frame cannot be mistaken for a written one.
            let mut buf = vec![UNTOUCHED; STATION_TOTAL];
            buf[STATION_BYTES..].fill(0);
            let ring = Arc::new(SlotRing::primed(SLOTS, first_frame_id / FPS));
            let asm = StationAssembler::new(
                P,
                FPS,
                FRAME_STRIDE,
                SLOT_STRIDE,
                ring.clone(),
                buf.as_mut_ptr(),
                first_frame_id,
                STATION_BYTES,
                MASK_BYTES,
            );
            Fixture {
                buf,
                asm,
                ring,
                first: first_frame_id,
            }
        }

        /// Packet payload whose bytes encode the counter, so tests can tell
        /// frames apart by content and spot a stale frame immediately.
        fn payload(counter: u64) -> Vec<u8> {
            let mut v = vec![0u8; SAMPLES_PER_PACKET as usize];
            for (i, b) in v.iter_mut().enumerate() {
                *b = ((counter as usize).wrapping_mul(31).wrapping_add(i)) as u8;
            }
            v
        }

        fn send(&mut self, counter: u64) {
            let p = Self::payload(counter);
            self.asm.on_packet(counter, &p).unwrap();
        }

        /// All frames from `first` to `first + FPS*SLOTS`, so every slot has
        /// been used once.
        fn send_full_ring(&mut self) {
            let n = FPS * SLOTS as u64;
            for c in 0..n * P {
                self.send(self.first * P + c);
            }
        }

        fn frame(&self, frame_id: u64) -> &[u8] {
            let slot_seq = frame_id / FPS;
            let slot = (slot_seq % SLOTS as u64) as usize;
            let fi = (frame_id % FPS) as usize;
            let off = slot * SLOT_STRIDE + fi * FRAME_STRIDE;
            &self.buf[off..off + FRAME_STRIDE]
        }

        /// The staging mask byte holding `frame_id`'s bit.
        fn mask_byte(&self, frame_id: u64) -> u8 {
            let slot = ((frame_id / FPS) % SLOTS as u64) as usize;
            let f = (frame_id % FPS) as usize;
            self.buf[STATION_BYTES + slot * MASK_BYTES + f / 8]
        }

        fn expect_frame_content(&self, frame_id: u64) {
            let expect0 = Self::payload(frame_id * P);
            assert_eq!(
                &self.frame(frame_id)[..SAMPLES_PER_PACKET as usize],
                &expect0[..],
                "frame {frame_id} first packet"
            );
            let expect1 = Self::payload(frame_id * P + 1);
            assert_eq!(
                &self.frame(frame_id)[SAMPLES_PER_PACKET as usize..],
                &expect1[..],
                "frame {frame_id} second packet"
            );
        }

        /// The frame is declared missing: its bit is clear, which is what makes the
        /// device zero the whole frame -- including any packets of it that did
        /// arrive, exactly as the old zero fill discarded them. The bytes are *not*
        /// this thread's business any more, so nothing is asserted about them here.
        fn expect_frame_missing(&self, frame_id: u64) {
            let f = (frame_id % FPS) as usize;
            assert_eq!(
                self.mask_byte(frame_id) & (1u8 << (f % 8)),
                0,
                "frame {frame_id} must be declared missing"
            );
        }

        /// Nothing at all was written into this frame, so it still holds what the
        /// staging held before: for a frame that never had a packet, that is the
        /// sentinel, and the assembler must not have zeroed it on the way past.
        fn expect_frame_untouched(&self, frame_id: u64) {
            assert!(
                self.frame(frame_id).iter().all(|b| *b == UNTOUCHED),
                "frame {frame_id} never arrived, so nothing may have been written there"
            );
        }

        /// The frame is declared present.
        fn expect_frame_valid(&self, frame_id: u64) {
            let f = (frame_id % FPS) as usize;
            assert_ne!(
                self.mask_byte(frame_id) & (1u8 << (f % 8)),
                0,
                "frame {frame_id} arrived whole and must be declared valid"
            );
        }

        /// The pipeline releases a slot once the GPU has copied it out -- and, as
        /// `corr_row_slot_free` does in the library, hands its mask back empty. That
        /// clear is the other half of this design: it is what stops a slot nobody
        /// wrote this lap from inheriting the previous lap's bits, which would make
        /// the device correlate stale samples as data.
        /// One slot handed back the way `corr_row_slot_free` hands it back: the ring
        /// is told the copy is done, and the slot's mask is cleared so the next lap
        /// starts from "nothing present".
        fn release(&mut self, slot_seq: u64) {
            self.ring.release(slot_seq);
            let slot = (slot_seq % SLOTS as u64) as usize;
            let off = STATION_BYTES + slot * MASK_BYTES;
            self.buf[off..off + MASK_BYTES].fill(0);
        }

        fn release_upto(&mut self, last_slot_seq: u64) {
            for sq in 0..=last_slot_seq {
                self.release(sq);
            }
        }
    }

    #[test]
    fn the_mask_marks_exactly_the_frames_that_did_not_arrive_whole() {
        // The mask is the *only* statement of which frames are real: the device
        // reads it, with the samples, and zeroes every frame whose bit is clear. So
        // the bit has to mean "every packet of this frame landed", and the bytes of
        // a frame it calls missing have to be left exactly as they were -- if this
        // thread also wrote zeros there, this test would pass while the mask was
        // saying nothing at all.
        const P: u64 = 2; // packets per frame
        const FPS: u64 = 4; // frames per slot
        const FRAME: usize = 2 * SAMPLES_PER_PACKET as usize;
        const SLOTS: usize = 3;
        const MASK_BYTES: usize = (FPS as usize).div_ceil(8);

        let samples = SLOTS * FPS as usize * FRAME;
        let mut buf = vec![UNTOUCHED; samples + SLOTS * MASK_BYTES];
        buf[samples..].fill(0); // as corr_create leaves the mask
        let ring = Arc::new(SlotRing::primed(SLOTS, 0));
        let base = buf.as_mut_ptr();
        let mut a = StationAssembler::new(
            P, FPS, FRAME, FRAME * FPS as usize, ring, base, 0, samples, MASK_BYTES,
        );
        let payload = vec![7i8 as u8; SAMPLES_PER_PACKET as usize];

        // Frames 0, 1 and 3 arrive whole; frame 2 delivers only its first packet.
        for counter in [0u64, 1, 2, 3, 4, 6, 7] {
            a.on_packet(counter, &payload).expect("in-order packets");
        }
        // Frame 3's arrival is what walks past frame 2 and finalises it.
        assert!(a.progress() >= 4, "frame 2 must have been walked past");

        let mask = buf[samples];
        assert_eq!(
            mask & 0b0000_1111,
            0b0000_1011,
            "bits 0, 1 and 3 are whole frames; bit 2 did not complete and must stay clear"
        );
        // And the frame the mask calls missing is left to the device: the packet of
        // it that did arrive stays where it landed (the device discards it by
        // zeroing the whole frame), and the part that never arrived is untouched --
        // if this thread had zeroed either, the mask would be saying nothing.
        let f2 = &buf[2 * FRAME..3 * FRAME];
        assert_eq!(
            &f2[..SAMPLES_PER_PACKET as usize],
            &payload[..],
            "the packet of frame 2 that did arrive must not be touched either way"
        );
        assert!(
            f2[SAMPLES_PER_PACKET as usize..].iter().all(|&b| b == UNTOUCHED),
            "the rest of frame 2 must be left alone: the zeroing is the device's job now"
        );
        assert_eq!(a.stats().zero_filled_frames, 1);
        let f0 = &buf[0..FRAME];
        assert!(f0.iter().all(|&b| b == 7), "frame 0 must hold its samples");
    }

    #[test]
    fn advancing_progress_always_publishes() {
        // The scheduler releases a staging slot only when the published progress
        // says the station is done with it, so an unpublished advance is a
        // deadlock. This asserts the property directly rather than trusting that
        // every call site remembers.
        use std::sync::Arc;

        let ring = Arc::new(SlotRing::primed(3, 0));
        let mut a = StationAssembler::new(2, 4, 8, 32, ring, std::ptr::null_mut(), 0, 0, 0);
        let prog = Arc::new(StationProgress::new(0));
        a.set_liveness(prog.clone());
        assert_eq!(prog.get(), 0, "nothing published before anything happened");

        a.advance_to(5);
        assert_eq!(a.progress(), 5);
        assert_eq!(prog.get(), 5, "advance_to must publish the new progress");

        // And a slot boundary reached by the ordinary packet path publishes too.
        a.advance_to(8);
        assert_eq!(prog.get(), 8);
        assert!(a.progress() % 4 == 0, "the frame involved is a slot boundary here");
    }

    #[test]
    fn a_clean_stream_lands_exactly_where_the_geometry_says() {
        let mut f = Fixture::new(1000);
        f.send_full_ring();
        for i in 0..(FPS * SLOTS as u64) {
            f.expect_frame_content(1000 + i);
        }
        assert_eq!(f.asm.progress(), 1000 + FPS * SLOTS as u64);
        let s = f.asm.stats();
        assert_eq!(s.packets, FPS * SLOTS as u64 * P);
        assert_eq!(s.zero_filled_frames, 0);
        assert_eq!(s.late_packets, 0);
    }

    #[test]
    fn the_slot_index_comes_from_frames_per_slot_not_from_the_row() {
        // A row is far longer than a slot: if the slot were derived from the row
        // index, every one of these frames would land in the same slot.
        let mut f = Fixture::new(0);
        let mut seen_slots = std::collections::BTreeSet::new();
        for frame in 0..(FPS * SLOTS as u64) {
            seen_slots.insert((frame / FPS) % SLOTS as u64);
        }
        assert_eq!(seen_slots.len(), SLOTS, "the ring must actually rotate");
        f.send_full_ring();
        assert_eq!(f.asm.progress(), FPS * SLOTS as u64);
    }

    #[test]
    fn a_lost_packet_declares_only_that_frame_missing() {
        let mut f = Fixture::new(0);
        f.send(0); // frame 0 packet 0
        f.send(1); // frame 0 packet 1  -> frame 0 complete
        f.send(2); // frame 1 packet 0
        // frame 1 packet 1 lost; the next packet we do see is frame 2 packet 0
        f.send(4);
        f.send(5);

        f.expect_frame_content(0);
        f.expect_frame_valid(0);
        f.expect_frame_missing(1);
        f.expect_frame_content(2);
        f.expect_frame_valid(2);

        let s = f.asm.stats();
        assert_eq!(s.lost_packets, 1);
        assert_eq!(s.zero_filled_frames, 1);
        assert_eq!(f.asm.progress(), 3, "frames 0,1,2 are final");
    }
    #[test]
    fn a_whole_lost_frame_is_detected_without_a_timeout() {
        let mut f = Fixture::new(0);
        f.send(0);
        f.send(1);
        // frame 1 entirely lost
        f.send(4);
        f.send(5);

        f.expect_frame_content(0);
        f.expect_frame_missing(1);
        f.expect_frame_untouched(1);
        f.expect_frame_content(2);
        assert_eq!(f.asm.stats().zero_filled_frames, 1);
        assert_eq!(f.asm.progress(), 3);
    }

    #[test]
    fn a_recycled_slot_does_not_inherit_the_previous_laps_bits() {
        // The failure mode a fully-zero staging buffer hides: on the second time
        // round the ring, a frame that is missing must end up with a clear bit even
        // though the same bit was set by the previous use of that physical slot.
        // (The clear itself is `corr_row_slot_free`'s, which `release_upto` mirrors;
        // what is asserted here is that nothing sets it back.)
        let mut f = Fixture::new(0);
        f.send_full_ring();
        f.release_upto(SLOTS as u64);

        // Frame 12 reuses slot 0, which held frames 0..3 and their bits.
        f.send(12 * P); // packet 0 of frame 12
        f.send(12 * P + 1);
        f.send(13 * P + 1); // frame 13 packet 1 arrives; frame 13 packet 0 lost
        f.send(14 * P);
        f.send(14 * P + 1);

        f.expect_frame_content(12);
        f.expect_frame_missing(13);
        f.expect_frame_content(14);
        assert_eq!(f.asm.stats().zero_filled_frames, 1);
    }

    #[test]
    fn packets_that_arrive_after_their_frame_was_finalized_are_dropped() {
        let mut f = Fixture::new(0);
        f.send(0);
        f.send(1);
        f.send(2); // frame 1 packet 0
        f.send(4); // 3 is frame 1 packet 1 -- but it arrives after the gap was declared
        let before = f.asm.stats();
        let progress_before = f.asm.progress();
        f.send(3);
        assert_eq!(f.asm.stats().late_packets, 1);
        assert_eq!(f.asm.stats().packets, before.packets + 1);
        assert_eq!(f.asm.progress(), progress_before);
        assert_eq!(f.asm.stats().before_start_packets, 0, "not a startup packet");
        f.expect_frame_missing(1);
    }

    #[test]
    fn out_of_order_within_a_frame_still_completes_it() {
        let mut f = Fixture::new(0);
        f.send(1); // frame 0 packet 1 first
        f.send(0); // then packet 0
        f.expect_frame_content(0);
        assert_eq!(f.asm.progress(), 1);
        assert_eq!(f.asm.stats().out_of_order, 1);
        assert_eq!(f.asm.stats().zero_filled_frames, 0);
    }

    #[test]
    fn a_duplicate_packet_does_not_advance_anything() {
        let mut f = Fixture::new(0);
        f.send(0);
        f.send(1);
        f.send(1);
        f.send(0);
        f.expect_frame_content(0);
        assert_eq!(f.asm.progress(), 1, "a repeat must not finalize anything new");

        // The first repeat is a duplicate; the second is counted as out of order,
        // because CounterTracker measures against the last in-order counter. What
        // matters is that neither is treated as data: the frame is untouched and
        // exactly one frame is final.
        let s = f.asm.stats();
        assert_eq!(s.packets, 4);
        assert_eq!(s.duplicates + s.out_of_order, 2, "both repeats are unusable");
        assert_eq!(s.finalized_frames, 1);
        assert_eq!(s.zero_filled_frames, 0);
    }

    #[test]
    fn writing_waits_for_the_pipeline_to_release_the_slot() {
        let mut f = Fixture::new(0);
        f.send_full_ring();
        // Nothing has been released, so the ring is full: a write to the next
        // sequence must not proceed.
        assert!(!f.ring.is_free(SLOTS as u64));
        assert!(f.ring.is_free(0) && f.ring.is_free(1) && f.ring.is_free(2));

        // Sequence 3 maps to physical slot 0, so it is sequence 0 that has to be
        // released -- not "the last three".
        f.ring.release(0);
        assert!(f.ring.is_free(SLOTS as u64), "releasing sequence 0 frees sequence 3");
        assert!(
            !f.ring.is_free(SLOTS as u64 + 1),
            "sequence 4 maps to slot 1 and must still be blocked"
        );
        f.ring.release(1);
        assert!(f.ring.is_free(SLOTS as u64 + 1));

        // A release for an older sequence must never undo a newer one.
        f.ring.release(0);
        assert!(f.ring.is_free(SLOTS as u64 + 1));
    }

    #[test]
    fn finalize_upto_declares_a_station_that_stopped_missing() {
        let mut f = Fixture::new(0);
        f.send(0);
        f.send(1); // frame 0 complete
        f.send(2); // frame 1 partial
        assert_eq!(f.asm.progress(), 1);

        f.asm.finalize_upto(3).expect("nothing here needs the ring");
        assert_eq!(f.asm.progress(), 3);
        assert_eq!(f.asm.finalized_to(), Some(3));
        f.expect_frame_content(0);
        f.expect_frame_valid(0);
        f.expect_frame_missing(1);
        f.expect_frame_missing(2);
        f.expect_frame_untouched(2);
        assert_eq!(f.asm.stats().zero_filled_frames, 2);

        // Idempotent, and never walks backwards.
        f.asm.finalize_upto(2).expect("nothing to do");

        assert_eq!(f.asm.progress(), 3);
    }

    #[test]
    fn an_aborted_ring_releases_its_waiters_instead_of_spinning_for_ever() {
        // Regression guard: without this, a run whose pipeline stopped while a
        // receive thread was waiting for a slot hangs the join for ever, and
        // SIGTERM does not help because the handler only sets a flag that the
        // blocked thread never looks at.
        let mut f = Fixture::new(0);
        f.send_full_ring(); // fills the ring; nothing has been released
        assert!(!f.ring.is_free(SLOTS as u64));
        assert!(!f.ring.is_aborted());

        f.ring.abort();
        let (free, _) = f.ring.wait_free(SLOTS as u64, None);
        assert!(!free, "an aborted wait must report failure, not claim the slot");

        // And a write into that slot must now be refused rather than performed.
        let err = f
            .asm
            .on_packet(SLOTS as u64 * FPS * P, &Fixture::payload(0))
            .unwrap_err();
        assert_eq!(err, AssembleError::Aborted);
    }

    #[test]
    fn an_unprimed_ring_would_block_a_run_that_starts_mid_counter() {
        // Regression guard for a real hang: the shipped run starts at frame ids
        // near 1e10, so slot sequences near 2.5e9. A ring that has "released
        // nothing" would wait for sequence first-slots, which is never issued.
        let ring = SlotRing::new(SLOTS);
        assert!(!ring.is_free(2_500_000_000), "an unprimed ring is not writable there");

        let ring = SlotRing::primed(SLOTS, 2_500_000_000);
        for q in 2_500_000_000..2_500_000_000 + SLOTS as u64 {
            assert!(ring.is_free(q), "sequence {q} should be free on a primed ring");
        }
        assert!(
            !ring.is_free(2_500_000_000 + SLOTS as u64),
            "a primed ring must free exactly one lap, not more"
        );
    }

    #[test]
    fn packets_before_the_aligned_start_are_not_counted_as_late() {
        // The scheduler discards everything before the first row boundary, which
        // at the real frame rate is a second of good packets. Calling those
        // "late" makes a clean startup look like a reordering problem.
        let mut f = Fixture::new(100);
        for frame in 90..100 {
            f.send(frame * P);
            f.send(frame * P + 1);
        }
        let s = f.asm.stats();
        assert_eq!(s.before_start_packets, 20);
        assert_eq!(s.late_packets, 0);
        assert_eq!(f.asm.progress(), 100, "the watermark must not move backwards");
    }

    #[test]
    fn a_startup_counter_jump_moves_the_window_instead_of_hanging() {
        // Regression guard for the produce-mode hang: the run's start frame is
        // computed from sniffed counters before any receive thread exists, so a
        // stream that moved on while the sockets were being opened would otherwise
        // wait for a release no pipeline will ever issue -- with nothing
        // submitted, no error and no output.
        let ring = SlotRing::primed(SLOTS, 100);
        assert!(!ring.is_free(400), "400 is far outside the primed window");
        assert!(ring.is_window_open());

        let (free, _) = ring.wait_free(400, None);
        assert!(free, "an open window must move to where the first packet is");
        assert_eq!(ring.reprimes(), 1);
        assert!(ring.is_free(400) && ring.is_free(400 + SLOTS as u64 - 1));
        assert!(
            !ring.is_free(400 + SLOTS as u64),
            "moving the window must still free exactly one lap"
        );
        // Sequences behind the new window stay writable: they are slots of the
        // same lap that no thread has touched.
        assert!(ring.is_free(399));

        // Once the pipeline has taken a slot, the window is frozen and the same
        // wait becomes a real one. Unwound by aborting, which is how shutdown
        // releases a blocked receive thread.
        ring.close_window();
        assert!(!ring.is_window_open());
        assert!(!ring.is_free(400 + SLOTS as u64));
        let ring = Arc::new(ring);
        let waiter = ring.clone();
        let h = std::thread::spawn(move || waiter.wait_free(400 + SLOTS as u64, None));
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!h.is_finished(), "a frozen window must not move on its own");
        ring.abort();
        let (free, _) = h.join().expect("the waiter must unwind");
        assert!(!free);
    }

    #[test]
    fn a_startup_jump_wider_than_the_ring_is_refused_not_declared() {
        // FPS*SLOTS = 12 frames of staging. A first packet exactly one lap past
        // the start frame would be written over the slot the declared range
        // still occupies -- the mask for a physical slot cannot describe two
        // sequences at once -- so it must be refused rather than silently
        // mis-addressed.
        let mut f = Fixture::new(1000);
        let counter = (1000 + FPS * SLOTS as u64) * P;
        let err = f
            .asm
            .on_packet(counter, &Fixture::payload(counter))
            .unwrap_err();
        assert_eq!(
            err,
            AssembleError::StartupGap {
                gap: FPS * SLOTS as u64,
                window: FPS * SLOTS as u64
            }
        );
        assert!(err.is_fatal(), "the run cannot continue, and must say so");
        assert!(!AssembleError::ShortPacket { len: 1 }.is_fatal());

        // One frame less still fits: the gap is declared missing and the packet
        // lands.
        let mut f = Fixture::new(1000);
        let last = 1000 + FPS * SLOTS as u64 - 1;
        f.send(last * P);
        f.send(last * P + 1);
        for frame in 1000..last {
            f.expect_frame_missing(frame);
        }
        f.expect_frame_content(last);
        assert_eq!(f.asm.progress(), last + 1);

        // The check is a startup check only: once a packet has been placed, a jump
        // is ordinary loss and is declared missing as before.
        let mut f = Fixture::new(1000);
        f.send(1000 * P);
        let counter = (1000 + 4 * (FPS * SLOTS as u64)) * P;
        f.asm.on_packet(counter, &Fixture::payload(counter)).unwrap();
    }

    #[test]
    fn a_jump_is_declared_without_waiting_for_the_ring_or_touching_staging() {
        // This is the fix, stated as a test. The old path zero filled the gap one
        // slot at a time and waited for the ring before each step, so a 150 ms gap
        // cost this station 150 ms of not reading its socket and held every other
        // station's slots meanwhile. Now the walk is host bookkeeping: it must not
        // wait for a single slot, must not write a single byte of staging, and must
        // leave `progress` past the whole gap.
        let mut f = Fixture::new(0);
        f.send_full_ring(); // sequences 0, 1, 2 written
        f.release_upto(SLOTS as u64 - 1); // the pipeline consumed that lap
        f.ring.close_window(); // now it has taken over: the window is frozen
        let spins_before = f.asm.stats().slot_wait_spins;

        // Frames 12..1000 span 247 slots, not one of which is writable -- and not
        // one of which this station has any business writing.
        f.asm.finalize_upto(1000).expect("a jump must not need the ring");

        assert_eq!(f.asm.progress(), 1000, "progress must cross the gap in one step");
        assert_eq!(
            f.asm.stats().slot_wait_spins, spins_before,
            "declaring the gap missing must not wait for the ring"
        );
        assert_eq!(f.asm.stats().zero_filled_frames, 1000 - FPS * SLOTS as u64);
        // The slots of that lap were handed back empty and nothing has set a bit in
        // them since, so the device will treat every frame of the gap as silence ...
        for frame in 12..(12 + FPS * SLOTS as u64) {
            f.expect_frame_missing(frame);
        }
        // ... and nothing was written into the staging either: frames 12.. are the
        // same bytes frames 0.. left there, lap for lap.
        for frame in 12..(12 + FPS * SLOTS as u64) {
            f.expect_frame_content(frame - FPS * SLOTS as u64);
        }
    }

    #[test]
    fn the_packet_that_ends_a_gap_still_waits_for_its_own_slot() {
        // The walk over the gap needs no slot, but the write that ends it does: the
        // pipeline owns a physical slot until its H2D copy is done, and
        // corr_row_slot_free is cudaEventSynchronize on that copy. Dropping the wait
        // from the walk must not have dropped it from the write -- that would put a
        // live packet into a slot being copied right now, and the row would report
        // every frame valid while some of them are the previous lap's.
        let mut f = Fixture::new(0);
        f.send_full_ring(); // sequences 0, 1, 2 written; nothing released
        f.ring.close_window(); // the pipeline has taken over: the window is frozen
        // The pipeline has finished copying sequences 0 and 2 -- so the gap's first
        // slots are back in the receive side's hands, with empty masks, which is
        // what makes frames 12..15 correctly missing -- but not sequence 1, which is
        // the physical slot the incoming packet (sequence 4) needs.
        f.release(0);
        f.release(2);

        assert!(!f.ring.is_free(4), "the packet's own slot is still the GPU's");
        assert!(f.ring.is_free(3), "the gap's first slot is writable");

        let ring = f.ring.clone();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            ring.release(1);
        });
        // Frames 12..15 are the gap; frame 16 is the packet that revealed it.
        f.send(16 * P);
        f.send(16 * P + 1);
        releaser.join().expect("the releaser must finish");

        for frame in 12..16 {
            f.expect_frame_missing(frame);
        }
        f.expect_frame_content(16);
        f.expect_frame_valid(16);
        assert_eq!(f.asm.progress(), 17);
        assert!(
            f.asm.stats().slot_wait_spins > 0,
            "the write must have waited for the ring, not gone straight over it"
        );
    }

    #[test]
    fn a_station_that_fabricates_too_much_silence_stops_the_run() {
        // The scheduler caps the frames it *asks* for; this caps the frames a
        // station invents for itself when its own counter jumps. Without it a
        // station that stops producing data can keep the correlator grinding out
        // rows of silence for ever.
        let mut f = Fixture::new(0);
        f.asm.set_zero_fill_limit(3);
        f.send(0); // half of frame 0, so progress is still 0
        // A jump to frame 6 leaves frames 0..5 to fabricate. The count is checked
        // once per staging slot, so the figure reported is the end of the slot
        // that crossed the limit -- here the first one, frames 0..3.
        let counter = 6 * P;
        let err = f
            .asm
            .on_packet(counter, &Fixture::payload(counter))
            .unwrap_err();
        assert_eq!(
            err,
            AssembleError::ZeroFillCapExceeded {
                fabricated: 4,
                limit: 3
            }
        );
        assert!(err.is_fatal(), "the run must stop, not keep inventing rows");

        // Under the cap the same jump is absorbed.
        let mut f = Fixture::new(0);
        f.asm.set_zero_fill_limit(100);
        f.send(0);
        f.send(counter);
        f.send(counter + 1);
        assert_eq!(f.asm.stats().zero_filled_frames, 6);
        assert_eq!(f.asm.progress(), 7);

        // 0 disables the cap, which is what an unconfigured assembler does.
        let mut f = Fixture::new(0);
        f.asm.set_zero_fill_limit(0);
        f.send(0);
        f.send(5_000 * P);
        assert!(f.asm.stats().zero_filled_frames > 4_000);
    }

    #[test]
    fn short_packets_are_rejected_rather_than_read_past() {
        let mut f = Fixture::new(0);
        let err = f.asm.on_packet(0, &[0u8; 16]).unwrap_err();
        assert_eq!(err, AssembleError::ShortPacket { len: 16 });
    }

    #[test]
    fn every_frame_in_a_long_run_lands_in_the_right_place() {
        // A longer run than the ring, so slots are recycled many times and the
        // arrival bitmaps have to be reset exactly when the slot turns over.
        let mut f = Fixture::new(0);
        let frames = FPS * SLOTS as u64 * 5;
        let mut expected = Vec::new();
        for frame in 0..frames {
            for slot in 0..P {
                let c = frame * P + slot;
                f.send(c);
                expected.push(c);
            }
            // Release the slot as soon as the pipeline would have copied it.
            let slot_seq = frame / FPS;
            if slot_seq >= SLOTS as u64 - 1 {
                f.ring.release(slot_seq - (SLOTS as u64 - 1));
            }
        }
        assert_eq!(f.asm.progress(), frames);
        assert_eq!(f.asm.stats().zero_filled_frames, 0);
        assert_eq!(f.asm.stats().late_packets, 0);
        // Only the frames still resident in the ring can be inspected: anything
        // older has been overwritten by design, which is the whole point of a
        // ring. Check the last full lap.
        for frame in (frames - FPS * SLOTS as u64)..frames {
            f.expect_frame_content(frame);
        }
    }
}
