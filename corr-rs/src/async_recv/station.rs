//! One station's receive task: the async twin of `receiver::recv_thread`.
//!
//! The work is the same in both -- read a batch with `recvmmsg`, feed the packets
//! to the assembler, publish progress, honour the scheduler's request to declare a
//! silent station's frames missing -- and so are the rules. What differs is what
//! happens when the *staging ring* has no free slot for the frame a packet belongs
//! to.
//!
//! A thread-per-station receiver simply blocks there: `SlotRing::wait_free` parks
//! that station's thread on a condition variable, and nothing else on that thread
//! is affected, because there is nothing else on that thread. A task that shares
//! its thread with nineteen other stations may not do that -- parking would stop
//! nineteen streams to wait for one -- so this file asks the ring the same
//! question without waiting (`StationAssembler::try_on_packet`), puts the packet
//! back where it came from, and yields to the other tasks on the worker. When the
//! worker comes back the packet is still first in line, so the stream is never
//! reordered by the detour: a per-station queue of one.
//!
//! Two timeouts keep the task honest, and both have a counterpart on the thread
//! path:
//!
//! * the interval (from `receiver.recv_timeout_ms`, the same number the thread
//!   path puts in `SO_RCVTIMEO`): a station with no traffic must still wake up and
//!   service `StationProgress::requested`, or a stalled station would freeze the
//!   cross-station watermark for ever;
//! * the park delay, which is the async stand-in for the condition variable's
//!   1 ms backstop and is deliberately shorter (200 us).

use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::net::UdpSocket;

use crate::assemble::{AssembleError, FeedOutcome, SlotRing, StationAssembler};
use crate::net::{Packet, PACKET_BYTES};
use crate::receiver::RecvOutcome;
use crate::run::StagingPtr;
use crate::sched::StationProgress;
use crate::stop::{RunStop, StopReason};

/// How many packets one station may place before it yields to its siblings.
///
/// Not an optimisation: it is what keeps one busy stream from monopolising the
/// worker it shares. A socket that always has data -- which is the normal state of
/// a 195 kpps stream -- means `AsyncFd::readable` is ready on every poll, so a
/// task that never yields would never let the reactor run another station's task.
/// 512 packets is about 2 MB of copying, tens of microseconds: short enough that
/// five stations on one worker interleave finely, long enough that the yield is
/// not itself the cost.
const DRAIN_BUDGET_PACKETS: u64 = 512;

/// Consecutive refusals to spin through before parking on the ring's release.
///
/// Four cheap yields first: the pipeline thread releases slots continuously while
/// a row is being accumulated, so the release a station is refused is often
/// already in flight. A yield costs nothing where a park costs a wakeup.
const PARK_YIELDS: u64 = 4;

/// A future that becomes ready when the ring's slot can be written.
///
/// This is the async half of `SlotRing::poll_free`: it registers the task's
/// waker with the ring and is woken by the pipeline's `release`. Waiting on a
/// timer instead is what made a single-worker run deliver a quarter of the
/// stream -- the release being waited for was microseconds away and the sleep
/// quantised it to the timer's millisecond. See `SlotRing::poll_free`.
struct RingReady<'a> {
    ring: &'a SlotRing,
    slot_seq: u64,
    /// How many times this wait was polled, so a diagnostic can tell "the
    /// pipeline is keeping up" from "this station is short of slots".
    polls: u64,
}

impl<'a> RingReady<'a> {
    fn new(ring: &'a SlotRing, slot_seq: u64) -> Self {
        RingReady {
            ring,
            slot_seq,
            polls: 0,
        }
    }
}

impl std::future::Future for RingReady<'_> {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        self.polls += 1;
        if self.ring.poll_free(self.slot_seq, cx.waker()) {
            std::task::Poll::Ready(())
        } else {
            std::task::Poll::Pending
        }
    }
}

/// Everything the tasks on one worker have in common.
///
/// Shared by `Arc` rather than copied: the ring and the progress records must be
/// the *same* objects the scheduler and the report see, not equal ones.
pub(crate) struct RunGeometry {
    pub(crate) ring: Arc<SlotRing>,
    pub(crate) stop: Arc<RunStop>,
    pub(crate) frames_per_slot: u64,
    pub(crate) frame_stride: usize,
    pub(crate) packets_per_frame: u64,
    pub(crate) validity_offset: usize,
    pub(crate) validity_bytes_per_slot: usize,
    pub(crate) start_frame: u64,
    pub(crate) zero_fill_limit_frames: u64,
    /// Longest a station may sit in the reactor without servicing the scheduler's
    /// request. Comes from `receiver.recv_timeout_ms`.
    pub(crate) poll_interval: Duration,
    /// When a packet's frame number means the counter changed epoch. See
    /// [`crate::assemble::EpochLimits`].
    pub(crate) epoch_limits: crate::assemble::EpochLimits,
}

/// One station, as handed to its task.
pub(crate) struct StationInit {
    pub(crate) idx: usize,
    /// The socket, owned by `tokio` from here on: it holds the descriptor and its
    /// reactor registration together.
    pub(crate) sock: UdpSocket,
    /// Where the socket receives into. `std`'s and `tokio`'s sockets both take the
    /// buffer from the caller rather than lending one, which is what keeps a
    /// datagram from being copied into an owned item.
    pub(crate) buf: Vec<u8>,
    pub(crate) staging: StagingPtr,
    pub(crate) progress: Arc<StationProgress>,
}

/// What stopped the walk over a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Batch {
    /// Every buffered packet has been placed (or deliberately dropped).
    Drained,
    /// The frame's staging slot is still the pipeline's: nothing was consumed.
    SlotBusy,
    /// The run has to unwind.
    Fatal,
}

/// Counters for diagnosing a worker that is busy but slow (or idle and slow).
///
/// Set `CORR_ASYNC_STATS=1`. They answer the only question that matters when a
/// run under-delivers: was the task reading, or was it waiting, and on what?
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct TaskStats {
    pub(crate) iterations: u64,
    pub(crate) tick_fires: u64,
    pub(crate) ready_events: u64,
    pub(crate) dry_reads: u64,
    pub(crate) batches: u64,
    pub(crate) yields: u64,
    pub(crate) parks: u64,
    pub(crate) ns_consume: u64,
    pub(crate) ns_wait: u64,
    pub(crate) ring_wait_polls: u64,
}

struct Station {
    idx: usize,
    sock: UdpSocket,
    /// One datagram, received by [`Station::receive_one`] and placed by
    /// [`Station::place_one`]. It holds a packet that a busy staging slot has not
    /// let through yet: nothing receives into it again until that packet is placed,
    /// which is what makes a refusal a retry rather than a loss.
    buf: Vec<u8>,
    /// How many bytes of `buf` the last receive filled.
    len: usize,
    /// Whether `buf` holds a packet that has not been placed yet.
    have_packet: bool,
    asm: StationAssembler,
    prog: Arc<StationProgress>,
    stop: Arc<RunStop>,
    g: Arc<RunGeometry>,
    /// Packets left before this task yields to its siblings.
    budget: u64,
    /// The slot the last refusal was about, which is what a park waits for.
    parked_slot: Option<u64>,
    /// Time spent parked on the staging ring, mirroring the thread path's
    /// `AssembleStats::ns_slot_wait`.
    ns_slot_wait: u64,
    // The same counters `RecvOutcome` carries for the thread path, filled in the
    // same places: a comparison between the two backends is only worth making if
    // the numbers mean the same thing.
    batches: u64,
    packets: u64,
    /// Datagrams longer than one packet can hold. Counted here rather than in the
    /// socket because this backend's socket is `tokio`'s; see `net::RecvStats`.
    oversized: u64,
    idle_polls: u64,
    ns_recv: u64,
    ns_assemble: u64,
    last_read: Instant,
    max_stall_ms: u64,
    stalls_over_20ms: u64,
    /// How many times this station was refused a staging slot. The async
    /// equivalent of `AssembleStats::slot_wait_spins`, and the number that says
    /// whether the pipeline -- not the network -- is what a run is waiting for.
    ring_refusals: u64,
    dbg: TaskStats,
}

impl Station {
    fn new(init: StationInit, g: Arc<RunGeometry>) -> Self {
        let mut asm = StationAssembler::new(
            g.packets_per_frame,
            g.frames_per_slot,
            g.frame_stride,
            g.frame_stride * g.frames_per_slot as usize,
            g.ring.clone(),
            init.staging.as_ptr(),
            g.start_frame,
            g.validity_offset,
            g.validity_bytes_per_slot,
            g.epoch_limits,
        );
        // Waiting for a staging slot is being alive, not being silent -- the same
        // statement the thread path makes. Here the wait happens in this file's
        // park path, which is what marks the station alive while it happens.
        asm.set_liveness(init.progress.clone());
        asm.set_zero_fill_limit(g.zero_fill_limit_frames);
        Station {
            idx: init.idx,
            sock: init.sock,
            buf: init.buf,
            len: 0,
            have_packet: false,
            asm,
            prog: init.progress,
            stop: g.stop.clone(),
            g,
            budget: DRAIN_BUDGET_PACKETS,
            parked_slot: None,
            ns_slot_wait: 0,
            batches: 0,
            packets: 0,
            oversized: 0,
            idle_polls: 0,
            ns_recv: 0,
            ns_assemble: 0,
            last_read: Instant::now(),
            max_stall_ms: 0,
            stalls_over_20ms: 0,
            ring_refusals: 0,
            dbg: TaskStats::default(),
        }
    }

    /// Publish this station's progress, exactly where the thread path does.
    fn publish(&self) {
        self.prog.set(self.asm.progress());
        self.prog.set_zero_filled_frames(self.asm.stats().zero_filled_frames);
        self.prog.set_lost_packets(self.asm.stats().lost_packets);
    }

    /// Honour the scheduler's request to declare frames up to `requested()` final.
    ///
    /// Returns false when the run must unwind. This is the same decision, with the
    /// same two failure modes, as the thread path's copy of it.
    fn service_request(&mut self) -> bool {
        let want = self.prog.requested();
        if want <= self.asm.progress() {
            return true;
        }
        match self.asm.finalize_upto(want) {
            Ok(()) => {
                self.prog.clear_request(want);
                self.prog.set(self.asm.progress());
                true
            }
            Err(AssembleError::Aborted) => {
                self.stop.stop(StopReason::Fatal {
                    station: self.idx,
                    message: "assemble aborted".to_string(),
                });
                false
            }
            Err(e) => {
                eprintln!("corr-async: station {}: {e}", self.idx);
                self.stop.stop(StopReason::Fatal {
                    station: self.idx,
                    message: e.to_string(),
                });
                false
            }
        }
    }

    /// Place the one packet this station is holding, if it is holding one.
    ///
    /// A refusal is not consumption: the packet stays in the buffer and the caller
    /// retries it after parking, which is what the thread path's `wait_free` loop
    /// does with its own packet.
    fn place_one(&mut self) -> Batch {
        let t0 = Instant::now();
        let mut outcome = Batch::Drained;
        if self.have_packet {
            // Split borrows: the packet borrows the receive buffer, the placement
            // decision is the assembler's, and the two are disjoint fields.
            let Some(pkt) = Packet::parse(&self.buf, self.len) else {
                // Shorter than a counter: the thread path ignores it too, and
                // counting it as anything else would make the two reports disagree
                // on a case neither can do anything about.
                self.have_packet = false;
                self.publish();
                self.ns_assemble += t0.elapsed().as_nanos() as u64;
                return Batch::Drained;
            };
            match self.asm.try_on_packet(pkt.counter, pkt.samples) {
                Ok(FeedOutcome::Taken) => {
                    self.packets += 1;
                    self.budget = self.budget.saturating_sub(1);
                    self.have_packet = false;
                }
                Ok(FeedOutcome::SlotBusy { slot_seq }) => {
                    // Nothing consumed: `have_packet` stays set so the packet is
                    // tried again once the slot is released.
                    self.parked_slot = Some(slot_seq);
                    outcome = Batch::SlotBusy;
                }
                Err(e @ AssembleError::CounterEpochChange { from, to, back }) => {
                    // The run's observation is over; say so as itself, not as a
                    // fatal error, because the caller can start again.
                    eprintln!("corr-async: station {}: {e}", self.idx);
                    self.stop.stop(StopReason::CounterEpochChange {
                        station: self.idx,
                        from,
                        to,
                        back,
                    });
                    outcome = Batch::Fatal;
                }
                Err(e) if e.is_fatal() => {
                    if !matches!(e, AssembleError::Aborted) {
                        eprintln!("corr-async: station {}: {e}", self.idx);
                        self.stop.stop(StopReason::Fatal {
                            station: self.idx,
                            message: e.to_string(),
                        });
                    }
                    outcome = Batch::Fatal;
                }
                Err(e) => {
                    eprintln!("corr-async: station {}: {e}", self.idx);
                    self.have_packet = false;
                }
            }
        }
        let ns = t0.elapsed().as_nanos() as u64;
        self.ns_assemble += ns;
        self.dbg.ns_consume += ns;
        // Publish *before* the caller can wait, whatever stopped the walk --
        // including a refused slot. This is invariant 1 (docs/ARCHITECTURE.md
        // section 3) and it is what the thread path gets for free by publishing at
        // the end of every packet: the scheduler's permission to release the slot
        // this station is about to wait for is derived from the progress published
        // here. Parking without publishing is the documented "one packet short of
        // the boundary" deadlock, and on the 16384-channel profile it cost the run
        // its rows entirely: the watermark stalled, releases stopped, and the only
        // thing that moved the run along was the 200 ms stall detector.
        self.publish();
        outcome
    }

    /// One datagram off the socket, if it has one.
    ///
    /// `Ok(false)` is a dry socket. That is not just information for this loop:
    /// `tokio` holds readiness in a cache of its own and clears it when an
    /// operation says `WouldBlock`, so this is also how the reactor learns that it
    /// has to wait for the next edge.
    fn receive_one(&mut self) -> io::Result<bool> {
        match self.sock.try_recv_from(&mut self.buf) {
            Ok((n, _peer)) => {
                // The peer address is discarded: which board a stream comes from is
                // decided by the group this socket joined, not by the datagram.
                self.len = n;
                if n > PACKET_BYTES {
                    self.oversized += 1;
                }
                self.dbg.ready_events += 1;
                Ok(true)
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// The station's task, from the first poll to the last.
    ///
    /// The socket is owned here, and so is its reactor registration: `tokio`'s
    /// `UdpSocket` holds the descriptor. That is what makes the drop order somebody
    /// else's problem -- the version before this one registered a *copy* of the raw
    /// fd with `AsyncFd` and had to borrow it rather than own it, so that the
    /// registration went first. Getting that wrong deregisters whatever file the
    /// process opens next, in a run that otherwise looks healthy.
    async fn run(&mut self) -> RecvOutcome {
        let mut tick = tokio::time::interval(self.g.poll_interval);
        // Late ticks must not stack up: a worker that was busy for 10 ms would
        // otherwise fire ten immediate ticks, each of which would run a full
        // iteration of the loop for nothing. `Delay` re-arms from now instead of
        // catching up.
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await; // the first tick completes at once; discard it

        let mut park_rounds = 0u64;
        while !self.stop.is_stopped() {
            self.dbg.iterations += 1;
            // A ring that has been aborted will never release another slot, so
            // nothing this station reads can be placed again. The thread path
            // discovers this the moment its next packet is refused (the packet
            // arrives, `wait_free` returns false, `AssembleError::Aborted` unwinds
            // the thread) -- which it cannot do while a station is quiet, and that
            // is the known "the stream stops, then the duration expires" hang.
            // Asking the ring directly costs one atomic load per iteration and
            // closes that window: a shutdown no longer depends on more packets
            // arriving.
            if self.g.ring.is_aborted() {
                break;
            }
            // Service the scheduler's request first, before anything that can wait:
            // the same ordering rule the thread path follows, and the reason a
            // station that has gone quiet cannot freeze the watermark.
            if !self.service_request() {
                break;
            }

            if self.have_packet {
                match self.place_one() {
                    Batch::Drained => {
                        park_rounds = 0;
                        // The publish is inside `place_one`, before anything can wait.
                    }
                    Batch::SlotBusy => {
                        // The ring belongs to the pipeline. Park *this* station:
                        // the others on this worker keep draining, which is the
                        // whole reason this backend exists.
                        park_rounds += 1;
                        self.ring_refusals += 1;
                        self.dbg.parks += 1;
                        // Blocked on the ring is not silent: the front end is still
                        // sending and the packets are queueing in the socket.
                        self.prog.mark_packet();
                        if park_rounds <= PARK_YIELDS {
                            // Cheap first: let the siblings run, then ask again.
                            tokio::task::yield_now().await;
                        } else {
                            // Then wait for the release itself. The slot the next
                            // packet needs is the one this station is refused, and
                            // the scheduler's request has to keep being serviced
                            // while we wait, so the ring and the tick race -- the
                            // tick is the only thing that can end the wait for a
                            // station whose stream really has stopped.
                            let slot_seq = self.parked_slot.unwrap_or(0);
                            let mut ready = RingReady::new(&self.g.ring, slot_seq);
                            let t_wait = Instant::now();
                            tokio::select! {
                                _ = &mut ready => {}
                                _ = tick.tick() => {
                                    self.dbg.tick_fires += 1;
                                }
                            }
                            // Reported through the assembler so the run log's
                            // "of which Y s waiting for a staging slot" means the
                            // same thing for this backend as for the thread one.
                            let ns = t_wait.elapsed().as_nanos() as u64;
                            self.ns_slot_wait += ns;
                            self.asm.note_slot_wait(ns);
                            self.prog.set_slot_wait_ns(self.ns_slot_wait);
                            self.dbg.ring_wait_polls += ready.polls;
                        }
                    }
                    Batch::Fatal => break,
                }
                if !self.stop.is_stopped() && self.budget == 0 {
                    // Fairness, not throughput: see DRAIN_BUDGET_PACKETS.
                    self.budget = DRAIN_BUDGET_PACKETS;
                    self.dbg.yields += 1;
                    tokio::task::yield_now().await;
                }
                continue;
            }

            // Nothing held: wait for the socket, but never for ever. The tick is
            // what bounds how long a silent station may ignore a request.
            //
            // The tick's own counter is bumped after the select rather than inside
            // its branch body: `self.sock.readable()` holds a borrow of a field of
            // `self` for as long as the select lasts, and keeping the bodies to
            // plain locals keeps that from mattering.
            let t0 = Instant::now();
            let mut ticked = false;
            let ready = tokio::select! {
                _ = tick.tick() => {
                    ticked = true;
                    None
                }
                r = self.sock.readable() => Some(r),
            };
            if ticked {
                self.dbg.tick_fires += 1;
            }
            let dry = match ready {
                // A tick: not a packet, but not nothing either -- the loop goes
                // round and services the scheduler's request, which is the point.
                None => true,
                Some(Err(e)) => {
                    eprintln!("corr-async: station {}: reactor error: {e}", self.idx);
                    true
                }
                // Readable. One packet now; draining the rest of what the socket
                // holds is the next thing to try, and the counters are here to say
                // whether it is worth it.
                Some(Ok(())) => match self.receive_one() {
                    Ok(got) => !got,
                    Err(e) => {
                        eprintln!("corr-async: station {}: receive error: {e}", self.idx);
                        true
                    }
                },
            };
            let waited = t0.elapsed().as_nanos() as u64;
            self.ns_recv += waited;
            if dry {
                // A tick, or a readiness that yielded nothing: either way the
                // socket was dry, which is what an empty poll means here.
                self.dbg.ns_wait += waited;
                self.idle_polls += 1;
                self.dbg.dry_reads += 1;
                continue;
            }

            let stall_ms = self.last_read.elapsed().as_millis() as u64;
            self.last_read = Instant::now();
            if stall_ms > self.max_stall_ms {
                self.max_stall_ms = stall_ms;
            }
            if stall_ms >= 20 {
                self.stalls_over_20ms += 1;
            }
            self.batches += 1;
            self.dbg.batches += 1;
            self.prog.mark_packet();
            self.prog.add_packets(1);
            self.have_packet = true;
        }

        if std::env::var_os("CORR_ASYNC_STATS").is_some() {
            let d = self.dbg;
            eprintln!(
                "corr-async: station {:2}: iters={} ticks={} ready={} dry={} \
                 batches={} yields={} parks={} consume={:.3}s wait={:.3}s pkts={} \
                 ring_refusals={} ring_wait_polls={}",
                self.idx,
                d.iterations,
                d.tick_fires,
                d.ready_events,
                d.dry_reads,
                d.batches,
                d.yields,
                d.parks,
                d.ns_consume as f64 / 1e9,
                d.ns_wait as f64 / 1e9,
                self.packets,
                self.ring_refusals,
                d.ring_wait_polls,
            );
        }
        // One last publish, so the report reflects everything consumed.
        self.prog.set(self.asm.progress());
        RecvOutcome {
            stats: self.asm.stats(),
            progress: self.asm.progress(),
            batches: self.batches,
            packets: self.packets,
            // The socket's own counter stays zero here, and always did: this path
            // reads through its own non-blocking calls, and an empty socket is an
            // empty poll. What it does not have that the socket would is a place to
            // count an oversized datagram, so this station counts its own.
            empty_polls: self.idle_polls,
            oversized: self.oversized,
            ns_recv: self.ns_recv,
            ns_assemble: self.ns_assemble,
            max_stall_ms: self.max_stall_ms,
            stalls_over_20ms: self.stalls_over_20ms,
        }
    }
}

/// Run one station to completion. Never returns an error: a failure that ends the
/// run is reported through `RunStop`, exactly as the thread path does it.
pub(crate) async fn station_task(init: StationInit, g: Arc<RunGeometry>) -> (usize, RecvOutcome) {
    let idx = init.idx;
    // No drop dance here any more: the socket carries its own reactor
    // registration, so the order they go in cannot be got wrong.
    let mut station = Station::new(init, g);
    let outcome = station.run().await;
    (idx, outcome)
}
