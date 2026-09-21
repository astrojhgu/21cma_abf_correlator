//! The scheduler loop and the GPU hand-off.
//!
//! This is the part of the run that decides when a staging slot goes to the
//! device, when a row is closed and when it is fetched. It used to be the middle
//! of `run_receive`, which also opened sockets, started threads, joined them and
//! built the report -- so changing the loop meant reading the receive path too.
//!
//! Everything it needs is a field, deliberately: the borrows are the run's
//! long-lived state, the owned values are the loop's own bookkeeping. Two
//! consequences a reader will trip over otherwise:
//!
//!  * it owns the device handle and the writer, because `run_receive` reads them
//!    back after the loop (to snapshot per-device counters and close the files);
//!  * the submit path keeps `block_slots - ring_lead` slots in flight, and the
//!    lead is what the receive side has to read ahead into -- see
//!    `docs/ARCHITECTURE.md` section 3.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::assemble::SlotRing;
use crate::config::RunPlan;
use crate::run::RunOptions;
use crate::gpu::Correlator;
use crate::monitor::Monitor;
use crate::run::flush_rows;
use crate::stop::{RunStop, StopReason};
use crate::sched::{Action, RowScheduler, StationProgress};
use crate::writer::Writer;

/// The scheduler loop and the GPU hand-off, as one object.
///
/// This used to be the middle 400 lines of `run_receive`, which also opened
/// sockets, started threads, joined them and assembled the report. The state it
/// needs is explicit here: what it borrows (the plan, the device, the ring, the
/// writer, the stations' progress records, the stop flag) and what it owns (the
/// scheduler, the slots in flight, the fetch buffer and the diagnostics counters).
/// Reading the loop no longer requires reading the receive path.
pub(crate) struct Pipeline<'a> {
    pub(crate) plan: &'a RunPlan,
    pub(crate) opts: RunOptions,
    pub(crate) gpu: Option<Correlator>,
    pub(crate) ring: &'a Arc<SlotRing>,
    pub(crate) writer: Option<Writer>,
    /// Where each written row's autocorrelations go, when the status endpoint is on
    /// and somebody can read them.
    pub(crate) autospec: Option<Arc<crate::autospectra::Cell>>,
    pub(crate) progress: &'a [Arc<StationProgress>],
    pub(crate) stop: &'a Arc<RunStop>,
    pub(crate) n_station: usize,
    pub(crate) max_in_flight: usize,
    pub(crate) sched: RowScheduler,
    pub(crate) in_flight: std::collections::VecDeque<u64>,
    pub(crate) zf_mark: Vec<u64>,
    pub(crate) fetch_buf: Vec<u8>,
    pub(crate) pending_meta: Vec<(u64, i64)>,
    pub(crate) timeline: Vec<(u64, u64, u64)>,
    pub(crate) last_dbg: Instant,
    pub(crate) stall_dbg: Instant,
    pub(crate) last_sample: Instant,
    pub(crate) sampled_once: bool,
    pub(crate) prev_lost: u64,
    pub(crate) prev_prog: u64,
    /// Consecutive one-second samples in which no station advanced its finalized
    /// progress at all, while every station was still receiving. See
    /// [`StopReason::StreamNotAdvancing`].
    pub(crate) flat_seconds: u64,
    /// Consecutive one-second samples in which no station read a single packet:
    /// the front end, or the path to it, has stopped. See
    /// [`StopReason::StreamQuiet`].
    pub(crate) quiet_seconds: u64,
    /// Packets read across all stations at the previous sample.
    pub(crate) prev_packets: u64,
    pub(crate) prev_zf: u64,
    /// How many times the loop asked the scheduler for work and got nothing, and
    /// what those polls cost. The device report says how long the *library* calls
    /// took; whatever is left of the run is this loop, so when a row shorter than
    /// a few milliseconds underperforms, this is where the missing time went --
    /// each `Idle` costs a whole poll back-off.
    pub(crate) idle_polls: u64,
    pub(crate) idle_ns: u64,
    /// Time inside `RowScheduler::step`, the loop's own deciding rather than
    /// waiting. `step` walks every station to spot a stalled one, so it is the
    /// one part of the loop whose cost grows with the station count.
    pub(crate) step_ns: u64,
    /// Per-arm wall clock, and the two library calls inside the row-closing arm
    /// that the device counters do not cover.
    pub(crate) ns_submit_arm: u64,
    pub(crate) ns_close_arm: u64,
    pub(crate) ns_row_end: u64,
    pub(crate) ns_pending: u64,
    /// Inside the row fetch: `[autospec, write_row, the rest of flush_rows]`, ns.
    pub(crate) flush_ns: [u64; 3],
    pub(crate) t0: Instant,
    pub(crate) elapsed: Duration,
    /// Stop once the watermark has been lifted this long; `None` disables the check.
    pub(crate) lift_stop_after: Option<Duration>,
    /// The live line. `None` when `monitoring.interval_ms` is 0.
    pub(crate) monitor: Option<Monitor>,
}

impl<'a> Pipeline<'a> {
    /// Run until the duration elapses, the stop flag is set, or a station
    /// fabricates more silence than the cap allows. The caller keeps ownership of
    /// everything borrowed and reads the outcome back off this struct.
    pub(crate) fn run(&mut self) -> Result<()> {
        let t0 = Instant::now();
        while !self.stop.is_stopped() {
            if let Some(d) = self.opts.duration {
                if t0.elapsed() >= d {
                    break;
                }
            }
            // A station that has declared more missing frames than the configured
            // cap allows is not coming back. Stop rather than keep writing rows of
            // silence that look like observations, and rather than let the receive
            // buffers grow to hold data nobody will use.
            // The *second* line of defence. A station's own receive thread caps
            // its fabricated frames first (AssembleError::ZeroFillCapExceeded, which
            // arrives here as a fatal reason), so this fires only when that thread is
            // not honouring the requests at all -- a dead or wedged station. Kept
            // for that case, and it is the one reason in RunStop that a live run has
            // not exercised.
            if let Some(ex) = self.sched.zero_fill_exceeded() {
                // The station's *name* is presentation; the reason carries the
                // index and the numbers, and `StopReason::describe` is what turns
                // it into the sentence the run report has always printed.
                self.stop.stop(StopReason::ZeroFillCap {
                    name: self
                        .plan
                        .enabled
                        .get(ex.station)
                        .map(|e| e.name.clone())
                        .unwrap_or_else(|| format!("index {}", ex.station)),
                    station: ex.station,
                    fabricated: ex.frames,
                    limit: ex.limit,
                });
                break;
            }
            let step_t0 = Instant::now();
            let action = self.sched.step(step_t0);
            self.step_ns += step_t0.elapsed().as_nanos() as u64;
            match action {
                Action::Submit {
                    slot_seq,
                    frames,
                    row,
                } => {
                    let _ = row;
                    let arm_t0 = Instant::now();
                    // The first hand-over is what makes the staging the consumer's:
                    // from here on the self.ring may not move its window, because a copy
                    // may be in flight for any physical slot in it. Doing this before
                    // the accumulate is what makes "window open" mean "nothing has
                    // been read", not "nothing has been read yet, probably".
                    self.ring.close_window();
                    match self.gpu.as_ref() {
                        Some(corr) => {
                            corr.row_accumulate(slot_seq, frames)
                                .with_context(|| format!("accumulating slot sequence {slot_seq}"))?;
                            self.in_flight.push_back(slot_seq);
                            // Release the oldest slot, keeping `lead` slots writable
                            // for the receive side.
                            //
                            // The lead is the whole story of this loop, and it was
                            // measured the hard way. `block_slots - 1` in flight --
                            // "use the self.ring to the brim" -- leaves exactly *one*
                            // writable slot, which forces a receive thread to acquire
                            // a slot once per slot period with no margin whatsoever.
                            // Any jitter then costs packets that cannot be recovered,
                            // because reading ahead needs a writable slot: at 24
                            // stations that configuration received 1.68M of 3.12M
                            // packets per station and zero filled 41% of every row.
                            // Widening the lead to 7 (one slot in flight) received all
                            // 3.12M with 76 packets lost in 16 s, and did so *without*
                            // costing host->device throughput: 18.8 GB/s aggregate
                            // against 18.4 GB/s at four slots in flight, because a
                            // single copy stream per card already exceeds what one
                            // card's link can carry here.
                            // Computed once, before the loop; see `pipeline_slots`.
                            while self.in_flight.len() > self.max_in_flight {
                                let old = self.in_flight.pop_front().expect("non-empty");
                                corr.row_slot_free(old)
                                    .with_context(|| format!("releasing slot sequence {old}"))?;
                                self.ring.release(old);
                            }
                        }
                        None => {
                            // A dry run has no GPU: the slot is handed straight back,
                            // which is what the real path does after the copy.
                            self.ring.release(slot_seq);
                        }
                    }
                    self.ns_submit_arm += arm_t0.elapsed().as_nanos() as u64;
                }
                Action::CloseRow {
                    frames,
                    row,
                    open_ms,
                } => {
                    let arm_t0 = Instant::now();
                    match self.gpu.as_ref() {
                        Some(corr) => {
                            corr.row_end(frames).context("closing the row on the GPU")?;
                            self.ns_row_end += arm_t0.elapsed().as_nanos() as u64;
                            let pend_t0 = Instant::now();
                            let pending = corr.rows_pending()?;
                            self.ns_pending += pend_t0.elapsed().as_nanos() as u64;
                            self.pending_meta.push((row, open_ms));
                            if pending >= self.plan.rows_per_fetch as i32 {
                                // A file set that cannot be opened for the new
                                // local day stops the run *with a reason the report
                                // carries*, like the zero-fill cap: propagating it
                                // as a bare error would end the run without the
                                // artifact that says why.
                                let rows_out = match flush_rows(
                                    corr,
                                    self.plan,
                                    self.progress,
                                    &mut self.zf_mark,
                                    &mut self.pending_meta,
                                    &mut self.fetch_buf,
                                    self.writer.as_mut(),
                                    self.autospec.as_deref(),
                                    &mut self.flush_ns,
                                ) {
                                    Ok(n) => n,
                                    Err(e) => {
                                        if let Some(rf) =
                                            e.downcast_ref::<crate::writer::RolloverFailed>()
                                        {
                                            self.stop.stop(StopReason::RolloverFailed {
                                                prefix: rf.prefix.clone(),
                                                detail: rf.detail.clone(),
                                            });
                                            break;
                                        }
                                        return Err(e);
                                    }
                                };
                                let _ = rows_out;
                                self.sched.note_row_fetched(self.plan.rows_per_fetch);
                            }
                        }
                        None => {
                            if let Some(w) = self.writer.as_mut() {
                                // No GPU means no spectra; the dry run only exercises
                                // the receive path, so nothing is written.
                                let _ = w;
                            }
                            self.sched.note_row_fetched(1);
                        }
                    }
                    self.ns_close_arm += arm_t0.elapsed().as_nanos() as u64;
                }
                Action::Idle => {
                    // One sample per second of what every station did, taken from the
                    // thread that has the spare time to take it. A run that loses data
                    // can lose it steadily or in cascades, and the difference decides
                    // the fix, so the self.timeline is printed rather than guessed at.
                    let now = Instant::now();
                    self.idle_polls += 1;
                    let idle_t0 = now;
                    // A lifted watermark means a station is being left behind on purpose, and
                    // its frames are going into blocks.csv as missing. Unlike a socket backlog
                    // -- which drains once the machine quiets down -- this never clears itself,
                    // so past the configured grace period the honest thing is to stop rather
                    // than write more hours of data that is short on those frames.
                    if let Some(limit) = self.lift_stop_after {
                        let lifted = self.sched.note_lift(now);
                        if lifted >= limit {
                            let slots = self.sched.watermark_lift() as f64
                                / self.sched.frames_per_slot().max(1) as f64;
                            self.stop.stop(StopReason::LiftCap {
                                seconds: lifted.as_secs(),
                                slots_milli: (slots * 1000.0) as u64,
                                station: self.sched.watermark_station().unwrap_or(0),
                            });
                        }
                    }
                    // The live line, on its own interval (monitoring.interval_ms).
                    // Printed from here because this is the one thread that is idle
                    // by construction, and because both receivers reach it.
                    if let Some(m) = self.monitor.as_mut() {
                        // The set tag is read live rather than captured once: the
                        // thread opens the next set at a day boundary, and a monitor
                        // showing the previous set's name is worse than none.
                        // Everything the sampler needs to know about the writer,
                        // read once: the set tag (a rollover changes it mid-run),
                        // how full the queue is (the early warning) and how many rows
                        // it has had to drop (the damage).
                        let wstat = self.writer.as_ref().map(|w| crate::monitor::WriterStatus {
                            set_tag: w.current_set_tag(),
                            queued: w.queued(),
                            queue_depth: w.queue_depth() as u64,
                            max_queued: w.max_queue_depth(),
                            dropped: w.dropped_rows(),
                        });
                        m.tick(
                            &self.sched,
                            self.progress,
                            self.ring.as_ref(),
                            self.gpu.as_ref(),
                            wstat.as_ref(),
                            now,
                        );
                    }
                    if now.duration_since(self.last_sample).as_millis() >= 1000 {
                        self.last_sample = now;
                        let mut lost = 0u64;
                        let mut prog = 0u64;
                        let mut zf = 0u64;
                        for i in 0..self.n_station {
                            lost += self.sched.lost_of(i);
                            prog += self.sched.progress_of(i);
                            zf += self.sched.zero_filled_of(i);
                        }
                        // The first sample has no previous value to subtract from,
                        // and self.progress is a frame id (1e10 at the real rate), so a
                        // naive delta prints that instead of a rate.
                        let first = !self.sampled_once;
                        self.sampled_once = true;
                        let d_lost = if first { 0 } else { lost.saturating_sub(self.prev_lost) };
                        let d_prog = if first { 0 } else { prog.saturating_sub(self.prev_prog) };
                        let d_zf = if first { 0 } else { zf.saturating_sub(self.prev_zf) };
                        (self.prev_lost, self.prev_prog, self.prev_zf) = (lost, prog, zf);
                        if self.timeline.len() < 40 && (d_lost > 0 || self.timeline.len() < 3) {
                            self.timeline.push((d_prog / self.n_station as u64, d_lost / self.n_station as u64, d_zf / self.n_station as u64));
                        }

                        // Two watchdogs for the failures that leave no trace anywhere
                        // else, told apart by the one signal that cannot lie: how many
                        // packets the stations actually took off their sockets.
                        //
                        // `last_packet_ns` cannot be used for this -- a thread blocked
                        // on a full ring marks itself alive on purpose, which is right
                        // for the stall detector and useless here. Measured the hard
                        // way: a front end that went quiet for the night was reported
                        // as a frame-numbering mismatch, because "silent" looked the
                        // same as "busy".
                        let mut packets = 0u64;
                        for i in 0..self.n_station {
                            packets += self.sched.packets_of(i);
                        }
                        let d_packets = if first { 0 } else { packets.saturating_sub(self.prev_packets) };
                        self.prev_packets = packets;

                        if !first && d_prog == 0 {
                            if d_packets > 0 {
                                // Packets arriving, no frame placeable: they are all
                                // landing before the run's start frame or behind a frame
                                // already finalized. See StreamNotAdvancing for the two
                                // ways this happened for real. Consecutive zero-progress
                                // samples keep it rate independent -- a deliberately slow
                                // replay still advances several frames inside the window.
                                self.flat_seconds += 1;
                                self.quiet_seconds = 0;
                                if self.flat_seconds >= stream_flat_limit_s() {
                                    self.stop.stop(StopReason::StreamNotAdvancing {
                                        start_frame: self.sched.row_start_frame(),
                                        progress: (0..self.n_station)
                                            .map(|i| self.sched.progress_of(i))
                                            .min()
                                            .unwrap_or(0),
                                        seconds: self.flat_seconds,
                                        n_station: self.n_station,
                                    });
                                }
                            } else {
                                // Not one packet at any station: the front end stopped.
                                self.flat_seconds = 0;
                                self.quiet_seconds += 1;
                                if self.quiet_seconds >= stream_quiet_limit_s() {
                                    self.stop.stop(StopReason::StreamQuiet {
                                        seconds: self.quiet_seconds,
                                        start_frame: self.sched.row_start_frame(),
                                        progress: (0..self.n_station)
                                            .map(|i| self.sched.progress_of(i))
                                            .min()
                                            .unwrap_or(0),
                                        n_station: self.n_station,
                                    });
                                }
                            }
                        } else {
                            self.flat_seconds = 0;
                            self.quiet_seconds = 0;
                        }
                    }
                    if std::env::var_os("CORR_PIPE_DEBUG").is_some() {
                        let now = Instant::now();
                        if now.duration_since(self.last_dbg).as_millis() >= 500 {
                            self.last_dbg = now;
                            let low = self.sched.watermark_station().unwrap_or(0);
                            eprintln!(
                                "pipe: next={} watermark={} low_st={} low_prog={} low_silence={}ms \
                                 requested={} row_start={} slots={} self.in_flight={}",
                                self.sched.next_frame(),
                                self.sched.watermark(),
                                low,
                                self.sched.progress_of(low),
                                self.sched.silence_ms(low, now),
                                self.sched.requested_of(low),
                                self.sched.row_start_frame(),
                                self.sched.stats().slots_submitted,
                                self.in_flight.len()
                            );
                            // Per-station distribution, once the stall is old enough
                            // to be real: a single "low=752" cannot distinguish one
                            // station lagging a batch behind from every station parked
                            // on the same missing release.
                            if now.duration_since(self.stall_dbg).as_millis() >= 1500 {
                                self.stall_dbg = now;
                                let base = self.sched.row_start_frame();
                                let mut all = String::new();
                                for i in 0..self.n_station {
                                    all.push_str(&format!(" {}:{}", i, self.sched.progress_of(i) - base));
                                }
                                eprintln!("      offsets from row start (frames):{all}");
                            }
                            eprintln!(
                                "      high={} row_end={} {}",
                                self.sched.highest_progress(),
                                self.sched.row_start_frame() + self.plan.n_cum as u64,
                                self.ring.wait_snapshot()
                            );
                        }
                    }
                    std::thread::sleep(idle_backoff());
                    self.idle_ns += idle_t0.elapsed().as_nanos() as u64;
                }
            }
        }
        self.elapsed = self.t0.elapsed();
        Ok(())
    }
}

/// How many consecutive one-second samples with no progress at all the watchdog
/// in the pipeline loop tolerates before it stops the run. Ten is far above the
/// ~1 s a healthy run spends discarding pre-start packets, and far below the
/// hours a run in this state would otherwise sit there.
///
/// `CORR_STREAM_FLAT_S` overrides it, for a front end slow enough to advance less
/// than one frame per sample on purpose.
/// How long every station may go without reading a single packet before the run
/// stops. A front end that is coming back within seconds or a minute is common;
/// one that is not would otherwise cost the whole zero-fill budget in rows where
/// every frame is fabricated. `CORR_STREAM_QUIET_S` overrides it.
fn stream_quiet_limit_s() -> u64 {
    std::env::var("CORR_STREAM_QUIET_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v >= 5)
        .unwrap_or(60)
}

fn stream_flat_limit_s() -> u64 {
    std::env::var("CORR_STREAM_FLAT_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v >= 3)
        .unwrap_or(10)
}

/// How long the loop backs off when the scheduler has nothing to do.
///
/// This was suspected of capping the row rate, on the theory that a poll landing
/// one frame early would pay the whole back-off. Measured, it is not: at one
/// frame per row (40.96 us) the loop polls and finds nothing 3 or 4 times in ten
/// seconds, because a saturated pipeline is never idle, and spinning instead
/// (0 us) made the run *slower* by burning a core the receive threads want. So
/// the back-off stays, and it is a lag on an unsaturated loop rather than a cap
/// on a busy one: with the row path taking ~700 us of library time per row, the
/// loop only reaches this arm once rows are long enough for 200 us not to matter.
pub(crate) fn idle_backoff() -> Duration {
    Duration::from_micros(200)
}
