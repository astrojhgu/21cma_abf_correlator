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
    pub(crate) prev_zf: u64,
    pub(crate) t0: Instant,
    pub(crate) elapsed: Duration,
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
            // A station that has been asked to zero fill past the configured cap is
            // not coming back. Stop rather than keep writing rows of silence that
            // look like observations, and rather than let the receive buffers grow to
            // hold data nobody will use.
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
            match self.sched.step(Instant::now()) {
                Action::Submit {
                    slot_seq,
                    frames,
                    row,
                } => {
                    let _ = row;
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
                }
                Action::CloseRow {
                    frames,
                    row,
                    open_ms,
                } => {
                    match self.gpu.as_ref() {
                        Some(corr) => {
                            corr.row_end(frames).context("closing the row on the GPU")?;
                            self.pending_meta.push((row, open_ms));
                            if corr.rows_pending()? >= self.plan.rows_per_fetch as i32 {
                                let rows_out = flush_rows(
                                    corr,
                                    self.plan,
                                    self.progress,
                                    &mut self.zf_mark,
                                    &mut self.pending_meta,
                                    &mut self.fetch_buf,
                                    self.writer.as_mut(),
                                )?;
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
                }
                Action::Idle => {
                    // One sample per second of what every station did, taken from the
                    // thread that has the spare time to take it. A run that loses data
                    // can lose it steadily or in cascades, and the difference decides
                    // the fix, so the self.timeline is printed rather than guessed at.
                    let now = Instant::now();
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
                    std::thread::sleep(Duration::from_micros(200))
                }
            }
        }
        self.elapsed = self.t0.elapsed();
        Ok(())
    }
}
