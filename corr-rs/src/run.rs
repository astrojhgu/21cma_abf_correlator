//! The receive path: sockets, assemblers, and the scheduler loop.
//!
//! Startup order is the reason this is not just a few thread spawns, and it was
//! measured rather than reasoned about:
//!
//! 1. Everything slow is created **first**, before a single group is joined: the
//!    CUDA context, the cuFFT plan, the pinned staging, and the output files.
//!    Creating them after the sockets were opened cost a produce-mode run its
//!    whole startup: with no thread draining them, 512 MiB of socket buffer fills
//!    in about 0.6 s at the real packet rate, so by the time the receive threads
//!    existed the streams were seconds old.
//! 2. Then every socket is opened and joins its group.
//! 3. Each socket is drained until every station has been seen, and then for a
//!    short settle period, so that the counters the start frame is derived from
//!    are current rather than merely the first ones ever seen.
//! 4. A start frame is chosen on the global row grid, ahead of every station, and
//!    the staging ring is primed for its slot sequence.
//! 5. The receive threads are spawned.
//!
//! Step 4 cannot come first. A frame's position in the staging ring is
//! `frame_id % frames_per_slot` and its slot sequence is
//! `frame_id / frames_per_slot`, so a ring primed for frame 0 would have the
//! first real packet -- at frame ~5e9 -- wait for a slot release that no pipeline
//! will ever issue. Priming from a sniffed counter also means every station
//! starts on the same frame, which is what makes the watermark meaningful.
//!
//! Even so, the start frame is a prediction, and a station that arrives later
//! than the prediction cannot be served by a ring primed for it. That case is not
//! left to luck: see [`SlotRing::reprime`] for the ring moving itself forward at
//! startup, and [`crate::assemble::AssembleError::StartupGap`] for the refusal
//! that replaces what used to be a silent hang.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::assemble::SlotRing;
pub use crate::receiver::{open_sockets, packet_bytes, require_interface, sniff_first_counters};
use crate::receiver::{recv_thread, RecvThreadArgs};
use crate::pipeline::Pipeline;
use crate::stop::RunStop;
use crate::report::{DeviceReport, RunReport, StationAssembleReport};
use crate::config::{Config, RunPlan};
use crate::gpu::Correlator;
use crate::net::MulticastSocket;
use crate::sched::{RowScheduler, RowSchedulerParams, StationProgress};
use crate::writer::{RowBlock, Writer};

/// What a run is allowed to do.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Stop after this long. `None` runs until interrupted.
    pub duration: Option<Duration>,
    /// Also wait for the first packet of every station before starting.
    /// Turning this off makes a silent station obvious instead of making startup
    /// hang for ever.
    pub wait_for_first_packet: Duration,
    /// Where to write. `None` means do not write, which is what `--dry-run` does.
    pub output_dir: Option<std::path::PathBuf>,
    /// Use the GPU. `false` runs the receive and scheduling path alone.
    pub enable_gpu: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            duration: Some(Duration::from_secs(10)),
            wait_for_first_packet: Duration::from_secs(10),
            output_dir: None,
            enable_gpu: false,
        }
    }
}

/// How many staging slots the pipeline may hold, i.e. `block_slots - ring_lead`.
///
/// A pure function of the plan, deliberately: the previous version memoised the
/// answer in a `OnceLock`, which froze the *first* config's setting for the life
/// of the process -- two configs in one process (a test, or a future server) would
/// silently share the first one's ring layout. That is the same shape of mistake
/// as the ones that cost this project whole afternoons: a value that decides
/// whether data survives, computed once, somewhere else.
///
/// `CORR_RING_LEAD` remains as a temporary experiment override.
pub(crate) fn pipeline_slots(plan: &RunPlan) -> usize {
    let lead = std::env::var("CORR_RING_LEAD")
        .ok()
        .and_then(|v| v.parse().ok())
        .or(plan.ring_lead.map(|v| v as usize))
        .unwrap_or_else(|| (plan.block_slots as usize).saturating_sub(1));
    (plan.block_slots as usize).saturating_sub(lead).max(1)
}

/// Align `frame` up to the next multiple of `n_cum`.
pub fn align_up(frame: u64, n_cum: u64) -> u64 {
    if frame % n_cum == 0 {
        frame
    } else {
        frame + (n_cum - frame % n_cum)
    }
}




/// A staging region handed to exactly one receive thread.
///
/// A raw pointer is not `Send` by default, and it should not be in general. Here
/// it is justified by construction: the allocation is made once, handed to one
/// thread, never aliased, and freed after that thread has been joined.
#[derive(Clone, Copy)]
pub(crate) struct StagingPtr(pub(crate) *mut u8);

// SAFETY: as above -- sole ownership transferred to one thread.
unsafe impl Send for StagingPtr {}


/// Everything that has to exist before the sockets are opened.
///
/// This exists because of a measured failure, not for tidiness. Creating the CUDA
/// context and opening the output files takes seconds -- 300 files on a network
/// filesystem, plus a cuFFT plan -- and while that happened **with the sockets
/// already joined and no thread reading them**, each 512 MiB receive buffer filled
/// up and started dropping. By the time the receive threads existed the streams
/// were seconds old: the first packet they saw was far past the run's start frame,
/// its slot was outside the primed window, and the run hung with zero slots
/// submitted, no rows, 492696 lost packets and no error. A dry run of the same
/// configuration, which creates nothing, received every packet.
pub struct PreparedOutput {
    pub gpu: Option<Correlator>,
    pub writer: Option<Writer>,
    /// Rows already on disk before this run started.
    pub resumed_from_row: u64,
    /// Frame id just past the last row on disk, if there is one.
    ///
    /// Applied to the start frame *after* the counters are read, so a resume
    /// never re-covers a frame range that is already written.
    pub resume_floor: Option<u64>,
}

/// Create the GPU context and open the output files.
///
/// Must be called before any socket is opened: see [`PreparedOutput`].
pub fn prepare_output(cfg: &Config, plan: &RunPlan, opts: &RunOptions) -> Result<PreparedOutput> {
    let started = Instant::now();
    let gpu = if opts.enable_gpu {
        let devices = &cfg.gpu.devices;
        Some(
            Correlator::new(devices, cfg.gpu.split.abi_value(), plan).with_context(|| {
                format!(
                    "creating a corr_cuda context on device(s) {devices:?}, split {:?}",
                    cfg.gpu.split
                )
            })?,
        )
    } else {
        None
    };

    let mut writer = None;
    let mut resumed_from_row = 0u64;
    let mut resume_floor = None;
    if let Some(dir) = &opts.output_dir {
        let hash = crate::config::config_hash(cfg);
        let (w, state) =
            Writer::open(dir, plan, cfg, &hash, cfg.storage.fsync_every_rows)
            .with_context(|| format!("opening the output directory {}", dir.display()))?;
        if !state.is_fresh() {
            println!(
                "resuming {} at row {} (next frame {}), truncated {} partial rows",
                dir.display(),
                state.rows_written,
                state.next_frame_id.unwrap_or(0),
                state.truncated_rows
            );
        }
        resumed_from_row = state.rows_written;
        resume_floor = state.next_frame_id;
        writer = Some(w);
    }

    // The split is performance-relevant and a mis-assignment would show up as a
    // silently zero slice of every row, so it is printed even when nothing is
    // wrong: one line per device, with the baseline range it owns.
    //
    // Station gpu_id hints pointing at a card this run does not use are reported
    // too. They are allowed (that is what makes the fallback to one card a
    // one-line change) but they must not be invisible.
    let outside: Vec<&str> = plan
        .enabled
        .iter()
        .filter(|s| !cfg.gpu.devices.contains(&s.gpu_id))
        .map(|s| s.name.as_str())
        .collect();
    if !outside.is_empty() {
        println!(
            "note       : {} station(s) name a gpu_id outside gpu.devices {:?} ({}{}); \
             those hints are advisory and this run does not use them",
            outside.len(),
            cfg.gpu.devices,
            outside.iter().take(4).cloned().collect::<Vec<_>>().join(", "),
            if outside.len() > 4 { ", ..." } else { "" }
        );
    }
    if let Some(corr) = gpu.as_ref() {
        if let Ok(place) = corr.placement() {
            println!("gpu        : {} device(s), baselines split as", place.len());
            for (i, p) in place.iter().enumerate() {
                let stations = corr
                    .device_stations(i as i32, p.n_station_resident)
                    .map(|v| v.len())
                    .unwrap_or(0);
                println!(
                    "  [{i}] device {} holds {} station(s), baselines [{}..{}) = {}",
                    p.device_id,
                    stations,
                    p.first_baseline,
                    p.first_baseline + p.n_baseline,
                    p.n_baseline
                );
            }
        }
    }

    // Printed before any socket exists, so a run's log shows the order that keeps
    // the receive buffers from filling up while this work happens.
    println!(
        "prepared   : device context and {} output file(s) in {:.1} ms, before any socket was opened",
        plan.n_baseline,
        started.elapsed().as_secs_f64() * 1e3
    );

    Ok(PreparedOutput {
        gpu,
        writer,
        resumed_from_row,
        resume_floor,
    })
}

/// Run the receive path. Returns when the duration elapses or `stop` is set.
///
/// This is the dry run: sockets, assembly, the staging ring and the row
/// scheduler, without a GPU and without files. It is the only way to exercise the
/// parts that real traffic never triggers -- a lost packet, a station that stops,
/// a counter jump -- short of waiting for them to happen during an observation.
/// `resume_floor` is the frame id just past the last row already on disk, if any.
pub fn run_receive(
    cfg: &Config,
    plan: &RunPlan,
    sockets: Vec<MulticastSocket>,
    first_counters: &[u64],
    opts: RunOptions,
    prepared: PreparedOutput,
    stop: Arc<RunStop>,
) -> Result<RunReport> {
    let n_station = plan.n_station;
    if sockets.len() != n_station || first_counters.len() != n_station {
        bail!("sockets and counters must cover every enabled station");
    }

    let fft_len = plan.fft_len;
    let frames_per_slot = plan.frames_per_slot as u64;

    // Start ahead of every station, on the global row grid. n_cum is a multiple
    // of frames_per_slot (validated), so a row-aligned start is slot aligned too.
    let latest_first = first_counters.iter().copied().max().unwrap();
    let mut start_frame = align_up(latest_first / plan.packets_per_frame + 1, plan.n_cum as u64);

    let PreparedOutput {
        gpu,
        writer,
        resumed_from_row,
        resume_floor,
    } = prepared;

    let gpu_banner = gpu.as_ref().map(|_| crate::ffi::abi_banner());

    // The run's start is fixed before the ring is primed: a ring primed for one
    // slot sequence and a run starting at another is exactly the mismatch that
    // used to hang, so the two are computed on adjacent lines and from the same
    // value.
    if let Some(prev_end) = resume_floor {
        let floor = align_up(prev_end, plan.n_cum as u64);
        if floor > start_frame {
            println!("resume floor: starting at frame {floor}, past the {prev_end} already written");
            start_frame = floor;
        }
    }

    let ring = Arc::new(SlotRing::primed(plan.block_slots as usize, start_frame / frames_per_slot));
    let progress: Vec<Arc<StationProgress>> = (0..n_station)
        .map(|_| Arc::new(StationProgress::new(start_frame)))
        .collect();

    // --- GPU and output -------------------------------------------------
    // Created by `prepare_output`, which the caller runs *before* opening a
    // socket: see [`PreparedOutput`] for why the order is load bearing.

    // Where the receive threads write. This has to be the GPU context's pinned
    // staging whenever there is one: it is a different allocation from any buffer
    // we make here, and handing the assemblers local memory while the GPU reads
    // its own produces a run that looks perfect and writes nothing but zeros.
    let staging_bytes = plan.staging_bytes_per_station as usize;
    let staging_layout = std::alloc::Layout::from_size_align(staging_bytes, 4096)
        .expect("staging layout must be valid");
    let mut local_staging: Vec<StagingPtr> = Vec::new();
    let staging: Vec<StagingPtr> = match gpu.as_ref() {
        Some(corr) => corr
            .staging_all()?
            .iter()
            .map(|s| StagingPtr(s.base as *mut u8))
            .collect(),
        None => {
            local_staging = (0..n_station)
                .map(|_| {
                    // SAFETY: a fresh allocation of a non-zero size; freed below,
                    // after every receive thread has been joined.
                    StagingPtr(unsafe { std::alloc::alloc_zeroed(staging_layout) })
                })
                .collect();
            if local_staging.iter().any(|p| p.0.is_null()) {
                bail!("cannot allocate {staging_bytes} bytes of staging per station");
            }
            local_staging.clone()
        }
    };

    // The validity-mask layout comes from the C side, which owns the allocation;
    // `--check` already verified it against this side's arithmetic. A dry run has
    // no device and nothing to zero, so the mask stays disabled there.
    let (validity_offset, validity_bytes_per_slot) = match gpu.as_ref() {
        Some(corr) => {
            let s = corr.staging(0).context("reading the staging layout")?;
            (s.validity_offset, s.validity_bytes_per_slot)
        }
        None => (0, 0),
    };

    let mut handles = Vec::with_capacity(n_station);
    // Where a receive thread records an error the whole run has to answer.
    for (idx, sock) in sockets.into_iter().enumerate() {
        let args = RecvThreadArgs {
            sock,
            base: staging[idx],
            ring: ring.clone(),
            prog: progress[idx].clone(),
            stop: stop.clone(),
            zero_fill_limit_frames: cfg.integration.zero_fill_limit_frames,
            idx,
            cpu: plan.enabled[idx].recv_cpu,
            packets_per_frame: plan.packets_per_frame,
            frames_per_slot,
            frame_stride: fft_len as usize,
            validity_offset,
            validity_bytes_per_slot,
            start_frame,
        };
        handles.push(std::thread::spawn(move || recv_thread(args)));
    }

    let sched = RowScheduler::new(
        progress.clone(),
        RowSchedulerParams {
            frames_per_slot,
            n_cum: plan.n_cum as u64,
            rows_per_fetch: plan.rows_per_fetch,
            start_frame_id: start_frame,
            // Physical time: a station may deliver no packets for this long before
            // the scheduler concludes it has stopped.
            stall_timeout: Duration::from_millis(cfg.integration.block_timeout_ms),
            zero_fill_limit_frames: cfg.integration.zero_fill_limit_frames,
            block_slots: plan.block_slots,
        },
        Instant::now(),
    );

    // Everything the loop owns lives in `Pipeline`; naming it here, once, is the
    // whole of what this function has to know about the loop's state.
    let mut pipeline = Pipeline {
        plan,
        opts: opts.clone(),
        gpu,
        ring: &ring,
        writer,
        progress: &progress,
        stop: &stop,
        n_station,
        max_in_flight: pipeline_slots(plan),
        sched,
        in_flight: std::collections::VecDeque::new(),
        zf_mark: progress.iter().map(|p| p.zero_filled_frames()).collect(),
        fetch_buf: Vec::new(),
        pending_meta: Vec::new(),
        timeline: Vec::new(),
        last_dbg: Instant::now(),
        stall_dbg: Instant::now(),
        last_sample: Instant::now(),
        sampled_once: false,
        prev_lost: 0,
        prev_prog: 0,
        prev_zf: 0,
        t0: Instant::now(),
        elapsed: Duration::ZERO,
    };
    pipeline.run()?;
    let elapsed = pipeline.elapsed;

    // A receive thread may have stopped the run with an error it could not fix
    // (an impossible staging geometry, or a start frame the front end's counter
    // had already left behind). It set the stop flag, which is why the loop above
    // ended, so this is read rather than polled.
    let fatal_stop: Option<String> = stop.fatal_message();
    // Release any receive thread still waiting for a slot. Without this the join
    // below waits for ever: the thread is spinning on a release the pipeline will
    // never issue, and never looks at the stop flag.
    ring.abort();

    let mut stations = Vec::with_capacity(n_station);
    for (idx, h) in handles.into_iter().enumerate() {
        match h.join() {
            Ok(o) => stations.push(StationAssembleReport {
                name_idx: idx,
                stats: o.stats,
                progress: o.progress,
                recv_batches: o.batches,
                recv_packets: o.packets,
                recv_empty_polls: o.empty_polls,
                recv_truncated: o.truncated,
                ns_recv: o.ns_recv,
                ns_assemble: o.ns_assemble,
                ns_slot_wait: o.stats.ns_slot_wait,
                max_stall_ms: o.max_stall_ms,
                stalls_over_20ms: o.stalls_over_20ms,
            }),
            Err(_) => stations.push(StationAssembleReport {
                name_idx: idx,
                ..Default::default()
            }),
        }
    }

    // Whatever the last partial fetch batch left behind, and a clean close.
    // Destructured rather than reached through `pipeline.` one field at a time:
    // the device handle is borrowed while the fetch buffers are borrowed mutably,
    // and only disjoint field borrows let those coexist.
    // Scoped so the mutable borrow of `pipeline` ends here: the code below reads
    // the loop's counters, and holding the destructuring borrow alive would make
    // those reads a conflict.
    let rows_written = {
        let Pipeline {
            gpu,
            zf_mark,
            pending_meta,
            fetch_buf,
            writer,
            ..
        } = &mut pipeline;
        if let Some(corr) = gpu.as_ref() {
            if corr.rows_pending()? > 0 {
                flush_rows(
                    corr,
                    plan,
                    &progress,
                    zf_mark,
                    pending_meta,
                    fetch_buf,
                    writer.as_mut(),
                )?;
            }
        }
        match writer.as_mut() {
            Some(w) => {
                w.finish().context("flushing the output")?;
                w.rows_written()
            }
            None => 0,
        }
    };

    for p in &local_staging {
        // SAFETY: allocated above with this layout, and no receive thread holds
        // it any more. The GPU's staging is owned by the context and is not ours
        // to free.
        unsafe { std::alloc::dealloc(p.0, staging_layout) };
    }

    // Snapshot the counters while the context is still alive: they are the only
    // record of how much each card actually pulled, which is the question the
    // multi-device split exists to answer.
    let mut devices = Vec::new();
    if let Some(corr) = pipeline.gpu.as_ref() {
        if let Ok(place) = corr.placement() {
            for (i, p) in place.iter().enumerate() {
                let stats = corr.device_stats(i as i32).unwrap_or_default();
                devices.push(DeviceReport {
                    device_id: p.device_id,
                    n_station_resident: p.n_station_resident,
                    first_baseline: p.first_baseline,
                    n_baseline: p.n_baseline,
                    bytes_h2d: stats.bytes_h2d,
                    transfers: stats.transfers,
                    ns_accumulate: stats.ns_accumulate,
                    ns_slot_free: stats.ns_slot_free,
                    ns_rows_wait: stats.ns_rows_wait,
                    ns_fetch: stats.ns_fetch,
                });
            }
        }
    }

    let ss = pipeline.sched.stats();
    Ok(RunReport {
        start_frame,
        rows_closed: ss.rows_closed,
        slots_submitted: ss.slots_submitted,
        frames_scheduled: ss.frames_scheduled,
        stall_recoveries: ss.stall_recoveries,
        zero_fill_requested_frames: ss.zero_fill_requested_frames,
        zero_fill_limit_frames: cfg.integration.zero_fill_limit_frames,
        recoveries_per_station: ss.recoveries_per_station,
        requested_per_station: ss.requested_per_station,
        stall_timeout: pipeline.sched.stall_timeout(),
        stopped_zero_fill_exceeded: stop.zero_fill_message(),
        stopped_fatal: fatal_stop,
        rows_written,
        output_dir: opts.output_dir.clone(),
        resumed_from_row,
        used_gpu: pipeline.gpu.is_some(),
        gpu_banner,
        elapsed,
        stations,
        devices,
        ring_blocks: ring.block_events(),
        ring_blocked: ring.blocked_now(),
        timeline: pipeline.timeline,
    })
}


/// Wait for the pending rows, fetch them in one transfer, and write them out.
///
/// `pending_meta` and `zf_mark` are the per-row bookkeeping the GPU does not
/// have: which row index and wall clock each fetched row belongs to, and how many
/// frames each station declared missing since the previous row (those are the
/// frames that contributed silence, which is what the `zero_filled` column has
/// always meant).
#[allow(clippy::too_many_arguments)]
pub(crate) fn flush_rows(
    corr: &Correlator,
    plan: &RunPlan,
    progress: &[Arc<StationProgress>],
    zf_mark: &mut [u64],
    pending_meta: &mut Vec<(u64, i64)>,
    buf: &mut Vec<u8>,
    mut writer: Option<&mut Writer>,
) -> Result<u32> {
    corr.rows_wait().context("waiting for the rows")?;
    let need = corr.fetch_bytes()?;
    if buf.len() < need {
        buf.resize(need, 0);
    }
    let rows = corr.rows_fetch(&mut buf[..need])? as usize;
    if rows == 0 {
        return Ok(0);
    }
    let row_bytes = corr.row_bytes()?;

    // The zero-fill figures are read once, after the rows are known to be done, so
    // every row in this batch is charged against the same snapshot. With
    // rows_per_fetch > 1 that spreads the count across the batch rather than
    // attributing it to one row; the alternative is a per-row async channel from
    // every receive thread, which is not worth it for a diagnostic flag.
    let now_zf: Vec<u64> = progress.iter().map(|p| p.zero_filled_frames()).collect();
    let per_row: Vec<u64> = if rows == 1 {
        now_zf
            .iter()
            .zip(zf_mark.iter())
            .map(|(a, b)| a.saturating_sub(*b))
            .collect()
    } else {
        // Spread evenly: without per-row counts the honest thing is to divide.
        now_zf
            .iter()
            .zip(zf_mark.iter())
            .map(|(a, b)| a.saturating_sub(*b) / rows as u64)
            .collect()
    };

    for i in 0..rows {
        let (row_index, open_ms) = pending_meta
            .get(i)
            .copied()
            .unwrap_or((0, crate::sched::system_ms()));
        let spectrum = &buf[i * row_bytes..(i + 1) * row_bytes];
        if let Some(w) = writer.as_deref_mut() {
            let valid: Vec<u64> = per_row
                .iter()
                .map(|z| (plan.n_cum as u64).saturating_sub(*z))
                .collect();
            let block = RowBlock {
                row_index,
                first_frame_id: row_index * plan.n_cum as u64,
                n_cum: plan.n_cum,
                valid_frames_per_station: valid,
            };
            w.write_row(spectrum, open_ms, &block)
                .with_context(|| format!("writing row {row_index}"))?;
        }
    }
    for i in 0..rows {
        for (m, n) in zf_mark.iter_mut().zip(per_row.iter()) {
            *m = m.saturating_add(*n);
        }
        let _ = i;
    }
    pending_meta.drain(0..rows.min(pending_meta.len()));
    Ok(rows as u32)
}






/// Progress counters, exposed so a caller can watch a run without the report.
pub fn progress_of(p: &[Arc<StationProgress>]) -> Vec<u64> {
    p.iter().map(|x| x.get()).collect()
}

/// Used by tests to publish a progress value directly.
pub fn set_progress(p: &AtomicU64, v: u64) {
    p.store(v, Ordering::Release);
}

#[cfg(test)]
mod send_probe {
    use super::*;
    // Only the tests need this: the receive path itself no longer names the
    // assembler, which is the point of `receiver.rs`.
    use crate::assemble::StationAssembler;
    fn assert_send<T: Send>() {}
    #[test]
    fn the_receive_thread_payload_types_are_send() {
        assert_send::<MulticastSocket>();
        assert_send::<StagingPtr>();
        assert_send::<StationAssembler>();
    }
}
