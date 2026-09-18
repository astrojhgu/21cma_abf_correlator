//! One sample of live state: a line on stderr every `monitoring.interval_ms`,
//! and the same sample published to whoever listens.
//!
//! Two questions, and nothing else:
//!
//! 1. **Is the run healthy right now?** Rows closed and the rate they close at,
//!    the per-row zero fill (cumulative *and* over the last interval, because a
//!    steady trickle and a burst look the same in a running total), the packets
//!    the streams never delivered, and the receive side's own view -- how long it
//!    spent waiting for a staging slot, how far apart the stations are, and
//!    whether any of our sockets has dropped anything in the kernel.
//!
//! 2. **Is the machine's capacity being used up, and by what?** A run that keeps
//!    up because it has headroom is a different animal from one that keeps up
//!    because nothing else is running. This prints the things that can be
//!    exhausted, each against its own ceiling:
//!
//!      * **the GPU link** -- host-to-device bytes moved against the bytes the
//!        streams require (`n_station * fft_len * frame_rate`), which is what this
//!        profile is sized to need;
//!      * **the receive side** -- the fraction of station-seconds spent waiting
//!        for the pipeline to release a slot. Waiting is not loss, but a run that
//!        waits a lot has no jitter margin left;
//!      * **the host** -- the process's own cores against the cores online, and
//!        the machine load against them.
//!
//!    `capacity:` then names the first resource that is at its ceiling, so the
//!    answer to "what is the wall" is on the same line as the numbers.
//!
//! Everything is a delta against the previous line: one line says what changed,
//! a series says which way it is going, and the run report still holds the
//! totals. The cost is one `/proc/self/stat`, one `/proc/loadavg` and one
//! `/proc/net/udp` per interval at the 1 s default -- nothing next to a 19 GB/s
//! copy stream.
//!
//! | field | source |
//! |---|---|
//! | rows, lost, stalls | `RowScheduler` |
//! | zero fill, slot wait | `StationProgress`, summed over stations |
//! | spread | `highest_progress - watermark`, in staging slots |
//! | h2d, slot-free wait | `Correlator::device_stats`, summed over devices |
//! | cpu, load | `/proc/self/stat`, `/proc/loadavg` |
//! | socket drops | `/proc/net/udp`, the rows for this run's ports |
//!
//! The same sample is also published, as a [`Snapshot`], into the [`Cell`] this
//! monitor was built with: today that is the HTTP endpoint in `status.rs`, and
//! when `monitoring.http.enabled` is off (the default) nobody holds one. That is
//! the whole coupling between the sampler and the status interface -- this module
//! owns *what the numbers are*, `status.rs` owns *how they are served*, and the
//! type in between is plain data with no opinion about sockets.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::assemble::SlotRing;
use crate::config::RunPlan;
use crate::gpu::Correlator;
use crate::sched::{now_ns, system_ms, RowScheduler, StationProgress};

/// At what fraction of a ceiling a resource is called exhausted.
const RING_BUSY_LIMIT: f64 = 0.20;
const LINK_LIMIT: f64 = 0.95;
const CPU_LIMIT: f64 = 0.85;

/// Schema version of [`Snapshot`]. A reader should refuse what it does not know:
/// adding a field is compatible, changing the meaning of one is not.
pub(crate) const SCHEMA: u32 = 1;

/// One published sample, as an outside reader sees it.
///
/// Every rate here is a delta against the previous sample, exactly like the
/// printed line, and `capacity` is the *same* verdict that line ends with: one
/// function decides it, so a monitor page and the log cannot drift apart.
///
/// `docs/DESIGN-status-http.md` has the HTTP contract and the field table.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Snapshot {
    pub schema: u32,
    /// False until the first sample lands, so a reader that connects during
    /// startup is told that rather than given an error.
    pub ready: bool,
    pub ts_unix_ms: i64,
    pub pid: u32,
    /// When this process started the run, for uptime.
    pub started_unix_ms: i64,
    pub interval_ms: u64,
    pub run: RunInfo,
    pub cumulative: Totals,
    pub window: Window,
    pub recv: Recv,
    pub gpu: Gpu,
    pub host: Host,
    pub capacity: Capacity,
    pub stations: Vec<Station>,
}

/// What is being run, so a monitor can say *which* run it is watching.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RunInfo {
    pub name: String,
    pub output_dir: String,
    /// The set being written right now: `<file_prefix><stamp>`. Empty until the
    /// first row names one, and it changes at a local-midnight rollover.
    pub set_tag: String,
    pub n_station: usize,
    pub n_baseline: usize,
    pub n_freq: u32,
    /// Frames per integration row.
    pub n_cum: u32,
    pub integration_ms: f64,
    pub frames_per_slot: u32,
    pub block_slots: u32,
}

/// Since the run started.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Totals {
    pub rows: u64,
    pub frames: u64,
    pub zero_filled_frames: u64,
    pub zero_fill_pct: f64,
    /// Frames whose packets the streams never delivered, from the front end's
    /// counter: a gap in what arrived, before any zero filling.
    pub lost_packets: u64,
    pub stall_recoveries: u64,
    /// Kernel drops on this run's own sockets: non-zero means the receive side is
    /// behind the wire, whatever the rest of the numbers say.
    pub socket_drops: u64,
    pub ring_block_events: u64,
}

/// Over the last sampling interval. `dt_s == 0` means "no interval yet": the
/// very first sample has cumulative totals but no window.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Window {
    pub dt_s: f64,
    pub rows: u64,
    pub rows_per_s: f64,
    pub frames: u64,
    pub zero_filled_frames: u64,
    pub zero_fill_pct: f64,
    pub lost_packets: u64,
    pub socket_drops: u64,
    pub ring_block_events: u64,
    pub stall_recoveries: u64,
}

/// The receive side's own view.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Recv {
    /// Percent of station-seconds spent waiting for a staging slot.
    pub wait_pct: f64,
    /// Stations blocked on the ring at this instant.
    pub blocked_now: u64,
    /// `highest_progress - watermark`, in staging slots.
    pub spread_slots: f64,
}

/// The host-to-device link, which this profile's geometry is sized against.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Gpu {
    pub h2d_gbs: f64,
    /// `n_station * fft_len * frame_rate`: what the streams require.
    pub required_gbs: f64,
    pub link_pct: f64,
    /// Percent of the interval the pipeline spent waiting for a device slot back.
    pub slot_free_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Host {
    pub cpu_cores: f64,
    pub cores_online: u64,
    pub load1: f64,
}

/// The first resource at its ceiling, if any.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Capacity {
    /// Stable token, for a monitor to colour by: `ok`, `socket_drops`, `ring`,
    /// `gpu_link`, `machine_load` or `process_cpu`.
    pub bottleneck: String,
    /// The printed line's own words for it.
    pub detail: String,
    /// The ceilings the verdict was taken against, in percent.
    pub limits: Limits,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Limits {
    pub ring_busy: f64,
    pub link: f64,
    pub cpu: f64,
}

/// One station, so that "which input" is answerable without trusting the run
/// report's per-station totals.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Station {
    pub name: String,
    pub progress_frames: u64,
    pub zero_filled_frames: u64,
    pub lost_packets: u64,
    pub slot_wait_ms: f64,
    /// Seconds since a packet last arrived on this station's socket.
    pub silence_s: f64,
}

/// The one place a run's live state is handed to an outside reader.
///
/// A slot, not a channel: the reader polls, and only the latest sample is
/// interesting. Publishing swaps an `Arc`, so the sampler pays one uncontended
/// lock, and a reader cloning the `Arc` never holds the sampler while it
/// serialises.
pub(crate) struct Cell {
    slot: Mutex<Option<Arc<Snapshot>>>,
}

impl Cell {
    pub(crate) fn new() -> Arc<Cell> {
        Arc::new(Cell {
            slot: Mutex::new(None),
        })
    }

    /// Replace the published sample. Called from the pipeline thread.
    pub(crate) fn publish(&self, snap: Snapshot) {
        // A poisoned lock would mean a panic somewhere else; publishing is still
        // the right thing to do, and `panic = "abort"` in release makes this
        // unreachable in practice.
        let mut slot = match self.slot.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        *slot = Some(Arc::new(snap));
    }

    /// The latest sample, or `None` before the first one. Called from whichever
    /// thread is serving readers.
    ///
    /// Without the `status-http` feature there is no serving thread at all, so
    /// outside of tests this is dead code; the `allow` keeps a
    /// `--no-default-features` build warning-free without hiding a real one.
    #[cfg_attr(not(feature = "status-http"), allow(dead_code))]
    pub(crate) fn current(&self) -> Option<Arc<Snapshot>> {
        let slot = match self.slot.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        slot.clone()
    }
}

/// One sample. Consecutive samples are subtracted to get rates.
#[derive(Clone, Copy)]
struct Snap {
    at: Instant,
    rows: u64,
    zero_filled: u64,
    frames: u64,
    lost: u64,
    stalls: u64,
    slot_wait_ns: u64,
    ring_blocks: u64,
    socket_drops: u64,
    h2d_bytes: u64,
    slot_free_ns: u64,
    cpu_ticks: u64,
}

/// What is at its ceiling, and the line's words for it.
struct Verdict {
    bottleneck: &'static str,
    detail: String,
}

/// Everything one tick derived from two consecutive [`Snap`]s.
///
/// The printed line and the published snapshot are two renderings of this, which
/// is why the verdict is computed once, here, and not in either rendering.
struct Derived {
    dt: f64,
    cur: Snap,
    d_rows: u64,
    d_frames: u64,
    d_zf: u64,
    d_lost: u64,
    d_stalls: u64,
    d_blocks: u64,
    d_drops: u64,
    zf_total: f64,
    zf_window: f64,
    rate_rows: f64,
    wait_ratio: f64,
    slot_free_ratio: f64,
    h2d_gbs: f64,
    link_ratio: f64,
    cpu_cores: f64,
    cores: f64,
    load: f64,
    spread_slots: f64,
    blocked_now: u64,
    verdict: Verdict,
}

/// The live line printer. `interval_ms == 0` turns the *line* off; the endpoint
/// can still be on, and then samples are taken once a second and published
/// without being printed.
pub(crate) struct Monitor {
    interval: Duration,
    print: bool,
    /// Where samples go. `None` when nothing is listening.
    cell: Option<Arc<Cell>>,
    /// Bytes per second the streams carry: what has to reach the devices, and
    /// therefore the demand the GPU link is measured against.
    required_bytes_per_s: f64,
    /// This run's local ports, so `/proc/net/udp` can be filtered to our sockets.
    ports: Vec<u16>,
    frames_per_slot: u64,
    n_station: usize,
    n_cum: u64,

    // Run identity: not sampled, but a monitor needs it to say what it is
    // watching.
    run_name: String,
    output_dir: String,
    n_baseline: usize,
    n_freq: u32,
    integration_ms: f64,
    block_slots: u32,
    station_names: Vec<String>,
    started_unix_ms: i64,

    prev: Option<Snap>,
}

impl Monitor {
    /// `cell` is where samples are published; pass `None` for the line alone.
    pub(crate) fn new(
        interval_ms: u64,
        cell: Option<Arc<Cell>>,
        plan: &RunPlan,
    ) -> Option<Self> {
        // Neither printing nor publishing: nothing to do, and no `/proc` to read.
        if interval_ms == 0 && cell.is_none() {
            return None;
        }
        Some(Monitor {
            // With the line off and the endpoint on, samples are still taken once a
            // second: that is the granularity of a monitor page, and one sample a
            // second costs one `/proc` read.
            interval: Duration::from_millis(if interval_ms == 0 { 1000 } else { interval_ms }),
            print: interval_ms != 0,
            cell,
            required_bytes_per_s: plan.frame_rate_hz * plan.fft_len as f64 * plan.n_station as f64,
            ports: plan.enabled.iter().map(|s| s.dest_port).collect(),
            frames_per_slot: plan.frames_per_slot as u64,
            n_station: plan.n_station,
            n_cum: plan.n_cum as u64,
            run_name: plan.run_name.clone(),
            output_dir: plan.output_dir.display().to_string(),
            n_baseline: plan.n_baseline,
            n_freq: plan.n_freq,
            integration_ms: 1000.0 * plan.n_cum as f64 / plan.frame_rate_hz,
            block_slots: plan.block_slots,
            station_names: plan.enabled.iter().map(|s| s.name.clone()).collect(),
            started_unix_ms: system_ms(),
            prev: None,
        })
    }

    fn sample(
        &self,
        sched: &RowScheduler,
        progress: &[Arc<StationProgress>],
        ring: &SlotRing,
        gpu: Option<&Correlator>,
        at: Instant,
    ) -> Snap {
        let rows = sched.stats().rows_closed;
        let mut slot_wait_ns = 0;
        let mut zero_filled = 0;
        for p in progress {
            slot_wait_ns += p.slot_wait_ns();
            zero_filled += p.zero_filled_frames();
        }
        let (h2d_bytes, slot_free_ns) = match gpu {
            Some(corr) => {
                let mut bytes = 0;
                let mut slot_free = 0;
                if let Ok(place) = corr.placement() {
                    for i in 0..place.len() {
                        if let Ok(st) = corr.device_stats(i as i32) {
                            bytes += st.bytes_h2d;
                            slot_free += st.ns_slot_free;
                        }
                    }
                }
                (bytes, slot_free)
            }
            None => (0, 0),
        };
        Snap {
            at,
            rows,
            zero_filled,
            frames: rows * self.n_cum * self.n_station as u64,
            lost: (0..self.n_station).map(|i| sched.lost_of(i)).sum(),
            stalls: sched.stats().stall_recoveries,
            slot_wait_ns,
            ring_blocks: ring.block_events(),
            socket_drops: socket_drops(&self.ports),
            h2d_bytes,
            slot_free_ns,
            cpu_ticks: cpu_ticks(),
        }
    }

    /// Sample and publish if the interval has elapsed. One `Instant::now()` when
    /// it has not.
    pub(crate) fn tick(
        &mut self,
        sched: &RowScheduler,
        progress: &[Arc<StationProgress>],
        ring: &SlotRing,
        gpu: Option<&Correlator>,
        set_tag: Option<&str>,
        now: Instant,
    ) {
        let due = match self.prev {
            Some(p) => now.duration_since(p.at) >= self.interval,
            None => true,
        };
        if !due {
            return;
        }
        let cur = self.sample(sched, progress, ring, gpu, now);
        let previous = self.prev.replace(cur);
        // On the first tick there is nothing to subtract, so the sample is its own
        // baseline: every delta below comes out zero and `dt` is zero, which is
        // what a reader sees until the second sample lands.
        let prev = previous.unwrap_or(cur);
        let first = previous.is_none();
        let dt = if first {
            0.0
        } else {
            cur.at.duration_since(prev.at).as_secs_f64().max(1e-6)
        };

        let d_rows = cur.rows.saturating_sub(prev.rows);
        let d_frames = cur.frames.saturating_sub(prev.frames);
        let d_zf = cur.zero_filled.saturating_sub(prev.zero_filled);
        let d_lost = cur.lost.saturating_sub(prev.lost);
        let d_stalls = cur.stalls.saturating_sub(prev.stalls);
        let d_blocks = cur.ring_blocks.saturating_sub(prev.ring_blocks);
        let d_drops = cur.socket_drops.saturating_sub(prev.socket_drops);

        // Receive side: station-seconds spent waiting for a slot, as a fraction of
        // the station-seconds that just elapsed. 0 = never refused, 1 = every
        // station was waiting the whole time.
        let wait_ratio = if dt > 0.0 {
            (cur.slot_wait_ns.saturating_sub(prev.slot_wait_ns) as f64 / 1e9)
                / (dt * self.n_station as f64)
        } else {
            0.0
        };
        // Pipeline: how much of the interval it spent waiting for the device to
        // hand a slot back. One pipeline thread, so this is a fraction of a core.
        let slot_free_ratio = if dt > 0.0 {
            (cur.slot_free_ns.saturating_sub(prev.slot_free_ns) as f64 / 1e9) / dt
        } else {
            0.0
        };
        // Link: bytes moved against the bytes the streams require.
        let h2d_gbs = if dt > 0.0 {
            cur.h2d_bytes.saturating_sub(prev.h2d_bytes) as f64 / dt / 1e9
        } else {
            0.0
        };
        let link_ratio = if self.required_bytes_per_s > 0.0 && gpu.is_some() {
            h2d_gbs * 1e9 / self.required_bytes_per_s
        } else {
            0.0
        };
        // Host: our own cores, and the machine's load against its cores.
        let cpu_cores = if dt > 0.0 {
            cur.cpu_ticks.saturating_sub(prev.cpu_ticks) as f64 / ticks_per_second() / dt
        } else {
            0.0
        };
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0) as f64;
        let load = loadavg();

        let zf_total = 100.0 * cur.zero_filled as f64 / cur.frames.max(1) as f64;
        let zf_window = if d_frames > 0 {
            100.0 * d_zf as f64 / d_frames as f64
        } else {
            0.0
        };
        let spread_slots = sched
            .highest_progress()
            .saturating_sub(sched.watermark()) as f64
            / self.frames_per_slot.max(1) as f64;

        let d = Derived {
            dt,
            cur,
            d_rows,
            d_frames,
            d_zf,
            d_lost,
            d_stalls,
            d_blocks,
            d_drops,
            zf_total,
            zf_window,
            rate_rows: if dt > 0.0 { d_rows as f64 / dt } else { 0.0 },
            wait_ratio,
            slot_free_ratio,
            h2d_gbs,
            link_ratio,
            cpu_cores,
            cores,
            load,
            spread_slots,
            // The ring counts leases in and out, so its reading is signed; a
            // count published to a reader cannot be negative, so clamp.
            blocked_now: ring.blocked_now().max(0) as u64,
            verdict: verdict(d_drops, wait_ratio, link_ratio, cpu_cores, cores, load),
        };

        if self.print {
            if first {
                // Nothing to subtract yet. Say what the baseline is instead of a
                // row of zeroes that could be mistaken for a measurement.
                eprintln!(
                    "mon        : baseline: {} rows closed, {:.4}% zero filled, {} lost, \
                     {} stall recoveries",
                    d.cur.rows, d.zf_total, d.cur.lost, d.cur.stalls
                );
            } else {
                self.print_line(&d);
            }
        }
        if let Some(cell) = &self.cell {
            cell.publish(self.snapshot(&d, sched, progress, set_tag));
        }
    }

    fn print_line(&self, d: &Derived) {
        eprintln!(
            "mon {rows}r {rate:+.2}r/s | zf {zf_total:.4}% (win {zf_window:.4}%) | lost +{d_lost} \
             | recv wait {wait_r:.0}% spread {spread:.1}sl blocked {blocked} blocks +{d_blocks} \
             stalled +{d_stalls} | cpu {cpu:.1}/{cores:.0}c load {load:.1} | h2d {h2d:.1}GB/s \
             {link:.0}% slotfree {sfree:.0}% | sockdrop +{drops} | capacity: {verdict}",
            rows = d.cur.rows,
            rate = d.rate_rows,
            zf_total = d.zf_total,
            zf_window = d.zf_window,
            d_lost = d.d_lost,
            wait_r = d.wait_ratio * 100.0,
            spread = d.spread_slots,
            blocked = d.blocked_now,
            d_blocks = d.d_blocks,
            d_stalls = d.d_stalls,
            cpu = d.cpu_cores,
            cores = d.cores,
            load = d.load,
            h2d = d.h2d_gbs,
            link = d.link_ratio * 100.0,
            sfree = d.slot_free_ratio * 100.0,
            drops = d.d_drops,
            verdict = d.verdict.detail,
        );
    }

    /// The same sample as a value an outside reader can render. Everything here
    /// either came from the sample or was fixed when the monitor was built; the
    /// only live input is the set tag, because a rollover changes it mid-run.
    fn snapshot(
        &self,
        d: &Derived,
        sched: &RowScheduler,
        progress: &[Arc<StationProgress>],
        set_tag: Option<&str>,
    ) -> Snapshot {
        let at_ns = now_ns();
        let stations = (0..self.n_station)
            .map(|i| {
                let p = progress.get(i);
                Station {
                    name: self.station_names.get(i).cloned().unwrap_or_default(),
                    progress_frames: sched.progress_of(i),
                    zero_filled_frames: p.map(|p| p.zero_filled_frames()).unwrap_or(0),
                    lost_packets: sched.lost_of(i),
                    slot_wait_ms: p.map(|p| p.slot_wait_ns()).unwrap_or(0) as f64 / 1e6,
                    silence_s: p.map(|p| p.silence_ns(at_ns)).unwrap_or(0) as f64 / 1e9,
                }
            })
            .collect();
        Snapshot {
            schema: SCHEMA,
            ready: true,
            ts_unix_ms: system_ms(),
            pid: std::process::id(),
            started_unix_ms: self.started_unix_ms,
            interval_ms: self.interval.as_millis() as u64,
            run: RunInfo {
                name: self.run_name.clone(),
                output_dir: self.output_dir.clone(),
                set_tag: set_tag.unwrap_or_default().to_string(),
                n_station: self.n_station,
                n_baseline: self.n_baseline,
                n_freq: self.n_freq,
                n_cum: self.n_cum as u32,
                integration_ms: self.integration_ms,
                frames_per_slot: self.frames_per_slot as u32,
                block_slots: self.block_slots,
            },
            cumulative: Totals {
                rows: d.cur.rows,
                frames: d.cur.frames,
                zero_filled_frames: d.cur.zero_filled,
                zero_fill_pct: d.zf_total,
                lost_packets: d.cur.lost,
                stall_recoveries: d.cur.stalls,
                socket_drops: d.cur.socket_drops,
                ring_block_events: d.cur.ring_blocks,
            },
            window: Window {
                dt_s: d.dt,
                rows: d.d_rows,
                rows_per_s: d.rate_rows,
                frames: d.d_frames,
                zero_filled_frames: d.d_zf,
                zero_fill_pct: d.zf_window,
                lost_packets: d.d_lost,
                socket_drops: d.d_drops,
                ring_block_events: d.d_blocks,
                stall_recoveries: d.d_stalls,
            },
            recv: Recv {
                wait_pct: d.wait_ratio * 100.0,
                blocked_now: d.blocked_now,
                spread_slots: d.spread_slots,
            },
            gpu: Gpu {
                h2d_gbs: d.h2d_gbs,
                required_gbs: self.required_bytes_per_s / 1e9,
                link_pct: d.link_ratio * 100.0,
                slot_free_pct: d.slot_free_ratio * 100.0,
            },
            host: Host {
                cpu_cores: d.cpu_cores,
                cores_online: d.cores as u64,
                load1: d.load,
            },
            capacity: Capacity {
                bottleneck: d.verdict.bottleneck.to_string(),
                detail: d.verdict.detail.clone(),
                limits: Limits {
                    ring_busy: RING_BUSY_LIMIT * 100.0,
                    link: LINK_LIMIT * 100.0,
                    cpu: CPU_LIMIT * 100.0,
                },
            },
            stations,
        }
    }
}

/// The first resource that is at its ceiling, if any.
///
/// One function, two renderings: the printed line's tail and the snapshot's
/// `capacity` both come from here, so the log and a monitor page cannot disagree
/// about what the wall is.
fn verdict(
    d_drops: u64,
    wait_ratio: f64,
    link_ratio: f64,
    cpu_cores: f64,
    cores: f64,
    load: f64,
) -> Verdict {
    if d_drops > 0 {
        Verdict {
            bottleneck: "socket_drops",
            detail: format!("收包侧已溢出（socket 丢 {d_drops}）"),
        }
    } else if wait_ratio > RING_BUSY_LIMIT {
        Verdict {
            bottleneck: "ring",
            detail: format!("接收侧被环顶住（等环 {:.0}%）", wait_ratio * 100.0),
        }
    } else if link_ratio >= LINK_LIMIT {
        Verdict {
            bottleneck: "gpu_link",
            detail: format!("GPU 链路贴顶（H2D {:.0}%）", link_ratio * 100.0),
        }
    } else if cores > 0.0 && load >= cores {
        Verdict {
            bottleneck: "machine_load",
            detail: format!("整机满（load {load:.1} / {cores:.0} 核）"),
        }
    } else if cores > 0.0 && cpu_cores / cores >= CPU_LIMIT {
        Verdict {
            bottleneck: "process_cpu",
            detail: format!("本进程贴满 CPU（{cpu_cores:.1}/{cores:.0} 核）"),
        }
    } else {
        Verdict {
            bottleneck: "ok",
            detail: "有余量".to_string(),
        }
    }
}

/// utime + stime of this process, in clock ticks.
fn cpu_ticks() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else {
        return 0;
    };
    // Field 2 is the command in parentheses and may itself contain spaces, so
    // count from the last closing parenthesis.
    let Some(i) = s.rfind(')') else { return 0 };
    let f: Vec<&str> = s[i + 1..].split_whitespace().collect();
    // utime and stime are fields 14 and 15 of the line, i.e. 11 and 12 here.
    let u = f.get(11).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let st = f.get(12).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    u + st
}

/// Clock ticks per second, from the kernel rather than from an assumption.
fn ticks_per_second() -> f64 {
    static HZ: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *HZ.get_or_init(|| {
        // SAFETY: sysconf is a pure query.
        let v = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if v > 0 {
            v as f64
        } else {
            100.0
        }
    })
}

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(0.0)
}

/// Kernel drops on *our* receiving sockets: the `/proc/net/udp` rows whose local
/// port belongs to this run. A non-zero delta means the receive side is already
/// behind the wire, whatever the rest of the line says.
fn socket_drops(ports: &[u16]) -> u64 {
    if ports.is_empty() {
        return 0;
    }
    let Ok(text) = std::fs::read_to_string("/proc/net/udp") else {
        return 0;
    };
    let mut total = 0u64;
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 13 {
            continue;
        }
        let Some((_, port_hex)) = f[1].split_once(':') else {
            continue;
        };
        let Ok(port) = u16::from_str_radix(port_hex, 16) else {
            continue;
        };
        if ports.contains(&port) {
            if let Some(d) = f.last().and_then(|v| v.parse::<u64>().ok()) {
                total += d;
            }
        }
    }
    total
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The verdict is the one thing two renderings share, so it is the one thing
    /// worth pinning: a socket drop outranks everything, and each ceiling is
    /// reported by its own token.
    #[test]
    fn the_verdict_names_the_first_resource_at_its_ceiling() {
        let ok = verdict(0, 0.0, 0.0, 4.0, 32.0, 1.0);
        assert_eq!(ok.bottleneck, "ok");

        // A drop is the wall even when the ring is also busy: it is the one that
        // has already lost data.
        let dropped = verdict(3, 0.5, 1.5, 31.0, 32.0, 40.0);
        assert_eq!(dropped.bottleneck, "socket_drops");

        assert_eq!(verdict(0, 0.5, 0.0, 1.0, 32.0, 1.0).bottleneck, "ring");
        assert_eq!(verdict(0, 0.0, 0.96, 1.0, 32.0, 1.0).bottleneck, "gpu_link");
        assert_eq!(
            verdict(0, 0.0, 0.0, 1.0, 32.0, 33.0).bottleneck,
            "machine_load"
        );
        assert_eq!(
            verdict(0, 0.0, 0.0, 28.0, 32.0, 1.0).bottleneck,
            "process_cpu"
        );
    }

    /// An unpublished cell is a normal state, not an error: it is what a reader
    /// sees between the socket being opened and the first sample landing.
    #[test]
    fn an_empty_cell_has_no_sample_and_publishing_replaces_it() {
        let cell = Cell::new();
        assert!(cell.current().is_none());
        cell.publish(minimal_snapshot(1));
        assert_eq!(cell.current().expect("published").ts_unix_ms, 1);
        cell.publish(minimal_snapshot(2));
        assert_eq!(cell.current().expect("published").ts_unix_ms, 2);
    }

    /// A snapshot with every field present, so the test above is about the cell
    /// and not about constructing one.
    pub(crate) fn minimal_snapshot(ts: i64) -> Snapshot {
        Snapshot {
            schema: SCHEMA,
            ready: true,
            ts_unix_ms: ts,
            pid: 1,
            started_unix_ms: 0,
            interval_ms: 1000,
            run: RunInfo {
                name: "t".into(),
                output_dir: "/tmp".into(),
                set_tag: String::new(),
                n_station: 1,
                n_baseline: 1,
                n_freq: 8,
                n_cum: 10,
                integration_ms: 1000.0,
                frames_per_slot: 4,
                block_slots: 8,
            },
            cumulative: Totals {
                rows: 0,
                frames: 0,
                zero_filled_frames: 0,
                zero_fill_pct: 0.0,
                lost_packets: 0,
                stall_recoveries: 0,
                socket_drops: 0,
                ring_block_events: 0,
            },
            window: Window {
                dt_s: 0.0,
                rows: 0,
                rows_per_s: 0.0,
                frames: 0,
                zero_filled_frames: 0,
                zero_fill_pct: 0.0,
                lost_packets: 0,
                socket_drops: 0,
                ring_block_events: 0,
                stall_recoveries: 0,
            },
            recv: Recv {
                wait_pct: 0.0,
                blocked_now: 0,
                spread_slots: 0.0,
            },
            gpu: Gpu {
                h2d_gbs: 0.0,
                required_gbs: 0.0,
                link_pct: 0.0,
                slot_free_pct: 0.0,
            },
            host: Host {
                cpu_cores: 0.0,
                cores_online: 1,
                load1: 0.0,
            },
            capacity: Capacity {
                bottleneck: "ok".into(),
                detail: "有余量".into(),
                limits: Limits {
                    ring_busy: RING_BUSY_LIMIT * 100.0,
                    link: LINK_LIMIT * 100.0,
                    cpu: CPU_LIMIT * 100.0,
                },
            },
            stations: vec![],
        }
    }
}
