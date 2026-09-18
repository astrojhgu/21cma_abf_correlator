//! One line of live state, every `monitoring.interval_ms`, on stderr.
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

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::assemble::SlotRing;
use crate::config::RunPlan;
use crate::gpu::Correlator;
use crate::sched::{RowScheduler, StationProgress};

/// At what fraction of a ceiling a resource is called exhausted.
const RING_BUSY_LIMIT: f64 = 0.20;
const LINK_LIMIT: f64 = 0.95;
const CPU_LIMIT: f64 = 0.85;

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

/// The live line printer. `interval_ms == 0` turns it off entirely.
pub(crate) struct Monitor {
    interval: Duration,
    /// Bytes per second the streams carry: what has to reach the devices, and
    /// therefore the demand the GPU link is measured against.
    required_bytes_per_s: f64,
    /// This run's local ports, so `/proc/net/udp` can be filtered to our sockets.
    ports: Vec<u16>,
    frames_per_slot: u64,
    n_station: usize,
    n_cum: u64,
    prev: Option<Snap>,
}

impl Monitor {
    pub(crate) fn new(interval_ms: u64, plan: &RunPlan) -> Option<Self> {
        if interval_ms == 0 {
            return None;
        }
        Some(Monitor {
            interval: Duration::from_millis(interval_ms),
            required_bytes_per_s: plan.frame_rate_hz * plan.fft_len as f64 * plan.n_station as f64,
            ports: plan.enabled.iter().map(|s| s.dest_port).collect(),
            frames_per_slot: plan.frames_per_slot as u64,
            n_station: plan.n_station,
            n_cum: plan.n_cum as u64,
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

    /// Print a line if the interval has elapsed. One `Instant::now()` when it has
    /// not.
    pub(crate) fn tick(
        &mut self,
        sched: &RowScheduler,
        progress: &[Arc<StationProgress>],
        ring: &SlotRing,
        gpu: Option<&Correlator>,
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
        let Some(prev) = self.prev.replace(cur) else {
            // Nothing to subtract yet. Say what the baseline is instead of a row
            // of zeroes that could be mistaken for a measurement.
            eprintln!(
                "mon        : baseline: {} rows closed, {:.4}% zero filled, {} lost, \
                 {} stall recoveries",
                cur.rows,
                100.0 * cur.zero_filled as f64 / cur.frames.max(1) as f64,
                cur.lost,
                cur.stalls
            );
            return;
        };

        let dt = cur.at.duration_since(prev.at).as_secs_f64().max(1e-6);
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
        let wait_ratio =
            (cur.slot_wait_ns.saturating_sub(prev.slot_wait_ns) as f64 / 1e9) / (dt * self.n_station as f64);
        // Pipeline: how much of the interval it spent waiting for the device to
        // hand a slot back. One pipeline thread, so this is a fraction of a core.
        let slot_free_ratio =
            (cur.slot_free_ns.saturating_sub(prev.slot_free_ns) as f64 / 1e9) / dt;
        // Link: bytes moved against the bytes the streams require.
        let h2d_gbs = cur.h2d_bytes.saturating_sub(prev.h2d_bytes) as f64 / dt / 1e9;
        let link_ratio = if self.required_bytes_per_s > 0.0 && gpu.is_some() {
            h2d_gbs * 1e9 / self.required_bytes_per_s
        } else {
            0.0
        };
        // Host: our own cores, and the machine's load against its cores.
        let cpu_cores = cur.cpu_ticks.saturating_sub(prev.cpu_ticks) as f64 / ticks_per_second() / dt;
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
        let blocked_now = ring.blocked_now();

        let verdict = if d_drops > 0 {
            format!("收包侧已溢出（socket 丢 {d_drops}）")
        } else if wait_ratio > RING_BUSY_LIMIT {
            format!("接收侧被环顶住（等环 {:.0}%）", wait_ratio * 100.0)
        } else if link_ratio >= LINK_LIMIT {
            format!("GPU 链路贴顶（H2D {:.0}%）", link_ratio * 100.0)
        } else if cores > 0.0 && load >= cores {
            format!("整机满（load {load:.1} / {cores:.0} 核）")
        } else if cores > 0.0 && cpu_cores / cores >= CPU_LIMIT {
            format!("本进程贴满 CPU（{:.1}/{cores:.0} 核）", cpu_cores)
        } else {
            "有余量".to_string()
        };

        eprintln!(
            "mon {rows}r {rate:+.2}r/s | zf {zf_total:.4}% (win {zf_window:.4}%) | lost +{d_lost} \
             | recv wait {wait_r:.0}% spread {spread:.1}sl blocked {blocked} blocks +{d_blocks} \
             stalled +{d_stalls} | cpu {cpu:.1}/{cores:.0}c load {load:.1} | h2d {h2d:.1}GB/s \
             {link:.0}% slotfree {sfree:.0}% | sockdrop +{drops} | capacity: {verdict}",
            rows = cur.rows,
            rate = d_rows as f64 / dt,
            zf_total = zf_total,
            zf_window = zf_window,
            d_lost = d_lost,
            wait_r = wait_ratio * 100.0,
            spread = spread_slots,
            blocked = blocked_now,
            d_blocks = d_blocks,
            d_stalls = d_stalls,
            cpu = cpu_cores,
            cores = cores,
            load = load,
            h2d = h2d_gbs,
            link = link_ratio * 100.0,
            sfree = slot_free_ratio * 100.0,
            drops = d_drops,
            verdict = verdict,
        );
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
