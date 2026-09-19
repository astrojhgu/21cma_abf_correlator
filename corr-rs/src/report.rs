//! The run report: what a finished run says about itself.
//!
//! Split out of `run.rs`, which had grown to hold the receive path, the scheduler
//! loop, the GPU hand-off and this. The report is pure data plus formatting: it
//! depends on nothing in `run.rs`, so it can be read and changed on its own -- and
//! it is the part a maintainer is most likely to touch (every field added to a
//! report is a question asked by someone reading a run).
//!
//! Everything here is constructed by `run.rs` and printed by `main.rs`; nothing in
//! this module runs threads or touches hardware.

use std::time::Duration;

use crate::assemble::AssembleStats;
use crate::config::RunPlan;

/// Everything a finished run reports: what it did, what it moved, what was missing.
///
/// Constructed by `run.rs` at the end of the run, printed by `main.rs`, and used by
/// the tests to assert on a run without a terminal.
#[derive(Debug, Default, Clone)]
pub struct RunReport {
    pub start_frame: u64,
    pub rows_closed: u64,
    pub slots_submitted: u64,
    pub frames_scheduled: u64,
    pub stall_recoveries: u64,
    /// Frames the scheduler asked stations to zero fill, and the configured cap.
    pub zero_fill_requested_frames: u64,
    pub zero_fill_limit_frames: u64,
    /// Recoveries and requested frames per station, in station order.
    pub recoveries_per_station: Vec<u64>,
    pub requested_per_station: Vec<u64>,
    /// How long a station may deliver no packets before it counts as stopped.
    pub stall_timeout: Duration,
    /// Set when the run stopped because a station exceeded the zero-fill cap.
    pub stopped_zero_fill_exceeded: Option<String>,
    /// Set when the run stopped because the watermark stayed lifted for too long.
    pub stopped_lift_exceeded: Option<String>,
    /// Set when a receive thread hit an error the whole run had to answer -- see
    /// [`crate::assemble::AssembleError::is_fatal`]. The output is still complete
    /// rows; it just stops early and says why.
    pub stopped_fatal: Option<String>,
    /// Rows handed to the writer, and where they went. `None` when not writing.
    pub rows_written: u64,
    pub output_dir: Option<std::path::PathBuf>,
    pub resumed_from_row: u64,
    /// True when the run actually accumulated on the GPU.
    pub used_gpu: bool,
    pub gpu_banner: Option<String>,
    pub elapsed: Duration,
    /// Per station, indexed by station index.
    pub stations: Vec<StationAssembleReport>,
    /// Per device: what it was given and how much it actually moved. Empty in a
    /// dry run, which has no device at all.
    pub devices: Vec<DeviceReport>,
    /// How often a receive thread had to stop and wait for a staging slot, and
    /// how many were still waiting when the run ended. This -- not the spin
    /// count -- is what says whether the staging ring throttled the link.
    pub ring_blocks: u64,
    pub ring_blocked: i64,
    /// Per second: (frames each station finalised, packets it lost, frames it had
    /// to zero fill). Sampled by the pipeline thread.
    pub timeline: Vec<(u64, u64, u64)>,
}

/// One device's share, as the library reported it at the end of the run.
#[derive(Debug, Clone)]
pub struct DeviceReport {
    pub device_id: i32,
    pub n_station_resident: i32,
    pub first_baseline: i32,
    pub n_baseline: i32,
    pub bytes_h2d: u64,
    pub transfers: u64,
    /// Where the pipeline thread's time went on this device's behalf.
    pub ns_accumulate: u64,
    pub ns_slot_free: u64,
    pub ns_rows_wait: u64,
    pub ns_fetch: u64,
}

/// One receive thread's outcome: what it saw, what it lost, where its time went.
#[derive(Debug, Default, Clone, Copy)]
pub struct StationAssembleReport {
    pub name_idx: usize,
    pub stats: AssembleStats,
    pub progress: u64,
    pub recv_batches: u64,
    pub recv_packets: u64,
    pub recv_empty_polls: u64,
    pub recv_truncated: u64,
    /// Time this station's thread spent waiting for packets, and time it spent
    /// turning them into staged frames. The split is the whole diagnosis when a
    /// run loses data: waiting means the link (or the sender) is behind, while
    /// assembling means this thread is the limit -- and those are different
    /// problems with different fixes.
    pub ns_recv: u64,
    pub ns_assemble: u64,
    /// Of `ns_assemble`, how much was spent waiting for the pipeline to release a
    /// staging slot rather than turning packets into frames.
    pub ns_slot_wait: u64,
    /// Longest this thread went without reading a batch, and how many times it
    /// went 20 ms or more. These size the socket-buffer + ring-lead budget: the
    /// budget has to cover this distribution, so it is measured rather than
    /// guessed at (HANDOFF 6.15.3b).
    pub max_stall_ms: u64,
    pub stalls_over_20ms: u64,
}

impl RunReport {
    /// One-line-per-station summary, shared by the CLI and the tests.
    pub fn render(&self, plan: &RunPlan) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "start frame {} | {:.2} s | {} slots | {} rows | {} frames",
            self.start_frame,
            self.elapsed.as_secs_f64(),
            self.slots_submitted,
            self.rows_closed,
            self.frames_scheduled
        );
        let _ = writeln!(
            s,
            "  stall: {} recoveries, {} frames zero filled against a limit of {} \
             (a station counts as stopped after {:?} with no packet at all)",
            self.stall_recoveries,
            self.zero_fill_requested_frames,
            self.zero_fill_limit_frames,
            self.stall_timeout
        );
        if let Some(why) = &self.stopped_zero_fill_exceeded {
            let _ = writeln!(s, "  STOPPED: {why}");
        }
        if let Some(why) = &self.stopped_lift_exceeded {
            let _ = writeln!(s, "  STOPPED: {why}");
        }
        if let Some(why) = &self.stopped_fatal {
            let _ = writeln!(s, "  STOPPED: {why}");
        }
        // How often the receive threads found the staging ring full and had to
        // wait for the pipeline. The waiting itself now blocks instead of
        // spinning -- measured at 24 stations that took the process from 21.5
        // cores to 6.1 -- so this number costs almost nothing and is purely a
        // statement about who is the bottleneck.
        let waits: u64 = self
            .stations
            .iter()
            .map(|st| st.stats.slot_wait_spins)
            .sum();
        // `slot_wait_spins` counts *spin iterations*, so at 24 stations it runs
        // to billions even when nothing is ever blocked: it says how long the
        // threads spent looking at a full ring, not whether the ring was the
        // limit. The blocking counter is the one that answers that, so it is
        // reported instead, and the spin count is kept only as a hint.
        let _ = writeln!(
            s,
            "  pipeline: {} ring block episode(s), {} thread(s) still blocked; \
             {waits} spin iterations (a large spin count alone means nothing)",
            self.ring_blocks, self.ring_blocked
        );
        match &self.output_dir {
            Some(d) => {
                let _ = writeln!(
                    s,
                    "  output: {} rows in {} (resumed from row {})",
                    self.rows_written,
                    d.display(),
                    self.resumed_from_row
                );
            }
            None if self.used_gpu => {
                let _ = writeln!(s, "  output: not writing (no output_dir)");
            }
            None => {}
        }
        if let Some(b) = &self.gpu_banner {
            let _ = writeln!(s, "  gpu: {b}");
        }
        // The achieved host-link rate per card. This is the number that says
        // whether the split fits: with every card holding every station, a card
        // pulling 19.2 GB/s at 24 stations is at 87% of the best rate ever
        // measured on one card here, and the aggregate is what 40 stations has
        // to fit into.
        if !self.devices.is_empty() {
            let secs = self.elapsed.as_secs_f64().max(1e-9);
            let total: u64 = self.devices.iter().map(|d| d.bytes_h2d).sum();
            let _ = writeln!(
                s,
                "  devices: {} | host->device {:.1} GB total = {:.1} GB/s aggregate",
                self.devices.len(),
                total as f64 / 1e9,
                total as f64 / 1e9 / secs
            );
            for d in &self.devices {
                let _ = writeln!(
                    s,
                    "    dev {}  stations={:<3} baselines={:<4} h2d={:.2} GB = {:.2} GB/s  \
                     transfers={}",
                    d.device_id,
                    d.n_station_resident,
                    d.n_baseline,
                    d.bytes_h2d as f64 / 1e9,
                    d.bytes_h2d as f64 / 1e9 / secs,
                    d.transfers
                );
            }
            // Where the pipeline thread's time goes. `busy` is what it spent
            // inside the library calls (queueing work and waiting for results);
            // the rest of the run it was in the scheduler, the writer, or idle.
            let busy: u64 = self
                .devices
                .iter()
                .map(|d| d.ns_accumulate + d.ns_slot_free + d.ns_rows_wait + d.ns_fetch)
                .sum();
            let total_busy_ms = busy as f64 / 1e6;
            let _ = writeln!(
                s,
                "  pipeline thread: {:.1} s inside library calls ({:.0}% of the run): \
                 queue {:.1} s, release-wait {:.1} s, rows-wait {:.1} s, fetch {:.1} s",
                total_busy_ms / 1e3,
                total_busy_ms / 1e3 / secs * 100.0,
                self.devices.iter().map(|d| d.ns_accumulate).sum::<u64>() as f64 / 1e6 / 1e3,
                self.devices.iter().map(|d| d.ns_slot_free).sum::<u64>() as f64 / 1e6 / 1e3,
                self.devices.iter().map(|d| d.ns_rows_wait).sum::<u64>() as f64 / 1e6 / 1e3,
                self.devices.iter().map(|d| d.ns_fetch).sum::<u64>() as f64 / 1e6 / 1e3,
            );
        }
        for st in &self.stations {
            let name = plan
                .enabled
                .get(st.name_idx)
                .map(|e| e.name.as_str())
                .unwrap_or("?");
            let _ = writeln!(
                s,
                "  {:<5} pkts={:<10} gaps={:<6} lost={:<8} zero_filled={:<6} \
                 late={:<6} before_start={:<8} recoveries={:<5} progress={}",
                name,
                st.stats.packets,
                st.stats.gaps,
                st.stats.lost_packets,
                st.stats.zero_filled_frames,
                st.stats.late_packets,
                st.stats.before_start_packets,
                self.recoveries_per_station
                    .get(st.name_idx)
                    .copied()
                    .unwrap_or(0),
                st.progress
            );
        }
        // Where each receive thread's time actually went. A station that spends
        // its life waiting is limited by the link; one that spends it assembling
        // is limited by this thread. Both look like "packets missing" in the
        // counters above, and they need opposite fixes, so the split is printed.
        let secs = self.elapsed.as_secs_f64().max(1e-9);
        let recv_s: f64 = self.stations.iter().map(|x| x.ns_recv as f64).sum::<f64>() / 1e9;
        let asm_s: f64 = self.stations.iter().map(|x| x.ns_assemble as f64).sum::<f64>() / 1e9;
        let n = self.stations.len().max(1) as f64;
        let wait_s: f64 = self.stations.iter().map(|x| x.ns_slot_wait as f64).sum::<f64>() / 1e9;
        let total_lost: u64 = self.stations.iter().map(|x| x.stats.lost_packets).sum();
        let total_zf: u64 = self
            .stations
            .iter()
            .map(|x| x.stats.zero_filled_frames)
            .sum();
        if !self.timeline.is_empty() && (total_lost > 0 || total_zf > 0) {
            let _ = writeln!(s, "  per second (per station): frames lost_then_zero_filled");
            let mut line = String::from("   ");
            for (sec, (p, l, z)) in self.timeline.iter().enumerate() {
                line.push_str(&format!(" {sec}: {p}/{l}/{z} |"));
            }
            let _ = writeln!(s, "{line}");
        }
        let worst_stall = self.stations.iter().map(|x| x.max_stall_ms).max().unwrap_or(0);
        let stalls: u64 = self.stations.iter().map(|x| x.stalls_over_20ms).sum();
        let _ = writeln!(
            s,
            "  receive threads: worst read stall {} ms, {} stalls >= 20 ms across all stations",
            worst_stall,
            stalls
        );
        let _ = writeln!(
            s,
            "  receive threads: {:.1} s/station waiting for packets, {:.1} s/station \
             assembling them (of which {:.1} s waiting for a staging slot), \
             {:.0}% of the run spent assembling",
            recv_s / n,
            asm_s / n,
            wait_s / n,
            100.0 * asm_s / (n * secs)
        );
        s
    }

    pub fn ok(&self) -> bool {
        self.stations.iter().all(|s| s.recv_truncated == 0)
    }
}
