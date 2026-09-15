//! The receive side: sockets, the priming sniff, and one thread per station.
//!
//! Split out of `run.rs`, which held the receive path, the scheduler loop and the
//! GPU hand-off in one 437-line function. This module owns everything that touches
//! a socket or a receive thread; nothing here knows about rows, slots or devices,
//! which is what makes it readable on its own.
//!
//! The rule that is easy to break from here: a receive thread must publish a
//! station's progress *before* it waits for anything (see `assemble.rs` and
//! `docs/ARCHITECTURE.md` section 3). The deadlock that rule prevents cost a day.

use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::{bail, Context, Result};
use crate::assemble::{AssembleStats, SlotRing, StationAssembler};
use crate::config::{Config, RunPlan};
use crate::net::{pin_to_cpu, MulticastSocket, PACKET_BYTES};
use crate::run::StagingPtr;
use crate::sched::StationProgress;
use crate::stop::{RunStop, StopReason};

/// Open one socket per enabled station, joining each group on the configured
/// interface.
/// How long to keep draining after every station has been seen, before the
/// counters the start frame is derived from are taken.
///
/// The alternative -- the counter of the very first packet each station sent --
/// can be seconds apart between stations whose streams begin at different times,
/// and a station handed a start frame it is already seconds past has nothing left
/// to write into the ring. A few milliseconds of extra draining is free: it is
/// data the run discards anyway, because the start frame is on the next row
/// boundary.
const SNIFF_SETTLE: Duration = Duration::from_millis(5);

pub fn open_sockets(cfg: &Config, plan: &RunPlan) -> Result<Vec<MulticastSocket>> {
    let mut out = Vec::with_capacity(plan.n_station);
    for s in &plan.enabled {
        let sock = MulticastSocket::open(&cfg.receiver, s.group_ip, s.dest_port).with_context(
            || {
                format!(
                    "joining {}:{} on {} ({})",
                    s.group_ip, s.dest_port, cfg.receiver.interface_ip, cfg.receiver.interface
                )
            },
        )?;
        out.push(sock);
    }
    Ok(out)
}

/// Drain until the first packet of every station has been seen, then keep
/// draining for [`SNIFF_SETTLE`], and return **the latest counter seen** per
/// station.
///
/// Stations are polled round-robin rather than one at a time: a station that is
/// already flowing would otherwise pile up in its socket buffer while the others
/// are waited for, and with a large `SO_RCVBUF` that is harmless but pointless.
pub fn sniff_first_counters(
    sockets: &mut [MulticastSocket],
    timeout: Duration,
) -> Result<Vec<u64>> {
    let n = sockets.len();
    let mut latest: Vec<Option<u64>> = vec![None; n];
    let mut remaining = n;
    let deadline = Instant::now() + timeout;
    let mut settle_until: Option<Instant> = None;

    loop {
        let settling = settle_until.map(|t| Instant::now() < t).unwrap_or(true);
        if remaining == 0 && !settling {
            break;
        }
        if Instant::now() >= deadline {
            let missing: Vec<usize> = (0..n).filter(|i| latest[*i].is_none()).collect();
            bail!(
                "no packets from station index {missing:?} within {:?}; either the front \
                 end is not transmitting on that group, or the switch is not forwarding \
                 it to this host",
                timeout
            );
        }
        let mut progressed = false;
        for i in 0..n {
            let got = sockets[i].recv_batch()?;
            if got > 0 {
                // The last packet of the batch is the station's latest counter;
                // see the note above on why the first one is not good enough.
                if let Some((c, _)) = sockets[i].packet(got - 1) {
                    // Count the station, not the batch. Decrementing on every batch
                    // let one busy station drive `remaining` to zero on its own, so
                    // the loop stopped early and the code below unwrapped a station
                    // that had never been heard from -- a panic during start-up,
                    // reachable whenever one station comes up late.
                    if latest[i].is_none() {
                        remaining = remaining.saturating_sub(1);
                    }
                    latest[i] = Some(c);
                    progressed = true;
                }
            }
        }
        if remaining == 0 && settle_until.is_none() {
            settle_until = Some(Instant::now() + SNIFF_SETTLE);
        }
        if !progressed {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    latest
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            c.ok_or_else(|| {
                anyhow::anyhow!(
                    "station index {i} sent nothing during priming; either it is not \
                     transmitting yet or its group is not reaching this host"
                )
            })
        })
        .collect()
}

/// Statistics a receive thread publishes when it exits.
pub(crate) struct RecvOutcome {
    pub(crate) stats: AssembleStats,
    pub(crate) progress: u64,
    pub(crate) batches: u64,
    pub(crate) packets: u64,
    pub(crate) empty_polls: u64,
    pub(crate) truncated: u64,
    pub(crate) ns_recv: u64,
    pub(crate) ns_assemble: u64,
    /// Longest this thread went without reading a batch, and how many times it
    /// went 20 ms or more. Sizes the socket-buffer + ring-lead budget.
    pub(crate) max_stall_ms: u64,
    pub(crate) stalls_over_20ms: u64,
}

/// Everything one receive thread needs. A named struct rather than a closure so
/// that a non-`Send` field is reported by name instead of as an anonymous
/// `*mut u8` somewhere inside a closure.
pub(crate) struct RecvThreadArgs {
    pub(crate) sock: MulticastSocket,
    pub(crate) base: StagingPtr,
    pub(crate) ring: Arc<SlotRing>,
    pub(crate) prog: Arc<StationProgress>,
    pub(crate) stop: Arc<RunStop>,
    /// Cap on the frames one station may fabricate as silence.
    pub(crate) zero_fill_limit_frames: u64,
    pub(crate) idx: usize,
    pub(crate) cpu: u32,
    pub(crate) packets_per_frame: u64,
    pub(crate) frames_per_slot: u64,
    pub(crate) frame_stride: usize,
    pub(crate) start_frame: u64,
}

/// One station's receive loop: poll the socket, assemble frames into the staging
/// ring, and honour the scheduler's stall-recovery requests between polls.
pub(crate) fn recv_thread(a: RecvThreadArgs) -> RecvOutcome {
    let mut sock = a.sock;
    if let Err(e) = pin_to_cpu(a.cpu) {
        // Not fatal: an unpinned thread still works, it is just less predictable.
        eprintln!("corr: station {}: cannot pin to CPU {}: {e}", a.idx, a.cpu);
    }

    let mut asm = StationAssembler::new(
        a.packets_per_frame,
        a.frames_per_slot,
        a.frame_stride,
        a.frame_stride * a.frames_per_slot as usize,
        a.ring,
        a.base.0,
        a.start_frame,
    );
    // Waiting for a staging slot is being alive, not being silent: the front end
    // is still sending and the packets are queueing in the socket.
    asm.set_liveness(a.prog.clone());
    // The same cap the scheduler's recovery path is bounded by: this station may
    // not fabricate unbounded silence, whether the scheduler asked for it or its
    // own counter jumped.
    asm.set_zero_fill_limit(a.zero_fill_limit_frames);

    let mut batches = 0u64;
    let mut packets = 0u64;
    let mut ns_recv = 0u64;
    let mut ns_assemble = 0u64;
    // How long this thread went without reading, and how often that was long
    // enough to matter. The socket buffer is the only thing standing between a
    // preempted thread and lost packets, so the budget the ring and the socket
    // have to cover *is* this distribution -- measuring it is how the next
    // parameter change stops being a guess (see HANDOFF 6.15.3b).
    let mut last_read = Instant::now();
    let mut max_stall_ms = 0u64;
    let mut stalls_over_20ms = 0u64;

    while !a.stop.is_stopped() {
        // Service the scheduler's stall recovery first: if this station has gone
        // quiet, its frames must be zero filled before the watermark can move
        // past them. It can fail -- the ring may be unwinding, or the station may
        // have fabricated more silence than the cap allows -- and both mean the
        // run is over, so this is reported like any other fatal assemble error.
        let want = a.prog.requested();
        if want > asm.progress() {
            match asm.finalize_upto(want) {
                Ok(()) => {
                    a.prog.clear_request(want);
                    a.prog.set(asm.progress());
                }
                Err(e) => {
                    if !matches!(e, crate::assemble::AssembleError::Aborted) {
                        eprintln!("corr: station {}: {e}", a.idx);
                    }
                    if !matches!(e, crate::assemble::AssembleError::Aborted) {
                        a.stop.stop(StopReason::Fatal {
                            station: a.idx,
                            message: e.to_string(),
                        });
                    }
                    a.stop.stop(StopReason::Fatal {
                        station: a.idx,
                        message: "assemble aborted".to_string(),
                    });
                    break;
                }
            }
        }

        let t_recv = Instant::now();
        let batch = sock.recv_batch();
        ns_recv += t_recv.elapsed().as_nanos() as u64;
        let t_assemble = Instant::now();
        match batch {
            Ok(0) => {}
            Ok(n) => {
                let stall_ms = last_read.elapsed().as_millis() as u64;
                last_read = Instant::now();
                if stall_ms > max_stall_ms {
                    max_stall_ms = stall_ms;
                }
                if stall_ms >= 20 {
                    stalls_over_20ms += 1;
                }
                batches += 1;
                a.prog.mark_packet();
                let mut fatal_here = false;
                for i in 0..n {
                    if let Some((counter, payload)) = sock.packet(i) {
                        match asm.on_packet(counter, payload) {
                            Ok(()) => packets += 1,
                            // Shutting down: stop parsing this batch and unwind.
                            Err(e) if e.is_fatal() => {
                                if !matches!(e, crate::assemble::AssembleError::Aborted) {
                                    eprintln!("corr: station {}: {e}", a.idx);
                                    a.stop.stop(StopReason::Fatal {
                                        station: a.idx,
                                        message: e.to_string(),
                                    });
                                }
                                fatal_here = true;
                                break;
                            }
                            Err(e) => eprintln!("corr: station {}: {e}", a.idx),
                        }
                    }
                }
                a.prog.set(asm.progress());
                a.prog.set_zero_filled_frames(asm.stats().zero_filled_frames);
                a.prog.set_lost_packets(asm.stats().lost_packets);
                if fatal_here {
                    break;
                }
            }
            Err(e) => {
                eprintln!("corr: station {}: receive error: {e}", a.idx);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        ns_assemble += t_assemble.elapsed().as_nanos() as u64;
    }

    // One last publish so the report reflects everything consumed.
    a.prog.set(asm.progress());
    let rs = sock.stats();
    RecvOutcome {
        stats: asm.stats(),
        progress: asm.progress(),
        batches,
        packets,
        empty_polls: rs.empty_polls,
        truncated: rs.truncated,
        ns_recv,
        ns_assemble,
        max_stall_ms,
        stalls_over_20ms,
    }
}

/// A `PACKET_BYTES` sanity check used by the replayer and by tests.
pub fn packet_bytes() -> usize {
    PACKET_BYTES
}

/// Convenience for the CLI: build the io error a missing interface would give.
pub fn require_interface(cfg: &Config) -> Result<()> {
    if cfg.receiver.interface_ip.is_unspecified() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "receiver.interface_ip is unset",
        ))
        .context("cannot join a multicast group without an interface address");
    }
    Ok(())
}
