//! The async receive backend: a pool of worker threads, each running a
//! single-threaded runtime over its share of the stations' sockets.
//!
//! # Why this shape
//!
//! The thread-per-station receiver needs one core per stream, because a station's
//! thread is either reading its socket or waiting for a staging slot -- and while
//! it waits, that core is idle (or, with the condvar path, asleep with the socket
//! filling up behind it). That is fine while the number of streams fits in the
//! machine. It is not fine at 40 streams on 32 cores, which is this project's next
//! target: the threads then time-slice each other, every preemption happens with
//! packets in flight, and the per-thread state (a 256 KiB `recvmmsg` batch buffer,
//! an arrival bitmap per slot, a stack) is multiplied by a thread count that
//! exceeds the core count.
//!
//! Here the number of *threads* is chosen instead of inherited from the number of
//! streams. One worker owns several stations and multiplexes them with a single
//! epoll set, so 40 streams can be received by 8 workers on 8 cores, or by 32
//! workers on 32 cores, or by one worker on one core -- the same code, one flag.
//!
//! # What each worker is
//!
//! One OS thread, pinned to one CPU, running a `current_thread` tokio runtime: its
//! own epoll reactor, its own task queue, no work-stealing and therefore no task
//! migration between cores. Each station on it is one task whose state machine is
//! in `station.rs`. A blocking call in a task would stall the whole worker, so the
//! two places that block on the thread path -- waiting for a staging slot, and
//! waiting for a packet -- are both non-blocking here: the first yields and
//! retries, the second waits on the reactor.
//!
//! # Partitioning
//!
//! Stations are dealt round-robin to workers. Every station carries the same
//! traffic, so round-robin is balanced by construction, and it also spreads each
//! worker's stations over the station index range rather than clustering them --
//! which matters when the front end's boards are not equally fast.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{bail, Result};
use tokio::io::unix::AsyncFd;

use crate::backend::{RecvSetup, ReceiveBackend};
use crate::net::{pin_to_cpu, MulticastSocket};
use crate::receiver::RecvOutcome;
use crate::run::StagingPtr;

use super::station::{station_task, RunGeometry, StationInit};

/// One station as the backend holds it between `start` and the worker spawn.
struct Pending {
    idx: usize,
    sock: MulticastSocket,
    staging: StagingPtr,
    progress: Arc<crate::sched::StationProgress>,
}

/// The receive backend of `corr-async`.
pub struct AsyncBackend {
    /// How many worker threads to run. Capped at the station count; one worker
    /// may serve many stations, which is the entire point.
    workers: usize,
    /// Pin the workers to this CPU list (in order) instead of the plan's.
    cpus: Option<Vec<u32>>,
    handles: Vec<JoinHandle<Vec<(usize, RecvOutcome)>>>,
}

impl AsyncBackend {
    pub(crate) fn new(workers: usize, cpus: Option<Vec<u32>>) -> Self {
        AsyncBackend {
            workers: workers.max(1),
            cpus,
            handles: Vec::new(),
        }
    }

    /// The cores the workers will be pinned to, resolved the same way `start`
    /// resolves them, so the caller can print the layout before anything runs.
    pub(crate) fn resolve_cpus(&self, plan: &crate::RunPlan) -> Vec<u32> {
        if let Some(c) = &self.cpus {
            if !c.is_empty() {
                return c.clone();
            }
        }
        // The plan's receive cores, deduplicated in station order: for the
        // thread-per-station backend those are one per thread, and here they are
        // the cores the workers divide between them. Reading them from the plan
        // rather than from a flag of its own is what makes the two backends
        // comparable on one configuration.
        let mut out: Vec<u32> = Vec::new();
        for s in &plan.enabled {
            if !out.contains(&s.recv_cpu) {
                out.push(s.recv_cpu);
            }
        }
        out
    }
}

impl ReceiveBackend for AsyncBackend {
    fn name(&self) -> &'static str {
        "async"
    }

    fn start(&mut self, setup: RecvSetup<'_>) -> Result<()> {
        let RecvSetup {
            cfg,
            plan,
            sockets,
            staging,
            ring,
            progress,
            stop,
            start_frame,
            frame_stride,
            packets_per_frame,
            frames_per_slot,
            validity_offset,
            validity_bytes_per_slot,
        } = setup;

        let n = plan.n_station;
        if sockets.len() != n || staging.len() != n || progress.len() != n {
            bail!("sockets, staging and progress must cover every enabled station");
        }
        if ring.slots() < 2 {
            bail!("a staging ring of {} slots cannot overlap anything", ring.slots());
        }

        let workers = self.workers.min(n);
        let cpus = self.resolve_cpus(plan);
        if cpus.is_empty() {
            bail!("no receive core to pin the async workers to");
        }

        // The interval that bounds how long a silent station may ignore the
        // scheduler's request. Deliberately the same configuration field the
        // thread path feeds to SO_RCVTIMEO: it is the same decision, and a second
        // knob for it would be a second thing to get wrong.
        let poll_interval = Duration::from_millis(cfg.receiver.recv_timeout_ms.max(1));

        let geometry = Arc::new(RunGeometry {
            ring: ring.clone(),
            stop: stop.clone(),
            frames_per_slot,
            frame_stride,
            packets_per_frame,
            validity_offset,
            validity_bytes_per_slot,
            start_frame,
            zero_fill_limit_frames: cfg.integration.zero_fill_limit_frames,
            poll_interval,
        });

        // Deal the stations out, then hand each worker its own batch. The sockets
        // move into the worker threads: one socket, one task, one owner -- the
        // same rule the thread path has, for the same reason.
        let mut groups: Vec<Vec<Pending>> = (0..workers).map(|_| Vec::new()).collect();
        for (idx, sock) in sockets.into_iter().enumerate() {
            groups[idx % workers].push(Pending {
                idx,
                sock,
                staging: staging[idx],
                progress: progress[idx].clone(),
            });
        }

        println!(
            "async      : {workers} worker(s) on cores {:?} for {n} stream(s) ({:.2} stream(s) per core)",
            (0..workers).map(|w| cpus[w % cpus.len()]).collect::<Vec<_>>(),
            n as f64 / workers as f64
        );
        if workers > cpus.len() {
            println!(
                "  WARNING: {workers} workers share {} core(s); with more workers than cores the \
                 per-worker state is again multiplied by more threads than the machine can run at \
                 once, which is what this backend exists to avoid",
                cpus.len()
            );
        }
        if workers < cpus.len() {
            println!(
                "  note: {} of {} receive core(s) left idle; pass --workers {} to use them all",
                cpus.len() - workers,
                cpus.len(),
                cpus.len()
            );
        }

        for (w, group) in groups.into_iter().enumerate() {
            let cpu = cpus[w % cpus.len()];
            let geo = geometry.clone();
            let stations: Vec<usize> = group.iter().map(|p| p.idx).collect();
            let handle = std::thread::Builder::new()
                .name(format!("corr-async-rx{w}"))
                .spawn(move || worker_main(w, cpu, group, geo))
                .map_err(|e| anyhow::anyhow!("spawning receive worker {w}: {e}"))?;
            println!("  worker {w} on CPU {cpu}: station(s) {stations:?}");
            self.handles.push(handle);
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Vec<RecvOutcome> {
        let mut collected: Vec<(usize, RecvOutcome)> = Vec::new();
        let mut max_idx = 0usize;
        for h in self.handles {
            match h.join() {
                Ok(v) => {
                    for (idx, o) in v {
                        max_idx = max_idx.max(idx);
                        collected.push((idx, o));
                    }
                }
                Err(_) => eprintln!("corr-async: a receive worker panicked"),
            }
        }
        let mut out: Vec<Option<RecvOutcome>> = vec![None; max_idx + 1];
        for (idx, o) in collected {
            out[idx] = Some(o);
        }
        // A worker that panicked, or a station whose task was lost with it, has no
        // outcome to report: the default is zero packets and zero progress, which
        // is exactly what the thread path's `unwrap_or_default` shows and is
        // impossible to mistake for a healthy station in the report.
        out.into_iter().map(|o| o.unwrap_or_default()).collect()
    }
}

/// One worker thread: pin, build a runtime, spawn one task per station, wait.
///
/// The runtime is created inside the thread rather than shared: a
/// `current_thread` runtime belongs to the thread that drives it, and giving each
/// worker its own reactor is what keeps a task on one core for its whole life.
fn worker_main(
    w: usize,
    cpu: u32,
    group: Vec<Pending>,
    g: Arc<RunGeometry>,
) -> Vec<(usize, RecvOutcome)> {
    if let Err(e) = pin_to_cpu(cpu) {
        // Not fatal: an unpinned worker still receives, it is just less
        // predictable -- the same judgement the thread path makes.
        eprintln!("corr-async: worker {w}: cannot pin to CPU {cpu}: {e}");
    }
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("corr-async: worker {w}: cannot build a runtime: {e}");
            return Vec::new();
        }
    };

    rt.block_on(async move {
        let mut ready = Vec::with_capacity(group.len());
        for p in group {
            // The reactor needs a non-blocking descriptor: an empty socket must
            // say EAGAIN at once instead of parking the worker for SO_RCVTIMEO.
            if let Err(e) = p.sock.set_nonblocking() {
                eprintln!(
                    "corr-async: worker {w}: station {}: cannot set O_NONBLOCK: {e}",
                    p.idx
                );
                continue;
            }
            let fd = p.sock.fd();
            match AsyncFd::new(fd) {
                Ok(afd) => ready.push((
                    StationInit {
                        idx: p.idx,
                        sock: p.sock,
                        staging: p.staging,
                        progress: p.progress,
                    },
                    afd,
                )),
                Err(e) => eprintln!(
                    "corr-async: worker {w}: station {}: cannot register with the reactor: {e}",
                    p.idx
                ),
            }
        }

        let mut set = tokio::task::JoinSet::new();
        for (init, afd) in ready {
            set.spawn(station_task(init, g.clone(), afd));
        }
        let mut out = Vec::with_capacity(set.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(v) => out.push(v),
                Err(e) => eprintln!("corr-async: worker {w}: a station task failed: {e}"),
            }
        }
        out
    })
}
