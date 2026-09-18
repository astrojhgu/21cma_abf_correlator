//! `corr-async` -- the correlator with a readiness-driven receive path.
//!
//! This module is the whole of the second receive backend: the worker pool
//! (`backend`) and one task per station (`station`). It lives in this crate rather
//! than in one of its own because everything it is *not* -- configuration, the
//! staging ring, the frame assembler, the row scheduler, the GPU hand-off, the
//! writer, the report -- is here, and a separate crate had to re-export all of it
//! through a public seam plus a runpath workaround for `libcorr_cuda.so`. Behind
//! the `async-recv` feature, which is on by default: `cargo build --release`
//! builds `corr`, `corr-replay` and `corr-async`, and `--no-default-features`
//! builds the first two without pulling `tokio` in at all.
//!
//! Same configuration, same sockets, same frame assembler, same staging ring, same
//! row scheduler, same GPU ABI, same writer, same report. One thing is different:
//! how many threads read the sockets.
//!
//! The thread-per-station receiver needs one core per stream. When the array has
//! more streams than the machine has cores -- 40 streams on 32 cores is this
//! project's next target, and `docs/ARCHITECTURE.md` lists "40 receive threads >
//! 32 cores" as one of the three walls in the way -- the threads time-slice each
//! other, and every one of those switches is a switch taken with packets in
//! flight. This binary replaces "one thread per stream" with "one worker per core,
//! as many streams per worker as there are", multiplexed through one epoll set per
//! worker.
//!
//! `--workers` is the whole interface to that decision: give it the number of
//! cores the run may have and the streams divide themselves between them. Left
//! alone it is one worker per receive core the plan names, which is the closest
//! thing to the thread-per-station layout the same configuration produces.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;

use crate::config::Config;
use crate::run::{self, RunOptions};
use crate::stop::{RunStop, StopReason};
use crate::RunPlan;

pub(crate) mod backend;
pub(crate) mod station;

/// Set by the SIGINT/SIGTERM handler, read by the scheduler loop.
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_sig: libc::c_int) {
    // Only an atomic store, so it is safe in a signal handler.
    STOP.store(true, Ordering::Relaxed);
}

fn install_signal_handlers() {
    // SAFETY: `on_signal` only touches an atomic, which is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "corr-async",
    version,
    about = "Correlator with an async receive path: more streams than cores",
    long_about = "Receives the same multicast UDP streams as `corr`, through the same frame \
                  assembler, staging ring, scheduler, GPU ABI and writer, but with a pool of \
                  readiness-driven worker threads instead of one thread per station. The number of \
                  workers is chosen with --workers, so N streams can be received on W < N cores."
)]
struct Cli {
    /// Path to the run configuration.
    #[arg(short, long, default_value = "config/correlator.yaml", env = "CORR_CONFIG")]
    config: PathBuf,

    /// Print the resolved run plan (stations, baselines, derived rates) and exit.
    #[arg(long)]
    print_plan: bool,

    /// How long to run, in seconds. Default: until interrupted.
    #[arg(long)]
    duration: Option<f64>,

    /// Receive and schedule only: no GPU, no files.
    #[arg(long, env = "CORR_DRY_RUN")]
    dry_run: bool,

    /// Override run.output_dir.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// How many receive workers to run. Default: one per receive core in the plan.
    ///
    /// Each worker is one thread with its own epoll set and its own tasks; a worker
    /// holds as many stations as it takes to cover every stream. Fewer workers than
    /// streams is the case this binary exists for; more workers than cores is
    /// allowed and warned about, since it is the thread-per-station problem again.
    #[arg(long, env = "CORR_ASYNC_WORKERS")]
    workers: Option<usize>,

    /// Pin the workers to these CPUs instead of the plan's receive cores
    /// (e.g. `--worker-cpus 16-23`, the kernel's own cpulist syntax).
    #[arg(long, value_name = "CPULIST")]
    worker_cpus: Option<String>,

    /// Start a second file set for today even though one already exists.
    #[arg(long)]
    allow_second_set: bool,

    /// Append to this file set prefix instead of today's.
    #[arg(long, value_name = "PREFIX")]
    set: Option<String>,
}

/// Entry point for the `corr-async` binary.
///
/// Public, and the only public item here: `src/bin/corr_async.rs` is three lines
/// that call this. Everything else in the module -- the worker pool, the per
/// station task, the seam -- stays `pub(crate)`, so adding this backend did not
/// widen the crate's API.
pub fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("corr-async: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();

    let mut cfg = Config::from_yaml_file(&cli.config)?;
    if let Some(d) = cli.output_dir.clone() {
        cfg.run.output_dir = d;
    }
    cfg.resolve_run_name();
    let plan = cfg.compile().context("configuration does not compile")?;

    println!("corr-async {}", env!("CARGO_PKG_VERSION"));
    println!("config     : {}", cli.config.display());
    println!("plan       : {}", plan.summary());

    if cli.print_plan {
        println!("\nenabled stations, in baseline index order");
        print!("{plan}");
        return Ok(ExitCode::SUCCESS);
    }

    let set_policy = if let Some(p) = cli.set.clone() {
        crate::writer::SetPolicy::Explicit(p)
    } else if cli.allow_second_set {
        crate::writer::SetPolicy::NewSet
    } else {
        crate::writer::SetPolicy::Auto
    };
    execute(&cfg, &plan, &cli, set_policy)
}

fn execute(
    cfg: &Config,
    plan: &RunPlan,
    cli: &Cli,
    set_policy: crate::writer::SetPolicy,
) -> Result<ExitCode> {
    install_signal_handlers();
    run::require_interface(cfg)?;

    let dry_run = cli.dry_run || cfg.run.dry_run;
    let output_dir = if dry_run {
        None
    } else {
        let dir = cfg.run.output_dir.join(&plan.run_name);
        println!("output     : {}", dir.display());
        Some(dir)
    };

    let opts = RunOptions {
        duration: cli.duration.map(Duration::from_secs_f64),
        wait_for_first_packet: Duration::from_secs_f64(
            cli.duration.unwrap_or(10.0).max(1.0) + 5.0,
        ),
        output_dir,
        enable_gpu: !dry_run,
        set_policy,
    };

    // Everything slow happens before a single group is joined, for the reason
    // `run::prepare_output` documents: with the sockets joined and nothing reading
    // them, 512 MiB of receive buffer fills in about 0.6 s at the real packet rate.
    let prepared = run::prepare_output(cfg, plan, &opts)?;

    let worker_cpus = match &cli.worker_cpus {
        Some(s) => {
            let v = crate::config::parse_cpulist(s);
            if v.is_empty() {
                bail!("--worker-cpus {s:?} names no CPU");
            }
            Some(v)
        }
        None => None,
    };
    let workers = cli.workers.unwrap_or_else(|| {
        backend::AsyncBackend::new(1, worker_cpus.clone())
            .resolve_cpus(plan)
            .len()
            .max(1)
    });
    let backend = backend::AsyncBackend::new(workers, worker_cpus);
    let cpus = backend.resolve_cpus(plan);
    println!("receiver   : async, {workers} worker(s) over cores {cpus:?}");

    println!("opening {} multicast sockets", plan.n_station);
    for note in &plan.cpu_note {
        println!("cpu        : {note}");
    }
    let mut sockets = run::open_sockets(cfg, plan)?;
    for (i, s) in sockets.iter().enumerate() {
        let granted = s.rcvbuf().unwrap_or(0);
        println!(
            "  [{i}] {}:{}  SO_RCVBUF={} MiB granted, SO_BUSY_POLL={}",
            plan.enabled[i].group_ip,
            plan.enabled[i].dest_port,
            granted / (1024 * 1024),
            if s.busy_poll_granted() { "yes" } else { "no (EPERM)" }
        );
    }

    let wait = Duration::from_secs_f64(cli.duration.unwrap_or(10.0).max(1.0) + 5.0);
    println!("draining every station until all of them are seen (up to {wait:?})");
    let first = run::sniff_first_counters(&mut sockets, wait)?;
    for (i, c) in first.iter().enumerate() {
        println!("  [{i}] {:<6} counter {c}", plan.enabled[i].name);
    }

    let stop = RunStop::new();
    {
        // Bridge the signal flag into the shared stop flag the run watches.
        let stop = stop.clone();
        std::thread::spawn(move || loop {
            if STOP.load(Ordering::Relaxed) {
                stop.stop(StopReason::Signal);
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        });
    }

    println!(
        "receiving ({}); Ctrl-C to stop",
        if dry_run {
            "dry run: no GPU, no output"
        } else {
            "GPU accumulation + writing"
        }
    );
    let report = run::run_receive_with(
        cfg,
        plan,
        sockets,
        &first,
        opts,
        prepared,
        stop,
        Box::new(backend),
    )?;
    print!("{}", report.render(plan));
    if !report.ok() {
        bail!("the run reported truncated packets; the receive buffer is too small");
    }
    if let Some(why) = report.stopped_fatal.or(report.stopped_zero_fill_exceeded) {
        bail!("the run stopped early: {why}");
    }
    Ok(ExitCode::SUCCESS)
}
