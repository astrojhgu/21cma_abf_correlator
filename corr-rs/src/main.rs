//! `corr` -- the correlator's entry point.
//!
//! In this build the binary's job is to prove that the configuration, the two
//! halves of the project and the ABI agree, before any traffic or GPU work is
//! involved. `--check` is the command that does it and is safe to run at any
//! time: it allocates the pinned staging region, exercises the ABI, and exits
//! without opening a socket.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;

use corr_rs::config::Config;
use corr_rs::gpu::Correlator;
use corr_rs::run::{self, RunOptions};
use corr_rs::{ffi, RunPlan};

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
    name = "corr",
    version,
    about = "Real-time correlator for RFSoC array data",
    long_about = "Receives multicast UDP streams from the RFSoC front end, applies a per-station \
                  R2C FFT, accumulates Y_j*conj(Y_k) for every baseline, and writes one \
                  complex64[ntime][n_freq] file per baseline."
)]
struct Cli {
    /// Path to the run configuration.
    #[arg(
        short,
        long,
        default_value = "config/correlator.yaml",
        env = "CORR_CONFIG"
    )]
    config: PathBuf,

    /// Validate the configuration and the CUDA ABI, then exit without receiving.
    #[arg(long)]
    check: bool,

    /// Print the resolved run plan (stations, baselines, derived rates).
    #[arg(long)]
    print_plan: bool,

    /// Override run.output_dir.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Override run.name.
    #[arg(long)]
    run_name: Option<String>,

    /// Receive, assemble and schedule rows, but do not touch the GPU or write
    /// anything. This is the only mode that runs end to end today.
    #[arg(long)]
    dry_run: bool,

    /// Stop after this many seconds. Without it a dry run continues until
    /// interrupted.
    #[arg(long)]
    duration: Option<f64>,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("corr: {e:#}");
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
    if let Some(n) = cli.run_name.clone() {
        cfg.run.name = Some(n);
    }
    cfg.resolve_run_name();

    let plan = cfg.compile().context("configuration does not compile")?;

    println!("corr-rs {}", env!("CARGO_PKG_VERSION"));
    println!("config     : {}", cli.config.display());
    println!("plan       : {}", plan.summary());

    if cli.print_plan || cli.check {
        println!("\nenabled stations, in baseline index order");
        print!("{plan}");
        println!("\nbaselines (j <= k, row major)");
        for (i, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let jn = &plan.enabled[*j].name;
            let kn = &plan.enabled[*k].name;
            let kind = if j == k { "auto " } else { "cross" };
            println!("  [{i:3}] ({j:2},{k:2})  {jn:<5} x {kn:<5} {kind}");
        }
        println!(
            "\noutput     : {}/{}",
            plan.output_dir.display(),
            plan.run_name
        );
        println!(
            "per block  : staging {} MiB, spectrum {} bytes over {} files",
            plan.staging_total_bytes / (1024 * 1024),
            plan.out_bytes_per_row,
            plan.n_baseline
        );
    }

    if cli.check {
        check_abi(&cfg, &plan)?;
    }

    if cli.check || cli.print_plan {
        return Ok(ExitCode::SUCCESS);
    }

    let mode = if cli.dry_run || cfg.run.dry_run {
        Mode::DryRun
    } else {
        Mode::Produce
    };
    execute(&cfg, &plan, cli.duration, mode)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Receive and schedule only: no GPU, no files.
    DryRun,
    /// Accumulate on the GPU and write the output.
    Produce,
}

/// Run the correlator.
fn execute(
    cfg: &Config,
    plan: &RunPlan,
    duration: Option<f64>,
    mode: Mode,
) -> Result<ExitCode> {
    install_signal_handlers();
    run::require_interface(cfg)?;

    let output_dir = match mode {
        Mode::DryRun => None,
        Mode::Produce => {
            let dir = cfg.run.output_dir.join(&plan.run_name);
            println!("output     : {}", dir.display());
            Some(dir)
        }
    };

    let wait = Duration::from_secs_f64(duration.unwrap_or(10.0).max(1.0) + 5.0);

    let opts = RunOptions {
        duration: duration.map(Duration::from_secs_f64),
        wait_for_first_packet: wait,
        output_dir,
        enable_gpu: mode == Mode::Produce,
    };

    // Everything slow happens before a single group is joined. With the sockets
    // open and no thread draining them, each 512 MiB receive buffer fills in about
    // 0.6 s at the real packet rate, and a produce run that spent seconds creating
    // its CUDA context and 300 output files here started on a stream that had
    // already moved several seconds past its start frame -- which is what hung it,
    // with zero slots submitted and no error.
    let prepared = run::prepare_output(cfg, plan, &opts)?;

    println!("opening {} multicast sockets", plan.n_station);
    for note in &plan.cpu_note {
        println!("cpu        : {note}");
    }
    if let Some((threads, cores)) = plan.recv_cpus_shared {
        // Loud on purpose. Every measurement in HANDOFF §6.13 that shows a clean
        // run was taken with one core per receive thread; sharing cores changes
        // the jitter story, and the ring lead is what has to absorb it.
        println!(
            "  WARNING: {threads} receive threads share {cores} cores ({} per core). \
             The one-core-per-station layout this project has been measured with is \
             not available on this machine; if loss appears, widen the ring lead \
             (gpu.block_slots) before blaming the link.",
            threads.div_ceil(cores)
        );
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

    let wait = Duration::from_secs_f64(duration.unwrap_or(10.0).max(1.0) + 5.0);
    println!("draining every station until all of them are seen (up to {wait:?})");
    let first = run::sniff_first_counters(&mut sockets, wait)?;
    for (i, c) in first.iter().enumerate() {
        println!("  [{i}] {:<6} counter {c}", plan.enabled[i].name);
    }

    let stop = Arc::new(AtomicBool::new(false));
    {
        // Bridge the signal flag into the shared stop flag the run watches.
        let stop = stop.clone();
        std::thread::spawn(move || loop {
            if STOP.load(Ordering::Relaxed) {
                stop.store(true, Ordering::Relaxed);
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        });
    }

    println!(
        "receiving ({}); Ctrl-C to stop",
        if mode == Mode::Produce {
            "GPU accumulation + writing"
        } else {
            "dry run: no GPU, no output"
        }
    );
    let report = run::run_receive(cfg, plan, sockets, &first, opts, prepared, stop)?;
    print!("{}", report.render(plan));
    if !report.ok() {
        bail!("the run reported truncated packets; the receive buffer is too small");
    }
    // A run that stopped itself -- the staging geometry never agreed with the
    // front end's counter, or a station was zero filled past the cap -- is not a
    // successful observation, even though its files are complete rows. Exit
    // non-zero so a wrapper script or a monitoring check notices.
    if let Some(why) = report.stopped_fatal.or(report.stopped_zero_fill_exceeded) {
        bail!("the run stopped early: {why}");
    }
    Ok(ExitCode::SUCCESS)
}

/// Validate that the shared object, the configuration and Rust's own view of the
/// baseline numbering all agree.
fn check_abi(cfg: &Config, plan: &RunPlan) -> Result<()> {
    println!("\n--- ABI check ---");
    println!("library    : {}", ffi::abi_banner());

    // Baseline numbering is defined by the library, so Rust must not merely
    // assume it agrees.
    let lib_pairs = Correlator::baseline_pairs(plan.n_station as i32)
        .context("reading baseline numbering from libcorr_cuda")?;
    let rust_pairs: Vec<(i32, i32)> = plan
        .baseline_pairs()
        .into_iter()
        .map(|(j, k)| (j as i32, k as i32))
        .collect();
    if lib_pairs != rust_pairs {
        bail!(
            "baseline numbering disagrees: library {:?} vs corr-rs {:?}",
            lib_pairs,
            rust_pairs
        );
    }
    println!(
        "baselines  : {} pairs agree between libcorr_cuda and corr-rs",
        lib_pairs.len()
    );

    // The devices the engine spans come from gpu.devices, and every station's
    // declared gpu_id has to be one of them (config.rs validates that already).
    // A station's gpu_id says which card receives its stream; it does not
    // restrict which card may compute a baseline with it, which is why one card
    // can be selected here -- `gpu.devices: [0]` -- while the station table keeps
    // its wider assignment. That is the fallback to a single card.
    let devices = cfg.gpu.devices.clone();
    let corr = Correlator::new(&devices, cfg.gpu.split.abi_value(), plan)
        .with_context(|| format!("creating a corr_cuda context on device(s) {devices:?}"))?;

    println!(
        "context    : {} device(s) {:?} n_station={} n_freq={} frames_per_slot={} \
         chunk_frames={} slots={} rows_per_fetch={}",
        corr.n_devices().unwrap_or(-1),
        devices,
        corr.n_station(),
        corr.n_freq(),
        corr.frames_per_slot(),
        cfg.gpu.chunk_frames,
        corr.block_slots(),
        corr.rows_per_fetch()
    );

    // The staging geometry is the contract the receive threads will rely on, so
    // verify it rather than trusting it.
    let staging = corr.staging_all()?;
    let expect_frame = 2 * plan.n_freq as usize;
    let expect_slot = expect_frame * plan.frames_per_slot as usize;
    let expect_station = expect_slot * plan.block_slots as usize;
    for (i, st) in staging.iter().enumerate() {
        if st.frame_stride != expect_frame {
            bail!(
                "station {i}: frame_stride {} but 2*n_freq = {expect_frame}",
                st.frame_stride
            );
        }
        if st.slot_stride != expect_slot {
            bail!(
                "station {i}: slot_stride {} but frames_per_slot*2*n_freq = {expect_slot}",
                st.slot_stride
            );
        }
        // base is advanced to this station, so total_bytes is its extent, not
        // the size of the whole allocation.
        if st.total_bytes != expect_station {
            bail!(
                "station {i}: total_bytes {} but station extent = {expect_station}",
                st.total_bytes
            );
        }
        if st.base.is_null() {
            bail!("station {i}: staging base is NULL");
        }
    }
    // Stations are laid out end to end, so the scheduler can index them
    // arithmetically. The public struct hides the station stride, but the
    // distance between consecutive bases is exactly it.
    let station_gap = expect_station;
    for w in staging.windows(2) {
        // SAFETY: both pointers are inside the same pinned allocation, and the
        // offset is exactly the stride between consecutive stations.
        if w[1].base != unsafe { w[0].base.add(station_gap) } {
            bail!("station staging regions are not contiguous");
        }
    }
    println!(
        "staging    : {} stations, frame_stride={expect_frame} slot_stride={expect_slot} \
         station_stride={expect_station} total={} MiB",
        staging.len(),
        expect_station * staging.len() / (1024 * 1024)
    );

    // Touching the staging through the ABI is what the receive threads will do
    // millions of times per second; a mistake here would stay invisible until
    // the spectra came out wrong.
    let s0 = &staging[0];
    let probe = s0.frame_stride * 3 + 7;
    // SAFETY: probe is within one frame of a station-sized region.
    unsafe {
        *s0.base.add(probe) = -128;
        if *s0.base.add(probe) != -128 {
            bail!("pinned staging write did not read back");
        }
        *s0.base.add(probe) = 0;
    }
    println!("staging    : pinned writes verified");

    println!("result     : CUDA ABI, config and numbering are consistent");
    Ok(())
}
