//! `corr-replay` -- a synthetic front end, for testing the receive path.
//!
//! This is **not** a pcap replayer. The host cannot capture packets (`tcpdump`
//! and `AF_PACKET` need privileges `jhgu` does not have), so there is nothing to
//! replay. What this does instead is generate the front end's traffic from
//! scratch: the same 4104-byte payloads, the same little-endian global packet
//! counter, the same multicast groups and ports as the configuration it is
//! pointed at.
//!
//! Why it exists: the real link is clean enough (120 s, zero gaps, zero
//! reordering) that the branches worth testing never occur naturally. A lost
//! packet, a counter jump, a station that stops for a second, an offset between
//! stations -- those have to be manufactured.
//!
//! It uses the configuration's own station table, so the groups and ports come
//! from one place. Point it at `config/correlator-test.yaml`, whose groups
//! (239.1.200.x) are deliberately different from the live front end's, and a
//! replay can run while the telescope is observing without the two streams
//! interleaving their counters.
//!
//! It deliberately does **not** try to reproduce the real 195 kpps per station:
//! frame assembly, the slot ring and the watermark are rate independent, and
//! asking for real rate would mean four threads pushing 3.2 GB/s of loopback for
//! no extra coverage.

use std::net::Ipv4Addr;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;

use corr_rs::config::{Config, SAMPLES_PER_PACKET};

/// The counter is 8 bytes ahead of the samples.
const HEADER: usize = 8;
const DATAGRAM: usize = HEADER + SAMPLES_PER_PACKET as usize;

#[derive(Parser, Debug)]
#[command(
    name = "corr-replay",
    version,
    about = "Synthetic RFSoC front end for testing the correlator's receive path",
    long_about = "Sends 4104-byte UDP multicast packets -- an 8-byte little-endian packet counter \
                  followed by 4096 int8 samples -- to the groups in the configuration, so that \
                  the receive path can be driven deterministically. Faults (loss, gaps, a silent \
                  station, a per-station counter offset) are injected on request."
)]
struct Cli {
    /// Configuration whose station table and groups to send to.
    #[arg(short, long, default_value = "config/correlator-test.yaml", env = "CORR_CONFIG")]
    config: std::path::PathBuf,

    /// How long to transmit.
    #[arg(long, default_value_t = 10.0)]
    duration: f64,

    /// Packets per second per station. The real front end does 195312.5; a test
    /// does not need to, and the receive logic does not care.
    #[arg(long, default_value_t = 20000.0)]
    rate: f64,

    /// First packet counter. Defaults to something in the range the real front
    /// end is at, so the arithmetic under test matches production.
    #[arg(long, default_value_t = 10_000_000_000)]
    start_counter: u64,

    /// Drop this fraction of packets, uniformly. 0.001 is one in a thousand.
    #[arg(long, default_value_t = 0.0)]
    drop_rate: f64,

    /// Skip this many consecutive counters starting at `--start-counter + gap-at`.
    #[arg(long, default_value_t = 0)]
    gap_len: u64,

    /// Where in the run the gap starts, in packets after the first one.
    #[arg(long, default_value_t = 0)]
    gap_at: u64,

    /// Make station index INDEX go silent for SECONDS, starting `--stall-at`
    /// seconds in. Format: INDEX:SECONDS.
    #[arg(long, value_parser = parse_pair)]
    stall: Option<(u64, f64)>,

    /// When the stall starts, in seconds after the first packet.
    #[arg(long, default_value_t = 2.0)]
    stall_at: f64,

    /// Start station index INDEX this many packets ahead of the rest, to prove
    /// the alignment does not depend on a per-station offset. Format: INDEX:N.
    #[arg(long, value_parser = parse_pair_u64)]
    offset: Option<(u64, u64)>,

    /// Sample waveform: "sine" puts a tone in a known bin (useful for checking a
    /// spectrum by hand), "counter" makes every sample a function of the packet
    /// counter (useful for spotting a frame assembled out of order).
    #[arg(long, default_value = "counter", value_parser = ["sine", "counter", "zero", "prng"])]
    pattern: String,

    /// Frequency of the tone for `--pattern sine`, as a fraction of the sample
    /// rate.
    #[arg(long, default_value_t = 0.05)]
    tone: f64,

    /// Seed for the synthetic sample stream, so runs are reproducible.
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// Delay one station's *sample* stream by N samples. Format: INDEX:N.
    ///
    /// The delay is applied to the underlying sample sequence, not to the packet
    /// boundaries, so it crosses packets exactly as a real geometric delay does:
    /// packet c of the delayed station carries samples starting one before packet
    /// c of the reference. A correlator that got the packet alignment, the
    /// conjugation or the FFT bin order wrong would show it in the phase.
    /// Repeatable: `--sample-delay 3:12 --sample-delay 7:40`.
    #[arg(long, value_parser = parse_pair_i64)]
    sample_delay: Vec<(u64, i64)>,

    /// Give station `i` a sample delay of `i * N`, which is the pattern a
    /// geometric delay across a linear array produces.
    ///
    /// One flag instead of one per station: a 24-station phase test needs 24
    /// different delays, and writing 24 `--sample-delay` arguments by hand is a
    /// transcription error waiting to become an unexplained phase slope. A delay
    /// gradient is the *first* thing a correlator should be tested with, since
    /// the slope of phase against channel is exactly the delay difference and
    /// nothing else can imitate it across all 300 baselines at once.
    #[arg(long, default_value_t = 0)]
    delay_stride: i64,

    /// Step the whole delay ladder every this many frames (0 = no ladder).
    ///
    /// With `--delay-stride s`, station i is delayed `s*i*k` samples while the
    /// ladder step is k, where k = 1 + (frame - origin)/FRAMES. Step boundaries
    /// fall on multiples of FRAMES in the global frame numbering, and the
    /// correlator's rows are aligned to `n_cum` frames, so FRAMES = n_cum makes
    /// the two grids coincide: every output row integrates exactly one step.
    ///
    /// `origin` is the first multiple of FRAMES at or after the first frame sent,
    /// which is what makes the alignment hold regardless of `--start-counter`.
    /// The delay is a property of the *sample stream*, so a step boundary is a
    /// discontinuity in that stream: the correlation of a row is unaffected (the
    /// delay is constant across it), and the phase measured from row to row steps
    /// by `2*pi*f*s*dN/fs` -- which is the relation this exists to test.
    #[arg(long, default_value_t = 0)]
    ladder_frames: u64,

    /// Stop counting up after this many ladder steps (0 = no cap). Keeps the
    /// phase from growing without bound in a long run.
    #[arg(long, default_value_t = 10)]
    ladder_steps: u64,

    /// Print what would be sent without sending it.
    #[arg(long)]
    dry_run: bool,
}

fn parse_pair(s: &str) -> Result<(u64, f64), String> {
    let (a, b) = s.split_once(':').ok_or("expected INDEX:VALUE")?;
    Ok((
        a.parse().map_err(|_| "bad index")?,
        b.parse().map_err(|_| "bad value")?,
    ))
}

fn parse_pair_i64(s: &str) -> Result<(u64, i64), String> {
    let (a, b) = s.split_once(':').ok_or("expected INDEX:VALUE")?;
    Ok((
        a.parse().map_err(|_| "bad index")?,
        b.parse().map_err(|_| "bad value")?,
    ))
}

fn parse_pair_u64(s: &str) -> Result<(u64, u64), String> {
    let (a, b) = s.split_once(':').ok_or("expected INDEX:VALUE")?;
    Ok((
        a.parse().map_err(|_| "bad index")?,
        b.parse().map_err(|_| "bad value")?,
    ))
}

/// Point multicast sends at one interface, by its address.
fn set_multicast_interface(fd: std::os::unix::io::RawFd, iface: Ipv4Addr) -> Result<()> {
    let addr = libc::in_addr {
        s_addr: u32::from(iface).to_be(),
    };
    // SAFETY: addr is a fully initialised in_addr of the given length.
    let rc = unsafe {
        libc::setsockopt(
            fd,
            libc::IPPROTO_IP,
            libc::IP_MULTICAST_IF,
            &addr as *const _ as *const libc::c_void,
            std::mem::size_of::<libc::in_addr>() as libc::socklen_t,
        )
    };
    if rc != 0 {
        return Err(anyhow::Error::from(std::io::Error::last_os_error()))
            .context("setsockopt(IP_MULTICAST_IF)");
    }
    Ok(())
}

/// Deterministic int8 sample at an absolute sample index.
///
/// Indexed rather than generated sequentially, because a delayed station has to
/// ask for sample `i - delay` for any `i`, including negative indices at the very
/// start. splitmix64 is enough for a test signal and has no state to carry.
fn prng_sample(i: i64, seed: u64) -> u8 {
    let mut x = (i as u64)
        .wrapping_add(seed)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    // The front end's samples sit in -4..=4; matching that keeps the test's
    // amplitude realistic and catches any accidental float scaling.
    (((x >> 8) % 9) as i8 - 4) as u8
}

/// Build one packet: counter, then samples. `sample_delay` shifts this station's
/// view of the underlying sequence.
fn build_packet(counter: u64, station: usize, sample_delay: i64, cli: &Cli, out: &mut [u8]) {
    debug_assert_eq!(out.len(), DATAGRAM);
    out[..HEADER].copy_from_slice(&counter.to_le_bytes());
    let samples = &mut out[HEADER..];

    match cli.pattern.as_str() {
        "zero" => samples.iter_mut().for_each(|b| *b = 0),
        "counter" => {
            // A cheap deterministic hash of the counter and the position inside
            // the packet: a frame assembled from the wrong packets is then
            // visible in the numbers rather than merely in the statistics.
            let mut x = counter
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(station as u64 * 0x1234_5678);
            for b in samples.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                // Keep it inside the range the front end actually emits: -4..=4
                // as a two's complement byte. Computing this in u8 would wrap.
                *b = (((x >> 3) % 9) as i8 - 4) as u8;
            }
        }
        "prng" => {
            let first = (counter as i64) * SAMPLES_PER_PACKET as i64 - sample_delay;
            for (k, b) in samples.iter_mut().enumerate() {
                *b = prng_sample(first + k as i64, cli.seed);
            }
        }
        "sine" => {
            let phase = (counter as f64 + cli.seed as f64 * station as f64) * cli.tone
                * std::f64::consts::TAU;
            for (i, b) in samples.iter_mut().enumerate() {
                let t = phase + (i as f64) * cli.tone * std::f64::consts::TAU
                    / SAMPLES_PER_PACKET as f64;
                *b = (t.sin() * 4.0).round() as i8 as u8;
            }
        }
        _ => unreachable!("clap restricts the value"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.rate <= 0.0 {
        bail!("--rate must be positive");
    }
    if !(0.0..1.0).contains(&cli.drop_rate) {
        bail!("--drop-rate must be in [0, 1)");
    }

    let cli = Arc::new(cli);
    let cfg = Config::from_yaml_file(&cli.config)?;
    let plan = cfg.compile()?;
    if plan.n_station == 0 {
        bail!("no station is enabled in {}", cli.config.display());
    }
    // The guard that matters is *which groups* this would transmit into, not how
    // many stations there are. Counting stations was a proxy for "this looks like
    // the live configuration", and it blocked the one thing the 40-station target
    // needs: a 40-station replay into the test group space. The configuration
    // already says which is which -- every board is named in `board`, and the
    // replayer's own configs say `replay` -- so that is what is checked.
    let live_boards: Vec<&str> = plan
        .enabled
        .iter()
        .filter_map(|e| {
            let b = cfg.stations.get(e.cfg_index).map(|s| s.board.as_str())?;
            (b != "replay").then_some(b)
        })
        .collect();
    if !live_boards.is_empty() && plan.n_station > 16 {
        bail!(
            "{} stations are enabled and {} of them are not marked `board: replay` \
             (first: {}); refusing to transmit a synthetic counter into what may be \
             the live front end's groups",
            plan.n_station,
            live_boards.len(),
            live_boards[0]
        );
    }

    // Frames per packet, and the ladder's frame grid. `origin` must be the first
    // multiple of the ladder step at or after the first frame sent: the
    // correlator's rows are aligned to `n_cum` frames in the same numbering, so
    // that is what makes one row integrate exactly one ladder step.
    let packets_per_frame = (2 * cfg.fft.n_freq as u64) / SAMPLES_PER_PACKET as u64;
    let first_frame = (cli.start_counter * SAMPLES_PER_PACKET as u64) / (2 * cfg.fft.n_freq as u64);
    let ladder_origin = if cli.ladder_frames > 0 {
        first_frame.div_ceil(cli.ladder_frames) * cli.ladder_frames
    } else {
        0
    };

    println!(
        "corr-replay: {} stations, {:.0} pps each ({:.1} MB/s total), {:.1} s",
        plan.n_station,
        cli.rate,
        cli.rate * DATAGRAM as f64 * plan.n_station as f64 / 1e6,
        cli.duration
    );
    if cli.ladder_frames > 0 {
        println!(
            "  ladder: station i delayed {} * i * k samples, k stepping every {} frames \
             ({} per frame), first frame {} -> origin {}, {} steps{}",
            cli.delay_stride,
            cli.ladder_frames,
            packets_per_frame,
            first_frame,
            ladder_origin,
            cli.ladder_steps,
            if cli.delay_stride == 0 { " -- WARNING: stride 0, every step is a no-op" } else { "" }
        );
    }
    for s in &plan.enabled {
        println!("  [{:2}] {:<6} -> {}:{}", s.index, s.name, s.group_ip, s.dest_port);
    }

    if cli.dry_run {
        println!("corr-replay: dry run, nothing sent");
        return Ok(());
    }

    let stop = Arc::new(AtomicBool::new(false));
    let sent = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();

    for st in plan.enabled.iter() {
        let group = st.group_ip;
        let port = st.dest_port;
        let index = st.index as u64;
        let stop = stop.clone();
        let sent = sent.clone();
        let start_counter = cli.start_counter
            + cli.offset
                .filter(|(i, _)| *i == index)
                .map(|(_, n)| n)
                .unwrap_or(0);
        let sample_delay = cli
            .sample_delay
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, n)| *n)
            .unwrap_or_else(|| cli.delay_stride * index as i64);
        // The fixed part of the ladder: `k` below multiplies this.
        let ladder_base = sample_delay;
        let cli = cli.clone();
        let iface_ip = cfg.receiver.interface_ip;
        let iface_name = cfg.receiver.interface.clone();

        handles.push(std::thread::spawn(move || -> Result<u64> {
            let cli: &Cli = &cli;
            let sock = std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))
                .context("binding the replay socket")?;
            sock.set_multicast_loop_v4(true)
                .context("enabling multicast loopback")?;
            // A modest send buffer is plenty at test rates.
            let _ = sock.set_write_timeout(Some(Duration::from_millis(100)));

            // Sending out of the right interface is not optional. Without
            // IP_MULTICAST_IF the kernel picks the interface from the routing
            // table, which on this host is not the one the correlator joined the
            // group on -- the packets go out somewhere else and the receiver sees
            // nothing at all, with no error anywhere.
            set_multicast_interface(sock.as_raw_fd(), iface_ip).with_context(|| {
                format!("routing replay traffic out of {iface_ip} ({iface_name})")
            })?;

            let mut packet = vec![0u8; DATAGRAM];
            let mut sample_delay = sample_delay;
            let start = Instant::now();
            let total = (cli.duration * cli.rate) as u64;
            let mut n = 0u64;
            let mut dropped = 0u64;

            while n < total && !stop.load(Ordering::Relaxed) {
                // A silent station: this thread simply stops sending.
                if let Some((si, secs)) = cli.stall {
                    let t = start.elapsed().as_secs_f64();
                    if si == index && t >= cli.stall_at && t < cli.stall_at + secs {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                }

                let counter = start_counter + n;
                // A gap: a run of counters that never goes out on the wire. The
                // receives side cannot tell this from a lost burst by construction,
                // which is exactly what makes it a useful test.
                let in_gap = cli.gap_len > 0
                    && n >= cli.gap_at
                    && n < cli.gap_at + cli.gap_len;
                // Uniform loss.
                let lost = cli.drop_rate > 0.0 && {
                    let mut x = counter
                        .wrapping_mul(0x2545_F491_4F6C_DD1D)
                        .wrapping_add(index);
                    x ^= x >> 33;
                    (x % 1_000_000) < (cli.drop_rate * 1_000_000.0) as u64
                };
                if in_gap || lost {
                    dropped += 1;
                } else {
                    if cli.ladder_frames > 0 {
                        let frame = counter / packets_per_frame;
                        let mut k = if frame >= ladder_origin {
                            1 + (frame - ladder_origin) / cli.ladder_frames
                        } else {
                            1
                        };
                        if cli.ladder_steps > 0 {
                            k = k.min(cli.ladder_steps);
                        }
                        sample_delay = ladder_base * k as i64;
                    }
                    build_packet(counter, index as usize, sample_delay, cli, &mut packet);
                    // A full send buffer would block; dropping is the honest
                    // outcome and the receive path is expected to survive it.
                    match sock.send_to(&packet, (group, port)) {
                        Ok(_) => {
                            sent.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => dropped += 1,
                    }
                }
                n += 1;

                // Pace against the start time rather than sleeping a fixed
                // amount per packet, so the rate does not drift with the cost of
                // the send itself.
                let target = start + Duration::from_secs_f64(n as f64 / cli.rate);
                let now = Instant::now();
                if target > now {
                    std::thread::sleep(target - now);
                }
            }
            let _ = dropped;
            Ok(n)
        }));
    }

    // Stop when the longest station finishes.
    let monitor = {
        let stop = stop.clone();
        let duration = cli.duration;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(duration + 0.5));
            stop.store(true, Ordering::Relaxed);
        })
    };

    let mut total = 0u64;
    for h in handles {
        match h.join() {
            Ok(Ok(n)) => total += n,
            Ok(Err(e)) => eprintln!("corr-replay: {e:#}"),
            Err(_) => eprintln!("corr-replay: a sender thread panicked"),
        }
    }
    let _ = monitor.join();

    println!(
        "corr-replay: {} slots attempted, {} packets sent",
        total,
        sent.load(Ordering::Relaxed)
    );
    Ok(())
}
