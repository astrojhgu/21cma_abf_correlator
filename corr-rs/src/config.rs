//! Run configuration: parsing, validation, and the derived numbers the rest of
//! the program works from.
//!
//! Two rules shape this module:
//!
//! 1. **Refuse to start on anything inconsistent.** A correlator that runs with
//!    a mismatched `n_freq` or a duplicated station silently produces wrong
//!    spectra, and wrong spectra are far more expensive than a failed startup.
//!    So unknown fields are rejected, and every cross-field invariant is checked
//!    before a single socket is opened.
//!
//! 2. **Derive, do not duplicate.** Frame rate, bin width, block period, staging
//!    size and output rate are all computed here from `sample_rate_hz`,
//!    `n_freq`, `n_cum` and the enabled station set, and written into the run
//!    metadata. Nothing downstream recomputes them.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Samples carried by one UDP packet payload, after the 8-byte counter.
pub const SAMPLES_PER_PACKET: u64 = 4096;

/// Configuration format version accepted by this build.
pub const CONFIG_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Raw configuration, one struct per YAML section
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub run: RunCfg,
    pub frontend: FrontendCfg,
    pub receiver: ReceiverCfg,
    pub fft: FftCfg,
    pub integration: IntegrationCfg,
    pub gpu: GpuCfg,
    pub storage: StorageCfg,
    pub cpu: CpuCfg,
    pub monitoring: MonitoringCfg,
    pub stations: Vec<StationCfg>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunCfg {
    /// `None` means: derive `corr_YYYYmmdd_HHMMSS` at startup.
    pub name: Option<String>,
    pub output_dir: PathBuf,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendCfg {
    /// Front-end quantisation shift. Metadata only; it never enters the FFT.
    pub data_shift_count: u32,
    pub sample_dtype: String,
    pub sample_format: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiverCfg {
    pub interface: String,
    pub interface_ip: Ipv4Addr,
    pub recv_buffer_bytes: usize,
    pub busy_poll_us: u32,
    pub recvmmsg_batch: u32,
    /// `SO_RCVTIMEO` for the receive socket. Short by design: an idle poll is how
    /// a receive thread notices that it should service other work, such as the
    /// scheduler's request to zero fill frames for a station that went quiet.
    pub recv_timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FftCfg {
    pub n_freq: u32,
    pub sample_rate_hz: u64,
    pub window: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrationCfg {
    pub n_cum: u32,
    pub block_timeout_ms: u64,
    pub start_frame_id: Option<u64>,
    /// Cap on how many frames one station may be asked to zero fill for.
    ///
    /// Zero filling keeps a run alive across a brief interruption, but past some
    /// point the output is rows of silence dressed up as observations, and the
    /// receive-side buffers keep growing to hold data nobody will use. When a
    /// station exceeds this, the run stops cleanly instead.
    ///
    /// 0 disables the cap, which is only sensible for short tests.
    pub zero_fill_limit_frames: u64,
}

/// How the pair space is divided over `gpu.devices`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMode {
    /// Every device holds every station and owns a contiguous range of
    /// baselines. N devices move N times the run's bytes; right for one device,
    /// and for several while the station count is small.
    BaselineRange,
    /// Each device holds a subset of the stations and owns the baselines among
    /// them. Total host traffic is the sum of the resident set sizes over the
    /// station count -- 2.17x for 24 stations over 4 cards instead of 4x -- which
    /// is what makes a station count too large for one card's host link fit.
    PairCovering,
    /// Split by *frames*: every device holds every station and computes every
    /// baseline, but only for its share of the slots; the library adds the
    /// partial rows when they are fetched. The host traffic is then exactly the
    /// run's data rate (R = 1), which is what makes a station count that does not
    /// fit one card's host link possible without replicating anything.
    TimeSlice,
}

impl SplitMode {
    pub fn abi_value(self) -> i32 {
        match self {
            SplitMode::BaselineRange => crate::ffi::CORR_SPLIT_BASELINE_RANGE,
            SplitMode::PairCovering => crate::ffi::CORR_SPLIT_PAIR_COVERING,
            SplitMode::TimeSlice => crate::ffi::CORR_SPLIT_TIME_SLICE,
        }
    }
}

fn default_split_mode() -> SplitMode {
    SplitMode::BaselineRange
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuCfg {
    pub devices: Vec<i32>,
    /// How the pair space is divided over `devices`. Defaults to the simple
    /// range split, which is also the single-device behaviour, so a config that
    /// does not mention it keeps working unchanged.
    #[serde(default = "default_split_mode")]
    pub split: SplitMode,
    /// Frames one staging slot holds per station. Together with `block_slots`
    /// this fixes the pinned footprint, and it is deliberately independent of
    /// the integration length: a 1 s row and a 10 ms row need the same staging.
    pub frames_per_slot: u32,
    /// Frames per FFT + correlate launch on the device. Must be <=
    /// frames_per_slot.
    pub chunk_frames: u32,
    /// Staging slots in the ring. >= 2, so the receive threads always have
    /// somewhere to write while a copy is in flight.
    pub block_slots: u32,
    /// Staging slots reserved for the receive side; the pipeline holds
    /// `block_slots - ring_lead` of them.
    ///
    /// This is the setting that decides whether a *long* run keeps its data, and
    /// it took a 15-minute run to see why. The receive threads may only read as
    /// far ahead as they have writable slots, so `ring_lead` is the entire buffer
    /// between the front end's jitter and packet loss. At `block_slots=8` with the
    /// old "everything but one slot in flight" policy the receive side had 7
    /// writable slots (37 ms at 24 stations, nch=8192): over 20 s that looks
    /// perfect, because the socket buffer absorbs the deficit, but over 300 s at
    /// the same settings the loss settles at 25% (nch=4096) / 10% (nch=8192) and
    /// never recovers. Measured with 12 writable slots and 4 copies in flight:
    /// 60 packets lost out of 58.6M over 300 s, receive threads idle 83%.
    ///
    /// Defaults to `block_slots - 1`, i.e. one slot in flight.
    #[serde(default)]
    pub ring_lead: Option<u32>,
    /// Integration rows held in device memory between host transfers.
    ///
    /// This is what decouples the integration length from the transfer and
    /// writer cadence. With a 1 s integration, 1 is right. With a 10 ms
    /// integration, fetching every row would push n_baseline*n_freq*8 bytes
    /// across PCIe 100 times a second and wake the writer 100 times a second;
    /// setting this to about (1 second / integration length) restores a 1 Hz
    /// cadence with no change to the integration itself.
    pub rows_per_fetch: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StorageCfg {
    pub writers: usize,
    pub queue_bytes: u64,
    pub fsync_on_stop: bool,
    /// How many rows to write before syncing the baseline files to the platter.
    ///
    /// 1 means every row is durable before the sidecars claim it exists, which
    /// costs one `fsync` per baseline file per row. At a 1 s integration that is
    /// 300 fsyncs a second for 24 stations and no problem at all; at 10 ms it is
    /// 28610 a second, which no network filesystem will take. The trade is
    /// explicit: a larger value means a crash can lose up to that many rows, and
    /// the sidecars are still written after the data, so a lost row is missing
    /// from all four files rather than half written.
    pub fsync_every_rows: u64,
    /// Write each local day's rows into their own set of files, named with the
    /// local day and the local time of that set's first row.
    ///
    /// On by default, for every run: a file should hold the day it says it holds,
    /// and that should not depend on how the run was invoked. Turn it off to get
    /// the old unprefixed names back (used by short tests and by anything that
    /// globs `baseline_*.bin`).
    ///
    /// Details, and the two risks this accepts (a ~0.5 s stall at the boundary,
    /// and no completeness marker for a day whose process died): see
    /// docs/DESIGN-daily-files.md.
    #[serde(default = "default_daily_files")]
    pub daily_files: bool,
    /// A prefix put verbatim at the front of every file name of a set, with no
    /// path component and no strftime directive in it.
    ///
    /// It is the run-level half of a name; the other half is the local time of
    /// the set's first row, formatted with `file_prefix_format`. So a set looks
    /// like
    ///
    ///   <file_prefix><station_j><station_k>-<stamp>.bin   one file per baseline
    ///   <file_prefix><stamp>-time.txt                     per-row wall clock
    ///   <file_prefix><stamp>-time.bin                     the same, int64 LE
    ///   <file_prefix><stamp>-blocks.csv                   per-row validity
    ///   <file_prefix><stamp>-metadata.yaml                how to read the above
    ///
    /// Nothing is inserted between the halves, so whatever separators a name
    /// should have belong in this string. With the default empty prefix:
    ///
    ///   S01W17-20260918-144142.bin
    ///   20260918-144142-time.txt
    ///
    /// and with `file_prefix: "corr24-"`:
    ///
    ///   corr24-S01W17-20260918-144142.bin
    ///   corr24-20260918-144142-time.txt
    ///
    /// Empty is the default, because a name is read far more often than it is
    /// written and the station pair plus the time already identify a file. Use it
    /// to tell runs apart when several land in one directory (see
    /// `docs/DESIGN-daily-files.md`).
    #[serde(default)]
    pub file_prefix: String,
    /// strftime format for the local time a file set is named after: the local
    /// time of that set's first row. Must contain at least one `%` directive and
    /// must not contain a path separator -- both are validated, because a format
    /// that produces the same name every day would collide at the first boundary.
    #[serde(default = "default_file_prefix_format")]
    pub file_prefix_format: String,
}

fn default_daily_files() -> bool {
    true
}

fn default_file_prefix_format() -> String {
    DEFAULT_FILE_PREFIX_FORMAT.to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CpuCfg {
    /// Receive cores, one per station, in station order.
    ///
    /// Empty is allowed only with [`CpuCfg::auto_layout`], which fills it in.
    #[serde(default)]
    pub recv_cpus: Vec<u32>,
    /// Ignored (and filled in) when `auto_layout` is set.
    #[serde(default)]
    pub pipeline_cpu: u32,
    #[serde(default)]
    pub writer_cpus: Vec<u32>,
    #[serde(default)]
    pub monitor_cpu: u32,
    /// NICs whose NUMA node must stay free of receive threads.
    ///
    /// This is the parameter that was missing, and it is not cosmetic. Measured
    /// on this machine (24 stations, nch=8192): the front-end NIC and its mlx5
    /// completion IRQs live on NUMA node 0, the receive threads were pinned to
    /// cores 0-23 -- the same cores the NAPI work runs on -- and the RX ring is
    /// only 1024 descriptors deep (about 0.23 ms of traffic), so a NAPI delay
    /// the receive threads themselves caused dropped packets *in hardware*
    /// (`rx_discards_phy`, matching the run's own loss one for one). Moving the
    /// receive threads to cores 4-27 and leaving node 0 to the NIC cut the
    /// factory-fabricated frames from 1.61% to 0.23% *at 1.5x the machine load*.
    ///
    /// Naming an interface here removes its node's cores from the pool that
    /// `auto_layout` draws from, and reports the overlap if the explicit lists
    /// still use them.
    #[serde(default)]
    pub avoid_nic_interfaces: Vec<String>,
    /// Derive `recv_cpus`, `pipeline_cpu`, `writer_cpus` and `monitor_cpu` from
    /// the online CPUs minus the avoided NUMA nodes, instead of listing cores by
    /// hand.
    ///
    /// The derived layout is deliberately the one that was measured to work:
    /// receive threads take the head of the pool, and the pipeline, the writers
    /// and the monitor take its last `writers + 2` cores, so those four share a
    /// node instead of being scattered.
    #[serde(default)]
    pub auto_layout: bool,
}

/// A resolved CPU assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuLayout {
    pub recv_cpus: Vec<u32>,
    pub pipeline_cpu: u32,
    pub writer_cpus: Vec<u32>,
    pub monitor_cpu: u32,
}

/// `"0-3,8-11,20"` -> `[0,1,2,3,8,9,10,11,20]`, the format the kernel uses in
/// `/sys/devices/system/{cpu/online,node/nodeN/cpulist}`.
pub fn parse_cpulist(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                    out.extend(a..=b);
                }
            }
            None => {
                if let Ok(v) = part.parse::<u32>() {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// Split `pool` into a receive run at the head and the pipeline/writer/monitor
/// cores at the tail.
///
/// Taking the tail rather than the head is what keeps those four on one node:
/// the pool is built node by node, so its last few cores are contiguous.
fn layout_from_pool(pool: &[u32], n_station: usize, writers: usize) -> Option<CpuLayout> {
    let tail = writers + 2;
    if pool.len() < n_station + tail {
        return None;
    }
    let split = pool.len() - tail;
    Some(CpuLayout {
        recv_cpus: pool[..split].to_vec(),
        pipeline_cpu: pool[split],
        writer_cpus: pool[split + 1..split + 1 + writers].to_vec(),
        monitor_cpu: pool[split + 1 + writers],
    })
}

/// The NUMA node an interface lives on, and that node's cores.
fn nic_node_cpus(iface: &str) -> Option<(i32, Vec<u32>)> {
    let node: i32 = std::fs::read_to_string(format!(
        "/sys/class/net/{iface}/device/numa_node"
    ))
    .ok()?
    .trim()
    .parse()
    .ok()?;
    if node < 0 {
        return None;
    }
    let list = std::fs::read_to_string(format!(
        "/sys/devices/system/node/node{node}/cpulist"
    ))
    .ok()?;
    Some((node, parse_cpulist(&list)))
}

/// Online CPUs, per the kernel.
fn online_cpus() -> Option<Vec<u32>> {
    Some(parse_cpulist(
        &std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?,
    ))
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MonitoringCfg {
    /// How often the pipeline thread prints one line of live state to stderr:
    /// rows closed and their rate, per-row zero fill (cumulative and over the
    /// interval), packets the streams never delivered, how much of the interval
    /// the receive side spent waiting for a staging slot, the host-to-device rate
    /// against what the streams require, this process's cores against the cores
    /// online, and the machine's load -- ending in `capacity:`, which names the
    /// first resource that is at its ceiling.
    ///
    /// **0 disables it**, which is the right setting for a measurement whose
    /// output is being parsed line by line.
    pub interval_ms: u64,
    /// Append the same numbers to `<output_dir>/<run_name>/stats.csv`.
    ///
    /// **Not implemented yet**: the field is accepted and ignored, so that a
    /// configuration written for it does not silently mean something else. The
    /// line on stderr is the live view; the run report is the record.
    pub stats_csv: bool,
    /// The read-only status endpoint, for a monitor that runs as its own program
    /// (`tools/corr-web.py` is one): `GET /status.json` returns one JSON object,
    /// rendered per request from the sampler's latest snapshot.
    ///
    /// **Off unless this says `enabled: true`**: a production run should not open
    /// a listening socket because of a default. Nothing in the data path reads
    /// this -- the sampler publishes whether or not anybody is listening, and the
    /// endpoint is served from a thread of its own
    /// (`docs/DESIGN-status-http.md`).
    #[serde(default)]
    pub http: HttpCfg,
}

/// The status endpoint's configuration.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HttpCfg {
    /// Serve `GET /status.json` for as long as the run is up.
    #[serde(default)]
    pub enabled: bool,
    /// `IP:port` to listen on; the loopback by default.
    ///
    /// Validated while the configuration is loaded rather than when the socket is
    /// bound, so a typo stops the run at startup instead of leaving an operator
    /// waiting for a monitor page that never appears.
    #[serde(default = "default_http_bind")]
    pub bind: String,
}

impl Default for HttpCfg {
    fn default() -> Self {
        HttpCfg {
            enabled: false,
            bind: default_http_bind(),
        }
    }
}

fn default_http_bind() -> String {
    DEFAULT_HTTP_BIND.to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StationCfg {
    pub name: String,
    pub group_ip: Ipv4Addr,
    pub dest_port: u16,
    pub interface: String,
    pub source_ip: Ipv4Addr,
    pub board: String,
    /// Ordering key. Enabled stations are compacted into 0..n_station-1 by this
    /// key, and that compacted index is the j/k of the baseline formulas.
    pub order: u32,
    pub gpu_id: i32,
    pub enabled: bool,
}

/// A short, stable fingerprint of a configuration.
///
/// Written into `metadata.yaml` and re-checked when a run directory is reopened:
/// resuming with different geometry would silently misread every existing row.
/// This is a fingerprint, not a security measure -- it only has to notice that
/// two configurations differ.
///
/// **`monitoring` is left out on purpose.** It decides how a run is *watched* --
/// how often the stderr line is printed, and whether the read-only status endpoint
/// listens -- and none of that changes what a stored row means. Including it meant
/// that switching the status endpoint on refused to append to the day's existing
/// set, reporting a hash mismatch instead of the harmless change behind it: a
/// false alarm of exactly the kind this check exists to avoid. Everything else
/// stays in, and the question to ask before excluding anything else is the same
/// one this check asks -- would that change make an existing row mean something
/// else?
pub fn config_hash(cfg: &Config) -> String {
    // FNV-1a over the serialised form: stable, dependency free, and enough to
    // catch a changed n_freq, n_cum, station set or ordering.
    let mut doc = serde_yaml::to_value(cfg).unwrap_or(serde_yaml::Value::Null);
    if let serde_yaml::Value::Mapping(map) = &mut doc {
        map.remove(serde_yaml::Value::String("monitoring".to_string()));
    }
    let text = serde_yaml::to_string(&doc).unwrap_or_default();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{h:016x}")
}

// ---------------------------------------------------------------------------
// Derived plan
// ---------------------------------------------------------------------------

/// An enabled station, resolved and compacted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnabledStation {
    /// Compacted index 0..n_station-1; this is `j`/`k` in `Y_j * conj(Y_k)`.
    pub index: usize,
    /// Index into `Config::stations`.
    pub cfg_index: usize,
    pub name: String,
    pub group_ip: Ipv4Addr,
    pub dest_port: u16,
    pub gpu_id: i32,
    /// CPU this station's receive thread is pinned to.
    pub recv_cpu: u32,
}

/// Everything derived from the raw configuration, in the units the data path
/// actually uses.
#[derive(Debug, Clone)]
pub struct RunPlan {
    pub run_name: String,
    pub output_dir: PathBuf,
    pub dry_run: bool,

    pub n_station: usize,
    pub n_baseline: usize,
    /// `Some((threads, cores))` when there are fewer receive cores than stations,
    /// so threads share cores. Printed at startup: it is the difference between
    /// "each thread owns a core" and "the timing story is a hope".
    pub recv_cpus_shared: Option<(usize, usize)>,
    /// What the resolver decided about pinning, and what it excluded.
    pub cpu_note: Vec<String>,

    /// UDP packets per FFT frame: `2 * n_freq / 4096`.
    pub packets_per_frame: u64,
    /// FFT length in real samples.
    pub fft_len: u64,
    /// Retained bins, `n_freq`. The R2C transform yields `n_freq + 1` and the
    /// Nyquist bin is dropped, so this is also the spectrum length.
    pub n_freq: u32,

    /// Frame rate as an exact rational `sample_rate_hz / (2 * n_freq)`.
    /// Frame rate and bin width are the same number: one frame spans `2*n_freq`
    /// samples, so a block of `n_cum` frames spans `n_cum / bin_width` seconds
    /// exactly. Keeping it rational avoids a rounding decision.
    pub frame_rate_hz: f64,
    pub frame_rate_num: u64,
    pub frame_rate_den: u64,
    pub bin_width_hz: f64,
    pub block_period_s: f64,
    pub block_rate_hz: f64,

    /// Frames one staging slot holds, per station.
    pub frames_per_slot: u32,
    /// Frames per device FFT + correlate launch.
    pub chunk_frames: u32,
    /// Staging slots in the receive ring.
    pub block_slots: u32,
    /// Slots left writable for the receive side; `None` means `block_slots - 1`.
    pub ring_lead: Option<u32>,
    /// Integration rows held in device memory between host transfers.
    pub rows_per_fetch: u32,

    /// int8 bytes of one staging slot, one station.
    pub staging_bytes_per_slot: u64,
    /// int8 bytes of one station's whole staging ring.
    pub staging_bytes_per_station: u64,
    /// int8 bytes of the whole pinned region handed to the GPU side.
    pub staging_total_bytes: u64,
    /// Device bytes of the row stack: rows_per_fetch rows of complex64.
    pub row_stack_bytes: u64,

    /// Spectrum bytes of one integration row, all baselines.
    pub out_bytes_per_row: u64,
    /// Spectrum bytes pulled across PCIe per fetch (rows_per_fetch rows).
    pub fetch_bytes: u64,
    pub out_bytes_per_sec: f64,
    /// Seconds between host transfers, i.e. how often the writer wakes.
    pub fetch_interval_s: f64,

    pub enabled: Vec<EnabledStation>,
    /// Frames per integration row. This is the integration length; the staging
    /// footprint does not depend on it.
    pub n_cum: u32,
}

impl RunPlan {
    /// Compacted index of a station by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.enabled.iter().find(|s| s.name == name).map(|s| s.index)
    }

    /// `(j, k)` for each baseline, `j <= k`, row major. This must agree with
    /// `corr_baseline_pairs()` in the shared object; a test asserts that it does.
    pub fn baseline_pairs(&self) -> Vec<(usize, usize)> {
        let mut v = Vec::with_capacity(self.n_baseline);
        for j in 0..self.n_station {
            for k in j..self.n_station {
                v.push((j, k));
            }
        }
        v
    }

    /// One-line human summary, used by `corr --check` and by the startup log.
    pub fn summary(&self) -> String {
        format!(
            "{} stations -> {} baselines | n_freq={} | row {} frames ({:.3} s) | \
             frame rate {:.4} Hz, bin width {:.4} Hz | staging {:.1} MiB | \
             row stack {:.1} MiB | fetch every {} row(s) = {:.3} s | output {:.3} MB/s",
            self.n_station,
            self.n_baseline,
            self.n_freq,
            self.n_cum,
            self.block_period_s,
            self.frame_rate_hz,
            self.bin_width_hz,
            self.staging_total_bytes as f64 / (1024.0 * 1024.0),
            self.row_stack_bytes as f64 / (1024.0 * 1024.0),
            self.rows_per_fetch,
            self.fetch_interval_s,
            self.out_bytes_per_sec / 1e6,
        )
    }
}

// ---------------------------------------------------------------------------
// Loading and compiling
// ---------------------------------------------------------------------------

/// Default strftime format for the local time in a file set's *file names*:
/// `20260918-144142`.
pub const DEFAULT_FILE_PREFIX_FORMAT: &str = "%Y%m%d-%H%M%S";

/// Default stamp for a run *directory* name: the same instant, in the older
/// underscore layout (`corr_20260918_144142`).
///
/// Separate from the file-name format on purpose: the directory is created once
/// per run and referred to by hand, and renaming it would orphan every path in
/// the project's notes, while the file names inside it are the ones a reader has
/// to parse.
pub const DEFAULT_RUN_NAME_FORMAT: &str = "%Y%m%d_%H%M%S";

/// Default address for the status endpoint: the loopback, so that turning it on
/// exposes a run to this host and not to the network.
///
/// 3081 rather than 3080, which this machine's DSH web GUI already holds.
pub const DEFAULT_HTTP_BIND: &str = "127.0.0.1:3081";

/// Current UTC time as `YYYYmmdd_HHMMSS`.
///
/// Goes through libc rather than pulling in a date crate: the host side has no
/// other use for calendar arithmetic, and a UTC stamp computed from
/// `gmtime_r` cannot be affected by the machine's timezone the way a
/// hand-rolled formatter could. It is now only used for *reporting* the UTC
/// time next to the local one; everything that names a file is local
/// (see [`local_stamp_at`] and docs/DESIGN-daily-files.md).
pub fn utc_stamp() -> String {
    stamp_at(crate::sched::system_ms(), "%Y%m%d_%H%M%S", true)
}

/// Format a wall-clock instant in the machine's local time zone.
///
/// `localtime_r`, not `gmtime_r`: a daily file set is named for the *local* day
/// (docs/DESIGN-daily-files.md §1), and "local" is whatever `TZ` and
/// `/etc/localtime` say at the moment of the call -- so a timezone change takes
/// effect on the next row rather than needing a restart. Being a call rather than
/// a cached offset is also what makes a daylight-saving day (23 or 25 hours) come
/// out right.
pub fn local_stamp_at(ms: i64, format: &str) -> String {
    stamp_at(ms, format, false)
}

/// The local calendar day of `ms`, as `YYYYmmdd`.
///
/// The day key that decides which set a row belongs to. Comparing these strings
/// compares days, because the format is fixed-width and big-endian.
pub fn local_day_at(ms: i64) -> String {
    stamp_at(ms, "%Y%m%d", false)
}

/// The `TZ` environment variable as the process sees it, for metadata. Empty when
/// unset, which means libc fell back to `/etc/localtime`.
pub fn local_tz_name() -> String {
    std::env::var("TZ").unwrap_or_default()
}

/// Seconds east of UTC that local time was at `ms`. Recorded per set so a reader
/// can tell what the local stamp in the file names meant without guessing.
pub fn local_utc_offset_s(ms: i64) -> i64 {
    // SAFETY: localtime_r writes only into the provided tm.
    unsafe {
        let secs = ms.div_euclid(1000) as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&secs, &mut tm).is_null() {
            return 0;
        }
        tm.tm_gmtoff as i64
    }
}

/// Shared body of the two stamps above. `utc` selects `gmtime_r`.
fn stamp_at(ms: i64, format: &str, utc: bool) -> String {
    use std::ffi::CStr;
    // SAFETY: both conversions take a valid time_t pointer and a properly sized
    // tm; strftime writes only into the provided buffer, and the buffer is long
    // enough that strftime returning 0 means "format failed", not "too long".
    unsafe {
        let secs = ms.div_euclid(1000) as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        let ok = if utc {
            !libc::gmtime_r(&secs, &mut tm).is_null()
        } else {
            !libc::localtime_r(&secs, &mut tm).is_null()
        };
        if !ok {
            return "unknown_time".to_string();
        }
        let mut buf = [0 as libc::c_char; 128];
        let mut fmt: Vec<u8> = format.as_bytes().to_vec();
        fmt.push(0);
        let n = libc::strftime(
            buf.as_mut_ptr(),
            buf.len(),
            fmt.as_ptr() as *const libc::c_char,
            &tm,
        );
        if n == 0 {
            return "unknown_time".to_string();
        }
        CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
    }
}

impl Config {
    /// Fill in a timestamped run name if the configuration left it empty.
    ///
    /// Local time, like the file-set prefixes inside it: a directory named in UTC
    /// containing files named in local time is exactly the kind of two-time-base
    /// mix-up that makes the daily data hard to reconcile later.
    ///
    /// Separate from [`Config::compile`] so that compiling stays pure, and
    /// separate from validation so that `--check` on an unmodified file does not
    /// silently invent a name.
    pub fn resolve_run_name(&mut self) {
        if self.run.name.as_deref().map_or(true, str::is_empty) {
            self.run.name = Some(format!(
                "corr_{}",
                local_stamp_at(crate::sched::system_ms(), DEFAULT_RUN_NAME_FORMAT)
            ));
        }
    }

    pub fn from_yaml_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config {}", path.display()))?;
        Self::from_yaml_str(&text).with_context(|| format!("in config {}", path.display()))
    }

    pub fn from_yaml_str(text: &str) -> Result<Self> {
        let cfg: Config = serde_yaml::from_str(text).context("YAML does not match the schema")?;
        Ok(cfg)
    }

    /// Validate every cross-field invariant. Pure: touches no sockets, no GPU
    /// and no filesystem, so it is cheap to unit test.
    pub fn validate(&self) -> Result<()> {
        if self.version != CONFIG_VERSION {
            bail!(
                "config version {} is not supported by this build (expected {})",
                self.version,
                CONFIG_VERSION
            );
        }

        // --- FFT ---------------------------------------------------------
        if self.fft.n_freq == 0 {
            bail!("fft.n_freq must be >= 1");
        }
        if self.fft.n_freq % 2048 != 0 {
            bail!(
                "fft.n_freq={} is not a multiple of 2048: a frame of 2*n_freq samples \
                 would not be a whole number of 4096-sample UDP packets, so frames \
                 could not be assembled from packet counters",
                self.fft.n_freq
            );
        }
        if self.fft.sample_rate_hz == 0 {
            bail!("fft.sample_rate_hz must be > 0");
        }
        if self.fft.window != "rectangular" {
            bail!(
                "fft.window={:?} is not implemented; only \"rectangular\" is supported \
                 (the algorithmic decision is a rectangular window with no correction)",
                self.fft.window
            );
        }

        // --- integration -------------------------------------------------
        if self.integration.n_cum == 0 {
            bail!("integration.n_cum must be >= 1");
        }
        if let Some(f) = self.integration.start_frame_id {
            if f % self.integration.n_cum as u64 != 0 {
                bail!(
                    "integration.start_frame_id={} is not a multiple of n_cum={}; \
                     block boundaries must sit on the global frame grid",
                    f,
                    self.integration.n_cum
                );
            }
        }
        if self.integration.block_timeout_ms == 0 {
            bail!(
                "integration.block_timeout_ms must be > 0; without a timeout a single \
                 dead station would stall the watermark for ever"
            );
        }
        if self.integration.zero_fill_limit_frames != 0 {
            // Below one row the cap would fire before a single degraded row could
            // be produced, which is not a useful setting.
            let min = self.integration.n_cum as u64;
            if self.integration.zero_fill_limit_frames < min {
                bail!(
                    "integration.zero_fill_limit_frames={} is below one row ({} frames); \
                     set it to 0 to disable, or to at least one row",
                    self.integration.zero_fill_limit_frames,
                    min
                );
            }
        }

        // --- gpu ---------------------------------------------------------
        if self.gpu.devices.is_empty() {
            bail!("gpu.devices must list at least one device");
        }
        if self.gpu.split == SplitMode::PairCovering && self.gpu.devices.len() < 2 {
            bail!(
                "gpu.split: pair_covering needs at least 2 devices; with one device the \
                 range split is already optimal"
            );
        }
        let mut seen_dev = BTreeSet::new();
        for d in &self.gpu.devices {
            if *d < 0 {
                bail!("gpu.devices contains a negative device id {}", d);
            }
            if !seen_dev.insert(*d) {
                bail!("gpu.devices lists device {} twice", d);
            }
        }
        if self.gpu.frames_per_slot == 0 {
            bail!("gpu.frames_per_slot must be >= 1");
        }
        if self.gpu.chunk_frames == 0 {
            bail!("gpu.chunk_frames must be >= 1");
        }
        if self.gpu.chunk_frames > self.gpu.frames_per_slot {
            bail!(
                "gpu.chunk_frames={} exceeds gpu.frames_per_slot={}; a chunk is the \
                 device-side work unit and cannot span staging slots",
                self.gpu.chunk_frames,
                self.gpu.frames_per_slot
            );
        }
        if self.gpu.rows_per_fetch == 0 {
            bail!("gpu.rows_per_fetch must be >= 1");
        }
        // A frame's position inside a staging slot is `frame_id % frames_per_slot`,
        // and the device side reads a slot starting from its first frame. A row
        // that began part way through a slot would therefore have its earliest
        // frames read from the wrong offset -- a silent numerical error, so it is
        // refused at startup instead.
        if self.integration.n_cum as u64 % self.gpu.frames_per_slot as u64 != 0 {
            bail!(
                "integration.n_cum={} is not a multiple of gpu.frames_per_slot={}; one \
                 row must be a whole number of staging slots ({} slots per row is not \
                 an integer). Pick n_cum = frames_per_slot * k, e.g. 97792 = 256 * 382.",
                self.integration.n_cum,
                self.gpu.frames_per_slot,
                self.integration.n_cum as f64 / self.gpu.frames_per_slot as f64
            );
        }
        if let Some(lead) = self.gpu.ring_lead {
            if lead == 0 {
                bail!("gpu.ring_lead=0 leaves the receive side nothing to read ahead into");
            }
            if lead >= self.gpu.block_slots {
                bail!(
                    "gpu.ring_lead={} leaves the pipeline no slot to copy (block_slots={}); \
                     at least one slot must be in flight",
                    lead,
                    self.gpu.block_slots
                );
            }
        }
        if self.gpu.block_slots < 2 {
            bail!(
                "gpu.block_slots={} must be >= 2: with a single slot the receive \
                 threads would have nowhere to write while the GPU reads",
                self.gpu.block_slots
            );
        }

        // --- receiver ----------------------------------------------------
        if self.receiver.recv_buffer_bytes == 0 {
            bail!("receiver.recv_buffer_bytes must be > 0");
        }
        if self.receiver.recvmmsg_batch == 0 {
            bail!("receiver.recvmmsg_batch must be >= 1");
        }
        if self.receiver.recv_timeout_ms == 0 {
            bail!(
                "receiver.recv_timeout_ms must be > 0; a receive thread that blocks \
                 for ever cannot honour the scheduler's stall recovery"
            );
        }
        if self.receiver.interface_ip.is_unspecified() {
            bail!("receiver.interface_ip must be set; it is the IP_ADD_MEMBERSHIP interface");
        }

        // --- monitoring ---------------------------------------------------
        // Only checked when the endpoint is on: a `bind` nobody asked to listen on
        // is not a reason to refuse a run, the same way a junk prefix format is not
        // when daily files are off.
        if self.monitoring.http.enabled
            && self.monitoring.http.bind.parse::<std::net::SocketAddr>().is_err()
        {
            bail!(
                "monitoring.http.bind={:?} is not an `IP:port` address (the default is \
                 {DEFAULT_HTTP_BIND}); a host name would have to be resolved, and a run should \
                 not fail later because DNS answered differently",
                self.monitoring.http.bind
            );
        }

        // --- storage -----------------------------------------------------
        if self.storage.writers == 0 {
            bail!("storage.writers must be >= 1");
        }
        if self.storage.queue_bytes == 0 {
            bail!("storage.queue_bytes must be > 0");
        }
        if self.storage.fsync_every_rows == 0 {
            bail!("storage.fsync_every_rows must be >= 1");
        }
        // Only checked when the feature is on: a junk format in a config that has
        // daily files switched off is not a reason to refuse the run.
        if self.storage.file_prefix.contains('/') {
            bail!(
                "storage.file_prefix={:?} contains a path separator; it is put verbatim at the \
                 front of every file name, so a directory belongs in run.output_dir",
                self.storage.file_prefix
            );
        }
        if self.storage.daily_files {
            let fmt = &self.storage.file_prefix_format;
            if fmt.is_empty() {
                bail!("storage.file_prefix_format is empty; the file sets would have no name");
            }
            if fmt.contains('/') {
                bail!(
                    "storage.file_prefix_format={fmt:?} contains a path separator; the prefix \
                     goes in front of a file name, not in front of a path"
                );
            }
            if !fmt.contains('%') {
                bail!(
                    "storage.file_prefix_format={fmt:?} has no strftime directive, so every \
                     day would name its files the same and the second day would collide with \
                     the first"
                );
            }
            // strftime returns 0 for a format it cannot produce; catching that here
            // means a bad format refuses to start rather than failing at midnight.
            let probe = local_stamp_at(0, fmt);
            if probe == "unknown_time" || probe.is_empty() {
                bail!("storage.file_prefix_format={fmt:?} does not format to anything");
            }
        }

        // --- stations ----------------------------------------------------
        if self.stations.is_empty() {
            bail!("stations is empty");
        }

        let mut names = BTreeSet::new();
        let mut orders: BTreeMap<u32, &str> = BTreeMap::new();
        let mut groups: BTreeMap<(Ipv4Addr, u16), &str> = BTreeMap::new();

        for s in &self.stations {
            if s.name.is_empty() {
                bail!("a station has an empty name");
            }
            if !names.insert(s.name.as_str()) {
                bail!("duplicate station name {:?}", s.name);
            }
            if let Some(other) = orders.insert(s.order, &s.name) {
                bail!(
                    "stations {:?} and {:?} both use order {}; order must be unique",
                    other,
                    s.name,
                    s.order
                );
            }
            if s.dest_port == 0 {
                bail!("station {:?} has dest_port 0", s.name);
            }
            if !s.interface.is_empty() && s.interface != self.receiver.interface {
                bail!(
                    "station {:?} names interface {:?} but receiver.interface is {:?}; \
                     this build receives everything on one NIC",
                    s.name,
                    s.interface,
                    self.receiver.interface
                );
            }
            if s.enabled {
                if let Some(other) = groups.insert((s.group_ip, s.dest_port), &s.name) {
                    bail!(
                        "enabled stations {:?} and {:?} share the same group:port {}:{}",
                        other,
                        s.name,
                        s.group_ip,
                        s.dest_port
                    );
                }
                // gpu_id is *advisory*: it says which card should receive this
                // station's stream, and it is deliberately allowed to name a card
                // that this run does not use. That is what makes falling back to
                // one card a one-line change -- `gpu.devices: [0]` -- instead of
                // rewriting every station's assignment, which is also how the
                // production layout stays documented in the table while a
                // single-card test runs. `gpu.devices` is what the engine spans;
                // stations whose hint is outside it are reported at startup rather
                // than silently obeyed or silently dropped.
                if s.gpu_id < 0 {
                    bail!(
                        "enabled station {:?} has a negative gpu_id {}",
                        s.name,
                        s.gpu_id
                    );
                }
            }
        }

        // --- cpu ---------------------------------------------------------
        let n_enabled = self.stations.iter().filter(|s| s.enabled).count();
        if n_enabled == 0 {
            bail!("no station has enabled: true");
        }
        // One core per receive thread is the design's assumption, but it is not
        // available on every machine: this box has 32 cores and the 40-station
        // target needs 40 receive threads. Refusing to compile made "40 stations
        // on this hardware" impossible to even try, which is worse than trying it
        // with a warning -- the whole point of the pinning is *jitter margin*, and
        // whether sharing cores costs that margin is an empirical question (see
        // §6.13: the ring lead is what absorbs jitter, and it is configurable).
        // So sharing is allowed, round-robin, and reported loudly at startup.
        // With a derived layout there is nothing to check here: the resolver runs
        // in compile(), where /sys is readable and a clear error can name the pool
        // it actually found.
        if self.cpu.auto_layout {
            return Ok(());
        }
        if self.cpu.recv_cpus.is_empty() {
            bail!(
                "cpu.recv_cpus is empty; either list one core per station or set \
                 cpu.auto_layout: true to derive the layout (see avoid_nic_interfaces)"
            );
        }
        if self.cpu.writer_cpus.len() < self.storage.writers {
            bail!(
                "cpu.writer_cpus has {} entries but storage.writers={}",
                self.cpu.writer_cpus.len(),
                self.storage.writers
            );
        }

        // Every core actually handed out must be distinct, otherwise two busy
        // spinning threads fight over one core and the whole timing story is
        // fiction. Only the used prefix of recv_cpus/writer_cpus matters.
        let mut used: BTreeMap<u32, String> = BTreeMap::new();
        let claim = |cpu: u32, who: String, used: &mut BTreeMap<u32, String>| -> Result<()> {
            if let Some(other) = used.insert(cpu, who.clone()) {
                bail!("CPU {} is assigned to both {} and {}", cpu, other, who);
            }
            Ok(())
        };
        for i in 0..n_enabled {
            let cpu = self.cpu.recv_cpus[i % self.cpu.recv_cpus.len()];
            // Deliberate sharing is allowed above; the check that matters is that
            // no receive core is also the pipeline's, a writer's or the monitor's.
            let who = format!("recv[{i}]");
            match used.get(&cpu) {
                Some(other) if other.starts_with("recv[") => {}
                Some(other) => bail!("CPU {} is assigned to both {} and {}", cpu, other, who),
                None => {
                    used.insert(cpu, who);
                }
            }
        }
        claim(self.cpu.pipeline_cpu, "pipeline".to_string(), &mut used)?;
        for i in 0..self.storage.writers {
            claim(self.cpu.writer_cpus[i], format!("writer[{i}]"), &mut used)?;
        }
        claim(self.cpu.monitor_cpu, "monitor".to_string(), &mut used)?;

        Ok(())
    }

    /// Resolve the CPU assignment, either from the explicit lists or from the
    /// policy (`avoid_nic_interfaces` + `auto_layout`).
    ///
    /// Returns the layout and human-readable notes: what was excluded and which
    /// cores ended up doing what. The notes are printed at startup because a
    /// pinning decision that is invisible is a pinning decision that will be
    /// wrong on the next machine without anyone noticing.
    fn resolve_cpu_layout(&self, n_station: usize) -> Result<(CpuLayout, Vec<String>)> {
        let mut excluded: Vec<u32> = Vec::new();
        let mut notes = Vec::new();
        for iface in &self.cpu.avoid_nic_interfaces {
            match nic_node_cpus(iface) {
                Some((node, cpus)) => {
                    notes.push(format!(
                        "{iface} is on NUMA node {node}: cores {cpus:?} kept free of receive threads"
                    ));
                    excluded.extend(cpus);
                }
                None => notes.push(format!(
                    "{iface}: no NUMA node to avoid (virtual interface, or /sys is unreadable)"
                )),
            }
        }

        let explicit = CpuLayout {
            recv_cpus: self.cpu.recv_cpus.clone(),
            pipeline_cpu: self.cpu.pipeline_cpu,
            writer_cpus: self.cpu.writer_cpus.clone(),
            monitor_cpu: self.cpu.monitor_cpu,
        };
        if !self.cpu.auto_layout {
            let clash: Vec<&u32> = explicit
                .recv_cpus
                .iter()
                .filter(|c| excluded.contains(c))
                .collect();
            if !clash.is_empty() {
                notes.push(format!(
                    "WARNING: receive cores {clash:?} are on a NIC's NUMA node; that is the \
                     layout measured to cost 7x more fabricated frames (HANDOFF 6.16)"
                ));
            }
            return Ok((explicit, notes));
        }

        let online = online_cpus().with_context(|| "reading /sys/devices/system/cpu/online")?;
        let pool: Vec<u32> = online.into_iter().filter(|c| !excluded.contains(c)).collect();
        let writers = self.storage.writers;
        let layout = layout_from_pool(&pool, n_station, writers).with_context(|| {
            format!(
                "cpu.auto_layout needs {} cores outside the avoided NUMA nodes ({} stations + \
                 {} writers + pipeline + monitor), but only {} are available: {pool:?}",
                n_station + writers + 2,
                n_station,
                writers,
                pool.len()
            )
        })?;
        notes.push(format!(
            "auto layout: receive on {:?}, pipeline {}, writers {:?}, monitor {}",
            layout.recv_cpus, layout.pipeline_cpu, layout.writer_cpus, layout.monitor_cpu
        ));
        Ok((layout, notes))
    }

    /// Validate and derive. Fails on the first inconsistency.
    pub fn compile(&self) -> Result<RunPlan> {
        self.validate()?;

        let n_station = self.stations.iter().filter(|s| s.enabled).count();
        let n_baseline = n_station * (n_station + 1) / 2;
        let (cpu, cpu_notes) = self.resolve_cpu_layout(n_station)?;

        // Compaction by `order` is what makes "add a station at the end without
        // renumbering the others" possible, and it is the reason baseline index
        // is not simply derived from the YAML line order.
        let mut enabled_src: Vec<&StationCfg> =
            self.stations.iter().filter(|s| s.enabled).collect();
        enabled_src.sort_by_key(|s| s.order);

        let mut enabled = Vec::with_capacity(n_station);
        for (index, s) in enabled_src.iter().enumerate() {
            enabled.push(EnabledStation {
                index,
                cfg_index: self
                    .stations
                    .iter()
                    .position(|o| o.name == s.name)
                    .expect("station came from stations"),
                name: s.name.clone(),
                group_ip: s.group_ip,
                dest_port: s.dest_port,
                gpu_id: s.gpu_id,
                recv_cpu: cpu.recv_cpus[index % cpu.recv_cpus.len()],
            });
        }

        let n_freq = self.fft.n_freq as u64;
        let fft_len = 2 * n_freq;
        let packets_per_frame = fft_len / SAMPLES_PER_PACKET;

        let frame_rate_num = self.fft.sample_rate_hz;
        let frame_rate_den = fft_len;
        let frame_rate_hz = frame_rate_num as f64 / frame_rate_den as f64;
        // One frame spans 2*n_freq samples, so frame rate and bin width coincide.
        let bin_width_hz = frame_rate_hz;

        let n_cum = self.integration.n_cum as u64;
        let block_period_s = n_cum as f64 / frame_rate_hz;
        let block_rate_hz = 1.0 / block_period_s;

        // One bit per frame of a slot for the validity mask, rounded up to whole
        // bytes: the GPU side appends this to each station's samples (see
        // `corr_staging` in corr_cuda.h), so the plan's staging figure has to
        // include it or the two sides disagree at `--check`.
        let mask_bytes_per_slot = (self.gpu.frames_per_slot as u64).div_ceil(8);
        let staging_bytes_per_slot =
            self.gpu.frames_per_slot as u64 * fft_len + mask_bytes_per_slot;
        let staging_bytes_per_station = staging_bytes_per_slot * self.gpu.block_slots as u64;
        let staging_total_bytes = staging_bytes_per_station * n_station as u64;
        let row_stack_bytes = n_baseline as u64 * n_freq * 8 * self.gpu.rows_per_fetch as u64;

        let out_bytes_per_row = n_baseline as u64 * n_freq * 8;
        let fetch_bytes = out_bytes_per_row * self.gpu.rows_per_fetch as u64;
        let out_bytes_per_sec = out_bytes_per_row as f64 * block_rate_hz;
        let fetch_interval_s = 1.0 / (block_rate_hz / self.gpu.rows_per_fetch as f64);

        let run_name = match &self.run.name {
            Some(n) if !n.is_empty() => n.clone(),
            // `resolve_run_name` normally fills this in before compile; the
            // fallback keeps `compile` total for callers that only validate.
            _ => "corr_run".to_string(),
        };

        Ok(RunPlan {
            run_name,
            output_dir: self.run.output_dir.clone(),
            dry_run: self.run.dry_run,
            n_station,
            n_baseline,
            packets_per_frame,
            fft_len,
            n_freq: self.fft.n_freq,
            frame_rate_hz,
            frame_rate_num,
            frame_rate_den,
            bin_width_hz,
            block_period_s,
            block_rate_hz,
            frames_per_slot: self.gpu.frames_per_slot,
            chunk_frames: self.gpu.chunk_frames,
            block_slots: self.gpu.block_slots,
            ring_lead: self.gpu.ring_lead,
            rows_per_fetch: self.gpu.rows_per_fetch,
            staging_bytes_per_slot,
            staging_bytes_per_station,
            staging_total_bytes,
            row_stack_bytes,
            out_bytes_per_row,
            fetch_bytes,
            out_bytes_per_sec,
            fetch_interval_s,
            enabled,
            n_cum: self.integration.n_cum,
            recv_cpus_shared: if cpu.recv_cpus.len() < n_station {
                Some((n_station, cpu.recv_cpus.len()))
            } else {
                None
            },
            cpu_note: cpu_notes,
        })
    }
}

impl fmt::Display for RunPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for s in &self.enabled {
            writeln!(
                f,
                "  [{:2}] {:<5} {:>15}:{:<6} gpu={} cpu={}",
                s.index, s.name, s.group_ip, s.dest_port, s.gpu_id, s.recv_cpu
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The config that actually ships, so a change to it is caught here.
    fn shipped_yaml() -> String {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("config/correlator.yaml");
        std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
    }

    /// Watching a run is not part of its geometry: turning the status endpoint on,
    /// or changing how often the line is printed, must not make the day's existing
    /// set unresumable. Not hypothetical -- adding `monitoring.http` to a running
    /// profile did exactly that, and the refusal named a hash mismatch.
    #[test]
    fn the_config_hash_ignores_monitoring_and_still_notices_geometry() {
        let mut cfg = Config::from_yaml_str(&shipped_yaml()).unwrap();
        let base = config_hash(&cfg);

        cfg.monitoring.interval_ms = 250;
        cfg.monitoring.http.enabled = true;
        cfg.monitoring.http.bind = "0.0.0.0:9999".to_string();
        assert_eq!(
            config_hash(&cfg),
            base,
            "monitoring is observation, not geometry"
        );

        cfg.integration.n_cum += 1;
        assert_ne!(
            config_hash(&cfg),
            base,
            "a changed integration length is geometry"
        );
        cfg.integration.n_cum -= 1;
        assert_eq!(config_hash(&cfg), base, "and undoing it gives the hash back");
    }

    #[test]
    fn shipped_config_parses_and_validates() {
        let cfg = Config::from_yaml_str(&shipped_yaml()).expect("shipped config must parse");
        cfg.validate().expect("shipped config must validate");
        assert_eq!(cfg.stations.len(), 24, "all 24 stations belong in the file");
    }

    #[test]
    fn shipped_config_derives_the_expected_numbers() {
        let cfg = Config::from_yaml_str(&shipped_yaml()).unwrap();
        let p = cfg.compile().unwrap();

        assert_eq!(p.n_station, 4);
        assert_eq!(p.n_baseline, 10);
        assert_eq!(p.packets_per_frame, 2, "n_freq=4096 spans two 4096-sample packets");
        assert_eq!(p.fft_len, 8192);

        // 800e6 / 8192 = 97656.25 exactly.
        assert!((p.frame_rate_hz - 97656.25).abs() < 1e-9);
        assert!((p.bin_width_hz - 97656.25).abs() < 1e-9);

        // 10 baselines * 4096 bins * 8 bytes per row.
        assert_eq!(p.out_bytes_per_row, 10 * 4096 * 8);

        let names: Vec<&str> = p.enabled.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["W17", "S01", "S06", "N04"]);
        assert!(p.enabled.iter().all(|s| s.gpu_id == 0));
    }

    #[test]
    fn baseline_pairs_match_the_row_major_definition() {
        let cfg = Config::from_yaml_str(&shipped_yaml()).unwrap();
        let p = cfg.compile().unwrap();
        assert_eq!(
            p.baseline_pairs(),
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 2),
                (2, 3),
                (3, 3)
            ]
        );
    }

    #[test]
    fn scaling_to_15_and_24_stations_gives_the_expected_rates() {
        let mut cfg = Config::from_yaml_str(&shipped_yaml()).unwrap();

        // Enable the first 15 stations by order.
        let mut by_order: Vec<usize> = (0..cfg.stations.len()).collect();
        by_order.sort_by_key(|&i| cfg.stations[i].order);
        for &i in by_order.iter().take(15) {
            cfg.stations[i].enabled = true;
        }
        // 15 receive threads need 15 distinct cores.
        cfg.cpu.recv_cpus = (0..24).collect();
        let p15 = cfg.compile().unwrap();
        assert_eq!(p15.n_station, 15);
        assert_eq!(p15.n_baseline, 120);
        // One row is 120 * 4096 * 8 bytes, independent of the integration length.
        assert_eq!(p15.out_bytes_per_row, 120 * 4096 * 8);
        // With a 1 second integration the writer wakes about once a second, so
        // the sustained output rate is the row divided by the row period.
        assert!((p15.out_bytes_per_sec / 1e6 - 3.93).abs() < 0.05);
        // The fetch interval is defined as rows_per_fetch row periods, so assert
        // the relationship rather than a magic number: 1 is not expressible as a
        // whole number of frames, and the row length must also be a whole number
        // of staging slots, so the shipped row is 1.00139 s rather than 1.00000 s.
        let row_period = p15.n_cum as f64 / p15.frame_rate_hz;
        assert!((p15.fetch_interval_s - row_period * p15.rows_per_fetch as f64).abs() < 1e-9);
        assert!(
            (row_period - 1.0).abs() < 0.01,
            "the default row should be about a second, got {row_period}"
        );

        for s in cfg.stations.iter_mut() {
            s.enabled = true;
        }
        let p24 = cfg.compile().unwrap();
        assert_eq!(p24.n_station, 24);
        assert_eq!(p24.n_baseline, 300);
        assert_eq!(p24.out_bytes_per_row, 300 * 4096 * 8);
        assert!((p24.out_bytes_per_sec / 1e6 - 9.83).abs() < 0.05);
    }

    #[test]
    fn a_short_integration_can_keep_a_one_second_fetch_cadence() {
        // The point of rows_per_fetch: a 10 ms integration must not turn into
        // 100 host transfers and 100 writer wake-ups per second. Raising
        // rows_per_fetch restores the cadence without touching the integration.
        let mut cfg = Config::from_yaml_str(&shipped_yaml()).unwrap();
        // 4 slots of 256 frames is 10.49 ms, which is as close to 10 ms as the
        // whole-slot rule allows at this slot size.
        cfg.integration.n_cum = 4 * cfg.gpu.frames_per_slot;
        cfg.gpu.rows_per_fetch = 96; // 96 x 10.49 ms ~= 1.007 s
        let p = cfg.compile().unwrap();

        assert!((p.block_period_s - 0.0105).abs() < 1e-3, "10 ms integration");
        assert!(
            (p.fetch_interval_s - 1.0).abs() < 0.02,
            "96 rows of 10.5 ms should be fetched about once a second, got {}",
            p.fetch_interval_s
        );
        assert_eq!(p.fetch_bytes, p.out_bytes_per_row * 96);

        // Now hold rows_per_fetch fixed and change only the integration length.
        // Neither the pinned staging nor the device row stack may move: they are
        // functions of frames_per_slot and rows_per_fetch respectively.
        let mut long = Config::from_yaml_str(&shipped_yaml()).unwrap();
        long.integration.n_cum = cfg.integration.n_cum * 10;
        long.gpu.rows_per_fetch = cfg.gpu.rows_per_fetch;
        let pl = long.compile().unwrap();
        assert_eq!(
            pl.staging_total_bytes, p.staging_total_bytes,
            "lengthening the integration must not touch the pinned staging"
        );
        assert_eq!(
            pl.row_stack_bytes, p.row_stack_bytes,
            "the device row stack depends on rows_per_fetch, not on the row length"
        );
        assert_eq!(pl.fetch_bytes, p.fetch_bytes);

        // ... and it is rows_per_fetch that does move the stack.
        let mut one = Config::from_yaml_str(&shipped_yaml()).unwrap();
        one.gpu.rows_per_fetch = 1;
        let p1 = one.compile().unwrap();
        assert_eq!(p1.row_stack_bytes * 96, p.row_stack_bytes);
    }

    #[test]
    fn rejects_a_chunk_larger_than_a_staging_slot() {
        let mut c = base_config();
        c.gpu.chunk_frames = c.gpu.frames_per_slot + 1;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("chunk_frames"), "unexpected error: {err}");

        let mut c = base_config();
        c.gpu.rows_per_fetch = 0;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("rows_per_fetch"), "unexpected error: {err}");
    }

    fn base_config() -> Config {
        Config::from_yaml_str(&shipped_yaml()).unwrap()
    }

    #[test]
    fn rejects_n_freq_that_is_not_a_whole_number_of_packets() {
        let mut c = base_config();
        c.fft.n_freq = 4096 + 1024;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("multiple of 2048"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_duplicate_order_and_duplicate_name() {
        let mut c = base_config();
        c.stations[1].order = c.stations[0].order;
        assert!(c.validate().unwrap_err().to_string().contains("order"));

        let mut c = base_config();
        c.stations[1].name = c.stations[0].name.clone();
        assert!(c.validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate station name"));
    }

    #[test]
    fn rejects_duplicate_group_port_between_enabled_stations() {
        let mut c = base_config();
        c.stations[1].group_ip = c.stations[0].group_ip;
        c.stations[1].dest_port = c.stations[0].dest_port;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("same group:port"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_cpu_overlap_including_pipeline_and_writers() {
        let mut c = base_config();
        c.cpu.pipeline_cpu = c.cpu.recv_cpus[0];
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("assigned to both"), "unexpected error: {err}");
    }

    #[test]
    fn cpulists_parse_the_way_the_kernel_writes_them() {
        assert_eq!(parse_cpulist("0-3,8-11,20"), vec![0, 1, 2, 3, 8, 9, 10, 11, 20]);
        assert_eq!(parse_cpulist(" 5\n"), vec![5]);
        assert_eq!(parse_cpulist(""), Vec::<u32>::new());
        assert_eq!(parse_cpulist("junk,7"), vec![7]);
    }

    #[test]
    fn auto_layout_takes_receive_cores_from_the_head_and_the_rest_from_the_tail() {
        // This machine's shape: node 0 = cores 0-3 holds the NIC, so the pool is
        // 4-31; 24 stations and 2 writers must reproduce the layout that was
        // measured to be 7x better than sharing node 0.
        let pool: Vec<u32> = (4..32).collect();
        let l = layout_from_pool(&pool, 24, 2).expect("24+4 fits in 28");
        assert_eq!(l.recv_cpus, (4..28).collect::<Vec<u32>>());
        assert_eq!(l.pipeline_cpu, 28);
        assert_eq!(l.writer_cpus, vec![29, 30]);
        assert_eq!(l.monitor_cpu, 31);
        // The four non-receive cores stay contiguous, so they share a node.
        assert!(!l.recv_cpus.contains(&l.pipeline_cpu));
        assert!(!l.recv_cpus.contains(&l.monitor_cpu));
    }

    #[test]
    fn auto_layout_says_no_when_the_pool_is_too_small() {
        let pool: Vec<u32> = (4..24).collect();
        assert!(layout_from_pool(&pool, 24, 2).is_none());
    }

    #[test]
    fn too_few_receive_cores_is_allowed_but_reported() {
        // This used to be a hard error ("one dedicated core per receive thread is
        // the whole point of the pinning"), which made the 40-station target
        // impossible to even attempt on a 32-core machine. Sharing is now legal
        // and visible: the plan carries it, and the run prints it.
        let mut c = base_config();
        c.cpu.recv_cpus = vec![0, 1, 2];
        let plan = c.compile().expect("sharing cores must compile");
        assert_eq!(plan.recv_cpus_shared, Some((plan.n_station, 3)));
        // Sharing must not be silent: two stations land on the same core.
        let assign = |p: &RunPlan| p.enabled.iter().map(|s| s.recv_cpu).collect::<Vec<_>>();
        assert_eq!(assign(&plan)[0], assign(&plan)[3]);
    }

    #[test]
    fn a_receive_core_may_not_be_the_pipeline_core() {
        let mut c = base_config();
        c.cpu.recv_cpus = vec![0, 1, 2];
        c.cpu.pipeline_cpu = 2;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("assigned to both"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_start_frame_id_off_the_block_grid() {
        let mut c = base_config();
        c.integration.start_frame_id = Some(1500);
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("multiple of n_cum"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_single_block_slot() {
        let mut c = base_config();
        c.gpu.block_slots = 1;
        assert!(c.validate().unwrap_err().to_string().contains("block_slots"));
    }

    #[test]
    fn rejects_unknown_fields_rather_than_ignoring_them() {
        let yaml = shipped_yaml().replace("n_freq: 4096", "n_freq: 4096\n  n_freq_typo: 1");
        let err = Config::from_yaml_str(&yaml).unwrap_err().to_string();
        assert!(err.contains("YAML does not match"), "unexpected error: {err}");
    }

    #[test]
    fn a_station_may_name_a_gpu_this_run_does_not_use() {
        // The single-card fallback depends on this: `gpu.devices: [0]` with a
        // station table that still records the production assignment must
        // compile, because rewriting 24 station lines to test on one card is
        // exactly how a fallback stops being used.
        let mut c = base_config();
        c.stations[0].gpu_id = 3;
        c.validate().expect("an advisory gpu_id outside gpu.devices must be allowed");
        assert_eq!(c.gpu.devices, vec![0]);

        // A negative one is still a typo, not a hint.
        let mut c = base_config();
        c.stations[0].gpu_id = -1;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("negative gpu_id"), "unexpected error: {err}");
    }

    #[test]
    fn disabled_stations_are_ignored_by_numbering() {
        let mut c = base_config();
        // Take the station with the largest order out of the enabled set.
        c.stations.sort_by_key(|s| s.order);
        let last = c.stations.len() - 1;
        c.stations[last].enabled = false;
        c.stations[last].order = 100;
        let p = c.compile().unwrap();
        assert_eq!(p.n_station, 4);
        assert_eq!(p.index_of("W17"), Some(0));
        assert_eq!(p.index_of("N04"), Some(3));
        assert_eq!(p.index_of("N11"), None);
    }
}
