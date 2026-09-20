//! Output: one raw spectrum file per baseline, plus sidecar files, in **sets**.
//!
//! Layout of a set, with `<P>` = `storage.file_prefix` (empty by default) and
//! `<T>` = the local time of the set's first row, formatted with
//! `storage.file_prefix_format` (`20260918-144142` by default):
//!
//! ```text
//!   <P><station_j><station_k>-<T>.bin   complex64[ntime][n_freq], one row per integration
//!   ...                                 one per baseline, j <= k, station names
//!   <P><T>-time.bin                     int64 little-endian Unix milliseconds, one per row
//!   <P><T>-time.txt                     local "YYYY-mm-DD HH:MM:SS.mmm" per line
//!   <P><T>-blocks.csv                   row_index, station, frame range, valid frames
//!   <P><T>-metadata.yaml                everything needed to read the raw files
//! ```
//!
//! Nothing is inserted between the halves, so an empty prefix reads
//! `S01W17-20260918-144142.bin` and `20260918-144142-time.txt`, while
//! `file_prefix: "corr24-"` reads `corr24-S01W17-20260918-144142.bin` and
//! `corr24-20260918-144142-time.txt`. A file is identified by the station pair it
//! correlates and the time it starts at, which is what a person looking at the
//! directory wants to know; the tag `<P><T>` is the set's identity and is what
//! `--set` takes.
//!
//! A **set** is the four sidecars plus one file per baseline -- 304 files for the
//! 24-station geometry -- sharing one tag. The tag names the *local* day and the
//! *local* time of the set's first row, so a file set holds the day it says it
//! holds, and the boundary between two days is a boundary between two sets
//! (docs/DESIGN-daily-files.md).
//!
//! Three rules make the output safe to resume from:
//!
//! 1. **Every file is always a whole number of rows.** The raw spectrum files are
//!    the only place where a partial write would be invisible: they carry no
//!    framing at all, so a half-written row is not detectable downstream and a
//!    reader cannot tell it from a short experiment. On open, every file is
//!    truncated back to its last complete row.
//!
//! 2. **The sidecars never claim a row that the data files do not have.** A row's
//!    spectrum is written and flushed first; only then are `rows_ms.*` and
//!    `blocks.csv` appended. A crash between the two leaves the sidecars one row
//!    short, which is the direction that costs one row rather than producing a
//!    row nobody can interpret.
//!
//! 3. **Resumption goes by the shortest file.** The number of complete rows is
//!    the minimum over all the files, so a crash anywhere yields a consistent
//!    prefix rather than a guess.
//!
//! The row index is the set's whole state: `ntime` is `rows_written`, and a reader
//! that wants frame ids or timestamps joins against `blocks.csv` and
//! `rows_ms.bin` by row index.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;

use anyhow::{bail, Context, Result};

use crate::pinned::RowBuf;

use crate::config::{
    local_day_at, local_stamp_at, local_tz_name, local_utc_offset_s, Config, QueueFull, RunPlan,
};

/// Longest row still worth page-locking, in bytes.
///
/// Above this the row buffer is an ordinary allocation. The threshold is not
/// derived, it is the boundary between the two geometries this program is
/// actually run at: 4 stations at `n_freq = 16384` is a 1.31 MB row, where
/// page-locking buys 1.35x on the fetch, and 24 stations is a 39 MB row, where
/// page-locking costs a factor of 159 in lost frames (see `pin_rows`).
const MAX_PINNED_ROW: usize = 4 << 20;

/// Bytes in one row of one baseline's file.
pub fn baseline_row_bytes(n_freq: u32) -> usize {
    n_freq as usize * 8 // complex64
}

/// How one set's files are named.
///
/// One place, because the writer, the resume scan, `--set` and every reader in
/// `tools/` have to agree on it; a naming rule spread over five call sites is a
/// naming rule that drifts. See the module documentation for the layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetNames {
    /// `storage.file_prefix`, put verbatim at the front; empty by default.
    pub file_prefix: String,
    /// Local time of this set's first row, from `storage.file_prefix_format`.
    pub stamp: String,
}

impl SetNames {
    pub fn new(file_prefix: &str, stamp: &str) -> Self {
        SetNames {
            file_prefix: file_prefix.to_string(),
            stamp: stamp.to_string(),
        }
    }

    /// The set's identity: `<prefix><stamp>`, which is what goes into
    /// `metadata.yaml` as `set_prefix`, what `--set` matches, and what the sidecar
    /// names are built from.
    ///
    /// No separator is inserted between the two halves: whatever `file_prefix`
    /// ends with is what the name has (write `corr24-` to get `corr24-20260918-...`).
    /// An implicit separator would double up on a prefix that already ends in one,
    /// which is exactly what a prefix is for.
    pub fn tag(&self) -> String {
        format!("{}{}", self.file_prefix, self.stamp)
    }

    /// `<prefix><station_j><station_k>-<stamp>.bin`.
    ///
    /// The station pair comes first so that a directory listing groups by pair,
    /// and so that a file can be identified without reading `metadata.yaml`. It
    /// also means the tag cannot be recovered from a baseline's name alone --
    /// which is why `discover_sets` goes by the sidecars.
    pub fn baseline(&self, station_j: &str, station_k: &str) -> String {
        format!(
            "{}{}{}-{}.bin",
            self.file_prefix, station_j, station_k, self.stamp
        )
    }

    pub fn time_txt(&self) -> String {
        format!("{}-time.txt", self.tag())
    }

    pub fn time_bin(&self) -> String {
        format!("{}-time.bin", self.tag())
    }

    pub fn blocks(&self) -> String {
        format!("{}-blocks.csv", self.tag())
    }

    pub fn metadata(&self) -> String {
        format!("{}-metadata.yaml", self.tag())
    }
}

/// How a set came to exist, recorded in its metadata so a reader can tell a
/// day's first set from a second one on the same day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetKind {
    /// The run's first set, on a day that had none.
    Start,
    /// The set opened when the local day changed under a running process.
    Rollover,
    /// An existing set that a restarted process appended to.
    Resume,
    /// A second set on a day that already had one (`--allow-second-set`).
    SecondSet,
}

impl SetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SetKind::Start => "start",
            SetKind::Rollover => "rollover",
            SetKind::Resume => "resume",
            SetKind::SecondSet => "second_set",
        }
    }
}

/// Which set a starting process should open.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SetPolicy {
    /// Append to the set that already belongs to today, if there is one.
    #[default]
    Auto,
    /// Ignore today's set and start a new one (the explicit opt-in to two sets on
    /// one day, which the default refuses because the alternative is mixing two
    /// geometries into one file).
    NewSet,
    /// Append to the named set, whatever day it belongs to.
    Explicit(String),
}

/// The output set for a new day could not be opened.
///
/// A type rather than a bare message so the pipeline can stop the run *with a
/// reason the report carries*, instead of dying with an error and no report --
/// the same treatment the zero-fill cap gets, and for the same reason: a run that
/// ends has to say why in the artifact it leaves behind.
#[derive(Debug)]
pub struct RolloverFailed {
    pub prefix: String,
    pub detail: String,
}

impl std::fmt::Display for RolloverFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "could not open the output file set for prefix {:?} after 3 attempts: {}",
            self.prefix, self.detail
        )
    }
}

impl std::error::Error for RolloverFailed {}

/// The open files of one set. No `Debug`: `File` has none, and printing a set's
/// handles would say nothing a reader wants anyway.
struct SetFiles {
    baselines: Vec<File>,
    ms_bin: File,
    ms_txt: BufWriter<File>,
    blocks: BufWriter<File>,
}

/// One set found in the directory, from its metadata and file lengths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundSet {
    /// The set's tag (`<prefix>-<stamp>`), which is what `--set` takes.
    pub prefix: String,
    /// How its files are named, read back from the set's own metadata.
    pub names: SetNames,
    /// Local day the set belongs to, from its metadata.
    pub day: String,
    pub kind: String,
    pub rows_written: u64,
    pub next_frame_id: Option<u64>,
}

/// What a resumed run found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeState {
    /// Rows already complete on disk; the next row to write has this index.
    pub rows_written: u64,
    /// First frame id of the next row, from the last `blocks.csv` entry.
    /// `None` for a fresh run.
    pub next_frame_id: Option<u64>,
    /// True if anything had to be truncated away.
    pub truncated_rows: u64,
}

impl ResumeState {
    pub fn is_fresh(&self) -> bool {
        self.rows_written == 0
    }
}

/// Per-row accounting that only the receive side knows.
///
/// The missing-frame flag belongs to a *station*, not to a baseline: every
/// baseline touching a station is missing exactly the same frames. So this
/// records one entry per station and `blocks.csv` gets one line per station per
/// row, rather than the same number duplicated across 300 baseline files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowBlock {
    pub row_index: u64,
    pub first_frame_id: u64,
    pub n_cum: u32,
    /// Per station, in station order: frames of this row that the station
    /// actually delivered. Fewer than `n_cum` means the rest were zero filled.
    pub valid_frames_per_station: Vec<u64>,
}

/// One row on its way to the writer thread.
enum Job {
    Row {
        /// The buffer this row lives in, shared with the other rows of its fetch
        /// batch. The last row of a batch is the one that puts it back in the
        /// pool -- see `run_writer_thread`.
        buf: std::sync::Arc<RowBuf>,
        /// Where this row starts inside `buf`, and how long it is. One row per
        /// fetch makes these `0` and `row_bytes`; a batch of `rows_per_fetch`
        /// rows shares one buffer and each row names its own slice.
        offset: usize,
        len: usize,
        row_ms: i64,
        blocks: RowBlock,
    },
    /// Wait until every queued row has been written (not fsynced), and answer.
    /// A point where the files on disk are what the caller expects, without
    /// stopping the thread.
    Sync(std::sync::mpsc::Sender<Result<u64, String>>),
    /// Flush and fsync everything, answer with the total row count, and stop.
    Finish(std::sync::mpsc::Sender<Result<u64, String>>),
}

/// The writing state: the open set, where the next one would go, and the geometry
/// to size it. Owned by the writer thread; the handle next to it only queues rows.
///
/// Split out from [`Writer`] because the two objects answer different questions:
/// this one owns the files and the durability, the handle owns the queue and the
/// backpressure. `docs/DESIGN-writer-thread.md` has the measurements behind that.
struct SetState {
    dir: PathBuf,
    plan: RunPlan,
    cfg: Config,
    config_hash: String,
    fsync_every: u64,
    daily: bool,

    /// Prefix of the set being written; empty until a set exists.
    prefix: String,
    /// The same tag, shared with the handle. A day boundary opens the next set in
    /// *this* thread, so [`Writer::current_set_tag`] can only answer "which set now"
    /// from a slot both sides can see.
    tag: Arc<Mutex<String>>,
    /// Local day of the current set; `None` until a set exists (its name has to
    /// come from its first row's timestamp, so it cannot exist before that row).
    set_day: Option<String>,
    set_kind: SetKind,
    set_first_row: Option<(u64, u64)>,
    set_continuation_of: Option<String>,
    rows_in_set: u64,
    backward_warned: bool,

    n_baseline: usize,
    per_baseline_row: usize,
    n_station: usize,
    row_bytes: usize,
    files: Option<SetFiles>,
    /// Rows this process has written, across every set it has opened.
    rows_total: u64,
}

/// Put `tag` where [`Writer::current_set_tag`] reads it.
///
/// Shared between the thread that opens sets and the handle that reports them. A
/// `Mutex` is enough: it is written once per set and read once per status sample.
fn publish_tag(slot: &Arc<Mutex<String>>, tag: String) {
    match slot.lock() {
        Ok(mut slot) => *slot = tag,
        Err(poisoned) => *poisoned.into_inner() = tag,
    }
}

impl SetState {
    /// Attach to an existing set, creating it if it is not there yet, and report
    /// where the run should resume.
    ///
    /// `names` decides every file name in the set; `found` is that set's discovered
    /// metadata when it already exists, and `None` means it is being created now.
    ///
    /// Runs on the calling thread at startup, before any socket is open: the run
    /// needs `ResumeState` to fix its row grid, and a failure here should stop the
    /// process before it starts receiving.
    // Eight arguments, every one of them this constructor's own input: the run's
    // shape, the set's naming, the metadata it is checked against, and the slot the
    // tag is published into. A parameter struct would only move the list elsewhere.
    #[allow(clippy::too_many_arguments)]
    fn attach(
        dir: &Path,
        plan: &RunPlan,
        cfg: &Config,
        config_hash: &str,
        fsync_every: u64,
        names: &SetNames,
        found: Option<&FoundSet>,
        tag: Arc<Mutex<String>>,
    ) -> Result<(Self, ResumeState)> {
        // One file per baseline, named by the two stations it correlates.
        let baseline_path = |j: usize, k: usize| {
            dir.join(names.baseline(&plan.enabled[j].name, &plan.enabled[k].name))
        };

        let n_baseline = plan.n_baseline;
        let per_baseline_row = baseline_row_bytes(plan.n_freq);
        let row_bytes = per_baseline_row * n_baseline;
        // A baseline file grows by exactly one row per integration -- not by a
        // whole block. Getting this wrong makes the resume scan report zero rows
        // for ever, which is exactly what it did before this comment existed.
        let baseline_bytes = per_baseline_row;

        // --- work out how many complete rows exist, and where to resume ---
        let mut complete = u64::MAX;
        let mut sizes: Vec<u64> = Vec::with_capacity(n_baseline);
        for (j, k) in plan.baseline_pairs().iter() {
            let p = baseline_path(*j, *k);
            let len = match std::fs::metadata(&p) {
                Ok(m) => m.len(),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
                Err(e) => return Err(e).with_context(|| format!("stat {}", p.display())),
            };
            sizes.push(len);
            complete = complete.min(len / baseline_bytes as u64);
        }
        if n_baseline == 0 {
            complete = 0;
        }
        let complete = if complete == u64::MAX { 0 } else { complete };

        let ms_len = file_len(&dir.join(names.time_bin()))?;
        let ms_rows = ms_len / 8;
        // blocks.csv has one line per station per row.
        let blocks_rows = count_csv_rows(&dir.join(names.blocks()))? / plan.n_station.max(1) as u64;
        // The shortest file wins: a crash can only ever leave one of them short,
        // and taking the minimum means we re-write a row rather than claim one
        // that is not fully there.
        let rows_written = complete.min(ms_rows).min(blocks_rows);
        let truncated_rows = complete.saturating_sub(rows_written);

        // --- truncate everything back to that prefix ---
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = baseline_path(*j, *k);
            let want = rows_written * baseline_bytes as u64;
            if sizes.get(idx).copied().unwrap_or(0) != want {
                let f = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&p)
                    .with_context(|| format!("opening {}", p.display()))?;
                f.set_len(want)
                    .with_context(|| format!("truncating {} to {want} bytes", p.display()))?;
            }
        }
        truncate_file(&dir.join(names.time_bin()), rows_written * 8)?;
        // blocks.csv carries one line per station per row.
        truncate_csv(
            &dir.join(names.blocks()),
            rows_written * plan.n_station as u64,
        )?;
        // The text view is derived from the binary one, so it is rebuilt rather
        // than truncated: two files that are supposed to agree are easier to keep
        // in step by regenerating one from the other than by truncating both.
        rebuild_ms_text(
            &dir.join(names.time_bin()),
            &dir.join(names.time_txt()),
            rows_written,
        )?;

        // --- the previous row's frame range, for resuming the grid ---
        let next_frame_id = read_last_block_first_frame(&dir.join(names.blocks()), plan.n_cum)?;

        // --- open everything for appending ---
        let mut baselines = Vec::with_capacity(n_baseline);
        for (j, k) in plan.baseline_pairs().iter() {
            let p = baseline_path(*j, *k);
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&p)
                .with_context(|| format!("opening {}", p.display()))?;
            // Append semantics without O_APPEND: we seek once and then rely on
            // every write being exactly one row, which we truncate back to on the
            // next start anyway.
            f.seek(SeekFrom::Start(rows_written * baseline_bytes as u64))?;
            baselines.push(f);
        }

        // Every handle has to be positioned at the end of what survives the
        // truncation. Opening without seeking leaves the position at zero, so a
        // resumed run silently overwrites its own sidecars from the start --
        // measured: 98 rows of spectrum with 49 timestamps and 49 block records.
        let mut ms_bin = open_rw(&dir.join(names.time_bin()))?;
        ms_bin.seek(SeekFrom::Start(rows_written * 8))?;
        let mut ms_txt_file = open_rw(&dir.join(names.time_txt()))?;
        ms_txt_file.seek(SeekFrom::End(0))?;
        let ms_txt = BufWriter::new(ms_txt_file);
        let mut blocks_file = open_rw(&dir.join(names.blocks()))?;
        blocks_file.seek(SeekFrom::End(0))?;
        let blocks = BufWriter::new(blocks_file);

        // A set's metadata.yaml is written once, when the set is created; opening
        // a set that already has rows checks it rather than rewriting it, because
        // rewriting geometry that existing rows were produced with is how a reader
        // gets misled. Missing metadata is the one case that is written rather
        // than checked: a crash between creating the files and writing metadata
        // leaves a set nobody has described yet, and the geometry in hand is the
        // only description available.
        let meta_path = dir.join(names.metadata());
        let created_now = rows_written == 0 || !meta_path.exists();
        let day = found
            .map(|s| s.day.clone())
            .unwrap_or_else(|| local_day_at(crate::sched::system_ms()));
        if created_now {
            write_metadata(
                &meta_path,
                plan,
                cfg,
                config_hash,
                names,
                &day,
                SetKind::Start,
                None,
                None,
                rows_written,
            )?;
        } else {
            check_metadata(&meta_path, config_hash)?;
        }

        let state = ResumeState {
            rows_written,
            next_frame_id,
            truncated_rows,
        };
        // Publish it now: on resume this is the set the rows go into, and a
        // rollover republishes from `try_create_set`.
        publish_tag(&tag, names.tag());
        Ok((
            SetState {
                dir: dir.to_path_buf(),
                prefix: names.tag(),
                tag,
                plan: plan.clone(),
                cfg: cfg.clone(),
                config_hash: config_hash.to_string(),
                fsync_every: fsync_every.max(1),
                daily: cfg.storage.daily_files,
                set_day: Some(day),
                set_kind: if created_now { SetKind::Start } else { SetKind::Resume },
                set_first_row: None,
                set_continuation_of: None,
                rows_in_set: rows_written,
                backward_warned: false,
                n_baseline,
                per_baseline_row,
                n_station: plan.n_station,
                row_bytes,
                files: Some(SetFiles {
                    baselines,
                    ms_bin,
                    ms_txt,
                    blocks,
                }),
                rows_total: rows_written,
            },
            state,
        ))
    }

    /// A state with no set yet: the first row's local day names the first set.
    fn unopened(
        dir: &Path,
        plan: &RunPlan,
        cfg: &Config,
        config_hash: &str,
        fsync_every: u64,
        first_kind: SetKind,
        tag: Arc<Mutex<String>>,
    ) -> Self {
        SetState {
            dir: dir.to_path_buf(),
            plan: plan.clone(),
            cfg: cfg.clone(),
            config_hash: config_hash.to_string(),
            fsync_every: fsync_every.max(1),
            daily: cfg.storage.daily_files,
            prefix: String::new(),
            tag,
            set_day: None,
            set_kind: first_kind,
            set_first_row: None,
            set_continuation_of: None,
            rows_in_set: 0,
            backward_warned: false,
            n_baseline: plan.n_baseline,
            per_baseline_row: baseline_row_bytes(plan.n_freq),
            n_station: plan.n_station,
            row_bytes: baseline_row_bytes(plan.n_freq) * plan.n_baseline,
            files: None,
            rows_total: 0,
        }
    }

    /// Make sure the row about to be written has a set, opening or rolling one as
    /// the row's own local day requires.
    ///
    /// The decision is a pure function of (current set's day, this row's day) --
    /// see [`decide`] -- so it is unit-testable without a timezone, and the
    /// timezone itself is read per row rather than cached, which is what makes a
    /// daylight-saving day come out right.
    fn ensure_set_for(&mut self, row_ms: i64, blocks: &RowBlock) -> Result<()> {
        if !self.daily {
            if self.files.is_none() {
                bail!("no output set is open and daily files are off; this is a writer bug");
            }
            return Ok(());
        }
        let day = local_day_at(row_ms);
        match decide(self.set_day.as_deref(), &day) {
            Decision::Open => {
                // `set_kind` was decided when the state was built: a day that
                // already had a set but was not appended to (a changed
                // configuration, `--allow-second-set`) opens a second one.
                let kind = self.set_kind;
                self.create_set(&day, row_ms, blocks, kind, None)
            }
            Decision::Keep => Ok(()),
            Decision::Roll => {
                let rows = self.rows_in_set;
                let from_day = self.set_day.clone().unwrap_or_default();
                let from = self.close_set()?;
                self.create_set(&day, row_ms, blocks, SetKind::Rollover, Some(from.clone()))?;
                // Loud, because this is the one event the daily layout exists for
                // and a multi-day run's log is the only place it can be seen while
                // it happens; the prefix is not knowable in advance (it is this
                // row's own local time), so it has to be printed.
                eprintln!(
                    "writer: local day changed {from_day} -> {day}: closed set {from} \
                     ({rows} rows) and opened {}",
                    self.prefix
                );
                Ok(())
            }
            Decision::WarnBackwards => {
                if !self.backward_warned {
                    self.backward_warned = true;
                    eprintln!(
                        "writer: the local day went backwards ({} -> {day}) with the wall clock; \
                         keeping the current set {} and not opening another, because the row \
                         index is monotonic and a set is never reopened once left",
                        self.set_day.as_deref().unwrap_or("?"),
                        self.prefix
                    );
                }
                Ok(())
            }
        }
    }

    /// Open a new set whose prefix is the local time of `row_ms`, retrying a few
    /// times and then failing as a [`RolloverFailed`].
    ///
    /// Retried because a transient metadata failure (ENOSPC while another writer
    /// finishes, a busy NFS-style backend) is worth three attempts; *not* retried
    /// indefinitely, and never fallen back on by writing into the previous set,
    /// because a file set whose name says one day and whose rows are another is
    /// worse than a stopped run.
    fn create_set(
        &mut self,
        day: &str,
        row_ms: i64,
        blocks: &RowBlock,
        kind: SetKind,
        continuation: Option<String>,
    ) -> Result<()> {
        // The set's identity is the configured prefix plus the local time of its
        // first row; the files inside it are named from the same two halves.
        let names = SetNames::new(
            &self.cfg.storage.file_prefix,
            &local_stamp_at(row_ms, &self.cfg.storage.file_prefix_format),
        );
        let mut last: Option<anyhow::Error> = None;
        for attempt in 1..=3 {
            match self.try_create_set(&names, day, kind, continuation.clone(), blocks) {
                Ok(()) => {
                    if kind != SetKind::Rollover {
                        eprintln!(
                            "writer: opened file set {} for local day {day} ({})",
                            names.tag(),
                            kind.as_str()
                        );
                    }
                    return Ok(());
                }
                Err(e) => {
                    eprintln!(
                        "writer: opening file set {} for local day {day} failed \
                         (attempt {attempt}/3): {e}",
                        names.tag()
                    );
                    last = Some(e);
                    if attempt < 3 {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        }
        Err(anyhow::Error::new(RolloverFailed {
            prefix: names.tag(),
            detail: format!("{:#}", last.expect("three attempts set it")),
        }))
    }

    fn try_create_set(
        &mut self,
        names: &SetNames,
        day: &str,
        kind: SetKind,
        continuation: Option<String>,
        blocks: &RowBlock,
    ) -> Result<()> {
        let baseline_path = |plan: &RunPlan, j: usize, k: usize| {
            self.dir
                .join(names.baseline(&plan.enabled[j].name, &plan.enabled[k].name))
        };
        // A set that already holds rows belongs to somebody else's run; reusing
        // its tag would interleave two processes' rows. An empty file is a
        // half-created set from a failed attempt: reuse it.
        let (j0, k0) = self
            .plan
            .baseline_pairs()
            .first()
            .copied()
            .unwrap_or((0, 0));
        let probe = baseline_path(&self.plan, j0, k0);
        if let Ok(m) = std::fs::metadata(&probe) {
            if m.len() > 0 {
                bail!(
                    "{} already holds {} bytes: the set tag {} is in use",
                    probe.display(),
                    m.len(),
                    names.tag()
                );
            }
        }

        let mut baselines = Vec::with_capacity(self.n_baseline);
        let pairs = self.plan.baseline_pairs();
        for (j, k) in pairs.iter() {
            let p = baseline_path(&self.plan, *j, *k);
            let f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&p)
                .with_context(|| format!("creating {}", p.display()))?;
            baselines.push(f);
        }
        let time_bin_path = self.dir.join(names.time_bin());
        let time_txt_path = self.dir.join(names.time_txt());
        let blocks_path = self.dir.join(names.blocks());
        let ms_bin = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&time_bin_path)
            .with_context(|| format!("creating {}", time_bin_path.display()))?;
        let ms_txt = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&time_txt_path)
                .with_context(|| format!("creating {}", time_txt_path.display()))?,
        );
        let blocks_file = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&blocks_path)
                .with_context(|| format!("creating {}", blocks_path.display()))?,
        );

        write_metadata(
            &self.dir.join(names.metadata()),
            &self.plan,
            &self.cfg,
            &self.config_hash,
            names,
            day,
            kind,
            continuation.as_deref(),
            Some((blocks.row_index, blocks.first_frame_id)),
            0,
        )?;

        let tag = names.tag();
        self.prefix = tag.clone();
        publish_tag(&self.tag, tag);
        self.set_day = Some(day.to_string());
        self.set_kind = kind;
        self.set_continuation_of = continuation;
        self.set_first_row = Some((blocks.row_index, blocks.first_frame_id));
        self.rows_in_set = 0;
        self.files = Some(SetFiles {
            baselines,
            ms_bin,
            ms_txt,
            blocks: blocks_file,
        });
        Ok(())
    }

    /// Flush and fsync the current set, and let go of its files.
    ///
    /// Called at the day boundary and at the end of the run. The boundary case is
    /// why this is not optional: whatever `fsync_every_rows` says, a set that is
    /// about to be closed is one whose rows have to be on the platter.
    fn close_set(&mut self) -> Result<String> {
        self.finish_set()?;
        let closed = self.prefix.clone();
        self.files = None;
        Ok(closed)
    }

    /// Append one integration row to the open set.
    ///
    /// The spectrum goes to disk first and is flushed; the sidecars follow. That
    /// order is the whole reason a crash can only cost a row and never produce an
    /// unreadable one.
    fn write_one(&mut self, spectrum: &[u8], row_ms: i64, blocks: &RowBlock) -> Result<()> {
        self.ensure_set_for(row_ms, blocks)?;

        let files = self
            .files
            .as_mut()
            .expect("ensure_set_for leaves an open set");
        debug_assert_eq!(
            files.baselines.len(),
            self.n_baseline,
            "one file per baseline, checked because a mismatch would shuffle spectra \
             between baselines silently"
        );
        for (idx, f) in files.baselines.iter_mut().enumerate() {
            let start = idx * self.per_baseline_row;
            // One write per baseline, of exactly one row. Not write_all in a loop:
            // a partially applied row is what we are trying to avoid.
            f.write_all(&spectrum[start..start + self.per_baseline_row])
                .with_context(|| format!("writing row {} baseline {idx}", self.rows_total))?;
        }

        if self.rows_total % self.fsync_every == 0 {
            // Flush the data before the sidecars claim it exists.
            for f in files.baselines.iter_mut() {
                f.flush()?;
            }
            for f in files.baselines.iter() {
                f.sync_data()?;
            }
        }

        files
            .ms_bin
            .write_all(&row_ms.to_le_bytes())
            .context("writing rows_ms.bin")?;
        writeln!(files.ms_txt, "{}", text_time(row_ms)).context("writing rows_ms.txt")?;

        for (station, valid) in blocks.valid_frames_per_station.iter().enumerate() {
            writeln!(
                files.blocks,
                "{},{},{},{},{},{}",
                blocks.row_index,
                station,
                blocks.first_frame_id,
                blocks.n_cum,
                valid,
                blocks.n_cum as u64 - *valid
            )
            .context("writing blocks.csv")?;
        }

        // Flush the sidecars on the same schedule as the rows they describe. They go
        // through `BufWriter`s, and `rows_ms.txt` is about 24 bytes a row, so its own
        // 8 KiB flush is 341 rows -- close to six minutes during which a reader
        // watching that file concludes the run has stopped (measured: 121 rows behind
        // `rows_ms.bin`, which is written unbuffered).
        //
        // This has to sit *after* the writes above. The first attempt put it in the
        // fsync branch, which runs earlier in this function, so it flushed an empty
        // buffer and changed nothing at all -- the branch was entered, the call
        // returned Ok, and the file stayed empty until the next `sync`.
        if self.rows_total % self.fsync_every == 0 {
            files.ms_txt.flush()?;
            files.blocks.flush()?;
        }

        self.rows_total += 1;
        self.rows_in_set += 1;
        Ok(())
    }

    /// Push the buffered sidecars to the OS. Not a durability point: `rows_ms.bin`
    /// is written directly, `blocks.csv` and `rows_ms.txt` go through a `BufWriter`.
    fn flush_sidecars(&mut self) -> Result<()> {
        if let Some(files) = self.files.as_mut() {
            files.ms_txt.flush()?;
            files.blocks.flush()?;
        }
        Ok(())
    }

    /// Flush and fsync everything in the current set.
    fn finish_set(&mut self) -> Result<()> {
        let Some(files) = self.files.as_mut() else {
            return Ok(()); // never had a set: nothing was written
        };
        for f in files.baselines.iter_mut() {
            f.flush()?;
        }
        for f in files.baselines.iter() {
            f.sync_all()?;
        }
        files.ms_bin.sync_all()?;
        files.ms_txt.flush()?;
        files.ms_txt.get_ref().sync_all()?;
        files.blocks.flush()?;
        files.blocks.get_ref().sync_all()?;
        Ok(())
    }
}

/// The writer thread: take one row at a time off the queue, write it, hand the
/// buffer back, and never touch anything the pipeline thread owns.
fn run_writer_thread(
    mut state: SetState,
    rx: std::sync::mpsc::Receiver<Job>,
    free: std::sync::mpsc::Sender<RowBuf>,
    rows: Arc<AtomicU64>,
    first_error: Arc<Mutex<Option<anyhow::Error>>>,
) {
    let started = Instant::now();
    let mut busy = Duration::ZERO;
    let mut written = 0u64;
    while let Ok(job) = rx.recv() {
        match job {
            Job::Row {
                buf,
                offset,
                len,
                row_ms,
                blocks,
            } => {
                let t0 = std::time::Instant::now();
                let r = state.write_one(&buf[offset..offset + len], row_ms, &blocks);
                busy += t0.elapsed();
                match r {
                    Ok(()) => {
                        written += 1;
                        rows.store(state.rows_total, Ordering::Release);
                        // The last slice of a batch is the one that owns the
                        // buffer: every other row's clone is already gone by the
                        // time it is written, because they are written in order
                        // and the pipeline dropped its own handle when it queued
                        // them. `try_unwrap` is how that is decided without a
                        // counter of our own.
                        if let Ok(owned) = std::sync::Arc::try_unwrap(buf) {
                            let _ = free.send(owned);
                        }
                    }
                    Err(e) => {
                        *first_error.lock().expect("not poisoned") = Some(e);
                        break;
                    }
                }
            }
            Job::Sync(ack) => {
                let r = state
                    .flush_sidecars()
                    .map(|_| state.rows_total)
                    .map_err(|e| format!("{e:#}"));
                let _ = ack.send(r);
            }
            Job::Finish(ack) => {
                let r = state
                    .finish_set()
                    .map(|_| state.rows_total)
                    .map_err(|e| format!("{e:#}"));
                let _ = ack.send(r);
                break;
            }
        }
    }
    // What only the thread knows: how long a row really costs it. If the busy
    // figure approaches the row period the writer cannot keep up at this rate --
    // the pipeline is then held by the queue, which the handle's own
    // `max_outstanding` line reports, not something the thread can see.
    let per_row = |d: Duration| {
        if written > 0 {
            1000.0 * d.as_secs_f64() / written as f64
        } else {
            0.0
        }
    };
    eprintln!(
        "writer: thread wrote {written} rows, {:.0} ms/row of its own time \
         ({:.0} ms/row of wall clock since it started)",
        per_row(busy),
        per_row(started.elapsed())
    );
}

/// Handle to the writer thread: hand it a row and do not wait for the disk.
///
/// The pipeline thread used to write the rows itself, which meant every `fsync`
/// (300 of them per row at one per baseline file) was time it was not releasing
/// staging slots. Measured at nch=16384 that cost 142 ms of a 1007 ms row period
/// on the pipeline thread, and because the GPU pipeline has no headroom at that
/// channel count it turned into 28-29% of frames zero filled. Here the row is
/// copied into a pooled buffer and queued; the queue is bounded by
/// `storage.queue_bytes`, so a writer that falls behind blocks the pipeline --
/// which is the old behaviour, kept as the honest fallback -- and a writer that
/// merely pauses costs nothing.
pub struct Writer {
    dir: PathBuf,
    row_bytes: usize,
    n_station: usize,
    /// The set this process starts on, for the startup line. Later sets (a day
    /// boundary) are announced by the thread, which is where they happen.
    prefix: String,
    /// The set being written right now, shared with that thread. See
    /// [`Writer::current_set_tag`].
    tag: Arc<Mutex<String>>,
    set_day: Option<String>,
    set_kind: SetKind,

    tx: std::sync::mpsc::SyncSender<Job>,
    rx_free: std::sync::mpsc::Receiver<RowBuf>,
    handle: Option<std::thread::JoinHandle<()>>,
    rows: Arc<AtomicU64>,
    first_error: Arc<Mutex<Option<anyhow::Error>>>,
    pool: Vec<RowBuf>,
    /// What to do when the queue cannot take another row.
    queue_full: QueueFull,
    /// Rows thrown away because the queue was full: a gap in the data, counted so
    /// the end-of-run report can state it. Only this handle touches it.
    dropped: u64,
    /// A run of consecutive drops, so the log says "dropped 52 rows (2301..2352)"
    /// once instead of printing 52 lines, and announces the recovery too.
    drop_streak: u64,
    drop_first_row: u64,
    drop_last_row: u64,
    /// Rows that may be in flight, from `storage.queue_bytes`.
    queue_depth: usize,
    pool_cap: usize,
    /// Whether row buffers are page-locked. True when one fetch produces one row,
    /// which is the only case where the row buffer *is* the fetch destination --
    /// see [`crate::pinned::RowBuf`].
    pin_rows: bool,
    /// Rows this process has handed to the thread, and the reader count it
    /// started from. The queue length is their difference against what the
    /// thread has finished -- counting buffers instead would drift, because a
    /// buffer is taken at two different places but only ever given back once.
    sent: u64,
    rows_at_open: u64,
    max_outstanding: u64,
    finished: bool,
    rows_final: u64,
}

/// The state an operator needs in a log line, without the open file handles.
impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("dir", &self.dir)
            .field("set_prefix", &self.prefix)
            .field("set_day", &self.set_day)
            .field("rows_written", &self.rows_written())
            .field("queued", &self.queued())
            .finish()
    }
}

impl Writer {
    /// Create or resume a run directory, appending to today's set if there is one.
    pub fn open(
        dir: &Path,
        plan: &RunPlan,
        cfg: &Config,
        config_hash: &str,
        fsync_every: u64,
    ) -> Result<(Self, ResumeState)> {
        Self::open_with(dir, plan, cfg, config_hash, fsync_every, &SetPolicy::Auto)
    }

    /// As [`Writer::open`], with an explicit policy for which set to use.
    pub fn open_with(
        dir: &Path,
        plan: &RunPlan,
        cfg: &Config,
        config_hash: &str,
        fsync_every: u64,
        policy: &SetPolicy,
    ) -> Result<(Self, ResumeState)> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating the run directory {}", dir.display()))?;

        let daily = cfg.storage.daily_files;
        // The shared slot for "which set is being written right now", empty until a
        // set exists. Created before the first attach so that both paths fill it.
        let tag: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let empty = ResumeState {
            rows_written: 0,
            next_frame_id: None,
            truncated_rows: 0,
        };
        let (state, resume) = if !daily {
            // No per-day splitting: one set for the whole run, stamped with the
            // local time the writer opened it. The files still carry the station
            // pair and a time, so a name stays self-describing; what is gone is
            // the rollover at the local midnight (docs/DESIGN-daily-files.md).
            SetState::attach(
                dir,
                plan,
                cfg,
                config_hash,
                fsync_every,
                &SetNames::new(
                    &cfg.storage.file_prefix,
                    &local_stamp_at(crate::sched::system_ms(), &cfg.storage.file_prefix_format),
                ),
                None,
                tag.clone(),
            )?
        } else {
            let today = local_day_at(crate::sched::system_ms());
            let found = discover_sets(dir, plan)?;
            if std::env::var_os("CORR_SET_DEBUG").is_some() {
                eprintln!("writer: daily sets in {}: {found:?}", dir.display());
            }
            // Which set to append to?
            let chosen: Option<FoundSet> = match policy {
                SetPolicy::Explicit(prefix) => {
                    let hit = found.iter().find(|s| s.prefix == *prefix).cloned();
                    match hit {
                        Some(s) => Some(s),
                        None => bail!(
                            "no file set with prefix {prefix:?} in {}; the sets there are {:?}",
                            dir.display(),
                            found.iter().map(|s| s.prefix.as_str()).collect::<Vec<_>>()
                        ),
                    }
                }
                SetPolicy::NewSet => None,
                // The *latest* set of the day, not the earliest. When a day has more
                // than one set -- `--allow-second-set` opened it because the
                // configuration changed -- the older ones are exactly the sets whose
                // config hash no longer matches, so picking the earliest guarantees
                // "resuming would misinterpret every existing row" on every later
                // run. Measured: with two sets of one day, `Auto` kept choosing the
                // 15:57 one over the 16:35 one and refused to start.
                SetPolicy::Auto => found
                    .iter()
                    .filter(|s| s.day == today)
                    .max_by(|a, b| a.prefix.cmp(&b.prefix))
                    .cloned(),
            };
            match chosen {
                Some(set) => {
                    // Appending to an existing set: the config hash has to agree,
                    // or two geometries would be mixed into one file set. The
                    // message names the way out, because the alternative is an
                    // operator guessing.
                    SetState::attach(
                        dir,
                        plan,
                        cfg,
                        config_hash,
                        fsync_every,
                        &set.names,
                        Some(&set),
                        tag.clone(),
                    )
                    .map_err(|e| {
                        if matches!(policy, SetPolicy::Auto) {
                            e.context(format!(
                                "appending to today's set {}: if the configuration changed on \
                                 purpose, either pass --set <prefix> to choose a set explicitly \
                                 or --allow-second-set to start another set for today",
                                set.prefix
                            ))
                        } else {
                            e
                        }
                    })?
                }
                None => {
                    // No set to append to: the first row will name one.
                    let second = found.iter().any(|s| s.day == today);
                    let kind = if second { SetKind::SecondSet } else { SetKind::Start };
                    (
                        SetState::unopened(
                            dir,
                            plan,
                            cfg,
                            config_hash,
                            fsync_every,
                            kind,
                            tag.clone(),
                        ),
                        empty,
                    )
                }
            }
        };

        let row_bytes = state.row_bytes;
        // Which side of the row-size gate this run landed on is worth being able
        // to see without guessing, at the two geometries that matter.
        if std::env::var("CORR_PIN_DEBUG").is_ok() {
            eprintln!(
                "writer: row_bytes={} rows_per_fetch={} -> pin_rows={}",
                row_bytes,
                plan.rows_per_fetch,
                plan.rows_per_fetch == 1 && row_bytes <= MAX_PINNED_ROW
            );
        }
        let n_station = state.n_station;
        let prefix = state.prefix.clone();
        let live_tag = state.tag.clone();
        let set_day = state.set_day.clone();
        let set_kind = state.set_kind;
        let rows_at_open = resume.rows_written;

        // How many rows may be in flight. `storage.queue_bytes` was in the schema
        // and validated from the start but never used; this is what it was for.
        // Capped, because the point is to absorb a hiccup, not to buffer a minute
        // of data in RAM: the writer needs ~150 ms per row and has ~1000 ms.
        let depth = (cfg.storage.queue_bytes as usize / row_bytes.max(1)).clamp(2, 8);
        let (tx, rx) = std::sync::mpsc::sync_channel(depth);
        let (free_tx, rx_free) = std::sync::mpsc::channel();
        let rows = Arc::new(AtomicU64::new(rows_at_open));
        let first_error: Arc<Mutex<Option<anyhow::Error>>> = Arc::new(Mutex::new(None));
        let handle = {
            let rows = rows.clone();
            let first_error = first_error.clone();
            std::thread::Builder::new()
                .name("corr-writer".to_string())
                .spawn(move || run_writer_thread(state, rx, free_tx, rows, first_error))
                .context("spawning the writer thread")?
        };

        Ok((
            Writer {
                dir: dir.to_path_buf(),
                row_bytes,
                n_station,
                prefix,
                tag: live_tag,
                set_day,
                set_kind,
                tx,
                rx_free,
                handle: Some(handle),
                rows,
                first_error,
                pool: Vec::new(),
                queue_full: cfg.storage.queue_full,
                dropped: 0,
                drop_streak: 0,
                drop_first_row: 0,
                drop_last_row: 0,
                queue_depth: depth,
                pool_cap: depth + 2,
                // One row per fetch is what makes the row buffer the fetch
                // destination (`flush_rows`), and that is the only case where
                // page-locking it changes anything.
                // One row per fetch is what makes the row buffer the fetch
                // destination (`flush_rows`), and that is one of the two
                // conditions for page-locking it to be worth anything. The other
                // is that the row has to be small: this path exists for
                // sub-millisecond rows, where the fetch is most of the row period
                // and the D2H is the whole cost.
                //
                // Measured, and the reason this is gated at all: at 24 stations
                // with a one-second integration a row is 39 MB, and pinning it
                // cost the receive side badly. Same configuration, 150 s each on
                // this machine:
                //
                //                    lost frames   socket drops   receive cost
                //   pageable (master)       31,155         11,293   19.3 s/station
                //   page-locked          4,960,459      1,748,749   25.9 s/station
                //
                // -- a factor of 159 in lost frames, zero fill 0.005% -> 0.74%.
                // Turning the same switch off in a 60 s run put it back at 12,011
                // lost and 0.005% with nothing else changed. So short rows get
                // the page-locked path and long ones stay as they always were.
                pin_rows: plan.rows_per_fetch == 1 && row_bytes <= MAX_PINNED_ROW,
                sent: 0,
                rows_at_open,
                max_outstanding: 0,
                finished: false,
                rows_final: rows_at_open,
            },
            resume,
        ))
    }

    /// Record a row thrown away because the queue was full.
    ///
    /// The first drop explains itself and names the row, the rest are counted
    /// silently, and the recovery reports the range. One line per dropped row would
    /// turn a 52-second stall into 52 lines of noise, while the range is what a
    /// reader needs: those row indices are the gap in `blocks.csv`, `time.bin` and
    /// every baseline file.
    fn note_dropped(&mut self, row_index: u64) {
        if self.drop_streak == 0 {
            self.drop_first_row = row_index;
            eprintln!(
                "writer: queue full ({depth} rows in flight), dropping row {row_index} and any \
                 others that will not fit -- the run keeps receiving; a dropped row is absent \
                 from every file of this set",
                depth = self.queue_depth
            );
        }
        self.drop_streak += 1;
        self.drop_last_row = row_index;
        self.dropped += 1;
    }

    /// Called after a row is handed over, so a run of drops gets closed out.
    fn note_queued(&mut self) {
        if self.drop_streak > 0 {
            eprintln!(
                "writer: caught up after dropping {n} row(s) (row_index {first}..={last}); \
                 blocks.csv and time.bin jump across them, and no baseline file holds a partial \
                 row",
                n = self.drop_streak,
                first = self.drop_first_row,
                last = self.drop_last_row
            );
            self.drop_streak = 0;
        }
    }

    /// Rows thrown away because the queue was full.
    pub fn dropped_rows(&self) -> u64 {
        self.dropped
    }

    /// How many rows the queue can hold.
    pub fn queue_depth(&self) -> usize {
        self.queue_depth
    }

    /// The deepest the queue has been, in rows: how close this run has come to
    /// dropping data.
    pub fn max_queue_depth(&self) -> u64 {
        self.max_outstanding
    }

    /// Take an error the writer thread hit, if any.
    ///
    /// One row late at most: the thread records the first failure and stops, and
    /// the next call here (a `write_row` or `finish`) returns it. That is the
    /// price of not making the pipeline thread wait for the disk, and at a 1 s
    /// integration it is at most one second of extra rows in flight.
    fn take_error(&mut self) -> Result<()> {
        let mut slot = self.first_error.lock().expect("not poisoned");
        match slot.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The set this process *starts* on: prefix and local day, for the startup
    /// line. It is fixed at open and never updated by the writer thread, so with
    /// `storage.daily_files` on and no set to resume it is empty until the thread
    /// names the set after the first row's timestamp -- and it stays the startup
    /// value across a day boundary. The sets a run actually produced are the ones
    /// in the directory; read those (and their metadata) instead of this --
    /// [`Writer::current_set_tag`] is the one that follows the thread.
    pub fn set_prefix(&self) -> &str {
        &self.prefix
    }

    /// The set being written *right now*, which is what a monitor wants: before a
    /// set exists, and again after a local-midnight rollover, this differs from
    /// [`Writer::set_prefix`] -- that one is fixed at open, by design.
    ///
    /// Empty until the first row (or a resume) has named a set.
    pub fn current_set_tag(&self) -> String {
        match self.tag.lock() {
            Ok(tag) => tag.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// The local day of the set this process starts on; `None` when the first
    /// row has yet to name it. See [`Writer::set_prefix`].
    pub fn set_day(&self) -> Option<&str> {
        self.set_day.as_deref()
    }

    /// How this process starts on that set. See [`Writer::set_prefix`].
    pub fn set_kind(&self) -> SetKind {
        self.set_kind
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Rows the writer has finished with. Exact after [`Writer::finish`].
    pub fn rows_written(&self) -> u64 {
        self.rows.load(Ordering::Acquire).max(self.rows_final)
    }

    /// Rows handed over but not yet written: the queue's length right now.
    pub fn queued(&self) -> u64 {
        self.sent
            .saturating_sub(self.rows_written().saturating_sub(self.rows_at_open))
    }

    /// Queue one integration row. Returns as soon as the row is handed over.
    ///
    /// The length checks stay here rather than in the thread so a caller that
    /// passes the wrong shape is told immediately, on its own stack.
    fn check_row(&mut self, len: usize, blocks: &RowBlock) -> Result<()> {
        if len != self.row_bytes {
            bail!(
                "write_row got {} bytes but one row is {}",
                len,
                self.row_bytes
            );
        }
        if blocks.valid_frames_per_station.len() != self.n_station {
            bail!(
                "write_row got {} per-station records but there are {} stations",
                blocks.valid_frames_per_station.len(),
                self.n_station
            );
        }
        self.take_error()
    }

    /// A buffer of exactly `need` bytes, reused from the pool when there is one.
    ///
    /// Exists so a caller can fetch a row *into* the buffer the writer is going to
    /// write. With `rows_per_fetch == 1` that is what makes the hand-over a move
    /// instead of a copy, and the copy is not optional overhead: one row of 24
    /// stations at `n_freq = 16384` is 39 MB, so at a one-second integration it is
    /// 39 MB/s and at a 40.96 us row it is 960 GB/s of `memcpy` -- more than the
    /// machine's whole memory bandwidth, spent moving bytes nobody reads.
    pub fn take_row_buffer(&mut self, need: usize) -> Result<RowBuf> {
        let reusable = self
            .pool
            .pop()
            .or_else(|| self.rx_free.try_recv().ok())
            // A pooled buffer of the wrong size is not reusable: the length is the
            // allocation's, and it is the length the fetch checks.
            .filter(|b| b.len() == need && b.is_pinned() == self.pin_rows);
        match reusable {
            Some(b) => Ok(b),
            None => RowBuf::new(need, self.pin_rows),
        }
    }

    pub fn write_row(&mut self, spectrum: &[u8], row_ms: i64, blocks: &RowBlock) -> Result<()> {
        self.check_row(spectrum.len(), blocks)?;
        let mut buf = self.take_row_buffer(spectrum.len())?;
        buf.as_mut_slice().copy_from_slice(spectrum);
        let shared = std::sync::Arc::new(buf);
        self.put_row(&shared, 0, spectrum.len(), row_ms, blocks)?;
        self.return_buffer(shared);
        Ok(())
    }

    /// Queue one row that already lives in `buf` at `offset`, without copying it.
    ///
    /// This is the hand-over `flush_rows` uses. One fetch produces
    /// `rows_per_fetch` rows contiguous in one buffer, so each row is a slice of
    /// a shared buffer rather than a buffer of its own: the buffer goes back to
    /// the pool when its last row is written, which for one row per fetch is
    /// immediately and for a batch is after the batch.
    pub fn write_row_slice(
        &mut self,
        buf: &std::sync::Arc<RowBuf>,
        offset: usize,
        len: usize,
        row_ms: i64,
        blocks: &RowBlock,
    ) -> Result<()> {
        self.check_row(len, blocks)?;
        self.put_row(buf, offset, len, row_ms, blocks)
    }

    /// Give a fetch buffer back to the pool. Used when a fetch returned no rows,
    /// and by `write_row` for the one-row case.
    pub fn return_buffer(&mut self, buf: std::sync::Arc<RowBuf>) {
        if let Ok(owned) = std::sync::Arc::try_unwrap(buf) {
            if self.pool.len() < self.pool_cap {
                self.pool.push(owned);
            }
        }
    }

    fn put_row(
        &mut self,
        buf: &std::sync::Arc<RowBuf>,
        offset: usize,
        len: usize,
        row_ms: i64,
        blocks: &RowBlock,
    ) -> Result<()> {
        let row_index = blocks.row_index;
        let job = Job::Row {
            buf: std::sync::Arc::clone(buf),
            offset,
            len,
            row_ms,
            blocks: blocks.clone(),
        };
        // `buf` holds this row and nothing else and one `Job` carries one row, so
        // this is where "all of it or none of it" is decided: a row not handed over
        // here is never seen by the thread, and no file can hold half of it. Whether
        // a full queue drops or blocks is `storage.queue_full`; dropping is the
        // default, because blocking here is what turns a slow disk into lost packets
        // (a 52 s stall became 240 M dropped packets on 2026-09-19).
        match self.queue_full {
            QueueFull::Drop => match self.tx.try_send(job) {
                Ok(()) => {
                    self.sent += 1;
                    self.note_queued();
                }
                Err(std::sync::mpsc::TrySendError::Full(job)) => {
                    // Give the buffer back for the next fetch to reuse when this was
                    // its last row: the row is the loss, the allocation is not.
                    if let Job::Row { buf, .. } = job {
                        self.return_buffer(buf);
                    }
                    self.note_dropped(row_index);
                    return self.take_error();
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                    // The thread stopped between the check above and this send. Its own
                    // error is the reason whenever it recorded one: a bare "the thread
                    // stopped" here would lose the type the pipeline needs (a failed
                    // day rollover is a `RolloverFailed`, and the report names it).
                    self.take_error()?;
                    bail!("the writer thread has stopped without reporting a reason");
                }
            },
            QueueFull::Block => {
                if self.tx.send(job).is_err() {
                    self.take_error()?;
                    bail!("the writer thread has stopped without reporting a reason");
                }
                self.sent += 1;
                self.note_queued();
            }
        }
        let queued = self.queued();
        if queued > self.max_outstanding {
            self.max_outstanding = queued;
        }
        while let Ok(b) = self.rx_free.try_recv() {
            if self.pool.len() < self.pool_cap {
                self.pool.push(b);
            }
        }
        self.take_error()
    }

    /// Wait until every row handed over so far has been written to the files.
    ///
    /// Not a durability point (that is [`Writer::finish`]): it exists so a caller
    /// that wants to look at the output while the run continues -- a test, a
    /// monitoring job -- sees what it handed over, and so the row count is exact
    /// at that moment.
    pub fn sync(&mut self) -> Result<()> {
        if self.finished {
            return self.take_error();
        }
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        if self.tx.send(Job::Sync(ack_tx)).is_err() {
            // A stopped thread answers nothing; whatever it recorded is the reason.
            self.take_error()?;
            bail!("the writer thread has stopped without reporting a reason");
        }
        match ack_rx.recv() {
            Ok(Ok(n)) => self.rows_final = n.max(self.rows_final),
            Ok(Err(msg)) => return Err(anyhow::anyhow!("{msg}")),
            // The thread died instead of answering (it hit an error and stopped);
            // `take_error` below turns that into the error it recorded.
            Err(_) => {}
        }
        self.take_error()
    }

    /// Drain the queue, fsync everything and stop the thread.
    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut ack_error: Option<anyhow::Error> = None;
        if self.tx.send(Job::Finish(ack_tx)).is_ok() {
            match ack_rx.recv() {
                Ok(Ok(n)) => self.rows_final = n,
                Ok(Err(msg)) => ack_error = Some(anyhow::anyhow!("{msg}")),
                Err(_) => {} // the thread is gone; the stored error says why
            }
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let dropped = self.dropped_rows();
            if dropped > 0 {
                // Only when it happened: "dropped 0 rows, every file is missing exactly those
                // rows" is noise pretending to be information.
                eprintln!(
                    "writer: dropped {dropped} row(s) while the queue was full; every file of the \
                     set is missing exactly those rows, and row_index jumps across them in \
                     blocks.csv"
                );
            }
        eprintln!(
            "writer: {} rows handed over, deepest queue {} of {} rows ({} MB); \
             the pipeline thread never wrote or fsynced a row",
            self.rows_written(),
            self.max_outstanding,
            self.queue_depth,
            self.row_bytes * self.queue_depth / (1024 * 1024),
        );
        if let Some(e) = ack_error {
            return Err(e);
        }
        self.take_error()
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // A run that ended without asking (an error on the way out) still has to
        // close its files: an unflushed sidecar is a row nobody can interpret.
        if !self.finished {
            let _ = self.finish();
        }
    }
}


/// What to do with a row, given the day the current set belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// No set yet: this row's day names the first one.
    Open,
    /// Same day: write into the set that is open.
    Keep,
    /// A later day: close this set and open one for the new day.
    Roll,
    /// An *earlier* day than the set being written. The wall clock moved
    /// backwards (NTP step, timezone change). Keep writing, do not reopen a set
    /// that has been left: `row_index` is monotonic and a reader would have to
    /// merge two files to see one day's rows in order.
    WarnBackwards,
}

/// Decide what a row's local day means for the set being written.
///
/// Pure, and the only place the rule lives: dates in `YYYYmmdd` compare as
/// strings because the format is fixed-width and most significant first.
pub fn decide(current_day: Option<&str>, row_day: &str) -> Decision {
    match current_day {
        None => Decision::Open,
        Some(cur) if row_day == cur => Decision::Keep,
        Some(cur) if row_day > cur => Decision::Roll,
        Some(_) => Decision::WarnBackwards,
    }
}

/// Every set already present in `dir`, from their metadata and file lengths.
///
/// Every file set in `dir`, sorted by tag, with the naming read back from each
/// set's own metadata.
///
/// A set is recognised by its `<tag>-blocks.csv`, and the two halves its file
/// names are built from (`file_prefix` and `stamp`) come from that set's
/// metadata rather than from parsing the tag: a prefix is free text that may
/// itself contain the separator, so parsing would be a second, drifting
/// definition of what a name means. A set whose metadata predates this layout is
/// skipped -- this binary would open files that are not there.
pub fn discover_sets(dir: &Path, plan: &RunPlan) -> Result<Vec<FoundSet>> {
    let per_baseline_row = baseline_row_bytes(plan.n_freq) as u64;
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e).with_context(|| format!("listing {}", dir.display())),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(tag) = name.strip_suffix("-blocks.csv") else {
            continue;
        };
        let meta = dir.join(format!("{tag}-metadata.yaml"));
        let Some(stamp) = read_metadata_field(&meta, "stamp") else {
            continue; // laid out before the names changed; not ours to resume
        };
        let names = SetNames::new(
            &read_metadata_field(&meta, "file_prefix").unwrap_or_default(),
            &stamp,
        );
        let day = read_metadata_field(&meta, "set_day").unwrap_or_default();
        let kind = read_metadata_field(&meta, "set_kind").unwrap_or_default();
        // Complete rows: the shortest of the set's own files, as on resume.
        let mut rows = u64::MAX;
        for (j, k) in plan.baseline_pairs().iter() {
            let p = dir.join(names.baseline(&plan.enabled[*j].name, &plan.enabled[*k].name));
            let len = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            rows = rows.min(len / per_baseline_row.max(1));
        }
        if rows == u64::MAX {
            rows = 0;
        }
        let time_rows = file_len(&dir.join(names.time_bin()))? / 8;
        let block_rows =
            count_csv_rows(&dir.join(names.blocks()))? / plan.n_station.max(1) as u64;
        out.push(FoundSet {
            next_frame_id: read_last_block_first_frame(&dir.join(names.blocks()), plan.n_cum)?,
            prefix: tag.to_string(),
            names,
            day,
            kind,
            rows_written: rows.min(time_rows).min(block_rows),
        });
    }
    out.sort_by(|a, b| a.prefix.cmp(&b.prefix));
    Ok(out)
}

/// One `key: value` line out of a metadata file, if the file and the key exist.
pub fn read_metadata_field(path: &Path, key: &str) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let want = format!("{key}:");
    s.lines()
        .find_map(|l| l.strip_prefix(&want))
        .map(|v| v.trim().to_string())
}

/// Rewrite `ms.txt` so that it lists exactly the first `rows` timestamps of
/// `ms.bin`. The text file is a convenience view, never a source of truth.
/// One row's wall clock, as a person reads it: local date, time, milliseconds.
///
/// Local, not UTC: that is what the startup line prints and what an operator
/// comparing the log against the data is looking at.
///
/// Both the writing path (`write_one`) and the rebuilding path (`rebuild_ms_text`)
/// go through here on purpose. Two spellings of the same format is exactly how a
/// text view and a binary view drift apart, and this file has paid for that lesson
/// once already (the naming change of 2026-09-18).
///
/// `rem_euclid` rather than `%`: for an instant before 1970 the latter is negative,
/// and "-123" milliseconds is not a thing.
fn text_time(row_ms: i64) -> String {
    format!(
        "{}.{:03}",
        local_stamp_at(row_ms, "%Y-%m-%d %H:%M:%S"),
        row_ms.rem_euclid(1000)
    )
}

fn rebuild_ms_text(bin: &Path, txt: &Path, rows: u64) -> Result<()> {
    let mut s = String::new();
    let len = file_len(bin)?;
    if len > 0 && rows > 0 {
        let mut f = File::open(bin)?;
        let mut raw = vec![0u8; (rows * 8) as usize];
        f.read_exact(&mut raw)
            .with_context(|| format!("reading {}", bin.display()))?;
        for chunk in raw.chunks_exact(8) {
            let ms = i64::from_le_bytes(chunk.try_into().expect("8 bytes"));
            s.push_str(&text_time(ms));
            s.push('\n');
        }
    }
    std::fs::write(txt, s).with_context(|| format!("rewriting {}", txt.display()))?;
    Ok(())
}

fn open_rw(p: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(p)
        .with_context(|| format!("opening {}", p.display()))
}

fn file_len(p: &Path) -> Result<u64> {
    match std::fs::metadata(p) {
        Ok(m) => Ok(m.len()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e).with_context(|| format!("stat {}", p.display())),
    }
}

fn truncate_file(p: &Path, len: u64) -> Result<()> {
    if file_len(p)? == len {
        return Ok(());
    }
    let f = open_rw(p)?;
    f.set_len(len).with_context(|| format!("truncating {}", p.display()))?;
    Ok(())
}

/// `blocks.csv` has a one-line header, so the number of data rows is one less.
fn count_csv_rows(p: &Path) -> Result<u64> {
    let len = file_len(p)?;
    if len == 0 {
        return Ok(0);
    }
    let mut f = File::open(p)?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).count() as u64)
}

/// Keep the header and the first `lines` data lines.
fn truncate_csv(p: &Path, lines: u64) -> Result<()> {
    let len = file_len(p)?;
    if len == 0 {
        return Ok(());
    }
    let mut f = File::open(p)?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    let head: Vec<&str> = s.lines().filter(|l| l.starts_with('#')).collect();
    let data: Vec<&str> = s
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if data.len() as u64 == lines {
        return Ok(());
    }
    let mut out = String::new();
    for h in head {
        out.push_str(h);
        out.push('\n');
    }
    for l in data.iter().take(lines as usize) {
        out.push_str(l);
        out.push('\n');
    }
    std::fs::write(p, out).with_context(|| format!("rewriting {}", p.display()))?;
    Ok(())
}

/// First frame id of the last row in `blocks.csv`, so a resumed run continues the
/// global frame grid rather than starting a new one.
fn read_last_block_first_frame(p: &Path, n_cum: u32) -> Result<Option<u64>> {
    let len = file_len(p)?;
    if len == 0 {
        return Ok(None);
    }
    let mut s = String::new();
    File::open(p)?.read_to_string(&mut s)?;
    let last = s
        .lines()
        .rfind(|l| !l.is_empty() && !l.starts_with('#'));
    let Some(line) = last else { return Ok(None) };
    // Columns: row_index, station, first_frame_id, n_cum, valid, zero_filled.
    let mut it = line.split(',');
    let _row: u64 = it.next().unwrap_or("0").parse().unwrap_or(0);
    let _station: u64 = it.next().unwrap_or("0").parse().unwrap_or(0);
    let first: u64 = it.next().unwrap_or("0").parse().unwrap_or(0);
    Ok(Some(first + n_cum as u64))
}

/// Write one set's `metadata.yaml`.
///
/// Everything a reader needs, in the one file that belongs to that set: the
/// configuration and geometry the rows were produced with, and the set's own
/// identity -- which local day it holds, which global row it starts at, and how
/// it came to exist (docs/DESIGN-daily-files.md §4).
#[allow(clippy::too_many_arguments)] // one call site per set creation; a struct would not help
fn write_metadata(
    p: &Path,
    plan: &RunPlan,
    cfg: &Config,
    hash: &str,
    names: &SetNames,
    day: &str,
    kind: SetKind,
    continuation_of: Option<&str>,
    first_row: Option<(u64, u64)>,
    rows_at_open: u64,
) -> Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# Written once, when this file set is created. A resumed run checks");
    let _ = writeln!(s, "# the config_hash rather than rewriting geometry that the existing");
    let _ = writeln!(s, "# rows were produced with.");
    let _ = writeln!(s, "config_version: {}", cfg.version);
    let _ = writeln!(s, "config_hash: {hash}");
    let _ = writeln!(s, "format: raw-complex64");
    let _ = writeln!(s, "byte_order: little_endian");
    let _ = writeln!(s, "component: {{real: float32, imag: float32}}");
    let _ = writeln!(s, "baseline_layout: baseline_major");
    let _ = writeln!(s, "n_freq: {}", plan.n_freq);
    let _ = writeln!(s, "n_cum: {}", plan.n_cum);
    let _ = writeln!(s, "n_station: {}", plan.n_station);
    let _ = writeln!(s, "n_baseline: {}", plan.n_baseline);
    let _ = writeln!(s, "sample_rate_hz: {}", cfg.fft.sample_rate_hz);
    let _ = writeln!(s, "bin_width_hz: {}", plan.bin_width_hz);
    let _ = writeln!(s, "frame_rate_hz: {}", plan.frame_rate_hz);
    let _ = writeln!(s, "row_period_s: {}", plan.block_period_s);
    let _ = writeln!(s, "fft: {{type: R2C, length: {}, window: rectangular, nyquist: dropped}}",
                     plan.fft_len);
    let _ = writeln!(s, "fft_normalisation: none");
    let _ = writeln!(s, "row_normalisation: divide_by_n_cum");
    let _ = writeln!(s, "missing_frames: zero_filled");
    let _ = writeln!(s, "fftshift: false");
    let _ = writeln!(s, "bin0: DC");
    let _ = writeln!(s, "data_shift_count: {}", cfg.frontend.data_shift_count);
    let _ = writeln!(s, "frames_per_slot: {}", plan.frames_per_slot);
    let _ = writeln!(s, "rows_per_fetch: {}", plan.rows_per_fetch);
    let _ = writeln!(s, "baseline_row_bytes: {}", baseline_row_bytes(plan.n_freq));

    // --- this set's identity ---
    //
    // `file_prefix` and `stamp` are the two halves the *file names* are built
    // from, written separately because they cannot be told apart once joined: a
    // prefix is free text and may contain the separator. A reader needs them to
    // reconstruct a baseline's name, which no longer starts with the tag.
    let _ = writeln!(s, "# --- the file set this metadata belongs to ---");
    let _ = writeln!(s, "set_prefix: {}", names.tag());
    // The text view is derived, but it is derived into a format, and a reader should
    // not have to guess which one.
    let _ = writeln!(s, "rows_ms_txt_format: \"local YYYY-mm-DD HH:MM:SS.mmm\"");
    let _ = writeln!(s, "file_prefix: {}", names.file_prefix);
    let _ = writeln!(s, "stamp: {}", names.stamp);
    let _ = writeln!(s, "set_day: {day}");
    let _ = writeln!(s, "set_kind: {}", kind.as_str());
    if let Some(prev) = continuation_of {
        let _ = writeln!(s, "set_continuation_of: {prev}");
    }
    match first_row {
        Some((row_index, first_frame_id)) => {
            let _ = writeln!(s, "first_row_index: {row_index}");
            let _ = writeln!(s, "first_frame_id: {first_frame_id}");
        }
        None => {
            let _ = writeln!(s, "first_row_index: 0");
            let _ = writeln!(s, "first_frame_id: 0");
        }
    }
    let _ = writeln!(s, "rows_at_open: {rows_at_open}");
    let _ = writeln!(s, "local_timezone: {}", local_tz_name());
    let _ = writeln!(
        s,
        "local_utc_offset_s: {}",
        local_utc_offset_s(crate::sched::system_ms())
    );

    let _ = writeln!(s, "stations:");
    for st in &plan.enabled {
        let _ = writeln!(
            s,
            "  - {{index: {}, name: {}, group_ip: {}, dest_port: {}, board: {}}}",
            st.index,
            st.name,
            st.group_ip,
            st.dest_port,
            cfg.stations[st.cfg_index].board
        );
    }
    // Each baseline's actual file name, so a reader never has to reconstruct the
    // naming rule: the writer is the only thing that knows it for certain.
    let _ = writeln!(s, "baselines:");
    for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
        let _ = writeln!(
            s,
            "  - {{index: {}, j: {}, k: {}, file: {}}}",
            idx,
            j,
            k,
            names.baseline(&plan.enabled[*j].name, &plan.enabled[*k].name)
        );
    }
    std::fs::write(p, s).with_context(|| format!("writing {}", p.display()))?;
    Ok(())
}

fn check_metadata(p: &Path, hash: &str) -> Result<()> {
    let s = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    let found = s
        .lines()
        .find_map(|l| l.strip_prefix("config_hash: "))
        .unwrap_or("")
        .trim()
        .to_string();
    if found != hash {
        bail!(
            "{} was written with config_hash {found} but this run has {hash}; resuming \
             would misinterpret every existing row",
            p.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn test_plan_and_cfg() -> (Config, RunPlan) {
        let (mut cfg, _) = shipped();
        // These tests pin the *legacy* layout -- no prefix, one set for ever --
        // which is still a supported mode (`storage.daily_files: false`). The
        // daily sets have their own tests below, and an end-to-end one in
        // tools/daily_rollover_test.sh where a timezone can be set for the
        // process instead of globally.
        cfg.storage.daily_files = false;
        let plan = cfg.compile().unwrap();
        (cfg, plan)
    }

    /// The shipped test configuration, without touching the daily-files switch.
    fn shipped() -> (Config, RunPlan) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("config/correlator-test.yaml");
        let cfg = Config::from_yaml_file(&path).unwrap();
        let plan = cfg.compile().unwrap();
        (cfg, plan)
    }

    /// The same, with daily file sets on.
    fn test_plan_and_cfg_daily() -> (Config, RunPlan) {
        let (mut cfg, _) = shipped();
        cfg.storage.daily_files = true;
        let plan = cfg.compile().unwrap();
        (cfg, plan)
    }

    /// Every set in `dir`, by tag, with its (time.bin length, blocks rows).
    ///
    /// `time.bin` is an unbuffered file, so its length is exact at any moment.
    /// `blocks.csv` goes through a `BufWriter` and is only flushed when the set is
    /// closed or the fsync cadence comes round, so for a set still being written
    /// the second number lags by up to a buffer -- which is why the assertions
    /// below take it after `finish()`.
    fn sets_on_disk(dir: &Path, plan: &RunPlan) -> Vec<(String, u64, u64)> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            let Some(tag) = name.strip_suffix("-blocks.csv") else {
                continue;
            };
            let prefix = tag.to_string();
            let ms = std::fs::metadata(dir.join(format!("{tag}-time.bin")))
                .map(|m| m.len())
                .unwrap_or(0)
                / 8;
            let blocks = std::fs::read_to_string(dir.join(format!("{tag}-blocks.csv")))
                .map(|s| s.lines().count() as u64)
                .unwrap_or(0)
                / plan.n_station.max(1) as u64;
            out.push((prefix, ms, blocks));
        }
        out.sort();
        out
    }

    /// The naming a set with this tag was written with, read from its metadata.
    fn names_for(dir: &Path, tag: &str) -> SetNames {
        let meta = dir.join(format!("{tag}-metadata.yaml"));
        SetNames::new(
            &read_metadata_field(&meta, "file_prefix").unwrap_or_default(),
            &read_metadata_field(&meta, "stamp").expect("stamp is in the metadata"),
        )
    }

    /// The tag of the only set in `dir` (the tests here write one).
    fn only_tag(dir: &Path) -> String {
        let mut tags: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                n.strip_suffix("-metadata.yaml").map(str::to_string)
            })
            .collect();
        tags.sort();
        assert_eq!(tags.len(), 1, "expected one set in {}, found {tags:?}", dir.display());
        tags.pop().unwrap()
    }

    fn names_of(dir: &Path) -> SetNames {
        names_for(dir, &only_tag(dir))
    }

    /// One baseline's file, by index into `plan.baseline_pairs()`.
    fn baseline_path(dir: &Path, plan: &RunPlan, idx: usize) -> PathBuf {
        let (j, k) = plan.baseline_pairs()[idx];
        dir.join(names_of(dir).baseline(&plan.enabled[j].name, &plan.enabled[k].name))
    }

    fn time_bin_path(dir: &Path) -> PathBuf {
        dir.join(names_of(dir).time_bin())
    }

    fn time_txt_path(dir: &Path) -> PathBuf {
        dir.join(names_of(dir).time_txt())
    }

    fn blocks_path(dir: &Path) -> PathBuf {
        dir.join(names_of(dir).blocks())
    }

    fn metadata_path(dir: &Path) -> PathBuf {
        dir.join(names_of(dir).metadata())
    }

/// One baseline of a named set, for the tests that write two sets in one run.
    fn baseline_path_for_tag(dir: &Path, plan: &RunPlan, tag: &str, idx: usize) -> PathBuf {
        let (j, k) = plan.baseline_pairs()[idx];
        dir.join(names_for(dir, tag).baseline(&plan.enabled[j].name, &plan.enabled[k].name))
    }

    fn baseline_paths_for_tag(dir: &Path, plan: &RunPlan, tag: &str) -> Vec<PathBuf> {
        (0..plan.n_baseline)
            .map(|i| baseline_path_for_tag(dir, plan, tag, i))
            .collect()
    }


    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "corr-writer-test-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A row of recognisable bytes: baseline b, bin c, real/imag.
    fn make_row(plan: &RunPlan, row: u64) -> Vec<u8> {
        let row_bytes = baseline_row_bytes(plan.n_freq);
        let mut v = vec![0u8; row_bytes * plan.n_baseline];
        for b in 0..plan.n_baseline {
            for c in 0..plan.n_freq as usize {
                let off = (b * row_bytes) + c * 8;
                let re = (row as u32 * 1000 + b as u32) as f32 + c as f32 / 1024.0;
                let im = -(re) - 0.5;
                v[off..off + 4].copy_from_slice(&re.to_le_bytes());
                v[off + 4..off + 8].copy_from_slice(&im.to_le_bytes());
            }
        }
        v
    }

    fn blocks_for(plan: &RunPlan, row: u64, valid: u64) -> RowBlock {
        RowBlock {
            row_index: row,
            first_frame_id: row * plan.n_cum as u64,
            n_cum: plan.n_cum,
            valid_frames_per_station: vec![valid; plan.n_station],
        }
    }

    // -----------------------------------------------------------------------
    // daily file sets
    // -----------------------------------------------------------------------

    /// The text view answers "when was this row" for a person: local time with
    /// milliseconds, and both paths that write it share one formatter.
    #[test]
    fn the_text_sidecar_is_local_time_with_milliseconds() {
        let t = 1_789_732_717_123_i64;
        let s = text_time(t);
        assert!(
            s.starts_with(&local_stamp_at(t, "%Y-%m-%d %H:%M:%S")),
            "{s} should start with the local date and time"
        );
        assert!(s.ends_with(".123"), "{s}");
        assert_eq!(s.len(), 23, "{s} is YYYY-mm-DD HH:MM:SS.mmm");
        assert!(text_time(t + 1).ends_with(".124"), "{}", text_time(t + 1));
        // Pre-1970 instants must not produce "-123" milliseconds.
        assert!(text_time(-1).ends_with(".999"), "{}", text_time(-1));
        assert!(text_time(0).ends_with(".000"), "{}", text_time(0));
    }

    /// A full queue drops whole rows and keeps going instead of stalling the
    /// pipeline: blocking there is what turns a slow disk into lost packets.
    #[test]
    fn a_full_queue_drops_whole_rows_and_keeps_receiving() {
        let (mut cfg, plan) = test_plan_and_cfg();
        // A two-row queue, so "full" is reachable deterministically.
        let row_bytes = baseline_row_bytes(plan.n_freq) * plan.n_baseline;
        cfg.storage.queue_bytes = row_bytes as u64 * 2;
        let d = TempDir::new("queue-full");
        let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        assert_eq!(w.queue_depth, 2, "this test needs a shallow queue");

        // Swap the handle's channel for one that is already full, so the outcome does
        // not depend on how fast the writer thread happens to be. `_rx` stays alive so
        // the channel does not look disconnected.
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        tx.send(Job::Row {
            buf: std::sync::Arc::new(RowBuf::new(row_bytes, false).expect("row buffer")),
            offset: 0,
            len: row_bytes,
            row_ms: 0,
            blocks: blocks_for(&plan, 0, plan.n_cum as u64),
        })
        .expect("room for one");
        w.tx = tx;

        let t0 = midday_now();
        for i in 0..3u64 {
            w.write_row(
                &make_row(&plan, i),
                t0 + i as i64,
                &blocks_for(&plan, i, plan.n_cum as u64),
            )
            .expect("a full queue is no longer an error: the row is dropped instead");
        }
        assert_eq!(w.dropped_rows(), 3, "three rows offered, none fit");

        // Whole rows only: nothing reached the writer thread, so nothing was written,
        // and in particular no file holds a fraction of a row. This is the property
        // that makes a drop safe to read downstream -- every file of the set is
        // missing exactly the same rows, and `row_index` jumps across them.
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 1, "{sets:?}");
        assert_eq!((sets[0].1, sets[0].2), (0, 0), "no rows in the sidecars: {sets:?}");
        for p in baseline_paths_for_tag(d.path(), &plan, &sets[0].0) {
            assert_eq!(
                std::fs::metadata(&p).unwrap().len(),
                0,
                "{} must be empty: a dropped row writes to no baseline at all",
                p.display()
            );
        }
    }

    /// A reader watching `-time.txt` has to see rows as they are written. The file
    /// goes through an 8 KiB `BufWriter` whose natural flush is 341 rows (~5.7
    /// minutes), which reads as "the text view stopped growing".
    #[test]
    fn the_text_sidecar_keeps_up_with_the_rows() {
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("sidecar-live");
        let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        let t0 = midday_now();
        w.write_row(&make_row(&plan, 0), t0, &blocks_for(&plan, 0, plan.n_cum as u64))
            .unwrap();

        let tag = only_tag(d.path());
        let txt = d.path().join(format!("{tag}-time.txt"));
        let count = || {
            std::fs::read_to_string(&txt)
                .map(|s| s.lines().count())
                .unwrap_or(9999)
        };
        // Poll without syncing: that is the property under test.
        let mut before = 0;
        for _ in 0..300 {
            before = count();
            if before > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // The same must hold for `blocks.csv`, which is also a `BufWriter`: one line
        // per station, visible without a sync. Read both before syncing, or the test
        // would be measuring the sync.
        let blocks = std::fs::read_to_string(d.path().join(format!("{tag}-blocks.csv")))
            .map(|s| s.lines().count())
            .unwrap_or(9999);
        // `write_row` only surfaces a writer-thread error on the *next* call.
        let err = w.take_error().err().map(|e| format!("{e:#}"));
        w.sync().unwrap();
        let after = count();
        assert_eq!(err, None, "the writer thread failed");
        assert_eq!(before, 1, "rows_ms.txt: one line without syncing");
        assert_eq!(blocks, plan.n_station, "blocks.csv: one line per station, no sync");
        assert_eq!(after, 1, "and still one line after a sync");
    }

    /// A wall-clock instant on the current local day, kept well away from local
    /// midnight so that "two rows on one day" cannot race the day boundary.
    fn midday_now() -> i64 {
        let mut t = crate::sched::system_ms();
        let hour: u32 = local_stamp_at(t, "%H").parse().unwrap_or(12);
        if hour >= 23 || hour == 0 {
            t -= 3 * 3600 * 1000;
        }
        t
    }

    #[test]
    fn the_day_rule_is_open_keep_roll_or_warn() {
        // No set yet: this row's day names the first one.
        assert_eq!(decide(None, "20260916"), Decision::Open);
        assert_eq!(decide(Some("20260916"), "20260916"), Decision::Keep);
        // A later day rolls. Fixed-width YYYYmmdd compares as a string, which is
        // the whole reason the format is that shape -- checked across a month and
        // a year boundary, where a naive comparison goes wrong.
        assert_eq!(decide(Some("20260916"), "20260917"), Decision::Roll);
        assert_eq!(decide(Some("20260930"), "20261001"), Decision::Roll);
        assert_eq!(decide(Some("20261231"), "20270101"), Decision::Roll);
        // An earlier day never reopens a set: the clock went backwards.
        assert_eq!(decide(Some("20260917"), "20260916"), Decision::WarnBackwards);
        assert_eq!(decide(Some("20270101"), "20261231"), Decision::WarnBackwards);
    }

    #[test]
    fn the_first_rows_local_day_names_the_set_and_a_new_day_opens_another() {
        let (cfg, plan) = test_plan_and_cfg_daily();
        let d = TempDir::new("daily-roll");
        let (mut w, st) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        assert_eq!(st.rows_written, 0);
        // The name has to come from the first row's timestamp, so no set can exist
        // before that row is in hand.
        assert!(w.set_prefix().is_empty(), "no set before the first row");
        assert!(
            w.current_set_tag().is_empty(),
            "and nothing is being written yet"
        );
        assert!(sets_on_disk(d.path(), &plan).is_empty());

        let t0 = midday_now();
        w.write_row(&make_row(&plan, 0), t0, &blocks_for(&plan, 0, plan.n_cum as u64))
            .unwrap();
        // The set is opened by the writer thread now, so its name is read from the
        // directory rather than from the handle: the files are the fact, and the
        // handle's own copy is only the set this process started on.
        w.sync().unwrap();
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 1, "{sets:?}");
        let p1 = sets[0].0.clone();
        let fmt = cfg.storage.file_prefix_format.clone();
        assert_eq!(p1, local_stamp_at(t0, &fmt), "prefix is the first row's local time");
        // The handle follows the thread, which is what the live status reads: this
        // is the accessor that answers "which set is being written now", as opposed
        // to `set_prefix`, which is fixed at open.
        assert_eq!(w.current_set_tag(), p1);
        let meta1 = d.path().join(format!("{p1}-metadata.yaml"));
        assert_eq!(read_metadata_field(&meta1, "set_day").as_deref(), Some(local_day_at(t0).as_str()));
        assert_eq!(read_metadata_field(&meta1, "set_kind").as_deref(), Some("start"));
        assert_eq!(read_metadata_field(&meta1, "set_prefix").as_deref(), Some(p1.as_str()));
        assert_eq!(read_metadata_field(&meta1, "first_row_index").as_deref(), Some("0"));
        assert_eq!(read_metadata_field(&meta1, "first_frame_id").as_deref(), Some("0"));
        // Every baseline file of the set exists with exactly one row in it.
        for p in baseline_paths_for_tag(d.path(), &plan, &p1) {
            assert_eq!(
                std::fs::metadata(&p).unwrap().len(),
                baseline_row_bytes(plan.n_freq) as u64,
                "{} should hold one row",
                p.display()
            );
        }

        // A second row on the same local day stays in the same set.
        w.write_row(&make_row(&plan, 1), t0 + 1000, &blocks_for(&plan, 1, plan.n_cum as u64))
            .unwrap();
        w.sync().unwrap();
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 1, "same day, same set: {sets:?}");
        assert_eq!(sets[0].0, p1, "and the set keeps its name");

        // A row on the next local day opens a second set, and closes the first.
        let t1 = t0 + 24 * 3600 * 1000 + 60_000;
        if local_day_at(t1) == local_day_at(t0) {
            // Only reachable with a pathological timezone; nothing to assert.
            return;
        }
        w.write_row(&make_row(&plan, 2), t1, &blocks_for(&plan, 2, plan.n_cum as u64))
            .unwrap();
        w.sync().unwrap();
        let sets = sets_on_disk(d.path(), &plan);
        let p2 = sets
            .iter()
            .map(|(p, _, _)| p.clone())
            .find(|p| *p != p1)
            .expect("the new day opened a second set");
        assert_ne!(p1, p2);
        assert_eq!(p2, local_stamp_at(t1, &fmt));
        // The rollover happened in the writer thread; the handle reports the new set
        // without the caller doing anything about it.
        assert_eq!(w.current_set_tag(), p2);
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 2, "{sets:?}");
        let s1 = sets.iter().find(|(p, _, _)| *p == p1).unwrap();
        let s2 = sets.iter().find(|(p, _, _)| *p == p2).unwrap();
        assert_eq!((s1.1, s1.2), (2, 2), "the closed set keeps its two rows");
        assert_eq!(s2.1, 1, "the new set starts at row one");
        w.finish().unwrap();
        let sets = sets_on_disk(d.path(), &plan);
        let s2 = sets.iter().find(|(p, _, _)| *p == p2).unwrap();
        assert_eq!((s2.1, s2.2), (1, 1), "and its sidecars agree once flushed");

        let meta2 = d.path().join(format!("{p2}-metadata.yaml"));
        assert_eq!(read_metadata_field(&meta2, "set_kind").as_deref(), Some("rollover"));
        assert_eq!(
            read_metadata_field(&meta2, "set_continuation_of").as_deref(),
            Some(p1.as_str())
        );
        assert_eq!(read_metadata_field(&meta2, "first_row_index").as_deref(), Some("2"));
        assert_eq!(
            read_metadata_field(&meta2, "first_frame_id").as_deref(),
            Some((2 * plan.n_cum as u64).to_string().as_str())
        );
        // row_index must not restart per set: it is the global grid.
        let b2 = std::fs::read_to_string(d.path().join(format!("{p2}-blocks.csv"))).unwrap();
        assert!(b2.starts_with("2,0,"), "row_index continues across sets: {b2:?}");
        // ... and the first row of the new set is on the new local day.
        assert_eq!(local_day_at(t1), local_day_at(t0 + 24 * 3600 * 1000 + 60_000));
    }

    #[test]
    fn a_rollover_that_cannot_open_its_set_stops_with_that_reason() {
        // The day boundary now happens in the writer thread, so this is the path
        // that has to keep its *type* across the thread boundary: the pipeline
        // downcasts `RolloverFailed` to stop the run with a reason the report
        // carries, and a stringified error would end the run without saying why.
        let (cfg, plan) = test_plan_and_cfg_daily();
        let d = TempDir::new("daily-roll-fail");
        let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        let t0 = midday_now();
        w.write_row(&make_row(&plan, 0), t0, &blocks_for(&plan, 0, plan.n_cum as u64))
            .unwrap();
        w.sync().unwrap();
        let p1 = sets_on_disk(d.path(), &plan)[0].0.clone();

        let t1 = t0 + 24 * 3600 * 1000 + 60_000;
        if local_day_at(t1) == local_day_at(t0) {
            return; // pathological timezone, as in the rollover test above
        }
        // The new day's set cannot be created: something already owns that name.
        let next = local_stamp_at(t1, &cfg.storage.file_prefix_format);
        let names = SetNames::new(&cfg.storage.file_prefix, &next);
        // The name the new day's set would take, so the squatter is exactly the
        // file the rollover probes for.
        let (j0, k0) = plan.baseline_pairs().first().copied().unwrap_or((0, 0));
        let squatter = d
            .path()
            .join(names.baseline(&plan.enabled[j0].name, &plan.enabled[k0].name));
        std::fs::write(&squatter, b"not ours").unwrap();

        // The thread hits it on the row that crosses the boundary, so that row is
        // not written; the caller is told at the next `write_row` (or here, at
        // `finish`, which waits for the thread and then reports what it hit --
        // that join is what makes this test deterministic rather than a race
        // against the thread).
        w.write_row(&make_row(&plan, 1), t1, &blocks_for(&plan, 1, plan.n_cum as u64))
            .unwrap();
        let err = w.finish().unwrap_err();
        let rf = err
            .downcast_ref::<RolloverFailed>()
            .unwrap_or_else(|| panic!("typed, not a string: {err:#}"));
        assert_eq!(rf.prefix, names.tag());
        assert!(rf.detail.contains("already holds"), "{:?}", rf.detail);

        // And the set that was left is closed cleanly, with the rows it did get:
        // the boundary is not a reason to lose the day that ended.
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 1, "{sets:?}");
        assert_eq!(sets[0].0, p1);
        assert_eq!((sets[0].1, sets[0].2), (1, 1), "the crossing row was not written");
        assert_eq!(std::fs::read(&squatter).unwrap(), b"not ours");
    }

    #[test]
    fn a_restart_appends_to_the_days_set_instead_of_opening_a_second() {
        let (cfg, plan) = test_plan_and_cfg_daily();
        let d = TempDir::new("daily-resume");
        let t0 = midday_now();
        let p1;
        {
            let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
            w.write_row(&make_row(&plan, 0), t0, &blocks_for(&plan, 0, plan.n_cum as u64))
                .unwrap();
            w.write_row(&make_row(&plan, 1), t0 + 1000, &blocks_for(&plan, 1, plan.n_cum as u64))
                .unwrap();
            w.finish().unwrap();
            p1 = sets_on_disk(d.path(), &plan)[0].0.clone();
        }
        // A new process on the same day finds that set and appends.
        let (mut w2, st2) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        assert_eq!(st2.rows_written, 2);
        assert_eq!(w2.set_prefix(), p1, "same prefix, not a second set");
        assert_eq!(w2.set_kind(), SetKind::Resume);
        assert_eq!(st2.next_frame_id, Some(2 * plan.n_cum as u64));
        w2.write_row(&make_row(&plan, 2), t0 + 2000, &blocks_for(&plan, 2, plan.n_cum as u64))
            .unwrap();
        assert_eq!(sets_on_disk(d.path(), &plan).len(), 1, "still one set");
        w2.finish().unwrap();
        let s = sets_on_disk(d.path(), &plan);
        assert_eq!((s[0].1, s[0].2), (3, 3), "the row was appended, not written over");
        // Appending must not rewrite the set's origin.
        let meta = d.path().join(format!("{p1}-metadata.yaml"));
        assert_eq!(read_metadata_field(&meta, "set_kind").as_deref(), Some("start"));
        assert_eq!(read_metadata_field(&meta, "first_row_index").as_deref(), Some("0"));
    }

    #[test]
    fn a_changed_configuration_is_refused_and_the_way_out_is_named() {
        let (cfg, plan) = test_plan_and_cfg_daily();
        let d = TempDir::new("daily-hash");
        let t0 = midday_now();
        let p1;
        {
            let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
            w.write_row(&make_row(&plan, 0), t0, &blocks_for(&plan, 0, plan.n_cum as u64))
                .unwrap();
            w.sync().unwrap();
            p1 = sets_on_disk(d.path(), &plan)[0].0.clone();
        }
        // Same day, different geometry: appending would mix two geometries into
        // one set, so it is refused -- with the two ways out in the message.
        let err = Writer::open(d.path(), &plan, &cfg, "hash-b", 1).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("--allow-second-set"), "{msg}");
        assert!(msg.contains("--set"), "{msg}");
        assert!(msg.contains(&p1), "the message names the set it refused: {msg}");

        // --allow-second-set opens a second set for the same day.
        let (mut w2, st2) =
            Writer::open_with(d.path(), &plan, &cfg, "hash-b", 1, &SetPolicy::NewSet).unwrap();
        assert_eq!(st2.rows_written, 0);
        w2.write_row(&make_row(&plan, 0), t0 + 1000, &blocks_for(&plan, 0, plan.n_cum as u64))
            .unwrap();
        assert_ne!(w2.set_prefix(), p1.as_str());
        w2.sync().unwrap();
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 2, "{sets:?}");
        let p2 = sets
            .iter()
            .map(|(p, _, _)| p.clone())
            .find(|p| *p != p1)
            .expect("the second set has the other prefix");
        let meta2 = d.path().join(format!("{p2}-metadata.yaml"));
        assert_eq!(read_metadata_field(&meta2, "set_kind").as_deref(), Some("second_set"));

        // --set <prefix> appends to the named set instead.
        let (w3, st3) =
            Writer::open_with(d.path(), &plan, &cfg, "hash-a", 1, &SetPolicy::Explicit(p1.clone()))
                .unwrap();
        assert_eq!(st3.rows_written, 1);
        assert_eq!(w3.set_prefix(), p1);

        // Auto appends to the *latest* set of the day, and it has to: the older one
        // is the set the changed configuration replaced, so its hash is the one that
        // no longer matches. Choosing the earliest made every run after a
        // `--allow-second-set` fail with "resuming would misinterpret existing rows".
        drop(w3);
        drop(w2);
        let (w4, st4) =
            Writer::open_with(d.path(), &plan, &cfg, "hash-b", 1, &SetPolicy::Auto).unwrap();
        assert_eq!(st4.rows_written, 1, "the second set is the one being written");
        assert_eq!(w4.current_set_tag(), p2);
    }

    #[test]
    fn a_wall_clock_that_goes_backwards_keeps_writing_the_current_set() {
        let (cfg, plan) = test_plan_and_cfg_daily();
        let d = TempDir::new("daily-backwards");
        let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        let t0 = midday_now();
        w.write_row(&make_row(&plan, 0), t0, &blocks_for(&plan, 0, plan.n_cum as u64))
            .unwrap();
        // The name comes from the first row's timestamp and is decided in the
        // writer thread, so it is read back from the directory.
        w.sync().unwrap();
        let p1 = sets_on_disk(d.path(), &plan)[0].0.clone();
        // A row stamped a day earlier (NTP step, timezone change) must not reopen
        // a set that has been left.
        w.write_row(&make_row(&plan, 1), t0 - 24 * 3600 * 1000, &blocks_for(&plan, 1, plan.n_cum as u64))
            .unwrap();
        w.sync().unwrap();
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 1, "the earlier stamp opened nothing: {sets:?}");
        assert_eq!(sets[0].0, p1, "and the row stayed in the current set");
        w.finish().unwrap();
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 1);
        assert_eq!((sets[0].1, sets[0].2), (2, 2));
    }

    #[test]
    fn a_fresh_run_writes_every_file_and_is_resumable_from_a_record_of_it() {
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("fresh");
        let (mut w, st) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        assert!(st.is_fresh());
        assert_eq!(st.truncated_rows, 0);

        for row in 0..3u64 {
            w.write_row(&make_row(&plan, row), 1_700_000_000_000 + row as i64,
                        &blocks_for(&plan, row, plan.n_cum as u64)).unwrap();
        }
        w.finish().unwrap();
        assert_eq!(w.rows_written(), 3);

        // Every baseline file holds exactly three rows.
        let row_bytes = baseline_row_bytes(plan.n_freq) as u64;
        for idx in 0..plan.n_baseline {
            let p = baseline_path(d.path(), &plan, idx);
            assert_eq!(std::fs::metadata(&p).unwrap().len(), 3 * row_bytes, "{}", p.display());
        }
        assert_eq!(file_len(&time_bin_path(d.path())).unwrap(), 3 * 8);
        // One line per station per row.
        assert_eq!(
            count_csv_rows(&blocks_path(d.path())).unwrap(),
            3 * plan.n_station as u64
        );
        assert!(metadata_path(d.path()).exists());

        // The content is what we wrote, baseline-major and row by row.
        let mut buf = Vec::new();
        File::open(baseline_path(d.path(), &plan, 0))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let expect = make_row(&plan, 0);
        assert_eq!(&buf[..row_bytes as usize], &expect[..row_bytes as usize]);
    }

    #[test]
    fn reopening_resumes_at_the_last_complete_row() {
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("resume");
        {
            let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
            for row in 0..2u64 {
                w.write_row(&make_row(&plan, row), 1000, &blocks_for(&plan, row, plan.n_cum as u64))
                    .unwrap();
            }
            w.finish().unwrap();
        }
        let (w2, st) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        assert_eq!(st.rows_written, 2, "two complete rows should be found");
        assert_eq!(st.next_frame_id, Some(2 * plan.n_cum as u64));
        assert_eq!(w2.rows_written(), 2);
    }

    #[test]
    fn a_half_written_row_is_truncated_rather_than_left_to_be_misread() {
        // The failure this exists for: the raw files carry no framing, so a
        // partial row is invisible downstream. Reopening must cut it off.
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("half");
        let row_bytes = baseline_row_bytes(plan.n_freq) as u64;
        {
            let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
            for row in 0..2u64 {
                w.write_row(&make_row(&plan, row), 1000, &blocks_for(&plan, row, plan.n_cum as u64))
                    .unwrap();
            }
            w.finish().unwrap();
        }
        // Simulate a crash halfway through the third row, and after the spectrum
        // was flushed but before the sidecars caught up.
        let p0 = baseline_path(d.path(), &plan, 0);
        let f = OpenOptions::new().write(true).open(&p0).unwrap();
        f.set_len(2 * row_bytes + row_bytes / 2).unwrap();
        drop(f);

        let (_w, st) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        assert_eq!(st.rows_written, 2);
        assert_eq!(st.truncated_rows, 0, "the sidecars only had two rows anyway");
        assert_eq!(std::fs::metadata(&p0).unwrap().len(), 2 * row_bytes);
    }

    #[test]
    fn sidecars_that_ran_ahead_do_not_create_rows_that_are_not_there() {
        // The other direction: if the sidecars somehow have more rows than the
        // data, the shortest file wins and the extra sidecar rows are dropped.
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("ahead");
        {
            let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
            // Claim four rows but only hand over one row of spectrum.
            for row in 0..4u64 {
                w.write_row(&make_row(&plan, row), 1000, &blocks_for(&plan, row, plan.n_cum as u64))
                    .unwrap();
            }
            w.finish().unwrap();
        }
        // Cut every baseline file back to a single row, as a crash mid-write would.
        let row_bytes = baseline_row_bytes(plan.n_freq) as u64;
        for idx in 0..plan.n_baseline {
            let p = baseline_path(d.path(), &plan, idx);
            let f = OpenOptions::new().write(true).open(&p).unwrap();
            f.set_len(row_bytes).unwrap();
        }

        let (_w, st) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        assert_eq!(st.rows_written, 1, "the data files are the truth");
        assert_eq!(st.truncated_rows, 0, "the data file was already short");
        assert_eq!(file_len(&time_bin_path(d.path())).unwrap(), 8);
        assert_eq!(
            count_csv_rows(&blocks_path(d.path())).unwrap(),
            plan.n_station as u64
        );
    }

    #[test]
    fn resuming_appends_to_the_sidecars_instead_of_overwriting_them() {
        // Regression guard. The baseline files were seeked to the right offset on
        // resume but the sidecars were not, so a resumed run overwrote its own
        // timestamps and block records from position zero -- measured, 98 rows of
        // spectrum with 49 timestamps.
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("sidecar-append");
        {
            let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
            for row in 0..3u64 {
                w.write_row(&make_row(&plan, row), 1000 + row as i64,
                            &blocks_for(&plan, row, plan.n_cum as u64))
                    .unwrap();
            }
            w.finish().unwrap();
        }
        {
            let (mut w, st) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
            assert_eq!(st.rows_written, 3);
            for row in 3..5u64 {
                w.write_row(&make_row(&plan, row), 2000 + row as i64,
                            &blocks_for(&plan, row, plan.n_cum as u64))
                    .unwrap();
            }
            w.finish().unwrap();
        }

        let ms = std::fs::read(time_bin_path(d.path())).unwrap();
        assert_eq!(ms.len(), 5 * 8, "five timestamps, not three");
        let stamps: Vec<i64> = ms
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(stamps, vec![1000, 1001, 1002, 2003, 2004], "the old rows survive");

        let txt = std::fs::read_to_string(time_txt_path(d.path())).unwrap();
        assert_eq!(txt.lines().count(), 5);
        // The format itself is pinned by `the_text_sidecar_is_local_time_with_milliseconds`;
        // what matters here is that the text view follows the binary one.
        assert_eq!(txt.lines().next().unwrap(), text_time(1000));

        assert_eq!(
            count_csv_rows(&blocks_path(d.path())).unwrap(),
            5 * plan.n_station as u64
        );
        let row_bytes = baseline_row_bytes(plan.n_freq) as u64;
        for idx in 0..plan.n_baseline {
            let p = baseline_path(d.path(), &plan, idx);
            assert_eq!(std::fs::metadata(&p).unwrap().len(), 5 * row_bytes);
        }
    }

    #[test]
    fn resuming_with_a_different_configuration_is_refused() {
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("hash");
        {
            let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
            w.write_row(&make_row(&plan, 0), 1000, &blocks_for(&plan, 0, plan.n_cum as u64))
                .unwrap();
            w.finish().unwrap();
        }
        let err = Writer::open(d.path(), &plan, &cfg, "hash-b", 1).unwrap_err().to_string();
        assert!(err.contains("config_hash"), "unexpected error: {err}");
    }

    #[test]
    fn a_row_of_the_wrong_size_is_refused_rather_than_written_short() {
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("short");
        let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        let err = w.write_row(&[0u8; 16], 1000, &blocks_for(&plan, 0, plan.n_cum as u64));
        assert!(err.is_err(), "a short spectrum buffer must be refused");
        assert_eq!(w.rows_written(), 0);

        let mut short = blocks_for(&plan, 0, plan.n_cum as u64);
        short.valid_frames_per_station.pop();
        let err = w.write_row(&make_row(&plan, 0), 1000, &short);
        assert!(err.is_err(), "missing per-station records must be refused");
        assert_eq!(w.rows_written(), 0);
    }

    #[test]
    fn blocks_csv_records_the_frame_range_and_the_missing_frame_count() {
        let (cfg, plan) = test_plan_and_cfg();
        let d = TempDir::new("blocks");
        let (mut w, _) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        let valid = plan.n_cum as u64 - 7;
        w.write_row(&make_row(&plan, 0), 4242, &blocks_for(&plan, 0, valid)).unwrap();
        w.finish().unwrap();

        let s = std::fs::read_to_string(blocks_path(d.path())).unwrap();
        let lines: Vec<&str> = s.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(lines.len(), plan.n_station, "one line per station");
        let f: Vec<&str> = lines[0].split(',').collect();
        assert_eq!(f[0], "0", "row index");
        assert_eq!(f[1], "0", "station index");
        assert_eq!(f[2], "0", "first frame id");
        assert_eq!(f[3], plan.n_cum.to_string(), "n_cum");
        assert_eq!(f[4], valid.to_string(), "valid frames");
        assert_eq!(f[5], "7", "zero filled frames");

        let ms = std::fs::read(time_bin_path(d.path())).unwrap();
        assert_eq!(i64::from_le_bytes(ms[..8].try_into().unwrap()), 4242);
        let txt = std::fs::read_to_string(time_txt_path(d.path())).unwrap();
        assert_eq!(txt.trim(), text_time(4242));
    }
}
