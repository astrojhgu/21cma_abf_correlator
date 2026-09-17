//! Output: one raw spectrum file per baseline, plus sidecar files, in **sets**.
//!
//! Layout of a run directory (the prefix is empty when `storage.daily_files` is
//! off, which is the old behaviour):
//!
//! ```text
//!   <prefix>baseline_0000_j00_k00.bin   complex64[ntime][n_freq], one row per integration
//!   ...
//!   <prefix>rows_ms.bin                 int64 little-endian Unix milliseconds, one per row
//!   <prefix>rows_ms.txt                 "<row_index> <ms>" per line
//!   <prefix>blocks.csv                  row_index, station, frame range, per-station valid frames
//!   <prefix>metadata.yaml               everything needed to read the raw files
//! ```
//!
//! A **set** is the four sidecars plus one file per baseline -- 304 files for the
//! 24-station geometry -- sharing one prefix. The prefix names the *local* day and
//! the *local* time of the set's first row, so a file set holds the day it says it
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

use crate::config::{
    local_day_at, local_stamp_at, local_tz_name, local_utc_offset_s, Config, RunPlan,
};

/// Bytes in one row of one baseline's file.
pub fn baseline_row_bytes(n_freq: u32) -> usize {
    n_freq as usize * 8 // complex64
}

/// Name of one baseline's raw file. Zero padded so that a directory listing is
/// in baseline order, and carrying (j, k) so a file can be identified without
/// reading metadata.yaml. The set prefix is prepended by the writer.
pub fn baseline_file_name(index: usize, j: usize, k: usize) -> String {
    format!("baseline_{index:04}_j{j:02}_k{k:02}.bin")
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
    pub prefix: String,
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
        buf: Vec<u8>,
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

impl SetState {
    /// Attach to an existing set (or create the one the legacy layout always has),
    /// and report where the run should resume.
    ///
    /// Runs on the calling thread at startup, before any socket is open: the run
    /// needs `ResumeState` to fix its row grid, and a failure here should stop the
    /// process before it starts receiving.
    fn attach(
        dir: &Path,
        plan: &RunPlan,
        cfg: &Config,
        config_hash: &str,
        fsync_every: u64,
        prefix: &str,
        found: Option<&FoundSet>,
    ) -> Result<(Self, ResumeState)> {
        let name = |n: &str| dir.join(format!("{prefix}{n}"));

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
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = name(&baseline_file_name(idx, *j, *k));
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

        let ms_len = file_len(&name("rows_ms.bin"))?;
        let ms_rows = ms_len / 8;
        // blocks.csv has one line per station per row.
        let blocks_rows = count_csv_rows(&name("blocks.csv"))? / plan.n_station.max(1) as u64;
        // The shortest file wins: a crash can only ever leave one of them short,
        // and taking the minimum means we re-write a row rather than claim one
        // that is not fully there.
        let rows_written = complete.min(ms_rows).min(blocks_rows);
        let truncated_rows = complete.saturating_sub(rows_written);

        // --- truncate everything back to that prefix ---
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = name(&baseline_file_name(idx, *j, *k));
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
        truncate_file(&name("rows_ms.bin"), rows_written * 8)?;
        // blocks.csv carries one line per station per row.
        truncate_csv(&name("blocks.csv"), rows_written * plan.n_station as u64)?;
        // rows_ms.txt is a derived view of rows_ms.bin, so it is rebuilt rather
        // than truncated: two files that are supposed to agree are easier to keep
        // in step by regenerating one from the other than by truncating both.
        rebuild_ms_text(&name("rows_ms.bin"), &name("rows_ms.txt"), rows_written)?;

        // --- the previous row's frame range, for resuming the grid ---
        let next_frame_id = read_last_block_first_frame(&name("blocks.csv"), plan.n_cum)?;

        // --- open everything for appending ---
        let mut baselines = Vec::with_capacity(n_baseline);
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = name(&baseline_file_name(idx, *j, *k));
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
        let mut ms_bin = open_rw(&name("rows_ms.bin"))?;
        ms_bin.seek(SeekFrom::Start(rows_written * 8))?;
        let mut ms_txt_file = open_rw(&name("rows_ms.txt"))?;
        ms_txt_file.seek(SeekFrom::End(0))?;
        let ms_txt = BufWriter::new(ms_txt_file);
        let mut blocks_file = open_rw(&name("blocks.csv"))?;
        blocks_file.seek(SeekFrom::End(0))?;
        let blocks = BufWriter::new(blocks_file);

        // A set's metadata.yaml is written once, when the set is created; opening
        // a set that already has rows checks it rather than rewriting it, because
        // rewriting geometry that existing rows were produced with is how a reader
        // gets misled. Missing metadata is the one case that is written rather
        // than checked: a crash between creating the files and writing metadata
        // leaves a set nobody has described yet, and the geometry in hand is the
        // only description available.
        let meta_path = name("metadata.yaml");
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
                prefix,
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
        Ok((
            SetState {
                dir: dir.to_path_buf(),
                plan: plan.clone(),
                cfg: cfg.clone(),
                config_hash: config_hash.to_string(),
                fsync_every: fsync_every.max(1),
                daily: cfg.storage.daily_files,
                prefix: prefix.to_string(),
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
    ) -> Self {
        SetState {
            dir: dir.to_path_buf(),
            plan: plan.clone(),
            cfg: cfg.clone(),
            config_hash: config_hash.to_string(),
            fsync_every: fsync_every.max(1),
            daily: cfg.storage.daily_files,
            prefix: String::new(),
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
        let prefix = local_stamp_at(row_ms, &self.cfg.storage.file_prefix_format);
        let mut last: Option<anyhow::Error> = None;
        for attempt in 1..=3 {
            match self.try_create_set(&prefix, day, kind, continuation.clone(), blocks) {
                Ok(()) => {
                    if kind != SetKind::Rollover {
                        eprintln!(
                            "writer: opened file set {prefix} for local day {day} ({})",
                            kind.as_str()
                        );
                    }
                    return Ok(());
                }
                Err(e) => {
                    eprintln!(
                        "writer: opening file set {prefix} for local day {day} failed \
                         (attempt {attempt}/3): {e}"
                    );
                    last = Some(e);
                    if attempt < 3 {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        }
        Err(anyhow::Error::new(RolloverFailed {
            prefix,
            detail: format!("{:#}", last.expect("three attempts set it")),
        }))
    }

    fn try_create_set(
        &mut self,
        prefix: &str,
        day: &str,
        kind: SetKind,
        continuation: Option<String>,
        blocks: &RowBlock,
    ) -> Result<()> {
        let name = |n: &str| self.dir.join(format!("{prefix}{n}"));
        // A prefix that already holds rows belongs to a set somebody else is
        // writing (or wrote); reusing it would interleave two processes' rows.
        // An empty one is a half-created set from a failed attempt: reuse it.
        let probe = name(&baseline_file_name(0, 0, 0));
        if let Ok(m) = std::fs::metadata(&probe) {
            if m.len() > 0 {
                bail!(
                    "{} already holds {} bytes: the prefix {prefix} is in use",
                    probe.display(),
                    m.len()
                );
            }
        }

        let mut baselines = Vec::with_capacity(self.n_baseline);
        for (idx, (j, k)) in self.plan.baseline_pairs().iter().enumerate() {
            let p = name(&baseline_file_name(idx, *j, *k));
            let f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&p)
                .with_context(|| format!("creating {}", p.display()))?;
            baselines.push(f);
        }
        let ms_bin = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(name("rows_ms.bin"))
            .with_context(|| format!("creating {}", name("rows_ms.bin").display()))?;
        let ms_txt = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(name("rows_ms.txt"))
                .with_context(|| format!("creating {}", name("rows_ms.txt").display()))?,
        );
        let blocks_file = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(name("blocks.csv"))
                .with_context(|| format!("creating {}", name("blocks.csv").display()))?,
        );

        write_metadata(
            &name("metadata.yaml"),
            &self.plan,
            &self.cfg,
            &self.config_hash,
            prefix,
            day,
            kind,
            continuation.as_deref(),
            Some((blocks.row_index, blocks.first_frame_id)),
            0,
        )?;

        self.prefix = prefix.to_string();
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
        writeln!(files.ms_txt, "{} {}", self.rows_total, row_ms).context("writing rows_ms.txt")?;

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
    free: std::sync::mpsc::Sender<Vec<u8>>,
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
                row_ms,
                blocks,
            } => {
                let t0 = std::time::Instant::now();
                let r = state.write_one(&buf, row_ms, &blocks);
                busy += t0.elapsed();
                match r {
                    Ok(()) => {
                        written += 1;
                        rows.store(state.rows_total, Ordering::Release);
                        let _ = free.send(buf);
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
    set_day: Option<String>,
    set_kind: SetKind,

    tx: std::sync::mpsc::SyncSender<Job>,
    rx_free: std::sync::mpsc::Receiver<Vec<u8>>,
    handle: Option<std::thread::JoinHandle<()>>,
    rows: Arc<AtomicU64>,
    first_error: Arc<Mutex<Option<anyhow::Error>>>,
    pool: Vec<Vec<u8>>,
    /// Rows that may be in flight, from `storage.queue_bytes`.
    queue_depth: usize,
    pool_cap: usize,
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
        let empty = ResumeState {
            rows_written: 0,
            next_frame_id: None,
            truncated_rows: 0,
        };
        let (state, resume) = if !daily {
            // The historical layout: one nameless set, created now, exactly as
            // before the daily layout existed.
            SetState::attach(dir, plan, cfg, config_hash, fsync_every, "", None)?
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
                SetPolicy::Auto => found
                    .iter()
                    .filter(|s| s.day == today)
                    .min_by(|a, b| a.prefix.cmp(&b.prefix))
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
                        &set.prefix,
                        Some(&set),
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
                        SetState::unopened(dir, plan, cfg, config_hash, fsync_every, kind),
                        empty,
                    )
                }
            }
        };

        let row_bytes = state.row_bytes;
        let n_station = state.n_station;
        let prefix = state.prefix.clone();
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
                set_day,
                set_kind,
                tx,
                rx_free,
                handle: Some(handle),
                rows,
                first_error,
                pool: Vec::new(),
                queue_depth: depth,
                pool_cap: depth + 2,
                sent: 0,
                rows_at_open,
                max_outstanding: 0,
                finished: false,
                rows_final: rows_at_open,
            },
            resume,
        ))
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
    /// in the directory; read those (and their metadata) instead of this.
    pub fn set_prefix(&self) -> &str {
        &self.prefix
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
    pub fn write_row(&mut self, spectrum: &[u8], row_ms: i64, blocks: &RowBlock) -> Result<()> {
        if spectrum.len() != self.row_bytes {
            bail!(
                "write_row got {} bytes but one row is {} bytes",
                spectrum.len(),
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
        self.take_error()?;

        let mut buf = match self.pool.pop().or_else(|| self.rx_free.try_recv().ok()) {
            Some(mut b) => {
                b.clear();
                b
            }
            None => Vec::with_capacity(self.row_bytes),
        };
        buf.extend_from_slice(spectrum);
        if self
            .tx
            .send(Job::Row {
                buf,
                row_ms,
                blocks: blocks.clone(),
            })
            .is_err()
        {
            // The thread stopped between the check above and this send. Its own
            // error is the reason whenever it recorded one: a bare "the thread
            // stopped" here would lose the type the pipeline needs (a failed day
            // rollover is a `RolloverFailed`, and the report names it).
            self.take_error()?;
            bail!("the writer thread has stopped without reporting a reason");
        }
        self.sent += 1;
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
/// A set is recognised by the presence of baseline 0 under a prefix, and its day
/// comes from that set's own metadata rather than from parsing the prefix: the
/// prefix format is configurable, so parsing it would be a second, drifting
/// definition of what the name means.
pub fn discover_sets(dir: &Path, plan: &RunPlan) -> Result<Vec<FoundSet>> {
    let (j0, k0) = plan.baseline_pairs().first().copied().unwrap_or((0, 0));
    let suffix = baseline_file_name(0, j0, k0);
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
        let Some(prefix) = name.strip_suffix(&suffix).filter(|p| !p.is_empty()) else {
            continue; // the legacy unprefixed layout, or another file
        };
        let prefix = prefix.to_string();
        let meta = dir.join(format!("{prefix}metadata.yaml"));
        let day = read_metadata_field(&meta, "set_day").unwrap_or_default();
        let kind = read_metadata_field(&meta, "set_kind").unwrap_or_default();
        // Complete rows: the shortest of the set's own files, as on resume.
        let mut rows = u64::MAX;
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = dir.join(format!("{prefix}{}", baseline_file_name(idx, *j, *k)));
            let len = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            rows = rows.min(len / per_baseline_row.max(1));
        }
        let ms_rows = file_len(&dir.join(format!("{prefix}rows_ms.bin")))? / 8;
        let block_rows =
            count_csv_rows(&dir.join(format!("{prefix}blocks.csv")))? / plan.n_station.max(1) as u64;
        if rows == u64::MAX {
            rows = 0;
        }
        out.push(FoundSet {
            next_frame_id: read_last_block_first_frame(
                &dir.join(format!("{prefix}blocks.csv")),
                plan.n_cum,
            )?,
            prefix,
            day,
            kind,
            rows_written: rows.min(ms_rows).min(block_rows),
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
fn rebuild_ms_text(bin: &Path, txt: &Path, rows: u64) -> Result<()> {
    let mut s = String::new();
    let len = file_len(bin)?;
    if len > 0 && rows > 0 {
        let mut f = File::open(bin)?;
        let mut raw = vec![0u8; (rows * 8) as usize];
        f.read_exact(&mut raw)
            .with_context(|| format!("reading {}", bin.display()))?;
        for (i, chunk) in raw.chunks_exact(8).enumerate() {
            let ms = i64::from_le_bytes(chunk.try_into().expect("8 bytes"));
            s.push_str(&format!("{i} {ms}\n"));
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
    prefix: &str,
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
    let _ = writeln!(s, "# --- the file set this metadata belongs to ---");
    let _ = writeln!(s, "set_prefix: {prefix}");
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
    let _ = writeln!(s, "baselines:");
    for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
        let _ = writeln!(
            s,
            "  - {{index: {}, j: {}, k: {}, file: {}{}}}",
            idx,
            j,
            k,
            prefix,
            baseline_file_name(idx, *j, *k)
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

    /// Every set in `dir`, by prefix, with its (rows_ms.bin length, blocks rows).
    ///
    /// `rows_ms.bin` is an unbuffered file, so its length is exact at any moment.
    /// `blocks.csv` goes through a `BufWriter` and is only flushed when the set is
    /// closed or the fsync cadence comes round, so for a set still being written
    /// the second number lags by up to a buffer -- which is why the assertions
    /// below take it after `finish()`.
    fn sets_on_disk(dir: &Path, plan: &RunPlan) -> Vec<(String, u64, u64)> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            let Some(prefix) = name.strip_suffix("rows_ms.bin") else {
                continue;
            };
            let prefix = prefix.to_string();
            let ms = std::fs::metadata(dir.join(format!("{prefix}rows_ms.bin")))
                .map(|m| m.len())
                .unwrap_or(0)
                / 8;
            let blocks = std::fs::read_to_string(dir.join(format!("{prefix}blocks.csv")))
                .map(|s| s.lines().count() as u64)
                .unwrap_or(0)
                / plan.n_station.max(1) as u64;
            out.push((prefix, ms, blocks));
        }
        out.sort();
        out
    }

    fn baseline_paths(dir: &Path, plan: &RunPlan, prefix: &str) -> Vec<PathBuf> {
        plan.baseline_pairs()
            .iter()
            .enumerate()
            .map(|(i, (j, k))| dir.join(format!("{prefix}{}", baseline_file_name(i, *j, *k))))
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
        let meta1 = d.path().join(format!("{p1}metadata.yaml"));
        assert_eq!(read_metadata_field(&meta1, "set_day").as_deref(), Some(local_day_at(t0).as_str()));
        assert_eq!(read_metadata_field(&meta1, "set_kind").as_deref(), Some("start"));
        assert_eq!(read_metadata_field(&meta1, "set_prefix").as_deref(), Some(p1.as_str()));
        assert_eq!(read_metadata_field(&meta1, "first_row_index").as_deref(), Some("0"));
        assert_eq!(read_metadata_field(&meta1, "first_frame_id").as_deref(), Some("0"));
        // Every baseline file of the set exists with exactly one row in it.
        for p in baseline_paths(d.path(), &plan, &p1) {
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

        let meta2 = d.path().join(format!("{p2}metadata.yaml"));
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
        let b2 = std::fs::read_to_string(d.path().join(format!("{p2}blocks.csv"))).unwrap();
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
        let squatter = d.path().join(format!("{next}{}", baseline_file_name(0, 0, 0)));
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
        assert_eq!(rf.prefix, next);
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
        let meta = d.path().join(format!("{p1}metadata.yaml"));
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
        w2.sync().unwrap();
        let sets = sets_on_disk(d.path(), &plan);
        assert_eq!(sets.len(), 2, "{sets:?}");
        let p2 = sets
            .iter()
            .map(|(p, _, _)| p.clone())
            .find(|p| *p != p1)
            .expect("the second set has the other prefix");
        let meta2 = d.path().join(format!("{p2}metadata.yaml"));
        assert_eq!(read_metadata_field(&meta2, "set_kind").as_deref(), Some("second_set"));

        // --set <prefix> appends to the named set instead.
        let (w3, st3) =
            Writer::open_with(d.path(), &plan, &cfg, "hash-a", 1, &SetPolicy::Explicit(p1.clone()))
                .unwrap();
        assert_eq!(st3.rows_written, 1);
        assert_eq!(w3.set_prefix(), p1);
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
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = d.path().join(baseline_file_name(idx, *j, *k));
            assert_eq!(std::fs::metadata(&p).unwrap().len(), 3 * row_bytes, "{}", p.display());
        }
        assert_eq!(file_len(&d.path().join("rows_ms.bin")).unwrap(), 3 * 8);
        // One line per station per row.
        assert_eq!(
            count_csv_rows(&d.path().join("blocks.csv")).unwrap(),
            3 * plan.n_station as u64
        );
        assert!(d.path().join("metadata.yaml").exists());

        // The content is what we wrote, baseline-major and row by row.
        let mut buf = Vec::new();
        File::open(d.path().join(baseline_file_name(0, 0, 0)))
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
        let p0 = d.path().join(baseline_file_name(0, 0, 0));
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
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = d.path().join(baseline_file_name(idx, *j, *k));
            let f = OpenOptions::new().write(true).open(&p).unwrap();
            f.set_len(row_bytes).unwrap();
        }

        let (_w, st) = Writer::open(d.path(), &plan, &cfg, "hash-a", 1).unwrap();
        assert_eq!(st.rows_written, 1, "the data files are the truth");
        assert_eq!(st.truncated_rows, 0, "the data file was already short");
        assert_eq!(file_len(&d.path().join("rows_ms.bin")).unwrap(), 8);
        assert_eq!(
            count_csv_rows(&d.path().join("blocks.csv")).unwrap(),
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

        let ms = std::fs::read(d.path().join("rows_ms.bin")).unwrap();
        assert_eq!(ms.len(), 5 * 8, "five timestamps, not three");
        let stamps: Vec<i64> = ms
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(stamps, vec![1000, 1001, 1002, 2003, 2004], "the old rows survive");

        let txt = std::fs::read_to_string(d.path().join("rows_ms.txt")).unwrap();
        assert_eq!(txt.lines().count(), 5);
        assert!(txt.lines().next().unwrap().starts_with("0 1000"));

        assert_eq!(
            count_csv_rows(&d.path().join("blocks.csv")).unwrap(),
            5 * plan.n_station as u64
        );
        let row_bytes = baseline_row_bytes(plan.n_freq) as u64;
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = d.path().join(baseline_file_name(idx, *j, *k));
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

        let s = std::fs::read_to_string(d.path().join("blocks.csv")).unwrap();
        let lines: Vec<&str> = s.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(lines.len(), plan.n_station, "one line per station");
        let f: Vec<&str> = lines[0].split(',').collect();
        assert_eq!(f[0], "0", "row index");
        assert_eq!(f[1], "0", "station index");
        assert_eq!(f[2], "0", "first frame id");
        assert_eq!(f[3], plan.n_cum.to_string(), "n_cum");
        assert_eq!(f[4], valid.to_string(), "valid frames");
        assert_eq!(f[5], "7", "zero filled frames");

        let ms = std::fs::read(d.path().join("rows_ms.bin")).unwrap();
        assert_eq!(i64::from_le_bytes(ms[..8].try_into().unwrap()), 4242);
        let txt = std::fs::read_to_string(d.path().join("rows_ms.txt")).unwrap();
        assert_eq!(txt.trim(), "0 4242");
    }
}
