//! Output: one raw spectrum file per baseline, plus sidecar files.
//!
//! Layout of a run directory:
//!
//! ```text
//!   baseline_0000_j00_k00.bin   complex64[ntime][n_freq], one row per integration
//!   ...
//!   rows_ms.bin                 int64 little-endian Unix milliseconds, one per row
//!   rows_ms.txt                 "<row_index> <ms>" per line
//!   blocks.csv                  row_index, frame range, per-station valid frames
//!   metadata.yaml               everything needed to read the raw files
//! ```
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
//! The row index is the file's whole state: `ntime` is `rows_written`, and a
//! reader that wants frame ids or timestamps joins against `blocks.csv` and
//! `rows_ms.bin` by row index.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::config::{Config, RunPlan};

/// Bytes in one row of one baseline's file.
pub fn baseline_row_bytes(n_freq: u32) -> usize {
    n_freq as usize * 8 // complex64
}

/// Name of one baseline's raw file. Zero padded so that a directory listing is
/// in baseline order, and carrying (j, k) so a file can be identified without
/// reading metadata.yaml.
pub fn baseline_file_name(index: usize, j: usize, k: usize) -> String {
    format!("baseline_{index:04}_j{j:02}_k{k:02}.bin")
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

/// Writes one run's output.
#[derive(Debug)]
pub struct Writer {
    dir: PathBuf,
    n_baseline: usize,
    /// Bytes of one baseline's row: the unit every raw file is made of. Each
    /// baseline file grows by exactly this much per integration row.
    per_baseline_row: usize,
    n_station: usize,
    /// Bytes of a whole integration row across all baselines: what
    /// `corr_rows_fetch` hands over and what `write_row` expects.
    row_bytes: usize,

    baselines: Vec<File>,
    ms_bin: File,
    ms_txt: BufWriter<File>,
    blocks: BufWriter<File>,

    rows_written: u64,
    fsync_every: u64,
}

impl Writer {
    /// Create or resume a run directory.
    ///
    /// `cfg` is written to metadata.yaml and hashed into it; on resume the hash
    /// must match, because the geometry being different would silently misread
    /// every existing row.
    pub fn open(
        dir: &Path,
        plan: &RunPlan,
        cfg: &Config,
        config_hash: &str,
        fsync_every: u64,
    ) -> Result<(Self, ResumeState)> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating the run directory {}", dir.display()))?;

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
            let p = dir.join(baseline_file_name(idx, *j, *k));
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

        let ms_len = file_len(&dir.join("rows_ms.bin"))?;
        let ms_rows = ms_len / 8;
        // blocks.csv has one line per station per row.
        let blocks_rows = count_csv_rows(&dir.join("blocks.csv"))? / plan.n_station.max(1) as u64;
        // The shortest file wins: a crash can only ever leave one of them short,
        // and taking the minimum means we re-write a row rather than claim one
        // that is not fully there.
        let rows_written = complete.min(ms_rows).min(blocks_rows);
        let truncated_rows = complete.saturating_sub(rows_written);

        // --- truncate everything back to that prefix ---
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = dir.join(baseline_file_name(idx, *j, *k));
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
        truncate_file(&dir.join("rows_ms.bin"), rows_written * 8)?;
        // blocks.csv carries one line per station per row.
        truncate_csv(&dir.join("blocks.csv"), rows_written * plan.n_station as u64)?;
        // rows_ms.txt is a derived view of rows_ms.bin, so it is rebuilt rather
        // than truncated: two files that are supposed to agree are easier to keep
        // in step by regenerating one from the other than by truncating both.
        rebuild_ms_text(&dir.join("rows_ms.bin"), &dir.join("rows_ms.txt"), rows_written)?;

        // --- the previous row's frame range, for resuming the grid ---
        let next_frame_id = read_last_block_first_frame(&dir.join("blocks.csv"), plan.n_cum)?;

        // --- open everything for appending ---
        let mut baselines = Vec::with_capacity(n_baseline);
        for (idx, (j, k)) in plan.baseline_pairs().iter().enumerate() {
            let p = dir.join(baseline_file_name(idx, *j, *k));
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
        let mut ms_bin = open_rw(&dir.join("rows_ms.bin"))?;
        ms_bin.seek(SeekFrom::Start(rows_written * 8))?;
        let ms_txt_path = dir.join("rows_ms.txt");
        let mut ms_txt_file = open_rw(&ms_txt_path)?;
        ms_txt_file.seek(SeekFrom::End(0))?;
        let ms_txt = BufWriter::new(ms_txt_file);
        let mut blocks_file = open_rw(&dir.join("blocks.csv"))?;
        blocks_file.seek(SeekFrom::End(0))?;
        let blocks = BufWriter::new(blocks_file);

        // metadata.yaml is written once, at the start. Resume checks it rather
        // than rewriting it.
        let meta_path = dir.join("metadata.yaml");
        if rows_written == 0 || !meta_path.exists() {
            write_metadata(&meta_path, plan, cfg, config_hash)?;
        } else {
            check_metadata(&meta_path, config_hash)?;
        }

        let state = ResumeState {
            rows_written,
            next_frame_id,
            truncated_rows,
        };
        Ok((
            Writer {
                dir: dir.to_path_buf(),
                n_baseline,
                n_station: plan.n_station,
                per_baseline_row,
                row_bytes,
                baselines,
                ms_bin,
                ms_txt,
                blocks,
                rows_written,
                fsync_every: fsync_every.max(1),
            },
            state,
        ))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn rows_written(&self) -> u64 {
        self.rows_written
    }

    /// Append one integration row.
    ///
    /// `spectrum` is `n_baseline * n_freq` complex64, baseline major -- the whole
    /// row across every baseline, which is exactly what `corr_rows_fetch` returns
    /// for one row. `write_row` splits it into one write per baseline file.
    /// `blocks` carries one entry per station. `row_ms` is the wall
    /// clock of the row's first frame. `blocks` carries the per-row frame
    /// accounting.
    ///
    /// The spectrum goes to disk first and is flushed; the sidecars follow. That
    /// order is the whole reason a crash can only cost a row and never produce an
    /// unreadable one.
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

        debug_assert_eq!(
            self.baselines.len(),
            self.n_baseline,
            "one file per baseline, checked because a mismatch would shuffle spectra \
             between baselines silently"
        );
        for (idx, f) in self.baselines.iter_mut().enumerate() {
            let start = idx * self.per_baseline_row;
            // One write per baseline, of exactly one row. Not write_all in a loop:
            // a partially applied row is what we are trying to avoid.
            f.write_all(&spectrum[start..start + self.per_baseline_row])
                .with_context(|| format!("writing row {} baseline {idx}", self.rows_written))?;
        }

        if self.rows_written % self.fsync_every == 0 {
            // Flush the data before the sidecars claim it exists.
            for f in self.baselines.iter_mut() {
                f.flush()?;
            }
            for f in self.baselines.iter() {
                f.sync_data()?;
            }
        }

        self.ms_bin
            .write_all(&row_ms.to_le_bytes())
            .context("writing rows_ms.bin")?;
        writeln!(self.ms_txt, "{} {}", self.rows_written, row_ms).context("writing rows_ms.txt")?;

        for (station, valid) in blocks.valid_frames_per_station.iter().enumerate() {
            writeln!(
                self.blocks,
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

        self.rows_written += 1;
        Ok(())
    }

    /// Flush and fsync everything. Called on a clean stop.
    pub fn finish(&mut self) -> Result<()> {
        for f in self.baselines.iter_mut() {
            f.flush()?;
        }
        for f in self.baselines.iter() {
            f.sync_all()?;
        }
        self.ms_bin.sync_all()?;
        self.ms_txt.flush()?;
        self.ms_txt.get_ref().sync_all()?;
        self.blocks.flush()?;
        self.blocks.get_ref().sync_all()?;
        Ok(())
    }
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

fn write_metadata(p: &Path, plan: &RunPlan, cfg: &Config, hash: &str) -> Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# Written once at the start of a run. A resumed run checks the");
    let _ = writeln!(s, "# config_hash rather than rewriting geometry that the existing");
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
            "  - {{index: {}, j: {}, k: {}, file: {}}}",
            idx,
            j,
            k,
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
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("config/correlator-test.yaml");
        let cfg = Config::from_yaml_file(&path).unwrap();
        let plan = cfg.compile().unwrap();
        (cfg, plan)
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
