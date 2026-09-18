//! The autocorrelation spectra of the row being written: `|Y_j|^2` per station, in
//! dB, one point per group of channels so that a page can draw it.
//!
//! **Why it is recorded here and not read back off the disk.** The row is in hand
//! when it is written -- that is what `flush_rows` is doing -- the autocorrelations
//! are 24 of its 300 baselines (8% of the bytes), and reading them back would mean
//! hitting the same NFS the run is pushing 39 MB/s into. The pipe is therefore:
//!
//! | stage | file | knows |
//! |---|---|---|
//! | pick the `j == k` rows out of a row | `pipeline.rs` | when a row is written |
//! | reduce them to one dB curve per station | this file | channels, dB, decimation |
//! | serve the latest one | `status.rs` | HTTP, one thread, one listener |
//! | draw it | `tools/corr-web.py` | nothing about the correlator |
//!
//! Nothing in the running correlator waits for a reader: publishing swaps one
//! `Arc`, and with `monitoring.http.enabled` off (the default) no cell is even
//! created, so a row costs nothing extra.
//!
//! What the figure is for: the autocorrelation *is* the power spectrum, and it is
//! the fastest way to see whether an input is healthy at all -- a dead port is
//! flat noise, a port with the wrong gain sits at a different level, a spur is a
//! spike. `tools/autospectra.py` draws the same quantity offline at full
//! resolution; this one trades resolution for being live.

#![cfg_attr(not(feature = "status-http"), allow(dead_code))]

use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::config::RunPlan;

/// Bumped when a field changes meaning. Adding a field is compatible.
pub(crate) const SCHEMA: u32 = 1;

/// Points per station curve. 16384 channels across ~240 px is 68 channels per
/// pixel: drawing them all would paint the noise as a solid block.
pub(crate) const POINTS: usize = 256;

/// Channels at the start that are neither drawn nor allowed to set the dB
/// reference. Bin 0 of an R2C transform is the front end's DC offset, orders of
/// magnitude above the band (measured: 500x to 130000x the band's median), so
/// including it would flatten every other channel into a line.
pub(crate) const SKIP_BINS: usize = 1;

/// The dB scale's span below the median: far enough to see structure, not so far
/// that one spur decides the axis.
const SPAN_BELOW_MEDIAN_DB: f64 = 45.0;

/// One station's curve, already reduced.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct StationSpectrum {
    pub name: String,
    /// The compacted station index, i.e. the `j` of `(j, j)`.
    pub index: usize,
    /// One point per `step` channels, in dB relative to [`Snapshot::db_ref`].
    pub db: Vec<f64>,
    pub peak_db: f64,
    pub peak_mhz: f64,
    pub median_db: f64,
}

/// The latest row's autocorrelations, as a reader sees them.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Snapshot {
    pub schema: u32,
    pub ready: bool,
    pub ts_unix_ms: i64,
    /// Row number in the run (the same one `blocks.csv` and `time.bin` carry) and
    /// the wall clock of that row's start.
    pub row_index: u64,
    pub row_ms: i64,
    pub n_freq: u32,
    pub bin_width_hz: f64,
    pub points: usize,
    /// Channels per drawn point.
    pub step: usize,
    pub skip_bins: usize,
    /// Frequency of the first and last drawn point, MHz.
    pub freq_mhz: [f64; 2],
    /// What 0 dB is. Here rather than in the page, so the JSON explains itself.
    pub db_ref: String,
    /// Suggested vertical range for a shared axis across every station, so that
    /// the level difference *between* ports stays visible.
    pub ymin: f64,
    pub ymax: f64,
    pub stations: Vec<StationSpectrum>,
}

/// The one published sample, and the geometry needed to reduce the next one.
///
/// A slot, not a queue: a reader wants the newest row, and rows arrive once a
/// second. The reduction runs on the pipeline thread (it already has the bytes)
/// and costs a few milliseconds of it per row.
pub(crate) struct Cell {
    slot: Mutex<Option<Arc<Snapshot>>>,

    // Fixed for the run; taken from the plan once, at startup.
    n_freq: usize,
    bin_width_hz: f64,
    row_bytes_per_baseline: usize,
    /// `(baseline index, station index)` for every `j == k` baseline, in order.
    autos: Vec<(usize, usize)>,
    station_names: Vec<String>,
    step: usize,
    kept: usize,
}

impl Cell {
    pub(crate) fn new(plan: &RunPlan) -> Arc<Cell> {
        let n_freq = plan.n_freq as usize;
        let step = (n_freq / POINTS).max(1);
        let kept = ((n_freq - SKIP_BINS) / step).max(1);
        let autos = plan
            .baseline_pairs()
            .iter()
            .enumerate()
            .filter(|(_i, (j, k))| j == k)
            .map(|(i, (j, _k))| (i, *j))
            .collect();
        Arc::new(Cell {
            slot: Mutex::new(None),
            n_freq,
            bin_width_hz: plan.bin_width_hz,
            row_bytes_per_baseline: 8 * n_freq, // complex64
            autos,
            station_names: plan.enabled.iter().map(|s| s.name.clone()).collect(),
            step,
            kept,
        })
    }

    /// Reduce one row's spectrum to the dB curves and publish them.
    ///
    /// `spectrum` is one row as the writer takes it: `n_baseline` blocks of
    /// `n_freq` complex64, in `plan.baseline_pairs()` order.
    pub(crate) fn publish_row(
        &self,
        spectrum: &[u8],
        row_index: u64,
        row_ms: i64,
        now_unix_ms: i64,
    ) {
        // A short buffer means the geometry and the row disagree; that is a
        // writer-side bug, and not something to crash a running correlator over
        // from here.
        if let Some(snap) = self.reduce(spectrum, row_index, row_ms, now_unix_ms) {
            let mut slot = match self.slot.lock() {
                Ok(slot) => slot,
                Err(poisoned) => poisoned.into_inner(),
            };
            *slot = Some(Arc::new(snap));
        }
    }

    pub(crate) fn current(&self) -> Option<Arc<Snapshot>> {
        let slot = match self.slot.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        slot.clone()
    }

    /// The reduction itself, separated from the publishing so that a test can
    /// check the numbers without a pipeline around them.
    fn reduce(
        &self,
        spectrum: &[u8],
        row_index: u64,
        row_ms: i64,
        now_unix_ms: i64,
    ) -> Option<Snapshot> {
        let need = self
            .autos
            .iter()
            .map(|(bi, _)| (bi + 1) * self.row_bytes_per_baseline)
            .max()
            .unwrap_or(0);
        if self.autos.is_empty() || spectrum.len() < need {
            return None;
        }

        // Power per drawn point, in channel groups, one station at a time.
        let mut per_station: Vec<Vec<f64>> = Vec::with_capacity(self.autos.len());
        for (bi, _j) in &self.autos {
            let base = bi * self.row_bytes_per_baseline;
            let row = &spectrum[base..base + self.row_bytes_per_baseline];
            let mut points = vec![0.0f64; self.kept];
            for (seg, out) in points.iter_mut().enumerate() {
                let lo = SKIP_BINS + seg * self.step;
                let hi = lo + self.step;
                let mut acc = 0.0f64;
                for c in lo..hi {
                    let o = c * 8;
                    let re = f32::from_le_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]]);
                    let im =
                        f32::from_le_bytes([row[o + 4], row[o + 5], row[o + 6], row[o + 7]]);
                    let (re, im) = (re as f64, im as f64);
                    acc += re * re + im * im;
                }
                *out = acc / self.step as f64;
            }
            per_station.push(points);
        }

        // One reference for every station: the strongest channel in the band. A
        // station that is 10x stronger then draws 10 dB higher instead of being
        // normalised away, which is half of what the figure is for.
        let ref_power = per_station
            .iter()
            .flat_map(|p| p.iter())
            .fold(0.0f64, |a, &b| a.max(b));
        // NaN would slip through a `<= 0.0` test and make every dB below it NaN, so
        // ask for a finite positive reference explicitly.
        if !ref_power.is_finite() || ref_power <= 0.0 {
            return None;
        }

        let db = |v: f64| 10.0 * (v / ref_power).max(1e-12).log10();
        let mut stations = Vec::with_capacity(self.autos.len());
        let mut all_db: Vec<f64> = Vec::with_capacity(self.autos.len() * self.kept);
        for (i, points) in per_station.iter().enumerate() {
            let j = self.autos[i].1;
            let curve: Vec<f64> = points
                .iter()
                .map(|&v| (db(v) * 10.0).round() / 10.0)
                .collect();
            let peak = points
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut sorted = curve.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            all_db.extend(curve.iter().copied());
            stations.push(StationSpectrum {
                name: self
                    .station_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("station {j}")),
                index: j,
                peak_db: curve[peak],
                peak_mhz: (SKIP_BINS as f64 + (peak as f64 + 0.5) * self.step as f64)
                    * self.bin_width_hz
                    / 1e6,
                median_db: sorted[sorted.len() / 2],
                db: curve,
            });
        }

        let mut all_sorted = all_db.clone();
        all_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = all_sorted[all_sorted.len() / 2];
        let hi = all_sorted[all_sorted.len() - 1];
        Some(Snapshot {
            schema: SCHEMA,
            ready: true,
            ts_unix_ms: now_unix_ms,
            row_index,
            row_ms,
            n_freq: self.n_freq as u32,
            bin_width_hz: self.bin_width_hz,
            points: self.kept,
            step: self.step,
            skip_bins: SKIP_BINS,
            freq_mhz: [
                (SKIP_BINS as f64 + 0.5 * self.step as f64) * self.bin_width_hz / 1e6,
                (SKIP_BINS as f64 + (self.kept as f64 - 0.5) * self.step as f64)
                    * self.bin_width_hz
                    / 1e6,
            ],
            db_ref: format!(
                "0 dB = 带内最强通道（已排除 DC 与前 {SKIP_BINS} 个通道）"
            ),
            ymin: (median - SPAN_BELOW_MEDIAN_DB).min(hi).round(),
            ymax: (hi + 2.0).round(),
            stations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    /// The configuration that ships, so the geometry under test is the real one.
    fn plan() -> RunPlan {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("config/correlator.yaml");
        let cfg = Config::from_yaml_str(&std::fs::read_to_string(&p).expect("read config"))
            .expect("parse config");
        cfg.compile().expect("compile the plan")
    }

    /// One row: every channel of a baseline at the same amplitude, except channel 0
    /// which is the front end's DC offset. Every expected dB is then known by hand,
    /// so the only things that can move it are a wrong stride or a wrong reference.
    fn flat_row(plan: &RunPlan, amp: impl Fn(usize) -> f32, dc: f32) -> Vec<u8> {
        let n_freq = plan.n_freq as usize;
        let mut out = Vec::with_capacity(plan.n_baseline * n_freq * 8);
        for (j, _k) in plan.baseline_pairs() {
            let v = amp(j);
            for c in 0..n_freq {
                let x = if c == 0 { dc } else { v };
                out.extend_from_slice(&x.to_le_bytes()); // real
                out.extend_from_slice(&0.0f32.to_le_bytes()); // imaginary
            }
        }
        out
    }

    /// The reduction must not invent structure: a flat row is a flat curve, one per
    /// station, and the reference makes the strongest one 0 dB.
    #[test]
    fn a_flat_row_is_a_flat_curve_per_station() {
        let plan = plan();
        let cell = Cell::new(&plan);
        // 4x the amplitude on station 0 is 16x the power, i.e. 12.04 dB above the rest.
        let row = flat_row(&plan, |j| if j == 0 { 4.0 } else { 1.0 }, 1.0);
        let snap = cell.reduce(&row, 7, 1_700_000_000_000, 0).expect("reduced");

        assert_eq!(snap.schema, SCHEMA);
        assert_eq!(snap.row_index, 7);
        assert_eq!(snap.row_ms, 1_700_000_000_000);
        assert_eq!(snap.stations.len(), plan.n_station);
        assert_eq!(snap.points, (plan.n_freq as usize - SKIP_BINS) / snap.step);
        assert_eq!(snap.stations[0].name, plan.enabled[0].name);

        for s in &snap.stations {
            let first = s.db[0];
            assert!(
                s.db.iter().all(|d| (d - first).abs() < 0.11),
                "{} is not flat: {:?}",
                s.name,
                &s.db[..4]
            );
        }
        // The reference is the strongest channel of any port, so station 0 is 0 dB
        // and the others are 16x down.
        let strong = &snap.stations[0];
        let weak = &snap.stations[1];
        assert!(
            strong.median_db.abs() < 0.11,
            "the strongest channel is 0 dB by definition, got {}",
            strong.median_db
        );
        assert!(
            (weak.median_db + 12.04).abs() < 0.2,
            "16x the power is 12.04 dB down, got {}",
            weak.median_db
        );
    }

    /// The DC channel is neither drawn nor allowed to set the reference: with a
    /// 1000x offset in channel 0 the band must still read 0 dB, not -60.
    #[test]
    fn the_dc_channel_is_excluded_from_the_curve_and_the_reference() {
        let plan = plan();
        let cell = Cell::new(&plan);
        let row = flat_row(&plan, |_| 1.0, 1.0e3);
        let snap = cell.reduce(&row, 1, 0, 0).expect("reduced");

        assert_eq!(snap.skip_bins, SKIP_BINS);
        assert!(snap.freq_mhz[0] > 0.0, "the first drawn point is not DC");
        for s in &snap.stations {
            assert!(
                s.median_db.abs() < 0.11,
                "the band is the reference, DC is not: {} reads {}",
                s.name,
                s.median_db
            );
        }
        // ... and the suggested axis stays where the band is, not where the offset is.
        assert!(snap.ymax < 3.0, "ymax {} was set by something in the band", snap.ymax);
        assert!(snap.ymin >= snap.stations[0].median_db - SPAN_BELOW_MEDIAN_DB - 1.0);
    }

    /// A buffer that is too short for the geometry is not a panic: it is the row
    /// and the plan disagreeing, which the writer would report anyway.
    #[test]
    fn a_short_row_is_refused_rather_than_indexed_out_of_bounds() {
        let plan = plan();
        let cell = Cell::new(&plan);
        assert!(cell.reduce(&[0u8; 64], 0, 0, 0).is_none());
        assert!(cell.current().is_none());
    }
}
