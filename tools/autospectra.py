#!/usr/bin/env python3
"""autospectra.py -- the 24 autocorrelation spectra, one panel per station.

The autocorrelation is the power spectrum: `Y_j * conj(Y_j)`, real, one per
station, and the fastest way to see whether the array is healthy at all. It is
also the place the instrument's own artefacts live -- a dead port is flat noise, a
port with the wrong gain sits at a different level, a spur shows as a spike, and
the R2C DC bin is the front end's DC offset.

Two things this checks that a plot alone would not:

  * **the imaginary part is zero.** An autocorrelation of a real signal has no
    imaginary part; anything else means a wrong conjugation or a transposed
    layout. Printed as max|Im| / mean(Re) per station, and it should be ~1e-5.
  * **the row count actually averaged.** A short run's first row can be partly
    zero-filled while the frame grid catches up, which drags the whole average
    down without looking wrong. The per-row zero fill is printed with it.

Usage:
    tools/autospectra.py RUN_DIR                      # newest file set
    tools/autospectra.py RUN_DIR --prefix 20260917_000000 --db
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corr_common as cc


def station_names(set_info, n_station):
    """`index -> name` from the metadata's station list."""
    out = {}
    try:
        for line in open(set_info["metadata"]):
            m = re.search(r"index:\s*(\d+),\s*name:\s*([^,}]+)", line)
            if m:
                out[int(m.group(1))] = m.group(2).strip()
    except FileNotFoundError:
        pass
    return {i: out.get(i, f"station {i}") for i in range(n_station)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prefix", default=None, help="which file set (local day); default the newest")
    ap.add_argument("--rows", type=int, default=0, help="rows to average; 0 = all")
    ap.add_argument("--out", default="/tmp/autospectra.png")
    ap.add_argument("--overlay", default=None, help="also write all 24 in one panel here")
    ap.add_argument("--db", action="store_true", help="10*log10 instead of linear power")
    ap.add_argument("--mhz", default=None, help="frequency window, e.g. 0,100")
    ap.add_argument(
        "--skip-bins",
        type=int,
        default=0,
        help="do not plot the first N channels. The R2C DC bin carries the front "
        "end's offset, which is orders of magnitude above the band and hides it",
    )
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = cc.pick_set(a.run_dir, a.prefix)
    nch = int(cc.metadata_field(s, "n_freq") or 0)
    bin_hz = float(cc.metadata_field(s, "bin_width_hz") or 0.0)
    n_station = int(cc.metadata_field(s, "n_station") or 0)
    if not nch or not bin_hz:
        sys.exit(f"{s['metadata']}: no n_freq/bin_width_hz; is this a run's metadata?")
    names = station_names(s, n_station)

    # Per-row zero fill, so a start-up row cannot silently flatten the average.
    zf = [0] * n_station
    per_row = {}
    for line in open(s["blocks"]):
        f = line.strip().split(",")
        if len(f) >= 6 and f[0].isdigit():
            per_row.setdefault(int(f[0]), [0] * n_station)[int(f[1])] = int(f[5])
    n_cum = int(cc.metadata_field(s, "n_cum") or 1)
    rows_avail = len(per_row)
    take = a.rows or rows_avail
    for r in sorted(per_row)[-take:]:
        for i, v in enumerate(per_row[r]):
            zf[i] += v

    # The autocorrelations: j == k.
    autos = [(j, p) for _, j, k, p in cc.set_baselines(a.run_dir, prefix=a.prefix) if j == k]
    autos.sort()
    if not autos:
        sys.exit(f"{s['blocks']}: no j == k baseline in this set")

    freqs = np.arange(nch) * bin_hz / 1e6  # MHz, bin c = c * fs/(2*nch)
    spectra = {}
    print(f"{s['prefix']}: {len(autos)} autocorrelations, {nch} channels, "
          f"{bin_hz/1e6:.4f} MHz per bin, averaging {take} of {rows_avail} rows")
    print(f"{'station':10s} {'rows used':>9s} {'zero-filled':>11s} {'median P':>12s} "
          f"{'peak MHz':>10s} {'peak/median':>11s} {'max|Im|/Re':>11s} {'DC/median':>10s}")
    for idx, (j, path) in enumerate(autos):
        raw = np.fromfile(path, dtype="<c8")
        rows = raw.size // nch
        if raw.size % nch:
            raw = raw[: rows * nch]
        rows = min(rows, take)
        data = raw[: rows * nch].reshape(rows, nch)
        mean = data.mean(axis=0)
        p = mean.real
        imag = np.abs(mean.imag)
        spectra[j] = p
        med = float(np.median(p))
        pk = int(np.argmax(p))
        print(f"{names[j]:10s} {rows:9d} {zf[j]:11d} {med:12.4g} "
              f"{freqs[pk]:10.3f} {p[pk]/med if med else 0:11.3f} "
              f"{imag.max()/med if med else 0:11.2e} {p[0]/med if med else 0:10.3f}")

    # One reference for every panel, so a station that is 10x stronger is drawn
    # 10 dB higher instead of being normalised away: the level difference between
    # ports is half of what this figure is for.
    ref = max(float(np.max(v)) for v in spectra.values())
    lo = a.skip_bins
    fig, axes = plt.subplots(4, 6, figsize=(19, 10), dpi=120, sharex=True, sharey=True)
    for idx, j in enumerate(sorted(spectra)):
        p = spectra[j]
        ax = axes[idx // 6][idx % 6]
        y = 10 * np.log10(np.maximum(p, ref * 1e-12)) if a.db else p
        ax.plot(freqs[lo:], y[lo:], lw=0.5, color="#1f4e79")
        ax.set_title(f"{names[j]}  ({j},{j})", fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)

    ylim = axes[0][0].get_ylim()
    if not a.db:
        # A spike would otherwise flatten every other panel: clip the shared scale
        # to a high percentile of the *pooled* power rather than to the maximum.
        pooled = np.concatenate([spectra[j][lo:] for j in sorted(spectra)])
        lo_v, hi_v = np.percentile(pooled, [0.1, 99.99])
        lo, hi = lo_v, hi_v
        for row in axes:
            for ax in row:
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    if a.mhz:
        lo_mhz, hi_mhz = (float(v) for v in a.mhz.split(","))
        for row in axes:
            for ax in row:
                ax.set_xlim(lo_mhz, hi_mhz)
    for ax in axes[-1]:
        ax.set_xlabel("MHz", fontsize=8)
    for row in axes:
        row[0].set_ylabel("power" if not a.db else "dB", fontsize=8)
    fig.suptitle(
        f"{s['prefix']}  |  autocorrelation |Y|^2 per station, {take} rows averaged, "
        f"{'10log10, 0 dB = strongest channel of any port' if a.db else 'linear'}  |  this is "
        f"the power spectrum, arbitrary units (int8 counts, no FFT normalisation)"
        + (f"  |  first {a.skip_bins} bin(s) not drawn" if a.skip_bins else ""),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out)
    print(f"wrote {a.out}")

    if a.overlay:
        fig2, ax = plt.subplots(figsize=(12, 5), dpi=120)
        for j in sorted(spectra):
            p = spectra[j]
            y = 10 * np.log10(np.maximum(p, ref * 1e-12)) if a.db else p
            ax.plot(freqs[a.skip_bins :], y[a.skip_bins :], lw=0.5, label=names[j])
        ax.set_xlabel("MHz")
        ax.set_ylabel("power" if not a.db else "dB")
        ax.grid(alpha=0.2, lw=0.4)
        ax.legend(fontsize=6, ncol=4)
        ax.set_title(f"{s['prefix']}: all {len(spectra)} autocorrelations overlaid",
                     fontsize=10)
        fig2.tight_layout()
        fig2.savefig(a.overlay)
        print(f"wrote {a.overlay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
