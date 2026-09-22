#!/usr/bin/env python3
"""autospectra.py -- the autocorrelation spectra: 24 panels, or one station up close.

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

`--station NAME` draws that one station instead of the 24-panel grid: one panel,
the mean over the rows, and the **row-to-row spread** as a band around it -- a
mean spectrum hides whether a channel is steady or swinging by a factor of ten,
and the spread is what says which. The rows are streamed in blocks, so a whole
night of one station (1.5 GB here) never has to be in memory at once.

Usage:
    tools/autospectra.py RUN_DIR                      # newest file set
    tools/autospectra.py RUN_DIR --prefix 20260917_000000 --db
    tools/autospectra.py RUN_DIR --station W17 --db   # one station, mean +- spread
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


def zero_fill_per_station(set_info, n_station, take):
    """Zero-filled frames per station over the last `take` rows of blocks.csv."""
    zf = [0] * n_station
    per_row = {}
    try:
        with open(set_info["blocks"]) as fh:
            for line in fh:
                f = line.strip().split(",")
                if len(f) >= 6 and f[0].isdigit():
                    per_row.setdefault(int(f[0]), [0] * n_station)[int(f[1])] = int(f[5])
    except FileNotFoundError:
        return zf, 0
    for r in sorted(per_row)[-take:]:
        for i, v in enumerate(per_row[r][:n_station]):
            zf[i] += v
    return zf, len(per_row)


def station_average(path, nch, take, chunk=256, first=0):
    """Mean and row-to-row standard deviation of one autocorrelation file.

    Blocks of rows, not the whole file: a night of one station is 1.5 GB and the
    numbers wanted from it are two 16384-long arrays. Returns
    `(mean, std, rows_used)` with both in the file's own units.

    `first` skips whole rows from the front, which is how a set that a second run
    of the same day appended to (a gap in the middle, one file set per local day)
    gets averaged in the stretch that was asked for rather than across the gap.
    """
    rows_total = os.path.getsize(path) // (8 * nch)
    avail = max(rows_total - first, 0)
    rows = min(avail, take) if take else avail
    total = np.zeros(nch, dtype=np.float64)
    total_sq = np.zeros(nch, dtype=np.float64)
    done = 0
    with open(path, "rb") as fh:
        fh.seek(first * 8 * nch)
        while done < rows:
            want = min(chunk, rows - done)
            blk = np.fromfile(fh, dtype="<c8", count=want * nch)
            got = blk.size // nch
            if not got:
                break
            blk = blk[: got * nch].reshape(got, nch).real
            total += blk.sum(axis=0)
            total_sq += (blk.astype(np.float64) ** 2).sum(axis=0)
            done += got
    if not done:
        return None, None, 0
    mean = total / done
    var = np.maximum(total_sq / done - mean * mean, 0.0)
    return mean, np.sqrt(var), done


def plot_stations(set_info, names, wanted, spectra, rows_used, zf, freqs, a,
                  bin_hz, span, rows_avail):
    """One panel per requested station: mean spectrum and its row-to-row spread."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(wanted), 1, figsize=(13, 4.6 * len(wanted)),
                             dpi=110, squeeze=False)
    print(f"{'station':10s} {'rows used':>9s} {'zero-filled':>11s} {'median P':>12s} "
          f"{'peak MHz':>10s} {'peak/median':>11s} {'spread/median':>13s} {'DC/median':>10s}")
    for ax, (name, (mean, std, rows)) in zip(axes[:, 0], wanted):
        med = float(np.median(mean))
        pk = int(np.argmax(mean))
        # median of the per-channel row-to-row sigma: how much a typical channel moves
        rel = float(np.median(std / np.maximum(mean, 1e-30)))
        print(f"{name:10s} {rows:9d} {zf:11d} {med:12.4g} {freqs[pk]:10.3f} "
              f"{mean[pk] / med if med else 0:11.3f} {rel:13.4f} {mean[0] / med if med else 0:10.3f}")
        use_db = a.db
        lo = a.skip_bins
        f = freqs[lo:]
        # The band has to be scaled the same way as the mean, and in the file's
        # units before it is: scaling only the mean leaves the sigma in counts and
        # the panel's y-range becomes +-1e6 instead of a few dB.
        ref = float(np.max(mean))

        def scale(v):
            return 10 * np.log10(np.maximum(v, ref * 1e-12) / ref) if use_db else v

        y = scale(mean[lo:])
        y_lo = scale(np.maximum(mean[lo:] - std[lo:], 0.0))
        y_hi = scale(mean[lo:] + std[lo:])
        unit = ("dB, 0 dB = this station's strongest channel" if use_db
                else "power (int8 counts$^2$, no FFT normalisation)")
        ax.fill_between(f, y_lo, y_hi, color="#1f4e79", alpha=0.18, lw=0,
                        label="mean $\\pm$ 1$\\sigma$ across rows")
        ax.plot(f, y, lw=0.7, color="#1f4e79", label=f"mean of {rows} rows")
        ax.set_xlim(f[0], f[-1])
        if a.mhz:
            ax.set_xlim(*(float(v) for v in a.mhz.split(",")))
        ax.set_xlabel("MHz")
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(alpha=0.2, lw=0.4)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(
            f"{name}  autocorrelation |Y|$^2$   {set_info['prefix']}   "
            f"{rows} rows averaged, {span}"
            + (f", {zf} zero-filled frames" if zf else "")
            + f"   |   peak {freqs[pk]:.1f} MHz at {mean[pk] / med:.2f}x median"
            + (f"   |   first {a.skip_bins} bin(s) not drawn" if a.skip_bins else ""),
            fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prefix", default=None, help="which file set (local day); default the newest")
    ap.add_argument("--rows", type=int, default=0, help="rows to average; 0 = all")
    ap.add_argument("--out", default=None,
                    help="output PNG; default /tmp/autospectra.png, or <run_dir>/autospectra-<station>-<set>.png")
    ap.add_argument("--station", action="append", default=None, metavar="NAME",
                    help="draw this station alone (repeatable, or comma-separated); "
                         "default: the 24-panel grid")
    ap.add_argument("--row-range", default=None, metavar="A:B",
                    help="with --station: average rows A:B (0-based, half-open). One file set "
                         "per local day, so a day a second run appended to has a gap in the "
                         "middle and the whole-set average mixes the two stretches")
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

    # --- one station up close -------------------------------------------------
    if a.station:
        by_name = {names[j]: (j, p) for j, p in autos}
        wanted_names = [n.strip() for spec in a.station for n in spec.split(",") if n.strip()]
        missing = [n for n in wanted_names if n not in by_name]
        if missing:
            sys.exit(f"{a.run_dir}: no such station {', '.join(missing)}; "
                     f"this set has {' '.join(sorted(by_name))}")

        # Which rows: --row-range, else the last `take`, else everything.
        order = sorted(per_row)
        if a.row_range:
            lo_s, _, hi_s = a.row_range.partition(":")
            first = int(lo_s) if lo_s.strip() else 0
            n_sel = (int(hi_s) if hi_s.strip() else len(order)) - first
        else:
            first, n_sel = max(len(order) - take, 0), take
        n_sel = max(0, min(n_sel, len(order) - first))
        sel_rows = order[first:first + n_sel]
        if not sel_rows:
            sys.exit(f"row range {a.row_range!r} selects no row of {len(order)} in the set")
        # Zero fill over the rows actually being averaged, not over the whole set.
        zf_sel = {names[j]: 0 for j, _ in autos}
        for r in sel_rows:
            for i, v in enumerate(per_row[r][:n_station]):
                if names.get(i) in zf_sel:
                    zf_sel[names[i]] += v

        try:
            times = np.fromfile(s["rows_ms"], dtype="<i8") if os.path.exists(s["rows_ms"]) else None
        except OSError:
            times = None
        import datetime
        def stamp(v):
            return datetime.datetime.utcfromtimestamp(v / 1000.0 + 8 * 3600).strftime("%H:%M:%S")
        span = (f"rows {first}..{first + n_sel - 1} of {len(order)}, local "
                f"{stamp(int(times[first]))} .. {stamp(int(times[first + n_sel - 1]))}"
                if times is not None and times.size > first + n_sel - 1 else f"rows {first}..{first + n_sel - 1}")

        wanted, spectra = [], {}
        print(f"{s['prefix']}: {nch} channels, {bin_hz/1e6:.4f} MHz per bin, {span}")
        for name in wanted_names:
            j, path = by_name[name]
            mean, std, rows = station_average(path, nch, n_sel, first=first)
            if mean is None:
                sys.exit(f"{path}: no complete row to read")
            spectra[name] = (mean, std, rows)
            wanted.append((name, (mean, std, rows)))
        if not a.out:
            a.out = os.path.join(
                a.run_dir, f"autospectra-{'-'.join(wanted_names)}-{s['prefix']}"
                           + (f"-rows{first}-{first + n_sel}" if a.row_range else "") + ".png")
        return plot_stations(s, names, wanted, spectra, None, zf_sel[wanted_names[0]],
                             freqs, a, bin_hz, span, len(order))

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
