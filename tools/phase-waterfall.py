#!/usr/bin/env python3
"""Time-order consistency: a phase waterfall per baseline, over a delay ladder.

`corr-replay --delay-stride s --ladder-frames N --ladder-steps K` delays station
i by `s*i*k` samples while the ladder step is k, stepping k every N frames. With
N = the correlator's `n_cum`, the step boundaries coincide with the row
boundaries, so row k integrates exactly one delay: the pair (i, j) sees

    dN(k) = s * k * (i - j)  samples   ->   phi(ch, k) = -2*pi*ch*dN(k)/M

with M = 2*n_freq. This script draws that as a waterfall -- frequency across,
step down, phase as colour, wrapped to +-180 degrees -- for every correlation
pair, and checks each (baseline, step) against the exact model.

What the picture is for. A delay that grows with the row makes the phase at each
channel rotate by a fixed amount per step, and the amount is proportional to
frequency, so lines of constant phase are the hyperbolas `ch * k = const`: they
crowd towards the top right. A correlator that paired the wrong frames, mixed up
the row order, or lost the time axis would not produce that.

    tools/phase-waterfall.py --dir /dev/shm/ladder/phasetest24 \\
        --stride 1 --ladder-frames 1024 --sample-rate 81920000 \\
        --outdir /tmp/phase-plots
"""

import argparse
import collections
import csv
import glob
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

NAME_RE = __import__("re").compile(r"[A-Za-z]+\d+")


def stem_pair(basename):
    for field in basename.replace(".bin", "").split("-"):
        found = NAME_RE.findall(field)
        if len(found) == 2:
            return found[0], found[1]
    return None


def station_order(pairs):
    nodes = sorted({x for p in pairs for x in p})
    succ = {x: set() for x in nodes}
    indeg = {x: 0 for x in nodes}
    for a, b in pairs:
        if a != b and b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1
    out, left = [], set(nodes)
    while left:
        ready = sorted(x for x in left if indeg[x] == 0)
        if not ready:
            out.extend(sorted(left))
            break
        x = ready[0]
        left.discard(x)
        out.append(x)
        for y in succ[x]:
            if y in left:
                indeg[y] -= 1
    return out


def load_steps(directory, n_freq, steps):
    """{(j, k): (phase_deg[steps, n_freq], sum_of_phase_deg_for_the_baseline)}.

    Only the first `steps` rows of each file are read: the file is written row by
    row, so those are ladder steps 1..steps, which `blocks.csv` confirms by
    giving each row's first frame id.
    """
    files = {}
    for path in glob.glob(os.path.join(directory, "*.bin")):
        base = os.path.basename(path)
        if base.endswith("-time.bin"):
            continue
        pair = stem_pair(base)
        if pair:
            files[pair] = path
    names = station_order(list(files))
    pos = {x: i for i, x in enumerate(names)}

    # Which file row is which ladder step. `blocks.csv` is headerless
    # `row_index,station,first_frame_id,frames,valid_frames,missing_frames`, one
    # record per station per row, and rows are written in order, so the i-th row
    # of a file is the i-th row of the log.
    rowk = {}
    csvs = glob.glob(os.path.join(directory, "*-blocks.csv"))
    if csvs:
        per_row = collections.defaultdict(dict)
        with open(csvs[0]) as fh:
            for rec in csv.reader(fh):
                if len(rec) < 3 or not rec[0].strip().isdigit():
                    continue                      # tolerate a header if one appears
                per_row[int(rec[0])][rec[1]] = int(rec[2])
        order = sorted(per_row)
        if order:
            base_ff = min(per_row[order[0]].values())
            span = (min(per_row[order[1]].values()) - base_ff) if len(order) > 1 else 0
            span = span or (min(per_row[order[0]].values()) and 0) or 1
            for i, r in enumerate(order):
                ff = min(per_row[r].values())
                rowk[i] = 1 + (ff - base_ff) // abs(span) if span else 1
            rowk["_base_ff"] = base_ff
            rowk["_span"] = span

    out = {}
    for (a, b), path in files.items():
        z = np.fromfile(path, dtype=np.complex64, count=steps * n_freq)
        z = z.reshape(-1, n_freq)
        if z.shape[0] < steps:
            continue
        out[(pos[a], pos[b])] = np.degrees(np.angle(z.astype(np.complex128)))
    return names, out, rowk


def check(names, data, steps, n_freq, sample_rate, stride):
    """Every (baseline, step) against the exact model; returns the worst error."""
    M = 2 * n_freq
    ch = np.arange(n_freq, dtype=np.float64)
    f_mhz = ch * (sample_rate / M) / 1e6
    design = np.vstack([f_mhz, np.ones_like(f_mhz)]).T
    worst, per_step = 0.0, {}
    for k in range(1, steps + 1):
        worst_k = 0.0
        for (i, j), ph in data.items():
            if i == j:
                continue
            dn = stride * k * (i - j)
            model = -2 * np.pi * ch * dn / M
            resid = np.unwrap(np.radians(ph[k - 1]) - model)
            coef, *_ = np.linalg.lstsq(design, resid, rcond=None)
            err_fs = -coef[0] / (2 * np.pi) * 1e9        # f in MHz -> fs
            worst_k = max(worst_k, abs(err_fs))
        per_step[k] = worst_k
        worst = max(worst, worst_k)
    return worst, per_step, f_mhz


def draw_grid(names, data, f_mhz, steps, out, panel_w=2.4, panel_h=0.86, dpi=90):
    n = len(names)
    fig = plt.figure(figsize=(panel_w * n + 2.0, panel_h * n + 2.2))
    gs = fig.add_gridspec(n, n, hspace=0.06, wspace=0.06,
                          left=0.03, right=0.93, top=0.955, bottom=0.05)
    norm = Normalize(-180, 180)
    for i in range(n):
        for j in range(i, n):
            ax = fig.add_subplot(gs[i, j])
            ph = data.get((i, j))
            if ph is None:
                ax.axis("off")
                continue
            ax.imshow(ph, aspect="auto", cmap="hsv", norm=norm,
                      extent=[f_mhz[0], f_mhz[-1], steps + 0.5, 0.5],
                      interpolation="nearest")
            ax.tick_params(labelsize=5.5, length=1.5, pad=1)
            if j > i:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[i], fontsize=7)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("%s (MHz)" % names[j], fontsize=7)
            for s in ax.spines.values():
                s.set_linewidth(0.4)
    cax = fig.add_axes([0.945, 0.35, 0.012, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="hsv"), cax=cax)
    cb.set_label("phase (deg, wrapped)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        "TIME-ORDER CONSISTENCY — phase waterfall for all %d correlation pairs\n"
        "station i delayed $s\\cdot i\\cdot k$ samples (%d sample) during ladder step k, "
        "one step per row of %d frames;  x = frequency, y = step 1..%d, colour = wrapped phase\n"
        "lines of constant phase are hyperbolas $\\nu \\cdot k$ = const: a row-order or "
        "frame-pairing error would not make this fan\n"
        "note: a long baseline wraps 100+ times across the band, which is past what a "
        "panel this size can show, so the far panels read as texture — "
        "waterfall-by-baseline.png resolves every length"
        % (len(data), 1, 1024, steps), fontsize=12, y=0.998, linespacing=1.6)
    fig.savefig(out, dpi=dpi)
    print("wrote", out)


def draw_selected(names, data, f_mhz, steps, out, picks, dpi=110):
    fig, axes = plt.subplots(len(picks), 1, figsize=(16, 1.5 * len(picks) + 1.6),
                             sharex=True)
    norm = Normalize(-180, 180)
    for ax, (i, j) in zip(axes, picks):
        ph = data[(i, j)]
        im = ax.imshow(ph, aspect="auto", cmap="hsv", norm=norm,
                       extent=[f_mhz[0], f_mhz[-1], steps + 0.5, 0.5],
                       interpolation="nearest")
        ax.set_ylabel("step k", fontsize=9)
        ax.set_yticks(range(1, steps + 1))
        ax.tick_params(labelsize=8)
        ax.set_title("%s%s   |i-j| = %d   (predicted slope %d samples of delay per step)"
                     % (names[i], names[j], abs(i - j), abs(i - j)), fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("phase (deg, wrapped)", fontsize=9)
    axes[-1].set_xlabel("frequency (MHz)", fontsize=10)
    fig.suptitle("phase waterfall, four baselines of different length", fontsize=12)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print("wrote", out)


def draw_by_length(names, data, f_mhz, steps, out, dpi=100):
    """One full-width waterfall per baseline length |i-j| = 1..n-1.

    In the all-pairs grid a long baseline's pattern is 100+ cycles wide inside a
    panel a couple of hundred pixels across, which is past the display limit and
    reads as texture. Here each panel gets the full page width, so the fan is
    resolved for every baseline length -- which is the thing to look at.
    """
    n = len(names)
    lengths = list(range(1, n))
    fig, axes = plt.subplots(len(lengths), 1,
                             figsize=(16, 0.95 * len(lengths) + 1.8), sharex=True)
    norm = Normalize(-180, 180)
    for ax, d in zip(axes, lengths):
        ph = data[(0, d)]
        im = ax.imshow(ph, aspect="auto", cmap="hsv", norm=norm,
                       extent=[f_mhz[0], f_mhz[-1], steps + 0.5, 0.5],
                       interpolation="nearest")
        ax.set_yticks(range(1, steps + 1))
        ax.tick_params(labelsize=7)
        ax.set_ylabel("step k", fontsize=8)
        ax.text(0.995, 0.90, "|i-j| = %d   (%.1f wraps across the band at k=%d)"
                % (d, d * steps * 16384 / 32768, steps), transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="k",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("phase (deg, wrapped)", fontsize=9)
    axes[-1].set_xlabel("frequency (MHz)", fontsize=10)
    fig.suptitle("phase waterfall for every baseline length, |i-j| = 1..%d "
                 "(baseline R01R<1+|i-j|>)" % (n - 1), fontsize=12)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--outdir", default="/tmp/phase-plots")
    ap.add_argument("--n-freq", type=int, default=16384)
    ap.add_argument("--sample-rate", type=float, default=81920000.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--ladder-frames", type=int, default=1024)
    ap.add_argument("--origin", type=int, default=0,
                    help="the frame the replayer's ladder started at, to check "
                         "against the first row's frame id")
    args = ap.parse_args()

    names, data, rowk = load_steps(args.dir, args.n_freq, args.steps)
    print("%d stations, %d baselines, %d steps each" % (len(names), len(data), args.steps))
    if rowk:
        base_ff, span = rowk["_base_ff"], rowk["_span"]
        print("blocks.csv: first row starts at frame %d, rows %d frames apart" % (base_ff, span))
        if args.origin:
            print("  replayer's ladder origin was %d -> %s"
                  % (args.origin, "MATCH" if base_ff == args.origin
                     else "MISMATCH, the first row is not step 1"))
        if span != args.ladder_frames:
            print("  WARNING: row spacing %d != --ladder-frames %d"
                  % (span, args.ladder_frames))
        print("file row -> step: %s"
              % {r: k for r, k in sorted(
                  (r, k) for r, k in rowk.items() if isinstance(r, int))[:args.steps]})

    worst, per_step, f_mhz = check(names, data, args.steps, args.n_freq,
                                   args.sample_rate, args.stride)
    print("\n每步的最大 |时延误差| (对全部 %d 条基线):" % len(data))
    for k, v in per_step.items():
        print("  step %2d: %.1f fs   (注入延迟差 %d..%d 采样)"
              % (k, v, args.stride * k, args.stride * k * (len(names) - 1)))
    print("worst over all steps: %.1f fs" % worst)

    os.makedirs(args.outdir, exist_ok=True)
    draw_grid(names, data, f_mhz, args.steps,
              os.path.join(args.outdir, "waterfall-all-pairs.png"))
    long = len(names) - 1
    picks = [(0, 1), (0, max(2, long // 3)), (0, max(3, 2 * long // 3)), (0, long)]
    draw_selected(names, data, f_mhz, args.steps,
                  os.path.join(args.outdir, "waterfall-selected.png"), picks)
    draw_by_length(names, data, f_mhz, args.steps,
                   os.path.join(args.outdir, "waterfall-by-baseline.png"))


if __name__ == "__main__":
    main()
