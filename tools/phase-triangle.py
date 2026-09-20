#!/usr/bin/env python3
"""All correlation pairs as an upper-triangular grid of phase-vs-frequency.

The layout is the usual corner-plot one: row `j` is station j, column `k` is
station k, and only `k >= j` is filled, so the diagonal down the left edge is the
autocorrelations and the grid fans out to the longest baselines at the top right.
Panel (j, k) is baseline (j, k), i.e. `Z_jk = Y_j conj(Y_k)`, as phase against
frequency. Frequency is in MHz and phase in **turns** (cycles), on one common
scale for the whole grid: with a delay gradient the ramp steepens as |j-k| grows,
and that fan is the point of drawing them all together. A per-panel scale would
make every panel look identical.

Two modes:

  `--mode replay`  the data came from `corr-replay --delay-stride`, so the
                   expected phase is exact: phi = -2*pi*k*dN/M, i.e. -k*dN/M
                   turns. It is drawn dashed over the measurement, and the number
                   in the corner is the error against it, in fs.

  `--mode real`    the live front end: no model, so the least-squares line is
                   drawn instead and the number in the corner is its delay. Only
                   pairs that share a front-end board are coherent at all; the
                   rest are noise, their "ramps" are unwrapped random walks and
                   their fitted delay means nothing. Drawing them all is the
                   point: the coherent pair is the one that stands out.

**Station order comes from the files, not from sorting the names.** A baseline
file is `<file_prefix><name_j><name_k>-<stamp>.bin` with `j <= k` in the *plan's*
station order, so every file asserts an ordering constraint. Sorting the names
alphabetically gets that wrong whenever the plan order is not alphabetical -- for
the front end it is `W17, S01, S06, N04`, so `W17N04` is baseline (0,3), while
alphabetically it looks like (3,0): outside the triangle, and every pair the
coherent station is in would silently vanish.

    tools/phase-triangle.py --mode replay --dir /tmp/data24-stride64 \\
        --stride 64 --sample-rate 81920000 --out /tmp/phase-plots/triangle-24st.png
    tools/phase-triangle.py --mode real --dir /dev/shm/phase/corr \\
        --sample-rate 800000000 --out /tmp/phase-plots/triangle-real.png
"""

import argparse
import glob
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

BLUE, RED = "#1f77b4", "#d62728"

# Station names look like `W17`, `S01`, `R24`, `T04`: letters then digits.
NAME_RE = re.compile(r"[A-Za-z]+\d+")


def stem_pair(basename):
    """(name_j, name_k) from a baseline file name, prefix and stamp and all.

    The prefix may itself contain a dash (`corr24-`), so the fields are scanned
    and the first that holds two station-shaped names is the pair: a date field
    holds none, and a prefix holds at most one.
    """
    for field in basename.replace(".bin", "").split("-"):
        found = NAME_RE.findall(field)
        if len(found) == 2:
            return found[0], found[1]
    return None


def station_order(pairs):
    """Names in plan order, from the ordering every file name asserts.

    Each file says `name_j <= name_k`, so a topological sort of those constraints
    recovers the order the plan used. Ties break alphabetically, so the result is
    deterministic.
    """
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
        if not ready:                      # only if the file names disagree
            out.extend(sorted(left))
            break
        x = ready[0]
        left.discard(x)
        out.append(x)
        for y in succ[x]:
            if y in left:
                indeg[y] -= 1
    return out


def collect(directory):
    mapping = {}
    for path in glob.glob(os.path.join(directory, "*.bin")):
        base = os.path.basename(path)
        if base.endswith("-time.bin"):     # sidecar, not a baseline
            continue
        pair = stem_pair(base)
        if pair:
            mapping[pair] = path
    names = station_order(list(mapping))
    pos = {x: i for i, x in enumerate(names)}
    return names, {(pos[a], pos[b]): p for (a, b), p in mapping.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "real"], required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-freq", type=int, default=16384)
    ap.add_argument("--sample-rate", type=float, required=True)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--delays", default=None,
                    help="explicit per-station delays in samples, comma separated")
    ap.add_argument("--step", type=int, default=6, help="plot every Nth channel")
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--panel", type=float, default=1.95, help="panel size, inches")
    ap.add_argument("--yscale", choices=["panel", "common"], default="panel",
                    help="'panel' fills each panel with its own ramp (best for "
                         "reading one baseline); 'common' puts every panel on one "
                         "scale so the fan across |j-k| is visible")
    args = ap.parse_args()

    names, files = collect(args.dir)
    n = len(names)
    if not names or not files:
        raise SystemExit("no baseline files found in %s" % args.dir)
    print("%d stations in plan order: %s" % (n, ", ".join(names)))
    print("%d baselines" % len(files))

    delays = ([int(x) for x in args.delays.split(",")] if args.delays
              else [i * args.stride for i in range(n)])
    nf = args.n_freq
    m_len = 2.0 * nf
    chan = np.arange(nf, dtype=np.float64)
    f_mhz = chan * (args.sample_rate / m_len) / 1e6
    sel = np.arange(0, nf, args.step)

    data, ymax, worst, rows = {}, 0.0, 0.0, 0
    per_panel = args.yscale == "panel"
    for (j, k), path in files.items():
        z = np.fromfile(path, dtype=np.complex64).reshape(-1, nf)
        rows = z.shape[0]
        m = z.astype(np.complex128).mean(axis=0)
        # Unwrap, and drop the constant: the model also starts at zero, so the two
        # can be drawn on top of each other. The wrapped phase would not be a ramp.
        unwrapped = np.unwrap(np.angle(m))
        unwrapped -= unwrapped[0]
        ph = unwrapped / (2 * np.pi)                          # turns
        if args.mode == "replay":
            dn = delays[j] - delays[k]
            model = -chan * dn / m_len                       # turns
            resid = np.unwrap(np.angle(
                m * np.conj(np.exp(1j * (-2 * np.pi * chan * dn / m_len)))))
            w = np.abs(m)
            design = np.vstack([f_mhz, np.ones_like(f_mhz)]).T
            coef, *_ = np.linalg.lstsq(design * w[:, None], resid * w, rcond=None)
            # f is in MHz, so tau[s] = -coef/(2*pi*1e6); in fs that is *1e15.
            err_fs = -coef[0] / (2 * np.pi) * 1e9
            note = ("$\\Delta N$=%d ns\n%.1f fs" % (dn * 1e9 / args.sample_rate, err_fs)
                    if j != k else "auto")
            if j != k:
                worst = max(worst, abs(err_fs))
            span = max(np.abs(model).max(), np.abs(ph).max(), 1e-12)
            ymax = max(ymax, span)
        else:
            good = np.abs(m) > 0
            coef = np.polyfit(f_mhz[good], unwrapped[good], 1)
            model = (coef[0] * f_mhz + coef[1]) / (2 * np.pi)   # turns
            note = ("%.0f ns" % (-coef[0] / (2 * np.pi) * 1e3)) if j != k else "auto"
            span = max(np.abs(ph).max(), np.abs(model).max(), 1e-12)
            ymax = max(ymax, span)
        data[(j, k)] = (ph, model, m, note, span)

    ymax = max(ymax, 1e-12) * 1.08
    fig = plt.figure(figsize=(args.panel * n + 2.4, args.panel * n + 2.6))
    # add_gridspec would otherwise use the `figure.subplot.*` margins and leave a
    # 12% border on the left and a 12% one on top, for a grid that wants the page.
    gs = fig.add_gridspec(n, n, hspace=0.07, wspace=0.07,
                          left=0.03, right=0.995, top=0.962, bottom=0.038)
    for j in range(n):
        for k in range(j, n):
            ax = fig.add_subplot(gs[j, k])
            if (j, k) not in data:
                ax.axis("off")
                continue
            ph_turns, model, m, note, span = data[(j, k)]
            lim = (span if per_panel else ymax) * 1.1
            shown = ph_turns[sel]
            shown = shown - shown[0] + model[sel][0]        # no wrap jumps in the plot
            ax.plot(f_mhz[sel], shown, "-", color=BLUE, lw=0.8)
            ax.plot(f_mhz[sel], model[sel], "--", color=RED, lw=0.8)
            ax.set_xlim(0, f_mhz[-1])
            ax.set_ylim(-lim, lim)
            if j == k:
                ax.text(0.5, 0.06, "auto  |Im|/|Re|=%.0e"
                        % (np.abs(m.imag).max() / max(np.abs(m.real).max(), 1e-30)),
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.4, color=RED)
            else:
                ax.text(0.97, 0.06, note, transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=6.4, color=RED, linespacing=1.15)
            ax.text(0.04, 0.87, "%s%s" % (names[j], names[k]), transform=ax.transAxes,
                    fontsize=6.2, color="k")
            ax.tick_params(labelsize=6.2, length=2.0, pad=1.0)
            if k > j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[j], fontsize=8, labelpad=2)
            if j < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("%s   (MHz)" % names[k], fontsize=7.0, labelpad=1)
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            ax.grid(alpha=0.25, lw=0.3)

    if args.mode == "replay":
        head = ("ALL %d CORRELATION PAIRS — replayed delay gradient:  phase vs frequency\n"
                "%d stations in plan order, %d baselines, %d rows each;   blue = measured, "
                "red dashed = exact $\\varphi_k = -2\\pi k\\,\\Delta N/M$\n"
                "corner label = injected $\\Delta N$ in ns, and the error of the "
                "measurement against it (worst %.2f fs)   —   one common phase scale, in turns"
                ) % (len(files), n, len(files), rows, worst)
    else:
        head = ("ALL %d CORRELATION PAIRS — live front end:  phase vs frequency\n"
                "%d stations in plan order, %d baselines, %d rows each;   blue = measured, "
                "red = least-squares fit;   corner label = fitted delay\n"
                "only pairs on the same front-end board are coherent; the rest are "
                "unwrapped noise, so their delays mean nothing"
                ) % (len(files), n, len(files), rows)
    fig.suptitle(head, fontsize=13, y=0.999, linespacing=1.6)
    fig.savefig(args.out, dpi=args.dpi)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
