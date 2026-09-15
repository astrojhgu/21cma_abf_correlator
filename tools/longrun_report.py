#!/usr/bin/env python3
"""Verdict table for long runs: is this configuration's data intact, and at what load?

A correlator that silently fabricates 2% of its frames is worse than one that
crashes, and the run report cannot be trusted to say so: it prints a per-station
total, which hid a per-row trickle for weeks (see `rows_trend.py`).  This is the
one command that answers "is this configuration acceptable", from the artefacts
the run left behind rather than from anything it says about itself:

    tools/longrun_report.py /scratch/corr24-nf8192/long30/*/ \
                            /scratch/corr24-nf8192/pair100/*/ \
        --load /tmp/pair100-samples.txt

Every input is a run directory (the one holding blocks.csv).  `--load` is a file
of lines containing `load=NN`, as a sampling loop writes; the load range is
printed because this machine is shared and has been measured to move a run
between 0.0001% and 25% fabricated frames at identical settings.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def load_blocks(path: Path):
    """(row, station, valid, zero_filled), tolerating a half-written last line."""
    out = []
    with open(path) as f:
        for rec in csv.reader(f):
            if len(rec) < 6 or not rec[0].isdigit():
                continue
            try:
                out.append((int(rec[0]), int(rec[1]), int(rec[4]), int(rec[5])))
            except ValueError:
                continue
    return out


def summarise(d: Path):
    blocks = d / "blocks.csv"
    if not blocks.is_file():
        return None
    recs = load_blocks(blocks)
    if not recs:
        return None
    rows = sorted({r for r, _, _, _ in recs})
    # Sum over every record, not over the first record of each row: the first
    # version did the latter and reported 24x the true fabrication rate, which is
    # exactly the kind of quiet factor of n_station that this tool exists to catch.
    frames = sum(v + z for _, _, v, z in recs)
    zf_per_row: dict[int, int] = {}
    for row, _st, _v, zf in recs:
        zf_per_row[row] = zf_per_row.get(row, 0) + zf
    zf = sum(zf_per_row.values())
    n = len(rows)
    third = max(n // 3, 1)
    def s(sub):
        return sum(zf_per_row[r] for r in sub)
    return {
        "run": d.parent.name + "/" + d.name,
        "rows": n,
        "frames": frames,
        "zf": zf,
        "pct": 100.0 * zf / max(frames, 1),
        "clean": sum(1 for r in rows if zf_per_row[r] == 0),
        "worst": max(zf_per_row.values()) if rows else 0,
        "thirds": (s(rows[:third]), s(rows[third : 2 * third]), s(rows[2 * third :])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--load", action="append", default=[], help="file(s) with 'load=NN'")
    ap.add_argument("--seconds-per-row", type=float, default=1.00139008)
    a = ap.parse_args()

    loads: list[float] = []
    for p in a.load:
        for line in open(p):
            m = re.search(r"load=([\d.]+)", line)
            if m:
                loads.append(float(m.group(1)))

    print(f"{'run':34} {'rows':>5} {'zero-filled':>9} {'%':>8} {'clean rows':>11} "
          f"{'thirds (M)':>22}")
    worst = 0.0
    for r in a.runs:
        d = Path(r)
        if not d.is_dir():
            d = d.parent
        info = summarise(d)
        if info is None:
            print(f"{str(d):34} (no blocks.csv)")
            continue
        worst = max(worst, info["pct"])
        t = "/".join(f"{v / 1e6:.1f}" for v in info["thirds"])
        print(
            f"{info['run']:34} {info['rows']:>5} {info['zf']:>9,} {info['pct']:>7.4f}% "
            f"{info['clean']:>5}/{info['rows']:<5} {t:>22}"
        )
    if loads:
        print(f"load during the runs: {min(loads):.1f} .. {max(loads):.1f}")
    print(f"verdict: worst configuration fabricates {worst:.4f}% of frames")
    print("  acceptance (proposed): < 0.01% at load >= 24, with rows on schedule")
    return 0 if worst < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
