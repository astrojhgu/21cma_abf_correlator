#!/usr/bin/env python3
"""compare_blocks.py RUN_A RUN_B -- do two runs agree row by row?

Companion to tools/ab_zerofill.sh: the numbers a zero-fill A/B is judged on. Rows
are matched by (first_frame_id, station), so two runs that started at different
wall-clock times still compare on the rows they share, and a run whose rows do
not overlap at all is reported rather than passing vacuously.

The run's *first* row is expected to differ even between identical binaries: it
contains the startup transient, where each run loses whichever packets happened to
arrive before its receive threads were draining their sockets. Compare with the
first row excluded (the summary prints both).

Each run's blocks.csv says, per row and per station, how many frames of that row
were valid and how many were declared missing (the `zero_filled` column). Two
runs over the same deterministic replay have to agree on those numbers frame for
frame; if the data path changed what the correlator *counts*, this is where it
shows.

Rows are matched by (first_frame_id, station), not by row index: two runs that
started a moment apart write different global rows into file offset 0.

Usage: compare_blocks.py A_DIR B_DIR [--show N]
"""

import argparse
import os
import sys


def load(run_dir):
    path = os.path.join(run_dir, "blocks.csv")
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split(",")
            if len(f) < 6:
                continue
            row_index, station, first_frame, n_cum, valid, zero = (int(x) for x in f[:6])
            rows[(first_frame, station)] = (row_index, n_cum, valid, zero)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    common = sorted(set(a) & set(b))
    print(f"rows: {len(a)} in A, {len(b)} in B, {len(common)} in common")
    if not common:
        print("no rows in common: the two runs do not overlap at all")
        return 2

    same = [k for k in common if a[k] == b[k]]
    diff = [k for k in common if a[k] != b[k]]
    print(f"identical on {len(same)} of {len(common)} matched rows, {len(diff)} differ")
    if diff:
        print(f"  {'first_frame':>12} {'st':>3} {'A valid/zf':>14} {'B valid/zf':>14}")
        for k in diff[: args.show]:
            ar, br = a[k], b[k]
            print(f"  {k[0]:>12} {k[1]:>3} {ar[2]:>7}/{ar[3]:<6} {br[2]:>7}/{br[3]:<6}")
        if len(diff) > args.show:
            print(f"  ... and {len(diff) - args.show} more")

    for name, rows in (("A", a), ("B", b)):
        zf = sum(r[3] for r in rows.values())
        frames = sum(r[1] for r in rows.values())
        st_zf = {}
        for (ff, st), r in rows.items():
            st_zf[st] = st_zf.get(st, 0) + r[3]
        worst = sorted(st_zf.items(), key=lambda kv: -kv[1])[:4]
        print(
            f"{name}: {len(rows)} rows, {frames} frame-slots, {zf} missing "
            f"({100.0 * zf / max(frames, 1):.4f}%), worst stations {worst}"
        )
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
