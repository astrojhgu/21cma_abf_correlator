#!/usr/bin/env python3
"""Read a run's blocks.csv and say, row by row, how much data was fabricated.

Why this exists: the run report prints a per-station total --- "zero_filled=2804"
over 38 rows looks like a clean run.  It is not.  A station can lose a little on
*every* row, and a total that small hides exactly the failure that matters for a
long run: a steady trickle means the receive side is marginally slower than the
front end, and the socket buffer is absorbing the deficit until it runs out.

`blocks.csv` has one line per row per station:

    row, station, start_frame, n_frames, valid_frames, zero_filled

so the trend across rows is visible, which is the whole question.

    tools/rows_trend.py /scratch/corr24-nf8192/<run dir>
    tools/rows_trend.py <run dir> --worst 8 --step 10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read(path: Path) -> list[tuple[int, int, int, int]]:
    """(row, station, valid, zero_filled) from a blocks.csv."""
    out = []
    with open(path) as f:
        for rec in csv.reader(f):
            if len(rec) < 6 or not rec[0].isdigit():
                continue
            # A live run is the main use case here, and its last line can be a
            # half-written record: the writer appends row by row as rows close.
            # A torn line is not corrupt data, it is a row that has not finished
            # being written, so it is skipped rather than raised.
            try:
                out.append((int(rec[0]), int(rec[1]), int(rec[4]), int(rec[5])))
            except ValueError:
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory containing blocks.csv")
    ap.add_argument("--step", type=int, default=1, help="print every Nth row")
    ap.add_argument("--worst", type=int, default=5, help="worst rows to list")
    a = ap.parse_args()

    d = Path(a.run_dir)
    blocks = d / "blocks.csv"
    if not blocks.is_file():
        print(f"no blocks.csv in {d}", file=sys.stderr)
        return 2
    recs = read(blocks)
    if not recs:
        print("blocks.csv has no data rows", file=sys.stderr)
        return 2

    rows = sorted({r for r, _, _, _ in recs})
    per_row: dict[int, list[tuple[int, int, int]]] = {r: [] for r in rows}
    for row, st, valid, zf in recs:
        per_row[row].append((st, valid, zf))

    stations = len(per_row[rows[0]])
    total_frames = sum(v + z for per in per_row.values() for _, v, z in per)
    total_zf = sum(z for per in per_row.values() for _, _, z in per)

    print(f"{d.name}: {len(rows)} rows x {stations} stations")
    print(
        f"  frames {total_frames:,} | zero filled {total_zf:,} "
        f"({100.0 * total_zf / max(total_frames, 1):.4f}%)"
    )
    clean = sum(1 for r in rows if all(z == 0 for _, _, z in per_row[r]))
    print(f"  rows with no zero fill at all: {clean}/{len(rows)}")

    worst = sorted(rows, key=lambda r: -sum(z for _, _, z in per_row[r]))[: a.worst]
    if worst:
        print("  worst rows:")
        for r in worst:
            per = per_row[r]
            zs = sorted((z for _, _, z in per), reverse=True)
            print(
                f"    row {r:5d}  zero_filled {sum(zs):7d}  worst station {zs[0]:6d}  "
                f"rows affected {sum(1 for z in zs if z)}"
            )

    # The trend is the point: a trickle that keeps rising means the deficit is
    # permanent, not a one-off burst that the socket buffer absorbed.
    print(f"  trend (every {a.step} row(s): zero-filled frames):")
    line = []
    for i, r in enumerate(rows):
        if i % a.step:
            continue
        line.append(f"{r}:{sum(z for _, _, z in per_row[r])}")
        if len(line) == 12:
            print("    " + " ".join(line))
            line = []
    if line:
        print("    " + " ".join(line))

    # Split the run in thirds to say whether it is getting worse.
    if len(rows) >= 9:
        n = len(rows) // 3
        def s(sub):
            return sum(z for r in sub for _, _, z in per_row[r])
        first, mid, last = s(rows[:n]), s(rows[n : 2 * n]), s(rows[2 * n :])
        print(f"  thirds: first {first:,} | middle {mid:,} | last {last:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
