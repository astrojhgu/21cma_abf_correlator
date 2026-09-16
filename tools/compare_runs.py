#!/usr/bin/env python3
"""compare_runs.py -- do two runs of the same input agree, baseline by baseline?

The multi-card path has one job: produce the *same numbers* as the single-card
path. Every baseline is computed on exactly one card, from the same frames, in
the same order, by the same kernels, so agreement is expected to be exact -- not
"close". This compares the rows the two runs share, byte for byte, and says where
the first difference is when they disagree.

Usage:
    compare_runs.py DIR_A DIR_B [--tolerance REL]

Rows are matched by `first_frame_id` from blocks.csv, so two runs that started at
different times still compare on the rows they have in common; a run whose rows
do not overlap at all is reported as such rather than passing vacuously.

Exit status: 0 identical (within tolerance), 1 differences, 2 could not compare.
"""

import argparse
import glob
import os
import sys

import corr_common

import numpy as np


def read_blocks(run_dir, prefix=None):
    """first_frame_id -> file row offset, from blocks.csv.

    The `row` column of blocks.csv is the *global* row index (first_frame_id /
    n_cum), not a file offset: two runs that started at different times write
    different global rows into the same file offset 0. Matching runs therefore
    has to be done on first_frame_id, and reading a row needs that run's own
    offset for it.
    """
    first_seen = []
    path = corr_common.pick_set(run_dir, prefix)["blocks"]
    if not os.path.exists(path):
        sys.exit(f"{run_dir}: no blocks.csv")
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            first_frame = int(parts[2])
            if not first_seen or first_seen[-1] != first_frame:
                first_seen.append(first_frame)
    return {ff: off for off, ff in enumerate(first_seen)}


def load_metadata(run_dir, prefix=None):
    meta = {}
    with open(corr_common.pick_set(run_dir, prefix)["metadata"]) as fh:
        for line in fh:
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


def read_row(run_dir, row, n_baseline, n_freq):
    """complex64[n_baseline][n_freq] of one row, from the per-baseline files."""
    out = np.empty((n_baseline, n_freq), dtype=np.complex64)
    row_bytes = n_freq * 8
    for b in range(n_baseline):
        # The set's prefix goes in front of every file, so the glob carries it.
        path = glob.glob(
            os.path.join(run_dir, f"{corr_common.pick_set(run_dir, prefix)['prefix']}baseline_{b:04d}_*.bin")
        )
        if not path:
            sys.exit(f"{run_dir}: no file for baseline {b}")
        with open(path[0], "rb") as fh:
            fh.seek(row * row_bytes)
            raw = fh.read(row_bytes)
        if len(raw) != row_bytes:
            sys.exit(f"{run_dir}: baseline {b} is short at row {row}")
        out[b] = np.frombuffer(raw, dtype=np.complex64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument(
        "--prefix",
        default=None,
        help="which file set (local day) to compare in both runs; default is the newest",
    )
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="relative RMS tolerance; 0 means require bit equality")
    ap.add_argument("--max-rows", type=int, default=5,
                    help="how many shared rows to compare")
    args = ap.parse_args()

    meta_a = load_metadata(args.dir_a, args.prefix)
    meta_b = load_metadata(args.dir_b, args.prefix)
    for key in ("n_baseline", "n_freq", "n_cum"):
        if meta_a.get(key) != meta_b.get(key):
            sys.exit(f"runs differ in {key}: {meta_a.get(key)} vs {meta_b.get(key)}")
    n_baseline = int(meta_a["n_baseline"])
    n_freq = int(meta_a["n_freq"])

    blocks_a = read_blocks(args.dir_a, args.prefix)
    blocks_b = read_blocks(args.dir_b, args.prefix)
    shared = sorted(set(blocks_a) & set(blocks_b))
    if not shared:
        sys.exit(f"no shared rows: A covers {len(blocks_a)} rows "
                 f"({min(blocks_a) if blocks_a else '-'}..{max(blocks_a) if blocks_a else '-'}), "
                 f"B covers {len(blocks_b)} "
                 f"({min(blocks_b) if blocks_b else '-'}..{max(blocks_b) if blocks_b else '-'}); "
                 f"run both against the same input range")

    print(f"A = {args.dir_a}")
    print(f"B = {args.dir_b}")
    print(f"{n_baseline} baselines x {n_freq} bins, {len(shared)} shared rows, "
          f"comparing {min(len(shared), args.max_rows)}")

    worst = 0.0
    compared = 0
    for first_frame in shared[: args.max_rows]:
        row = first_frame
        a = read_row(args.dir_a, blocks_a[first_frame], n_baseline, n_freq)
        b = read_row(args.dir_b, blocks_b[first_frame], n_baseline, n_freq)
        if np.array_equal(a, b):
            print(f"  first frame {first_frame} (A row {blocks_a[first_frame]}, B row {blocks_b[first_frame]}): identical")
            compared += 1
            continue
        diff = np.abs(a - b)
        scale = np.abs(a).mean() if np.abs(a).mean() > 0 else 1.0
        rel = float(diff.max() / scale)
        worst = max(worst, rel)
        where = np.unravel_index(int(np.argmax(diff)), diff.shape)
        print(f"  first frame {first_frame} (A row {blocks_a[first_frame]}, B row {blocks_b[first_frame]}): differs, "
              f"max |d| = {diff.max():.3e} ({rel:.3e} relative) at "
              f"baseline {where[0]} bin {where[1]}: {a[where]} vs {b[where]}")
        if args.tolerance > 0 and rel <= args.tolerance:
            compared += 1

    if compared == min(len(shared), args.max_rows):
        print("RESULT: same numbers within the given tolerance"
              + (" (exact)" if args.tolerance == 0 else f" (worst {worst:.2e})"))
        return 0
    print("RESULT: DIFFERENT -- the multi-card split does not reproduce the "
          "single-card result")
    return 1


if __name__ == "__main__":
    sys.exit(main())
