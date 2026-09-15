#!/usr/bin/env python3
"""Generate a golden input/output pair for the corr_cuda numerical test.

This is an *independent* reference: numpy's rfft plus a straightforward
Python-level accumulation, written from the frozen decisions in HANDOFF.md
section 4 rather than from the CUDA code. Its only job is to disagree loudly if
the kernel gets the FFT stride, the Nyquist bin, the normalisation, the baseline
numbering or the real/imaginary convention wrong -- errors that would otherwise
produce plausible-looking spectra.

usage: gen_golden.py N_STATION N_FREQ N_CUM FRAMES_PER_SLOT ROWS OUTDIR [SEED]

writes into OUTDIR:
    meta.txt      header "n_station n_freq n_cum frames_per_slot rows", then
                  baseline -> (j,k) for every baseline
    in.bin        int8, layout [row][station][frame][2*n_freq]
    expect.bin    complex64 (two little-endian float32),
                  [row][baseline][n_freq]

Several rows are generated so that the device-side row stack (rows_per_fetch) is
exercised, and n_cum is deliberately not a multiple of frames_per_slot so the
last staging slot of a row is partial.
"""

import os
import sys

import numpy as np


def baseline_pairs(n_station):
    """(j, k) for every baseline with j <= k, row major. This is the definition
    corr_baseline_pairs() implements on the C side."""
    return [(j, k) for j in range(n_station) for k in range(j, n_station)]


def main():
    if len(sys.argv) < 6:
        print(__doc__)
        return 2

    n_station = int(sys.argv[1])
    n_freq = int(sys.argv[2])
    n_cum = int(sys.argv[3])
    frames_per_slot = int(sys.argv[4])
    rows = int(sys.argv[5])
    outdir = sys.argv[6]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 20260914

    if n_freq % 2048 != 0:
        print(f"n_freq={n_freq} is not a multiple of 2048")
        return 2

    fft_len = 2 * n_freq
    os.makedirs(outdir, exist_ok=True)

    # int8 in the range the front end actually produces (measured samples sit in
    # -4..+4), with a few extremes so sign handling is exercised.
    rng = np.random.default_rng(seed)
    samples = rng.integers(-4, 5, size=(rows, n_station, n_cum, fft_len)).astype(np.int8)
    samples[0, 0, 0, 0] = -128
    samples[0, 0, 0, 1] = 127

    samples.tofile(os.path.join(outdir, "in.bin"))

    # Reference spectrum. float64 throughout so the golden file is limited by
    # the CUDA side's float32, not by the reference.
    x = samples.astype(np.float64)
    y = np.fft.rfft(x, axis=3)[:, :, :, :n_freq]  # drop the Nyquist bin

    pairs = baseline_pairs(n_station)
    corr = np.zeros((rows, len(pairs), n_freq), dtype=np.complex128)
    for r in range(rows):
        for b, (j, k) in enumerate(pairs):
            corr[r, b] = (y[r, j] * np.conj(y[r, k])).sum(axis=0)

    # The frozen decision: divide by n_cum, never by the valid frame count.
    corr /= n_cum

    corr.astype(np.complex64).tofile(os.path.join(outdir, "expect.bin"))

    with open(os.path.join(outdir, "meta.txt"), "w") as f:
        f.write(f"{n_station} {n_freq} {n_cum} {frames_per_slot} {rows}\n")
        for b, (j, k) in enumerate(pairs):
            f.write(f"{b} {j} {k}\n")

    # Sanity properties of the reference itself, so a broken generator is not
    # mistaken for a broken kernel.
    diag = [b for b, (j, k) in enumerate(pairs) if j == k]
    worst_imag = max(abs(corr[r, b].imag).max() for r in range(rows) for b in diag)
    scale = max(abs(corr[r, b]).max() for r in range(rows) for b in diag)
    print(f"golden: n_station={n_station} n_freq={n_freq} n_cum={n_cum} "
          f"frames_per_slot={frames_per_slot} rows={rows} baselines={len(pairs)}")
    print(f"golden: input {samples.size} int8 bytes, output {corr.nbytes} bytes")
    print(f"golden: autocorr |max| = {scale:.6g}, max |imag| on the diagonal = "
          f"{worst_imag:.3g} (should be ~0: a real signal has a real autospectrum)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
