#!/usr/bin/env python3
"""Check the correlator's phase against a replayed delay we chose ourselves.

`corr-replay --pattern prng --sample-delay` / `--delay-stride` generates every
station from one deterministic sample stream, shifted by a known whole number of
samples. That makes the expected phase of every baseline exact:

    Z_jk[k] = Y_j[k] conj(Y_k[k]) = |R[k]|^2 * exp(-2*pi*i*k*(N_j - N_k)/M)

with M = 2 * n_freq the transform length and N_i the injected delay of station i
in samples. There is no fitted parameter and no constant to absorb an error: a
wrong conjugation, a reversed bin order, a wrong transform length or a
mis-assembled frame all move the phase off this line.

Three things are worth knowing before reading the output.

* **Zero delay is not a weak test.** With `--delay-stride 0` every station gets
  the same stream, so every baseline must come out at exactly zero phase --
  measured at the 1e-11 rad level, the correlator's arithmetic floor. A
  station-dependent phase error anywhere in the chain shows up immediately.

* **The residual has a predicted size.** The two stations of a baseline look at
  *different windows* of the one stream (they overlap on M - |tau| of M samples),
  so even a perfect correlator sees a per-channel phase error of

      sigma = sqrt(|tau| / M) / sqrt(n_frames)

  Everything else here -- windowing, the FFT, the pairwise product, the
  accumulation, the host-side sum -- is exact, so that is what the residual
  should be. Larger means a real phase error; at the prediction means the test
  signal's own fault.

* **Estimate the delay by removing the model first.** Both stations' phases are
  ramps of thousands of radians (1472 samples is 736 turns across the band), so
  unwrapping the measured and the expected phase separately and subtracting the
  two is numerically fragile; wrapping the *difference* and unwrapping that is
  not. The delay error is then the residual's linear term, and it comes out in
  femtoseconds rather than picoseconds.

Usage:

    corr-replay --config <cfg> --rate 20000 --duration 60 --pattern prng \\
        --delay-stride 64 &
    corr-async  --config <cfg> --workers 24 --duration 25 --output-dir <dir>
    tools/phase-replay-check.py <dir> --stride 64 --n-station 24
"""

import argparse
import glob
import os
import re
import sys

import numpy as np


def station_names(n_station, pad=2):
    return ["R%0*d" % (pad, i + 1) for i in range(n_station)]


def infer_names(stems):
    """Station names from the file names, prefix and all."""
    names = set()
    for stem in stems:
        pair = stem_pair(stem)
        if pair:
            names.update(pair)
    return sorted(names)


def parse_pair(basename, names):
    """`R07R19-<stamp>.bin` -> (6, 18), also with a file prefix in front."""
    pair = stem_pair(basename)
    if pair and pair[0] in names and pair[1] in names:
        return names.index(pair[0]), names.index(pair[1])
    return None

# Station names look like `W17`, `S01`, `R24`, `T04`: letters then digits. The
# file is `<file_prefix><name_j><name_k>-<stamp>.bin`, and the prefix can itself
# end in letters and digits (`corr24-`), so the two names are taken as the last
# two matches in the stem rather than by halving it.
NAME_RE = re.compile(r"[A-Za-z]+\d+")


def stem_pair(basename):
    """(name_j, name_k) from a baseline file name, prefix and stamp and all.

    The name is `<file_prefix><name_j><name_k>-<stamp>.bin` and the prefix may
    itself contain a dash (`corr24-`), so the fields are scanned and the first one
    that holds two station-shaped names is the pair. A date field holds none and
    a prefix holds at most one, which is what makes this unambiguous.
    """
    for field in basename.replace(".bin", "").split("-"):
        found = NAME_RE.findall(field)
        if len(found) == 2:
            return found[0], found[1]
    return None



def measure(path, dn, n_freq, sample_rate):
    """One baseline's phase against the exact model for `dn` samples of delay.

    Returns (tau_ns, tau_expected_ns, residual_rms, predicted_rms, offset_rad,
    n_frames)."""
    z = np.fromfile(path, dtype=np.complex64).reshape(-1, n_freq)
    # complex128 for the average: the phase of a float32 mean is quantised at
    # ~1e-7 rad, which is the same order as the effects being measured.
    m = z.astype(np.complex128).mean(axis=0)
    rows = z.shape[0]
    k = np.arange(n_freq, dtype=np.float64)
    m_len = 2.0 * n_freq
    model = -2.0 * np.pi * k * dn / m_len
    # Wrap the difference, then unwrap: the residual is ~1e-3 rad, so this cannot
    # slip, while unwrapping two 4000-rad ramps separately can and does.
    resid = np.unwrap(np.angle(m * np.conj(np.exp(1j * model))))
    f = k * (sample_rate / m_len)
    w = np.abs(m)
    design = np.vstack([f, np.ones_like(f)]).T
    coef, *_ = np.linalg.lstsq(design * w[:, None], resid * w, rcond=None)
    detrended = resid - design @ coef
    tau = dn / sample_rate * 1e9 - coef[0] / (2 * np.pi) * 1e9
    predicted = (np.sqrt(abs(dn) / m_len) / np.sqrt(rows * n_freq / n_freq * 1024)
                 if dn else 0.0)
    return tau, dn / sample_rate * 1e9, detrended.std(), predicted, coef[1], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory holding one .bin per baseline")
    ap.add_argument("--n-station", type=int, default=24)
    ap.add_argument("--stride", type=int, default=0,
                    help="--delay-stride passed to corr-replay")
    ap.add_argument("--delays", default=None,
                    help="explicit per-station delays in samples, comma separated")
    ap.add_argument("--n-freq", type=int, default=16384)
    ap.add_argument("--frames-per-row", type=int, default=1024)
    ap.add_argument("--sample-rate", type=float, default=81920000.0)
    ap.add_argument("--tol-ps", type=float, default=5.0)
    ap.add_argument("--quiet", action="store_true", help="summary only")
    ap.add_argument("--dump-csv", default=None,
                    help="write per-baseline numbers here, for plotting")
    args = ap.parse_args()

    stems = [os.path.basename(x) for x in glob.glob(os.path.join(args.dir, "*.bin"))
             if not x.endswith("-time.bin")]
    names = infer_names(stems)
    if not names:
        names = station_names(args.n_station)
    args.n_station = len(names)
    print("stations inferred from file names: %s" % ", ".join(names))
    delays = ([int(x) for x in args.delays.split(",")] if args.delays
              else [i * args.stride for i in range(args.n_station)])

    files = {}
    for path in glob.glob(os.path.join(args.dir, "*.bin")):
        base = os.path.basename(path)
        if base.endswith("-time.bin"):
            continue
        pair = parse_pair(base, names)
        if pair:
            files[pair] = path
    if not files:
        print("no baseline files found in %s" % args.dir)
        return 2

    m_len = 2 * args.n_freq
    print("injected delays (samples): %s" % delays)
    print("baselines: %d\n" % len(files))
    if not args.quiet:
        print("%-9s %7s %9s %15s %15s %10s %13s %13s %8s"
              % ("baseline", "delta N", "frames", "expected ns", "measured ns",
                 "error ps", "residual rad", "predicted", "ratio"))
    worst_err, worst_ratio, null_worst, worst_off = 0.0, 0.0, 0.0, 0.0
    rows = None
    dumped = []
    for (j, k) in sorted(files):
        if j == k:
            continue
        dn = delays[j] - delays[k]
        tau, tau_exp, resid, predicted, offset, rows = measure(
            files[(j, k)], dn, args.n_freq, args.sample_rate)
        err_ps = (tau - tau_exp) * 1000.0
        ratio = resid / predicted if predicted else float("nan")
        if not args.quiet:
            print("%-9s %7d %9d %15.4f %15.4f %10.3f %13.3e %13s %8s"
                  % ("%s%s" % (names[j], names[k]), dn, rows, tau_exp, tau, err_ps,
                     resid, ("%.3e" % predicted) if predicted else "0 (exact)",
                     ("%.3f" % ratio) if predicted else "-"))
        dumped.append((j, k, dn, tau_exp, tau, err_ps, resid, predicted))
        worst_err = max(worst_err, abs(err_ps))
        worst_off = max(worst_off, abs(offset))
        if predicted:
            worst_ratio = max(worst_ratio, ratio)
        else:
            null_worst = max(null_worst, resid)

    worst_im = 0.0
    for i in range(args.n_station):
        path = files.get((i, i))
        if not path:
            continue
        z = np.fromfile(path, dtype=np.complex64).reshape(-1, args.n_freq)
        scale = np.abs(z.real).max()
        if scale > 0:
            worst_im = max(worst_im, np.abs(z.imag).max() / scale)

    print("\nautocorrelation |Im|/|Re| worst : %.2e   (arithmetic noise only)" % worst_im)
    if null_worst:
        print("zero-delay baselines residual  : %.2e rad" % null_worst)
    print("worst |delay error|            : %.4f ps   (tolerance %.1f ps)"
          % (worst_err, args.tol_ps))
    print("worst |phase offset| at k=0    : %.2e rad" % worst_off)
    if np.isfinite(worst_ratio):
        print("worst residual / prediction    : %.3f      (1.00 = the test signal's "
              "window overlap explains it)" % worst_ratio)

    # Closure: every baseline delay must be the difference of per-station delays.
    # This is what a per-baseline check cannot see.
    obs, model = [], []
    for (j, k) in sorted(files):
        if j == k:
            continue
        tau, _, _, _, _, _ = measure(files[(j, k)], delays[j] - delays[k],
                                     args.n_freq, args.sample_rate)
        obs.append(tau)
        row = np.zeros(args.n_station)
        row[j], row[k] = 1.0, -1.0
        model.append(row)
    obs, model = np.array(obs), np.array(model)
    fit, *_ = np.linalg.lstsq(model, obs, rcond=None)
    closure = obs - model @ fit
    print("\nclosure over %d baselines: worst residual %.4f ps"
          % (len(obs), np.abs(closure).max() * 1000))
    # Only differences are observable, so compare both series after removing their
    # own means: the absolute level is degenerate by construction.
    rel_fit = fit - fit.mean()
    rel_inj = (np.array(delays, dtype=float) - np.mean(delays)) / args.sample_rate * 1e9
    print("  per-station delays, fitted - injected (ns): %s"
          % " ".join("%+.4f" % (a - b) for a, b in zip(rel_fit, rel_inj)))

    if args.dump_csv:
        import csv
        with open(args.dump_csv, "w", newline="") as fh:
            wtr = csv.writer(fh)
            wtr.writerow(["j", "k", "dn_samples", "tau_expected_ns", "tau_measured_ns",
                          "error_ps", "residual_rad", "predicted_rad", "frames"])
            for j, k, dn, te, tm, er, rs, pr in dumped:
                wtr.writerow([j, k, dn, "%.6f" % te, "%.6f" % tm, "%.6f" % er,
                              "%.6e" % rs, "%.6e" % pr, rows])
        print("  wrote %s" % args.dump_csv)

    ok = (worst_err <= args.tol_ps
          and np.abs(closure).max() * 1000 <= args.tol_ps
          and worst_im < 1e-5
          and (not np.isfinite(worst_ratio) or worst_ratio < 1.15))
    print("\nVERDICT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
