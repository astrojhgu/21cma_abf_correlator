#!/usr/bin/env python3
"""raw_spectrum.py -- FFT a raw capture and plot it, against the correlator if asked.

`tools/raw_capture.c` writes one station's packets verbatim (8-byte counter, 4096
int8 samples). This turns that into the same quantity the correlator stores: the
mean periodogram. Frames are groups of `n_freq/2048` consecutive packets, exactly
as the frame assembler defines them, and the FFT is R2C of `2*n_freq` samples with
no window and no normalisation -- the correlator's own settings -- so the two
curves are directly comparable in shape *and* level.

Why this exists: when one station's spectrum looks wrong, the first question is
whether the raw stream already looks like that. The correlator has been through
frame assembly, an FFT and an average; if the raw periodogram reproduces the
correlator's curve, the anomaly is in the data and not in the processing.

Usage:
    tools/raw_spectrum.py CAPTURE.bin --n-freq 16384 --out raw.png \
        [--against RUN_DIR [--prefix P] [--station 21]]
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corr_common as cc

DATAGRAM = 4104
HEADER = 8
SAMPLES = 4096


def frames(path, n_freq, max_frames=0):
    """Yield (counter, int8 samples) for every complete frame in the capture.

    A frame is `n_freq/2048` consecutive counters, so the capture is placed by
    counter rather than by arrival order: a gap in the middle of a frame drops
    that frame instead of splicing two moments together, which would raise the
    noise floor in a way that looks like a measurement.
    """
    per_frame = n_freq // 2048
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % DATAGRAM:
        raw = raw[: raw.size - raw.size % DATAGRAM]
    recs = raw.reshape(-1, DATAGRAM)
    counters = recs[:, :HEADER].copy().view("<u8").reshape(-1)
    samples = recs[:, HEADER:]
    order = np.argsort(counters, kind="stable")
    counters, samples = counters[order], samples[order]

    first = int(counters[0])
    # Where does each packet belong? frame = counter // per_frame, slot = counter % per_frame.
    frame_id = (counters - first) // per_frame
    slot = (counters - first) % per_frame

    out_frames = []
    start = 0
    while start < len(frame_id):
        fid = frame_id[start]
        end = start
        while end < len(frame_id) and frame_id[end] == fid:
            end += 1
        slots_here = slot[start:end]
        if len(slots_here) == per_frame and len(set(slots_here.tolist())) == per_frame:
            buf = np.empty(n_freq * 2, dtype=np.int8)
            for k, s in zip(slots_here, samples[start:end]):
                buf[int(k) * SAMPLES : (int(k) + 1) * SAMPLES] = s.view(np.int8)
            out_frames.append(buf)
            if max_frames and len(out_frames) >= max_frames:
                break
        start = end
    return out_frames, per_frame


def periodogram(path, n_freq, max_frames=0, chunk=512):
    """Mean |FFT|^2 over whole frames. Same estimator as the correlator's row."""
    fr, per_frame = frames(path, n_freq, max_frames)
    if not fr:
        sys.exit(f"{path}: no complete frame of {per_frame} packets")
    n = 2 * n_freq
    acc = np.zeros(n_freq, dtype=np.float64)
    done = 0
    for i in range(0, len(fr), chunk):
        block = np.stack(fr[i : i + chunk]).astype(np.float32)
        y = np.fft.rfft(block, axis=1)[:, :n_freq]  # drop the Nyquist bin
        acc += (y.real.astype(np.float64) ** 2 + y.imag.astype(np.float64) ** 2).sum(axis=0)
        done += block.shape[0]
    return acc / done, done, per_frame


def correlator_auto(run_dir, prefix, station):
    """The same station's autocorrelation as the correlator stored it."""
    s = cc.pick_set(run_dir, prefix)
    nch = int(cc.metadata_field(s, "n_freq") or 0)
    for i, j, k, path in cc.set_baselines(run_dir, prefix=prefix):
        if j == k == station:
            raw = np.fromfile(path, dtype="<c8")
            return raw.reshape(-1, nch).mean(axis=0).real, nch
    return None, nch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+", help="raw capture file(s), optionally NAME=path")
    ap.add_argument("--n-freq", type=int, required=True, help="n_freq the frames are cut to")
    ap.add_argument("--max-frames", type=int, default=0, help="use only the first N frames")
    ap.add_argument("--against", default=None, help="run directory to overlay the correlator's own")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--station", type=int, default=None, help="station index for --against")
    ap.add_argument("--out", default="/tmp/raw_spectrum.png")
    ap.add_argument("--mhz", default="0,410")
    ap.add_argument("--skip-bins", type=int, default=2)
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_hz = 800e6 / (2 * a.n_freq)  # sample_rate_hz / fft_len
    freqs = np.arange(a.n_freq) * bin_hz / 1e6

    fig, ax = plt.subplots(figsize=(13, 6), dpi=120)
    lo, hi = (float(v) for v in a.mhz.split(","))
    curves = {}
    print(f"n_freq={a.n_freq}, fft_len={2*a.n_freq}, bin={bin_hz/1e3:.3f} kHz, "
          f"band 0..{freqs[-1]:.1f} MHz")
    for item in a.captures:
        label, _, path = item.partition("=")
        if not path:
            label, path = os.path.basename(item), item
        p, nframes, per_frame = periodogram(path, a.n_freq, a.max_frames)
        curves[label] = p
        pk = int(np.argmax(p[a.skip_bins:])) + a.skip_bins
        print(f"{label:22s} frames {nframes:6d} ({per_frame} pkts each)  "
              f"peak {freqs[pk]:8.3f} MHz  band med {np.median(p[int(50e6/bin_hz):int(150e6/bin_hz)]):.4g}  "
              f"DC {p[0]:.4g}")
    ref = max(float(np.max(v)) for v in curves.values())

    for label, p in curves.items():
        ax.plot(freqs[a.skip_bins :], 10 * np.log10(np.maximum(p, ref * 1e-12))[a.skip_bins :],
                lw=0.6, label=f"raw FFT: {label}")

    if a.against:
        auto, nch = correlator_auto(a.against, a.prefix, a.station if a.station is not None else 0)
        if auto is None:
            print(f"note: station {a.station} has no j==k baseline in {a.against}")
        else:
            if nch != a.n_freq:
                sys.exit(f"n_freq mismatch: capture {a.n_freq} vs run {nch}")
            ax.plot(freqs[a.skip_bins :],
                    10 * np.log10(np.maximum(auto, ref * 1e-12))[a.skip_bins :],
                    lw=0.6, ls="--", color="k",
                    label=f"correlator: station {a.station}")
            band = slice(int(50e6 / bin_hz), int(150e6 / bin_hz))
            rms = np.sqrt(np.mean((10 * np.log10(np.maximum(auto[band], 1e-12))
                                   - 10 * np.log10(np.maximum(list(curves.values())[0][band], 1e-12))) ** 2))
            print(f"correlator vs first capture, 50-150 MHz, RMS of the dB difference: {rms:.3f} dB")

    ax.set_xlim(lo, hi)
    ax.set_xlabel("MHz")
    ax.set_ylabel("power, dB (0 dB = strongest channel of any curve)")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=8)
    ax.set_title(f"raw FFT of the captured stream, {a.n_freq} channels "
                 f"({bin_hz/1e3:.1f} kHz per bin), mean periodogram over whole frames",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
