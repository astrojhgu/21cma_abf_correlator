#!/usr/bin/env python3
"""Compare every baseline's phase-versus-frequency with the injected delays, and
draw the upper triangle.

The test this is built for: drive the correlator with `corr-replay --delay-stride 1`,
so station `i`'s sample stream is delayed by exactly `i` samples relative to a
common broadband source. A delay of `d` samples multiplies the spectrum by
`exp(-2*pi*i*k*d/nch)`, so the cross-power `Y_j * conj(Y_k)` has a phase that is
a *straight line* in channel `k`, with slope `-2*pi*(d_j - d_k)/fft_len`. Nothing
about the instrument can imitate that: the packet-to-frame alignment, the FFT bin
order, the conjugation and the multi-card partial-sum all break it, and they break
it differently for different baselines.

Two estimates per baseline, because they fail differently:

  * the **slope** of unwrapped phase against channel, weighted by |S| -- precise,
    but it assumes the unwrap is right;
  * the **lag** of the inverse transform -- one number, no unwrapping, and it lands
    on an exact integer when the delay is an exact integer. Trust it when the two
    disagree.

Panels are drawn in one of two ways, and both have been wrong once already:

  * `--mosaic phase` (data *and* model): the measured phase (blue) and the delay's
    straight line (red). The channel window must be narrow enough that the steepest
    delay does not wrap -- at 8192 channels a 23-sample delay spans 72 rad and wraps
    eleven times, which at 80 px per panel is indistinguishable from two diverging
    lines. Both curves are referenced to their own value at the first strong
    channel, so what the eye compares is the *slope*.
  * `--mosaic residual` (default): measured minus model on a shared scale, flat at
    zero when the phase is right, and its shape says *where* it is wrong when it is
    not.

The y-range is measured from the data in a pre-pass, never estimated: an earlier
version derived it from the model alone and drew the measured curve outside the
axes.

    tools/phase_triangle.py /scratch/phase24/<run dir> --mosaic phase
    tools/phase_triangle.py /scratch/phase24/<run dir> --zoom 0,23
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

import matplotlib

from corr_common import delay_from_slope, delay_per_channel_rad, fft_len, wrap_deg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def baselines(run_dir: Path) -> dict[tuple[int, int], Path]:
    out = {}
    for p in sorted(run_dir.glob("baseline_*_j*_k*.bin")):
        m = re.search(r"_j(\d+)_k(\d+)\.bin$", p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return out


def spectra(path: Path, nch: int) -> np.ndarray:
    """Coherent average over rows: complex64[ntime][nch] -> [nch]."""
    raw = np.fromfile(path, dtype="<c8")
    if raw.size % nch:
        raw = raw[: raw.size - raw.size % nch]
    return raw.reshape(-1, nch).mean(axis=0)


def estimate(s: np.ndarray) -> tuple[float, int, float]:
    """(delay from the slope, lag from the inverse transform, fit residual rms).

    The channel axis is the retained half of an R2C transform, so the FFT length is
    `2 * nch`, not `nch`: a delay of `d` samples multiplies bin `k` by
    `exp(-2*pi*i*k*d/nch)`. Using `nch` reports exactly half the delay, which
    looks like a plausible number -- worth stating here rather than buried.
    """
    n = s.size
    n_fft = fft_len(n)
    mag = np.abs(s)
    strong = mag > 0.1 * np.median(mag[mag > 0]) if np.any(mag > 0) else np.zeros(n, bool)
    k = np.arange(n)
    slope = float("nan")
    resid = float("nan")
    if strong.sum() > 16:
        ph = np.unwrap(np.angle(s[strong]))
        kk = k[strong]
        w = mag[strong]
        fit = np.polyfit(kk, ph, 1, w=w)
        slope = float(fit[0])
        resid = float(np.sqrt(np.average((ph - np.polyval(fit, kk)) ** 2, weights=w)))
    # Rebuild the full Hermitian spectrum (R2C drops the Nyquist bin) so the
    # correlation peak lands on whole samples of the real sample rate.
    full = np.zeros(n_fft, dtype=np.complex128)
    full[:n] = s
    full[n + 1 :] = np.conj(s[1:][::-1])
    ac = np.fft.ifft(full)
    lag = int(np.argmax(np.abs(ac)))
    if lag > n_fft // 2:
        lag -= n_fft
    delay = delay_from_slope(slope, n) if np.isfinite(slope) else float("nan")
    return delay, lag, resid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stride", type=int, default=1, help="delay of station i is i*stride samples")
    ap.add_argument("--sign", type=int, default=0, help="+1/-1 to force the expected sign, 0 to infer")
    ap.add_argument("--out", default="/tmp/phase_triangle.png")
    ap.add_argument(
        "--mosaic",
        choices=("residual", "phase"),
        default="residual",
        help="each panel shows the residual (flat at zero) or the phase and its model line",
    )
    ap.add_argument(
        "--phase-window",
        type=int,
        default=1536,
        help="channels per panel for --mosaic phase; the default shows about two "
        "wraps for the steepest injected delay, so the sawtooth stays legible",
    )
    ap.add_argument("--max-ch", type=int, default=1400, help="channels drawn per residual panel")
    ap.add_argument("--zoom", default=None, help="one baseline large, as J,K")
    a = ap.parse_args()

    d = Path(a.run_dir)
    if not d.is_dir():
        d = d.parent
    bl = baselines(d)
    if not bl:
        print(f"no baseline_*_j*_k*.bin in {d}", file=sys.stderr)
        return 2
    nst = max(max(j, k) for j, k in bl) + 1
    nch = 8192
    n_fft = fft_len(nch)
    win = min(a.phase_window, nch)
    print(f"{d.name}: {len(bl)} baselines, {nst} stations, {nch} channels")

    spec = {jk: spectra(p, nch) for jk, p in bl.items()}

    # Infer the sign from the data rather than assuming it: the stored cross-power is
    # Y_j*conj(Y_k), and whether the lag is d_j - d_k or its negative is exactly what
    # a test should discover.
    if a.sign:
        sign = a.sign
    else:
        num = den = 0.0
        for (j, k), s in spec.items():
            _, lag, _ = estimate(s)
            want = a.stride * (j - k)
            if want:
                num += lag * want
                den += want * want
        sign = 1 if num >= 0 else -1
        print(f"inferred sign: delay(j,k) = {sign:+d} * (d_j - d_k)")

    panels: dict[tuple[int, int], tuple] = {}
    errs: list[int] = []
    slope_errs: list[float] = []
    overlay_err: list[float] = []
    resid_fit: list[float] = []
    for (j, k), s in spec.items():
        if j >= k:
            continue
        want = sign * a.stride * (j - k)
        delay_fit, lag, one_resid = estimate(s)
        errs.append(abs(lag - want))
        if np.isfinite(delay_fit):
            slope_errs.append(abs(delay_fit - want))
        if np.isfinite(one_resid):
            resid_fit.append(one_resid)

        x = np.arange(nch)
        mag = np.abs(s)
        strong = mag > 0.1 * np.median(mag[mag > 0])
        ph = np.unwrap(np.angle(s))
        pred = delay_per_channel_rad(want, nch) * x
        k0 = int(np.argmax(strong))
        pred += ph[k0] - pred[k0]
        overlay_err.append(float(np.max(np.abs(ph[strong] - pred[strong]))))

        # Windowed copies for the phase mosaic: *wrapped* phase in degrees, each
        # referenced to its own first strong channel (the absolute phase offset of a
        # cross-power is not what this test is about -- the slope is). Wrapping is
        # what allows the fixed +/-180 degree axis the whole figure shares, and it
        # turns the model into a sawtooth of the same slope, so what the eye checks
        # is that the two sawtooths line up.
        xs = x[:win]
        su = mag[:win]
        strongw = su > 0.1 * np.median(su[su > 0]) if np.any(su > 0) else np.zeros(win, bool)
        kw = int(np.argmax(strongw))
        phw_deg = wrap_deg(np.degrees(np.angle(s[:win])) - np.degrees(np.angle(s[kw])))
        predw_deg = wrap_deg(np.degrees(pred[:win]) - np.degrees(pred[kw]))
        panels[(j, k)] = (xs, phw_deg, predw_deg, strongw, ph, pred, strong, lag, want)

    # One fixed range for every panel, in degrees, as asked: +/-180. Every panel is
    # therefore directly comparable, and nothing can be drawn outside the axes --
    # the two earlier versions of this figure managed that in both possible ways
    # (an unwrapped curve wrapping eleven times, then a data-derived range that was
    # computed from the model and clipped the measurement).
    phase_ylim = (-180.0, 180.0)

    if a.zoom:
        j, k = (int(v) for v in a.zoom.split(","))
        _, _, _, _, ph, pred, strong, lag, want = panels[(j, k)]
        dev = float(np.max(np.abs(ph[strong] - pred[strong])))
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=130)
        ch = np.arange(nch)
        axs[0].set_ylim(-180, 180)
        axs[0].plot(ch[::4], wrap_deg(np.degrees(ph))[::4], lw=0.9, color="#1f4e79",
                    label="measured (wrapped)")
        axs[0].plot(ch[::4], wrap_deg(np.degrees(pred))[::4], lw=1.6, color="#c00000",
                    alpha=0.55, label="injected-delay line (wrapped)")
        axs[0].set_title(
            f"baseline j={j} k={k}: unwrapped phase vs channel, max deviation "
            f"{dev:.5f} rad (Delta-d={want} samples, {int(abs(want) / 2)} wraps over {nch} ch)",
            fontsize=10)
        axs[0].set_xlabel("channel")
        axs[0].set_ylabel("phase (rad)")
        axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.25)
        axs[1].plot(ch[::4], np.degrees(ph - pred)[::4], lw=0.9, color="#0a7d28")
        axs[1].axhline(0.0, color="#c00000", lw=1.0, alpha=0.6)
        axs[1].set_title("measured minus predicted: flat at zero means they coincide", fontsize=10)
        axs[1].set_xlabel("channel")
        axs[1].set_ylabel("residual (degrees)")
        axs[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(a.out)
        print(f"wrote {a.out} (zoom j={j} k={k}, Delta-d={want}, deviation {dev:.6f} rad)")
        return 0 if dev < 0.01 else 1

    rows = sorted({j for j, _ in bl})
    cols = sorted({k for _, k in bl})
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(0.62 * len(cols), 0.62 * len(rows)), dpi=130)
    for j in rows:
        for k in cols:
            ax = axes[j][k]
            ax.set_xticks([])
            ax.set_yticks([])
            if (j, k) not in panels:
                ax.axis("off")
                continue
            _, phw_deg, predw_deg, strongw, ph, pred, strong, lag, want = panels[(j, k)]
            ok = lag == want
            if a.mosaic == "phase":
                ax.set_ylim(*phase_ylim)
                ax.plot(np.arange(win)[strongw], phw_deg[strongw], lw=0.6, color="#1f4e79")
                ax.plot(np.arange(win), predw_deg, lw=1.0, color="#c00000", alpha=0.6)
            else:
                ax.set_ylim(-3e-3, 3e-3)
                ax.axhline(0.0, color="#c00000", lw=0.5, alpha=0.8)
                step = max(1, nch // a.max_ch)
                ax.plot(np.arange(nch)[::step][strong[::step]],
                        (ph - pred)[::step][strong[::step]], lw=0.7, color="#0a7d28")
            ax.set_title(f"{j},{k}  d={lag:+d}", fontsize=4.6, pad=1.2,
                         color="#0a7d28" if ok else "#c00000")
            for sp in ax.spines.values():
                sp.set_color("#0a7d28" if ok else "#c00000")
                sp.set_linewidth(0.6)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if a.mosaic == "phase":
        what = (f"wrapped phase in degrees (blue) and the injected-delay line (red), channels "
                f"0-{win - 1} of {nch}, each referenced to its own first strong channel; "
                f"fixed axis -180..+180 degrees, so the two sawtooths must line up")
    else:
        what = ("measured phase minus the injected-delay line (green) against zero (red), "
                "shared scale")
    fig.suptitle(
        f"{stamp}  |  source {d.name}  |  corr-replay {nst} stations, sample delay = "
        f"{a.stride}*i, nch={nch}, {len(panels)} baselines\n{what}\n"
        f"lag matches the injected delay on {sum(1 for e in errs if e == 0)}/{len(errs)} "
        f"baselines, max |error| {max(errs)} sample(s), drawn overlay deviation "
        f"{np.max(overlay_err):.5f} rad",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(a.out)

    print(f"wrote {a.out}")
    print(f"  panels {len(panels)}; phase axis fixed at {phase_ylim[0]:.0f}..{phase_ylim[1]:.0f} deg")
    print(f"  integer lag == injected delay on {sum(1 for e in errs if e == 0)}/{len(errs)} baselines")
    print(f"  max |lag error| = {max(errs)} sample(s)")
    if slope_errs:
        print(f"  slope-derived delay: max |error| {np.max(slope_errs):.4f} sample(s)")
    print(f"  drawn overlay (measured vs injected-delay line): max deviation "
          f"{np.max(overlay_err):.5f} rad")
    if resid_fit:
        print(f"  phase residual after the linear fit: mean {np.mean(resid_fit):.4f} rad, "
              f"max {np.max(resid_fit):.4f} rad")
    return 0 if max(errs) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
