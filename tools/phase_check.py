#!/usr/bin/env python3
"""Validate the correlator's phase-vs-frequency response against theory.

The test signal is one pseudorandom int8 stream. Station 0 carries it as-is;
station 1 carries the same stream delayed by `tau` samples. Then

    x_1(t) = x_0(t - tau)
    Y_1(f) = Y_0(f) * exp(-i 2*pi*f*tau)
    C(f)   = Y_0(f) * conj(Y_1(f)) = |Y_0(f)|^2 * exp(+i 2*pi*f*tau)

so the phase of the cross-power spectrum must be a straight line in frequency
with slope 2*pi*tau, i.e. `phase(bin c) = 2*pi*c*tau/N` with N the FFT length.

This one figure therefore exercises, end to end: the packet-to-frame alignment,
the conjugation (a flipped conj would give -2*pi*c*tau/N), the R2C bin order
(a reversed bin axis would negate the slope), the Nyquist drop, and the writer's
complex64 layout.

usage: phase_check.py OUT.png [RUN_DIR ...]
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Prefer a CJK font so the labels are readable to the people who will look at
# this; fall back silently rather than failing on a machine without one.
for cand in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

N_FFT = 8192          # 2 * n_freq
N_FREQ = 4096         # retained bins


def load_baseline(run_dir, index, j, k):
    name = f"baseline_{index:04d}_j{j:02d}_k{k:02d}.bin"
    path = os.path.join(run_dir, name)
    a = np.fromfile(path, dtype=np.complex64)
    if a.size % N_FREQ:
        raise SystemExit(f"{path}: {a.size} complex values is not a whole number of rows")
    return a.reshape(-1, N_FREQ)


def read_sample_rate(run_dir):
    with open(os.path.join(run_dir, "metadata.yaml")) as f:
        for line in f:
            if line.startswith("sample_rate_hz:"):
                return float(line.split(":", 1)[1])
    return 0.0


def fit_slope(phase, bins):
    """Least-squares slope of phase against bin, unwrapped."""
    unwrapped = np.unwrap(phase)
    A = np.vstack([bins, np.ones_like(bins)]).T
    slope, intercept = np.linalg.lstsq(A, unwrapped, rcond=None)[0]
    return slope, intercept, unwrapped


def main():
    out = sys.argv[1]
    runs = sys.argv[2:] or ["/tmp/corr-phase/delay0", "/tmp/corr-phase/delay1"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    ax_phase, ax_mag, ax_res = axes

    bins = np.arange(N_FREQ, dtype=float)
    summary = []

    palette = ("tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange")
    for run_dir, colour in zip(runs, palette):
        # The delay is taken from the directory name so that adding a case means
        # adding a run, not editing this script.
        base = os.path.basename(run_dir.rstrip("/"))
        digits = "".join(ch for ch in base if ch.isdigit())
        tau = int(digits) if digits else 0
        x = load_baseline(run_dir, 1, 0, 1)   # the (0,1) cross term
        fs = read_sample_rate(run_dir)
        rows = x.shape[0]

        # Averaging the complex cross spectra is the integration: the phase is the
        # same on every row, so the mean has the same phase and sqrt(rows) less
        # noise. Averaging the *phases* instead would be wrong wherever the phase
        # wraps.
        mean = x.mean(axis=0)
        phase = np.angle(mean)
        mag = np.abs(mean) / rows

        freq_mhz = bins * fs / N_FFT / 1e6
        slope, intercept, unwrapped = fit_slope(phase, bins)
        tau_fit = slope * N_FFT / (2 * np.pi)
        # Theory: phase(c) = 2*pi*c*tau/N, wrapped into (-pi, pi].
        theory = np.angle(np.exp(1j * 2 * np.pi * bins * tau / N_FFT))
        resid = np.angle(np.exp(1j * (phase - theory)))   # wrapped difference
        summary.append((run_dir, tau, rows, slope, tau_fit, np.abs(resid).max(),
                        np.abs(resid).mean()))

        ax_phase.plot(freq_mhz, phase, ".", ms=2, color=colour, alpha=0.6,
                      label=f"τ = {tau} 样本（实测 {rows} 行平均）")
        ax_phase.plot(freq_mhz, theory, "-", lw=1.2, color=colour, alpha=0.75,
                      label=f"τ = {tau} 理论值 2πc·τ/N")
        ax_mag.semilogy(freq_mhz, np.maximum(mag, mag.max() * 1e-12), "-", lw=0.8,
                        color=colour, label=f"τ = {tau}")
        ax_res.plot(freq_mhz, resid, ".", ms=2, color=colour, alpha=0.6,
                    label=f"τ = {tau}（最大残差 {np.abs(resid).max():.3f} rad）")

    ax_phase.set_xlabel("频率 (MHz)   [bin c ↔ c × %.4f MHz]" % (read_sample_rate(runs[0]) / N_FFT / 1e6))
    ax_phase.set_ylabel("互相关相位 (rad)")
    ax_phase.set_title("互相关相位 vs 频率：直线即延迟，斜率 = 2πτ/N（τ≥2 时会绕圈）")
    ax_phase.legend(loc="upper left", fontsize=9)
    ax_phase.grid(alpha=0.3)

    ax_mag.set_xlabel("频率 (MHz)")
    ax_mag.set_ylabel("|C| (每行平均)")
    ax_mag.set_title("互相关幅度：伪随机白噪声应平坦，无周期结构")
    ax_mag.legend(fontsize=9)
    ax_mag.grid(alpha=0.3)

    ax_res.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax_res.set_xlabel("频率 (MHz)")
    ax_res.set_ylabel("实测 − 理论 (rad)")
    ax_res.set_title("残差：应落在数值噪声内")
    ax_res.legend(fontsize=9)
    ax_res.grid(alpha=0.3)

    fig.suptitle("相关机功能验证：伪随机时域数据，站 1 相对站 0 延迟 τ 个样本",
                 fontsize=13)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}\n")
    print(f"{'运行':<28} {'τ':>3} {'行数':>6} {'拟合斜率':>12} {'拟合 τ(样本)':>13} {'最大残差':>10} {'平均残差':>10}")
    for d, tau, rows, slope, tf, mx, mn in summary:
        print(f"{d:<28} {tau:>3} {rows:>6} {slope:>12.6f} {tf:>13.4f} {mx:>10.4f} {mn:>10.4f}")


if __name__ == "__main__":
    main()
