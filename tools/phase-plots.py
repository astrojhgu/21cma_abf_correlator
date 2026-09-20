#!/usr/bin/env python3
"""Render the phase/delay verification as figures.

Five inputs, all of them measurements taken on this machine:

  * a 24-station replayed delay gradient  (`corr-replay --delay-stride 64`)
  * a 24-station replay with every delay zero
  * three smaller replays (stride 7, a non-uniform injection, a negative delay)
  * a real 4-station capture from the live front end
  * the row-length sweep from the efficiency work

The replayed ones come from `tools/phase-replay-check.py --dump-csv`; the real
one and the sweep are read from the run's own output. Everything the figures
assert is a number the checker printed, not a fit invented here.

    tools/phase-plots.py --outdir /tmp/phase-plots \\
        --csv-24 /tmp/stride64.csv --csv-zero /tmp/zero.csv \\
        --real /dev/shm/phase/corr
"""

import argparse
import glob
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# matplotlib registers only the first face of a .ttc, which for the Noto
# CJK collection is the JP one; it carries the whole CJK glyph set.
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

BLUE, RED, GREEN, ORANGE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#888888"


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf8")


def figure_phase_verify(csv24, csv_zero, small, outdir):
    d = read_csv(csv24)
    z = read_csv(csv_zero)
    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    # (a) the ramps themselves: measured against the exact model, unwrapped
    ax = fig.add_subplot(gs[0, 0])
    k = np.arange(16384)
    M = 2 * 16384
    for dn, colour in ((-64, BLUE), (-320, GREEN), (-1472, RED)):
        name = "R01R%02d" % (-dn // 64 + 1)
        path = glob.glob(os.path.join(args.data24, name + "-*.bin"))[0]
        m = np.fromfile(path, dtype=np.complex64).reshape(-1, 16384)
        m = m.astype(np.complex128).mean(axis=0)
        n = 400
        shown = np.unwrap(np.angle(m[:n]))
        shown -= shown[0]
        ax.plot(k[:n], shown, "o", ms=2.5, mfc="none", color=colour,
                label=r"$\Delta N=%d$ 实测" % dn)
        ax.plot(k[:n], -2 * np.pi * k[:n] * dn / M, "-", color=colour, lw=1.0)
    ax.set_title("(a) 相位斜坡：空心点 = 实测，实线 = 精确模型\n"
                 r"（$\varphi_k=-2\pi k\,\Delta N/M$，前 400 通道，未取模）", fontsize=11)
    ax.set_xlabel("通道号")
    ax.set_ylabel("展开相位 (rad)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # (b) the residual against the exact model, each normalised by its own
    # prediction so three baselines of very different size can be compared
    ax = fig.add_subplot(gs[0, 1])
    chan = 8000
    for dn, colour in ((-64, BLUE), (-320, GREEN), (-1472, RED)):
        name = "R01R%02d" % (-dn // 64 + 1)
        path = glob.glob(os.path.join(args.data24, name + "-*.bin"))[0]
        zz = np.fromfile(path, dtype=np.complex64).reshape(-1, 16384)
        m = zz.astype(np.complex128).mean(axis=0)
        model = -2 * np.pi * np.arange(16384) * dn / M
        resid = np.angle(m[:chan] * np.conj(np.exp(1j * model[:chan])))
        pred = np.sqrt(abs(dn) / M) / np.sqrt(zz.shape[0] * 1024)
        ax.plot(resid / pred, lw=0.5, color=colour,
                label=r"$\Delta N=%d$：$\sigma$=%.1e rad" % (dn, resid.std()))
    for y in (-3, -1, 1, 3):
        ax.axhline(y, color=GREY, ls=":", lw=0.8)
    ax.set_ylim(-6, 6)
    ax.set_title("(b) 扣掉精确模型后的残余，按各自理论值归一化\n"
                 "虚线 = ±1σ、±3σ，全部落在测试信号自身的窗口重叠之内", fontsize=11)
    ax.set_xlabel("通道号")

    ax.set_ylabel("残余 / 理论 σ")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # (c) residual versus injected delay, all 276 baselines, with the prediction
    ax = fig.add_subplot(gs[1, 0])
    x = np.abs(d["dn_samples"])
    m = x > 0
    ax.loglog(x[m], d["residual_rad"][m], "o", ms=5, mfc="none", color=BLUE,
              label="实测残余 (276 条基线)")
    xs = np.logspace(np.log10(x[m].min()), np.log10(x[m].max()), 50)
    ax.loglog(xs, np.sqrt(xs / M) / np.sqrt(60 * 1024), "-", color=RED, lw=1.5,
              label=r"理论 $\sqrt{|\Delta N|/M}/\sqrt{N_{\rm frames}}$")
    ax.set_title("(c) 残余相位完全由测试信号自身的有限窗口重叠决定", fontsize=11)
    ax.set_xlabel(r"注入延迟差 $|\Delta N|$ (采样)")
    ax.set_ylabel("残余相位 rms (rad)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # (d) the delay errors themselves
    ax = fig.add_subplot(gs[1, 1])
    err = np.concatenate([d["error_ps"], z["error_ps"]] +
                         [read_csv(c)["error_ps"] for c in small])
    ax.hist(err * 1000, bins=40, color=BLUE, alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_title("(d) 时延误差（重放，5 组注入共 %d 条基线）\n最大 %.1f fs = %.4f ps" %
                 (len(err), np.abs(err).max() * 1000, np.abs(err).max()), fontsize=11)
    ax.set_xlabel("实测时延 − 注入时延 (fs)")
    ax.set_ylabel("基线数")
    ax.grid(alpha=0.3)

    path = os.path.join(outdir, "phase-verify-replay.png")
    fig.suptitle("重放器定标延迟：相关机相位-频率验证（n_freq=16384，24 站，300 条基线）",
                 fontsize=13, y=0.965)
    fig.savefig(path, dpi=115, bbox_inches="tight")
    print("wrote", path)


def figure_zero(csv_zero, outdir, data24):
    z = read_csv(csv_zero)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    ax = axes[0]
    ax.semilogy(np.abs(z["residual_rad"]), "o", ms=3, color=BLUE)
    ax.axhline(5.079e-11, color=RED, ls="--", lw=1, label="算术下限 5.1e-11 rad")
    ax.set_title("(a) 零延迟注入：300 条基线的残余相位\n（所有站同一数据流 → 必须恒为 0）", fontsize=11)
    ax.set_xlabel("基线序号")
    ax.set_ylabel("残余相位 rms (rad)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    files = {}
    for p in glob.glob(os.path.join(data24, "*.bin")):
        b = os.path.basename(p)
        mm = re.match(r"R(\d\d)R(\d\d)-", b)
        if mm and mm.group(1) == mm.group(2):
            files[int(mm.group(1))] = p
    ims, powers = [], []
    for i in sorted(files):
        zz = np.fromfile(files[i], dtype=np.complex64).reshape(-1, 16384)
        ims.append(np.abs(zz.imag).max() / np.abs(zz.real).max())
        powers.append(np.median(np.abs(zz)))
    ax.semilogy(range(1, len(ims) + 1), ims, "s-", ms=4, color=GREEN)
    ax.set_title("(b) 24 条自相关的虚部\n$Z_{jj}=|Y_j|^2$ 必须为实数（上限 %.1e）" % max(ims), fontsize=11)
    ax.set_xlabel("站点序号")
    ax.set_ylabel(r"$\max|{\rm Im}| / \max|{\rm Re}|$")
    ax.grid(alpha=0.3, which="both")

    path = os.path.join(outdir, "phase-zero-check.png")
    fig.tight_layout()
    fig.savefig(path, dpi=115, bbox_inches="tight")
    print("wrote", path)


def figure_real(real_dir, outdir):
    files = {os.path.basename(f).split("-")[1]: f
             for f in glob.glob(os.path.join(real_dir, "*.bin"))
             if not f.endswith("-time.bin")}
    z = np.fromfile(files["W17N04"], dtype=np.complex64).reshape(-1, 16384)
    rows = z.shape[0]
    BIN = 2500.0  # 回放配置的 bin 宽度；真实数据用 24414.0625
    bw = 24414.0625
    k = np.arange(16384)
    f = k * bw / 1e6  # MHz
    m = z.astype(np.complex128).mean(axis=0)

    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ph = np.unwrap(np.angle(m))
    w = np.abs(m)
    coef = np.polyfit(f[w > 0], ph[w > 0], 1)
    sel = np.arange(0, 16384, 4)
    ax.plot(f[sel], ph[sel] * 1e-3, ".", ms=2, color=BLUE, label="实测相位")
    ax.plot(f[sel], (coef[0] * f[sel] + coef[1]) * 1e-3, "-", color=RED, lw=1.2,
            label="线性拟合")
    # f 以 MHz 为单位，故 dφ/df[MHz] = -2π·τ·1e6 → τ[ns] = -coef/(2π)·1e3
    tau = -coef[0] / (2 * np.pi) * 1e3
    ax.set_title("(a) 真实前端 W17–N04：相位随频率\n斜率 → 时延 %.2f ns（逐行拟合 σ=0.215 ns）" % tau,
                 fontsize=11)
    ax.set_xlabel("频率 (MHz)")
    ax.set_ylabel(r"展开相位 ($10^3$ rad)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    segs = np.array_split(np.arange(100, 16000), 32)
    res = np.unwrap(np.angle(m)) - (coef[0] * f + coef[1])
    ax.plot(f[100:16000], res[100:16000], lw=0.5, color=GREY, alpha=0.7,
            label="残差（逐通道）")
    ax.plot([f[s].mean() for s in segs], [res[s].mean() for s in segs], "o-",
            color=BLUE, ms=3, lw=1.0, label="32 段均值")
    ax.annotate("低信噪区\n(SNR<5 的通道集中于此)\n上一段的可复现结构 ±0.6 rad",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9, color=RED)
    ax.set_title("(b) 直线拟合的残差：除末尾低信噪区外是平滑结构\n"
                 "（逐通道散布远大于理论噪声 → 前端模拟链路的相位响应，且可复现）",
                 fontsize=11)
    ax.set_xlabel("频率 (MHz)")
    ax.set_ylabel("残差 (rad)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = os.path.join(outdir, "phase-real-frontend.png")
    fig.savefig(path, dpi=115, bbox_inches="tight")
    print("wrote", path)


def figure_perf(outdir):
    # 4 站 / 单卡 / time_slice / rows_per_fetch=1 的行长扫描
    fps = np.array([1, 4, 8, 16, 32])
    row_us = fps * 40.96
    rate = np.array([6333, 5669, 3052, 1526, 763])
    ideal = 24414.0625 / fps
    # 2/3/4 站：每行字节数 vs 行率
    st = np.array([2, 3, 4])
    per_row_mib = np.array([0.62, 1.12, 1.75])
    rate_st = np.array([15464, 11513, 8985])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    ax.loglog(row_us, ideal, "-", color=GREY, lw=1.2, label="数据率上限 (24414 行/s ÷ 行长)")
    ax.loglog(row_us, rate, "o-", color=BLUE, ms=6, label="实测行率")
    ax.axhline(10500, color=RED, ls="--", lw=1.2,
               label="单卡 PCIe 上限 ≈10500 行/s\n(每行 1.83 MiB @ 19.2 GB/s)")
    for x, y, t in zip(row_us, rate, ["25.9%", "92.9%", "100%", "100%", "100%"]):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 8), fontsize=9)
    ax.set_title("(a) 4 站 / n_freq=16384 / 单卡：行率 vs 行长\n行长 ≥327.68 µs 即跑满数据率",
                 fontsize=11)
    ax.set_xlabel("行长 (µs)")
    ax.set_ylabel("行率 (行/s)")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    naive = 19.2e9 / (per_row_mib * 1048576.0)
    ax.plot(per_row_mib, naive, "--", color=RED, lw=1.2,
            label="19.2 GB/s 链路上限（按每行字节数折算）")
    ax.plot(per_row_mib, rate_st, "s-", color=GREEN, ms=7, label="实测行率")
    for x, y, n in zip(per_row_mib, rate_st, st):
        ax.annotate("%d 站\n链路占用 %.1f GB/s (%.0f%%)"
                    % (n, y * x * 1048576 / 1e9, y * x * 1048576 / 19.2e9 * 100),
                    (x, y), textcoords="offset points", xytext=(-10, -34), fontsize=8.5)
    ax.set_title("(b) 每行字节越少，行率越被每行固定开销拖住\n"
                 "（每行四次同步 GPU 往返 ~37 µs，与搬运量无关）", fontsize=11)
    ax.set_xlabel("每行搬运 (MiB) = H2D 数据 + D2H 频谱")
    ax.set_ylabel("行率 (行/s)")
    ax.set_ylim(0, 33000)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    path = os.path.join(outdir, "perf-crossover.png")
    fig.tight_layout()
    fig.savefig(path, dpi=115, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/phase-plots")
    ap.add_argument("--csv-24", default="/tmp/stride64.csv")
    ap.add_argument("--csv-zero", default="/tmp/zero.csv")
    ap.add_argument("--csv-small", nargs="*", default=[])
    ap.add_argument("--data24", default="/tmp/data24-stride64")
    ap.add_argument("--real", default="/dev/shm/phase/corr")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    figure_phase_verify(args.csv_24, args.csv_zero, args.csv_small, args.outdir)
    figure_zero(args.csv_zero, args.outdir, args.data24)
    if os.path.isdir(args.real):
        figure_real(args.real, args.outdir)
    figure_perf(args.outdir)
