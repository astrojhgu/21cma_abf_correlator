#!/usr/bin/env python3
"""loss_attribution.py —— 补零的包到底丢在哪一层。

corr 的 `capacity:` 说的是**应用侧**先顶到谁（环、GPU 链路、CPU）。补零变多时它可能
仍然显示"有余量"，因为包可能丢在这台机器的更下面，甚至根本没到这台机器。这个脚本把
四层计数并排打出来，用增量判断责任：

    rx_prio0_discards 在涨  → 网卡 RX 队列（硬件）丢的：ring 太小、或中断被推迟
    softnet dropped 在涨    → 内核软中断 backlog 丢的（net.core.netdev_max_backlog）
    corr socket_drops 在涨  → 我们自己的 socket 缓冲丢的（SO_RCVBUF）
    以上都不涨、loss 涨      → 包在这台机器之前就没了（前端 / 线路 / 交换机）
    以上都不涨、补零涨       → 应用侧（staging 环 / GPU / 写盘）

为什么不用 `doctor.py`：它会 join 组播组做空闲丢包基线，**不能在别的 corr 正在跑的时候
用**（会和它抢包）。这个脚本只读 `/proc` 与 `ethtool -S`，跑测量的时候也能安全地看。

用法::

    tools/loss_attribution.py                     # 每 5 s 一行，Ctrl-C 停
    tools/loss_attribution.py --interval 1 --count 60
    tools/loss_attribution.py --nic ens2f1np1 --status-url http://127.0.0.1:8000/status.json
    tools/loss_attribution.py --count 1           # 只要一次快照
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

# 网卡上要跟的计数器（mlx5 的名字；别的驱动缺几个也不影响判读）。
NIC_KEYS = (
    # 缓冲类：ring 太小或中断被推迟时这些先动，而 drop 计数可能还是 0
    "rx_buff_alloc_err",
    "rx_prio0_buf_discard",
    "rx_prio0_cong_discard",
    "rx_buffer_passed_thres_phy",
    "rx_packets_phy",
    "rx_bytes_phy",
    "rx_prio0_discards",
    "rx_discards_phy",
    "rx_out_of_buffer",
    "rx_steer_missed_packets",
    "rx_pause_ctrl_phy",
    "tx_global_pause",
    "rx_crc_errors_phy",
    "rx_symbol_err_phy",
)


def nic_counters(nic):
    try:
        out = subprocess.run(["ethtool", "-S", nic], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"ethtool: {exc}"
    if out.returncode != 0:
        return {}, f"ethtool -S {nic} 失败：{out.stderr.strip()[:60]}"
    d = {}
    for line in out.stdout.splitlines():
        key, _, val = line.partition(":")
        key = key.strip()
        if key in NIC_KEYS:
            try:
                d[key] = int(val.strip())
            except ValueError:
                pass
    return d, None


def nic_ring_depth(nic):
    """RX ring 的当前与最大深度（描述符数）。

    ring 是网卡侧唯一的缓冲：一个包进来没被驱动取走就占一个描述符，占满了就丢在网卡里
    （`rx_out_of_buffer`），应用侧的 lost/sockdrop/udp drops 一个都看不到。所以"它多深、
    相当于多少毫秒的到达量"是判断还能撑多久的输入。改它需要 root（ethtool -G）。
    """
    try:
        out = subprocess.run(["ethtool", "-g", nic], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    current = maximum = None
    section = None
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith("Pre-set maximums"):
            section = "max"
        elif s.startswith("Current hardware settings"):
            section = "cur"
        elif s.startswith("RX:"):          # 不是 RX Mini / RX Jumbo
            try:
                value = int(s.split()[1])
            except (IndexError, ValueError):
                continue
            if section == "max" and maximum is None:
                maximum = value
            elif section == "cur" and current is None:
                current = value
    if current is None:
        return None
    return {"rx_current": current, "rx_max": maximum}


def softnet():
    """(总 dropped, 总 time_squeeze, 每个 CPU 的 time_squeeze)。"""
    dropped = squeeze = 0
    per_cpu = []
    try:
        rows = [l.split() for l in open("/proc/net/softnet_stat")]
    except OSError:
        return 0, 0, []
    for r in rows:
        if len(r) < 3:
            continue
        d, s = int(r[1], 16), int(r[2], 16)
        dropped += d
        squeeze += s
        per_cpu.append(s)
    return dropped, squeeze, per_cpu


def mlx5_irq_per_cpu():
    """每个 CPU 上 mlx5_comp* 的中断计数之和。"""
    totals = None
    try:
        fh = open("/proc/interrupts")
    except OSError:
        return []
    with fh:
        for line in fh:
            if "mlx5_comp" not in line:
                continue
            head = line.split("mlx5_comp", 1)[0]
            vals = [int(n) for n in head.split(":", 1)[-1].split() if n.isdigit()]
            if totals is None:
                totals = [0] * len(vals)
            for i, v in enumerate(vals[: len(totals)]):
                totals[i] += v
    return totals or []


def corr_status(url):
    try:
        req = urllib.request.Request(url, headers={"Cache-Control": "no-store"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        return d if d.get("ready") else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def sample(nic, url):
    n, err = nic_counters(nic)
    dropped, squeeze, per_cpu = softnet()
    irq = mlx5_irq_per_cpu()
    st = corr_status(url)
    return {
        "t": time.time(),
        "nic": n,
        "nic_err": err,
        "softnet_dropped": dropped,
        "softnet_squeeze": squeeze,
        "squeeze_per_cpu": per_cpu,
        "irq": irq,
        "lost": (st or {}).get("cumulative", {}).get("lost_packets"),
        "sockdrop": (st or {}).get("cumulative", {}).get("socket_drops"),
        "rows": (st or {}).get("cumulative", {}).get("rows"),
        "zf_win": (st or {}).get("window", {}).get("zero_fill_pct"),
        "bottleneck": (st or {}).get("capacity", {}).get("bottleneck"),
        "running": st is not None,
    }


def delta(a, b, key):
    x, y = a.get(key), b.get(key)
    if x is None or y is None:
        return None
    return y - x


def verdict(d):
    """一行说清责任。顺序就是证据的强度：越靠下越是我们自己的事。"""
    if d["nic_prio0"]:
        return "网卡 RX 队列（硬件）：ring 太小或中断被推迟 —— 需要 root 调 ring/IRQ"
    if d["nic_discards"] and not d["nic_prio0"]:
        return "网卡（PHY/端口层）丢的 —— 查 rx_crc/symbol/pause 那几个计数"
    if d["softnet_dropped"]:
        return "内核软中断 backlog 丢的 —— net.core.netdev_max_backlog"
    if d["sockdrop"]:
        return "我们自己的 socket 缓冲丢的 —— receiver.recv_buffer_bytes"
    if d["lost"]:
        return "包在到这台机器之前就没了（前端/线路/交换机）"
    if d["zf_win"]:
        return "应用侧（staging 环 / GPU / 写盘）—— 看 corr 的等环与等槽释放"
    return "无损"


def main():
    ap = argparse.ArgumentParser(description="补零的包丢在哪一层（只读）")
    ap.add_argument("--nic", default="ens2f0np0", help="收包网卡（默认 ens2f0np0）")
    ap.add_argument("--status-url", default="http://127.0.0.1:8000/status.json",
                    help="corr 的状态接口；不可达时只打网卡/软中断两层")
    ap.add_argument("--interval", type=float, default=5.0, help="采样间隔秒（默认 5）")
    ap.add_argument("--count", type=int, default=0, help="打几行后退出（0 = 直到 Ctrl-C）")
    a = ap.parse_args()

    prev = sample(a.nic, a.status_url)
    if prev["nic_err"]:
        print(f"注意：{prev['nic_err']}", file=sys.stderr)
    if not prev["running"]:
        print(f"注意：{a.status_url} 不可达（corr 没在跑？）—— 只打网卡与软中断两层",
              file=sys.stderr)

    line = 0
    try:
        while True:
            time.sleep(a.interval)
            cur = sample(a.nic, a.status_url)
            dt = max(cur["t"] - prev["t"], 1e-6)

            d = {
                "nic_prio0": delta(prev["nic"], cur["nic"], "rx_prio0_discards") or 0,
                "nic_discards": delta(prev["nic"], cur["nic"], "rx_discards_phy") or 0,
                "nic_oob": delta(prev["nic"], cur["nic"], "rx_out_of_buffer") or 0,
                "nic_steer": delta(prev["nic"], cur["nic"], "rx_steer_missed_packets") or 0,
                "pause": delta(prev["nic"], cur["nic"], "tx_global_pause") or 0,
                "pkts": delta(prev["nic"], cur["nic"], "rx_packets_phy") or 0,
                "crc": delta(prev["nic"], cur["nic"], "rx_crc_errors_phy") or 0,
                "softnet_dropped": delta(prev, cur, "softnet_dropped") or 0,
                "softnet_squeeze": delta(prev, cur, "softnet_squeeze") or 0,
                "lost": delta(prev, cur, "lost") or 0,
                "sockdrop": delta(prev, cur, "sockdrop") or 0,
                "zf_win": cur["zf_win"] or 0.0,
                "rows": cur["rows"],
                "bottleneck": cur["bottleneck"],
            }

            # 哪个 CPU 的软中断被预算掐断得最多、哪个 CPU 的 mlx5 中断涨得最少：
            # 这两条是"谁在被抢"的唯一线索，而无 root 时我们改不了它。
            sq = []
            if prev["squeeze_per_cpu"] and len(prev["squeeze_per_cpu"]) == len(cur["squeeze_per_cpu"]):
                sq = sorted(
                    ((cur["squeeze_per_cpu"][i] - prev["squeeze_per_cpu"][i], i)
                     for i in range(len(cur["squeeze_per_cpu"]))),
                    reverse=True,
                )[:3]
            irq = []
            if prev["irq"] and len(prev["irq"]) == len(cur["irq"]):
                irq = sorted(
                    ((cur["irq"][i] - prev["irq"][i], i) for i in range(len(cur["irq"]))),
                    reverse=True,
                )[:3]

            stamp = time.strftime("%H:%M:%S")
            print(
                f"[{stamp}] {dt:.1f}s | 到达 {d['pkts']/dt/1e6:.2f} Mpps | "
                f"网卡丢 +{d['nic_discards']:,}（prio0 +{d['nic_prio0']:,}） | "
                f"软中断丢 +{d['softnet_dropped']:,}（squeeze +{d['softnet_squeeze']:,}） | "
                f"sockdrop +{d['sockdrop']:,} | corr lost +{d['lost']:,} "
                f"{('zf ' + format(d['zf_win'], '.4f') + '%') if cur['running'] else '(未运行)'}"
                + (f" | 发 pause +{d['pause']}" if d["pause"] else "")
                + (f" | crc +{d['crc']}" if d["crc"] else "")
            )
            if sq:
                print("          软中断 squeeze 最多的核: "
                      + " ".join(f"cpu{i} +{v:,}" for v, i in sq if v))
            if irq:
                print("          mlx5 中断最多的核: "
                      + " ".join(f"cpu{i} +{v:,}" for v, i in irq if v))
            print(f"          → {verdict(d)}")

            prev = cur
            line += 1
            if a.count and line >= a.count:
                break
    except KeyboardInterrupt:
        print("\n(停止)")


if __name__ == "__main__":
    main()
