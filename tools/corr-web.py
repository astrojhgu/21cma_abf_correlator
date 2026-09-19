#!/usr/bin/env python3
"""corr-web —— corr 的独立只读监测界面。

corr 进程自己（`monitoring.http.enabled: true`）在一个独立线程上 serve 一份 JSON
快照：`GET http://127.0.0.1:8000/status.json`。本程序每秒取一次，自己攒历史，
再用一个页面把它画出来。分工是刻意的：

  * corr 只提供**当前**一份快照，接口小到可以一眼看完，且不在数据通路上
    （详见 docs/DESIGN-status-http.md）；
  * 历史、曲线、颜色、表格都是"看的人"的事，都在这里，重启 corr 不影响它。

依赖：只有 Python 3 标准库。不需要 root，不写 /scratch，不碰 corr 的输出目录。

用法::

    tools/corr-web.py                       # 端口从配置里读（monitoring.http.bind）
    tools/corr-web.py --config config/correlator-24-nf16384.yaml
    tools/corr-web.py --url http://127.0.0.1:3099/status.json
    tools/corr-web.py --bind 0.0.0.0:8080   # 让别的机器也能看（内网可见，谨慎）
    tools/corr-web.py --interval 0.5 --history 7200
    tools/corr-web.py --spectra-interval 0          # 关掉自相关谱那一段

corr 的地址**只定义一处**：配置里的 `monitoring.http.bind`。本程序默认去那份配置里读它
（读不到才退回 127.0.0.1:8000），所以在配置里改了端口不用在这里再改一次。

本程序只做两件事：轮询 corr 的两个只读端点，画出来。

    GET <bind>/status.json        运行状态（每秒）
    GET <bind>/autospectra.json   最新那一行的自相关谱，每站一条 dB 曲线（默认 5 s）

**自相关谱不是从盘上读回来的**：`j == k` 的那 24 条本来就在写盘端手里（一行 300 个基线
里的 24 个，8% 的字节），corr 在写这一行时顺手把它们归纳成曲线（见
`corr-rs/src/autospectra.rs`）。去 NFS 读回来只会和正在跑的写盘抢带宽，而这台机器
的写盘就是那条要被观察的路径本身。

页面上的 `connected: false` 不是错误，是"corr 现在没在跑或没开接口"——它会把最后
一份已知快照继续留着，并显示已经断了多久，所以跑完一腿之后仍然能回看。
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 网卡层的计数与判读只有一处实现：`loss_attribution.py`。这里直接借用它的取样函数，
# 而不是再写一遍 —— 这台机器上"补零"跟"网卡硬件队列丢包"经常是同一件事，两边口径
# 必须一致。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loss_attribution as la

# 取数地址会先试配置里的 `monitoring.http.bind`（见 `read_bind`），因为端口只该在
# 一处定义；这两个常量是读不到时的退路。
DEFAULT_URL = "http://127.0.0.1:8000/status.json"
DEFAULT_BIND = "127.0.0.1:8080"
DEFAULT_CONFIG = "config/correlator-24-nf16384-manual.yaml"


def read_bind(path):
    """从配置里读 `monitoring.http.bind`；读不到返回 None。

    手写解析而不是引 YAML：这里只要一个键，而标准库没有 YAML —— 为一个端口把脚本
    变成"需要装依赖"不划算。这段格式是我们自己写的，缩进稳定。
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    section = None
    sub = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            section, sub = line.split(":", 1)[0].strip(), None
            continue
        if section != "monitoring":
            continue
        stripped = line.strip()
        if indent == 2:
            sub = stripped.split(":", 1)[0].strip()
            continue
        if sub == "http" and stripped.startswith("bind:"):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def url_from_bind(bind):
    """`bind: 0.0.0.0:8000` 是"监听所有地址"，但我们是要**连**过去，不能连 0.0.0.0。"""
    host, _, port = bind.rpartition(":")
    if not port:
        return None
    if host in ("", "0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    return f"http://{host}:{port}/status.json"

# 页面里要画的每条曲线：字段取自 /api/state 的 history 元素。
# (key, 标题, 单位, 归一化上限或 None=按数据自适应)
SERIES = [
    ("zf", "补零率（窗口）", "%", None),
    ("rate", "行率", "r/s", None),
    ("cpu", "本进程 CPU", "核", None),
    ("link", "H2D 链路", "%", 120.0),
    ("wait", "收包侧等环", "%", 100.0),
    # 可写槽 = block_slots - spread - 1，所以这条曲线直接说明"等环为什么高"：贴近
    # 11 就是环被站间差占满。量程固定在环的槽数上，于是"离上限多近"是一眼可见的。
    ("spread", "站间进度差（环 11 槽）", "槽", 11.0),
    # 积压是"读不过来"的先行指标，而且是**唯一**会提前几小时动的量：在它撞到 socket
    # 缓冲上限（实测 granted 512 MiB，420 MB 时开始丢）之前，drops、零填充、等环、
    # spread 全都是 0。量程自适应：正常贴近 0，故障时到几百 MB。
    ("udp", "UDP 积压（最大 socket）", "MB", None),
]


class Collector(threading.Thread):
    """每秒取一次 corr 的状态快照，留一份历史。"""

    def __init__(self, url, spectra_url, interval, spectra_interval, history,
                 nic="", nic_interval=5.0):
        super().__init__(daemon=True, name="corr-web-collector")
        self.url = url
        self.spectra_url = spectra_url
        self.interval = interval
        self.spectra_interval = spectra_interval
        self.nic = nic
        self.nic_interval = nic_interval
        self.nic_prev = None
        self.nic_now = None
        self.lock = threading.Lock()
        self.samples = deque(maxlen=history)
        # 与 samples 一一对应：nic 采样频率与 status 轮询不同，逐条记录才能对齐。
        self.samples_udp = deque(maxlen=history)
        self.udp_backlog_mb = 0.0
        self.threads_now = None
        self._thread_prev = None
        self._thread_prev_t = None
        self._squeeze_prev = None
        self.latest = None
        self.error = None
        self.received_at = None
        self.count = 0
        self.spectra = None
        self.spectra_error = None
        self.spectra_updated = None
        self.nic_ring = None   # ring 深度不会变，读一次即可
        self.stop = threading.Event()

    def _get(self, url, timeout):
        req = urllib.request.Request(url, headers={"Cache-Control": "no-store"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _find_pid(self):
        """status.json 里没有 pid 时，自己去 /proc 里认领。"""
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open("/proc/%s/comm" % name) as fh:
                    comm = fh.read().strip()
            except OSError:
                continue
            if comm in ("corr-async", "corr"):
                return int(name)
        return None

    def _sample_threads(self):
        """收包线程的 CPU 分摊，加上 NAPI 没处理完的次数（time_squeeze）。

        **诊断用，不是预警**：健康运行时最慢的收包线程也只拿到 33%（它所在的核被网卡
        队列中断占了固定一份），最慢与最快能差 40 个百分点 —— 与故障时几乎一样，所以它
        区分不了好坏。真正敏感的信号是积压。这一块的用处是：积压开始涨的时候，在这里看
        出是哪几个线程被饿，以及内核是否已经在硬扛（`time_squeeze` 健康时应为 0）。
        """
        with self.lock:
            pid = (self.latest or {}).get("pid")
        if not pid:
            pid = self._find_pid()
        if not pid:
            return None

        now = time.monotonic()
        cur = {}
        try:
            for t in os.listdir("/proc/%d/task" % pid):
                try:
                    with open("/proc/%d/task/%s/stat" % (pid, t)) as fh:
                        raw = fh.read()
                    with open("/proc/%d/task/%s/comm" % (pid, t)) as fh:
                        comm = fh.read().strip()
                except OSError:
                    continue
                rest = raw[raw.rindex(")") + 2:].split()
                cur[t] = (comm, int(rest[11]) + int(rest[12]))
        except OSError:
            return None

        squeeze = 0
        try:
            with open("/proc/net/softnet_stat") as fh:
                for line in fh:
                    f = line.split()
                    if len(f) >= 3:
                        squeeze += int(f[2], 16)
        except OSError:
            pass

        prev, prev_t, prev_sq = self._thread_prev, self._thread_prev_t, self._squeeze_prev
        self._thread_prev, self._thread_prev_t, self._squeeze_prev = cur, now, squeeze
        if prev is None or prev_t is None or prev_sq is None:
            return {"warmup": True}

        dt = max(now - prev_t, 1e-6)
        hz = os.sysconf("SC_CLK_TCK")
        rx = []
        for t, (comm, ticks) in cur.items():
            if t in prev and comm.startswith("corr-async-rx"):
                rx.append((comm, (ticks - prev[t][1]) / hz / dt * 100.0))
        if not rx:
            return None
        rx.sort(key=lambda r: -r[1])
        # 线程名是 corr-async-rxN，而 N 就是站索引：实测 rx16 对应站 index 16（S08，端口
        # 10068），也正是事故里积压 420 MB 的那一路。所以逐线程的数可以直接贴到「每站」表。
        by_index = {}
        for name, pct in rx:
            suffix = name.rsplit("rx", 1)[-1]
            if suffix.isdigit():
                by_index[int(suffix)] = round(pct, 1)
        return {
            "warmup": False,
            "pid": pid,
            "count": len(rx),
            "by_index": by_index,
            "fastest": {"name": rx[0][0], "pct": rx[0][1]},
            "slowest": {"name": rx[-1][0], "pct": rx[-1][1]},
            "sum_pct": sum(r[1] for r in rx),
            "squeeze_per_s": max(0.0, (squeeze - prev_sq) / dt),
            "dt_s": dt,
        }

    def _udp_backlog(self):
        """每个 UDP socket 的接收积压与丢弃。

        积压是"读不过来"的**先行指标**，也是这次事故唯一早到的信号：实测里它涨到
        420 MB 之后才开始丢包，早了几个小时，而在此期间 drops、零填充、等环全都是 0，
        所以只盯 drops 的监控什么也看不见。这台机器 `net.core.rmem_max` 被设成 2 GB，
        socket 越大就把问题掩盖得越久 —— 于是更要直接看积压本身。
        """
        rows = []
        try:
            with open("/proc/net/udp") as fh:
                for line in fh.read().splitlines()[1:]:
                    f = line.split()
                    if len(f) < 13:
                        continue
                    try:
                        port = int(f[1].split(":")[1], 16)
                        rx_queue = int(f[4].split(":")[1], 16)
                        drops = int(f[12])
                    except (ValueError, IndexError):
                        continue
                    rows.append((port, rx_queue, drops))
        except OSError:
            return None
        if not rows:
            return None
        worst = max(rows, key=lambda r: r[1])
        return {
            "sockets": len(rows),
            "backlog_bytes_max": worst[1],
            "backlog_bytes_total": sum(r[1] for r in rows),
            "queued_sockets": sum(1 for r in rows if r[1] > 1_000_000),
            "drops_total": sum(r[2] for r in rows),
            "worst_port": worst[0],
        }

    def _poll_nic(self, now):
        """网卡层：硬件队列丢弃 / 软中断丢弃 / 我们发出的流控。见 loss_attribution.py。"""
        counters, err = la.nic_counters(self.nic)
        if err:
            self.nic_now = {"error": err}
            return
        if self.nic_ring is None:
            self.nic_ring = la.nic_ring_depth(self.nic) or {}
        dropped, squeeze, _per_cpu = la.softnet()
        cur = {"t": now, "counters": counters, "softnet_dropped": dropped,
               "softnet_squeeze": squeeze}
        prev = self.nic_prev
        self.nic_prev = cur
        if prev is None:
            self.nic_now = {"error": None, "warmup": True}
            return
        dt = max(cur["t"] - prev["t"], 1e-6)
        d = lambda k: (cur["counters"].get(k, 0) - prev["counters"].get(k, 0))  # noqa: E731
        udp = self._udp_backlog() or {}
        self.udp_backlog_mb = (udp.get("backlog_bytes_max", 0) or 0) / 1e6
        self.threads_now = self._sample_threads()
        self.nic_now = {
            "error": None,
            "dt_s": dt,
            "udp_sockets": udp.get("sockets", 0),
            "udp_backlog_bytes_max": udp.get("backlog_bytes_max", 0),
            "udp_backlog_mb": (udp.get("backlog_bytes_max", 0) or 0) / 1e6,
            "udp_queued_sockets": udp.get("queued_sockets", 0),
            "udp_drops_total": udp.get("drops_total", 0),
            "udp_worst_port": udp.get("worst_port"),
            "iface": self.nic,
            # ring 是网卡侧唯一的缓冲：占满就把包丢在网卡里，应用侧一个计数都看不到
            "ring_rx_current": self.nic_ring.get("rx_current"),
            "ring_rx_max": self.nic_ring.get("rx_max"),
            "passed_thres_per_s": d("rx_buffer_passed_thres_phy") / dt,
            "prio0_buf_discard_per_s": d("rx_prio0_buf_discard") / dt,
            "prio0_cong_discard_per_s": d("rx_prio0_cong_discard") / dt,
            "steer_missed_per_s": d("rx_steer_missed_packets") / dt,
            "buff_alloc_err_per_s": d("rx_buff_alloc_err") / dt,
            "discards_per_s": d("rx_prio0_discards") / dt,
            "phy_discards_per_s": d("rx_discards_phy") / dt,
            "out_of_buffer_per_s": d("rx_out_of_buffer") / dt,
            "pause_per_s": d("tx_global_pause") / dt,
            "crc_per_s": d("rx_crc_errors_phy") / dt,
            "mpps": d("rx_packets_phy") / dt / 1e6,
            "softnet_dropped_per_s": (cur["softnet_dropped"] - prev["softnet_dropped"]) / dt,
            "softnet_squeeze_per_s": (cur["softnet_squeeze"] - prev["softnet_squeeze"]) / dt,
        }

    def run(self):
        # Two endpoints, one thread: the page asks for both and they are cheap, and a
        # second thread would only add a second thing to shut down.
        last_spectra = 0.0
        last_nic = 0.0
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                snap = self._get(self.url, timeout=2.0)
                with self.lock:
                    if snap.get("ready"):
                        self.latest = snap
                        self.samples.append(snap)
                        self.samples_udp.append(self.udp_backlog_mb)
                    self.error = None
                    self.received_at = time.time()
                    self.count += 1
            except (urllib.error.URLError, OSError, ValueError) as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
            if self.nic and (started - last_nic) >= self.nic_interval:
                last_nic = started
                self._poll_nic(started)
            if self.spectra_interval > 0 and (started - last_spectra) >= self.spectra_interval:
                last_spectra = started
                try:
                    # A bigger timeout: the response carries 24 curves.
                    data = self._get(self.spectra_url, timeout=10.0)
                    with self.lock:
                        self.spectra = data
                        self.spectra_error = None
                        self.spectra_updated = time.time()
                except (urllib.error.URLError, OSError, ValueError) as exc:
                    with self.lock:
                        self.spectra_error = f"{type(exc).__name__}: {exc}"
            self.stop.wait(max(0.0, self.interval - (time.monotonic() - started)))

    def nic_state(self):
        with self.lock:
            return self.nic_now

    def spectra_state(self):
        with self.lock:
            return {
                "ready": self.spectra is not None and bool(self.spectra.get("ready")),
                "error": self.spectra_error,
                "updated_at": self.spectra_updated,
                "now": time.time(),
                "data": self.spectra,
                "url": self.spectra_url,
            }

    def latest_snapshot(self):
        with self.lock:
            return self.latest

    def state(self):
        """给页面用的紧凑视图：最新快照 + 每条曲线需要的字段。"""
        with self.lock:
            latest = self.latest
            error = self.error
            received_at = self.received_at
            count = self.count
            history = [
                {
                    "ts": s.get("ts_unix_ms", 0),
                    "zf": s["window"]["zero_fill_pct"],
                    "rate": s["window"]["rows_per_s"],
                    "cpu": s["host"]["cpu_cores"],
                    "cores": s["host"]["cores_online"],
                    "link": s["gpu"]["link_pct"],
                    "wait": s["recv"]["wait_pct"],
                    "spread": s["recv"]["spread_slots"],
                    "udp": udp_mb,
                }
                for s, udp_mb in zip(self.samples, self.samples_udp)
            ]
        return {
            "connected": error is None and latest is not None,
            "error": error,
            "received_at": received_at,
            "now": time.time(),
            "polls": count,
            "latest": latest,
            "history": history,
            "series": [{"key": k, "title": t, "unit": u, "max": m} for k, t, u, m in SERIES],
            "nic": self.nic_state(),
            "threads": self.threads_now,
            "url": self.url,
        }


PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>corr 状态</title>
<style>
:root { --bg:#111418; --fg:#e6e9ef; --dim:#8b94a3; --card:#181d24; --line:#252c36;
        --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
header { padding:14px 18px; border-bottom:1px solid var(--line); display:flex;
         gap:16px; align-items:baseline; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; font-weight:600; }
header .meta { color:var(--dim); font-size:12px; }
main { padding:14px 18px 40px; }
.cards { display:grid; gap:10px; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.card .k { color:var(--dim); font-size:11px; letter-spacing:.04em; text-transform:uppercase; }
.card .v { font-size:20px; margin-top:4px; }
.card .s { color:var(--dim); font-size:11px; margin-top:2px; }
#verdict.ok, .card .v.ok { color:var(--ok); }
#verdict.warn, .card .v.warn { color:var(--warn); }
#verdict.bad, .card .v.bad { color:var(--bad); }
section { margin-top:18px; }
h2 { font-size:13px; color:var(--dim); font-weight:600; margin:0 0 8px;
     letter-spacing:.05em; text-transform:uppercase; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { text-align:right; padding:3px 8px; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--dim); font-weight:500; }
tr.hot td { color:var(--bad); } tr.warm td { color:var(--warn); }
.charts { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
.chart { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.chart .t { color:var(--dim); font-size:12px; margin-bottom:6px; }
svg { width:100%; height:80px; display:block; }
polyline { fill:none; stroke:var(--accent); stroke-width:1.5; }
.chart .last { color:var(--fg); font-size:12px; float:right; }
.stale { color:var(--warn); }
.specs { display:grid; gap:8px; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); }
.spec { background:var(--card); border:1px solid var(--line); border-radius:6px;
        padding:6px 8px; }
.spec .n { display:flex; justify-content:space-between; color:var(--fg); font-size:12px; }
.spec .lvl { color:var(--dim); font-size:10px; margin-top:1px; }
.spec svg { height:72px; margin-top:4px; }
.spec polyline { fill:none; stroke:var(--accent); stroke-width:1; }
.spec.quiet polyline { stroke:#6e7681; }
.spec .axis { display:flex; justify-content:space-between; color:var(--dim); font-size:10px; }
.spec .axis .mid { color:var(--dim); }
/* 水位线：0 处的竖线就是水位线，落在线左边的红条是"落后于水位线"的部分——那些帧会被
   记成 missing；cap 线画在同一根轴上，所以"离被迫放弃还差多远"是看得见的。轴是对称的，
   0 永远在正中，两次刷新之间位置可比。这一列**并进「每站」表**：数值与条并排，一行同时
   给出"哪一路"和"差多少"，不再单独占一张图。 */
/* 最后一列要够宽，条才有分辨率：auto 布局会把富余空间分给别的列，结果条被压到几十
   像素，"哪一路领先多少"就看不出来了。 */
th:last-child, td:last-child { width:44%; }
.wmcell { display:flex; align-items:center; gap:6px; }
.wmcell .wmv { width:48px; flex:none; text-align:right; font-variant-numeric:tabular-nums; }
.wmcell .wmv.lag { color:var(--bad); }
.wmcell .wmv.lead { color:var(--ok); }
.wmtrack { position:relative; flex:1; min-width:200px; height:11px; background:var(--card);
           border:1px solid var(--line); border-radius:3px; }
.wmtrack i { position:absolute; top:1px; bottom:1px; opacity:.85; }
.wmtrack i.lag { background:var(--bad); }
.wmtrack i.lead { background:var(--ok); }
.wmtrack .zero, .wmtrack .cap { position:absolute; top:-1px; bottom:-1px; width:1px; }
.wmtrack .zero { background:var(--fg); opacity:.7; }
.wmtrack .cap { background:var(--warn); width:2px; opacity:.9; }
</style></head><body>
<header>
  <h1 id="run">corr 状态</h1>
  <span class="meta" id="conn"></span>
  <span class="meta" id="set"></span>
  <span class="meta" id="proc"></span>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="k">rows</div><div class="v" id="rows">–</div>
      <div class="s" id="rate">–</div></div>
    <div class="card"><div class="k">补零率</div><div class="v" id="zf">–</div>
      <div class="s" id="zfwin">–</div></div>
    <div class="card"><div class="k">lost / sockdrop</div><div class="v" id="lost">–</div>
      <div class="s" id="lostwin">–</div></div>
    <div class="card"><div class="k">收包侧等环</div><div class="v" id="wait">–</div>
      <div class="s" id="spread">–</div></div>
    <div class="card"><div class="k">水位线</div><div class="v" id="wm">–</div>
      <div class="s" id="wmsub">–</div></div>
    <div class="card"><div class="k">H2D</div><div class="v" id="h2d">–</div>
      <div class="s" id="slotfree">–</div></div>
    <div class="card"><div class="k">CPU / load</div><div class="v" id="cpu">–</div>
      <div class="s" id="load">–</div></div>
    <div class="card"><div class="k">写盘队列</div><div class="v" id="wq">–</div>
      <div class="s" id="wqsub">–</div></div>
    <div class="card"><div class="k">网卡缓冲（RX ring）</div><div class="v" id="nicbuf">–</div>
      <div class="s" id="nicbufsub">–</div></div>
    <div class="card"><div class="k">收包线程（诊断）</div><div class="v" id="threads">–</div>
      <div class="s" id="threadsub">–</div></div>
    <div class="card"><div class="k">capacity</div><div class="v" id="verdict">–</div>
      <div class="s" id="limits">–</div></div>
    <div class="card"><div class="k">网卡层（ring 溢出）</div><div class="v" id="nic">–</div>
      <div class="s" id="nicsub">–</div></div>
  </div>

  <section><h2>曲线（本程序自己攒的）</h2><div class="charts" id="charts"></div></section>

  <section><h2>自相关谱（每路，dB）</h2>
    <div class="meta" id="spec-meta">读取中…</div>
    <div class="specs" id="specs"></div>
  </section>

  <section><h2>每站</h2>
    <div class="meta" id="wm-meta">等待第一份快照…</div>
    <table><thead><tr>
      <th>station</th><th>progress</th><th>zero-filled</th><th>lost</th>
      <th>等环 ms</th><th>静默 s</th><th>线程 CPU</th><th>相对水位线（槽）</th>
    </tr></thead><tbody id="stations"></tbody></table>
  </section>
</main>
<script>
// 首屏快照：服务端吐 HTML 时已经把最新一份 state 内联进来了，所以打开页面立刻就有数据，
// 不必先看一秒空表（也让"截图那一刻"能看到真实内容）。之后 tick() 照常轮询覆盖它。
const BOOT = __BOOT__;
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : v.toFixed(d);
let chartsBuilt = false;

function buildCharts(series) {
  const box = document.getElementById("charts");
  box.innerHTML = "";
  for (const s of series) {
    const div = document.createElement("div");
    div.className = "chart";
    div.innerHTML = `<div class="t"><span class="last" id="last-${s.key}">–</span>${s.title}</div>
                     <svg viewBox="0 0 600 80" preserveAspectRatio="none"
                          ><polyline id="line-${s.key}" points=""></polyline></svg>`;
    box.appendChild(div);
  }
  chartsBuilt = true;
}

function drawSeries(s, hist) {
  const line = document.getElementById(`line-${s.key}`);
  const last = document.getElementById(`last-${s.key}`);
  if (!line || hist.length === 0) return;
  const vals = hist.map(p => p[s.key]);
  const cur = vals[vals.length - 1];
  // 量程用**整个保留历史**，不是当前画出来的这一段：用窗口内最大值定标时，峰值一旦
  // 滑出窗口曲线就自己重新缩放，纵轴于是永远读不出数（看到的现象）。同时把峰值写在
  // 当前值旁边，这样"回落到多少"是能读的，而不是猜的。
  const peak = Math.max(...vals);
  last.textContent = `${fmt(cur)} ${s.unit} · 峰值 ${fmt(peak)}`;
  let max = s.max !== null ? s.max : Math.max(peak, 1e-9);
  if (!(max > 0)) max = 1;
  const n = Math.max(hist.length - 1, 1);
  const pts = vals.map((v, i) => {
    const x = (i / n) * 600;
    const y = 78 - Math.min(Math.max(v / max, 0), 1) * 74;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  line.setAttribute("points", pts.join(" "));
}

// 水位线：每站相对水位线的位置，画进「每站」表的最后一列（数值 + 条），不再单独成图。
// 0 处的竖线是水位线；落在它左边（红）的站就是"帧会被记成 missing"的那几个 —— 也就是把
// 整环水位线按住不放的站。cap 线画在同一根轴上，于是"离被迫把它放弃还差多远"是看得见的。
// 轴取对称量程，0 永远在正中，两次刷新之间位置可比。
//
// 量程跟着数据走，**不**被 cap 撑开：cap 是"还差多远才被迫放弃"的参照，不是当前量程。正常
// 运行时站间差只有 0.0x 槽，若把 4 槽的 cap 一并画进轴里，24 个站会全挤在中线上，这一列
// 就等于没有（单独成图的第一版就是这样）。cap 落在轴外时在表头说明里点名。
function watermarkScale(offs, capv) {
  const maxAbs = (offs || []).reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const half = Math.max(maxAbs * 1.15, 0.05);
  return {
    half,
    capIn: capv !== null && capv !== undefined && capv <= half,
    at: v => ((v + half) / (2 * half)) * 100,
  };
}

// 水位线本身读 0.00，不带符号：它不是"领先了一点点"。
const wmText = v => Math.abs(v) < 0.005 ? "0.00" : (v > 0 ? "+" : "") + v.toFixed(2);

// 表格某一格的内容：数值 + 条。抽成独立函数是为了能在没有浏览器的环境里单独验它。
function wmBar(off, sc, capv) {
  if (off === null || off === undefined) {
    return `<div class="wmcell"><span class="wmv">–</span></div>`;
  }
  const a = Math.min(sc.at(0), sc.at(off));
  const b = Math.max(sc.at(0), sc.at(off));
  const what = off < 0
    ? `落后水位线 ${Math.abs(off).toFixed(2)} 槽（这些帧记成 missing）`
    : `领先水位线 ${off.toFixed(2)} 槽`;
  return `<div class="wmcell">`
    + `<span class="wmv ${off < 0 ? "lag" : "lead"}">${wmText(off)}</span>`
    + `<span class="wmtrack" title="${what}">`
    + `<i class="${off < 0 ? "lag" : "lead"}" `
    + `style="left:${a.toFixed(2)}%;width:${Math.max(b - a, 0.5).toFixed(2)}%"></i>`
    + (sc.capIn ? `<span class="cap" style="left:${sc.at(-capv).toFixed(2)}%" `
                 + `title="cap ${capv} 槽：再落后这么多就不再等它"></span>` : "")
    + `<span class="zero" style="left:${sc.at(0).toFixed(2)}%"></span>`
    + `</span></div>`;
}

// 网卡层卡片：判据是 ring 溢出，不是端口级丢弃。
//
// `rx_prio0_discards`（与 `rx_discards_phy` 是同一个寄存器的两个名字）是**端口级**计数：
// 进入这张卡的包里被网卡丢掉的总数。实测它偶发增长时，ring 溢出、`/proc/net/udp` 的 drops
// 和 corr 自己的 lost/sockdrop 往往全是 0 —— 那些包不是我们的（多为其它流量或未订阅的组播），
// 用它当报警条件会误报。真正会变成零填充的是 `rx_out_of_buffer`：RX ring 满、描述符不够。
// 所以主值只认它；端口级丢弃退到副行，并且只有当同一窗口的零填充也在涨时才变色 ——
// 那才叫"这次真的丢到我们了"。
function nicCard(nic, zfWinPct) {
  if (!nic) return { v: "–", cls: "v", sub: "未采样", title: "" };
  if (nic.error) return { v: "?", cls: "v warn", sub: nic.error, title: "" };
  if (nic.warmup) return { v: "…", cls: "v", sub: "需要两次采样才能算增量", title: "" };

  const ring = nic.out_of_buffer_per_s || 0;
  const port = nic.discards_per_s || 0;
  const zfHot = (zfWinPct || 0) > 0;
  const backlogMb = (nic.udp_backlog_bytes_max || 0) / 1e6;
  const queued = nic.udp_queued_sockets || 0;
  const sub = `到达 ${(nic.mpps || 0).toFixed(2)} Mpps · ring 溢出 ${ring.toFixed(0)}/s`
    + ` · UDP 积压 ${backlogMb < 0.05 ? "0" : backlogMb.toFixed(1)} MB（${queued} 个 socket >1 MB）`
    + ` · 端口级丢弃 ${port.toFixed(0)}/s · pause ${(nic.pause_per_s || 0).toFixed(0)}/s`
    + ` · 软中断丢 ${(nic.softnet_dropped_per_s || 0).toFixed(0)}/s`;

  if (ring > 0) {
    return {
      v: `溢出 ${ring.toFixed(0)}/s`, cls: "v bad", sub,
      title: "RX ring 满、描述符不够：这些包在网卡里就没了，应用侧的 lost/sockdrop/udp drops "
           + "都看不到它们——它们会直接变成零填充。调大 ring 才治这个（ethtool -G，需要 root）",
    };
  }
  if (backlogMb >= 128) {
    return {
      v: `积压 ${backlogMb.toFixed(0)} MB`, cls: "v bad", sub,
      title: "接收线程读不过来：socket 缓冲堆到了百 MB 级（实测事故里到过 420 MB）。此刻还没丢，"
           + "但延迟已经是几十毫秒；等到追上产能就开始成片丢包，最慢的那几站还会把水位线拖住，"
           + "把零填充摊到所有站上。先查谁的核被中断占了（这台机器 mlx5 的中断会落在全部 32 个核上）",
    };
  }
  if (port > 0 && zfHot) {
    return {
      v: `端口丢 ${port.toFixed(0)}/s`, cls: "v bad", sub,
      title: "端口级丢弃，而且这一窗口的零填充同时在涨：这次丢的里面确实有 corr 的包",
    };
  }
  if (backlogMb >= 16) {
    return {
      v: `积压 ${backlogMb.toFixed(0)} MB`, cls: "v warn", sub,
      title: "接收侧开始积压（还不是丢包）：读得比到得慢一点。这个数应当长期贴近 0",
    };
  }
  return {
    v: "0/s", cls: "v ok", sub,
    title: port > 0
      ? "端口级丢弃是有的，但 ring 没溢出、零填充没涨、socket 也没积压：这些包不是 corr 的"
      : "网卡三层（ring 溢出 / 中断 / 软中断）都没有丢弃，接收侧也没有积压",
  };
}

// 网卡缓冲卡片：ring 深度，以及它"相当于多少毫秒的到达量"。
//
// ring 是网卡侧唯一的缓冲：包进来没被驱动取走就占一个描述符，占满就丢在网卡里，应用侧的
// lost / sockdrop / udp drops 一个都看不到 —— 所以深度和余量得自己算：深度 ÷ 到达速率。
// 这台机器是 1024 个描述符配 4.7 Mpps，只有 0.22 ms，一次几百微秒的调度延迟就能填满。
// 深度要 root 才能改（ethtool -G），而它出厂默认只有最大值的 1/8。
function bufferCard(nic) {
  if (!nic) return { v: "–", cls: "v", sub: "未采样", title: "" };
  if (nic.error) return { v: "–", cls: "v", sub: nic.error, title: "" };
  if (nic.warmup) return { v: "…", cls: "v", sub: "需要两次采样才能算增量", title: "" };

  const cur = nic.ring_rx_current, max = nic.ring_rx_max;
  const mpps = nic.mpps || 0;
  const ms = (cur && mpps > 0) ? (cur / (mpps * 1e6)) * 1e3 : null;
  const oob = nic.out_of_buffer_per_s || 0;
  const thres = nic.passed_thres_per_s || 0;
  const bufD = nic.prio0_buf_discard_per_s || 0;
  const congD = nic.prio0_cong_discard_per_s || 0;
  const miss = nic.steer_missed_per_s || 0;
  const allocErr = nic.buff_alloc_err_per_s || 0;
  const iface = nic.iface || "iface";

  const v = (cur === null || cur === undefined) ? "–"
      : `${cur}${max ? " / " + max : ""} 描述符`;
  const sub = (ms === null ? "包率未知，无法换算成时间" : `≈ ${ms.toFixed(2)} ms 的到达量`)
    + ` · 缓冲丢 ${bufD.toFixed(0)}/s · 拥塞丢 ${congD.toFixed(0)}/s`
    + ` · 阈值外 ${thres.toFixed(0)}/s · 未命中 ${miss.toFixed(0)}/s`
    + (allocErr > 0 ? ` · 分配失败 ${allocErr.toFixed(0)}/s` : "");

  if (oob > 0) {
    return { v, cls: "v bad", sub,
      title: `ring 的 ${cur} 个描述符全占满了，包开始丢在网卡里（rx_out_of_buffer 在涨）。`
           + `调大它：ethtool -G ${iface} rx ${max || 8192}（需要 root）` };
  }
  if (thres > 0 || bufD > 0 || congD > 0) {
    return { v, cls: "v warn", sub,
      title: "有包走到了缓冲阈值 / 优先级缓冲边界：ring 还没满，但余量在被消耗" };
  }
  return { v, cls: "v ok", sub,
    title: `ring ${cur}/${max} 个描述符，没有缓冲类丢弃。当前深度是出厂默认（最大值的 1/8），`
         + `按 ${mpps.toFixed(2)} Mpps 只有 ${ms === null ? "?" : ms.toFixed(2)} ms 余量；`
         + `调大需要 root：ethtool -G ${iface} rx ${max || 8192}` };
}

// 收包线程卡片：诊断用，不是预警。
//
// 健康运行时最慢的收包线程也只拿到 33%（它所在的核被网卡队列中断占了固定一份），最慢
// 与最快能差 40 个百分点 —— 与故障时几乎一样，所以它区分不了好坏，别拿它当警报。它的
// 用处是：积压开始涨的时候，在这里看是哪几个线程被饿。主值取 time_squeeze 的增量，那
// 是"NAPI 一轮都没把队列收干净"的直接证据，健康运行时应为 0。
function threadsCard(t) {
  if (!t) return { v: "–", cls: "v", sub: "看不到收包进程（corr 未运行？）", title: "" };
  if (t.warmup) return { v: "…", cls: "v", sub: "需要两次采样才能算 CPU", title: "" };

  const sq = t.squeeze_per_s || 0;
  const slow = t.slowest || {}, fast = t.fastest || {};
  const sub = `rx ${t.count} 个 · 合计 ${(t.sum_pct / 100).toFixed(2)} 核`
    + ` · 最慢 ${slow.name || "?"} ${(slow.pct || 0).toFixed(1)}%`
    + ` · 最快 ${fast.name || "?"} ${(fast.pct || 0).toFixed(1)}%`
    + `（差 ${((fast.pct || 0) - (slow.pct || 0)).toFixed(0)} 个百分点）`
    + ` · 逐线程见下方「每站」表`;

  if (sq >= 100) {
    return { v: `squeeze ${sq.toFixed(0)}/s`, cls: "v bad", sub,
      title: "NAPI 一轮的预算用完了还没收干净队列，内核在硬扛，收包线程必然被饿。"
           + "这是积压涨起来时最该先看的一条" };
  }
  if (sq > 0) {
    return { v: `squeeze ${sq.toFixed(0)}/s`, cls: "v warn", sub,
      title: "偶发 NAPI 预算不足。健康运行时应为 0；持续非零说明管中断的那些核在过载" };
  }
  return { v: "0/s", cls: "v ok", sub,
    title: `收包线程 CPU 分布（诊断用）：最慢 ${(slow.pct || 0).toFixed(1)}%、`
         + `最快 ${(fast.pct || 0).toFixed(1)}%。这个差距健康时也有 40 个百分点，`
         + `所以它区分不了好坏 —— 积压才是那个信号，这里用来查原因` };
}

function render(s) {
  const snap = s.latest;
  const conn = document.getElementById("conn");
  if (!snap) {
    conn.textContent = s.error ? `未连接：${s.error}` : "等待 corr 的第一份快照…";
    conn.className = "meta stale";
    document.getElementById("charts").innerHTML = "";
    chartsBuilt = false;
    return;
  }
  const age = s.received_at ? (s.now - s.received_at) : 0;
  conn.className = "meta" + (s.connected ? "" : " stale");
  conn.textContent = s.connected
      ? `已连接 · ${s.url} · 轮询 ${s.polls} 次`
      : `连接断开 ${age.toFixed(0)} s（${s.error || "无数据"}）· 显示最后一份快照`;

  // 进程自身的占用：长期跑时这四个数字必须平。RSS 等于峰值意味着"从未比现在更大"；
  // 线程与 fd 在跨本地日切套前后应当不变（切套会关掉旧套的 300 个句柄）。
  const p = snap.process;
  if (p) {
    document.getElementById("proc").textContent =
      `RSS ${(p.rss_mib / 1024).toFixed(2)} GiB（峰 ${(p.peak_rss_mib / 1024).toFixed(2)}）`
      + ` · 线程 ${p.threads} · fd ${p.fds} · 已跑 ${(p.uptime_s / 3600).toFixed(2)} h`;
  }

  const run = snap.run, cum = snap.cumulative, win = snap.window;
  document.getElementById("run").textContent = run.name + "  " + run.n_station + " 路";
  document.getElementById("set").textContent =
      `${run.set_tag || "(未开套)"} · n_freq ${run.n_freq} · n_cum ${run.n_cum} · `
      + `${fmt(run.integration_ms, 1)} ms 积分 · ${run.output_dir}`;
  document.getElementById("rows").textContent = cum.rows;
  document.getElementById("rate").textContent =
      `${fmt(win.rows_per_s)} r/s（窗口 ${fmt(win.dt_s, 1)} s，${win.rows} 行）`;
  document.getElementById("zf").textContent = fmt(cum.zero_fill_pct, 4) + " %";
  document.getElementById("zfwin").textContent =
      `窗口 ${fmt(win.zero_fill_pct, 4)} %（${win.zero_filled_frames} 帧）`;
  document.getElementById("lost").textContent =
      cum.lost_packets.toLocaleString() + " / " + cum.socket_drops.toLocaleString();
  document.getElementById("lostwin").textContent =
      `窗口 +${win.lost_packets.toLocaleString()} / +${win.socket_drops.toLocaleString()} · `
      + `stalled ${cum.stall_recoveries}`;
  document.getElementById("wait").textContent = fmt(snap.recv.wait_pct, 0) + " %";
  document.getElementById("spread").textContent =
      `spread ${fmt(snap.recv.spread_slots, 1)} 槽 · blocked ${snap.recv.blocked_now}`;
  document.getElementById("h2d").textContent = fmt(snap.gpu.h2d_gbs, 1) + " GB/s";
  document.getElementById("slotfree").textContent =
      `需求 ${fmt(snap.gpu.required_gbs, 1)} · 链路 ${fmt(snap.gpu.link_pct, 0)}% · `
      + `槽释放 ${fmt(snap.gpu.slot_free_pct, 0)}%`;
  document.getElementById("cpu").textContent =
      `${fmt(snap.host.cpu_cores, 1)}/${snap.host.cores_online}`;
  document.getElementById("load").textContent = `load1 ${fmt(snap.host.load1, 1)}`;

  // 网卡层：主值只认 ring 溢出，端口级丢弃退到副行（判据见 nicCard 的注释）。
  const nc = nicCard(s.nic, snap.window ? snap.window.zero_fill_pct : 0);
  const nv = document.getElementById("nic");
  const ns = document.getElementById("nicsub");
  nv.textContent = nc.v; nv.className = nc.cls; nv.title = nc.title;
  ns.textContent = nc.sub;

  // 写盘队列：占用率是"丢之前"的先行指标，丢行数是已经发生的损害。丢一行 = 数据里一个
  // 洞，而且它在**所有**文件里都不存在（blocks.csv 的 row_index 会在那里跳跃），所以这一格
  // 一变红就是"这腿的数据有缺口"，而不是"有点慢"。
  const w = snap.writer;
  const wv = document.getElementById("wq");
  const ws = document.getElementById("wqsub");
  if (!w) {
    wv.textContent = "–"; wv.className = "v";
    ws.textContent = "本次运行没有 writer（dry run？）";
  } else {
    const dropping = (w.dropped_window || 0) > 0;
    const hot = (w.queued_pct || 0) >= 75;
    wv.textContent = dropping ? `丢 ${w.dropped_window} 行` : `${w.queued}/${w.queue_depth}`;
    wv.className = "v " + (dropping ? "bad" : (hot ? "warn" : "ok"));
    wv.title = dropping
      ? "写盘队列满，本窗口的行被丢弃：这些行在本次运行的每个文件里都不存在"
      : "队列占用是丢弃之前的先行指标；到 75% 以上就该留意写盘速度了";
    ws.textContent = `占用 ${(w.queued_pct || 0).toFixed(0)}% · 峰值 ${w.max_queued}/${w.queue_depth}`
      + ` · 累计丢 ${w.dropped_total} 行`
      + (dropping ? `（本窗口占 ${(w.dropped_pct_window || 0).toFixed(0)}% 的行）` : "");
  }

  // 网卡缓冲：ring 深度与它的余量（判据见 bufferCard）
  const bc = bufferCard(s.nic);
  const nbv = document.getElementById("nicbuf");
  const nbs = document.getElementById("nicbufsub");
  nbv.textContent = bc.v; nbv.className = bc.cls; nbv.title = bc.title;
  nbs.textContent = bc.sub;

  // 收包线程：诊断用（判据见 threadsCard）
  const tc = threadsCard(s.threads);
  const tv = document.getElementById("threads");
  const tsub = document.getElementById("threadsub");
  tv.textContent = tc.v; tv.className = tc.cls; tv.title = tc.title;
  tsub.textContent = tc.sub;

  const v = document.getElementById("verdict");
  v.textContent = snap.capacity.bottleneck;
  // 丢行单独点出来：红。它和 socket 丢包不同 —— 那些包没了就是没了，而丢一行是**每个
  // 文件都缺那一行**，事后补不回来。
  v.className = snap.capacity.bottleneck === "ok" ? "ok"
      : (snap.capacity.bottleneck === "gpu_link" || snap.capacity.bottleneck === "process_cpu")
        ? "warn"
        : (snap.capacity.bottleneck === "writer_drops" ? "bad" : "bad");
  v.title = snap.capacity.detail;
  const lim = snap.capacity.limits;
  document.getElementById("limits").textContent =
      `门限：环 ${fmt(lim.ring_busy, 0)}% · 链路 ${fmt(lim.link, 0)}% · CPU ${fmt(lim.cpu, 0)}%`;

  // 水位线：谁在拖后腿、拖多少、离 cap 多远。spread 只说差多少，这一块说差在谁身上。
  const offs = (snap.recv && snap.recv.stations_offset_slots) || [];
  const capv = snap.recv ? snap.recv.max_spread_slots : null;
  const wmv = document.getElementById("wm");
  const wms = document.getElementById("wmsub");
  const wmMeta = document.getElementById("wm-meta");
  // 量程只算一次，表格每一行共用：同一张表里的条必须同一把尺子。
  const sc = watermarkScale(offs, capv);
  if (offs.length === 0) {
    wmv.textContent = "–"; wmv.className = "v";
    wms.textContent = "还没有行闭合，各站进度尚未形成水位线";
    wmMeta.className = "meta stale";
    wmMeta.textContent = "末列的水位线条：等第一行闭合";
  } else {
    const lag = Math.max(0, -Math.min(...offs));
    const lead = Math.max(...offs);
    const slow = offs.indexOf(Math.min(...offs));
    const slowName = (snap.stations && snap.stations[slow]) ? snap.stations[slow].name : slow;
    const lifted = snap.recv.watermark_lift_slots || 0;
    wmv.textContent = lag.toFixed(2) + " 槽";
    wmv.className = "v " + (lag > 0.5 ? "bad" : (lag > 0.05 ? "warn" : "ok"));
    wmv.title = "最慢的站落后水位线多少；落后于水位线的帧会被记成 missing，也就是补零的来源";
    wms.textContent = `最慢 #${slowName} · 最快 +${lead.toFixed(2)} · spread ${fmt(snap.recv.spread_slots, 2)}`
      + (capv ? ` · cap ${capv}` : " · 无 cap")
      + (lifted > 0 ? ` · 已抬 ${lifted.toFixed(2)} 槽` : "");
    wmMeta.className = "meta";
    wmMeta.textContent = `末列：每站相对水位线的位置（槽）——竖线 = 水位线（0），绿 = 领先、`
      + `红 = 落后于水位线（落后部分的帧进 blocks.csv 时记成 missing）· 轴 ±${sc.half.toFixed(2)} 槽`
      + ((capv === null || capv === undefined) ? " · 未设 cap"
          : (sc.capIn ? ` · 橙线 = cap ${capv} 槽` : ` · cap ${capv} 槽在轴外（离得还远）`));
  }

  const tb = document.getElementById("stations");
  tb.innerHTML = "";
  // 逐线程的 CPU，按站索引对齐（线程 rxN ↔ 站 N）。
  const thrByIndex = (s.threads && s.threads.by_index) || {};
  snap.stations.forEach((st, i) => {
    const tr = document.createElement("tr");
    const off = offs.length > i ? offs[i] : null;
    const thr = thrByIndex[i];
    const hot = st.silence_s > 2 || st.lost_packets > 0 && snap.recv.wait_pct > 50;
    tr.className = st.silence_s > 2 ? "hot" : (st.zero_filled_frames > 0 ? "warm" : "");
    tr.innerHTML = `<td>${st.name}</td><td>${st.progress_frames}</td>
      <td>${st.zero_filled_frames}</td><td>${st.lost_packets.toLocaleString()}</td>
      <td>${fmt(st.slot_wait_ms, 0)}</td><td>${fmt(st.silence_s, 1)}</td>
      <td>${thr === null || thr === undefined ? "–" : thr.toFixed(1)}</td>
      <td>${wmBar(off, sc, capv)}</td>`;
    if (hot) tr.title = "这一路的静默或等待明显偏高";
    if (thr !== undefined) {
      // 不上色：这个值健康时也就 32%（那个核同时扛着本路队列的中断），与故障时几乎一样，
      // 所以它区分不了好坏。它回答的是"积压涨起来时是谁被饿"。
      tr.title = (tr.title ? tr.title + "；" : "")
        + `这一路的收包线程 ${thr.toFixed(1)}% 核（健康时最慢的也就 32% 上下，别当警报看）`;
    }
    if (off !== null && off < 0) {
      tr.title = (tr.title ? tr.title + "；" : "")
        + "这一路在水位线之下：落后部分按 missing 记录，行里表现为补零";
    }
    tb.appendChild(tr);
  });

  if (!chartsBuilt) buildCharts(s.series);
  for (const ser of s.series) drawSeries(ser, s.history);
}

async function tick() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    render(await r.json());
  } catch (e) {
    const conn = document.getElementById("conn");
    conn.className = "meta stale";
    conn.textContent = "界面自身取数失败：" + e;
  }
}

// 自相关谱：一格一路。所有格共用同一段纵轴（d.ymin..d.ymax），这样端口之间的
// 电平差看得出来 —— 那是这张图一半的用处。
let specCount = 0;
async function tickSpectra() {
  const meta = document.getElementById("spec-meta");
  let s;
  try {
    s = await (await fetch("/api/spectra", { cache: "no-store" })).json();
  } catch (e) {
    meta.className = "meta stale";
    meta.textContent = "频谱取数失败：" + e;
    return;
  }
  if (!s.ready) {
    meta.className = "meta stale";
    meta.textContent = "还没有谱：" + (s.error || "读取中…");
    return;
  }
  const d = s.data;
  const age = s.updated_at ? (s.now - s.updated_at) : 0;
  const when = d.row_ms ? new Date(d.row_ms).toLocaleTimeString() : "?";
  meta.className = "meta" + (age > 15 ? " stale" : "");
  meta.textContent =
    `第 ${d.row_index} 行（${when}） · ${d.stations.length} 路 · `
    + `每格 ${d.points} 点（${d.step} 通道/点，跳过前 ${d.skip_bins} 个通道） · `
    + `${d.freq_mhz[0]}–${d.freq_mhz[1]} MHz · 纵轴 ${d.ymin}..${d.ymax} dB · `
    + `${d.db_ref} · ${age.toFixed(0)} s 前更新`;

  const box = document.getElementById("specs");
  if (specCount !== d.stations.length) { box.innerHTML = ""; specCount = d.stations.length; }
  const span = Math.max(1e-9, d.ymax - d.ymin);
  d.stations.forEach((st, i) => {
    let el = box.children[i];
    if (!el) {
      el = document.createElement("div");
      el.className = "spec";
      el.innerHTML = '<div class="n"><span class="nm"></span><span class="pk"></span></div>'
        + '<svg viewBox="0 0 300 72" preserveAspectRatio="none"><polyline points=""/></svg>'
        + '<div class="lvl"></div><div class="axis"></div>';
      box.appendChild(el);
    }
    el.querySelector(".nm").textContent = `${st.name}  (${st.index},${st.index})`;
    el.querySelector(".pk").textContent = st.db.length ? `${st.peak_db.toFixed(1)} dB` : "—";
    el.querySelector(".lvl").textContent = st.db.length
      ? `峰 ${st.peak_mhz} MHz · 中位 ${st.median_db.toFixed(1)} dB`
      : "无数据";
    el.querySelector(".axis").innerHTML =
      `<span>${d.freq_mhz[0]}</span><span class="mid">MHz</span><span>${d.freq_mhz[1]}</span>`;
    el.classList.toggle("quiet", !st.db.length);
    const n = Math.max(st.db.length - 1, 1);
    el.querySelector("polyline").setAttribute("points", st.db.map((v, k) => {
      const x = (k / n) * 300;
      const y = 70 - Math.min(Math.max((v - d.ymin) / span, 0), 1) * 68;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" "));
  });
}

if (BOOT && BOOT.latest) render(BOOT);   // 首屏用内联快照，不等第一次轮询
tick();
setInterval(tick, 1000);
tickSpectra();
// 服务端每 --spectra-interval 秒才重读一次盘（默认 5 s），所以这里也按 5 s 取。
setInterval(tickSpectra, 5000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    collector = None
    server_version = "corr-web"

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler's naming)
        path = self.path.split("?", 1)[0]
        if path == "/":
            # 内联最新一份快照：首屏不必先空一秒。`</` 必须转义，否则 JSON 里的字符串
            # （运行名、输出目录这些）会提前结束 <script> 标签。
            boot = json.dumps(self.collector.state()).replace("</", "<\\/")
            body = PAGE.replace("__BOOT__", boot).encode("utf-8")
            ctype = "text/html; charset=utf-8"
        elif path == "/api/state":
            body = json.dumps(self.collector.state()).encode("utf-8")
            ctype = "application/json"
        elif path == "/api/spectra":
            body = json.dumps(self.collector.spectra_state()).encode("utf-8")
            ctype = "application/json"
        else:
            body, ctype = b'{"error":"no such path"}', "application/json"
            self.send_response(404)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Quiet: a page opening is not news, and this console shows corr's log."""


def split_bind(text):
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"--bind 需要 host:port，收到 {text!r}")
    return host, int(port)


def main():
    ap = argparse.ArgumentParser(description="corr 的独立只读监测界面")
    ap.add_argument("--url", default=None,
                    help="corr 的状态接口；默认按 --config 里的 monitoring.http.bind 拼")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help=f"从这份配置里读 monitoring.http.bind（默认 {DEFAULT_CONFIG}）")
    ap.add_argument("--bind", default=DEFAULT_BIND,
                    help=f"本程序的监听地址（默认 {DEFAULT_BIND}）")
    ap.add_argument("--interval", type=float, default=1.0, help="取数间隔秒（默认 1）")
    ap.add_argument("--history", type=int, default=3600, help="内存里保留多少个样本")
    ap.add_argument("--spectra-interval", type=float, default=5.0,
                    help="自相关谱的取数间隔秒，0 = 不取（默认 5）")
    ap.add_argument("--nic", default="ens2f0np0",
                    help="采样的网卡（硬件队列丢弃 / 软中断 / 流控）；空字符串 = 不采样")
    ap.add_argument("--nic-interval", type=float, default=5.0,
                    help="网卡层采样间隔秒（默认 5）")
    args = ap.parse_args()

    url, source = args.url, "--url"
    if not url:
        bind = read_bind(args.config)
        url = url_from_bind(bind) if bind else None
        source = f"{args.config} 的 monitoring.http.bind"
    if not url:
        url, source = DEFAULT_URL, "默认值（没读到配置）"
    # The two endpoints sit next to each other on the same server.
    spectra_url = url.rsplit("/", 1)[0] + "/autospectra.json"
    collector = Collector(url, spectra_url, args.interval, args.spectra_interval, args.history,
                          nic=args.nic, nic_interval=args.nic_interval)
    collector.start()
    host, port = split_bind(args.bind)
    Handler.collector = collector
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"corr-web: 页面 http://{host}:{port}/  取数 {url}（{source}），"
          f"每 {args.interval}s，保留 {args.history} 个样本", flush=True)
    if args.spectra_interval > 0:
        print(f"corr-web: 自相关谱取 {spectra_url}，每 {args.spectra_interval}s 一次",
              flush=True)
    if args.nic:
        print(f"corr-web: 网卡层采样 {args.nic}，每 {args.nic_interval}s 一次"
              f"（硬件队列丢弃 / 软中断 / 流控；只读）", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ncorr-web: 退出", flush=True)
    finally:
        collector.stop.set()
        httpd.server_close()


if __name__ == "__main__":
    main()
