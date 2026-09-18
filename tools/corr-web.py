#!/usr/bin/env python3
"""corr-web —— corr 的独立只读监测界面。

corr 进程自己（`monitoring.http.enabled: true`）在一个独立线程上 serve 一份 JSON
快照：`GET http://127.0.0.1:3081/status.json`。本程序每秒取一次，自己攒历史，
再用一个页面把它画出来。分工是刻意的：

  * corr 只提供**当前**一份快照，接口小到可以一眼看完，且不在数据通路上
    （详见 docs/DESIGN-status-http.md）；
  * 历史、曲线、颜色、表格都是"看的人"的事，都在这里，重启 corr 不影响它。

依赖：只有 Python 3 标准库。不需要 root，不写 /scratch，不碰 corr 的输出目录。

用法::

    tools/corr-web.py                       # 取 127.0.0.1:3081，页面开在 127.0.0.1:8080
    tools/corr-web.py --url http://127.0.0.1:3099/status.json
    tools/corr-web.py --bind 0.0.0.0:8080   # 让别的机器也能看（内网可见，谨慎）
    tools/corr-web.py --interval 0.5 --history 7200

页面上的 `connected: false` 不是错误，是"corr 现在没在跑或没开接口"——它会把最后
一份已知快照继续留着，并显示已经断了多久，所以跑完一腿之后仍然能回看。
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_URL = "http://127.0.0.1:3081/status.json"
DEFAULT_BIND = "127.0.0.1:8080"

# 页面里要画的每条曲线：字段取自 /api/state 的 history 元素。
# (key, 标题, 单位, 归一化上限或 None=按数据自适应)
SERIES = [
    ("zf", "补零率（窗口）", "%", None),
    ("rate", "行率", "r/s", None),
    ("cpu", "本进程 CPU", "核", None),
    ("link", "H2D 链路", "%", 120.0),
    ("wait", "收包侧等环", "%", 100.0),
]


class Collector(threading.Thread):
    """每秒取一次 corr 的状态快照，留一份历史。"""

    def __init__(self, url, interval, history):
        super().__init__(daemon=True, name="corr-web-collector")
        self.url = url
        self.interval = interval
        self.lock = threading.Lock()
        self.samples = deque(maxlen=history)
        self.latest = None
        self.error = None
        self.received_at = None
        self.count = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                req = urllib.request.Request(self.url, headers={"Cache-Control": "no-store"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    snap = json.loads(resp.read().decode("utf-8"))
                with self.lock:
                    if snap.get("ready"):
                        self.latest = snap
                        self.samples.append(snap)
                    self.error = None
                    self.received_at = time.time()
                    self.count += 1
            except (urllib.error.URLError, OSError, ValueError) as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
            self.stop.wait(max(0.0, self.interval - (time.monotonic() - started)))

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
                }
                for s in self.samples
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
#verdict.ok { color:var(--ok); } #verdict.warn { color:var(--warn); } #verdict.bad { color:var(--bad); }
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
</style></head><body>
<header>
  <h1 id="run">corr 状态</h1>
  <span class="meta" id="conn"></span>
  <span class="meta" id="set"></span>
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
    <div class="card"><div class="k">H2D</div><div class="v" id="h2d">–</div>
      <div class="s" id="slotfree">–</div></div>
    <div class="card"><div class="k">CPU / load</div><div class="v" id="cpu">–</div>
      <div class="s" id="load">–</div></div>
    <div class="card"><div class="k">capacity</div><div class="v" id="verdict">–</div>
      <div class="s" id="limits">–</div></div>
  </div>

  <section><h2>曲线（本程序自己攒的）</h2><div class="charts" id="charts"></div></section>

  <section><h2>每站</h2>
    <table><thead><tr>
      <th>station</th><th>progress</th><th>zero-filled</th><th>lost</th>
      <th>等环 ms</th><th>静默 s</th>
    </tr></thead><tbody id="stations"></tbody></table>
  </section>
</main>
<script>
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
  last.textContent = `${fmt(cur)} ${s.unit}`;
  let max = s.max !== null ? s.max : Math.max(...vals);
  if (!(max > 0)) max = 1;
  const n = Math.max(hist.length - 1, 1);
  const pts = vals.map((v, i) => {
    const x = (i / n) * 600;
    const y = 78 - Math.min(Math.max(v / max, 0), 1) * 74;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  line.setAttribute("points", pts.join(" "));
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

  const v = document.getElementById("verdict");
  v.textContent = snap.capacity.bottleneck;
  v.className = snap.capacity.bottleneck === "ok" ? "ok"
      : (snap.capacity.bottleneck === "gpu_link" || snap.capacity.bottleneck === "process_cpu")
        ? "warn" : "bad";
  v.title = snap.capacity.detail;
  const lim = snap.capacity.limits;
  document.getElementById("limits").textContent =
      `门限：环 ${fmt(lim.ring_busy, 0)}% · 链路 ${fmt(lim.link, 0)}% · CPU ${fmt(lim.cpu, 0)}%`;

  const tb = document.getElementById("stations");
  tb.innerHTML = "";
  for (const st of snap.stations) {
    const tr = document.createElement("tr");
    const hot = st.silence_s > 2 || st.lost_packets > 0 && snap.recv.wait_pct > 50;
    tr.className = st.silence_s > 2 ? "hot" : (st.zero_filled_frames > 0 ? "warm" : "");
    tr.innerHTML = `<td>${st.name}</td><td>${st.progress_frames}</td>
      <td>${st.zero_filled_frames}</td><td>${st.lost_packets.toLocaleString()}</td>
      <td>${fmt(st.slot_wait_ms, 0)}</td><td>${fmt(st.silence_s, 1)}</td>`;
    if (hot) tr.title = "这一路的静默或等待明显偏高";
    tb.appendChild(tr);
  }

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
tick();
setInterval(tick, 1000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    collector = None
    server_version = "corr-web"

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler's naming)
        path = self.path.split("?", 1)[0]
        if path == "/":
            body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
        elif path == "/api/state":
            body = json.dumps(self.collector.state()).encode("utf-8")
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
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"corr 的状态接口（默认 {DEFAULT_URL}）")
    ap.add_argument("--bind", default=DEFAULT_BIND,
                    help=f"本程序的监听地址（默认 {DEFAULT_BIND}）")
    ap.add_argument("--interval", type=float, default=1.0, help="取数间隔秒（默认 1）")
    ap.add_argument("--history", type=int, default=3600, help="内存里保留多少个样本")
    args = ap.parse_args()

    collector = Collector(args.url, args.interval, args.history)
    collector.start()
    host, port = split_bind(args.bind)
    Handler.collector = collector
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"corr-web: 页面 http://{host}:{port}/  取数 {args.url} 每 {args.interval}s，"
          f"保留 {args.history} 个样本", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ncorr-web: 退出", flush=True)
    finally:
        collector.stop.set()
        httpd.server_close()


if __name__ == "__main__":
    main()
