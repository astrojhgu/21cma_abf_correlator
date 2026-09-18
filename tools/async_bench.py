#!/usr/bin/env python3
"""A/B the thread-per-station receiver against the async one, interleaved.

What this measures, and why it is built the way it is:

* **The judgement is per-station loss, not a total.** The replayer sends a
  gapless packet counter, so every packet the receive path fails to place shows up
  as a counter gap (`lost=`) or as a frame declared missing (`zero_filled=`). A
  total over stations hides the interesting case -- the run that loses a little on
  every stream -- so both are printed, and the per-station maximum is what the
  verdict uses.
* **Legs are interleaved, never blocked.** This machine is shared and its load
  drifts on a ten-minute scale (`docs/ARCHITECTURE.md` section 5); a run of A for
  ten minutes and then B for ten minutes measures the machine, not the code. Every
  round runs each leg once, in the same order, and the load average is recorded
  with the result.
* **Three layers of loss are counted.** The kernel's own `UdpRcvbufErrors` and the
  NIC's `rx_discards_phy` are sampled around each leg, because a receive-side loss
  and a network-side loss call for opposite fixes and the receive path cannot tell
  them apart from inside.
* **Both legs run the same configuration.** The core pool is patched into the
  config (`cpu.recv_cpus`); `corr` then runs one thread per station round-robin
  over that pool, and `corr-async --workers W` runs W workers over the same pool.
  Nothing else differs -- same sockets, same assembler, same ring, same scheduler,
  same report -- which is what makes the difference attributable to the receiver.

Usage

    tools/async_bench.py --tag 24s-8c --stations 24 --pool 16-23 --rate 30000 \
        --duration 60 --rounds 3 --legs threads,async:8,async:4

    tools/async_bench.py --tag live-24s-8c --live --pool 16-23 \
        --duration 45 --rounds 2 --legs threads,async:8

Results are printed as a table and written to bench/<tag>/ (results.json,
results.csv, and the raw log of every leg).
"""

import argparse
import json
import os
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORR = os.path.join(ROOT, "corr-rs/target/release/corr")
CORR_ASYNC = os.path.join(ROOT, "corr-rs/target/release/corr-async")
REPLAY = os.path.join(ROOT, "corr-rs/target/release/corr-replay")
LIVE_CFG = os.path.join(ROOT, "config/correlator-24.yaml")
REPLAY_CFG = os.path.join(ROOT, "config/correlator-40-replay.yaml")
IFACE = "ens2f0np0"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True,
                          capture_output=True, text=True, **kw)


def parse_cpulist(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def fmt_cpulist(cpus):
    return ",".join(str(c) for c in cpus)


# ---------------------------------------------------------------------------
# Configuration: the same file for every leg, with only the core pool patched
# ---------------------------------------------------------------------------

def write_bench_config(path, stations, pool, live, recv_buffer, n_cum,
                       frames_per_slot=256, block_slots=8, ring_lead=None,
                       produce=False, replay_cpus="0-7"):
    """Derive a bench configuration from the repository's own replay profile.

    Deriving rather than writing one from scratch is deliberate: every field the
    real profiles were tuned for (socket buffer, ring lead, staging geometry)
    stays exactly as this project measured it, and only the two things a
    measurement has to control -- how many streams and how many cores -- are set
    here.
    """
    import yaml

    src = LIVE_CFG if live else REPLAY_CFG
    with open(src) as f:
        cfg = yaml.safe_load(f)

    if not live:
        kept = [s for s in cfg["stations"] if s.get("enabled")][:stations]
        for s in cfg["stations"]:
            s["enabled"] = s in kept
        assert len(kept) == stations, f"only {len(kept)} replay stations available"

    # The receive core pool is the knob: `corr` deals it round-robin to one thread
    # per station (sharing when there are more stations than cores), `corr-async`
    # to one worker per core.
    # The pipeline, writer and monitor threads must avoid the receive pool *and*
    # the replayer's cores: sharing a core with the injector makes the injector --
    # not the design under test -- decide how much traffic arrives, and that is
    # what a whole round of measurements was lost to.
    replay_set = set(parse_cpulist(replay_cpus))
    outside = [c for c in parse_cpulist(open("/sys/devices/system/cpu/online").read().strip())
               if c not in pool and c not in replay_set]
    cfg["cpu"]["recv_cpus"] = list(pool)
    cfg["cpu"]["auto_layout"] = False
    cfg["cpu"]["avoid_nic_interfaces"] = []
    # The pipeline, the writers and the monitor must not sit on a receive core.
    cfg["cpu"]["pipeline_cpu"] = outside[0]
    cfg["cpu"]["writer_cpus"] = outside[1:1 + cfg["storage"].get("writers", 2)]
    cfg["cpu"]["monitor_cpu"] = outside[1 + cfg["storage"].get("writers", 2)]

    # The zero-fill judgement needs `blocks.csv`, and `blocks.csv` is written by
    # the pipeline only when there is a device to fetch rows from: a dry run
    # releases each slot immediately and never opens an output file. So a
    # zero-fill comparison has to run the real thing (`--produce`), GPU included.
    cfg["run"]["dry_run"] = not produce
    cfg["run"]["name"] = None
    cfg["storage"]["daily_files"] = False
    cfg["receiver"]["recv_buffer_bytes"] = recv_buffer
    cfg["receiver"]["interface"] = IFACE
    cfg["integration"]["n_cum"] = n_cum
    cfg["gpu"]["frames_per_slot"] = frames_per_slot
    cfg["gpu"]["block_slots"] = block_slots
    if ring_lead is not None:
        cfg["gpu"]["ring_lead"] = ring_lead
    # No device work happens in a dry run; keep the split the profiles use so the
    # plan is identical in shape to a production one.
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# The three layers of loss
# ---------------------------------------------------------------------------

def snapshot(paths):
    out = {}
    for name, p in paths.items():
        try:
            with open(p) as f:
                out[name] = f.read()
        except OSError:
            out[name] = ""
    return out


def udp_drops(text):
    """`Udp: ... InErrors RcvbufErrors ...` from /proc/net/snmp.

    The header line's first field is `Udp:` -- with the colon. Comparing against
    `"Udp"` silently returned an empty dict, and the deltas that were computed
    from it were therefore always zero: a run that lost 600k packets a second in
    the kernel was reported as losing none. Every counter read out of a file has
    to be checked against a real file once.
    """
    lines = [l.split() for l in text.splitlines()]
    for hdr, val in zip(lines, lines[1:]):
        if hdr and hdr[0].rstrip(":") == "Udp":
            cols = {h: int(v) for h, v in zip(hdr[1:], val[1:])}
            return {"InErrors": cols.get("InErrors", 0),
                    "RcvbufErrors": cols.get("RcvbufErrors", 0),
                    "NoPorts": cols.get("NoPorts", 0)}
    return {}


def nic_discards(text):
    return int(text.strip() or 0)


def socket_drops(ports):
    """Kernel drops and backlog for *our* receiving sockets, from /proc/net/udp.

    The system-wide `Udp: RcvbufErrors` counter cannot be used for this: this host
    has resident tenants receiving their own multicast, and their drops land in
    the same counter -- a measured 1.5M per 25 s leg whether or not the run under
    test was keeping up. So the per-socket `drops` column is read instead, matched
    by the station's own port, together with `rx_queue` (how many packets are
    sitting unread, which is the receiver's distance behind the front end).

    The layout is not what the header suggests on every kernel: here `tx_queue`
    and `rx_queue` are two fields and `drops` is the last one, while other kernels
    print `tx:rx` as one. Parsed defensively against both, with `drops` taken from
    the end of the line, and `rx_queue` (bytes) reported as the receiver's
    distance behind the front end.
    """
    drops = 0
    queued = 0
    try:
        with open("/proc/net/udp") as f:
            next(f)
            for line in f:
                parts = line.split()
                if len(parts) < 13:
                    continue
                try:
                    port = int(parts[1].split(":")[1], 16)
                except (IndexError, ValueError):
                    continue
                if port in ports:
                    rxq = parts[4]
                    if ":" in rxq:
                        rxq = rxq.split(":")[1]
                    else:
                        rxq = parts[5]
                    queued += int(rxq, 16)
                    drops += int(parts[-1])
    except OSError:
        pass
    return drops, queued


def loadavg():
    return os.getloadavg()[0]


def children_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


# ---------------------------------------------------------------------------
# One leg
# ---------------------------------------------------------------------------

RE_STATION = re.compile(
    r"^\s+(?P<name>\S+)\s+pkts=(?P<pkts>\d+)\s+gaps=(?P<gaps>\d+)\s+"
    r"lost=(?P<lost>\d+)\s+zero_filled=(?P<zf>\d+)\s+late=(?P<late>\d+)\s+"
    r"before_start=(?P<before>\d+).*progress=(?P<progress>\d+)"
)
RE_ROWS = re.compile(r"^\s+output:\s+(?P<rows>\d+) rows")


def parse_report(log):
    st = []
    rows = None
    start_frame = None
    m = re.search(r"^start frame (\d+)", log, re.M)
    if m:
        start_frame = int(m.group(1))
    for line in log.splitlines():
        m = RE_STATION.match(line)
        if m:
            st.append({k: (int(v) if k != "name" else v) for k, v in m.groupdict().items()})
        m = RE_ROWS.match(line)
        if m:
            rows = int(m.group("rows"))
    stalls = 0
    m = re.search(r"stall:\s+(\d+) recoveries", log)
    if m:
        stalls = int(m.group(1))
    ring = None
    m = re.search(r"pipeline:\s+(\d+) ring block episode", log)
    if m:
        ring = int(m.group(1))
    return st, rows, stalls, ring, start_frame


def blocks_stats(run_dir):
    """Per-row zero fill from a run's blocks.csv: the judgement this project uses.

    The run report's per-station total hides the failure that matters -- a station
    that loses a little on every row (docs/ARCHITECTURE.md section 5) -- so the
    numbers here are per row: how many rows came out whole, what the worst row
    and the worst station did, and whether the rate is drifting between the first
    and the second half of the run.
    """
    import csv
    import glob
    files = glob.glob(os.path.join(run_dir, "*", "*blocks.csv"))
    if not files:
        return None
    path = max(files, key=os.path.getmtime)
    per_row = {}
    per_station = {}
    total_frames = 0
    total_zf = 0
    with open(path) as f:
        for rec in csv.reader(f):
            if len(rec) < 6 or not rec[0].isdigit():
                continue
            try:
                row, station, n_frames, zf = (int(rec[0]), int(rec[1]),
                                              int(rec[3]), int(rec[5]))
            except ValueError:
                continue
            per_row[row] = per_row.get(row, 0) + zf
            per_station[station] = per_station.get(station, 0) + zf
            total_frames += n_frames
            total_zf += zf
    if not per_row:
        return None
    rows = sorted(per_row)
    clean = sum(1 for r in rows if per_row[r] == 0)
    half = max(1, len(rows) // 2)
    first = [per_row[r] for r in rows[:half]]
    second = [per_row[r] for r in rows[half:]]
    frames_first = total_frames * len(first) // max(1, len(rows))
    frames_second = total_frames - frames_first
    return {
        "blocks_csv": path,
        "rows": len(rows),
        "rows_clean": clean,
        "rows_with_zf": len(rows) - clean,
        "frames_total": total_frames,
        "zf_total": total_zf,
        "zf_rate": round(total_zf / total_frames, 6) if total_frames else 0,
        "zf_worst_row": max(per_row.values()),
        "zf_worst_station": max(per_station.values()) if per_station else 0,
        "zf_rate_first_half": round(sum(first) / frames_first, 6) if frames_first else 0,
        "zf_rate_second_half": round(sum(second) / frames_second, 6) if frames_second else 0,
    }


def run_leg(tag, name, cmd, cfg, replay_rate, duration, replay_cpus, outdir,
            offered_pps=None, ports=(), out_dir_leg=None, packets_per_frame=None):
    """Run one leg with the replayer already flowing; return its measurements."""
    # The NIC's own drop counters. `rx_discards_phy` does not exist on this
    # interface and reading it silently produced a zero for every run; what this
    # NIC exposes is `rx_missed_errors`, the hardware receive-ring overflow
    # counter, which is the one that matters here -- the ring is 1024 descriptors
    # deep, roughly 0.22 ms of traffic at the front end's rate, so a receiver that
    # stops polling for a moment loses packets *in hardware* where no socket
    # counter can see them.
    paths = {
        "udp": "/proc/net/snmp",
        "missed": f"/sys/class/net/{IFACE}/statistics/rx_missed_errors",
        "dropped": f"/sys/class/net/{IFACE}/statistics/rx_dropped",
        "fifo": f"/sys/class/net/{IFACE}/statistics/rx_fifo_errors",
    }
    before = snapshot(paths)
    sock_before = socket_drops(ports)
    cpu_before = children_cpu()
    load_before = loadavg()

    log_path = os.path.join(outdir, f"{tag}-{name}.log")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    cpu = children_cpu() - cpu_before

    after = snapshot(paths)
    sock_after = socket_drops(ports)
    load_after = loadavg()
    with open(log_path) as f:
        log = f.read()

    u0, u1 = udp_drops(before["udp"]), udp_drops(after["udp"])
    stations, rows, stalls, ring, start_frame = parse_report(log)
    lost = [s["lost"] for s in stations]
    zf = [s["zf"] for s in stations]
    pkts = [s["pkts"] for s in stations]
    res = {
        "leg": name,
        "tag": tag,
        "cmd": " ".join(cmd),
        "exit": proc.returncode,
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "cpu_per_stream": round(cpu / max(1, len(stations)), 4),
        # Microseconds of process CPU per delivered packet, the number that says
        # what a design costs per byte rather than per stream.
        "cpu_us_per_pkt": (round(cpu * 1e6 / (statistics.mean(pkts) * len(stations)), 3)
                           if pkts and statistics.mean(pkts) else 0),
        "load_before": round(load_before, 2),
        "load_after": round(load_after, 2),
        "stations": len(stations),
        "rows": rows,
        "stall_recoveries": stalls,
        "ring_blocks": ring,
        "pkts_min": min(pkts) if pkts else 0,
        "pkts_max": max(pkts) if pkts else 0,
        "pkts_mean": round(statistics.mean(pkts), 1) if pkts else 0,
        # What each station actually received per second over the run. The
        # replayer's *requested* rate is a flag; this is the rate the machine
        # delivered, and it is the number the two legs have to agree on before
        # their loss figures can be compared at all.
        "achieved_pps": round(statistics.mean(pkts) / duration, 1) if pkts else 0,
        # What the receive path *failed to deliver*. The assembler's own `lost`
        # only counts counter gaps it actually saw, which is not the same thing:
        # a receiver that never catches up to the point where the kernel started
        # dropping never observes the gap, and reports a clean run while losing
        # most of the stream. The kernel's counter and the replayer's own count are
        # what make that visible, so all three are reported side by side.
        "offered_pps": offered_pps,
        "received_pps": round(statistics.mean(pkts) / duration, 1) if pkts else 0,
        "delivered_frac": (round(statistics.mean(pkts) / duration / offered_pps, 4)
                           if (pkts and offered_pps) else None),
        # What the *injector* actually offered, measured from the run itself: the
        # replayer is CPU-limited and shares this machine, so `--rate` is a
        # request, not a fact. The counters the receive path saw are the fact:
        # (progress advanced) x (packets per frame) / seconds.
        "offered_measured_pps": (
            round((statistics.mean([s["progress"] for s in stations]) - start_frame)
                  * packets_per_frame / duration, 1)
            if (stations and start_frame and packets_per_frame) else None),
        "lost_total": sum(lost),
        "lost_max": max(lost) if lost else 0,
        "lost_per_station_mean": round(statistics.mean(lost), 1) if lost else 0,
        "zero_filled_total": sum(zf),
        "zero_filled_max": max(zf) if zf else 0,
        # Our own sockets only: this is the number that says the receive path
        # could not keep up (as opposed to the host's other tenants dropping their
        # own traffic into the same system-wide counter).
        "socket_drops": sock_after[0] - sock_before[0],
        "socket_rx_queue_after": sock_after[1],
        "udp_rcvbuf_errors": u1.get("RcvbufErrors", 0) - u0.get("RcvbufErrors", 0),
        "udp_in_errors": u1.get("InErrors", 0) - u0.get("InErrors", 0),
        "nic_rx_missed": nic_discards(after["missed"]) - nic_discards(before["missed"]),
        "nic_rx_dropped": nic_discards(after["dropped"]) - nic_discards(before["dropped"]),
        "nic_rx_fifo": nic_discards(after["fifo"]) - nic_discards(before["fifo"]),
        "log": log_path,
    }
    if out_dir_leg:
        stats = blocks_stats(out_dir_leg)
        if stats:
            res.update(stats)
        else:
            res["blocks_csv"] = None
    return res


def start_replay(cfg, rate, seconds, cpus):
    # `--pattern zero` on purpose: this is a throughput measurement, and generating
    # the synthetic samples (`prng` is a splitmix64 per sample, ~8 us per 4096-byte
    # packet) costs the replayer more CPU than the send itself. At 30 kpps/station
    # that would put the replayer -- not the correlator -- at the limit of the 8
    # cores it is given, and the delivered rate would then depend on the machine's
    # load instead of on the flag. The receive path copies 4096 bytes either way.
    cmd = ["taskset", "-c", fmt_cpulist(cpus), REPLAY, "--config", cfg,
           "--pattern", "zero", "--rate", str(rate), "--duration", str(seconds)]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p


def stop_replay():
    subprocess.run(["pkill", "-x", "corr-replay"], capture_output=True)
    time.sleep(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stations", type=int, default=24)
    ap.add_argument("--pool", default="16-23",
                    help="receive core pool, patched into cpu.recv_cpus")
    ap.add_argument("--rate", type=float, default=30000.0, help="packets/s/station")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--legs", default="threads,async:8",
                    help="comma-separated: `threads` and/or `async:W`")
    ap.add_argument("--produce", action="store_true",
                    help="run the real GPU+writer path so blocks.csv exists; "
                         "this is what a zero-fill comparison needs")
    ap.add_argument("--live", action="store_true",
                    help="use the live front end's 24 streams; no replayer")
    ap.add_argument("--replay-cpus", default="0-7")
    ap.add_argument("--recv-buffer", type=int, default=268435456)
    ap.add_argument("--n-cum", type=int, default=8192)
    ap.add_argument("--block-slots", type=int, default=8)
    ap.add_argument("--ring-lead", type=int, default=None)
    ap.add_argument("--bench-dir", default=os.path.join(ROOT, "bench"))
    args = ap.parse_args()

    outdir = os.path.join(args.bench_dir, args.tag)
    os.makedirs(outdir, exist_ok=True)
    pool = parse_cpulist(args.pool)
    replay_cpus = parse_cpulist(args.replay_cpus)

    cfg = write_bench_config(os.path.join(outdir, "bench.yaml"), args.stations,
                             pool, args.live, args.recv_buffer, args.n_cum,
                             block_slots=args.block_slots, ring_lead=args.ring_lead,
                             produce=args.produce, replay_cpus=args.replay_cpus)

    import yaml as _yaml
    with open(cfg) as f:
        _cfg = _yaml.safe_load(f)
    ports = {int(s["dest_port"]) for s in _cfg["stations"] if s.get("enabled")}
    packets_per_frame = 2 * int(_cfg["fft"]["n_freq"]) // 4096
    print(f"our ports  : {len(ports)} socket(s) watched for kernel drops")

    legs = []
    for spec in args.legs.split(","):
        spec = spec.strip()
        if spec == "threads":
            legs.append(("threads", [CORR, "--config", cfg, "--duration",
                                     str(args.duration)]))
        elif spec.startswith("async:"):
            w = spec.split(":", 1)[1]
            legs.append((f"async{w}", [CORR_ASYNC, "--config", cfg, "--duration",
                                       str(args.duration), "--workers", w]))
        else:
            raise SystemExit(f"unknown leg {spec!r}")

    print(f"tag        : {args.tag}")
    print(f"config     : {cfg}")
    print(f"stations   : {args.stations}  pool={args.pool}  legs={[l[0] for l in legs]}")
    print(f"duration   : {args.duration}s x {args.rounds} rounds"
          f"  rate={'live' if args.live else f'{args.rate:.0f} pps/station'}")
    print(f"replay cpus: {args.replay_cpus}")
    print()

    results = []
    replay = None
    t_start = time.time()
    try:
        if not args.live:
            stop_replay()
            replay_seconds = (args.duration + 40) * args.rounds + 60
            replay = start_replay(cfg, args.rate, replay_seconds, replay_cpus)
            time.sleep(5)

        for rnd in range(1, args.rounds + 1):
            for name, cmd in legs:
                # A replayer whose streams stop early hangs the correlator
                # (HANDOVER section 3): the replayer is started with a duration that
                # covers every round, and checked here rather than assumed.
                if not args.live and replay.poll() is not None:
                    replay = start_replay(cfg, args.rate, (args.duration + 40) + 60,
                                          replay_cpus)
                    time.sleep(5)
                print(f"--- round {rnd}/{args.rounds}: {name}", flush=True)
                leg_out = None
                if args.produce:
                    leg_out = os.path.join(outdir, "runs", f"r{rnd}-{name}")
                    cmd = cmd + ["--output-dir", leg_out]
                res = run_leg(args.tag, f"r{rnd}-{name}", cmd, cfg, args.rate,
                              args.duration, replay_cpus, outdir,
                              offered_pps=None if args.live else args.rate,
                              ports=ports, out_dir_leg=leg_out,
                              packets_per_frame=packets_per_frame)
                res["round"] = rnd
                results.append(res)
                frac = res["delivered_frac"]
                if res.get("blocks_csv"):
                    print(f"    rows={res['rows']:>4} clean={res['rows_clean']:>4}"
                          f"  ZF RATE={res['zf_rate'] * 100:.4f}%"
                          f"  worst_row={res['zf_worst_row']:>6}"
                          f"  worst_station={res['zf_worst_station']:>7}"
                          f"  1st/2nd half={res['zf_rate_first_half'] * 100:.4f}%/"
                          f"{res['zf_rate_second_half'] * 100:.4f}%", flush=True)
                print(f"    offered(measured)={res['offered_measured_pps']}"
                      f"  lost_to_gaps={res['lost_total']}"
                      f"  ring_blocks={res['ring_blocks']}", flush=True)
                print(f"    pps/station={res['received_pps']:>9.0f}"
                      f" delivered={'' if frac is None else f'{frac * 100:5.1f}%'}"
                      f"  gaps_seen={res['lost_total']:>7}  zf={res['zero_filled_total']:>7}"
                      f"  OUR_DROPS={res['socket_drops']:>9}"
                      f"  nic_missed={res['nic_rx_missed']:>9}"
                      f"  cpu={res['cpu_s']:>6.2f}s ({res['cpu_us_per_pkt']:.2f}us/pkt)"
                      f"  load={res['load_before']:.1f}->{res['load_after']:.1f}",
                      flush=True)
    finally:
        if replay is not None:
            replay.terminate()
        stop_replay()

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    keys = ["tag", "round", "leg", "stations", "offered_pps", "offered_measured_pps",
            "ring_blocks", "received_pps",
            "delivered_frac", "cpu_us_per_pkt", "socket_drops",
            "socket_rx_queue_after", "nic_rx_missed", "nic_rx_dropped", "lost_total", "lost_max",
            "rows", "rows_clean", "rows_with_zf", "zf_total", "zf_rate", "zf_worst_row",
            "zf_worst_station", "zf_rate_first_half", "zf_rate_second_half",
            "lost_per_station_mean", "zero_filled_total", "cpu_s",
            "cpu_per_stream", "rows", "stall_recoveries", "ring_blocks",
            "udp_rcvbuf_errors", "udp_in_errors", "nic_rx_fifo",
            "load_before", "load_after", "wall_s"]
    with open(os.path.join(outdir, "results.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print()
    print("=== summary (mean over rounds) ===")
    print(f"{'leg':<10} {'pps/station':>11} {'delivered':>10} {'rows':>5} {'clean':>6} "
          f"{'ZF rate':>9} {'worst row':>10} {'OUR DROPS':>10} {'us/pkt':>7} {'load':>6}")
    for name, _ in legs:
        rs = [r for r in results if r["leg"].replace("r%d-" % r["round"], "") == name]
        if not rs:
            continue
        fracs = [r["delivered_frac"] for r in rs if r["delivered_frac"] is not None]
        zf = [r for r in rs if r.get("blocks_csv")]
        col_pps = f"{statistics.mean(r['received_pps'] for r in rs):>11.0f}"
        col_frac = (f"{statistics.mean(fracs) * 100:9.1f}%" if fracs else "        -")
        col_rows = f"{statistics.mean(r['rows'] for r in zf):>5.0f}" if zf else "    -"
        col_clean = f"{statistics.mean(r['rows_clean'] for r in zf):>6.0f}" if zf else "     -"
        col_zf = (f"{statistics.mean(r['zf_rate'] for r in zf) * 100:8.4f}%" if zf
                  else "        -")
        col_worst = (f"{max(r['zf_worst_row'] for r in zf):>10.0f}" if zf else "         -")
        col_drops = f"{statistics.mean(r['socket_drops'] for r in rs):>10.0f}"
        col_us = f"{statistics.mean(r['cpu_us_per_pkt'] for r in rs):>7.2f}"
        col_load = f"{statistics.mean(r['load_before'] for r in rs):>6.1f}"
        print(f"{name:<10} {col_pps} {col_frac} {col_rows} {col_clean} {col_zf} "
              f"{col_worst} {col_drops} {col_us} {col_load}")

    print()
    print(f"artifacts in {outdir}")
    print(f"total wall time: {time.time() - t_start:.0f} s over "
          f"{len(results)} leg(s)")

    if not args.live:
        stop_replay()
    return 0


if __name__ == "__main__":
    sys.exit(main())
