#!/usr/bin/env python3
"""Summarise the multi-arm recvmmsg A/B logs.

Reads every `r<N>-<arm>.log` under a bench directory and prints, per metric, each
arm's value per round and then the paired difference for each pair of arms.

The paired difference is the result. This front end's delivery rate drifts over
minutes, so two arms measured minutes apart are not comparable
(docs/baseline_test_sync.md 2.2); the arms are rotated within the same window and
judged on the sign of the difference, not on any single run's value.

The steady-state columns are the ones to judge on: every run opens with a couple of
seconds in which the socket buffers fill before all the receive threads are up,
which is not what the arms differ by. Those seconds are dropped by skipping the
first STEADY_SKIP monitor lines, which are printed once per second.

    tools/ab_report.py bench/recvmmsg-ab/20260922-181046
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path
from statistics import mean

STEADY_SKIP = 4

MON = re.compile(
    r"mon\s+(\d+)r\s+\+([\d.]+)r/s\s+\|\s+zf\s+([\d.]+)%\s+\(win\s+([\d.]+)%\)"
    r"\s+\|\s+lost\s+\+(\d+)\s+\|\s+recv wait\s+(\d+)%\s+spread\s+([\d.]+)sl"
    r"\s+blocked\s+(\d+)\s+blocks\s+\+(\d+)\s+stalled\s+\+(\d+)"
    r"\s+\|\s+cpu\s+([\d.]+)/(\d+)c\s+load\s+([\d.]+)"
    r"\s+\|\s+h2d\s+([\d.]+)GB/s\s+(\d+)%\s+slotfree\s+(\d+)%"
    r"\s+\|\s+sockdrop\s+\+(\d+)\s+wdrop\s+\+(\d+)"
)
STATION = re.compile(
    r"^\s+(\w+)\s+pkts=(\d+)\s+gaps=(\d+)\s+lost=(\d+)\s+zero_filled=(\d+)"
    r"\s+late=(\d+)\s+before_start=(\d+)\s+recoveries=(\d+)\s+progress=(\d+)",
    re.M,
)
STALL = re.compile(r"worst read stall (\d+) ms, (\d+) stalls >= 20 ms")
RECV = re.compile(
    r"([\d.]+) s/station waiting for packets, ([\d.]+) s/station assembling them"
    r" \(of which ([\d.]+) s waiting for a staging slot\), (\d+)% of the run spent"
)
UTIME = re.compile(r"User time \(seconds\): ([\d.]+)")
STIME = re.compile(r"System time \(seconds\): ([\d.]+)")
# The arm/binary/round line is space-separated key=value pairs, so each key has to
# be matched on its own; a `^arm=(.*)$` would swallow the other two.
ARM = re.compile(r"\barm=(\S+)")
ROUND = re.compile(r"\bround=(\d+)")
WALLS = re.compile(r"^wall_s=(\S+)$", re.M)
NICD = re.compile(r"^nic_out_of_buffer_delta=(-?\d+)$", re.M)

# key, label, grouping
METRICS = [
    ("sockdrop_steady", "socket drops/s, steady (sum)", "pressure"),
    ("sockdrop_max", "socket drops/s, worst second", "pressure"),
    ("lost_steady", "packets lost, steady (sum)", "pressure"),
    ("zf", "frames zero filled (sum)", "pressure"),
    ("gaps", "counter gaps (sum)", "pressure"),
    ("lost_worst_station", "packets lost, worst station", "pressure"),
    ("cpu_proc_s", "process CPU, user+sys (s)", "price"),
    ("user_s", "  of which user (s)", "price"),
    ("sys_s", "  of which sys (s)", "price"),
    ("cpu_avg", "cores used (monitor mean)", "price"),
    ("recv_s", "s/station inside the recv call", "attribution"),
    ("asm_s", "s/station after the recv call", "attribution"),
    ("asm_pct", "% of the run spent assembling", "attribution"),
    ("slotwait_s", "s/station waiting for a slot", "attribution"),
    ("worst_stall_ms", "worst read stall (ms)", "liveness"),
    ("stalls20", "stalls >= 20 ms", "liveness"),
    ("rowrate_avg", "rows/s", "liveness"),
    ("pkts", "packets received (sum)", "liveness"),
    ("nic_out_of_buffer_delta", "NIC rx_out_of_buffer delta", "floor"),
    ("wall_s", "wall clock (s)", "floor"),
]


def parse(path: Path) -> dict:
    txt = path.read_text(errors="replace")
    d: dict = {}
    if m := ARM.search(txt):
        d["arm"] = m.group(1)
    if m := ROUND.search(txt):
        d["round"] = m.group(1)
    if m := WALLS.search(txt):
        d["wall_s"] = float(m.group(1))
    if m := NICD.search(txt):
        d["nic_out_of_buffer_delta"] = float(m.group(1))

    mons = [MON.search(l) for l in txt.splitlines() if l.startswith("mon ")]
    mons = [m for m in mons if m]
    steady = mons[STEADY_SKIP:] or mons
    g = lambda ms, i: [float(m.group(i)) for m in ms]  # noqa: E731
    d["n_mon"] = len(mons)
    d["sockdrop_steady"] = sum(g(steady, 17))
    d["sockdrop_all"] = sum(g(mons, 17))
    d["sockdrop_max"] = max(g(mons, 17)) if mons else 0.0
    d["lost_steady"] = sum(g(steady, 5))
    d["cpu_avg"] = mean(g(steady, 11)) if steady else 0.0
    d["ncpu"] = int(mons[0].group(12)) if mons else 0
    d["rowrate_avg"] = mean(g(steady, 2)) if steady else 0.0

    st = STATION.findall(txt)
    if st:
        d["n_station"] = len(st)
        for name, idx in (("pkts", 1), ("gaps", 2), ("lost", 3), ("zf", 4)):
            d[name] = sum(int(r[idx]) for r in st)
        d["lost_worst_station"] = max(int(r[3]) for r in st)

    if m := STALL.search(txt):
        d["worst_stall_ms"] = int(m.group(1))
        d["stalls20"] = int(m.group(2))
    if m := RECV.search(txt):
        d["recv_s"] = float(m.group(1))
        d["asm_s"] = float(m.group(2))
        d["slotwait_s"] = float(m.group(3))
        d["asm_pct"] = float(m.group(4))
    if m := UTIME.search(txt):
        d["user_s"] = float(m.group(1))
    if m := STIME.search(txt):
        d["sys_s"] = float(m.group(1))
    d["cpu_proc_s"] = d.get("user_s", 0.0) + d.get("sys_s", 0.0)
    return d


def fmt(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if x == int(x):
        return f"{int(x)}"
    return f"{x:,.3f}".rstrip("0").rstrip(".")


def verdict(diffs: list[float]) -> str:
    signs = {1 if x > 1e-9 else (-1 if x < -1e-9 else 0) for x in diffs}
    n = len(diffs)
    if signs == {0}:
        return "identical"
    if len(signs) == 1:
        pos = 1 in signs
        word = "higher" if pos else "lower"
        if 0 in signs:
            return f"{word}, ties present"
        return f"{word} on every round ({n}/{n})"
    pos = sum(1 for x in diffs if x > 0)
    return f"mixed ({pos}+ / {n - pos}-)"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    logs = sorted(root.glob("r*-*.log"))
    if not logs:
        print(f"no logs under {root}", file=sys.stderr)
        return 2

    rows = [(p.name, parse(p)) for p in logs]
    arms = sorted({d["arm"] for _, d in rows})
    if len(arms) < 2:
        print(f"need at least two arms, found {arms}", file=sys.stderr)
        return 2
    rounds = sorted({d["round"] for _, d in rows}, key=int)
    ncpu = next((d["ncpu"] for _, d in rows if d.get("ncpu")), 0)

    by: dict[tuple[str, str], dict] = {(d["round"], d["arm"]): d for _, d in rows}
    complete = [
        r for r in rounds if all((r, a) in by for a in arms)
    ]

    print(f"dir    : {root}")
    print(f"machine: {ncpu} cores   arms: {', '.join(arms)}")
    print(f"rounds : {len(complete)} complete of {len(rounds)}   "
          f"steady state = monitor lines after the first {STEADY_SKIP}")
    if len(complete) != len(rounds):
        print(f"  (incomplete rounds dropped: "
              f"{sorted(set(rounds) - set(complete), key=int)})")
    print()

    pairs = list(itertools.combinations(arms, 2))
    for group in ("pressure", "price", "attribution", "liveness", "floor"):
        title = {
            "pressure": "did the receiver keep up, and was the data right?",
            "price": "what did it cost?",
            "attribution": "where did the cost go?",
            "liveness": "was anything else disturbed?",
            "floor": "the conditions both arms shared",
        }[group]
        print(f"-- {title} " + "-" * max(0, 68 - len(title)))
        for key, label in [(k, l) for k, l, grp in METRICS if grp == group]:
            if not any(key in by[(r, a)] for r in complete for a in arms):
                continue
            print(f"{label}")
            for a in arms:
                vals = [by[(r, a)].get(key) for r in complete]
                print(f"    {a:<8} " + "  ".join(f"{fmt(v):>9}" if v is not None else "     n/a"
                                                 for v in vals))
            for a, b in pairs:
                diffs = []
                for r in complete:
                    va, vb = by[(r, a)].get(key), by[(r, b)].get(key)
                    if va is None or vb is None:
                        continue
                    diffs.append(vb - va)
                if not diffs:
                    continue
                med = sorted(diffs)[len(diffs) // 2]
                print(f"    {a} -> {b:<8} diffs "
                      f"{[fmt(x) for x in diffs]}  median {fmt(med)}   {verdict(diffs)}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
