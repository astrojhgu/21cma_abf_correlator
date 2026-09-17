#!/usr/bin/env python3
"""Compare the writer-thread arms across the paired rounds.

Reads every <arm>-r<n> file set under a directory and prints, per round, the
arms side by side: fabricated frames, the pipeline thread's own stall
accounting, the receive threads' wait for a staging slot, the writer thread's
own cost, and how far behind the rows ran. Per-round rows are what matters --
this machine swings between a state where the staging ring stays open and one
where it is closed for 100 s/station, and an arm average over a drifting
machine says nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from longrun_report import summarise  # noqa: E402


def ms_rows(d: Path):
    sets = sorted(p for p in d.glob("*rows_ms.bin"))
    if not sets:
        return []
    b = sets[-1].read_bytes()
    return [int.from_bytes(b[i : i + 8], "little") for i in range(0, len(b) - 7, 8)]


def from_log(log: Path):
    out = {"pipeline": None, "release_wait": None, "staging": None, "writer_ms": None}
    if not log.exists():
        return out
    t = log.read_text()
    if m := re.search(r"pipeline thread: ([\d.]+) s inside.*release-wait ([\d.]+) s", t):
        out["pipeline"], out["release_wait"] = float(m.group(1)), float(m.group(2))
    if m := re.search(r"of which ([\d.]+) s waiting for a staging slot", t):
        out["staging"] = float(m.group(1))
    if m := re.search(r"thread wrote \d+ rows, (\d+) ms/row of its own time", t):
        out["writer_ms"] = int(m.group(1))
    return out


def fmt(v, width, spec=".1f"):
    return f"{'--':>{width}}" if v is None else f"{v:{width}{spec}}"


def main() -> int:
    root = Path(sys.argv[1])
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["b", "n1", "n10"]
    rounds = sorted(
        {
            int(m.group(1))
            for p in root.iterdir()
            if p.is_dir() and (m := re.search(r"-r(\d+)$", p.name))
        }
    )
    print(
        f"{'round':>5} {'arm':>4} {'rows':>5} {'fabricated':>11} {'pipeline s':>10} "
        f"{'release-wait':>12} {'staging-wait':>12} {'writer ms/row':>13} "
        f"{'worst row gap':>13} {'span s':>7}"
    )
    for r in rounds:
        for arm in arms:
            tag = f"{arm}-r{r}"
            run_root = root / tag
            if not run_root.is_dir():
                continue
            d = run_root
            if not any(d.glob("*blocks.csv")) and (d / tag).is_dir():
                d = d / tag  # --run-name puts the set one level down
            s = summarise(d)
            if s is None:
                print(f"{r:>5} {arm:>4}   (nothing to read yet)")
                continue
            l = from_log(root / f"{tag}-run.log")
            ms = ms_rows(d)
            gaps = [b - a for a, b in zip(ms, ms[1:])]
            span = (ms[-1] - ms[0]) / 1000.0 if len(ms) > 1 else 0.0
            print(
                f"{r:>5} {arm:>4} {s['rows']:>5} {s['pct']:>10.4f}% "
                f"{fmt(l['pipeline'], 10)} {fmt(l['release_wait'], 12)} "
                f"{fmt(l['staging'], 12)} {fmt(l['writer_ms'], 13, 'd')} "
                f"{fmt(max(gaps) if gaps else None, 13, 'd')} {span:>7.1f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
