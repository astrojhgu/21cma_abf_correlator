#!/usr/bin/env python3
"""Look at the machine and say what is wrong with *this* box's pinning and NIC.

Why this exists: every machine has a different NUMA shape, a different number of
CPUs per node, and puts its NICs and GPUs in different places, so a hand-tuned
`recv_cpus` list is wrong on the next box -- and wrong *silently*, because the
symptom is a few percent of fabricated frames that only shows up in a long run.
Between them, the checks below cost most of a day on the machine this was written
for (HANDOFF 6.16):

  * the receive threads were pinned onto the same cores as the NIC's completion
    IRQs, which delays NAPI and makes the NIC drop packets in hardware;
  * the RX ring was 1024 of a possible 8192 descriptors, i.e. 0.23 ms of slack
    at the front end's rate, so "NAPI is a little late" means "packets are gone";
  * `SO_BUSY_POLL` silently fails with EPERM, so the one mitigation that would
    hide that latency was never in play;
  * another tenant on the box held 15 of the front end's 24 multicast
    memberships, which both explains a mysterious 11.3 GiB/s of idle traffic and
    means every measurement on this box carries a load we do not control.

Nothing here changes anything: it reads /sys, /proc, ethtool, and asks the
correlator binary what layout it resolved (so this never duplicates the resolver).

    tools/doctor.py --config config/correlator-24-nf8192.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

OK, BAD, WARN, INFO = "  ok ", " FAIL", " warn", " info"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def read(p: str) -> str:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return ""


def parse_cpulist(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                out.extend(range(int(a), int(b) + 1))
            except ValueError:
                pass
        else:
            try:
                out.append(int(part))
            except ValueError:
                pass
    return out


def node_map() -> dict[int, list[int]]:
    out = {}
    for d in sorted(Path("/sys/devices/system/node").glob("node*")):
        if not d.name[4:].isdigit():
            continue
        out[int(d.name[4:])] = parse_cpulist(read(str(d / "cpulist")))
    return out


def core_of(node_map_: dict[int, list[int]], cpu: int) -> int:
    for n, cpus in node_map_.items():
        if cpu in cpus:
            return n
    return -1


def resolved_layout(binary: str, cfg: str) -> tuple[list[int], list[str], int]:
    """Ask the binary: which cores will each station's receive thread use?"""
    out = sh([binary, "--config", cfg, "--print-plan"])
    cpus = [int(m) for m in re.findall(r"\scpu=(\d+)", out)]
    notes = re.findall(r"^cpu\s+:\s+(.*)$", out, re.M)
    # Each station line carries exactly one `cpu=N`, so the resolved cores *are*
    # the station count. Counting the `[i]` prefixes instead was wrong -- it also
    # matched other bracketed lines and reported 324 stations for 24.
    return cpus, notes, len(cpus)


def nic_irq_cpus(bdf: str) -> tuple[list[int], list[int], int, bool]:
    """Where this device's IRQs are *allowed* to run, and where they have run.

    The allowed set comes from `/proc/irq/<n>/smp_affinity_list` and is what
    decides whether a pinning layout can avoid NAPI; the historical counts in
    `/proc/interrupts` are cumulative since boot and say nothing about the layout
    in force now -- an earlier version of this check used them and flagged an
    overlap that no longer existed.
    """
    allowed: set[int] = set()
    seen: set[int] = set()
    lines = 0
    readable = True
    try:
        text = Path("/proc/interrupts").read_text().splitlines()
    except OSError:
        return [], [], 0, False
    for line in text:
        if bdf not in line:
            continue
        lines += 1
        irq = line.split(":", 1)[0].strip()
        mask = read(f"/proc/irq/{irq}/smp_affinity_list")
        if mask:
            allowed.update(parse_cpulist(mask))
        else:
            readable = False
        _, _, rest = line.partition(":")
        for i, tok in enumerate(rest.split()):
            try:
                if int(tok) > 0:
                    seen.add(i)
            except ValueError:
                break
    return sorted(allowed), sorted(seen), lines, readable


def rx_ring(iface: str) -> tuple[int, int]:
    txt = sh(["ethtool", "-g", iface])
    mx = cur = 0
    block = ""
    for line in txt.splitlines():
        if "maximums" in line:
            block = "max"
        elif "Current" in line:
            block = "cur"
        m = re.match(r"\s*RX:\s*(\d+)", line)
        if m:
            if block == "max":
                mx = int(m.group(1))
            elif block == "cur":
                cur = int(m.group(1))
    return cur, mx


def busy_poll_allowed() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, 46, 50)  # SO_BUSY_POLL
        return s.getsockopt(socket.SOL_SOCKET, 46) > 0
    except OSError:
        return False
    finally:
        s.close()


def foreign_memberships(iface: str) -> list[str]:
    """Front-end groups someone else on this box has joined."""
    out: list[str] = []
    cur = None
    try:
        text = Path("/proc/net/igmp").read_text().splitlines()
    except OSError:
        return out
    for line in text:
        m = re.match(r"^\d+\s+(\S+)\s+:", line)
        if m:
            cur = m.group(1)
            continue
        g = re.match(r"^\s+([0-9A-F]{8})", line)
        if g and cur == iface:
            h = g.group(1)
            ip = f"{int(h[6:8],16)}.{int(h[4:6],16)}.{int(h[2:4],16)}.{int(h[0:2],16)}"
            if ip.startswith("239.1."):
                out.append(ip)
    return out


def discards(iface: str, seconds: float) -> tuple[int, int]:
    def val(name: str) -> int:
        for line in sh(["ethtool", "-S", iface]).splitlines():
            if line.strip().startswith(name + ":"):
                try:
                    return int(line.split(":")[1])
                except ValueError:
                    return 0
        return 0

    d0, p0 = val("rx_discards_phy"), val("rx_packets_phy")
    time.sleep(seconds)
    d1, p1 = val("rx_discards_phy"), val("rx_packets_phy")
    return d1 - d0, p1 - p0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/correlator.yaml")
    ap.add_argument("--binary", default="corr-rs/target/release/corr")
    ap.add_argument("--sample", type=float, default=5.0, help="seconds for the idle drop sample")
    a = ap.parse_args()

    fails: list[str] = []
    cfg = Path(a.config)
    iface = ""
    for line in (cfg.read_text().splitlines() if cfg.is_file() else []):
        m = re.match(r"\s*interface:\s*(\S+)", line)
        if m:
            iface = m.group(1)
            break
    if not iface:
        iface = "ens2f0np0"

    print(f"doctor: config={a.config} interface={iface}")
    nodes = node_map()
    print(f"  NUMA: {len(nodes)} nodes -> " + ", ".join(f"n{n}:{c[0]}-{c[-1]}" for n, c in sorted(nodes.items())))

    # --- what the binary resolved ------------------------------------------
    cpus, notes, stations = resolved_layout(a.binary, a.config)
    if not cpus:
        print(f"{BAD} could not resolve the layout (is {a.binary} built?)")
        return 2
    print(f"  layout for {stations} stations: receive cores {sorted(set(cpus))}")
    for n in notes:
        print(f"         {n}")

    # --- NIC placement ------------------------------------------------------
    dev = Path(f"/sys/class/net/{iface}/device")
    bdf = os.path.basename(os.path.realpath(dev)) if dev.exists() else ""
    node_txt = read(f"{dev}/numa_node")
    nic_node = int(node_txt) if node_txt.lstrip("-").isdigit() else -1
    allowed, seen, irq_lines, readable = nic_irq_cpus(bdf) if bdf else ([], [], 0, True)
    print(f"  NIC {iface} bdf={bdf} numa_node={nic_node} irq_lines={irq_lines}")
    if allowed:
        print(f"      IRQ affinity allows cores {allowed[:16]}{'...' if len(allowed) > 16 else ''}")
    if seen:
        print(f"      historically took interrupts on {seen[:16]}{'...' if len(seen) > 16 else ''}")

    if allowed:
        overlap = sorted(set(cpus) & set(allowed))
        free = sorted(set(allowed) - set(cpus))
        if overlap and free:
            fails.append(
                f"the NIC's IRQs may run on {overlap}, which the receive threads also own, "
                f"while {free} is free. Pin them apart (root): "
                f"for i in /proc/irq/*/smp_affinity_list; do echo {free[0]} > $i; done, "
                f"or add the overlap to cpu.avoid_nic_interfaces. Measured cost of sharing: "
                f"1.61% vs 0.23% fabricated frames (HANDOFF 6.16)"
            )
            print(f"{BAD} IRQ-allowed cores overlap receive cores: {overlap} (free: {free})")
        elif overlap:
            print(f"{WARN} IRQs may run on any core, including the receive cores; only root can narrow that")
        else:
            print(f"{OK} receive cores avoid every core the NIC's IRQs may use")
    elif not readable:
        print(f"{WARN} cannot read /proc/irq/*/smp_affinity_list (root only on this kernel)")

    if nic_node >= 0 and any(core_of(nodes, c) == nic_node for c in cpus):
        same = [c for c in cpus if core_of(nodes, c) == nic_node]
        fails.append(
            f"receive threads {same} sit on the NIC's own NUMA node {nic_node}; that is the "
            f"node the driver allocates RX buffers on and where its IRQs run"
        )
        print(f"{WARN} receive cores on the NIC's node: {same}")
    else:
        print(f"{OK} receive cores are off the NIC's NUMA node")

    # --- SMT ----------------------------------------------------------------
    sib = {}
    for c in set(cpus):
        s = read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        sib[c] = parse_cpulist(s)
    both = [c for c in set(cpus) if any(o != c and o in cpus for o in sib.get(c, []))]
    if both:
        print(f"{WARN} SMT siblings both used by receive threads: {sorted(both)[:8]}")
    else:
        print(f"{OK} no SMT sibling is shared by two receive threads")

    # --- NIC / kernel knobs -------------------------------------------------
    cur, mx = rx_ring(iface)
    pps = len(cpus) * 195_312.5
    if mx and cur < mx:
        fails.append(
            f"RX ring is {cur} of {mx} descriptors "
            f"(~{1000 * cur / pps:.2f} ms of slack at this station count). "
            f"Raise it: ethtool -G {iface} rx {mx}"
        )
        print(f"{BAD} RX ring {cur}/{mx}")
    else:
        print(f"{OK} RX ring {cur}/{mx}")

    backlog = int(read("/proc/sys/net/core/netdev_max_backlog") or 0)
    if 0 < backlog < 100_000:
        fails.append(
            f"net.core.netdev_max_backlog is {backlog}; raise it "
            f"(sysctl -w net.core.netdev_max_backlog=250000)"
        )
        print(f"{WARN} netdev_max_backlog {backlog}")
    else:
        print(f"{OK} netdev_max_backlog {backlog}")

    if busy_poll_allowed():
        print(f"{OK} SO_BUSY_POLL is permitted")
    else:
        fails.append(
            "SO_BUSY_POLL is not permitted (needs CAP_NET_ADMIN); granting it lets the "
            "receive threads poll the NIC instead of sleeping, which hides the NAPI latency"
        )
        print(f"{WARN} SO_BUSY_POLL: EPERM")

    # --- who else is on our groups -----------------------------------------
    foreign = foreign_memberships(iface)
    running = bool(sh(["pgrep", "-x", "corr"]).strip())
    if foreign:
        print(
            f"{WARN} {len(foreign)} front-end group(s) are joined by something that is not "
            f"this run: {foreign[:6]}{'...' if len(foreign) > 6 else ''}"
        )
        print(
            "         That traffic is copied through the kernel on every measurement you "
            "take here, and it explains idle rates you did not ask for."
        )
    else:
        print(f"{OK} no other process holds front-end memberships")

    # --- idle drop baseline -------------------------------------------------
    if running:
        print(f"{INFO} a corr process is running; skipping the idle drop sample")
    else:
        dis, pk = discards(iface, a.sample)
        pct = 100.0 * dis / max(pk, 1)
        line = f"{dis:,} discarded of {pk:,} received in {a.sample:.0f}s ({pct:.3f}%)"
        if dis:
            print(f"{WARN} idle drop sample: {line}")
        else:
            print(f"{OK} idle drop sample: {line}")

    print()
    if fails:
        print(f"{len(fails)} thing(s) to fix:")
        for i, f in enumerate(fails, 1):
            print(f"  {i}. {f}")
        return 1
    print("no known misconfiguration found on this machine")
    return 0


if __name__ == "__main__":
    sys.exit(main())
