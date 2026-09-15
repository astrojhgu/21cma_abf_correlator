#!/usr/bin/env python3
"""Measure what the front end is actually delivering to each station group.

Why this exists: a run that reports "stuck" can be stuck for two very different
reasons -- our receive path is throttled, or nobody is sending.  The two look
identical from inside the pipeline (the watermark stops moving), and they call
for opposite fixes.  The NIC aggregate is not enough either: 12 GB/s on the wire
says nothing about whether *our* 24 groups are still being delivered.

This joins the configured groups itself, counts packets for a few seconds, and
reports the frame-id span it saw.  It only reads; it never sends, so it cannot
collide with a producer.

    tools/mcast_rate.py config/correlator.yaml --seconds 4
"""

from __future__ import annotations

import argparse
import re
import socket
import struct
import sys
import time


def enabled_stations(
    path: str, include_disabled: bool = False
) -> list[tuple[str, str, int]]:
    """(name, group, port) for the stations in a correlator config.

    `include_disabled` is what answers "how many of the 40 stations the design is
    sized for are actually streaming": a board that is not in the enabled list is
    invisible to a run, but its group may still be carrying data, and the
    difference between "no board" and "board sending but not enabled" decides
    whether the 40-station target is a software problem or a hardware one.
    """
    out: list[tuple[str, str, int]] = []
    pat = re.compile(r"\{([^}]*)\}")
    for line in open(path):
        line = line.split("#", 1)[0]
        if "group_ip:" not in line:
            continue
        fields = dict(
            (m.group(1), m.group(2))
            for m in re.finditer(r"([A-Za-z_]+):\s*([^,}]+)", line)
        )
        if not include_disabled and fields.get("enabled", "false").strip() != "true":
            continue
        out.append(
            (
                fields.get("name", "?").strip(),
                fields["group_ip"].strip(),
                int(fields["dest_port"]),
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--iface", default=None, help="interface name, for its IPv4")
    ap.add_argument("--mbuf", type=int, default=8 << 20)
    ap.add_argument(
        "--include-disabled",
        action="store_true",
        help="probe every configured group, enabled or not",
    )
    a = ap.parse_args()

    stations = enabled_stations(a.config, a.include_disabled)
    if not stations:
        print(f"no enabled stations in {a.config}", file=sys.stderr)
        return 2

    iface = a.iface
    if iface is None:
        for line in open(a.config):
            m = re.search(r"^\s*interface:\s*(\S+)", line)
            if m:
                iface = m.group(1)
                break
    iface_ip = "0.0.0.0"
    if iface:
        # The config carries interface_ip, but reading the live address is more
        # honest: a stale config would otherwise silently join on the wrong one.
        import subprocess

        txt = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", iface], capture_output=True, text=True
        ).stdout
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", txt)
        if m:
            iface_ip = m.group(1)

    socks = []
    for name, group, port in stations:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, a.mbuf)
        except OSError:
            pass
        s.bind(("", port))
        mreq = socket.inet_aton(group) + socket.inet_aton(iface_ip)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        s.settimeout(0)
        socks.append((name, group, port, s))

    print(f"{len(socks)} groups on {iface} ({iface_ip}), {a.seconds:.0f} s")
    count = [0] * len(socks)
    first = [None] * len(socks)
    last = [None] * len(socks)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.seconds:
        for i, (_, _, _, s) in enumerate(socks):
            while True:
                try:
                    data = s.recv(8192)
                except BlockingIOError:
                    break
                count[i] += 1
                fid = struct.unpack_from("<Q", data, 0)[0] if len(data) >= 8 else 0
                if first[i] is None:
                    first[i] = fid
                last[i] = fid
        time.sleep(0.001)
    dt = time.monotonic() - t0

    total = 0
    print(f"{'station':8} {'pps(seen)':>10} {'counter/s':>10} {'span':>12} {'state':>6}")
    for i, (name, group, port, s) in enumerate(socks):
        span = (last[i] - first[i]) if (first[i] is not None and last[i] is not None) else 0
        # Two packets per frame at n_freq 4096, so frames/s is pps/2; the span is
        # how far the counter really walked, which is what a run cares about.
        rate = span / dt
        print(
            f"{name:8} {count[i] / dt:10.0f} {rate / 1000:9.1f}k {span:12d} "
            f"{'live' if rate > 1000 else 'DEAD':>6}"
        )
        total += count[i]
    print(f"total {total / dt / 1e6:.2f} Mpps, {total / dt * 4104 * 8 / 1e9:.1f} Gbit/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
