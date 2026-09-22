#!/usr/bin/env bash
# Where a run's CPU actually goes, sampled from the kernel rather than inferred.
#
# `report.rs` can say how long a receive thread spent inside the receive call, but
# not whether that time was on-CPU or descheduled -- and the two call for opposite
# conclusions about whether `recvmmsg` is buying anything. This samples
# /proc/<pid>/task/*/stat (utime/stime per thread, in clock ticks), the context
# switch counters, and `ss -u -m` (the socket receive queue: the direct answer to
# "is the receiver behind the front end, or ahead of it?").
#
# Usage: tools/recv_probe.sh <binary> [extra args...]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=$1; shift
CFG=${CFG:-config/correlator-24-nf16384-g05.yaml}
HZ=$(getconf CLK_TCK)
DUR=${DUR:-40}

cpu_of() { # <pid> -- summed utime/stime over every thread
  local pid=$1 u=0 s=0 n=0 rest t
  for t in /proc/"$pid"/task/*/stat; do
    [ -r "$t" ] || continue
    # Field 2 is comm, which may contain spaces and parens: drop through the last
    # ')' so the remaining fields start at field 3.
    rest=$(sed 's/^[^)]*) //' "$t")
    u=$((u + $(echo "$rest" | awk '{print $12}')))
    s=$((s + $(echo "$rest" | awk '{print $13}')))
    n=$((n + 1))
  done
  echo "threads=$n user_s=$(echo "scale=2; $u / $HZ" | bc) sys_s=$(echo "scale=2; $s / $HZ" | bc)"
}

ctxt_of() { # <pid>
  local pid=$1 vol=0 invol=0 st
  for st in /proc/"$pid"/task/*/status; do
    [ -r "$st" ] || continue
    vol=$((vol + $(awk '/^voluntary_ctxt_switches/{print $2}' "$st")))
    invol=$((invol + $(awk '/^nonvoluntary_ctxt_switches/{print $2}' "$st")))
  done
  echo "voluntary_ctxt=$vol involuntary_ctxt=$invol"
}

recvq() { # the UDP receive queues, in bytes (skmem r), largest last
  ss -u -a -n -m 2>/dev/null | sed -n 's/.*skmem:(r\([0-9]*\),.*/\1/p' | sort -n |
    awk '{a[NR] = $1; s += $1}
         END {if (NR == 0) {print "recvq none"; exit}
              printf "recvq n=%d min=%d median=%d max=%d sum=%.1fMiB\n",
                     NR, a[1], a[int((NR + 1) / 2)], a[NR], s / 1048576}'
}

"$BIN" --config "$CFG" --dry-run --duration "$DUR" "$@" >/tmp/probe-run.log 2>&1 &
PID=$!
sleep 12
[ -d /proc/$PID ] || { echo "process exited early"; tail -5 /tmp/probe-run.log; exit 1; }
echo "t=12s $(cpu_of $PID) $(ctxt_of $PID)"
recvq
sleep 12
[ -d /proc/$PID ] && echo "t=24s $(cpu_of $PID) $(ctxt_of $PID)"
recvq
wait $PID || true
echo "--- report ---"
grep -E "recv calls|mean batch|receive threads" /tmp/probe-run.log
