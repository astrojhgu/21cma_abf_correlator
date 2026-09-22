#!/usr/bin/env bash
# A/B bench for the receive syscall: `recvmmsg` vs one `recvfrom` per packet.
#
# The shipped receive path calls `recvmmsg` with a batch of 32 and no
# MSG_WAITFORONE. Two questions follow from that, and they have different answers:
#
#   1. Is the batched syscall necessary at all? (does a plain `recvfrom` path keep
#      up, and what does it cost?)
#   2. If it costs something, is that the syscall, or something the batched call
#      was amortising by accident?
#
# So the default arms are three:
#
#   base     `recvmmsg`, batch 32   -- what ships today
#   looped   up to 32 `recvfrom` per call, filled by a loop
#   single   one `recvfrom` per call
#
# `looped` exists to separate the two questions. It hands the caller a batch of
# the same shape as `base`, so the caller's per-iteration work stays batched, but
# it does one syscall per packet like `single`. base -> looped is therefore the
# syscall alone; looped -> single is the batching shape alone, at a fixed
# syscall-per-packet ratio.
#
# Method (docs/baseline_test_sync.md section 2.2): this front end's delivery rate
# drifts over minutes, so two arms measured minutes apart are not comparable. The
# arms run in the same window, the order is rotated cyclically every round so that
# over a full cycle each arm occupies each position once, and the result is read
# off the sign of the paired difference rather than off any single run's value.
#
# Every run is `--dry-run`: receive, assemble and drive the watermark, but no GPU
# and no files. The question is what the receive syscall costs; letting the
# pipeline, four GPUs and /scratch into the measurement would add three sources of
# variance that have nothing to do with it.
#
# Usage:
#   tools/recvmmsg_ab.sh [rounds] [seconds-per-run]
#
#   # a single contrast instead of the default three arms
#   ARMS="base=bench/baseline-bin/corr single=bench/no-recvmmsg-bin/corr" \
#     tools/recvmmsg_ab.sh 6 60
#
#   # the direct-call build, with the intermediate one kept as a control that the
#   # API change itself is performance-neutral (docs/DESIGN-no-recvmmsg.md 5.1)
#   ARMS="base=bench/baseline-bin/corr single=bench/no-recvmmsg-bin/corr \
#         direct=bench/recvfrom-direct-bin/corr" tools/recvmmsg_ab.sh 6 60
#
#   # the readiness-driven backend, which needs its worker count
#   ARMS="base=bench/baseline-bin/corr-async \
#         direct=bench/recvfrom-direct-bin/corr-async" \
#     EXTRA_ARGS="--workers 24" tools/recvmmsg_ab.sh 6 60
#
# Read the result with tools/ab_report.py <output dir>.
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS=${1:-6}
DUR=${2:-60}
CFG=${CFG:-config/correlator-24-nf16384-g05.yaml}
NIC=${NIC:-ens2f0np0}

# arm=path, in the order they should be tried in round 0.
ARMS_SPEC=${ARMS:-"base=bench/baseline-bin/corr \
looped=bench/no-recvmmsg-looped-corr \
single=bench/no-recvmmsg-bin/corr"}

# Extra arguments passed to every arm, which is how the readiness-driven binary is
# driven: EXTRA_ARGS="--workers 24" tools/recvmmsg_ab.sh 6 60
EXTRA_ARGS=${EXTRA_ARGS:-}

NAMES=(); PATHS=()
for spec in $ARMS_SPEC; do
  NAMES+=("${spec%%=*}")
  PATHS+=("${spec#*=}")
done
N=${#NAMES[@]}
[ "$N" -ge 2 ] || { echo "need at least two arms; ARMS=$ARMS_SPEC" >&2; exit 2; }

for p in "${PATHS[@]}"; do [ -x "$p" ] || { echo "missing binary $p" >&2; exit 2; }; done
[ -e "$CFG" ] || { echo "missing config $CFG" >&2; exit 2; }

# Two receivers on the same groups steal each other's packets and cores, and the
# numbers would belong to neither. Refuse rather than produce them.
for other in corr corr-async; do
  if pgrep -x "$other" >/dev/null; then
    echo "a $other is already running; refusing to measure against it" >&2
    exit 2
  fi
done

OUT=bench/recvmmsg-ab/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"

nic_drops() { ethtool -S "$NIC" 2>/dev/null | awk '/rx_out_of_buffer/{print $2}'; }

run_arm() { # <index> <round>
  local i=$1 round=$2 arm=${NAMES[$1]} bin=${PATHS[$1]}
  local log="$OUT/r${round}-${arm}.log" d0 d1 t0 t1
  d0=$(nic_drops); t0=$(date +%s.%N)
  timeout $((DUR + 60)) /usr/bin/time -v "$bin" \
      --config "$CFG" --dry-run --duration "$DUR" $EXTRA_ARGS >"$log" 2>&1 || true
  t1=$(date +%s.%N); d1=$(nic_drops)
  {
    echo "arm=$arm binary=$bin round=$round duration=$DUR extra_args='$EXTRA_ARGS'"
    echo "wall_s=$(echo "$t1 - $t0" | bc)"
    # The condition both arms share: this card drops on its own at this rate, and a
    # comparison that ignored it would blame the receiver for the NIC's losses.
    echo "nic_out_of_buffer_delta=$((d1 - d0))"
  } >>"$log"
  echo "  [r$round $arm] $(grep -m1 'start frame' "$log" || echo 'no summary line')"
}

echo "output : $OUT"
echo "config : $CFG"
echo "arms   : $(for i in $(seq 0 $((N - 1))); do echo -n "${NAMES[$i]}=${PATHS[$i]}  "; done)"
echo "rounds : $ROUNDS x ${DUR}s per arm, order rotated cyclically"
if [ $((ROUNDS % N)) -ne 0 ]; then
  echo "note   : rounds is not a multiple of the arm count, so position balance is partial"
fi
echo

for ((r = 0; r < ROUNDS; r++)); do
  for ((k = 0; k < N; k++)); do
    run_arm $(((r + k) % N)) "$r"
  done
done

echo
echo "logs in $OUT"
echo "report: tools/ab_report.py $OUT"
