#!/usr/bin/env bash
# The whole phase-versus-frequency test, in one command, with fixed parameters.
#
# Why a script: two people (or two sessions) comparing results need to be running
# the same thing. Every number this test produces depends on the injector
# parameters below, on which binary is used, and on the analysis tool's version --
# an earlier copy of phase_triangle.py reported exactly half the delay, because the
# channel axis is the retained half of an R2C transform (fft_len = 2*nch). Any
# comparison that does not pin those down is comparing two different tests.
#
#   tools/phase_test.sh                       # defaults: 24 stations, stride 1, 50 kpps
#   RATE=20000 DUR=20 tools/phase_test.sh     # slower injector, shorter run
#   CORR=./corr-rs/target/release/corr-async CORR_ARGS="--workers 6" tools/phase_test.sh
#                                             # the same test through the async receive
#                                             # backend (docs/DESIGN-async-receive.md)
#
# Artifacts land in $OUT (default /scratch/phase-test-<timestamp>):
#   replay.log  corr.log  analysis.txt  triangle.png  <run dir>/
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-config/correlator-24-replay.yaml}
RATE=${RATE:-50000}          # packets/s/station; the real front end does 195312.5
DUR=${DUR:-25}               # seconds of correlation to keep (rows)
STRIDE=${STRIDE:-1}          # station i is delayed by i*STRIDE samples
OUT=${OUT:-/scratch/phase-test-$(date +%Y%m%d-%H%M%S)}
CORR=${CORR:-./corr-rs/target/release/corr}
# Extra arguments for $CORR, word-split on purpose: `corr` takes none, and
# `corr-async` needs `--workers W` (docs/DESIGN-async-receive.md section 7).
CORR_ARGS=${CORR_ARGS:-}
REPLAY=${REPLAY:-./corr-rs/target/release/corr-replay}

echo "config=$CFG rate=$RATE dur=$DUR stride=$STRIDE out=$OUT"
echo "corr=$CORR $CORR_ARGS"
mkdir -p "$OUT"

# One producer at a time: two would interleave their counters and the test would
# measure nothing. -x, not -f: -f matches the shell running this script.
pkill -x corr-replay 2>/dev/null || true
sleep 1
if pgrep -x corr-replay >/dev/null; then
  echo "a corr-replay is still running; refusing to start a second producer" >&2
  exit 1
fi

nohup "$REPLAY" --config "$CFG" --pattern prng --delay-stride "$STRIDE" \
      --rate "$RATE" --duration 70 > "$OUT/replay.log" 2>&1 &
sleep 5
head -1 "$OUT/replay.log"

# shellcheck disable=SC2086  # CORR_ARGS is meant to be word-split
"$CORR" --config "$CFG" --duration "$DUR" --output-dir "$OUT" $CORR_ARGS 2>&1 \
  | tee "$OUT/corr.log" | grep -E "^start frame|^  R01 |^  R24 |output:" || true
pkill -x corr-replay 2>/dev/null || true

python3 tools/phase_triangle.py "$OUT"/*/ --stride "$STRIDE" --mosaic phase \
        --out "$OUT/triangle.png" 2>&1 | tee "$OUT/analysis.txt"

echo
echo "artifacts in $OUT"
