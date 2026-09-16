#!/bin/bash
# ab_zerofill.sh LABEL CORR_BIN [CORR_SECS] [REPLAY_SECS] [RATE]
#
# One leg of a zero-fill A/B: drive the same deterministic replay into one
# correlator binary and keep everything a judgement needs (blocks.csv, the run
# report, the load before and after).
#
# Run two legs per binary, interleaved A/B/A/B, and compare with
# tools/compare_blocks.py -- the machine's own load drifts on a ten-minute scale,
# which is the same order as a single run, so block-wise comparison drowns in it.
#
# Determinism, all four parts needed for two legs to be comparable row by row:
#   * --start-counter is fixed, so a given frame holds the same samples in both
#     legs (the replay is a function of the counter, not of a running state);
#   * the injected loss is a --gap-len/--gap-at counter hole, i.e. the fault a
#     receiver actually sees as a jump, rather than the machine's own loss;
#   * the replay starts first and the correlator a few seconds later, so both legs
#     sniff the same first counter and therefore align to the same start frame;
#   * the replay is paced against the wall clock, so those seconds are the same
#     number of packets every time.
#
# The replay must outlive the correlator's run: when the stream stops before
# --duration elapses the run hangs (both the old and the new binary do; see
# docs/DESIGN-zero-fill-decoupling.md section 8.7).
#
# Environment overrides: GAP_LEN, GAP_AT, STALL (e.g. STALL="--stall 3:3
# --stall-at 10"), PIPE_DEBUG=1 for the pipeline's 500 ms progress traces.
set -u

LABEL=${1:?usage: ab_zerofill.sh LABEL CORR_BIN [secs] [replay-secs] [rate]}
CORR_BIN=${2:?usage: ab_zerofill.sh LABEL CORR_BIN [secs] [replay-secs] [rate]}
CORR_SECS=${3:-40}
REPLAY_SECS=${4:-60}
RATE=${5:-20000}
GAP_LEN=${GAP_LEN:-40000}
GAP_AT=${GAP_AT:-500000}
STALL=${STALL:-}

HERE=$(cd "$(dirname "$0")/.." && pwd)
CFG=$HERE/config/correlator-24-replay.yaml
REPLAY=$HERE/corr-rs/target/release/corr-replay
OUT=${OUT_DIR:-/scratch/zerofill-ab}/$LABEL
mkdir -p "$OUT"
rm -rf "${OUT:?}"/*

{
    echo "load before: $(cat /proc/loadavg)"
} > "$OUT/load.txt"

pkill -x corr-replay 2>/dev/null
sleep 1

( cd "$HERE" && exec "$REPLAY" --config "$CFG" --pattern prng --rate "$RATE" \
    --gap-len "$GAP_LEN" --gap-at "$GAP_AT" $STALL \
    --duration "$REPLAY_SECS" ) > "$OUT/replay.log" 2>&1 &
REPLAY_PID=$!

sleep 6
( cd "$HERE" && CORR_PIPE_DEBUG=${PIPE_DEBUG:-} exec "$CORR_BIN" --config "$CFG" \
    --duration "$CORR_SECS" --output-dir "$OUT" ) > "$OUT/run.log" 2>&1
CORR_RC=$?
wait "$REPLAY_PID" 2>/dev/null

echo "load after: $(cat /proc/loadavg)" >> "$OUT/load.txt"
echo "== $LABEL (corr $CORR_BIN, rate $RATE, gap $GAP_LEN@$GAP_AT, ${CORR_SECS}s) rc=$CORR_RC =="
cat "$OUT/load.txt"
grep -E "  stall:|receive threads: .*staging slot" "$OUT/run.log"
ls -d "$OUT"/*/blocks.csv 2>/dev/null || echo "  no blocks.csv: the run refused to start or wrote nothing"
