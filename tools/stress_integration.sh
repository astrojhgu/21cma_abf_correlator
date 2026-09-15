#!/bin/bash
# Stress the write path at 24 stations with different integration lengths.
#
# Every row is the same size (300 baselines x 4096 bins x 8 B = 9.38 MiB); what
# changes is how often a row is produced. That is what turns a 9.8 MB/s trickle
# into a 937 MB/s stream, and it is also what multiplies the per-row costs that
# are invisible at one row per second: the D2H fetch, the row wait, and the fsync
# of every baseline file.
#
# usage: stress_integration.sh N_CUM ROWS_PER_FETCH FSYNC_EVERY_ROWS SECONDS [RUN_NAME]
set -u

N_CUM="$1"; RPF="$2"; FSYNC="$3"; SECS="$4"; NAME="${5:-stress-n${N_CUM}-r${RPF}-f${FSYNC}}"
BASE=/home/member/jhgu/src/correlator
OUT=/scratch/corr-stress
CFG=/tmp/correlator-n${N_CUM}-r${RPF}-f${FSYNC}.yaml

python3 - "$BASE" "$CFG" "$N_CUM" "$RPF" "$FSYNC" <<'PY'
import re, sys
base, out, n_cum, rpf, fsync = sys.argv[1:6]
s = open(f"{base}/config/correlator.yaml").read()
s = re.sub(r"output_dir: /scratch\b", "output_dir: /scratch/corr-stress", s)
s = re.sub(r"^  n_cum: \d+", f"  n_cum: {n_cum}", s, flags=re.M)
s = re.sub(r"^  rows_per_fetch: \d+", f"  rows_per_fetch: {rpf}", s, flags=re.M)
s = re.sub(r"^  fsync_every_rows: \d+", f"  fsync_every_rows: {fsync}", s, flags=re.M)
s = re.sub(r"enabled: false", "enabled: true", s)
open(out, "w").write(s)
PY

mkdir -p "$OUT"
rm -rf "$OUT/$NAME"

echo "########## n_cum=$N_CUM  rows_per_fetch=$RPF  fsync_every_rows=$FSYNC  ${SECS}s ##########"
grep -E "^  n_cum|^  rows_per_fetch|^  fsync_every_rows" "$CFG" | tr '\n' ' '; echo

cd "$BASE" || exit 1
pkill -9 -x corr 2>/dev/null; sleep 0.3

START=$(date +%s.%N)
./corr-rs/target/release/corr --config "$CFG" --duration "$SECS" --run-name "$NAME" > /tmp/stress-$NAME.log 2>&1 &
CP=$!

# sample the process and the output growth while it runs
: > /tmp/stress-$NAME.samples
for _ in $(seq 1 60); do
  sleep 2
  [ -d /proc/$CP ] || break
  R=$(awk '/VmRSS/{printf "%d", $2/1024}' /proc/$CP/status 2>/dev/null)
  F=$(ls /proc/$CP/fd 2>/dev/null | wc -l)
  D=$(du -sm "$OUT/$NAME" 2>/dev/null | cut -f1)
  echo "$(date +%s),$R,$F,$D" >> /tmp/stress-$NAME.samples
done
wait $CP 2>/dev/null
END=$(date +%s.%N)

WALL=$(python3 -c "print('%.2f' % ($END-$START))")
echo "--- 结果 ---"
grep -E "^plan " /tmp/stress-$NAME.log | sed 's/^/  /'
grep -E "^start frame|^  stall|^  pipeline|^  output|STOPPED|^  gpu" /tmp/stress-$NAME.log | sed 's/^/  /'
# The interesting per-station numbers: loss and fabricated frames at short
# integration are the first sign that the writer, not the link, is the limit.
grep -E "^  [A-Z][0-9]{2} " /tmp/stress-$NAME.log | head -3 | sed 's/^/  /'

python3 - "$OUT/$NAME" "$WALL" "$N_CUM" <<'PY'
import glob, os, sys
d, wall, n_cum = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
fps = 800_000_000 / 8192
period = n_cum / fps
fs = sorted(glob.glob(os.path.join(d, "baseline_*.bin")))
if not fs:
    print("  没有写出任何文件"); raise SystemExit
sizes = {os.path.getsize(f) for f in fs}
rows = min(sizes) // 32768
total = sum(os.path.getsize(f) for f in fs)
ms = os.path.getsize(os.path.join(d, "rows_ms.bin")) if os.path.exists(os.path.join(d, "rows_ms.bin")) else 0
blocks = 0; n_station = 0
bp = os.path.join(d, "blocks.csv")
if os.path.exists(bp):
    for line in open(bp):
        if not line.strip():
            continue
        blocks += 1
        n_station = max(n_station, int(line.split(",")[1]) + 1)
print(f"  文件: {len(fs)} 个, 大小{'一致' if len(sizes)==1 else '不一致: ' + str(sorted(sizes))}")
print(f"  实际: 墙钟 {wall:.1f} s, {rows} 行 (= {rows/wall:.2f} 行/s), {total/1e6:.1f} MB "
      f"(= {total/wall/1e6:.1f} MB/s)")
print(f"  期望: 行周期 {period*1e3:.3f} ms -> {1/period:.2f} 行/s, 应写 {wall/period:.0f} 行")
short = (wall / period - rows) / (wall / period) * 100
print(f"  差距: {'落后 %.1f%%' % short if short > 1 else '跟得上'}")
# blocks.csv 是每 station 每行一行，不是每 baseline 一行。
expect_blocks = rows * n_station
print(f"  一致性: rows_ms.bin={ms} B (应 {8*rows}), blocks.csv={blocks} 行 "
      f"(应 {rows}x{n_station}={expect_blocks})"
      f" -> {'一致' if ms == 8*rows and blocks == expect_blocks else '不一致'}")
PY
du -sh "$OUT/$NAME" 2>/dev/null | sed 's/^/  输出: /'
rm -rf "$OUT/$NAME"
