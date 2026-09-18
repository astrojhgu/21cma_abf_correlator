#!/usr/bin/env bash
# tools/run-24-nf16384.sh —— 手动跑一腿 24 路 / 16384 信道 / 1 s 积分
#
# 为什么要有这个脚本：手动跑这件事有四个固定步骤，而每一步都有踩过的坑 ——
#
#   1. **起跑前确认没有别的 corr / corr-async 在跑**。两个接收端同时收同一批组会互相
#      抢核、把补零率抬上去，测出来的数字既不属于谁也无法复现（文档里"两个生产端会
#      交错计数器"说的是发送端；接收端同样不能并存）。
#   2. **确认四张卡上没有别的 CUDA 进程**。GPU 被占时 H2D 变慢 → 管线线程拿不到槽释放
#      → 接收侧被拒槽 → socket 溢出 → 包丢在这里，而不是丢在网卡或前端。
#   3. **输出目录**。本档案把它固定在 `/scratch/jhgu/corr/`：配置里
#      `run.output_dir: /scratch/jhgu` + `run.name: corr`，合起来就是它，
#      **不再套一层运行目录**（默认档案会生成 `corr_YYYYmmdd_HHMMSS/`）。
#      代价：同一个本地日、同一份配置的第二次运行会**续写**当日那一套（daily_files
#      默认开）——这就是"固定目录"的用法；要另起一套加 `NEW_SET=1`。
#   4. **判读按逐行 `blocks.csv`**，不看运行报告的 per-station 汇总（汇总会把
#      "每行涓流丢一点"藏成一个很小的数）。跑完自动跑 `longrun_report.py`。
#
# 用法（默认就是异步后端，24 个 worker，300 s）：
#
#   tools/run-24-nf16384.sh                  # 异步，300 s
#   DUR=60 tools/run-24-nf16384.sh           # 异步，60 s
#   DUR=0  tools/run-24-nf16384.sh           # 一直跑到 Ctrl-C（会打印报告再退出）
#   MODE=sync tools/run-24-nf16384.sh        # 对照腿：线程后端
#   WORKERS=12 tools/run-24-nf16384.sh       # 收包 worker 数（异步腿）
#   NEW_SET=1 tools/run-24-nf16384.sh        # 今天已有套时另起一套，不续写
#   DOCTOR=1 tools/run-24-nf16384.sh         # 起跑前先跑 tools/doctor.py（慢，但值得）
#   TREND=1  tools/run-24-nf16384.sh         # 收尾再打一份逐行明细
#   DRY=1    tools/run-24-nf16384.sh         # 只打印将要执行的命令，不动机器
#
# 前置检查失败的退出码是 2，运行本身的退出码原样传出。

set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-config/correlator-24-nf16384-manual.yaml}
MODE=${MODE:-async}                 # async | sync
DUR=${DUR:-300}                     # 秒；0 = 不加 --duration，跑到 Ctrl-C
WORKERS=${WORKERS:-24}              # 异步腿的 worker 数；结论：按负载定，别照抄核数
NEW_SET=${NEW_SET:-0}
DOCTOR=${DOCTOR:-0}
TREND=${TREND:-0}
DRY=${DRY:-0}

case "$MODE" in
  async) BIN=corr-rs/target/release/corr-async; ARGS=(--workers "$WORKERS") ;;
  sync)  BIN=corr-rs/target/release/corr;       ARGS=() ;;
  *) echo "MODE 只能是 async 或 sync（现在是 '$MODE'）" >&2; exit 2 ;;
esac
[ -x "$BIN" ] || { echo "$BIN 不存在：先 cd corr-rs && cargo build --release" >&2; exit 2; }

# 二进制比源码旧就说话。踩过一次：改完命名后为了验证某个提交点 `git stash` 编译过，
# 源码恢复后忘了重建，于是脚本兴高采烈地跑了一个旧二进制、产出旧命名 —— 而报告看起来
# 完全正常。这里只警告不自动重建（重建是这个脚本之外的事，而且可能几秒到几分钟）。
newer=$(find corr-rs/src corr-rs/Cargo.toml -newer "$BIN" -print -quit 2>/dev/null || true)
if [ -n "$newer" ]; then
  echo "注意：$newer 比 $BIN 新 —— 现在跑的是旧二进制；先 (cd corr-rs && cargo build --release)" >&2
fi

# 输出目录从配置里读，不在脚本里再写一份：两份定义迟早会漂移。
OUT=$(python3 - "$CFG" <<'PY'
import sys, yaml
run = yaml.safe_load(open(sys.argv[1]))["run"]
name = run.get("name") or ""
print(run["output_dir"].rstrip("/") + ("/" + str(name).strip("/") if name else ""))
PY
)
[ -n "$DUR" ] && [ "$DUR" != "0" ] && ARGS+=(--duration "$DUR")
[ "$NEW_SET" = "1" ] && ARGS+=(--allow-second-set)

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$OUT/$MODE-$STAMP.log"

echo "config  : $CFG"
echo "mode    : $MODE   ($BIN ${ARGS[*]:-})"
echo "output  : $OUT        (配置里 run.output_dir/run.name 合成，无子目录)"
echo "log     : $LOG"
if [ "$DRY" = "1" ]; then
  echo
  echo "DRY=1：将执行 ——"
  echo "  mkdir -p $OUT"
  echo "  $BIN --config $CFG ${ARGS[*]:-} 2>&1 | tee $LOG"
  exit 0
fi

# --- 1. 前置检查 ---------------------------------------------------------
for other in corr corr-async; do
  if pgrep -x "$other" >/dev/null; then
    echo "已有 $other 在运行；先等它结束（两个接收端并存时的补零率没有意义）" >&2
    exit 2
  fi
done
cuda=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
if [ -n "$cuda" ]; then
  echo "注意：卡上已有 CUDA 进程，H2D 会被拖慢（接收侧因此更容易被拒槽）：" >&2
  echo "$cuda" | sed 's/^/  /' >&2
fi
if [ "$DOCTOR" = "1" ]; then
  echo "--- tools/doctor.py"
  python3 tools/doctor.py --config "$CFG" || true
fi

mkdir -p "$OUT"
echo "load    : $(cut -d' ' -f1-3 /proc/loadavg)"
echo "start   : $(date '+%F %T')"
echo

# --- 2. 跑 ---------------------------------------------------------------
# 前台管道：Ctrl-C 会发给整个前台进程组（包括相关机自己），于是它走正常的停止路径
# ——停止原因、报告、flush 都照常，而不是被 SIGKILL 打断在半行上。
set +e
"$BIN" --config "$CFG" ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo
echo "exit    : $rc    end: $(date '+%F %T')"

# --- 3. 判读（逐行，不看汇总）--------------------------------------------
# `*blocks.csv` 同时覆盖新命名（<file_prefix><时刻>-blocks.csv）与旧的带前缀布局
# （<prefix>blocks.csv）；不写死成 `*-blocks.csv`，否则读不到旧套。
if ls "$OUT"/*blocks.csv >/dev/null 2>&1; then
  echo "--- tools/longrun_report.py $OUT"
  python3 tools/longrun_report.py "$OUT" || true
  if [ "$TREND" = "1" ]; then
    echo "--- tools/rows_trend.py $OUT --worst 8"
    python3 tools/rows_trend.py "$OUT" --worst 8 || true
  fi
else
  echo "没有找到 blocks.csv：这次运行没有关过一行？" >&2
fi
exit "$rc"
