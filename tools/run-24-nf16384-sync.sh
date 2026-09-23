#!/usr/bin/env bash
# tools/run-24-nf16384-sync.sh —— 跑一腿 24 路 / 16384 信道 / 1 s 的**同步后端**（线程每站一条）
#
# 姊妹脚本 tools/run-24-nf16384.sh 默认跑异步后端（`corr-async`）。两条腿差的不只是
# "哪个二进制"，有四件事会让人比错：
#
#   |                | 异步（run-24-nf16384.sh，默认） | 同步（本脚本）                     |
#   | 收包实现       | 每核一个 worker、一个 epoll 集   | 每站一条线程                       |
#   | 二进制         | corr-rs/target/release/corr-async | corr-rs/target/release/corr      |
#   | 收包旋钮       | `--workers` / `--worker-cpus`    | 无：核数来自配置 `cpu.recv_cpus`   |
#   | 前端换纪元     | 自己重置并重跑（`--max-epoch-restarts`，默认 10），每轮**另起一套** | **停下**（`STOPPED: … starting a new epoch`），要人来重跑 |
#   | 配置/ABI 自检  | 没有 `--check`                   | `--check`：配置 + CUDA ABI，约 1.3 s |
#
# 所以本脚本 = "把 run-24-nf16384.sh 固定成 MODE=sync" + 同步腿独有的三件事：
#
#   1. 起跑前 `corr --check`（只有同步二进制有这个开关）：配置、CUDA ABI、编号一致性，
#      失败就别浪费一腿。`CHECK=0` 可跳过；
#   2. **拒绝会把这条腿变成异步腿的环境变量**（`MODE=async` / `WORKERS=N`）：这两个名字在
#      姊妹脚本里有意义，在这里只会让人以为在跑同步、实际不是。要跑异步就用姊妹脚本；
#   3. 收尾读日志里的 `STOPPED:` 行，按**同步腿**的语义给出下一步。最关键的是纪元变化：
#      同步腿按设计停在那里，异步腿会自己接着跑 —— 做长跑 A/B 时两条腿会因此**在不同的
#      时刻结束**，这是必须记账的差异。`RESTART_ON_EPOCH=1` 时本脚本自己重跑（剩余时长、
#      `NEW_SET=1` 另起一套），行为与 corr-async 的重置对齐（`async_recv/mod.rs`：restart
#      用 `SetPolicy::NewSet`，剩余时长不足 5 s 就不再起）。
#
# 为什么是"调用姊妹脚本"而不是把它的前置检查抄一份：同名进程检查、CUDA 进程提醒、20 s
# 网卡层静默检查、输出目录从配置读、日志命名、跑完按 `longrun_report.py` 判读 —— 这些
# 抄成两份迟早漂移，姊妹脚本自己就写着这条理由（"输出目录从配置里读，不在脚本里再写一份"）。
#
# 用法（环境变量与姊妹脚本同一套；`DUR` 默认 0 = 一直跑到 Ctrl-C）：
#
#   tools/run-24-nf16384-sync.sh                                # 同步腿，跑到 Ctrl-C
#   DUR=300 tools/run-24-nf16384-sync.sh                        # 固定 300 s
#   DUR=600 RESTART_ON_EPOCH=1 tools/run-24-nf16384-sync.sh     # 纪元变化时自己再起一腿
#   CHECK=0 tools/run-24-nf16384-sync.sh                        # 跳过起跑前的 --check
#   NEW_SET=1 / DOCTOR=1 / TREND=1 / CHK_NIC=0 / DRY=1 / CFG=… / NIC=…
#
# 与异步腿对比的正确做法（这台机器每 3–7 分钟换一次窗口，补零率能差 20 多个百分点）：
#   **同一个窗口内轮转、顺序对调**，每腿 100–300 s，比的是配对差值的方向，不是单次绝对值。
#   docs/baseline_test_sync.md 有同步腿的完整记录与噪声量级；异步腿见
#   docs/DESIGN-async-receive.md。
#
# 退出码：前置检查失败 2；跑完 = 最后一腿的退出码；因为纪元变化停下且不重跑 = 3
# （与 corr-async 同一个出口的含义一致）。

set -euo pipefail
cd "$(dirname "$0")/.."

SIBLING=tools/run-24-nf16384.sh
RUN=corr-rs/target/release/corr        # 同步后端。姊妹脚本里是 .../corr-async
DUR=${DUR:-0}
RESTART_ON_EPOCH=${RESTART_ON_EPOCH:-0}
CHECK=${CHECK:-1}
DRY=${DRY:-0}
# 与 corr-async 的 --max-epoch-restarts 默认值一致，好让两条腿的重跑上限可比。
MAX_RESTARTS=${MAX_RESTARTS:-10}

# --- 0. 这条腿的身份：别让人以为在跑同步 --------------------------------
if [ -n "${MODE:-}" ] && [ "${MODE}" != "sync" ]; then
  echo "MODE=${MODE}：本脚本只跑同步后端（线程每站一条）。" >&2
  echo "           异步腿请直接用 $SIBLING —— 它的 MODE 默认就是 async。" >&2
  exit 2
fi
if [ -n "${WORKERS:-}" ]; then
  echo "WORKERS=${WORKERS} 是异步腿的旋钮（corr-async 的 --workers），同步后端没有这个概念：" >&2
  echo "  同步腿每站一条线程，收包核数来自配置的 cpu.recv_cpus；要限制 worker 数请跑异步腿。" >&2
  exit 2
fi
[ -x "$RUN" ] || { echo "$RUN 不存在：先 cd corr-rs && cargo build --release" >&2; exit 2; }
[ -x "$SIBLING" ] || { echo "$SIBLING 不存在：本脚本复用它做前置检查与判读" >&2; exit 2; }

# --- 0b. 停止原因的模式必须与 corr-rs 的文案对得上 ------------------------
# 这一条是踩过之后加的：脚本按 `STOPPED:` 行里的短语分支，而那些短语是
# `corr-rs/src/stop.rs` 的 `StopReason::describe()` 写出来的。改文案不会让脚本报错，
# 只会让它**静默地**落到兜底分支、于是对一次"零填充上限"给出"看机器状态"这种错误建议
# （本次就发现两处：实际是 "has not advanced for" / "no packet has arrived"，
# 脚本里写的是别的词）。几毫秒的事，每次跑之前对一遍。
if [ -f corr-rs/src/stop.rs ]; then
  for pat in "exceeded the zero-fill cap" "watermark stayed lifted" "has not advanced for" \
             "no packet has arrived" "starting a new epoch" "could not be opened"; do
    grep -qF "$pat" corr-rs/src/stop.rs || {
      echo "注意：停止原因模式「$pat」已不在 corr-rs/src/stop.rs 里。" >&2
      echo "      本脚本对这类停止的解释会失效（落到兜底分支）。请对照 StopReason::describe()。" >&2
    }
  done
fi

# 配置解析规则与姊妹脚本相同（同名的那份优先，否则 manual）。这是本脚本**唯一**重复的
# 一处：`--check` 必须在姊妹脚本跑起来之前执行，而它没有"只告诉我配置路径"的开关。
HOST=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "")
if [ -n "${CFG:-}" ]; then
  :
elif [ -n "$HOST" ] && [ -f "config/correlator-24-nf16384-${HOST}.yaml" ]; then
  CFG=config/correlator-24-nf16384-${HOST}.yaml
else
  CFG=config/correlator-24-nf16384-manual.yaml
fi

echo "sync leg: $RUN   （同步后端：线程每站一条）"
echo "sibling : $SIBLING   （前置检查、日志、判读都在它里面；本脚本固定 MODE=sync）"
echo "config  : $CFG"
echo "duration: $([ "$DUR" = "0" ] && echo '无上限（Ctrl-C 停）' || echo "${DUR}s")   \
restart-on-epoch: $RESTART_ON_EPOCH   check: $CHECK"

if [ "$DRY" = "1" ]; then
  echo
  echo "DRY=1：本脚本先执行 ——"
  [ "$CHECK" = "1" ] && echo "  $RUN --config $CFG --check"
  echo "再执行 ——"
  echo "  MODE=sync DUR=$DUR $SIBLING    （其余环境变量原样传下去）"
  echo "重跑时（RESTART_ON_EPOCH=1）再加：NEW_SET=1 CHK_NIC=0，DUR=剩余时长"
  echo
  echo "--- 委托给姊妹脚本先做一次它自己的 DRY："
  exec env MODE=sync DRY=1 "$SIBLING"
fi

# --- 1. 起跑前：配置 + CUDA ABI（只有同步二进制有这个开关）---------------
if [ "$CHECK" = "1" ]; then
  echo "--- $RUN --check"
  chk=$(mktemp)
  # 注意别写成 `if ! "$RUN" …; then rc=$?`：`!` 之后 $? 是**取反后**的状态，永远是 0，
  # 于是失败信息里会印出 "rc=0"。显式取一次。
  set +e
  "$RUN" --config "$CFG" --check > "$chk" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    sed 's/^/  /' "$chk" >&2
    rm -f "$chk"
    echo "--check 失败（rc=$rc）：配置或 CUDA ABI 有问题，先修再跑" >&2
    exit 2
  fi
  grep -E "^(plan|result) " "$chk" | sed 's/^/  /'
  rm -f "$chk"
fi

# --- 2. 跑；必要时按纪元变化重跑 -----------------------------------------
# 每一腿都委托给姊妹脚本跑（它的 DRY/前置检查/日志/判读一条不落），本脚本只负责
# "这一腿跑多长、要不要另起一套，以及跑完怎么读 STOPPED"。
STARTED=$(date +%s)
leg=0
restarts=0
rc=0
logs=()
while :; do
  leg=$((leg + 1))
  extra=()
  leg_dur=$DUR
  if [ "$leg" -gt 1 ]; then
    extra=(NEW_SET=1 CHK_NIC=0)   # 另起一套（同 corr-async 的重置），网卡 20 s 检查不重复做
    if [ "$DUR" -gt 0 ]; then
      left=$(( DUR - ( $(date +%s) - STARTED ) ))
      if [ "$left" -lt 5 ]; then
        echo >&2
        echo "sync leg: 只剩 ${left}s（<5s），不值得再起一腿；停在这里" >&2
        echo "          —— 与 corr-async 的 \"less than 5s left\" 规则一致" >&2
        rc=3
        break
      fi
      leg_dur=$left
      echo
      echo "sync leg: 第 $leg 腿（纪元变化后重跑），剩余 ${leg_dur}s，另起一套"
    else
      echo
      echo "sync leg: 第 $leg 腿（纪元变化后重跑），无时长上限，另起一套"
    fi
  fi

  out=$(mktemp)
  set +e
  env MODE=sync DUR="$leg_dur" ${extra[@]+"${extra[@]}"} "$SIBLING" 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  set -e

  log=$(sed -n 's/^log     : //p' "$out" | head -1)
  [ -n "$log" ] && logs+=("$log")
  stop=$(grep -a "STOPPED:" "$out" | tail -1 || true)
  rm -f "$out"

  echo
  echo "sync leg: 第 $leg 腿结束 rc=$rc"
  if [ -n "$stop" ]; then
    echo "  $stop"
  else
    echo "  （日志里没有 STOPPED 行：正常时长结束或 Ctrl-C 收尾时不会有它）"
  fi

  case "$stop" in
    *"starting a new epoch"*)
      # 同步腿在这里停下是设计如此；异步腿会自己接着跑。这条差异会直接改变长跑 A/B
      # 的结束时刻，所以必须说清楚，而不是让它看起来像一次"莫名其妙的退出"。
      echo "  → 前端换了纪元。同步腿**按设计停下**（异步腿此时会自己重置并继续）。"
      if [ "$RESTART_ON_EPOCH" = "1" ]; then
        if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
          echo "  → 已经重跑过 $restarts 次（MAX_RESTARTS=$MAX_RESTARTS），不再重跑" >&2
          rc=3
          break
        fi
        restarts=$((restarts + 1))
        echo "  → RESTART_ON_EPOCH=1：重跑（第 $restarts/$MAX_RESTARTS 次）"
        continue
      fi
      echo "  → 想接着跑：DUR=<秒> RESTART_ON_EPOCH=1 $0    （或直接再执行一次本脚本，"
      echo "     它会续写当日那一套；加 NEW_SET=1 则另起一套）"
      rc=3
      break
      ;;
    # 下面每个模式对应 corr-rs/src/stop.rs 里的一个 StopReason 变体（改文案时对照这里）：
    #   "exceeded the zero-fill cap" → ZeroFillCap        （补零超限，数据面）
    #   "watermark stayed lifted"    → LiftCap            （水位线抬升不回落，数据面）
    #   "has not advanced for"       → StreamNotAdvancing （包在到、行号不动）
    #   "no packet has arrived"      → StreamQuiet        （前端停了）
    *"exceeded the zero-fill cap"*)
      echo "  → **数据面**：某站补零超过 integration.zero_fill_limit_frames，运行自己停了。"
      echo "     先 tools/rows_trend.py <输出目录> 看逐行补零率是"某一行烂"还是"一路烂"，"
      echo "     再对照 docs/baseline_test_sync.md §6 的窗口量级，判断是前端/线路还是本机。"
      break
      ;;
    *"watermark stayed lifted"*)
      echo "  → **数据面**：水位线抬升后一直不回落（某站被落下超过 max_spread_slots）。"
      echo "     这是设计上的停止（gap 不会自己合拢，只有重启才会），本套的行是完整的到停点为止；"
      echo "     先看是哪一个站、它当时是不是在等环或被网卡丢包顶住（tools/loss_attribution.py）。"
      break
      ;;
    *"has not advanced for"*)
      echo "  → **数据面**：包还在到、行号不动 —— 收包在读、纸面上不丢、但一行也写不出去。"
      echo "     常见成因是前端计数器开新纪元，或当天这一套被从流永远到不了的帧上续写；"
      echo "     查前端后另起一套（NEW_SET=1），本套已有的行保留。"
      break
      ;;
    *"no packet has arrived"*)
      echo "  → **数据面**：24 路全部静默（前端停了，或到它的路径断了）。"
      echo "     先 tools/mcast_rate.py 确认前端在不在发，再决定重跑；本套到缺口为止是完整的。"
      break
      ;;
    *"could not be opened"*)
      echo "  → **运维面**：跨本地日时新一天的文件集打不开（RolloverFailed）。"
      echo "     新一天的行**没有**写进任何文件；先看输出目录的权限/空间，再重跑。"
      break
      ;;
    *)
      # 匹配不上有两种情况：正常结束（上面已说明没有 STOPPED 行），或者 corr-rs 又加了
      # 一种停止原因而本脚本还没跟上。后者不能沉默——把原文与提醒一起打出来。
      if [ -n "$stop" ]; then
        echo "  → 这条停止原因本脚本还没有对应的建议（上面那行 STOPPED 是全文）。"
        echo "     对照 corr-rs/src/stop.rs 的 StopReason::describe() 补一个分支；"
        echo "     本脚本开头那条模式对账只在**文案改词**时会提醒，新增变体不会。"
      fi
      break
      ;;
  esac
done

echo
echo "sync leg: 共跑了 $leg 腿，重跑 $restarts 次"
for l in ${logs[@]+"${logs[@]}"}; do echo "  log: $l"; done
exit "$rc"
