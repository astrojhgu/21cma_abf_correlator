#!/bin/bash
# 把网卡队列中断收拢到指定的核上（改它需要 root；--dry-run 不需要）。
#
# 背景（2026-09-19 事故，全部为实测数据）：这张卡的 33 个 MSI-X 向量与 32 个核一一对应
# —— `effective_affinity_list` 显示 mlx5_compN → cpuN（注意不是 `smp_affinity_list`，
# 它在本机对每个向量都写着 0-31，只是允许范围，看不出落点）。而 corr 的接收线程也在
# 那些核上：rx14/15/16 与 comp18/19/20 同核，被 ksoftirqd 抢时间，实测那些线程只拿到
# 33-35% 的核（别的接收线程 50-71%），对应那几路 socket 积压到 420 MB 后开始成片丢包。
#
# 为什么挪一半就够：实测异步版 24 个接收线程合计 12.75 核，而非 corr 的 CPU 占用只有
# 约 5 核（整机 17.8/32 核），且 cpu0-3 当时只用了 2-24%。把 comp16 及以后的中断挪到
# cpu0-3，cpu18-27 就干净了，最差的那批接收线程能恢复满速。
#
# 用法：
#   tools/pin-mlx5-irqs.sh --dry-run            # 打印会改什么（不需要 root）
#   sudo tools/pin-mlx5-irqs.sh                 # 只挪 comp16+（推荐，先跑这个）
#   sudo tools/pin-mlx5-irqs.sh --all           # 33 个向量全挪
#   sudo tools/pin-mlx5-irqs.sh --cpus 0-4      # 换目标核
#   sudo tools/pin-mlx5-irqs.sh --revert        # 恢复成 0-31（或直接重启）
#
# 改完盯 corr-web 的「网卡缓冲（RX ring）」卡：ring 溢出一旦非零，说明目标核过载，
# 立刻用 --revert 回退。这台机器上 irqbalance 是 inactive，所以改动会保持到重启。
set -u

IFACE=ens2f0np0
CPUS=0-3
FROM=16          # 只挪 mlx5_compN，N >= FROM；--all 时全部
ALL=0
REVERT=0
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --iface)  IFACE=$2; shift 2 ;;
        --cpus)   CPUS=$2; shift 2 ;;
        --from)   FROM=$2; shift 2 ;;
        --all)    ALL=1; shift ;;
        --revert) REVERT=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "未知参数：$1（--help 看用法）" >&2; exit 2 ;;
    esac
done

MSI=/sys/class/net/$IFACE/device/msi_irqs
[ -d "$MSI" ] || { echo "找不到 $MSI：接口名对不对？" >&2; exit 1; }

target=$CPUS
[ "$REVERT" = 1 ] && target=0-31

if [ "$DRY" = 0 ] && [ "$(id -u)" != 0 ]; then
    echo "改中断亲和性需要 root（先用 --dry-run 看会改什么）" >&2
    exit 1
fi

name_of() {   # IRQ 号 → /proc/interrupts 里的动作名（mlx5_compN）
    awk -v want="$1:" '$1 == want { print $NF }' /proc/interrupts
}
effective_of() { cat "/proc/irq/$1/effective_affinity_list" 2>/dev/null || echo '?'; }

echo "接口 $IFACE，共 $(ls "$MSI" | wc -l) 个 MSI-X 向量"
if [ "$REVERT" = 1 ]; then
    echo "模式：恢复（全部 → $target）"
elif [ "$ALL" = 1 ]; then
    echo "模式：全部 → cpu$target"
else
    echo "模式：mlx5_comp$FROM 及以后 → cpu$target（其余不动）"
fi
[ "$DRY" = 1 ] && echo "（dry-run：不写入任何东西）"
echo

picked=0
skipped=0
for f in "$MSI"/*; do
    irq=$(basename "$f")
    name=$(name_of "$irq")
    [ -n "$name" ] || { echo "  跳过 IRQ $irq：/proc/interrupts 里找不到"; continue; }
    name=${name%%@*}      # mlx5_comp16@pci:0000:41:00.0 → mlx5_comp16

    if [ "$REVERT" = 0 ] && [ "$ALL" = 0 ]; then
        num=${name#mlx5_comp}
        case "$name" in
            mlx5_comp*) [ "$num" -ge "$FROM" ] 2>/dev/null || { skipped=$((skipped+1)); continue; } ;;
            *)          skipped=$((skipped+1)); continue ;;   # async 等向量不碰
        esac
    fi

    before=$(effective_of "$irq")
    if [ "$DRY" = 1 ]; then
        printf "  IRQ %-4s %-14s %s → %s\n" "$irq" "$name" "$before" "$target"
    else
        if echo "$target" > "/proc/irq/$irq/smp_affinity_list" 2>/dev/null; then
            printf "  IRQ %-4s %-14s %s → %s\n" "$irq" "$name" "$before" "$(effective_of "$irq")"
        else
            printf "  IRQ %-4s %-14s 写入失败（该向量可能不接受多核掩码）\n" "$irq" "$name" >&2
        fi
    fi
    picked=$((picked+1))
done

echo
echo "涉及 $picked 个向量，未动 $skipped 个"
echo
echo "改动后每核承载的向量数（期望：目标核变多、被腾空的核变 0）："
python3 - "$IFACE" <<'PY'
import collections, os, sys
iface = sys.argv[1]
share = collections.Counter()
for f in os.listdir('/sys/class/net/%s/device/msi_irqs' % iface):
    try:
        lst = open('/proc/irq/%s/effective_affinity_list' % f).read().strip()
    except OSError:
        continue
    for part in lst.split(','):
        if '-' in part:
            a, b = part.split('-')
            for c in range(int(a), int(b) + 1):
                share[c] += 1
        elif part.isdigit():
            share[int(part)] += 1
print('   ', ' '.join('cpu%d=%d' % (c, share.get(c, 0)) for c in range(32)))
PY

if [ "$DRY" = 0 ] && [ "$REVERT" = 0 ]; then
    cat <<'TIP'

下一步：
  1. 重启 corr，看启动行 "receive cores ordered by NIC IRQ assignment"（若显示
     "spread evenly ... left alone"，说明排序没有可用的区分度）。
  2. 盯 corr-web 的「网卡缓冲」卡：ring 溢出非零 = 目标核过载，立刻 --revert。
  3. 对照「每核」占用：cpu18-27 应当降下来，cpu0-3 升上去。
  4. 重启会失效 —— 要持久就写一个 oneshot systemd unit 调这个脚本（见 tools/README.md）。
TIP
fi
