# tools/ — 硬件与链路实测工具

这些工具存在的理由：本机**没有 root**，`tcpdump` / `AF_PACKET` 都不可用，很多
"HANDOFF 里写成假设" 的事实只能自己写收包程序去测。测完把程序和原始日志留在
仓库里，下一轮不用重新猜。

| 文件 | 用途 |
|---|---|
| `counter_probe.c` | 每 station 一个线程、各绑一核，测**跨站 counter 是否共享同一时钟** |
| `counter_probe_120s.log` | 2026-09-14 那次 120 s 长跑的原始输出（4 路，零丢包零漂移） |
| `phase_check.py` | 从相关机落盘的 baseline 文件拟合**相位随频率的斜率**，反推站间采样延迟 |
| `stress_integration.sh` | 用不同积分时间（1 s / 100 ms / 10 ms）压写盘路径，观察 fsync 与吞吐 |
| `daily_rollover_test.sh` | 每日文件集的端到端判据：用 POSIX TZ 把本地零点挪进测试窗口，跑 2–4 分钟真跨零点，检查 12 条（套名自洽、行数之和、row_index/帧网格连续、时间单调、归属、切套代价与逐行补零）|
| `ab_zerofill.sh` | 补零解耦的 A/B 单腿：同一份确定性回放（`--start-counter` 固定、`--gap-len/--gap-at` 造缺口、回放先起）打进指定二进制，存下 `blocks.csv`、运行报告与前后 load |
| `compare_blocks.py` | 两条腿的 `blocks.csv` 逐行比对（按 `first_frame_id`+station 匹配），打印共有行数、逐行是否相同、以及排除启动首行后的缺失率 |

## phase_check

回放器可以给某个站加**整数个采样的延迟**（`--sample-delay INDEX:N`，按绝对采样
序号算，会跨越包边界），相关之后两条基线之间的相位应该随频率线性增长，
斜率正比于延迟，拟合出的 τ 必须精确等于 N：

```sh
./corr-rs/target/release/corr-replay --config config/correlator-test.yaml \
    --pattern prng --sample-delay 1:2 --duration 40 &
./corr-rs/target/release/corr --config config/correlator-test.yaml --duration 30
python3 tools/phase_check.py <输出目录>
```

实测（τ=0/1/2 三组）：拟合 τ = 0.0000 / 1.0000 / 2.0000，最大残差 0.0002 rad。
2026-09-16 在 ABI 7 上复测：拟合 τ = 0.0000 / 1.0000 / 2.0000，最大残差 ≤1e-4 rad。

**τ 是从目录名里读出来的**（`int("".join(c for c in basename if c.isdigit()))`），所以要传
**以 τ 命名**的目录：传 `<output_dir>/<run_name>/` 那种带时间戳的运行目录会把
`20260916071744` 当成 τ，理论曲线完全错位（残差列会打印 π/2 这种"看着像坏了"的数），
而拟合出的 τ 仍然是对的 —— 别被那一列骗了。用符号链接取名 `d0/d1/d2` 即可。
这一条同时证明了拼帧对齐、共轭方向、FFT bin 顺序三件事都对——三者任一错都会
让相位斜率变成别的样子。图见仓库根目录的 `phase_check.png`。

## stress_integration

```sh
tools/stress_integration.sh N_CUM ROWS_PER_FETCH FSYNC_EVERY_ROWS SECONDS [RUN_NAME]
# 例：100 ms 积分（n_cum 必须是 frames_per_slot=256 的整数倍）
tools/stress_integration.sh 9728 10 10 25 D-100ms
```

每行固定 9.38 MiB，积分时间决定行多久来一次：1 s 档是 9.8 MB/s，
100 ms 档是 98.7 MB/s，10 ms 档是 937 MB/s。压测结果与解读见 HANDOFF §6.9
（要点：1 s 稳；100 ms 的瓶颈是写盘——换 tmpfs 就准点零丢包；
10 ms 即使存储免费也只有 86 行/s，嫌疑是"writer 在管线线程里同步跑"）。
脚本会把每行计数、`rows_ms.bin`/`blocks.csv` 一致性、以及落后百分比打出来，
跑完自动删掉输出目录（10 ms 档一次 20 s 就是 11 GB）。

## daily_rollover_test

按本地日期分文件集（`docs/DESIGN-daily-files.md`）唯一的验证方式：不能让日历等你，所以
用 POSIX TZ 字符串把**本地零点挪进测试窗口**：

```sh
tools/daily_rollover_test.sh 20000 150 1     # 20 kpps、跑 150 s、1 分钟后到本地零点
```

TZ 字符串形如 `CORR+11:26:34`（POSIX 的符号与直觉相反：`+` 表示西经/UTC-），脚本自己算出
偏移，使"现在"正好是本地 23:59。跑完逐条打印 12 项判据（全过才算通过）。**不要用文件 mtime
与墙钟相减**：本机 `/scratch` 报出的 mtime 比墙钟慢约 13 分钟，混用会凭空造出 13 分钟的
"停顿"；脚本改成用**自己的时钟**记录"什么时候看见变化"。

实测（24 站）：20 kpps 下切套 0 帧补零；50 kpps（管线贴极限）下紧随边界的一行掉 6.3%；
两种速率下 304 个文件的建立都是 ~0.5 s。

判读工具现在都支持带前缀的文件集：`--prefix <YYYYmmdd_HHMMSS>`，默认取最后一套。

## ab_zerofill + compare_blocks

判据一/判据四要用「同一份输入、旧行为 vs 新行为」的对照，不是两次随便跑的对比：

```sh
OLD=/scratch/corr-baseline/corr-rs/target/release/corr   # 停在改动前的 worktree
NEW=corr-rs/target/release/corr
for leg in "a1 $OLD" "b1 $NEW" "a2 $OLD" "b2 $NEW"; do set -- $leg
  tools/ab_zerofill.sh "$1" "$2" 40 60 20000        # A/B/A/B 交错，抵消负载漂移
done
python3 tools/compare_blocks.py /scratch/zerofill-ab/a1/*/ /scratch/zerofill-ab/b1/*/
```

实测（缺口 40000 个包落在运行中，load 18–22）：含缺口的那一行两条腿**一位不差**
（240000/2347008 = 10.2258%），`tools/compare_runs.py` 对同几行做逐字节比较也是
identical；唯一不同的是每轮的**启动首行**——那是启动瞬态，差异在输入不在数据通路。
解读与四条判据的完整实测见 `docs/DESIGN-zero-fill-decoupling.md` §8。

`--stall INDEX:SECONDS` **不是**丢包的站：它冻结该站的计数器，恢复后该站的包落在别人
已经走过的帧上（变 late），而其他站在它身后被环挡住。要造"缺口"用 `--gap-len/--gap-at`。

**不要让回放比关联机的 `--duration` 早结束**：新旧二进制都会挂住（停在 futex，无报错）。

**不要用全局 `sync` 计时**：本机 `sync` 要等所有用户的脏页，实测 2 GB 写完后
`sync` 超过 7 分钟没回来。判读一律用行数 / `blocks.csv` / 文件大小对齐。

## counter_probe

```sh
gcc -O2 -pthread -o /tmp/counter_probe tools/counter_probe.c
taskset -c 8-15 /tmp/counter_probe 120
```

它会：
- 加入 `W17 / S01 / S06 / N04` 四个组（刻意覆盖三块 RFSoC 板）；
- 每线程独占一核，`SO_RCVBUF=64 MiB`，用 C 而不是 Python 收包
  —— Python 单线程只能跑到 ~35 kpps（真实 195 kpps），socket 队列里积压的是旧包，
  各线程滞后不同，看上去就是"漂移"，会得出完全错误的结论；
- 每收 16 个包发布一次 `(counter, CLOCK_MONOTONIC)`，主线程每 200 ms 采样全部 station；
- 统计 counter 跳变（gaps / lost）与回退（reorder）。

判读方法：**不要看全段分位数，要看窗口最小值。**

极差里混了两种东西：

1. **真实时钟差** —— 若各板独立计时，它会**单调增长**；
2. **读线程积压** —— 某线程被抢占几毫秒，socket 队列积压几百个包，它恢复后按
   FIFO 读旧包，于是"最新发布的 counter"落后于线路上真实时刻。
   `age(us)` 列会**低估**这种滞后（主线程先读 counter 再读 timestamp，
   两次读之间若线程又发布了新值，读到的是"旧 counter + 新 timestamp"）。

决定性差异：**真实频率差不会自己缩小**。所以按 12 s 窗口算最小值
（近似"无积压"样本），看它是否随时间增长：

```sh
awk -F'|' 'NF>1 { split($1,a," ");
  if (a[1] !~ /^[0-9]/) next;                 # 跳过表头，否则会被当成数字算出极差 0
  v0=a[2]; v1=a[3]; v2=a[4]; v3=a[5];
  mx=v0; mn=v0; if(v1>mx)mx=v1; if(v2>mx)mx=v2; if(v3>mx)mx=v3;
  if(v1<mn)mn=v1; if(v2<mn)mn=v2; if(v3<mn)mn=v3; print a[1], mx-mn }' \
  counter_probe_120s.log | awk 'NR%60==1{w=$1} {if($2<min||min=="")min=$2; if(NR%60==0){print "window @"w"s min spread =", min; min=""}}'
```

输出（12 s 一窗，共 10 窗）：

```text
window @0.00s    min spread = 7
window @12.01s   min spread = 7
window @24.01s   min spread = 7
window @36.02s   min spread = 7
window @48.03s   min spread = 7
window @60.03s   min spread = 14
window @72.04s   min spread = 12
window @84.04s   min spread = 7
window @96.05s   min spread = 7
window @108.06s  min spread = 7
```

## 结论（详见 HANDOFF.md 2.5）

120 s、4 路、600 个样本：

```text
gaps = 0     lost = 0     reorder = 0
极差 min 7  p50 33  p90 46  p99 109  max 706   （单位：包，1 包 ≈ 5.1 µs）
窗口最小值（12 s）: 7 7 7 7 7 14 12 7 7 7     <- 恒定，无增长
窗口中位数        : 33 31 34 34 31 31 30 33 31 34   <- 平坦
对窗口最小值线性拟合: +0.011 包/s -> 外推 1 小时 +40 包（+0.2 ms），在噪声内
```

即 counter 是三块板共享的全局包计数，`frame_id = counter / P` 可直接用于跨站对齐，
**不需要估计站间 offset**。

> 踩过的坑：按"三等分中位数"会得到 21 → 33 → 44，看起来像漂移。
> 那只是因为第一段含启动后最初几个异常紧凑的样本，而 82.84 s 的 706 包积压尖峰
> 落在末段。**别用整段分位数判漂移。**

## mcast_rate.py

自己入组、只读不发地测"前端到底在不在发、发多快"：

```bash
tools/mcast_rate.py config/correlator-24.yaml --seconds 4
```

用途是排除最简单的那个可能性。一次"跑到一半停住"的排查里，它证明了 24 个组的
counter 都在按 ~197 k/s 前进（=195 kpps/组，标称值），于是问题一定在接收侧——
在那之前，所有推理都在"源是不是停了"和"我们是不是收不动"之间打转。
读到的 pps 本身不可信（Python 跟不上 4.7 Mpps，会自己丢包），
**能说明问题的是 counter 的跨度**，它由发送端决定。

## 判读与体检工具（2026-09-15 新增）

这些工具回答的是"这次运行到底怎么样"，它们的存在原因是同一个：**报告里的每站汇总
量会掩盖逐行的损失**，而肉眼看到的图会骗人。

| 工具 | 用途 |
|---|---|
| `doctor.py` | 只读体检：网卡 NUMA 节点与 IRQ 亲和是否与收包线程撞核、RX 环大小、`netdev_max_backlog`、`SO_BUSY_POLL` 权限、是否有别的进程占着前端的组、空闲丢包基线。退出码非零=有问题。它把"布局解析"交给二进制自己（`--print-plan`），不重复实现配置逻辑。 |
| `rows_trend.py` | 逐行核对 `blocks.csv` 的补零：给"哪些行不干净、趋势是在涨还是在收敛"。容忍正在写入的半行记录，因此可以在跑的过程中看。 |
| `longrun_report.py` | 验收判决：多个 run 目录并列，给出补零率、干净行数、三段趋势、负载区间，并把"是否达标"作为退出码。 |
| `phase_triangle.py` | 相位–频率检验：逐基线做"斜率拟合 + 逆变换 lag"两个独立估计，自动推断符号约定，画上三角拼图（`--mosaic phase` 看数据+模型，默认 `--mosaic residual` 看残差平线，`--zoom J,K` 放大单条）。判据与陷阱写在其文档字符串里。 |
| `corr_common.py` | 共用换算：`fft_len(nch) = 2*nch` 等。**通道轴是 R2C 的保留半谱**，这条算术独立错过三次，只允许在这里实现一次。 |
| `compare_runs.py` | 两次运行的逐基线对比（按 `first_frame_id` 对齐），用于"改了配置之后数据是否一致"。 |
| `mk_dataflow.py` | 画数据流图（`dataflow.svg/png`）。 |

**三条使用纪律**（都是踩出来的）：

- 判据用 `rows_trend.py` / `longrun_report.py` 的**逐行**数字，不要引用运行报告里的每站汇总量。
- 长跑对比要**同一窗口内交错**（A/B/A/B，各 5–10 分钟）；这台共享机器的负载漂移是十分钟量级，块式对比会被淹没（实测同配置两次跑差 3.2 倍）。
- 起跑前先 `doctor.py`；`pkill -f corr-replay` 会杀掉执行它的 shell 自己，用 `pkill -x`。
