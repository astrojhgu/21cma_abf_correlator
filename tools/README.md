# tools/ — 硬件与链路实测工具

这些工具存在的理由：本机**没有 root**，`tcpdump` / `AF_PACKET` 都不可用，很多
"HANDOFF 里写成假设" 的事实只能自己写收包程序去测。测完把程序和原始日志留在
仓库里，下一轮不用重新猜。

| 文件 | 用途 |
|---|---|
| `raw_capture.c` | 把**一个站**的 UDP 载荷原样落盘（8 字节 counter + 4096 int8），不重排不修补：C 收包才能跟住 195 kpps（Python 单线程只到 ~35 kpps，会把谱测成 Python 的性能）|
| `raw_spectrum.py` | 对抓到的裸数据做 FFT（帧定义、R2C、无窗无归一化都与相关机一致）并画谱，可与相关机自己存的自相关逐条对照 |
| `counter_probe.c` | 每 station 一个线程、各绑一核，测**跨站 counter 是否共享同一时钟** |
| `counter_probe_120s.log` | 2026-09-14 那次 120 s 长跑的原始输出（4 路，零丢包零漂移） |
| `phase_check.py` | 从相关机落盘的 baseline 文件拟合**相位随频率的斜率**，反推站间采样延迟 |
| `stress_integration.sh` | 用不同积分时间（1 s / 100 ms / 10 ms）压写盘路径，观察 fsync 与吞吐 |
| `autospectra.py` | 24 条**自相关谱**（j==k）一屏看：每站一格 + 全站叠加；dB 用全局参考（能看出端口间电平差）、`--skip-bins` 跳过 DC 尖峰。顺带检查自相关的虚部（应为 0）与每站被补零的帧数 |
| `daily_rollover_test.sh` | 每日文件集的端到端判据：用 POSIX TZ 把本地零点挪进测试窗口，跑 2–4 分钟真跨零点，检查 12 条（套名自洽、行数之和、row_index/帧网格连续、时间单调、归属、切套代价与逐行补零）|
| `ab_zerofill.sh` | 补零解耦的 A/B 单腿：同一份确定性回放（`--start-counter` 固定、`--gap-len/--gap-at` 造缺口、回放先起）打进指定二进制，存下 `blocks.csv`、运行报告与前后 load |
| `compare_blocks.py` | 两条腿的 `blocks.csv` 逐行比对（按 `first_frame_id`+station 匹配），打印共有行数、逐行是否相同、以及排除启动首行后的缺失率 |
| `run-24-nf16384.sh` | 手动跑一腿 24 路 / 16384 信道 / 1 s：前置检查（没有别的 corr 在跑、卡上有没有别的 CUDA 进程、二进制是否比源码旧）→ 起跑（`MODE=async|sync`、`WORKERS=N`，**默认一直跑到 Ctrl-C**，`DUR=<秒>` 可改成固定时长；起跑前还会花 20 s 做**网卡层静默检查**，已经在丢就当场警告，`CHK_NIC=0` 可跳过）→ 收尾按 `longrun_report.py` 判读；输出固定落 `/scratch/jhgu/corr/`，日志同目录 |
| `async_bench.py` | 收包形状的 A/B 台架：同一份配置只换核池，交错跑 `corr`（一线程一站）与 `corr-async --workers W`（默认构建即产出），每腿 25 s；判据是**送达率**（收到的包 / 回放发出的包）＋三层丢包（应用缺口、`/proc/net/udp` 的 `drops`、网卡 `rx_missed_errors`）＋CPU µs/包。设计与数据见 `docs/DESIGN-async-receive.md` |

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

## raw_capture + raw_spectrum

某个站的谱看着不对时，第一个问题是**裸数据是不是本来就那样** —— 相关机的结果已经过拼帧、
FFT 和平均。这套工具回答它：

```sh
gcc -O2 -o /tmp/raw_capture tools/raw_capture.c
taskset -c 28 /tmp/raw_capture 239.1.173.3 10073 172.23.0.11 1.0 /scratch/jhgu/s13-raw.bin
#             GROUP            PORT     IFACE_IP     秒   输出
python3 tools/raw_spectrum.py s13=/scratch/jhgu/s13-raw.bin --n-freq 16384 \
        --against <run_dir> --station 21 --out /scratch/jhgu/raw-vs-corr.png
```

一秒钟一个站是 ~800 MB（194 kpps × 4104 B）。`raw_spectrum.py` 用**与相关机完全相同的估计量**
（`n_freq/2048` 个连续包一帧、R2C、无窗、无归一化、帧间取 |Y|² 平均），所以两条曲线可以直接
比形状**和电平**。实测（2026-09-17，nch=16384）：S13 的裸数据 FFT 与相关机存的自相关在
50–150 MHz 上平均差 **−0.91 dB**，把该站 17% 的补零稀释补回去之后是 **−0.10 dB / RMS 0.12 dB**。

## autospectra

阵列健康检查的第一步：自相关就是功率谱，一眼能看出死端口、增益不对的端口、杂散和带内形状。

```sh
tools/autospectra.py /scratch/jhgu/corr/nf16384-1s-24                    # 最新一套
tools/autospectra.py <run_dir> --prefix 20260917_000000 --db --skip-bins 2 \
        --out /scratch/jhgu/auto.png --overlay /scratch/jhgu/auto-overlay.png
```

- `--db` 用**全局**参考（最强的那个通道 = 0 dB），所以端口间 10 dB 的差会直接看出来；线性模式下 y 轴按分位数裁，免得 DC 尖峰把整幅压平。
- `--skip-bins 2` 不画 DC 附近 —— nch=16384 时 R2C 的 DC bin 比带内高 3–4 个数量级。
- 顺手打印 **max|Im|/Re**（自相关必须是实数，正常 ~1e-10，不对就是共轭方向或布局错了）和每站被补零的帧数（短跑的首行会把平均值拉低，但不改变形状）。

**看单个站：`--station NAME`**（可重复或逗号分隔）。画一格，纵轴 0 dB = 该站自己最强的通道，
并叠上**逐行 ±1σ 的带子**——平均值本身看不出一个信道是稳的还是在十几倍地跳，带子才看得出来。
配合 `--row-range A:B` 指定行区间：

```sh
tools/autospectra.py /scratch/jhgu/corr-g05 --station W17 --prefix corr24-20260920-204159 --db
tools/autospectra.py /scratch/jhgu/corr-g05 --station W17 --prefix corr24-20260921-000000 \
        --row-range 0:5315 --db          # 只平均那一段
```

**为什么需要 `--row-range`**：一天一套（`daily_files`），所以**同一天的第二次运行是续写同一个套**。
2026-09-21 那套就是这样：00:00–01:29 一夜、然后 10:34 起又一次运行在写 —— 直接平均整套
等于把两段的仪器状态混在一起（实测同一个站 W17 的带内中位功率差 **64 倍**：夜里 1.36e6
对上午 2.1e4）。图上的标题会写出实际的本地时刻区间，别看套名就以为是连续的一段。

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
| `uvmap.py` | **单条基线的 uv 图**（不是 300 条一起做）：把一条基线一晚的 cross/自相关格点化到 (u,v) 平面，输出 `${out}r.fits` / `${out}i.fits`（实部/虚部，只依赖 numpy，自带极简 FITS 写盘）与 `--png` 预览。一次一条：`tools/uvmap.py /scratch/jhgu/corr W18 W19 --png`。详见下面「uvmap」一节。 |
| `uvmap_raw.py` | 同一张图，但格子上的值是**原始互相关**——**不除以 `sqrt(auto_j*auto_k)`**。`uvmap.py` 的归一化把带通、增益、自相关电平都除掉、换来一个 [0,1] 的数；这一支什么都不除，所以自相关携带的信息（带形、某端口增益变了、某信道一站着）仍在图上，振幅也带着文件自己的单位。它只读 cross 一个文件（I/O 是 1/3，活套里也就少一个"行数可能落后"的文件），输出前缀是 `uvmapraw-`，与上一条不互相覆盖。振幅跨几个量级，配 `--db` 才看得清：`tools/uvmap_raw.py /scratch/jhgu/corr-g05 W18 W19 --png --db`。 |
| `corr_common.py` | 共用换算：`fft_len(nch) = 2*nch` 等。**通道轴是 R2C 的保留半谱**，这条算术独立错过三次，只允许在这里实现一次。 |
| `compare_runs.py` | 两次运行的逐基线对比（按 `first_frame_id` 对齐），用于"改了配置之后数据是否一致"。 |
| `mk_dataflow.py` | 画数据流图（`dataflow.svg/png`）。 |
| `loss_attribution.py` | 补零的包**丢在哪一层**：把网卡（`rx_prio0_discards` / `rx_discards_phy` / `rx_out_of_buffer`）、内核软中断（`/proc/net/softnet_stat` 的 dropped / time_squeeze）、我们自己的 socket（corr 的 `socket_drops`）、corr 的 `lost` 四层增量并排打成一行，末尾一句责任判定；顺带指出 mlx5 中断与 squeeze 集中在**哪些核**（无 root 时这是唯一能看的线索）。**只读、不 join 组播**，所以别的 corr 正在跑时也能用 —— `doctor.py` 不行，它会收包。 |
**网卡层丢包：现状与那条改不动的命令（2026-09-18 实测）**

24 路 × 16384 档在 4.7 Mpps 下，补零变多时 `loss_attribution.py` 给出的答案是"网卡 RX
队列（硬件）"：`rx_prio0_discards` 的增量与 corr 的 `lost` 增量**逐个相等**，而软中断
dropped、我们自己的 socket drops、等环全都是 0。根因是 RX ring 只有 **1024**（硬件上限
8192）：4.7 Mpps 摊到 32 个 RSS 队列时每队列 147 kpps ≈ **7 ms** 的缓冲，看似够 ——
但实测突发会集中到少数队列与 CPU（5 s 窗口里 mlx5 中断压在 cpu18/19/20 上，各 2.1–2.3
万次），那时的**有效**缓冲掉到亚毫秒，任何中断被推迟就直接丢包。改它需要 root：

```bash
sudo ethtool -G ens2f0np0 rx 8192      # 临时生效，重启失效；持久化见 NetworkManager dispatcher
```

本机拿不到 root（`SO_BUSY_POLL` 是 EPERM、`AF_PACKET` 也是 EPERM，所以连"应用态大 ring"
这条绕行路都没有），所以这条命令只能留档，等有权限的人执行。在此之前，
`tools/corr-web.py` 页面的"网卡层"卡片与 `loss_attribution.py` 是同一份口径：**应用侧
全干净、网卡层在丢**就是当前这种局面的特征。卡片的主值只认 `rx_out_of_buffer`（RX ring 满，
包在网卡里就没了，应用侧三个计数都看不到，直接变成零填充）；`rx_prio0_discards`（= `rx_discards_phy`）
是端口级计数，实测它在涨时 ring 溢出与应用侧计数往往全是 0（那些包不是我们的），所以它退到副行，
只有当同一窗口零填充也在涨时才变色。

| `corr-web.py` | 独立运行的**状态监测界面**：轮询 corr 的两个只读端点——每秒取 `GET /status.json`（运行状态，自己攒历史画曲线），每 5 s 取 `GET /autospectra.json`（最新那一行的自相关谱，每路一格 dB 曲线）；页面默认在 `http://127.0.0.1:8080/`。只用 Python 标准库，**不读 NFS**（谱由写盘端记录，不是读回来的——这台机器上被测的就是写盘那条路径）；corr 停了也不崩，显示"断开 N 秒"并继续展示最后一份。接口契约见 `docs/DESIGN-status-http.md`。页面上的「水位线」卡片与「每站」表最后一列读的是 `recv.stations_offset_slots`：那一列把数值和条并排画在一起——竖线就是水位线（0），绿条是领先、红条是落后于水位线（落后部分的帧进 `blocks.csv` 时记成 missing），橙线是 `integration.max_spread_slots` 的 cap，落在轴外时改为在表头说明里点名。轴是量程跟数据走的对称轴，0 恒在正中，所以"离被迫放弃还差多远"是看得见的，也不用去减两个数。 |

**三条使用纪律**（都是踩出来的）：

- 判据用 `rows_trend.py` / `longrun_report.py` 的**逐行**数字，不要引用运行报告里的每站汇总量。
- 长跑对比要**同一窗口内交错**（A/B/A/B，各 5–10 分钟）；这台共享机器的负载漂移是十分钟量级，块式对比会被淹没（实测同配置两次跑差 3.2 倍）。
- 起跑前先 `doctor.py`；`pkill -f corr-replay` 会杀掉执行它的 shell 自己，用 `pkill -x`。

### uvmap：一条基线的 uv 图

这份 `uvmap.py` 原来不是为本系统写的（21CMA 那套：ctime 时间文件、手写的 XY/XX/YY
三个文件、手填的 delay、astropy 求恒星时）。**原版保留为 `uvmap-21cma-legacy.py`**
（它原来没进 git，改写前先留了一份）。现在的版本只换输入与依赖，**格点化公式一字
未改**，因为那张图就是它：

    channel k -> 半径 (k/nch)*(npix/2)      半径 ∝ 频率
    row i     -> 角度 = 该行的当地恒星时

于是"一晚"在一个环扇区里展开：扇区的径向范围是频段，张角就是天空转过的角度（7.4 h ≈ 113°）。
半径的像素尺度是 `|b|·f_max/c`，即按基线长度归一化；要按波长标注就 `--baseline-length-m`。

与旧脚本的差别（每一条都对应本系统的一个事实）：

- 文件从**元数据**里的 baseline 列表查（`corr_common`），不按站名拼文件名 —— 名字里的
  先后是基线序号顺序（`j >= k`），`N04` 配 `N01` 存的是 `...-N01N04-...`；
- 时间读 `<tag>-time.bin`（int64 毫秒 UTC），不再解析 ctime、不再在脚本里写站点时区
  （只有没有 `.bin` 时才退回 `<tag>-time.txt`，并用元数据自己的 `local_utc_offset_s` 换算）；
- 恒星时用 GMST 多项式（误差 ~0.1 s，环上 0.03 像素），**不需要 astropy**：本机没装，
  而 py3.9 的 astropy 会把 numpy 降到 <2（见下面的安装说明）；
- `--delay` 默认 **0**，即不做条纹对齐（要相位对齐再显式给米数）；
- **补零行不参与平均**（读 `blocks.csv`）：补零位置在积分类产品里是真零，混进去会把那个
  像素压下去；
- 默认**从信道 0 起**（`--ch-min 0`），即整个 0–400 MHz、含 R2C 的 DC bin。代价是低频
  信道全落在格点中心：几千个信道挤进几个像素（打印里的 "samples in the fullest" 就是这个
  堆叠程度 —— 0.43 h 的数据已有 2.5 万个样本堆在一个像素里；只从 50 MHz 起时是 1.7 千），
  而 bin 0 是前端直流偏置、比带内高几个量级，归一化后可以接近 1（就中心那一个像素）。
  要从 50 MHz 起就 `--ch-min 2048`（旧脚本 8192 信道下的 `//4` 边界正好是这个频率）。

    tools/uvmap.py /scratch/jhgu/corr W18 W19 --png            # 最新一套（= 0 点后那套）
    tools/uvmap.py /scratch/jhgu/corr N01 N02 --rows 0:4000 --step 4 --ch-min 1
    tools/uvmap.py /scratch/jhgu/corr N04 W19 --delay 12.5 --baseline-length-m 900 --png

**`uvmap_raw.py` 是同一张图的另一支**：格子上的值就是 cross 本身，**不除自相关**。
两支共用同一份实现（`uvmap.py` 里 `run(a, normalise=...)`），`uvmap_raw.py` 只是把
`normalise=False` 打进去——所以格点化、时间、丢补零行、输出格式这些只有一份定义，
不会漂移。什么时候用哪支：

- 要**可见度**（跟源、跟模型比）→ `uvmap.py`，值是 `cross/sqrt(auto_j*auto_k)`；
- 要看**仪器/带通本身**（带形、某端口增益变了、某信道一站着、归一化把东西除没了）→
  `uvmap_raw.py`，值是文件里的原始计数（`divide_by_n_cum` 之后的量纲）。它的振幅跨
  好几十年（实测中位数 ~5e3、p99 ~2e6，均值被低频拖着走），所以看 PNG 要加 `--db`。

**读图时注意两件事**（都被误读过一次）：

1. **一条径向线 = 一行**（≈1 s）。一晚上的数据在图上是一条条径向线拼出来的扇区，所以
   十几秒的短事件就是图上一条细线。实测 2026-09-20 那套的第 5935–5949 行（≈22:22:00 本地）
   raw 功率比邻行高 200–400 倍、同时 |V| 掉到中位的 0.55 倍 —— 它在图上就是**最暗**的那条
   径向线，而在 raw 图上是**最亮**的（归一化把这种"功率涨、相干掉"的事件压下去了，
   这正是留 raw 一支的理由）。
2. **扇区的两条边界不是数据**。边界那一列像素只被约 7 行覆盖，内部是约 13 行，所以边界
   噪一点、看着像有一条亮线；实测把它和内部逐半径比过，边界行（第 0–5 行 |V|=0.033–0.035
   对中位 0.0396）并不亮。这是硬边界的观感 + 边界像素行数减半，不是数据异常。

```bash
tools/uvmap_raw.py /scratch/jhgu/corr-g05 W18 W19 --png --db
```

输出默认与数据放在一起：`<run_dir>/uvmap-<站><站>-<套标签>{r,i}.fits` 与同名 `.png`
（raw 那支是 `uvmapraw-` 前缀，两者不互相覆盖）。
（2048² float32 的 FITS 一对约 34 MB，所以**不要**让它默认落在仓库里；用 `--out` 改前缀）。
一次一条基线：`uvmap.py` 读三个文件（cross + 两条自相关，一晚上 3×3.5 GB，/scratch 上约 11 s），
`uvmap_raw.py` 只读 cross。

**两处口径要靠人确认**（脚本里都留了开关，别当成默认值是对的）：

1. `--lon` 默认 `86.432772`（照抄旧脚本）。本机那份 21CMA 台址表
   （`~/raw2ms_lite-master/21CMA_data/antennas.txt`，只有 E/W 臂 40 个站的 ECEF）反算出来
   是 **86.7174 E, 43.1157 N**，差 0.285° = **1.1 分钟时角**（图上 0.28°，看不出，但别拿它
   当绝对定标）。
2. 库里没有站坐标，所以 u,v 是"以基线长度为单位的像素"，不是波长 —— 要给绝对 u,v
   需要一张 24 个站的 E/N 坐标表。

**给 py3.9 装 astropy 的坑**：`python3 -m pip install --user astropy` 在 py3.9 上会解析到
astropy 6.0.1（要求 `numpy<2`），从而把你 `~/.local/lib/python3.9` 里的 numpy 2.0.2
**降级成 1.26.4**，影响所有用 python3 的脚本。astropy ≥ 6.1 只支持 py≥3.10，本机有
python3.11，所以装到 3.11 才不碰现有环境：

    python3.11 -m pip install --user astropy     # 落在 ~/.local/lib/python3.11/，与 3.9 隔离

`uvmap.py` 自己不需要 astropy（上面这条只是为了用真读者校验它写出的 FITS 时方便）。



事故的机制：这张卡的 **33 个 MSI-X 向量与 32 个核一一对应** —— 看
`/proc/irq/<n>/effective_affinity_list`，它是 `mlx5_compN → cpuN`。注意**不要**看
`smp_affinity_list`：本机对每个向量都写着 `0-31`，那只是允许范围，据此判断会得出
"中断均匀散布、无处可避"的错误结论（我第一版就是这么误判的）。

而 corr 的接收线程也在那些核上：`rx14/15/16` 与 `comp18/19/20` 同核，被 ksoftirqd 抢时间，
60 s 窗口实测它们只拿到 **33-35%** 的核，而别的接收线程有 50-71%。对应的正是事故里丢包
最多的端口（10065/10068/10069）。socket 积压涨到 420 MB 后开始成片丢包：3 小时 lost 3.47 亿、
零填充 10.12%，7 个站报废（水位线被 cap 抬了 85 槽，才保住其余 17 站）。

**算力本身是够的**，缺的是分布：异步版 24 个接收线程合计 12.75 核，非 corr 的 CPU 占用
约 5 核，整机 17.8/32 核；同一窗口 cpu0-3 只用了 2-24%，cp28-31 也是 21-25%。
所以把**一半**中断（comp16 及以后）挪到 cpu0-3 即可，cpu18-27 随之干净：

    tools/pin-mlx5-irqs.sh --dry-run        # 先看会改什么，不需要 root
    sudo tools/pin-mlx5-irqs.sh             # 只挪 comp16+（推荐）
    sudo tools/pin-mlx5-irqs.sh --all       # 33 个向量全挪（4 核承担全部中断，风险更大）
    sudo tools/pin-mlx5-irqs.sh --revert    # 回退

改动不持久，重启即失效；要持久就写一个 oneshot systemd unit 调这个脚本。
这台机器上 `irqbalance` 是 inactive + disabled，所以手工绑定不会被覆盖。

改完盯 corr-web 的「网卡缓冲（RX ring）」卡：**ring 溢出非零说明目标核过载，立刻 --revert**。

顺便：**手工排 `cpu.recv_cpus` 解决不了这个问题**。24 站 + pipeline + 2 writer + monitor
需要 28 个核，而避开 node 0 与当时最忙的 6 个核之后只剩 22 个，怎么排都会有核与中断共享。
先把中断挪开，布局才有腾挪余地。

### 水位线抬升超时就停：`integration.stop_after_lift_seconds`

`zero_fill_limit_frames` 是**事后**指标：默认 5000 万帧 ≈ 34 分钟的满速丢包才停。2026-09-19
的事故跑了 3 小时，最惨的站只有 **839 万帧**，离上限差 6 倍，所以它一直没停（7 小时那次长跑
是唯一被它停过的）。

`stop_after_lift_seconds` 用的是**先行**指标：水位线一旦被抬升（`lift > 0`），就说明调度器
已经在主动放弃某个站，那些帧正在被记 missing，而且**这个状态不可逆** —— 所有站以相同帧率
前进，差距守恒，只有重启能复位。持续超过配置秒数就带原因退出。

- 健康运行时 `lift` 恒为 0；瞬时抖动造成的抬升会在下一次检查时清零，不会累积到阈值
- 实测触发（cap=1 + 占住 cpu18-25 共 45 s，阈值 5 s）：duration 设 120 s，**67 s 就退出**，
  报告写明 `station index 15 has been left behind by 75.07 slot(s)`
- 必须同时设 `max_spread_slots`：没有 cap 就永远不会抬升，配置校验会直接拒绝这种组合
  （一个永远不会触发的停止条件比没有更糟，它看起来像保护）
- 建议值 30 s：足够滤掉抖动，又能把事故的损失从 3 小时压到半分钟

顺手把 `zero_fill_limit_frames` 收紧（5000 万 → 100 万帧）是另一层保险：实测健康运行每站约
**0.83 帧/s**，所以 100 万帧要跑 **14 天**才可能误停，而事故的 839 万帧会稳稳触发。
