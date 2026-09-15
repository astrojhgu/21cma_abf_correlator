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
