# 实时相关机交接文档

## 0. 本轮状态摘要

**24 路真实流量下已经能做到"几乎不丢包、不补零"**（本轮新增 §6.13）：

```text
24 路 tslice（4 卡，block_slots=8，40 s，load≈14）：
  38 行 / 300 个 baseline 文件，每卡 H2D 4.68 GB/s（需求 4.84，97%）
  每站收 7,814,016 包 = 标称 100%；lost 5,131（0.065%）；zero_filled 2,804 帧（0.07%）
  收包线程：28.0 s/站 等包、12.0 s/站 组包（其中 8.6 s 等 staging 槽）—— 30% 忙
```

本轮修掉的是**两个真 bug + 一个被当成"机器噪声"的策略错误**，每一个都有实测定位：

1. **进度发布滞后一个批次 → 与调度器互锁**（§6.13.1）：补零把水位推到槽边界，
   但发布在整批处理完才做，而线程已阻塞在下一槽上。结果是"提交 2 槽后永久停住、
   0 行、也不报错"。修法：槽一完成立刻发布。
2. **`max_in_flight = block_slots-1`（"把环用满"）让接收侧只剩 1 个可写槽**（§6.13.2）：
   每站 16 s 只收到 1.68M/3.12M 包、每行 41% 的帧是补零出来的假数据。
   把接收侧领先量放到 7 槽后收到全部 3.12M 个包、丢 76 个，且 **H2D 吞吐不变**（18.8 vs 18.4 GB/s）。
3. **上一节把丢包归因给"共享机器的噪声"是错的**（§6.12 第 5 条，已在 §6.13.3 更正）：
   同一窗口、同一 load 下，改掉第 2 条后丢包从 41% 降到 0.065%。

**两半都已经是能跑的实现，并且已经在 24 路真实前端流量下端到端跑通过。**
CUDA 侧（C ABI v2）+ Rust 侧（收包 / 拼帧 / 水位线 / 落盘）全部落地，
`--dry-run` 与 produce（GPU 累加 + 写盘）两条路都实测过。

本轮（2026-09-14 下午）修掉两个真问题，都改动行为、都有实测与变异测试佐证：

1. **启动竞态，会让 produce 模式彻底卡死**（§6.4）——这正是之前
   「24 路 produce 0 槽 0 行、没有报错、也不退出」的直接原因；
2. **运行中跳号的补零绕过 staging 槽的所有权检查**（§6.8）——一个会让
   「整槽 256 帧好数据被悄悄写成零、而 `blocks.csv` 仍报 valid 满值」的竞态。
   现在补零逐个槽推进、每步等管线释放，并对单站补零总量设上限。

修完后的实测（§6.5）：

```text
24 路真实前端，40 s produce：
  start frame 6983033344 | 40.00 s | 15049 slots | 39 rows | 3852544 frames
  24 个 station 全部 gaps=0 lost=0 zero_filled=0 late=0，exit 0
  300 个 baseline 文件各 39 行，rows_ms.bin=312 B，blocks.csv=936 行（39×24）
  GPU 1809 MiB / 55→100%，进程 RSS 280 MB、28 线程、372 fd、CPU 21.5/32 核
```

**效率优化的结论（§6.12）**：把"等 ring"从自旋改成阻塞，24 路进程 CPU 从
**21.5 核降到 6.1 核**、吞吐不变；并且用同窗口对照证明：近期观察到的丢包
在**单卡上也同样出现**（load 15~25 的共享机器），不是多卡引入的。

**多卡（§6.10、§6.11）**：一个 handle 背后可以有多张卡，卡数来自配置
（`gpu.devices`，代码里没有硬编码的卡数；`n_devices = 0` 退回单卡）。
正确性已证：8 路回放数据下 **4 卡输出与 1 卡逐字节相同**，相位 τ=2.0000。
效率也量清了：4 卡**纯拷贝聚合 81 GB/s**，但**管线只跑到 60.4 GB/s**（74%），
每卡天花板约 15 GB/s —— 这是 40 路（covering 需每卡 16.8 GB/s）之前要跨过的坎。

累加模型就是最初要求的那个：**GPU 里一块内存反复累加，吐数那一拍取走并清零**，
通过 `corr_row_accumulate` × k + `corr_row_end` × r + `corr_rows_fetch` 实现；
短积分时间下「取数太频繁」由 `rows_per_fetch` 解决（§5.4）。

积分时间定为 **97792 帧 = 1.00139008 s**（不是早先的 97656）：`n_cum` 必须是
`frames_per_slot=256` 的整数倍，256 × 382 = 97792 是离 1 s 最近的合法值，
与 1 s 差 0.139%。注意 1 秒**永远**不是整数个 FFT 帧（帧率 = 800e6/(2·n_freq)，
分母恒为 2 的幂），这个差是数学事实，不是配置错误。

## 1. 项目目标

构建一套面向阵列望远镜原始数据的实时相关机：

- 原始 UDP 数据从 `ens2f0np0` 进入本机。
- 每个站点的数据发往一个独立的多播组和目的端口。
- 接收端按配置选择实际参与的 station。
- 对每个 station 的数据帧做 FFT。
- 对 `j <= k` 的所有 baseline 计算频域互功率谱并积分。
- 输出每条 baseline 的 `complex<float>[ntime][nch]` 文件，供后续离线处理。

当前阶段完成：

1. 原始 UDP 包格式实测验证；
2. 本机系统、网卡、GPU、DPDK、RDMA 环境探测；
3. 核心相关算法的设计决策确认（§4）；
4. 实现层设计决策确认（§5）；
5. **counter 全局同步性长跑实测（§2.5）—— 上一版的风险 #1 关闭**；
6. **socket 收包路径 4 路 120 s 零丢包实测（§2.6）**；
7. **采样率标定 = 800 MHz（§2.7）**；
8. 接口契约、Rust/CUDA 双工程骨架、C 侧与 Rust 侧自检（§6）；
9. **CUDA 侧完整实现（int8→float、cuFFT R2C、SMEM 分块相关、scale、
   拷贝/计算双流），与 numpy 独立参考对拍通过（精度 1.7e-07）**；
10. **PCIe H2D 带宽定量（26.8 GB/s，Gen4 x16）与单卡天花板定量
    （26~28 路，24 路 87.4%）——结论是单卡够用、不需要多卡（§2.9/§2.10）**。

**下一轮起点：见 §7 Phase 5**（最优先是 §8「仍然开放」第 1 条的
`finalize_range` 补零竞态，其次是短积分写盘压测与小时级长跑）。

---

## 2. 已验证事实

### 2.1 网卡和地址

```text
interface: ens2f0np0
PCI:       0000:41:00.0
型号:      Mellanox ConnectX-7 MT2910 Family [ConnectX-7] [15b3:1021]
驱动:      mlx5_core
驱动版本:  24.10-1.1.4
固件:      28.43.2026
速率:      200000 Mb/s
PCIe:      32.0 GT/s x16
MTU:       9000
状态:      UP, PROMISC
NUMA:      node 0
CPU:       0-3
```

本机绑定地址：

```text
单播:
  172.23.0.11/18

当前自动加入的多播组 /32:
  239.1.137.3   239.1.138.3   239.1.139.3   239.1.141.3
  239.1.142.3   239.1.143.3   239.1.144.3   239.1.161.3
  239.1.163.3   239.1.164.3   239.1.165.3   239.1.166.3
  239.1.167.3   239.1.168.3   239.1.169.3
```

### 2.2 目标端口和 station 映射

`rfsoc-init.json` 中有 3 块 RFSoC 板，每块 8 个通道，共 24 个 station/通道。

板卡源 IP 和源端口：

```text
rfsoc-21cma-1: 172.23.0.201, source_port = 23333
rfsoc-21cma-2: 172.23.0.202, source_port = 23333
rfsoc-21cma-3: 172.23.0.203, source_port = 23333
```

完整的 station 归属（**上一版缺这张表**，按 `rfsoc-init.json` 补全；`data_shift_count`
三块板都是 5）：

| board | station |
|---|---|
| rfsoc-21cma-1 (172.23.0.201) | S13 S12 S11 S10 S09 S08 S07 S06 |
| rfsoc-21cma-2 (172.23.0.202) | S05 S04 S03 S02 S01 N01 N02 N03 |
| rfsoc-21cma-3 (172.23.0.203) | N04 W17 W18 W19 N08 N09 N10 N11 |

本机当前实际绑定/加入的 15 路：

| station | group:port | board |
|---|---|---|
| W17 | `239.1.137.3:10037` | 21cma-3 |
| W18 | `239.1.138.3:10038` | 21cma-3 |
| W19 | `239.1.139.3:10039` | 21cma-3 |
| N01 | `239.1.141.3:10041` | 21cma-2 |
| N02 | `239.1.142.3:10042` | 21cma-2 |
| N03 | `239.1.143.3:10043` | 21cma-2 |
| N04 | `239.1.144.3:10044` | 21cma-3 |
| S01 | `239.1.161.3:10061` | 21cma-2 |
| S03 | `239.1.163.3:10063` | 21cma-2 |
| S04 | `239.1.164.3:10064` | 21cma-2 |
| S05 | `239.1.165.3:10065` | 21cma-2 |
| S06 | `239.1.166.3:10066` | 21cma-1 |
| S07 | `239.1.167.3:10067` | 21cma-1 |
| S08 | `239.1.168.3:10068` | 21cma-1 |
| S09 | `239.1.169.3:10069` | 21cma-1 |

JSON 中存在但本机当前未绑定的 9 路：

| station | group:port | board |
|---|---|---|
| S02 | `239.1.162.3:10062` | 21cma-2 |
| S10 | `239.1.170.3:10070` | 21cma-1 |
| S11 | `239.1.171.3:10071` | 21cma-1 |
| S12 | `239.1.172.3:10072` | 21cma-1 |
| S13 | `239.1.173.3:10073` | 21cma-1 |
| N08 | `239.1.148.3:10048` | 21cma-3 |
| N09 | `239.1.149.3:10049` | 21cma-3 |
| N10 | `239.1.150.3:10050` | 21cma-3 |
| N11 | `239.1.151.3:10051` | 21cma-3 |

注意：

- 临时 `join` JSON 中全部 24 路时，确实能收到 24 路数据；
- 但最终参与相关计算的 station 集合由运行时配置文件决定，不能假定 JSON 全部 24 路
  都属于本机处理范围；
- station 集合在进程启动时固定，运行中不动态变化。

### 2.3 UDP 包格式实测

```text
8 + 4096 = 4104 bytes UDP payload

[0:8]    uint64_t little-endian packet counter
[8:4104] 4096 × int8_t time-domain samples
```

实测结果：

```text
UDP payload length: 4104
source IP:         172.23.0.202 / 172.23.0.203
source port:       23333
```

前 8 字节按小端解释为递增 counter，按大端解释则不连续，因此确认：

```text
packet counter = uint64_t little-endian
```

后半部分 4096 字节：有符号二进制补码 int8，转 float32 时直接取 `int(signed int8)`。
实测抽样值集中在 `-4 ... +4`，与 `data_shift_count = 5` 相符。

二层帧长度验证：

```text
14(Ethernet) + 20(IP) + 8(UDP) + 4104(payload) = 4146 bytes
```

接口统计的平均帧长：本机当前 15 路约 4146.19 bytes；临时 join 24 路约 4145.92 bytes。

### 2.4 当前数据流量

```text
单路 W17 实测:      约 195,331 pps，UDP payload 约 801.6 MB/s，counter 基本连续

本机当前 15 路:     约 2.935 Mpps，约 12.169 GB/s，约 97.35 Gbit/s L2
临时 join 24 路:    约 4.693 Mpps，约 19.457 GB/s，约 155.65 Gbit/s L2
```

链路健康状态：

```text
rx_crc_errors_phy = 0
rx_discards_phy = 0
rx_out_of_buffer = 0
rx_pause_ctrl_phy = 0
rx_steer_missed_packets 历史值约 8,494,713，但当前采样窗口内 delta = 0
```

当前没有进程长期监听 `10037-10073` 端口；`ss -lunp` 中未发现相关消费者。

### 2.5 【本轮新增】counter 全局同步性实测 —— 风险 #1 关闭

**上一版把"counter 是全局同步帧号"列为需要长跑验证的假设。已实测确认成立。**

工具与原始日志：`tools/counter_probe.c`、`tools/counter_probe_120s.log`
（用法与判读方法见 `tools/README.md`）。

方法：每 station 一个 C 线程、各独占一核，`SO_RCVBUF=64 MiB`，
每收 16 包发布一次 `(counter, CLOCK_MONOTONIC)`，主线程每 200 ms 采样全部 station。
覆盖三块板：`W17`(21cma-3)、`S01`(21cma-2)、`S06`(21cma-1)、`N04`(21cma-3)。

120 s 结果：

```text
每路收包数          23,509,040（探针自计，含启动前 0.3 s 预跑，除数不精确）
counter gaps        0
lost packets        0
reorder             0
```

**准确的包率**应以两个采样时刻的 counter 差值为准（同一 station 自比，无跨站误差）：

```text
W17: 10050478327 → 10073889447，即 23,411,120 包 / 119.87 s = 195,304.3 pps
```

跨站 counter 极差（`max - min`，单位：包；195,312.5 pps 下 1 包 ≈ 5.1 µs）：

```text
样本数 600（每 200 ms 一个）
min 7   p50 33   p90 46   p95 50   p99 109   max 706
98.7% 的样本极差 <= 60 包（307 µs）
```

**关键在于怎么读这张分布。** 极差里混了两种完全不同的东西：

1. **真实的时钟差**（我们想测的）——若三块板各自独立计时，它会**单调增长**；
2. **读线程积压**（探针自身的伪影）——某个收包线程被抢占几百微秒到几毫秒，
   socket 队列里就积压下几百个包；它恢复后按 FIFO 把旧包读出来，
   于是它"最新发布的 counter"落后于线路上的真实时刻。
   注意 `age(us)` 列会**低估**这种滞后（采样线程先读 counter 再读 timestamp，
   两次读之间线程若又发布了新值，读到的就是"旧 counter + 新 timestamp"）。

两者的决定性差异是：**真实频率差不会自己缩小**。所以用滑动最小值来判别
（窗口最小值近似"无积压"状态）：

```text
窗口(12 s)   1    2    3    4    5    6    7    8    9   10
最小值(包)   7    7    7    7    7   14   12    7    7    7

窗口中位数  33   31   34   34   31   31   30   33   31   34   （平坦）
窗口最大值  59   53   51   60  100   99  706   57  159  130   （积压，会自愈）
```
- 对窗口最小值做线性拟合，斜率 **+0.011 包/s**，外推 1 小时仅 +40 包（+0.2 ms），
  而这还小于该统计量自身 ±7 包的噪声；
- 窗口**中位数**平坦在 30~34 包，没有增长趋势。

> 陷阱提醒：按"三等分中位数"看会得到 21 → 33 → 44，**看起来像漂移**。
> 那只是因为第一段包含了启动后最初几个异常紧凑的样本，而某些积压尖峰
> （如 82.84 s 的 706）落在了末段。必须用窗口最小值或窗口中位数，
> 不要用整段分位数。

因此结论成立：

```text
counter 是三块板共享的全局包计数
相同 counter = 同一采样时刻
跨 station 对齐不需要估计 offset
```

**重要澄清**：上一版 §2.2 里 `W17=9385655416 / W18=9385655436 / S01=9385655421`
差 5~20，一度看起来像"板间固定偏移"。那只是三次测量发生在**不同时刻**，
不是真偏移。不要再按"需要估 offset"去设计。

**遗留**：120 s 只证明了 2 分钟量级。小时级长跑仍未做（§8「仍然开放」第 4 条），
但按"三块板共享同一计数器"这一机制，没有预期会漂移。
复测方法见 `tools/README.md`：**看窗口最小值，不要看全段分位数**。

### 2.6 【本轮新增】socket 收包路径可行性实测

同一份 120 s 数据同时证明了**用内核 socket 收多播、每路独占一核，是可以做到零丢包的**：

```text
4 路同时，每路 195 kpps / 801.6 MB/s
SO_RCVBUF = 64 MiB（本次；配置默认给 256 MiB）
120 s：gaps = 0，lost = 0，reorder = 0，每路 23.5 M 包
```

这直接支撑了 §5.2 的收包选型，也说明扩到 15 路（每路仍是一个独立流）时，
瓶颈在**核数与内存带宽**，而不在 socket 机制本身。

### 2.7 【本轮新增】采样率标定 = 800 MHz

`rfsoc-init.json` 里**没有任何采样率/带宽字段**（只有 `data_shift_count` 和各通道
`adc_attenuation_N`），所以采样率是从实测包率反推的：

```text
Δcounter / Δt = 23,411,120 / 119.87 s = 195,304.3 pps
195,304.3 pps × 4096 samples/packet = 799.97 MHz
```

而 800 MHz / 4096 = 195,312.5 pps，与实测相差 42 ppm。
注意这个差**完全在测量分辨率内**：样本间隔 119.87 s 只打印到小数点后两位，
±5 ms 就是 42 ppm。因此取：

```text
sample_rate_hz = 800,000,000   （对应 RFSoC 4 GSPS ADC ÷ 5）
```

派生量（`n_freq = 4096` 时）：

```text
fft_len           = 2 × n_freq = 8192
每帧包数 P        = 8192 / 4096 = 2
帧率              = 800e6 / 8192 = 97,656.25 fps
bin_width         = 800e6 / (2 × n_freq) = 97,656.25 Hz
```

注意 **帧率与 bin_width 恒等**（一帧跨 `2·n_freq` 个样本，所以 `n_cum` 帧正好跨
`n_cum / bin_width` 秒）。`n_cum = 1000` → 块长**精确 10.24 ms**，97.66 块/s。

配置里 `sample_rate_hz` 用**整数 Hz** 表示。不要写成 `800.0e6`：
YAML 1.1 的浮点指数必须带符号，`800.0e6` 会被解析成**字符串**。

### 2.8 【本轮新增】磁盘与输出吞吐实测

输出量：`n_baseline × n_freq × 8 bytes` 每块，
15 路 = 120 baseline = **384 MB/s**，24 路 = 300 baseline = **960 MB/s**。

实测（`dd` + `oflag=direct`）：

| 目标 | 单流 | 4 并行流合计 | 文件系统 |
|---|---|---|---|
| `/scratch` | 885 MB/s | **1767 MB/s** | NFS `hdd1.internal` |
| `/data` | 773 MB/s | 1422 MB/s | NFS `hdd1.internal` |
| `/home/member/jhgu/...` | 948 MB/s | 1387 MB/s | NFS（**本工作区就在 NFS 上，不是本地盘**） |
| `/data1` | 83 MB/s | **472 MB/s** | idfs 并行文件系统 |
| 本地 `/`（893 G） | 445 MB/s | — | 本地 xfs |

结论：默认输出目录选 `/scratch`。`/data1` 空间最大（911 T）但**撑不住 24 路的
960 MB/s**，不要因为它大就选它。本机另有一块 5.8 T NVMe（`nvme0n1`，xfs）
**未挂载**，要用需要 root。

---

### 2.9 【本轮新增】PCIe H2D 带宽实测 —— 之前把瓶颈搞错了

工具：`corr-cuda/tests/h2d_bw.c`（按真实几何：24 路、每站一个积分块的
chunk 化 2D 拷贝）。

```text
链路（加载中）: Gen4 x16
  linear    26.82 GB/s     一整块连续拷贝
  per-row   25.54 GB/s     每 station 一次线性拷贝
  2d        26.94 GB/s     与实现相同的跨步 2D 描述符
```

**关键纠正**：`nvidia-smi` 在空闲时报 `pcie.link.gen.current = 1`，容易被读成
"链路被限速到 Gen1"。实测加载中是 **Gen4**，那只是 ASPM 降速状态。

所以旧管线只跑到 12–16 GB/s **既不是链路限制，也不是跨步拷贝的代价**，而是
拷贝与 kernel 在同一条流上轮流执行。分离成拷贝流 + 计算流之后，15 路实测
H2D 持续 26.18 GB/s（§2.10），基本吃满链路。

### 2.10 【本轮新增】单卡吞吐实测与天花板

工具：`corr-cuda/tests/bench.c`，1 秒积分（97656 帧/行），
`frames_per_slot=256, chunk_frames=256, block_slots=3, rows_per_fetch=1`。

| station | 串行单流（旧） | 拷贝/计算双流（新） | H2D 持续 |
|---|---|---|---|
| 4 | 30.8% | **18.8%** | 17.1 GB/s |
| 15 | 77.2% | **45.8%** | 26.2 GB/s |
| 24 | 156.9% | **87.4%** | 22.0 GB/s |

（利用率 = 前端帧率 / GPU 达成帧率，100% 表示刚好跟上。）

- **单卡天花板在 26~28 路附近**，24 路有余量；
- `chunk_frames` 扫描（24 路）：64 → 73.3%，128 → 86.1%，256 → 87.3%，
  512 → 87.6%，取 256 即接近平台；
- `frames_per_slot` 在 256~4096 之间对吞吐**没有影响**（拷贝走 2D 描述符，
  与槽大小无关），所以取小值省 pinned 内存：24 路时 151 MiB。

---

## 3. 本机系统环境

### 3.1 OS / CPU / NUMA / 内存

```text
OS:     Rocky Linux 9.3
Kernel: 5.14.0-362.8.1.el9_3.x86_64
CPU:    AMD EPYC 9174F 16-Core Processor
        2 sockets, 16 cores/socket, SMT off, 32 vCPU
NUMA:   8 nodes, 每 node 4 个 CPU
内存:   约 755 GiB
```

NUMA 对应关系：

```text
node 0: CPU 0-3      <- ens2f0np0 所在
node 1: CPU 4-7
node 2: CPU 8-11
node 3: CPU 12-15
node 4: CPU 16-19
node 5: CPU 20-23
node 6: CPU 24-27
node 7: CPU 28-31
```

CPU governor：`performance`。`tuned` 未运行，`irqbalance` 未运行。

启动参数注意：

```text
isolCPUs=0-11 nohz_full=0-11
```

`isolCPUs` 拼写错误，因此 `/sys/devices/system/cpu/isolated` 为空，
**实际 CPU isolation 未生效**；`nohz_full = 1-11`。

### 3.2 【本轮新增】权限边界 —— 这一节解释了很多"看起来可选"的方案为什么不能用

当前用户 `jhgu`（uid 1052，组 `member`），**没有 root，`sudo` 需要密码**。实测：

| 项目 | 实测结果 | 后果 |
|---|---|---|
| `sudo -n true` | `a password is required` | 无任何提权路径 |
| 写 `/proc/sys/...` | `Permission denied` | 不能调 `netdev_max_backlog` 等 sysctl |
| 写 `/proc/irq/*/smp_affinity` | 不可写 | **不能把 mlx5 IRQ 调成 NUMA-local** |
| `SCHED_FIFO` | 需要 `CAP_SYS_NICE`，不可得 | 收包线程**不能设实时优先级** |
| `/dev/hugepages`、`/dev/hugepages-1G` | `root:root 755`，建文件 `Permission denied` | **128 个 1G 大页虽然 free，但用不上** |
| `tcpdump` / `AF_PACKET` | 无权限 | 不能抓包做回放（故自建回放器，§5.7） |
| `/dev/nvidia0..3`、`nvidiactl` | `crw-rw-rw-` | GPU 可用 |
| `ulimit -l` | `unlimited` | `mlockall` / `cudaHostAlloc` 固定内存不受限 |
| `ulimit -n` | `1000000` | 300 条 baseline 文件同时打开没问题 |
| `net.core.rmem_max` | `2147483647` | **无需 root 即可开 256 MiB 的 `SO_RCVBUF`** |

**直接结论：DPDK、AF_XDP 全部出局**（两者都需要 root + hugepages）。
上一版把它们列为候选方案，本轮实测后判定不可用。内核 socket 不是"图省事"，
是在这个权限边界下的唯一可行路径 —— 而 §2.6 已证明它在 4 路下零丢包。

### 3.3 网卡队列 / 中断 / offload

```text
Combined queues: 32
MSI-X mlx5_comp 中断: 32 个
默认 IRQ affinity: mlx5_comp0..31 分别绑定 CPU 0..31
RX/TX ring: 当前 1024，最大 8192
Coalescing: adaptive, RX rx-usecs=32, rx-frames=64
RSS: on, 32 rings 均匀散列
GRO: on     LRO: off     RX checksum: on     ntuple: off
PTP hardware clock: 5
```

由于无 root 不能改 IRQ affinity，**收包线程必然与部分 IRQ 争核**（IRQ 散在 0-31）。
这是 §5.6 绑核策略的约束来源，不是可以优化掉的问题。

### 3.4 HugePages / IOMMU / DPDK（判定不可用，仅存档）

```text
1G hugepages: 128 个，全部 free       <- 但 /dev/hugepages 不可写，用不了
2M hugepages: 4096 个
IOMMU: iommu=pt, amd_iommu=on
DPDK:  23.11.1，/usr/local/bin/dpdk-testpmd, dpdk-devbind.py
       网卡仍绑定在 mlx5_core，未绑 vfio-pci
```

### 3.5 RDMA / GPU

```text
RDMA:
  mlx5_ib、ib_uverbs 已加载
  rdma link: mlx5_1/1 state ACTIVE, netdev ens2f0np0
  ibv_devinfo: mlx5_1 active_mtu = 4096

GPU: 4 × NVIDIA L20，每张约 46 GiB，compute capability 8.9，CUDA toolkit 12.4
  实测（nvidia-smi --query-gpu）：index 0..3 均可见且设备节点可读写

nvidia-smi topology:
  GPU0/GPU1 -> NUMA 2, CPU 8-11
  GPU2/GPU3 -> NUMA 7, CPU 28-31
  mlx5_1 (ens2f0np0) 与 4 张 GPU 均为 SYS（跨 socket）

GPUDirect: nvidia_peermem 模块可用（550.127.08），当前未加载
```

本方案不走 GPUDirect / RDMA，`nvidia_peermem` 不需要加载。

### 3.6 编译与工具链

```text
GCC/G++: 11.4.1
CMake:   3.26.5
NVCC:    12.4.131
CUDA:    /usr/local/cuda-12.4，libcufft.so.11 / libcudart.so.12 齐备
FFTW3 / HDF5 / TBB / libnuma / libpcap / OpenMP: 可用
Python:  numpy 1.20.1（无 cupy / pycuda / numba / mpi4py / scipy / h5py）

Rust:    【本轮安装】原本未安装
         rustup + stable 1.98.1（2026-09-01）装入 ~/.cargo
         注意：安装脚本可能在最后一步中断，若报
         "rustup could not choose a version of rustc"，执行 `rustup default stable`
```

---

## 4. 核心算法决策（已冻结，实现必须遵守）

以下决策已通过 grilling 确认。若要修改，需要重新确认。

### 4.1 输入与帧组装

- 每个 UDP 包 = 4096 个 `int8` 时域实样本。
- `int8` 按有符号二进制补码直接转 `float32`。
- `data_shift_count = 5` 不参与相关机 FFT 计算，只作为前端量化/动态范围元数据。
- `N_freq` 运行时设定，可取 2048、4096、8192 等。
- FFT 帧长度为 `2 × N_freq`。
- 每帧需要的 UDP 包数 `P = N_freq / 2048`：

```text
N_freq = 2048 -> P = 1 包/帧
N_freq = 4096 -> P = 2 包/帧     <- 默认
N_freq = 8192 -> P = 4 包/帧
```

- 帧按 packet counter 边界连续拼接；帧之间不重叠。
- FFT 前使用矩形窗，不做窗函数修正。
- **帧的坐标由 counter 直接给出**（§2.5 已实测支撑）：

```text
frame_id      = counter / P
slot_in_frame = counter % P
frame_index   = frame_id % N_cum      （块内行号）
```

### 4.2 FFT

- 类型：R2C，实数到复数。
- 输入：`2N_freq` 个实数 float32 样本。
- R2C 输出 `N_freq + 1` 个非负频点。
- 最终丢弃 Nyquist bin，保留 `0(DC), 1, 2, ..., N_freq-1`，实际输出频点数 `N_freq`。
- FFT 不做 `1/N` 或 `1/sqrt(N)` 归一化。
- FFT 输出 complex64。
- 输出 bin 顺序为 R2C 原始顺序，不做 `fftshift`。

### 4.3 相关计算

- 每个 FFT 帧中，每个 enabled station 只做一次 FFT：`Y_j = FFT(x_j)`。
- **不允许按 baseline 重复做 FFT。**
- baseline 包含 autocorrelation 和 cross-correlation：`j <= k`，
  `N_baseline = N_station(N_station+1)/2`。
- 每个 baseline 的累加公式：`corr[l] += Y_j * conj(Y_k)`。
- `corr` 使用 complex64 累加。
- 最终只输出频域互功率谱，不做 iFFT。

### 4.4 积分、缺帧、对齐

积分：

```text
N_cum = FFT 帧数，不是 UDP 包数
每个积分行 = 连续 N_cum 个 FFT 帧
积分行之间不重叠
默认 N_cum = 97792 帧 = 1.00139008 s（= 256 × 382；1 秒永远不是整数帧，见 §0）
```

注意 N_cum 只是**积分长度**：它不决定 pinned staging 的大小（那是
`frames_per_slot`），也不决定吐数/写盘频率（那是 `rows_per_fetch`）。
三者解耦是 C ABI v2 改动的核心目的，见 §5.4。

缺帧：

```text
缺失 station 的帧补零（Rust 侧写零，GPU 侧无感）
继续更新所有 baseline
记录缺帧 flag（blocks.csv 的 valid_frames 列）
最终统一除以 N_cum，不按有效帧数归一化
```

注意：不做逐 baseline 的有效帧数归一化；缺帧造成的幅度偏置由 flag 解释。

时间对齐：

```text
packet counter 是全局同步帧号   <- §2.5 已实测确认
相同 counter = 同一采样时刻
不引入额外 offset
```

几何延迟和条纹旋转：

```text
由 RFSoC / 前端完成
相关机侧只做 FFT、Y_j * conj(Y_k) 和积分
不再补相位
```

### 4.5 station 配置

使用独立的相关机配置文件。每个 station 至少包含
`name / group_ip / dest_port / interface / enabled / order`。

规则：

- 只统计 `enabled = true` 的 station；
- 按配置的 `order` 升序压紧成下标 `0..N_station-1`，该下标就是 baseline 公式里的 `j/k`；
- `N_station = N_enabled`，`N_baseline = N_station(N_station+1)/2`；
- station 列表在进程启动时固定，运行中不允许增删；
- 修改配置需要重启程序。

### 4.6 输出

每条 baseline 一个输出文件：`complex<float>[ntime][nch]`。

```text
nch  = N_freq
ntime = 积分块序号
```

- 每个积分块结束后，向每条 baseline 文件追加一行 `complex64[N_freq]`；
- 运行期间追加写入，停止后再做离线处理；
- 频谱顺序：`bin 0 = DC`，`bin 1 = Δf`，…，`bin N_freq-1`，不做 `fftshift`。

启动时生成全局 metadata 文件，至少包含：

```text
station list / order
group_ip:port
N_freq
sample_rate / bin_width
baseline index -> (j,k)
Nyquist dropped
window function
FFT normalization
N_cum
config version/hash
```

---

## 5. 【本轮新增】实现决策（grilling 逐条确认）

§4 是"算什么"，本节是"怎么算/怎么跑"。每条都是主动选择，括号内是理由。

### 5.1 工程与构建

- 目录：`corr-rs/`（Rust）+ `corr-cuda/`（CMake CUDA）+ `config/correlator.yaml`，
  两个工程**同级**。
- **边界**：CUDA 侧编译成导出 **C ABI** 的 `libcorr_cuda.so`，Rust 用 `build.rs`
  驱动 cmake 构建并经手写 `extern "C"` 链接。
  （C++/CUDA 侧可自由用 cuFFT/C++，Rust 侧只面对一个稳定头文件；两边可独立编译测试。）
- **构建耦合**：`cargo build` 一键构建全部；`build.rs` 调 `cmake` crate，
  `.so` 路径经 `cargo:rustc-link-search` 传出，rpath 同时设 **OUT_DIR 绝对路径**（开发）
  与 **`$ORIGIN`**（部署）。
- **不用 bindgen**：绑定手写，`struct_size` / `abi_version` 不匹配时**运行期报
  `CORR_ERR_ABI`**，而不是静默读错字段。
- Rust 侧**不引入 async 运行时**；单 crate + `src/bin/replay.rs`。
  依赖只用了 `anyhow / clap / libc / serde / serde_yaml`。
- `corr-cuda` 的 Debug 构建在 CMake 层被强制成 Release
  （拿 Debug 版 .so 做吞吐测量是个陷阱）。

### 5.2 收包

- **1 station = 1 OS 线程 + 1 socket**，绑核，busy-poll。
  （代码最简；§2.6 证明每路独占一核可零丢包；扩到 N 路线程数=N，CPU 占用可控。）
- socket 选项：**阻塞 `recvmmsg`（batch 32）+ `SO_BUSY_POLL`（50 µs）+
  `SO_RCVBUF` 256 MiB**。
  （`rmem_max` 已是 2 GiB，无需 root；`net.core.busy_poll` 默认为 0，但 per-socket
  设置不受其限制。）
- `bind((group_ip, dest_port))` + `IP_ADD_MEMBERSHIP`，接口地址 `172.23.0.11`。

### 5.3 数据通路与显存

- **收包线程自己拼帧**，直接写进 **pinned int8 staging 环**；
  没有独立拼帧线程，也没有逐帧消息传递。
  （pinned 区按 `[station][slot][frame][sample]` 组织，每个收包线程只写自己那一路，
  无锁、无跨线程通信、零额外拷贝。）
- **staging 只按 `frames_per_slot` 分配，与积分长度无关**（§5.4）。
  槽大小取 256 帧：4 路 24 MiB / 15 路 90 MiB / 24 路 151 MiB pinned。
- **同步原语**：per-station `AtomicU64` progress（recv 线程 Release store，
  pipeline 线程 Acquire load），水位线 = 各站最大 frame_id 的**最小值**。
  slot 数默认 3（triple buffer），GPU 还在拷贝某个 slot 时 recv 线程可以继续写下一个。
- **`int8 → float32` 必须在 GPU 侧做**：15 路是 12.0 GB/s int8，
  若主机侧转成 float32 再 H2D 就是 48.1 GB/s，而链路实测上限 27 GB/s。
  所以 staging 里存 **int8**。
- **H2D 粒度是一个 staging 槽**（15 路、256 帧时 60 MiB），一次 2D 跨步拷贝
  把各 station 的帧摆成设备上的连续布局；实测跨步与线性同速（26.9 vs 26.8 GB/s）。
- **行边界**：全局水位线跨过 `frame_id` 为 `N_cum` 整数倍的点即关闭一行；
  带**超时脱困**（默认 200 ms 无新帧则该站不再阻塞水位线，其缺失帧按补零处理）。
- **启动对齐**：第 0 行对齐到"所有站都已越过的第一个 `frame_id % N_cum == 0`"
  的全局网格点，此前的帧全部丢弃。
  （因为 counter 全局，每次运行的行边界落在同一张网格上，跨运行可直接拼接。）

### 5.4 CUDA 侧

**累加模型（C ABI v2，§6.2）**：显存里放一块累加内存，反复把 staging 槽交给
GPU 累加进去；到吐数那一拍取走并清零。**不是**「一次提交一个积分块」。

- 积分行（输出行）由多次 `corr_row_accumulate(slot, frames)` 累加而成；
- `corr_row_end(frames)` 把该行除以帧数并推进到下一行；
- `corr_rows_fetch()` 把暂存的若干行**一次 D2H 取回**，随后清零。

**为什么要一个"行栈"（`rows_per_fetch`）**：把积分长度与吐数/写盘频率解耦。
1 秒积分时取 1 即可；若将来用 10 ms 积分，每行都取就是每秒 100 次 D2H
（24 路 983 MB/s）和每秒 100 次唤醒 writer——把本项设为 round(1 s / 积分长度)
就能把节奏拉回 1 Hz，而积分长度本身不用改。

**为什么 staging 与积分长度解耦**：staging 只按 `frames_per_slot` 分配，
所以积分从 10 ms 改成 10 s，pinned 内存一个字节都不多占。旧模型把整行放在
pinned 里，1 秒积分意味着 24 路 58 GB pinned——这是改 ABI 的直接原因。

**为什么必须分离两条流**：拷贝（copy_stream）与 kernel（compute_stream）分开，
中间用 event 串接。实测这是 15 路从 77% 降到 46%、24 路从 157% 降到 87% 的原因
（§2.9/§2.10）。设备缓冲环的下标是**全局 chunk 计数器**，不是槽内序号——
用槽内序号时，`frames_per_slot == chunk_frames`（即每槽只有一个 chunk）会让所有
拷贝都落到同一个缓冲，重叠静默消失（实测退化回 157%）。

- **GPU 分配**：`station → gpu_id` 写在 YAML。实测单卡 24 路 87.4%，
  **不需要多卡**；多卡反而带来跨 GPU baseline 的难题（见 §8）。
- **缺帧由 Rust 侧写零**，CUDA 侧完全无感，只按 `corr_row_end(frames)` 给的
  帧数归一化。
- **FFT**：chunk 批量 out-of-place `cufftPlanMany`（R2C，`n = 2·n_freq`，
  `batch = n_station × chunk_frames`），输出 stride 固定 `n_freq+1`
  （保留 Nyquist 占位，corr 侧跳过）。
  计划的 batch 是一个完整 chunk，所以最后一个不满的 chunk 会多算几个变换，
  其输入是上一次留下的有限值，corr kernel 不会去读它们的输出。
- **相关 kernel**：**共享内存分块**——每 block 管 16 个频点，先把
  `Y[所有 station][frame_chunk]` 搬进 SMEM，再让各 baseline 线程在 SMEM 里
  循环累加。全局 `Y` 只读一遍。
  （朴素写法"每线程一个 (baseline,freq) 直循环"会把 `Y` 读 `N_station` 遍，
  15 路约 7.9 GB/块，逼近 L20 显存带宽上限。）
- **累加器**：`acc[rows_per_fetch][baseline][n_freq]` 连续 complex64；
  行内跨 chunk 累加，行末 scale kernel 乘以 1/frames。
  一次 `corr_rows_fetch` 只做一次连续 D2H。

### 5.5 落盘与背压

**1 秒积分把这一整节的量级改变了**：15 路输出 3.93 MB/s、24 路 9.83 MB/s
（不是 384/960 MB/s——那是 10 ms 积分的数字）。§8 风险 5 因此消失，
`/data1` 也重新可用。

- **每条 baseline 一个 raw 文件**，启动时定位到最后一行末尾再顺序写
  （**不用 `O_APPEND`**：续写要先把长度对齐到整行，见 §6.7），
  `complex64[n_freq]` 一行，`ntime` 由文件长度隐式给出；
  `ntime` 由文件长度隐式给出；
- 启动时写 `metadata.yaml`（station/order、`group_ip:port`、`N_freq`、
  `sample_rate`/`bin_width`、baseline→(j,k)、Nyquist dropped、window、
  FFT normalization、积分长度、config hash）；
- 另写 **`blocks.csv`**，逐行（积分行）记录 `frame_id` 区间与每站有效帧数
  （缺帧 flag 的载体）；
- **背压**：有界内存队列 + 独立 writer 线程；队列满则**阻塞并告警**，不静默丢块。
  现在每秒只有一块（9.8 MB @24 路），队列几乎不会积压。
- 输出目录默认 `/scratch/<run_name>`（§2.8 实测最快），`run_name` 为空时自动生成
  `corr_YYYYmmdd_HHMMSS`（UTC）；
- **退出**：SIGINT/SIGTERM → 停止收包 → **丢弃未写满的积分行**
  （保证每条 baseline 文件行数严格等于 `ntime`）→ 排空已算完的行 → fsync →
  补完 `blocks.csv` 与 metadata。

### 5.6 绑核

- YAML 显式配置：`recv_cpus`（第 i 个 enabled station 用 `recv_cpus[i]`）、
  `pipeline_cpu`、`writer_cpus`、`monitor_cpu`。
- 默认从 node0（CPU 0-3）起按序分配。
- 启动时校验**所有实际使用的核互不重复**（否则两个 busy-spin 线程抢一个核，
  整个时序故事就是虚构的）。
- 约束来源：网卡在 node0 只有 4 核，mlx5 IRQ 却散在 0-31，且**无 root 不能调 IRQ
  affinity / 不能设 SCHED_FIFO**（§3.2）。所以"与 IRQ 争核"避不开。
  哪种绑法更好（全贴 node0 vs 散到其他 node 跨 xGMI 读）留待 Phase 2 实测，
  改 YAML 即可。

### 5.7 监控与验证

- **监控**：**尚未实现**（见 §8 第 10 条）。目前的可观测性 = 运行结束时的
  stdout 报告（每站 pkts/gaps/lost/zero_filled/late/before_start/recoveries、
  槽数、行数、slot-wait spins）+ 逐行 `blocks.csv`。
  原计划：每 1 s 一行 JSON 到 stderr（per-station pps / gaps / lost /
  partial-frames、已完成块数、水位线滞后、写盘队列深度与吞吐、丢块数、
  各 GPU 显存、端到端块延迟），可选追写 `stats.csv`，不引入外部依赖。
- **验证**：自建 UDP **回放器**（`corr-replay`，`corr-rs` 的一个 bin），
  可注入丢包 / 乱序 / 跳号 / 站间 counter 偏移 / 站点停发 / 采样延迟；
  **它拒绝加载实时配置**（24 站 enabled 且看起来就是生产配置时直接报错退出），
  免得合成流量打进真实多播组、和前端 counter 交错。
  单测覆盖拼帧 / 水位线 / 补零 / 块边界 / 槽环；
  小参数 golden 端到端（小 `n_freq` + 小 `n_cum`，输入已知正弦，
  检查互相关峰值位置与 autocorr 实虚部）。
  （本机不能抓包，真实数据不可回放，这是唯一能确定性测试的路。）
- 另有 `--dry-run` 诊断模式（只收包统计、不提交 GPU、不落盘）。

### 5.8 默认配置的选择

- **24 路全部写进 YAML**（含 `group_ip/dest_port/board/source_ip/order`），
  但只 `enable` **W17 / S01 / S06 / N04** —— 覆盖 21cma-3 / 21cma-2 / 21cma-1
  三块板，且正是 §2.5 实测过 counter 同步的那 4 路。→ `N_baseline = 10`，
  1 秒积分下输出 0.328 MB/s。扩展时只改 `enabled` 与 `order`
  （24 路 → 300 条 baseline → 9.82 MB/s，已实测，§6.5）。
- `N_cum` 只给**帧数**（默认 97792 = 256 × 382），不给秒数
  （避免换算引入非整数歧义；而且 1 秒永远不是整数帧）。
- `sample_rate_hz` 用整数 Hz。

---

## 6. 当前代码状态

### 6.1 文件树

```text
correlator/
├── HANDOFF.md                    本文档
├── phase_check.png               相位验证图（τ=0/1/2 采样延迟，见 §6.3）
├── rfsoc-init.json               前端板卡原始配置（代码不读它，是 station 表的来源）
├── config/
│   ├── correlator.yaml           实时配置（24 路 + 全部参数 + 校验规则注释）
│   └── correlator-test.yaml      回放测试配置（239.1.200.x 独立多播组，n_cum=1024）
├── tools/
│   ├── counter_probe.c           跨站 counter 同步探针（§2.5）
│   ├── counter_probe_120s.log    120 s 长跑原始输出
│   ├── phase_check.py            从落盘文件拟合相位斜率，验证延迟（§6.3）
│   ├── compare_runs.py           两个 run 逐基线逐字节对拍（§6.10 用它证明 4 卡≡1 卡）
│   ├── stress_integration.sh     不同积分时间的写盘压测脚本
│   └── README.md                 用法与判读方法
├── corr-cuda/                    CMake + CUDA 工程（151 行 CMakeLists）
│   ├── include/corr_cuda.h       ★ C ABI v2 契约（两工程之间唯一的接口，280 行）
│   ├── src/corr_cuda.cpp         ABI 层：校验、上下文、pinned staging、槽状态机（396）
│   ├── src/kernels.cu            设备实现：转换、cuFFT、相关、scale、双流管线（737）
│   ├── src/gpu_internal.h        两层之间的内部接口（75）
│   └── tests/
│       ├── abi_smoke.c           C 侧自检（与 Rust 无关）
│       ├── gen_golden.py         numpy 独立参考
│       ├── golden.c              对拍驱动
│       ├── bench.c               吞吐/利用率
│       ├── h2d_bw.c              PCIe H2D 带宽与访问模式（单卡）
│       ├── multi_h2d_bw.c        ★ 多卡并发 H2D 聚合（§6.10 的 81 GB/s）
│       └── multi_device.c        ★ 多卡正确性：4 卡输出必须与 1 卡逐字节相同
└── corr-rs/                      Rust 工程
    ├── Cargo.toml, build.rs      驱动 cmake，rpath = OUT_DIR + $ORIGIN
    ├── src/
    │   ├── lib.rs                模块地图（48）
    │   ├── config.rs             YAML schema、跨字段校验、派生 RunPlan（985）
    │   ├── frame.rs              counter→frame 算术、缺帧判定（纯逻辑 + 单测，452）
    │   ├── net.rs                多播 socket、recvmmsg 批量、绑核（427）
    │   ├── assemble.rs           每站拼帧器 + staging 槽环 SlotRing（1092）
    │   ├── sched.rs              水位线、行关闭、脱困、补零上限（972）
    │   ├── writer.rs             baseline 文件 + rows_ms + blocks.csv + metadata + 续写（826）
    │   ├── run.rs                ★ 启动顺序、收包线程、管线线程、报告（945）
    │   ├── ffi.rs                C ABI v2 裸绑定（手写，不用 bindgen，152）
    │   ├── gpu.rs                C ABI 安全封装（Correlator，235）
    │   ├── main.rs               CLI：`--check` / `--print-plan` / `--dry-run` / 正式运行
    │   └── bin/replay.rs         合成回放器
    └── tests/abi.rs              对着真实 .so 的集成测试（253）
```

`run.rs` 顶部注释写明了**启动顺序**以及它为什么是现在这个顺序——那是 §6.4
那个 bug 的修复本身，不是风格问题。

### 6.2 C ABI v2（`corr-cuda/include/corr_cuda.h`）

**v1 → v2 的原因**写在 §5.4：v1 的「一次提交一个积分块」让 staging 随积分长度
线性膨胀（1 秒积分 24 路 = 58 GB pinned），而且拷贝与计算无法重叠。

```c
typedef struct corr_config {          /* abi_version / struct_size 双重校验 */
    uint32_t abi_version, struct_size;
    int32_t  device_id, n_station, n_freq;
    int32_t  frames_per_slot;         /* 一个 staging 槽放多少帧        */
    int32_t  chunk_frames;            /* 设备侧每次 launch 处理多少帧    */
    int32_t  block_slots;             /* staging 环大小，>= 2           */
    int32_t  rows_per_fetch;          /* 显存里暂存几行再一次性取回      */
} corr_config;

typedef struct corr_staging {         /* base 已偏移到该 station        */
    int8_t *base;
    size_t  frame_stride;             /* 2*n_freq                       */
    size_t  slot_stride;              /* frames_per_slot*frame_stride   */
    size_t  total_bytes;              /* block_slots*slot_stride        */
} corr_staging;
```

生命周期（就是 HANDOFF 作者描述的「GPU 里一块内存反复累加，吐数那拍取走并清零」）：

```c
corr_create(...)                                  /* 累加内存清零           */
corr_staging_info(ctx, g, &st)                    /* 收包线程写字的地方     */
corr_row_accumulate(ctx, slot, frames)   * K      /* 累加，异步             */
corr_row_slot_free(ctx, slot)                     /* H2D 完成，槽可重用     */
corr_row_end(ctx, frames)                * R      /* 该行除以帧数           */
corr_rows_wait(ctx)                               /* 所有已关闭的行算完     */
corr_rows_pending(ctx, &n)
corr_rows_fetch(ctx, out, bytes, &rows)           /* 一次 D2H + 清零        */
corr_destroy(ctx)
```

**写入地址**（station `g`、槽 `s`、帧 `f`）：

```text
base + s*slot_stride + (f % frames_per_slot)*frame_stride + (counter % P)*4096
```

`corr_baseline_pairs()` 仍是 baseline 编号的**唯一定义**，Rust 侧不重新实现，
`--check` 与集成测试都断言两侧一致。

### 6.3 已验证 / 未实现

| 验证 | 结果 |
|---|---|
| C 侧 `ctest`（5 项：ABI 冒烟 + 2 组 numpy 对拍 + 生成器） | **5/5 通过** |
| Rust 单测 + 集成测试 | **79 通过 / 0 失败**（72 + 7） |
| `cargo clippy --all-targets` | 无警告 |
| `corr --check` 端到端 | 全绿（含 abi v2 版本自检） |
| golden 精度（vs numpy） | 相对 RMS **1.6e-07 / 1.8e-07**，float32 舍入水平 |
| golden 变异测试 | 期望值乘 0.5 → 正确失败（rel_rms=1.0） |
| 24 路 dry-run（真实前端 4.69 Mpps / 155 Gbit/s） | 24 站全部 `gaps=0 lost=0 reorder=0` |
| **24 路 produce（真实前端，40 s）** | **15049 槽 / 39 行 / 300 文件一致 / 0 丢包 / exit 0**（§6.5） |
| 相位验证（回放器，τ=0/1/2 采样） | 拟合 τ = 0.0000 / 1.0000 / 2.0000，最大残差 0.0002 rad |
| 续写（resume） | 39 → 78 → 117 行，四个文件计数始终一致 |
| 注入 2000 包连续丢包（4 槽跳变） | 行数准点、`valid=24 zero_filled=1000` 如实记录、exit 0（§6.8） |

**实现过程中抓到的真 bug**（都写进了对应代码的注释，避免复现）：

1. **`corr_staging_info` 的 `total_bytes` 语义**：`base` 已偏移到该 station，
   但 `total_bytes` 仍报整区大小 → 调用方遍历必然越界。写 bench 时直接段错误。
   现在 `total_bytes` 是该 station 的区间长度。
2. **不满的槽会把残留数据当数据相关**：`corr_row_accumulate` 原本没有帧数参数，
   库按整槽处理。一行最后一槽只填 8/64 帧时，多出的 56 帧是上次的残留。
   4 路的用例之所以通过纯属侥幸（它的最后一槽是全新槽，残留是零，而零帧对互相
   关贡献恰好为 0）。现在必须显式传帧数。
3. **设备缓冲下标用槽内序号导致重叠静默失效**：当 `frames_per_slot ==
   chunk_frames` 时下标恒为 0，所有拷贝挤在同一缓冲，利用率退化回 157%。
   改成全局 chunk 计数器后，24 路参数扫描才变得单调（73/86/87%）。
4. **`SlotRing` 没给「本 run 的起始槽序号」做过 primed**（§6.4 的一半）：
   环按「什么都没释放过」初始化，而真实起始 frame 在 5e9 附近、槽序号在 2e9 附近，
   于是第一个包会等一个永远不会到来的释放。现在 `SlotRing::primed(slots, first)`。
5. **`SlotRing::wait_free` 没有出口** → Ctrl-C 之后收包线程仍在自旋，
   join 永久挂起：实测留下一个烧两个核 13 分钟的僵尸进程。
   现在 `abort()` 会释放所有等待者。
6. **脱困判据用绝对超时**：站在两个槽之间被连续误判为停发，
   实测 82 次恢复 / 20992 帧被补零。改成按**收到包的物理时间**判活（§5.7）。
7. **背压被当成静默**：ring 满了的站在 `wait_free` 里无法刷新存活时间戳，
   被误判停发（实测 28 次误恢复 / 3606 帧好数据被丢）。
   现在等待 ring 期间也算「活着」。
8. **GPU staging 与本地 staging 是两块内存**：dry-run 用本地那块，
   但 produce 时 assembler 写本地、GPU 读自己那块 → 一切正常但落盘全是零。
   现在统一用 `corr.staging_all()`。
9. **续写时 sidecar 没有 seek**：从 0 覆盖写，98 行频谱配 49 个时间戳；
   `rows_ms.txt` 还会越写越长。现在 sidecar 先 seek，txt 由 bin 重建。
10. **`baseline_bytes` 曾被算成 `per_baseline_row * n_cum`** → 续写扫描永远认为
    0 行可续。`blocks.csv` 也曾每 baseline 写一行（应为每 station 每行一行）。
11. **回放器两个 bug**：`((x>>3)%9) as i8 as u8 - 4` 下溢；
    缺 `IP_MULTICAST_IF` 导致包从错误网卡出去。
12. **运行中跳变的补零绕过 ring 守卫**（§6.8）：一趟扫完可能正好写在管线
    正在拷的物理槽上，且会重置到达位图 → **整槽 256 帧**被清零而 `valid` 仍报满值。
    现在逐个槽推进 + 每步等 ring + 单站补零上限。

### 6.4 本轮修复：produce 模式的启动卡死（已修，已实测）

**症状**：24 路 produce 启动后永远停在 `0 slots / 0 rows`，没有报错、没有退出；
同一份配置 dry-run 完全正常（24 站零丢包，见 §6.3）。station 统计是
`lost=492696, zero_filled=210926`。

**根因**（两个，缺一不可）：

1. **启动顺序错了。** 原来的顺序是「开 socket → 嗅探首包 → 建 CUDA 上下文 →
   建 300 个输出文件 → 起收包线程」。中间那两步要几百毫秒到几秒，
   而这段时间**没有任何线程在收包**：每路 `SO_RCVBUF=512 MiB`，
   按真实速率 0.6 s 就填满，填满后内核丢的是**最新**的包。
   于是收包线程真正开始读时，socket 里最早的一个包已经是起始 frame 之后几秒的
   位置了。
2. **起始 frame 是一个预测值，而槽的可写性由它决定。** 起始 frame 在起线程之前
   算好，`SlotRing` 也按它 primed；第一个包落在 $\text{start} + k$ 槽时，
   `wait_free` 要等 $\text{start}/256 + k - 3$ 号槽被释放，
   而管线要从 $\text{start}/256$ 才开始提交——**永远不会发生**。
   此时全机状态是「一个槽都没提交、一个字节都没读」，所以既不是 GPU 慢
   也不是站在静默，补零/脱困逻辑都够不着它：纯粹的互锁。

**修复**（三处，都在 `run.rs` / `assemble.rs`）：

1. **把慢的都挪到开 socket 之前**：`run::prepare_output()` 先建 CUDA 上下文和
   输出文件，`main.rs` 拿到它之后才 `open_sockets()`。日志里现在能直接看到顺序
   （`prepared ... before any socket was opened`）。
2. **嗅探返回「当前」counter 而不是「第一个」counter**：所有站都收到过首包之后
   再多排空 `SNIFF_SETTLE=5 ms`，返回每站最后一个 counter。否则各站首包可能
   相差数秒，最慢那站会被安排到一个它早就走过的起始 frame 上。
3. **兜底：`SlotRing::reprime()` + 拒绝不可救药的启动跳变。**
   - ring 在**管线取走第一个槽之前**允许把窗口前移（`window_open`）：
     此时一个字节都还没交给消费者，所以把某个物理槽重新声明为可写是安全的；
     管线一旦提交第一个槽就 `close_window()`，窗口冻结。
   - 跳变若**超过环本身的大小**（`block_slots × frames_per_slot` 帧），
     补零会把第一个真实包所在槽覆盖掉（管线会把后来的数据当早期帧相关出去），
     所以直接报 `AssembleError::StartupGap` 并干净停止，而不是补零。
     这条错误现在是 fatal：收包线程置 stop 标志、管线收尾、exit code 1，
     并打印一句能直接照做的说明（日志统一用英文，与代码一致）。

**验证方式**（都做了，不是推演）：

- **变异测试**：临时把 `reprime` 调用去掉 → 新单测**挂死**（timeout 20 s 杀死），
  正是这个 bug 的原始症状；把 `close_window` 改成空函数、把跳变上限去掉 →
  对应单测各自失败。三处守卫都证明是承重的。
- **故意制造陈旧起始 frame**：临时 `start_frame -= 2*n_cum`，回放器下运行 →
  两个站各自报 1619/1620 帧跳变、`0 rows`、exit 1，**毫秒级停止**，
  不再挂死。
- **真实 24 路**：见 §6.5。

### 6.5 24 路实测数据（真实前端，本轮）

```text
$ ./corr-rs/target/release/corr --config /tmp/correlator-24.yaml \
      --output-dir /scratch/corr24-fix --duration 40
prepared   : device context and 300 output file(s) in 511.3 ms, before any socket was opened
opening 24 multicast sockets   (每路 SO_RCVBUF=512 MiB granted, SO_BUSY_POLL=no(EPERM))
draining every station until all of them are seen (up to 45s)
start frame 6983033344 | 40.00 s | 15049 slots | 39 rows | 3852544 frames
  stall: 0 recoveries, 0 frames zero filled against a limit of 488281
  output: 39 rows in /scratch/corr24-fix/corr_20260914_132140
  gpu: corr_cuda 0.1.0 abi=2 cudart=12040 driver=12040 cufft=11201 arch=sm_89 target=L20
  W17   pkts=7807072  gaps=0 lost=0 zero_filled=0 late=0 before_start=101023
  ...（24 站全部一样，before_start 约 98k~101k，progress 全部 6986886xxx）
exit=0
```

- 7,806,944 包 / 40 s = **195,174 pps/站**，与前端标称 195,312.5 相符；
  说明这批流量是**真实前端**而不是回放器（回放器会拒绝加载实时配置，见 §5.7）。
- `before_start` 约 10 万包/站 = 约 0.5 s：就是「起始 frame 对齐到行网格」
  主动丢掉的那段，属预期，不是丢包。
- 落盘一致性：300 个 baseline 文件各 39 行（1,277,952 B = 39×32768），
  `rows_ms.bin=312 B`（8×39）、`rows_ms.txt=39` 行、
  `blocks.csv=936` 行（39×24），全部 `valid=97792 zero_filled=0`，合计 366 MB。
- 资源（40 s 内每 4 s 采样）：进程 RSS 稳定 280 MB、28 线程、372 fd、
  GPU 1809 MiB；GPU 利用率 55% → 100%；进程 CPU **21.5/32 核**（单独测 8 s 窗口）。
- 另跑 8 s / 12 s / 20 s 短测，行数分别 7 / 11 / 19，全部按 `1.0014 s/行`
  准点关闭，`slot-wait spins` 4.5e9/8 s —— **说明 24 路时瓶颈已经在 GPU 侧**
  （收包线程在 ring 上自旋等待），CPU 那 21.5 核就是自旋花掉的。
  报告里新增了这一行的汇总，未来一眼就能看出是 GPU 还是网络在卡。


### 6.6 怎么跑

```sh
cd corr-rs
cargo build --release            # build.rs 会自动 cmake 构建 corr-cuda 并接好 rpath

# 1) 只校验配置 + C ABI（不开 socket、不碰 GPU 显存，随时可跑）
./target/release/corr --check
# 2) 打印 24 路 / 300 条 baseline 的派生参数
./target/release/corr --print-plan
# 3) 只收包、走完拼帧与水位线（不碰 GPU、不落盘），用来诊断链路
./target/release/corr --dry-run --duration 10
# 4) 正式跑：GPU 累加 + 落盘。Ctrl-C 停止，文件保证是整数行
./target/release/corr --config config/correlator.yaml
# 5) 续写：同一个 run_name + 同一份配置，接着最后一个完整行往后写
./target/release/corr --config config/correlator.yaml --run-name corr_20260914_132140
```

回放测试（**不要**指向实时配置：回放器会拒绝，免得合成流量混进真实多播组）：

```sh
./target/release/corr-replay --config config/correlator-test.yaml \
    --rate 20000 --duration 30 --pattern prng &
./target/release/corr --config config/correlator-test.yaml --duration 20 \
    --output-dir /tmp/corr-test
# 故障注入：--drop-rate / --gap-len+--gap-at / --stall INDEX:SECONDS /
#           --offset INDEX:N / --sample-delay INDEX:N / --pattern sine|prng|counter|zero
```

相位验证（回放器给 T02 加 N 个采样延迟，再从落盘文件拟合相位斜率）：

```sh
./target/release/corr-replay --config config/correlator-test.yaml --pattern prng \
    --sample-delay 1:2 --duration 40 &
./target/release/corr --config config/correlator-test.yaml --duration 30
python3 ../tools/phase_check.py <输出目录>     # 拟合出的 τ 应等于 2.0000
```

### 6.7 输出布局与续写

```text
<output_dir>/<run_name>/
├── baseline_0000_j00_k00.bin   complex64[n_freq] × ntime，行优先，
│   ...                         baseline 编号由 C 库的 corr_baseline_pairs 定义
├── baseline_0299_j23_k23.bin   （24 路 → 300 条，j <= k）
├── rows_ms.bin                 int64 LE × ntime：每行第一帧变为 final 的时刻（Unix ms）
├── rows_ms.txt                 同样内容，纯文本 "row ms" 两列，方便直接看
├── blocks.csv                  row,station,first_frame_id,n_cum,valid,zero_filled
│                               每 station 每行一行，无表头（'#' 开头的行会被当作注释保留）
└── metadata.yaml               几何参数 + config_hash（只在新建时写一次）
```

**时间戳的语义**：不是「写盘时刻」而是**该行第一帧变成 final 的时刻**，
所以 writer 积压不会把时间戳整体推后。精度到毫秒，同一次 run 内单调。

**续写规则**（`Writer::open` 每次启动都执行）：

1. 读 `metadata.yaml` 的 `config_hash`，与本次配置不符 → **拒绝启动**
   （几何变了就不该往同一批文件里追加）；
2. 每个文件按自己的行长向下取整，得出各自「已完成行数」，取**最小值**为准
   （四个文件任何一个写了半行，都以最保守的那个为准）；
3. 对多出来的尾部做 `set_len` 截断（数据文件、`rows_ms.bin`、`blocks.csv`），
   `rows_ms.txt` 由 `rows_ms.bin` 重建，而不是继续追加；
4. 起始 frame 取 `max(本次嗅探对齐值, 上盘最后一帧向上取整到行网格)`
   ——保证不重复覆盖已经写过的帧区间。
   注意两者之间可能留下**空洞**（前端 counter 已走远），`blocks.csv` 的
   `first_frame_id` 能看出来，这是设计允许的。

**退出**：Ctrl-C/SIGTERM → 停收包 → 丢弃未写满的行 → 排空已算完的行 →
fsync → 完成 `blocks.csv`。停止后每个文件的行数都严格相等，可直接续写。

### 6.8 本轮修复之二：运行中跳变的补零现在走 ring 守卫

§6.4 修的是**启动**时的跳变；这一节修的是**运行中**的跳变——丢包让 counter 跳号，
`StationAssembler::finalize_range` 要把中间那些帧补零写进 staging。
原来的写法是**一趟扫完**，绕过了 `SlotRing`，有两个真问题：

1. **写的时候那些字节还不属于自己。** 一个物理槽只有在管线把它上一圈的数据
   拷走之后才能被覆盖（`corr_row_slot_free` = `cudaEventSynchronize` 那个槽的
   H2D 拷贝）。3 个槽、每槽拷贝约 2 ms、而一个槽每 2.6 ms 就被消费一个，
   也就是说管线大部分时间正在拷某个槽。一趟扫完有相当大概率正好写在
   正在被拷的那个物理槽上；而且回收一个槽会**重置它的到达位图**，
   于是被清零的不是"一帧"而是**整槽 256 帧**——而那一行在 `blocks.csv` 里
   报的 `valid` 还是满值。这就是"看起来完全正常的一行里混进 256 帧零"。
2. **等一个槽需要范围先推进。** 管线能释放某个槽的上一圈，取决于这条 station
   **已经发布**的 progress 够不够它提交到那里；而这条 station 不写完补零就
   不能发布 progress。一趟扫完的话，范围末尾要等的释放只有范围末尾自己能解锁
   ——死锁。

**修复**：`finalize_range` 改成**逐个 staging 槽推进**，每一步：

```text
发布 progress  →  ring.wait_free(该槽序号)  →  写这一步的零  →  progress 前移到步末
```

- 每一步都先**发布 progress**（`StationProgress`），这正是给管线"许可"的地方；
  两个线程因此能轮流推进而不是互锁。
- 每一步都**过 ring 守卫**，落地前一定确认那个物理槽的上一圈已经被拷走。

**代价是诚实的、有界的**：长跳变期间这条 station 会停止读 socket，时长约等于
管线消费掉缺失帧所需的时间。只要这个时长小于 `SO_RCVBUF`（本机实测 0.5 s/路）
就**一点数据都不丢**（socket 里存着live 数据）；超过之后它开始丢 live 数据，
这正是新加的**补零上限**要拦住的东西——见下面第 2 条。

**同时新增：单站补零上限**（`AssembleError::ZeroFillCapExceeded`，fatal）。
调度器原本只对"它**要求**补零"的帧数设上限；现在这条 station 因为自己跳号而
补的零也算进同一个配置项 `integration.zero_fill_limit_frames`。上限按槽检查，
所以报出来的数字最多超出一个槽，并且超限就干净停止（exit 1、行仍是整数行）。

**验证**：

- 单测：跳变时**必须**等到 ring 释放才写（变异掉 `wait_free` → 单测失败）；
  超限必须 fatal 停止（变异掉检查 → 单测失败）；闸门/上限两处守卫都承重。
- 端到端注入 2000 个连续丢包（≈ 4 个 staging 槽的跳变）：
  `gaps=1 lost=2000 zero_filled=1000`，**行数准点（145 行/15 s）**、
  有 `slot-wait spins`（守卫确实在等）、`exit=0`；
  受影响的那一行在 `blocks.csv` 里如实记为 `valid=24 zero_filled=1000` ✓。
- 真实 24 路 20 s produce：`gaps=0 lost=0 zero_filled=0`、19 行、
  300 文件一致、exit 0（正常链路下这条路径完全不触发，性能无回退）。

**未做**：把「等待」从自旋改成阻塞（现在靠 `yield` + 自旋，24 路时把 CPU
吃到 21.5/32 核）；以及更彻底的方案——给 GPU 传**有效性位图**让 kernel 直接把
缺帧当零，这样完全不需要主机侧补零。后者要动 ABI 和 kernel，见 §8。

### 6.9 不同积分时间的写盘压测（本轮实测，24 路真实前端）

`tools/stress_integration.sh N_CUM ROWS_PER_FETCH FSYNC_EVERY_ROWS SECONDS [NAME]`。
每行的大小固定（300 baseline × 4096 bin × 8 B = 9.38 MiB），积分时间决定的是
**行多久来一次**，也就是把 9.8 MB/s 的细流变成 937 MB/s 的洪水——同时把那些
在 1 行/秒时完全看不见的每行开销（D2H 取回、`rows_wait`、300 个文件的写与
fsync）放大 100 倍。

| # | n_cum（积分） | rows_per_fetch | fsync_every | 输出目标 | 需要速率 | 实测 | 结果 |
|---|---|---|---|---|---|---|---|
| A | 97792（1.001 s） | 1 | 1 | /scratch | 1.00 行/s、9.8 MB/s | **24 行 / 25.0 s**，9.0 MB/s | ✅ 准点，0 丢包 |
| B | 9728（99.6 ms） | 1 | 1 | /scratch | 10.04 行/s、98.7 MB/s | 5.77 行/s、56.8 MB/s | ❌ 14.2 s 后被上限干净停止 |
| C | 9728 | 1 | 10 | /scratch | 10.04 | 8.64 行/s、84.9 MB/s | ❌ 18.8 s 后停止 |
| D | 9728 | 10 | 10 | /scratch | 10.04 | 8.70 行/s、**run 内 9.55 行/s** | ❌ 但已很接近 |
| E | 9728 | 10 | 10 | **/dev/shm** | 10.04 | **10.00 行/s、98 MB/s** | ✅ **准点、0 丢包、GPU 80–95%** |
| F | 1024（10.5 ms） | 100 | 100 | /dev/shm | 95.4 行/s、937 MB/s | 84.2 行/s、805 MB/s | ❌ 13.7 s 后停止 |
| G | 1024 | 10 | 100 | /dev/shm | 95.4 | 82.7 行/s、835 MB/s | ❌ |
| H | 1024 | 400 | 100 | /dev/shm | 95.4 | 86.2 行/s、828 MB/s | ❌ |

**结论（逐条都是上表量出来的）**：

1. **1 秒积分稳。** 24 路 9.8 MB/s，任何存储都能扛。
2. **100 ms 的瓶颈在写盘，不在 GPU。** 同一份配置把输出目录换到 `/dev/shm`
   （E）就是**满速准点、零丢包**，GPU 只跑到 80–95% 还有余量；
   写 /scratch（D）就差 5%（94 MB/s vs 98.7 MB/s）。
   注意 D 的 `fsync_every_rows=10` 已经把 fsync 降到 30 次/s，所以**主要不是
   fsync 的问题**（B→C 显示 fsync 确实值 50%：56.8 → 84.9 MB/s，值得调，
   但不是全部）。
3. **10 ms 的瓶颈在管线自己。** 即使把存储变免费（F/G/H），也只到
   83–86 行/s（需要 95.4），而且**`rows_per_fetch` 从 10 调到 400 几乎没用**
   （82.7 → 86.2 行/s）——说明取数据的频率不是矛盾点。剩下的每行开销里，
   最大的一项是 `Writer::write_row`：**它在管线线程里同步执行**
   （`storage.writers` / `queue_bytes` 两个配置项目前是**声明了但没实现**）。
   单独量这个写法：300 个文件轮转写 32 KB，纯缓冲写 tmpfs 是
   **6136 MB/s（624 行/s，1.6 ms/行）**，写到 /scratch 也有
   **789 MB/s（80 行/s）；加 fsync（每 10 行）降到 498 MB/s**。
   也就是说写法本身够快，但它**占的是管线线程**：10.5 ms 的行周期里
   1.6 ms（15%）花在写盘上，管线就来不及驱动 GPU + 取数。
4. **失败方式是"干净停止"，不是写坏数据。** B/C/D/F/G/H 六次失败全部由
   本轮新加的**单站补零上限**触发：丢包让 counter 跳号 → 补零 → 补零期间
   这条 station 不读 socket → socket 溢出 → 再丢更多——这个正反馈最终撞到
   `zero_fill_limit_frames`，运行停下来并报出是哪一路。
   停止时六个文件都是**整数行**（300 个 baseline 大小一致、
   `rows_ms.bin`/`blocks.csv` 计数一致），可以直接续写。
   （反过来说：任何**持续的**管线缺口都是致命的。收包侧能吸收的滞后是有界的
   ——ring 3 槽 7.9 ms + socket 0.5 s——所以哪怕只差 5%，十几秒后也一定会
   演变成丢包级联。这是"短积分"真正的难点，不是带宽不够。）
5. **环境事实**：/scratch 的 NFS 写本身没问题（dd 1 GB 缓冲写 837 MB/s），
   但**全局 `sync` 会卡很久**（2 GB 写完后 sync 超过 7 分钟没回来，因为要等
   所有用户的脏页）——压测脚本里不要用 `sync` 计时，用行数与 `blocks.csv`
   对齐来判读。

**下一步（按收益排序）**：把 writer 从管线线程里拿出去（实现 `storage.writers`
与队列），然后用同一张表复测；100 ms 档很可能因此达标，10 ms 档再看是否只剩
937 MB/s 的存储问题。

### 6.10 实验 A：多卡引擎（本轮实现并实测）

**做了什么**：C ABI 升到 v3，一个 handle 背后是多张卡。`corr_config` 新增
`n_devices` + `devices[CORR_MAX_DEVICES]`（**卡数来自配置，代码里没有"4"**；
`n_devices = 0` 退回 `device_id`，即 v2 语义）。引擎内部：一块
`cudaHostAllocPortable` 的共享 staging（收包线程写一次，所有卡读）、每卡一套
stream/事件/cuFFT plan、baseline 轴按卡连续切分、`corr_row_slot_free` 变成
"所有持有副本的卡都读完"（AND）、`corr_rows_fetch` 把各卡 slice 拼回一条连续的
全局行。新增 `corr_n_devices` / `corr_device_plan_get` / `corr_device_stations` /
`corr_device_stats_get`，让调用方能**看见**切分而不是猜。

**Rust 侧**：收包、拼帧、水位线、写盘、blocks.csv、metadata **一行未改**——
这正是"把多卡藏在 ABI 后面"的验证。配置里 `gpu.devices` 决定引擎跨几张卡；
`stations[].gpu_id` 降级为**建议值**（允许指向本次不用的卡），所以
**回退到单卡就是改一行 `gpu.devices: [0]`**，station 表不用动。

**正确性（仿真数据，8 路回放，T02 延迟 2 个采样）**：

| 检查 | 结果 |
|---|---|
| 4 卡 vs 1 卡，相同输入 | **逐字节完全相同**（115 个共同行，`tools/compare_runs.py`） |
| 相位拟合（τ=2 采样） | 1 卡 τ=2.0000、4 卡 τ=2.0000，最大残差 0.0002 rad |
| C 侧 `ctest`（含新增 `multi_device`） | **6/6 通过**（golden 期望值是改动前生成的 ⇒ 单卡路径没变） |
| Rust 单测 + 集成 | **79 通过** |
| 真实 24 路单卡 12 s（回退路径） | 11 行/12.00 s 准点、**0 丢包**、18.1 GB/s |

**效率（真实前端，24 路，本轮最关键的测量）**：

| 配置 | 每卡需求 | 每卡实测 | 聚合实测 | 行数/12 s（应 11） |
|---|---|---|---|---|
| 8 路 × 4 卡 | 6.4 GB/s | 6.31 (99%) | 25.2 GB/s | 11 ✓ |
| 16 路 × 4 卡 | 12.8 | 12.43 (97%) | 49.7 | 11 ✓ |
| 24 路 × 4 卡 | 19.2 | **15.10 (79%)** | **60.4** | 9 ✗ 落后 |
| 24 路 × 2 卡 | 19.2 | 17.58 (92%) | 35.2 | — |
| 24 路 × 1 卡（今天） | 19.2 | 18.1 | 18.1 | 11 ✓ |

**纯拷贝对照**（新增 `corr-cuda/tests/multi_h2d_bw.c`，同样的 2D 分块模式、
同一块 portable pinned、多卡并发，不含任何管线）：

```text
1 卡 26.97 GB/s | 2 卡 52.94 GB/s | 4 卡 81.34 GB/s（每卡 20.3）
```

**结论（三条，都是量出来的）**：

1. **链路不是墙**：4 卡聚合 **81 GB/s**，比 24 路全复制需要的 76.8 还高一点，
   也高于 40 路 covering（R≈2.1）需要的 **67 GB/s**。
2. **管线是墙**：同样的 4 卡，真正跑起来只有 **60.4 GB/s**，是纯拷贝的 74%。
   饱和点在**每卡 ~15 GB/s、聚合 ~60 GB/s**（8/16 路都能跟得上，24 路跟不上）。
   所以实验 A 在 24 路全复制（需求 76.8）下**必然会落后**——这也解释了它为什么
   会走到补零上限而干净停止（这是设计好的行为，不是崩溃）。
3. **NUMA 落点值 +16%**：staging 绑在单个 NUMA 节点时 4 卡聚合 51.1 GB/s，
   `numactl --interleave=all` 后 **59.4 GB/s**。跨 socket（xGMI）反而无所谓
   （2 卡本节点 17.58 vs 跨节点 17.33，一样）。

**对 B / 40 路的定量含义**：40 路 covering 每卡 21 站 = 16.8 GB/s 需求，
而当前管线每卡天花板约 15 GB/s ⇒ **还差约 10~15%**。要跨过去，三条路（都不是
重新设计，都是已经量过的杠杆）：

- **提高管线拷贝效率**（74% → 85%+）：减少收包侧的主机内存占用与竞争
  （NIC DMA 32 + 收包读 32 + 写 staging 32 = 96 GB/s，与 60 GB/s 的 GPU 读抢
  内存带宽）、NUMA 交错、必要时调 `kDeviceBuffers`；
- **staging 打包 4 bit**（字节数减半 ⇒ 40 路只需 33.5 GB/s，非常宽裕）；
- **频谱时间抽取交换**（35 GB/s 一档）。

### 6.11 实验 B：pair-covering 放置（本轮实现并实测）

**做了什么**：ABI 升到 v4，新增 `corr_config.split_mode`：
`CORR_SPLIT_BASELINE_RANGE`（0，A 的行为：每卡持有全部 station、按 baseline 连续切）
与 `CORR_SPLIT_PAIR_COVERING`（1，B：每卡只持有部分 station，算自己 resident 集合
内部的那些基线）。配置里是 `gpu.split: baseline_range | pair_covering`。
新增 `corr_device_baselines` 让调用方能看到每卡到底拥有哪些基线。

**引擎侧**：`GpuPlacement` 从"baseline 区间"改成**显式列表**（covering 下每卡的
基线在全局编号里是散开的），并且：resident 集合不是恒等映射时走**逐站拷贝**
（每站一段连续内存，1 次 1D 拷贝；恒等时仍是一条 2D 描述符，保持 A 的路径），
行拼装对散开的设备走"先取到 scratch、再按全局基线号逐个放置"（1 秒积分下
26.9 MB/行 ≈ 可忽略）。

**规划器（引擎内，确定性）**：先按 `Σ C(s_d,2) ≥ C(n,2)` 求最小 resident 集
大小 s；然后**轮转填充**（每轮给每张卡各加一个站，因此各卡大小天然一致——
每卡的主机链路开销正比于自己的 resident 数，不均衡等于把余量花在最大的那张卡上），
再用"把缺口站的另一个站塞给当前最小的卡"修补，最后**验证每条基线恰好被拥有一次**。
不满足就加大 s 重试。

**规划器实测**（`multi_device` 测试会打印并断言）：

```text
 8 stations / 4 devices: resident 5..6  per device, R = 2.62, baselines 8..10   per device, all 36  covered
24 stations / 4 devices: resident 15..16 per device, R = 2.58, baselines 73..77 per device, all 300 covered
40 stations / 4 devices: resident 25..26 per device, R = 2.58, baselines 204..207 per device, all 820 covered
```

- **正确性**：8 路仿真下，covering 放置与单卡结果 **逐字节完全相同**
  （早些版本的顺序贪心会因为 resident 集不均衡导致 cuFFT batch 大小不同而出现
  ~2e-6 的相对差；轮转填充后这个差消失，测试仍保留 1e-5 的相对容差作为护栏）。
  `ctest` **6/6**、Rust **79 测试**全绿；单卡 `split_mode=0` 路径逐位不变。
- **效率（真实 24 路，4 卡，covering）**：每卡 H2D 从 A 的 **19.2 GB/s 需求**
  降到 **11.2~12.0 GB/s 实测**（聚合 46.3 GB/s，需求 49.6），
  行数 18~19/20 s（慢约 1 行），说明 **covering 的确把每卡负载压到了管线
  能承受的范围内**——这是 B 的全部意义。
- **但仍有丢包**（几十万包/站，run 间波动大），而同一时刻的**单卡对照 12 s
  是 0 丢包、11 行准点** ⇒ 丢包与本次机器负载有关（当时 load average **20.9**，
  共享机器），也与"同一 staging 被多卡读、slot 释放要等最慢的读者"带来的抖动有关。
  这条要在做效率优化时一起解决。

**已知差距（诚实记录）**：

1. **R = 2.58，而池化下界是 2.10**（40 路 4 卡：池化下界 Σs=84、s=21，
   当前规划器给到 25~26）。差的这 ~20% 直接是每卡 H2D：21 站 = 16.8 GB/s
   对 26 站 = 20.8 GB/s，而实测管线每卡天花板约 15 GB/s。桥接这种"紧凑覆盖"
   需要真正的 covering design 搜索（轮转+修补做不到接近下界）；
   但即使拿到 R=2.1，40 路仍是 67 GB/s > 管线 60 GB/s，**所以下一步仍然是先提管线效率**。
2. covering 下每卡基线是散开的 ⇒ 行拼装多一次 host 拷贝（1 s 积分下可忽略，
   10 ms 积分下 2.7 GB/s，要用设备侧打包 kernel 解决）。

### 6.12 效率优化：先分清"机器"和"设计"

本轮的优化从一个能分清责任边界的实验开始，结论改变了接下来该做什么。

**1) 机器不是墙，管线也不是——是"谁在等谁"** 。新增两件永久诊断：
`corr_device_stats` 增加 `ns_accumulate / ns_slot_free / ns_row_end / ns_rows_wait /
ns_fetch`（**管道线程在每次库调用里花了多少时间**），报告里加一行汇总。用它一看就清楚：

```text
pipeline thread: 10.0 s inside library calls (71% of the run): queue 0.6 s,
                 release-wait 10.0 s, rows-wait 0.0 s, fetch 0.0 s
```

即管道线程 71% 的时间卡在 `corr_row_slot_free`（等各卡的 H2D 拷完），而**排队只花 0.6 s**
——瓶颈不是"发得太慢"，而是"等得太久"。

**2) 纯拷贝对照**（4 卡，24 路同款 2D 分块模式）：单独跑 **84.8 GB/s**，
与真实收包路径（dry-run 24 路）**同时跑 96.7 GB/s** ⇒ **主机内存/NIC 不构成争用**，
机器侧的链路能力远高于管线实际拿到的 ~45 GB/s。

**3) 真正的优化：把"等 ring"从自旋改成阻塞**（`SlotRing` 现在是 `Mutex+Condvar`，
释放/abort 时唤醒，1 ms 超时兜底）。效果：

| | 改前 | 改后 |
|---|---|---|
| 24 路进程 CPU（load≈17 时实测） | **21.5 核** | **6.1 核** |
| 行数 / 12 s（同窗口对照） | 10 | 10 |
| 聚合 H2D | 44.3 GB/s | 43.7 GB/s |

CPU 降 3.5 倍、吞吐不变。在**共享机器**上这一步同时减少了自己对全机的压力
（自旋的线程抢的正是收包路径自己的核）。报告里那行也从 "slot-wait spins" 改名为
"staging-ring waits"，因为它现在是阻塞等待，不再消耗 CPU。

**4) 配置旋钮在这台机器上量不出差别**（fps 256/512、slots 3/5/6 在 load≥16 时
全部收敛到 43~47 GB/s、10~11 行）。两个副产品：
- `frames_per_slot=1024` 与出厂 `n_cum=97792` **不整除**（97792/1024=95.5），配置校验直接拒绝
  ⇒ 512 是最大合法值；
- 在较安静的窗口里 fps512 把 release-wait 从 4.2 s 降到 0.7 s（28%→5%），
  行数准点、丢包从 232k 降到 54k。**安静窗口下这个旋钮是有效的**，值得在真机上试。

**5) 最重要的一条：丢包是机器的锅，不是多卡的锅。** 同一时刻的三方对照：

| 配置（同一窗口，load≈15） | 行数/12 s（应 11） | 聚合 H2D | W17 丢包 |
|---|---|---|---|
| **单卡**（今天一直在用的配置） | 10 | 16.2 GB/s | **845,907** |
| 4 卡 range | 6 | 44.3 GB/s | — |
| 4 卡 covering | 9~10 | 43.7 GB/s | 904,780 |

**单卡在同样的窗口里也丢 84.6 万包**（load≈6~9 的窗口里同一配置是 0 丢包、行数准点）
⇒ 当时的结论是"丢包来自其他用户把机器压到 load 15~25"。**这个结论是错的，见 §6.13.3**：
真正的原因是同在这一轮引入的 `max_in_flight = block_slots-1` 策略让接收侧没有任何余量，
负载只是放大器。改掉之后，**在 load≈14 的同一窗口里丢包 0.065%**。
"单卡也丢"这个对照本身仍然有价值（它排除的是"多卡引入的丢包"），但它不能证明
"丢包是机器的锅"——两者当时同时成立，而只有后者是可修的。
反过来看，covering 在**同样的行节奏**下把聚合流量从 16.2 提到 43.7 GB/s（每卡 10.9），
这正是 B 想要的形状。

**下一步（按证据排序）**：
1. **在安静的窗口重测**（load < 8）：fps 512 + covering 的 24 路，看是否能到 0 丢包、
   行数准点；这是验证"多卡已可用"的唯一干净条件。
2. **减少字节**：4 bit staging 打包（H2D 减半 ⇒ 40 路只需 33.5 GB/s，余量立刻出来）。
   这是目前唯一不依赖机器空闲就能推进的结构性优化，代价是精度论证（样本实测集中在
   −4..+4，但打包必须证明无损或给出可接受的误差界）。
3. 收包侧内存触碰次数（NIC DMA + 收包读 + 写 staging 共 96 GB/s）没有便宜的办法绕开：
   `recvmmsg` 的目标地址依赖包内 counter，无法直接收到 staging。

### 6.13 本轮：两个真 bug 与一个"零余量"策略错误

这一轮从"为什么 24 路 produce 只剩 2 个槽、0 行"开始，最后变成三个独立发现。
三个都靠**先分清"谁在等谁"、再动手**定位，因此顺带留下了四件永久诊断。

#### 6.13.1 互锁：发布的水位比已知的进度晚一个批次

现象：`slots=2 in_flight=2`、水位停在 `row_start+752`（差 16 帧到槽边界），
24 个站全部停在同一位置，`low_silence=0ms`（包还在到达），0 行、不报错。

关键是把"每个站各自停在哪"打出来（`CORR_PIPE_DEBUG=1` 每 1.5 s 打一行偏移）：

```text
offsets from row start (frames): 0:753 1:758 … 12:767 14:768 17:752 … 23:755
```

散在 752~768 之间、也就是**每个站都停在自己某一批的中间**。槽边界是 768，
768−752 = 16 帧 = 32 包 = 恰好一个 `recvmmsg` 批次 ⇒ 结论：线程在批次**内部**
阻塞在 ring 上（等下一槽释放），而那批里已经处理完的帧（含把水位推到 768 的那一帧）
**还没发布**——发布在批次末尾。调度器要"水位 ≥ 槽末端"才提交下一槽，
而能解锁这个槽释放的正是这条没发布出去的水位：互相等死。

修法：**槽一完成就发布**，不等批次末尾（`on_packet` 里 `progress % frames_per_slot == 0`
时 `p.set(progress)`；补零循环里每一步也发布）。原理就一句：
**在可能阻塞之前，先把"能让对方放行"的数字发布出去。**

#### 6.13.2 零余量：`max_in_flight = block_slots-1` 让接收侧只剩 1 个可写槽

修完互锁后能跑，但**每行 41% 的帧是补零出来的**。归因链是这样走完的：

1. **源是好的**：新写的 `tools/mcast_rate.py` 自己入组读 3 s —— 24 个组的 counter
   都按 ~197 k/s 前进（=195 kpps/组，正是标称）。它只读不发，不会和生产端打架。
2. **丢在内核 socket 接收缓冲**：跑一次前后对比 `nstat`：
   `UdpRcvbufErrors += 39,812,514`，而同期该窗口各站 lost 合计 ≈ 42M ⇒ 一一对应。
   网卡侧 `rx_missed_errors` 只 +88,669，不是网卡丢。
3. **线程时间花在哪**：给报告加了"收包线程时间拆分"后一次看清——
   `5.1 s/站 等包、14.8 s/站 组包（其中 14.1 s 是在等 staging 槽）`。
   也就是**不是收不动，是根本没轮到它收**。
4. **不是自旋的开销**：`CORR_RING_SPINS` 从 20000 降到 32，等待时间 11.3→11.7 s，
   无差别 ⇒ 是真等待，不是自旋浪费。
5. **是领先量**：`block_slots=8` 下扫"接收侧领先槽数"（`CORR_RING_LEAD`）：

| in-flight | 可写槽 | 每站 16 s 收包 | 丢包 | 补零帧 |
|---|---|---|---|---|
| 7 | 1 | 1.68M | 1.26M ✗ | 624k |
| 4 | 4 | 2.97M | 72k | 36k |
| **1（默认）** | **7** | **3.12M** | **76** ✓ | **39** ✓ |
| 1 | 3（bs=4） | 2.42M | 609k ✗ | 301k |

内存代价 = `n_station × frames_per_slot × 2×n_freq × block_slots`（24 路 8 槽 = 402 MiB pinned）。
**决定成败的是"可写槽数"，不是并发拷贝数**：in-flight=1 时 H2D 反而略快
（18.8 vs 18.4 GB/s 聚合），因为单卡一条拷贝流已经吃满链路。

因此默认策略改为 **`max_in_flight = 1`（领先量 = block_slots−1）**，
`CORR_RING_LEAD` 保留为实验旋钮；四个配置文件的 `block_slots` 全部改为 8，
并新增 `config/correlator-24.yaml` 作为 24 路实跑的标准配置（tslice + 4 卡）。

#### 6.13.3 更正 §6.12 第 5 条，以及三种模式的最终对照

§6.12 说"丢包是机器的锅（单卡同样丢）"。现在看，那个对照排除的是"多卡引入的丢包"，
但**丢包的真正原因是本轮的零余量策略**（§6.13.2），负载只是放大器。
同一时间窗（load≈14）内的对照：

| 模式 | 每卡 H2D 实测 / 需求 | 每站收包 | 丢包 | 补零帧 | 判定 |
|---|---|---|---|---|---|
| **tslice（R=1）** | **4.68 / 4.84 = 97%** | 7.81M | **5,131（0.065%）** | **2,804（0.07%）** | ✅ 可用 |
| covering（R≈2.67） | 11.2 / 12.8 = 87% | 2.63M | 384k（13%） | 189k（13%） | ⚠️ 贴墙 |
| range（R=24） | 12.4 / 19.2 = 65% | — | — | — | ❌ 不可行（与 §2 的预测一致） |

这组数字把"R 是什么"落到了实处：**R=1 的 tslice 每卡只要 4.84 GB/s，
离这台机器单卡实测的 ~20 GB/s 还有 4 倍余量**，这正是 40 路（每卡 8 GB/s）可行的依据；
covering 的 12.8 GB/s/卡已经在墙边，range 的 19.2 GB/s/卡 直接不可能。

#### 6.13.4 新增的永久诊断（下次不用再猜）

- **报告里的"收包线程时间拆分"**：等包 / 组包（其中多少是等 staging 槽）/ 占比。
  一行就能区分"链路慢"和"自己慢"，这是本轮最有用的一件工具。
- **ring 阻塞计数**：`pipeline: N ring block episode(s)`，取代原来那行
  `staging-ring waits`（它是**自旋次数**，24 路下 3e9 起步，读不出任何结论）。
- **每秒时间线**（有丢包/补零时才打印）：`帧/丢包/补零` 每站每秒，
  一眼看出是"起手就稳定丢"还是"级联"——本轮正是靠它排除了"级联自持"这个假设
  （实测从第 1 秒起稳定丢 ~40%，不是启动冲击）。
- `tools/mcast_rate.py`：独立入组测"源到底在不在发、发多快"。

**下一步（按证据排序）**：
1. **24 路 tslice 长跑**（≥1 h）确认 0.065% 的残余丢包是否稳定，并看 `blocks.csv`
   的补零分布是否集中在少数几行。
2. **40 路投影**：tslice 每卡需求 = 40×0.8/4 = **8 GB/s**（x16 Gen4 实测 ~20 GB/s，
   余量 2.5×）。真正的墙变成另外两处：**网络**（40 路 = 259 Gbit/s，需 ≥400G 或双 200G bond）
   与 **CPU**（40 个收包线程 > 32 核）。
3. covering 若保留为模式：把 R 从 2.67 往 2.10 的覆盖设计下界推（§6.11）。
4. 静音窗口重测一次三种模式，把"负载放大器"的作用量化出来。

### 6.14 40 路：现在的判定、每项的实测依据、以及三个硬障碍

**判定：不是"可用"。40 路在软件结构上通、功能路径已在仿真源上端到端跑通，
但从未在 40 路真实速率下验证过，而且有三个必须先解决的硬障碍（源 / 端口 / 核预算）。**
24 路真实流量的全速率运行是**可用**的（§6.13.3，0 丢包 0 补零）。

#### 6.14.1 计算量的 N² 标定（`corr-cuda/tests/bench`，单卡、全基线、全帧）

`bench` 的语义就是"一张卡能否独自扛住整个前端"，行预算 1001.4 ms：

| N | 基线 B=N(N+1)/2 | 每站帧成本 | 单卡扛全部 | tslice 4 卡每卡占用 |
|---|---|---|---|---|
| 16 | 136 | 5.00 µs | 49% | 12% |
| 24 | 300 | 8.94 µs | **87%** | **22%** |
| 32 | 528 | 12.89 µs | 125% | 31% |
| 40 | 820 | 17.69 µs | **173%（差 1.73 卡）** | **43%（余量 2.3×）** |

拟合 `每站帧成本 ≈ 3.9 µs + 16.8 ns × B(N)`（24/40 两点定标，32 路验证误差 1%）：
**N² 项在 40 路时占 78%**。它是唯一按 N² 长的项；H2D/FFT/staging 内存/线程数都线性，
输出与分块求和虽是 N² 但基数极小（26.8 MB/s 与 107 MB/s）。

`bench` 同时暴露一条**硬上限**：

```text
corr_create: 1176 baselines on one device, but the correlation kernel maps one thread
per baseline and supports at most 1024 (n_station <= 44; split the baselines over more devices)
```

⇒ **相关核一线程一基线、单设备 ≤1024 基线 ⇒ 任何"一张卡算全部基线"的模式（tslice 就是）
单设备最多 44 路**；40 路已用掉 820/1024 = 80%。要上 44 路以上必须先改 kernel
（grid-stride 遍历基线，或按设备切基线块），加卡不能绕过这条。

#### 6.14.2 40 路功能路径已跑通（仿真源 `corr-replay`）

新增 `config/correlator-40-replay.yaml`（40 路，组用 239.1.200.x，与真实前端不重叠）：

```text
plan: 40 stations -> 820 baselines | staging 640.0 MiB | row stack 25.6 MiB | output 26.832 MB/s
  40 sockets、40 个收包线程、4 卡各 station=40/baselines=820、写 820 个 baseline 文件
  R01 … R39 全部 gaps=0 lost=0 zero_filled=0
```

**这是功能验证，不是速率验证**：回放工具自己只能到 ~37 kpps/站（请求 100 kpps、
实测 1.5 Mpps 合计），远低于真实 195 kpps/站；收包线程 92% 在等包。
原因是它按"每包 sleep 到下一个目标时刻"发送，天生跑不满（它自己的文档也这么说）。

顺带修的两处：
- `corr-replay` 原来按 `n_station > 16` 拒绝发送，本意是"别误灌真实前端的组"，
  但它挡住了 40 路回放这件事本身。改成按配置里的 `board` 字段判定（`replay` 才放行）。
- `bench` 释放的是"最新"而不是"最旧"的槽，提交 3 个槽后就报
  `slot 0 (sequence 3) is not free`——它其实一直是坏的，只是没被 ctest 覆盖到。

#### 6.14.3 三个硬障碍

1. **数据源只有 24 路**。按 `239.1.n.3 : 9900+n` 扫 130–180 全段（`tools/mcast_rate.py`
   的 `--include-disabled` 与一次性扫描都做过）：**只有 24 个组在发**
   （137/138/139/141-144/148-151/161-173），其余 27 个编号全无数据。
   40 路目标先要有 16 块新板在发。
2. **入口带宽 200G < 259 Gbit/s**。`ens2f0np0` = 200000 Mbps，40 路 ×6.46 Gbit/s = 259 Gbit/s。
   机上另有 `ens105np0`/`ens106np0` = 400G（NUMA 2 / 7，GPU 分支）为空闲，
   但实测**本地跨卡多播没有回送**（`ens105np0` 发、`ens106np0` 收，收不到），
   要挪入口就得动接线/交换，并重新评估 NUMA（入口现在在 node 0，GPU 在 node 2/7）。
3. **40 个收包线程 vs 32 核**。程序的前提是"一个收包线程独占一个核"，配置校验原本直接
   拒绝（`recv_cpus has 24 entries but 40 stations are enabled; one dedicated core per
   receive thread is the whole point of the pinning`）。本轮把这条改成**允许轮流共用核 +
   启动大字警告**（`config.rs` / `main.rs`），40 路才第一次跑得起来；
   40 路在这台机器上的可行布局 = 收包 0–27、pipeline 28、writer 29–30、monitor 31。
   **但共用核会吃掉抖动余量**，而 §6.13.2 刚刚证明余量不足会丢 40% 数据 ⇒
   40 路全速率的真正前提是**改收包模型**（一个线程用 epoll 管多个 socket，把线程数降到 ≤28），
   而不是靠共享核硬顶。

#### 6.14.4 负载敏感性（决定"40 路实时"能否写死）

同一份 24 路配置：load 12–14 → `lost=0`；load 25 → `lost=12`；load 32（本轮自测回放的余波）
→ `lost≈1.4×10^5`（5%）。**"实时"承诺必须绑定在一个负载上限上**，否则它会像 §6.12 那样
被误判成"机器噪声"。这也是为什么"一核一线程 + ring 领先量"这两条要守住。

#### 6.14.5 达标清单（40 路"可用"的定义）

| 项 | 状态 |
|---|---|
| 配置/规划 40 路（820 基线、分块、fetch 和）| ✅ 已验证（640 MiB staging、26.8 MB/s 输出）|
| 引擎 40 路几何（4 卡各 820 基线）| ✅ 实测（bench 43%/卡；kernel 上限 1024 尚够）|
| 40 路端到端功能（socket/线程/拼帧/水位/落盘）| ✅ 仿真源跑通，每站 0 丢包 |
| 40 路**全速率**接收（7.81 Mpps、32 GB/s）| ❌ 未验证，且当前**无法**验证（源只有 24 路）|
| 入口带宽 ≥259 Gbit/s | ❌ 需要 400G 或双 200G bond |
| 收包线程 ≤ 核数 | ❌ 需要 epoll 多 socket 改造 |
| GPU 每卡余量 ≥2× | ✅ 43%（4 卡 tslice），44 路以上需改 kernel |
| 长跑稳定性统计 | ❌ 缺（无小时级运行、无 stats.csv）|

### 6.15 nch=8192 / 积分 1 s：几何全通，但稳定性是"边缘的"

**结论先说：nch=8192 不需要动前端**——前端每包仍是 4096 样点、195,312.5 pps，
所以 nch=8192 只是"4 包拼一帧、帧率减半"，每站字节率、包率、网络负载**完全不变**。
新增 `config/correlator-24-nf8192.yaml`（24 路，n_freq=8192，n_cum=48896）。

#### 6.15.1 几何与派生量（实测 `--print-plan` / `--check`）

| 量 | nch=4096 | nch=8192 |
|---|---|---|
| 帧率 | 97,656.25 Hz | **48,828.125 Hz** |
| 包/帧 | 2 | **4** |
| 1 s 的合法积分长度 | 97,792 帧（=1.00139008 s）| **48,896 帧（=1.00139008 s）** |
| staging（24 路、fps=256、slots=8）| 402 MiB | **768 MiB** |
| 行栈（300 基线 × nch × 8 B）| 9.8 MiB | **18.8 MiB** |
| 输出 | 9.817 MB/s | **19.634 MB/s** |
| 相关总 MAC/s（∝ B×nch×帧率）| 不变 | **不变**（帧率减半 × 通道加倍）|
| FFT 工作量 | 5·N·log2(8192) | +7.7%（log2 13→14）|
| 每站包率 / 字节率 | 195,312.5 pps / 800 MB/s | **完全相同** |

`--check` 通过（`frame_stride=16384 slot_stride=4194304 station_stride=33554432 total=768 MiB`，
`pinned writes verified`）。引擎的通道分块是 `ceil_div(n_freq, kChannelsPerBlock)`，没有 4096 硬编码。

#### 6.15.2 满速率实跑（24 路真实前端）

```
26 s | 4750 slots | 24 rows | H2D 每卡 4.60 GB/s（聚合 18.4）| GPU 21–50% | 显存 3307 MiB
输出：304 文件，每个 baseline 1,572,864 B = 24 行 × 8192 ch × 8 B ✓
```

**但稳定性是边缘的**，三次满速率跑的结果：

| 跑次 | 丢包（每站）| 补零帧 | 说明 |
|---|---|---|---|
| #1 | **1,452,984（29%）** | 366k | 时间线：第 1–3 s 干净，**第 4 s 起持续丢且缓慢恶化**（10k→21k 补零帧/s）|
| #2 | 0 | 0 | 同配置、同窗口、load 5.8 |
| #3 | 0 | 0 | `UdpRcvbufErrors +0` |
| dry-run（无 GPU）| 0 | 0 | 接收路径本身满速率无问题（196 kpps/站）|

也就是说：**nch=8192 没有结构性错误，但它的"落后→补零→更落后"正反馈比 nch=4096 强一倍**
（补零一帧的 memset 从 8 KiB 变 16 KiB，而落后时补零与读包争抢同一个线程）。
nch=4096 在同样的机器上也会出现这种小亏空，只是幅度小：翻出 40 s 那次"干净"跑的
`blocks.csv` 发现它并非全零——38 行共 63,011 帧补零（每站每行约 69 帧，0.07%），
而另一次 20 s 跑则是严格的 0。**判据只能是 blocks.csv，不是报告里那个每站汇总。**

#### 6.15.3 长跑才是判据：15 分钟长跑暴露的是"余量"，不是 nch

15 分钟（900 s）满速率长跑（24 路、nch=8192、当时的默认 `block_slots=8`）：

```text
897 行 / 900 s（行节奏 100% 准点）| RSS 1.21 GB 稳定 | GPU 39–52% | H2D 聚合 19.2 GB/s
zero_filled = 110,464,467 / 1,052,633,088 帧 = 10.49%
三段持平 36.8M / 37.0M / 36.7M  → 稳定丢 10%，不发散、也没被吸收
收包线程：一半时间在组包，其中 324.7 s（36%）在等 staging 槽
```

**行节奏、内存、GPU、H2D 全都正常**——10% 的数据是被补零"掩盖"掉的。
同窗口、各 300 s 的对照把责任分清了：

| 跑次 | 配置 | 丢包（每站）| 补零帧 | 等槽时间 |
|---|---|---|---|---|
| A | **nch=8192，block_slots=16，ring_lead=12（4 拷贝在飞）** | **60 / 58.6M** ✓ | 17 | 18.8 s / 300 s |
| B | nch=4096，block_slots=8，ring_lead 默认（1 拷贝在飞 → 7 可写槽）| **36,595,219** ✗ | 18.3M 帧（25%）| 225.4 s / 300 s |

⇒ **丢包与 nch 无关**，是"接收侧可写槽只有 7 个（37 ms）"这个余量不足：
短跑（20 s）看不出来，因为 socket 缓冲（0.64 s）把亏空吃掉了；缓冲一满就变成稳定丢包 ✗。

**修复：`gpu.ring_lead` 成为配置项**（`block_slots - ring_lead` 个槽归 pipeline）。
24 路两个配置现在是 `block_slots: 16` + `ring_lead: 12`（staging 1.5 GiB）。
代价是 pinned 内存：领先时间 ≈ `ring_lead × frames_per_slot / 帧率`，
内存 ≈ 领先时间 × 总字节率 ⇒ 24 路 63 ms ≈ 1.5 GiB。

#### 6.15.3b socket 缓冲与 ring 领先量必须配对（实测 7.2× 改善）

两个缓冲是**同一个设计参数的两半**：汇编器只能领先 pipeline "可写槽数"这么远，
所以 socket 里能积压的量一旦超过领先量，积压就**只能涨不能消**——此后每次被抢占
都直接丢包。旧配置正是这个失配：socket 512 MiB/站 = **640 ms** 的数据，
而领先量只有 **63 ms**。

| 配置（24 路、nch=8192、load 25–27）| 补零率 | 备注 |
|---|---|---|
| socket 512 MiB ↔ 领先 63 ms（`block_slots=16/ring_lead=12`）| **1.9568%** | 1800 s，1797 行 |
| socket 64 MiB（80 ms）↔ 领先 105 ms（`block_slots=24/ring_lead=20`）| **0.3534%** | 1800 s，改善 5.5×；三段 1.7/2.6/3.1M **仍在涨** |
| 同上但 load 18（300 s）| 0.0001% | 说明负载是放大器 |

代价：pinned 从 1.6 GB 升到 2.4 GB（1 槽 = 24 站 × 256 帧 × 16384 B = 100 MB）。
仍未到 0.01% 的验收线 ⇒ 下一步要么把预算做大（领先 60 槽 = 315 ms，pinned 6.4 GB），
要么先**量出抢占的时长分布**再定预算——不要继续盲调。

#### 6.15.4 对"长时间跑"的含义

- 行节奏本身没问题（39 行/40 s、24 行/26 s、897 行/900 s、299 行/300 s 全部准点），
  说明水位线/调度没拖后腿，丢包是被补零"掩盖"掉的。
- **判据只能是 blocks.csv（`tools/rows_trend.py`），不是报告里的每站汇总量**：
  那次 40 s「干净」跑（每站 zero_filled=2,804）实际是 38 行里只有 6 行全零，
  三段递增 15,423 → 21,033 → 26,555。
- 真正的风险是**边缘稳定性**：接收侧的余量（ring 领先量 ~36 ms + socket 缓冲 0.64 s）
  与"补零成本"之间只隔一点点。所以长跑必须**用 blocks.csv 逐步核对**，
  并且把 load 一起记录（本轮实测：load 12–14 → 0，load 25 → 12 包，load 32 → 5%）。
- 若要给 40 路/长跑留余量，最直接的旋钮仍是 `gpu.block_slots`（领先时间 =
  `(block_slots-1) × frames_per_slot / 帧率`），代价是 pinned 内存线性增长；
  其次是降低补零的代价（见 §6.13.4 的"GPU 侧有效性掩码"设想）。

### 6.16 24 路补零的真正来源：网卡硬件丢包 + 绑核撞 NUMA 节点

**结论：24 路反复观察到的补零，主要不是我们的代码，而是主机收包路径的能力边界**
（网卡 RX 环 + NAPI 调度），而**绑核方式决定了我们撞不撞这条边界**。

#### 6.16.1 三段分别计数才看得见（方法论）

一路丢包必须分三层计数，否则每一层的丢包都会被误记到另一层：

| 层 | 计数器 | 24 路典型读数 |
|---|---|---|
| 网卡硬件 | `ethtool -S ens2f0np0` → **`rx_discards_phy`** | **11,802,893 / 90 s** |
| 内核 socket | `nstat -az` → `UdpRcvbufErrors` | 37,845（可忽略）|
| 我们的线程 | 报告里的 `worst read stall` / 等槽时间 | 49 ms / 1.3 s |

该跑各站丢包合计 **11,772,000**，与 `rx_discards_phy` **11,802,893** 一比一吻合 ⇒
**丢包 100% 发生在网卡把包交给内核之前** ✗。软件侧（我们的线程 255 ns/包、单线程上限
3.9 Mpps）压根不是瓶颈 —— 之前只看 socket 层计数器，因此被误导了很久。

#### 6.16.2 是否与我们有关：不在跑时不丢

| 状态 | 包率 | `rx_discards_phy` |
|---|---|---|
| 只有别的租户的 15 路（我们停）| 2.93 Mpps | **0** ✓ |
| 我们跑 20 路 | 3.91 Mpps | 0.8% ✗ |
| 我们跑 24 路 | 4.69 Mpps | 0.8–2.8% ✗ |

⇒ 主机收包能力边界落在 **3.9 与 4.69 Mpps 之间**（≈20→24 路）。

#### 6.16.3 能力曲线（同一窗口，各 90 s）

| 路数 | 需求 | load | 补零率 |
|---|---|---|---|
| 16 | 3.13 Mpps | 7.2 | **0.0000%** ✓ |
| 20 | 3.91 Mpps | 13.3 | **0.0000%** ✓ |
| 24 | 4.69 Mpps | 17.8、15.1、22.3、24.0 | 4.57% / 1.61% / 0.23% / 4.9% ✗ |

#### 6.16.4 根因：绑核撞上网卡的 NUMA 节点

```
NUMA：8 节点 × 4 核（NPS4）
  node 0: 核 0-3   ← 前端网卡 ens2f0np0 在这里
  node 2: 核 8-11  ← GPU0/GPU1
  node 7: 核 28-31 ← GPU2/GPU3
mlx5 中断实际落在：核 0,1,2（node 0）与 8,9,10,11（node 2）
原先收包线程钉在：核 0-23  ← 与 NAPI 同核 ✗
```

RX 环只有 **1024 深（可设 8192）**，在 4.4 Mpps 下只有 **0.23 ms** 的余量 ⇒
NAPI/软中断晚跑一点，环就溢出、包在硬件层被丢 ✗。而我们又把收包线程钉在
NAPI 所在的那几个核上，等于自己制造这种延迟。

**A/B 实测**（收包线程整块移出 node 0）：

| | 收包线程 | load | 补零率 | `rx_discards_phy` | ≥20 ms 停顿 |
|---|---|---|---|---|---|
| A | 核 0-23 | 15.1 | **1.6143%** | 850,402（0.199%）| 1869 |
| B | 核 4-27 | **22.3** | **0.2316%** | 449,737（0.106%）| 192 |

⇒ **B 在负载高 1.5 倍的条件下补零率低 7 倍**。三个 `correlator-24*/20*` 配置已按 B 布局
（收包核避开 node 0 与 pipeline/writer/monitor 核）。

#### 6.16.5 还差什么（系统级，需管理员）

1. `ethtool -G ens2f0np0 rx 8192`（环深 1024 → 8192：0.23 ms → 1.9 ms 余量）——
   单条收益最大。
2. `sysctl -w net.core.netdev_max_backlog=250000`（现 5000）。
3. 把 `mlx5_comp*` 中断钉到专属核，与收包线程不相交（需 root）；
   若可授予 `CAP_NET_ADMIN`，`SO_BUSY_POLL` 也能显著降低这一路延迟（现在返回 EPERM）。

**判据**：修完后 24 路 90 s 的 `rx_discards_phy` 应为 0、补零率 < 0.01%。

#### 6.16.6 绑核不再是手写核号：策略参数 + 体检程序

不同机器的 NUMA 形状、每节点核数、网卡/GPU 位置都不同，**手写 `recv_cpus` 必然在下
一台机器上静默出错**（症状只是长跑里几个百分点的补零）。所以加了两件：

1. **策略参数**（`cpu` 段，一行即可）：
   ```yaml
   cpu:
     avoid_nic_interfaces: [ens2f0np0]   # 该网卡所在 NUMA 节点的核从收包池里剔除
     auto_layout: true                   # 收包核取池首，pipeline/writers/monitor 取池尾
   ```
   解析结果在启动时打印（`cpu        : ...`），手写列表与规避集重叠时会打 WARNING。
   实测它推出的就是 §6.16.4 里 B 方案的布局（24 路 → 收包核 4-27、pipeline 28、
   writers 29-30、monitor 31）。
2. **`tools/doctor.py`**（只读，退出码 1 表示发现问题）：读 NUMA 拓扑、网卡 NUMA 节点、
   **IRQ 亲和掩码**（不是 `/proc/interrupts` 的历史计数——那是累计值，判断不了"现在能落在哪"）、
   RX 环大小、`netdev_max_backlog`、`SO_BUSY_POLL` 权限、其他进程占用的前端组、
   空闲丢包基线；并把"布局解析"交给二进制自己（`--print-plan`），避免工具与配置逻辑重复。
   它在本机报出 4 条：IRQ 掩码允许 0-31、RX 环 1024/8192、backlog 5000、busy_poll EPERM。

## 7. 建议实现路径

### Phase 0 ✅ 配置与元数据 schema

`config/correlator.yaml` + `corr-rs/src/config.rs`：24 路 station 表与
`rfsoc-init.json` 逐字段一致，`order` 连续，`--check` 拒绝任何不自洽组合
（未知字段、重复 order/name、重复 group:port、`n_freq` 非 2048 倍数、
`start_frame_id` 不在行网格上、CPU 重复分配、`chunk_frames > frames_per_slot` 等）。

### Phase 1 ✅ 接口契约与骨架

C ABI、CMake 工程、Rust 模块划分与构建接线、C/Rust 双侧自检。

### Phase 2 ✅ CUDA 侧离线自测（本轮完成）

- `kernels.cu`：int8→float32、chunk 批量 cuFFT R2C、SMEM 分块相关 kernel、
  scale kernel、拷贝/计算双流 + 设备缓冲环；
- 与 **numpy 独立参考对拍**：覆盖多槽行、非整除尾槽、多行栈、Nyquist 丢弃、
  无归一化、baseline 编号、autocorr 实部；
- `h2d_bw` 与 `bench` 定量回答了「单卡能不能跟上」，结论见 §2.9/§2.10。

### Phase 3 ✅ Rust 侧收包通路（已完成）

1. `corr-replay`：自建 UDP 多播回放器，可注入丢包 / 乱序 / 跳号 / 站间 counter
   偏移 / 站点停发 / 采样延迟（本机不能抓包，真实数据不可回放，这是唯一能确定性
   测试的路）。它**拒绝**加载实时配置，防止把合成流量打进真实多播组。
2. 收包线程（1 station = 1 线程 + 1 socket，绑核，`recvmmsg` + `SO_BUSY_POLL`）
   + 拼帧 + `SlotRing` + 水位线 + `--dry-run`，全部落地。
3. 回放器验证：干净流、丢包、跳号、站点停发、跨站偏移、乱序、采样延迟。
4. 4 路 → 24 路端到端到落盘（§6.5）。

### Phase 4 ✅ 扩路（已完成）

4 → 24 路都实测过；dry-run 24 路 4.69 Mpps / 155 Gbit/s 零丢包；
produce 24 路真实前端 40 s 零丢包、行数准点（§6.5）。
绑核策略与 IRQ 争核的问题见 §8。

### Phase 5 ⬅ **下一轮起点**

1. **在安静窗口复测多卡**（load < 8）：24 路 covering + `frames_per_slot: 512`，
   目标是 0 丢包、行数准点。§6.12 已证明当前窗口里连**单卡**都丢 84.6 万包，
   所以这是验证"多卡可用"的唯一干净条件。
2. **减字节：4 bit staging 打包**。H2D 减半 ⇒ 40 路 covering 只需 33.5 GB/s
   （当前管线在安静窗口能到 ~60 GB/s），余量立刻出来。代价是要给出精度论证
   （样本实测集中在 −4..+4，但打包必须证明无损或给出误差界）。这是唯一不依赖
   机器空闲就能推进的结构性优化。
3. **把 writer 搬出管线线程**（实现 `storage.writers` + `queue_bytes`）。
   §6.9：100 ms 档只差 5%（94 vs 98.7 MB/s），10 ms 档在免费存储下也只到 86 行/s
   （需要 95.4）——两者共同嫌疑都是"管线线程同步写盘"。
4. **小时级长跑**：丢包、水位线滞后、GPU 利用率、写盘吞吐、内存/句柄是否单调。
   现在的证据只到 140 s（produce）与 40 s（§6.5），多卡只到 20 s。
5. **接真实前端长时间跑**（在用户安排的时间窗内）。
6. 下游消费格式仍未定；`rows_per_fetch > 1` 时 `valid_frames` 是均分近似。


## 8. 待确认 / 风险

### 已关闭

1. ~~counter 全局同步假设需要长跑验证~~ → **已实测关闭**（§2.5）。
   并澄清了旧版"counter 差 5~20"只是采样时刻不同，不是真偏移。
2. ~~计算平台和数据通路尚未最终确定~~ → 内核 socket + CUDA；
   DPDK/AF_XDP 因无 root 出局（§3.2）。
3. ~~运行时配置文件格式尚未最终确定~~ → 已定稿并带完整校验（§6.1）。
4. ~~没有 tcpdump / AF_PACKET 权限~~ → 对策是自建回放器（§5.7）。
5. ~~输出速率能否持续（15 路 384 MB/s、24 路 960 MB/s）~~
   → 1 秒积分把它降到 3.93 / 9.83 MB/s，NFS 完全不成问题。
6. ~~单卡能否跟上 24 路~~ → **能，87.4%**（§2.10）；真实前端 24 路 40 s
   零丢包、行数准点（§6.5）。但见下面第 8 条：余量没有纸面上那么宽。
7. ~~交换机是否转发全部 24 个组~~ → **已实测关闭**：24 个组全部有数据
   （`ip maddr` 只列 15 个，是因为只有 15 个曾被加入过）。
8. ~~produce 模式 24 路启动即卡死~~ → **本轮修复**（§6.4）。

### 仍然开放

1. ~~数据正确性：`finalize_range` 的补零绕过 `SlotRing`~~ → **本轮修复**
   （§6.8）。修法：逐个 staging 槽推进、每步等 ring 释放、每步发布 progress，
   并对单站补零总量设上限。仍留给下一轮的是它**未做**的两件事：
   等待方式仍是自旋（24 路 CPU 21.5/32 核），以及"给 GPU 传有效性位图、
   在 kernel 里直接跳过缺帧"这个更彻底的方案（要动 ABI 与 kernel，
   能完全消掉主机侧补零与它的等待）。
   <details><summary>原描述（存档）</summary>


   `StationAssembler::finalize_range(progress, frame_id)` 在包号跳变时把中间帧
   补零写进 staging，但它**不检查那个物理槽是否还在被 GPU 读**。
   危险的是一次且仅有一次重叠：补零范围里存在某个槽序号 Q，使 `Q - block_slots`
   正好是管线**刚提交、还没释放**的那一个槽（即 `Q = Smax + block_slots`，
   `Smax` = 已提交的最大槽序号）——此时补零会覆盖管线正在 H2D 拷贝的同一物理槽
   上一圈的数据。更早的槽都已释放、更晚的（> Smax）从未提交，所以危险恰好是
   **一个物理槽 = 256 帧**：这 256 帧如果是完整的好数据，就会被悄悄写成零，
   而它们属于**已经提交过**的那一行，`blocks.csv` 里那一行的 `valid` 仍是
   满值——"看起来完全正常的一行里混进一帧零"，是最难发现的那类错误。
   不能简单地在补零前加 `wait_free(Q)`：范围开头几个槽等得起，
   但范围末尾等不起——管线要释放 `X` 必须先提交 `X + 1`，而那要求水位线 ≥
   `(X + 2)*fps`，水位线又卡在这个站当前的 `progress` 上，而补零正是为了推进
   `progress`：**补零等管线、管线等补零**，死锁。
   真正的修法是让**释放与提交解耦**（拷贝完成即可释放，或让管线在 Idle 分支
   主动回收已完成的槽），或者在跳变时不做原地补零、而是重新对齐并报告。
   （注意：本轮新增的**启动**跳变上限 `gap < block_slots*frames_per_slot`
   是安全的，因为那时管线一个槽都没提交过；这条风险说的是**运行中**的跳变。）
   </details>
2. **短积分时间：压测做了（§6.9），结论是结构问题不是参数问题。**
   `fsync_every_rows` 已经做成配置项（默认 1），实测它值 50%
   （100 ms 档 56.8 → 84.9 MB/s），但**主要缺口在别处**：
   `Writer::write_row` 在**管线线程里同步执行**，每行 1.6 ms（写法本身够快，
   tmpfs 6136 MB/s、/scratch 789 MB/s），在 10.5 ms 的行周期里就是 15%，
   管线来不及同时驱动 GPU 与取数。
   要做的：实现 `storage.writers` / `queue_bytes`（现在只是被校验的字段），
   把写盘搬到独立线程 + 有界队列，然后按 §6.9 的表复测。
   另注意：**任何持续的管线缺口都是致命的**——收包侧能吸收的滞后是有界的
   （ring 7.9 ms + socket 0.5 s），差 5% 也会在十几秒后变成丢包级联。
3. **24 路的真实余量比纸面窄。** 实测 GPU 利用率 55→100%、进程占用
   21.5/32 核（收包线程在 ring 上自旋等待，`slot-wait spins` 4.5e9/8 s）。
   本机是共享的（还有别的用户在跑满核）。若 GPU 或 CPU 再被挤，
   收包线程会让 socket 积压，500 ms 级积压就会开始丢包。
   缓解方向是把「等 ring」从自旋改成阻塞等待（代价是唤醒延迟）。
4. **小时级 counter 漂移未复测。** 2 分钟无漂移不排除小时级问题。
   建议用 `tools/counter_probe.c` 跑 1 小时。
5. **缺帧补零 + 统一除以行帧数会引入幅度偏置。**
   当前决策不按 baseline 有效帧数归一化，flag 载体是 `blocks.csv` 的
   `valid_frames` 列。**下游如何据此修正尚未定义。**
6. **输出目录不是"覆盖保护"而是"续写"。** 目录已存在且 `config_hash` 一致时，
   writer 会接着最后一个完整行往后写（这是设计要求）；
   但**没有**"这个 run_name 已经有一份完整观测、确定要追加吗"的确认。
   另外续写时若前端 counter 已经走远，中间那段帧会留下空洞（`blocks.csv`
   的 `first_frame_id` 能看出来），这是设计允许的，但下游需要知道。
7. **多卡方案在单机上不成立，且已确认不需要。**
   如果将来路数超过单卡天花板（26~28 路），加卡**解决不了**：把 station 分到
   两张卡后，跨卡 baseline（j 在卡 A、k 在卡 B）需要两边数据，单机上没有便宜
   的交换路径（频谱的体积是时域数据的 4 倍）。真正的出路是第二台接收机各自
   加入全部多播组、各算自己那份 baseline——这正是多播组存在的意义。
8. **`isolCPUs` 拼写错误导致 CPU isolation 未生效**，且无 root 不能修。
   影响：收包线程会被内核调度、可能被其他任务抢占。
9. **corr kernel 仍有优化空间。** 24 路时相关 kernel 约 1.1 TFLOP/s，
   仅为 L20 FP32 峰值的 ~2%；当前瓶颈是 H2D（19.2 GB/s，链路 26.8 GB/s）。
   在 24 路 GPU 已经跑满的实测下，这个"优化空间"值得重新评估优先级（§8「仍然开放」第 9 条）。
10. **监控 JSON / `stats.csv` 仍未实现**：`monitoring.stats_csv` 只是被校验的
    字段，没有产出。当前的可观测性靠 stdout 报告 + `blocks.csv`。


## 9. 关键数值速查

```text
UDP payload:              4104 bytes
counter:                  8 bytes, little-endian uint64, 三块板共享的全局包计数
data:                     4096 bytes, int8 two's complement
Ethernet frame:           4146 bytes = 14+20+8+4104

单路帧率:                 约 195,331 pps（精确值 195,312.5 = 800e6/4096）
单路 payload:             约 801.6 MB/s
本机当前 15 路:           约 2.935 Mpps, 97.35 Gbit/s
JSON 全部 24 路:          约 4.693 Mpps, 155.65 Gbit/s

sample_rate_hz:           800,000,000（整数；实测标定 799.97 MHz）
n_freq 默认:              4096（必须是 2048 的倍数）
fft_len:                  2 × n_freq = 8192
每帧包数 P:               n_freq / 2048 = 2
帧率 = bin_width:         97,656.25 Hz（两者恒等）
积分行默认:               97792 帧 = 1.00139008 s
                          （= 256 × 382，必须是 frames_per_slot 的整数倍；
                            1 秒永不整除，差 0.139%）
FFT:                      R2C，保留 N_freq 个频点，DC..N_freq-1，丢弃 Nyquist
精度:                     FFT complex64，corr complex64
归一化:                   每行除以行帧数，无 FFT 归一化（scale kernel 在 GPU 侧）
baseline:                 j <= k，N(N+1)/2，行优先（24 路 → 300 条）

GPU 管线默认:             frames_per_slot=256, chunk_frames=256,
                          block_slots=3, rows_per_fetch=1
staging（pinned）:        n_station × frames_per_slot × 2·n_freq × block_slots
                          4 路 24 MiB / 15 路 90 MiB / 24 路 151 MiB
行栈（显存）:             rows_per_fetch × n_baseline × n_freq × 8 B
                          4 路 0.3 MiB / 15 路 3.8 MiB / 24 路 9.4 MiB
                          （与积分长度无关：这是 rows_per_fetch 的用处）

单卡利用率（1 秒积分）:   4 路 18.8% / 15 路 45.8% / 24 路 87.4%
单卡天花板:               26~28 路附近
PCIe:                     Gen4 x16，线性 H2D 26.8 GB/s，跨步 2D 26.9 GB/s
                          （空闲时 nvidia-smi 报 Gen1 是降速，别误读）

输出（1 秒积分）:         4 路 0.33 MB/s / 15 路 3.93 MB/s / 24 路 9.83 MB/s
输出（10 ms 积分）:       15 路 384 MB/s / 24 路 960 MB/s（另一套量级，未压测）
输出目录:                 /scratch/<run_name>（4 并行流实测 1767 MB/s）

24 路实测（真实前端，1 s 积分，§6.5）:
  吞吐:                   195,174 pps/站（前端标称 195,312.5）
  落盘:                   9.82 MB/s、10 文件写/s、39 行/40 s，366 MB
  资源:                   RSS 280 MB、28 线程、372 fd、GPU 1809 MiB
  GPU 利用率:             55% → 100%（稳态跑满）
  进程 CPU:               21.5/32 核（收包线程等 ring 自旋，slot-wait spins 4.5e9/8 s）
  → 结论:                 24 路时瓶颈在 GPU 侧，且 CPU 余量只有约 1/3 台机器

启停:                     手动启动、Ctrl-C 停止；停止后所有文件都是整数行，可续写
续写:                     同一 run_name + 同一 config_hash → 接着最后一个完整行写；
                          不同 config_hash 直接拒绝启动

权限边界:                 无 root → 不能调 IRQ affinity / sysctl / SCHED_FIFO
                          大页不可用 → DPDK、AF_XDP 出局
                          可用：SO_RCVBUF 256 MiB（rmem_max 2 GiB，实测授予 512 MiB）、
                                mlockall（ulimit -l unlimited）、per-socket SO_BUSY_POLL
                                （但实测返回 EPERM，未生效 → 收包线程只能自旋）
```
