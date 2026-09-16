# 程序结构与维护指南

写给接手的人。目标不是复述代码，而是回答三个问题：**数据从哪来到哪去**、
**改一件事要去哪个文件**、**哪些不变量一旦破坏会以最难查的方式咬人**。

---

## 1. 数据路径（一句话一箭头）

```
RFSoC 前端 24 路组播 UDP（4104 B = 8 B 小端计数器 + 4096 int8）
  → net.rs        每个 station 一个 socket（recvmmsg 批量、SO_RCVBUF、SO_BUSY_POLL）
  → assemble.rs   拼帧（每帧 2·n_freq 样点 = P 个包）、写入 staging 环的某个物理槽
  → sched.rs      水位线 = 各站已发布进度的最小值；够了就提交一个 staging 槽
  → gpu.rs + corr-cuda/    H2D 拷贝 → R2C FFT → 每基线累加（Y_j·conj(Y_k)）
  → writer.rs     每行（1 s 积分）取回，写 n_baseline 个 complex64[ntime][nch] 文件
```

**四个"节奏"概念，别混**：

| 名字 | 含义 | 当前值（24 路 / nch=8192） |
|---|---|---|
| 帧 | 一次 FFT 的样点数 | 8192 样点（复数）/ 2·8192 = 16384 实样点 |
| 包 | 一个 UDP 数据报 | 4096 样点，所以每帧 **P = 2·n_freq/4096 = 4** 包 |
| staging 槽 | 环里的一块，每站 `frames_per_slot` 帧 | 256 帧 ≈ 5.24 ms |
| 行 | 一次积分、一次吐数 | `n_cum = 48896` 帧 = **1.00139008 s** |

帧率 = `sample_rate / (2·n_freq)` = 48828.125 Hz。**1 秒永远不是整数帧**（分母恒为 2 的幂），
`n_cum` 取 `frames_per_slot` 的最大整数倍，这是数学事实不是配置错误。

---

## 2. 模块地图（改什么去哪里）

| 文件 | 职责 | 什么时候改它 |
|---|---|---|
| `corr-rs/src/config.rs` | 配置解析、校验、派生 `RunPlan`、**绑核策略解析** | 加/改配置项；改绑核规则 |
| `corr-rs/src/net.rs` | socket 建/配、recvmmsg、绑核 syscall | 收包系统调用层、socket 选项 |
| `corr-rs/src/assemble.rs` | `SlotRing` + `StationAssembler` + 到达位图 + 有效性掩码；进度发布 | 拼帧、环的所有权、缺失帧的**声明**（零由设备补，ABI 7）|
| `corr-rs/src/sched.rs` | 水位线、提交/关闭行的状态机、停滞恢复、`StationProgress` | 调度策略、停滞判定 |
| `corr-rs/src/run.rs` | 启动顺序、收包线程、调度循环 + GPU 提交、取数写盘 | 主流程；**这里的函数最长，动它要最小心** |
| `corr-rs/src/report.rs` | `RunReport` 等类型 + 渲染（纯数据/格式化） | 加一条报告字段 |
| `corr-rs/src/writer.rs` | 输出文件、`blocks.csv`、resume | 落盘格式、写盘节奏 |
| `corr-rs/src/ffi.rs` / `gpu.rs` | C ABI 镜像与安全包装 | ABI 改动（见 §4） |
| `corr-cuda/src/kernels.cu` | FFT + 相关核 | 核内算法、性能 |
| `corr-cuda/src/corr_cuda.cpp` | staging、三种 split 规划、拷贝、计时 | 多卡放置、拷贝路径 |
| `corr-rs/src/bin/replay.rs` | 仿真前端（含 `--delay-stride`、丢包/停顿注入） | 造测试数据 |
| `tools/*.py` | 判读与体检（见 §5） | 判据、作图、体检项 |

---

## 3. 五条不变量（每一条都对应一次真实的、查了很久的故障）

1. **进度必须在"可能阻塞的等待"之前发布**（`assemble.rs`）。
   收包线程要等下一槽才能继续写，而调度器允许释放那一槽的依据正是它刚算出的进度；
   若发布晚了一个 `recvmmsg` 批次，两者互等死锁 —— 现象是"提交 2 槽、0 行、不报错"。

2. **socket 里能积压的量必须 ≤ 环允许领先的量**（`recv_buffer_bytes` ↔ `ring_lead`）。
   组帧器只能领先 pipeline "可写槽数"这么远，所以积压一旦超过领先量就**只能涨不能消**，
   此后每次被抢占都直接丢包。实测：640 ms 缓冲配 63 ms 领先 → 30 min 丢 1.96%；
   80 ms 配 105 ms → 0.35%（同负载）。
   **2026-09-16 复测（补零解耦之后）：这条的数量级变了。** 收包线程不再为补零长时间离岗，
   大缓冲因此不再是毒药：nch=4096 生产档（请求 256 MiB、实得 512 MiB ≈ 640 ms，配 39 ms
   领先）三次 300 s 实测丢包 0.0092% / 0.0178% / 0.0317%（load 均值 21.4 / 22.7 / 23.0）；
   而 nch=16384 档照抄别的档案的 32 MiB（84 ms）时丢 0.42%，改成 256 MiB 后 0.077%。
   所以这一项**要按档案实测，不能跨档案抄** —— 见 `config/README.md` 第 3 条。

3. **收包线程不要网卡 IRQ 所在的核，也不要网卡自己的 NUMA 节点**（`cpu.avoid_nic_interfaces`）。
   RX 环只有 1024 描述符（≈0.22 ms 余量），NAPI 一晚就硬件丢包（`rx_discards_phy`）。
   实测 A/B：撞核 1.61% vs 避开 0.23%，**后者负载还高 1.5 倍**。

4. **`block_slots - ring_lead` = pipeline 持有的槽数，至少 1**（`cpu`/`gpu` 配置校验强制）。
   把环"用到最满"（只留 1 个可写槽）在短跑里看不出问题、长跑必崩。

5. **FFT 长度是 `2·nch`，不是 `nch`**（所有相位相关代码与工具）。
   通道轴是 R2C 结果的**保留半谱**；把它当 FFT 长度会让延迟/斜率差**正好 2 倍** ——
   这个错误已经在三处出现过（含 `tools/phase_triangle.py` 两次）。

---

## 4. ABI 与多卡

- C ABI 版本号 `CORR_ABI_VERSION`、`struct_size` 运行期校验（C 侧）+ `ffi.rs` 里的
  **编译期 `offset_of!/size_of!` 断言**。改结构体两侧都要动，且两边都会拦你。
- 三种放置：`baseline_range`（每站数据复制到所有卡，R=N）、`pair_covering`（按站子集，
  R≈2.6）、`time_slice`（每卡拥有 1/n 个槽、算全部基线，**R=1**，取回时把 n 份部分行相加）。
  当前推荐 `time_slice`：每卡 H2D 只要 输入/卡数，实测 4 卡 24 路每卡 4.8 GB/s（需求 4.84）。
- 相关核**一线程一基线、单设备 ≤1024 基线** ⇒ 任何"一张卡算全部基线"的模式单设备最多 44 路。

---

## 5. 怎么判读一次运行（按顺序）

```bash
tools/doctor.py --config <cfg>                     # 机器侧体检：IRQ 撞核/RX 环/backlog/权限/租户
corr --config <cfg> --duration 300 --output-dir /scratch/xxx
tools/rows_trend.py  /scratch/xxx/<run>/            # 逐行补零：判据是"逐行"，不是报告里的汇总量
tools/longrun_report.py /scratch/xxx/<run>/         # 验收判决 + 三段趋势 + 负载区间
tools/phase_triangle.py /scratch/xxx/<run>/ --mosaic phase   # 相位正确性（回放数据用）
```

**三条方法论（都是踩出来的）**：

- 丢包要在**三层分别计数**：网卡硬件 `rx_discards_phy`、内核 `UdpRcvbufErrors`、
  我们线程的停顿/等槽。只看一层会把责任判错（我们曾把网卡丢包误判为"机器噪声"）。
- 长跑判据**只能是 `blocks.csv`**：报告里的每站汇总量会把"每行丢一点"藏起来
  （一次 40 s 跑汇总 0.07%，实际 38 行里只有 6 行全零）。
- 对比要在**同一窗口内交错**跑 A/B/A/B：这台共享机器的负载漂移是十分钟量级，
  同一配置两次跑实测差 3.2 倍。任何"跑 30 min A、再跑 30 min B"的结论都不可信。

---

## 6. 已知未完成（接手即可开工）

| 项 | 说明 |
|---|---|
| 40 路 | 三道硬墙：源只有 24 路；入口 200G < 259 Gbit/s；40 收包线程 > 32 核（需 epoll 多 socket） |
| 负载相关的残余丢包 | 网卡 RX 环 1024/8192、IRQ 掩码 0-31、`netdev_max_backlog=5000`、无 `CAP_NET_ADMIN` ⇒ 都要管理员 |
| ~~`run.rs::run_receive` 仍偏长~~ | 已完成：`receiver.rs` + `pipeline.rs` + `report.rs` 拆出，`run.rs` 1303 → 639 行 |
| ~~补零与读包争同一线程~~ | **已做完（ABI 6 → 7，`b76339e`）**：掩码随样本上设备、kernel 按掩码整帧清零、`corr_row_slot_free` 交还槽时清空掩码、组帧器停止补零也不再为补零等环。四条判据的实测见 `docs/DESIGN-zero-fill-decoupling.md` §8：判据 1、3 通过（逐字节 / 276-276），**判据 2 的后半（等环时间下降）与判据 4 的"更低"没有兑现** —— 那笔等待属于管线不属于补零 |
| **整槽缺失就不必上设备** | 下一步，收益最大：提交时主机已能读掩码，"24 个站一帧有效都没有"的槽无需 H2D/FFT/相关（贡献恒为零，行最后统一除以 `n_cum`）。饱和工况 ~310/693 个提交槽属于此类。设计 §8.6 |
| 流先停、`--duration` 后到会挂住 | 新旧二进制都一样（既有问题，不是补零解耦引入的）：让回放活得比关联机久就绕开了；要么把这条路径修掉 |
| C 侧测试没人跑 | `abi_smoke` 自 ABI 6（掩码区进 staging）起一直失败到 ABI 7 才被发现——它还在按"每站 = 纯样本"算 stride。已修并加了掩码生命周期检查；**改 ABI 就顺手跑 `ctest`** |
| writer 仍在 pipeline 线程里 | 短积分/高频吐数时需要独立 writer 线程 |
| ~~仓库没有版本控制~~ | 已完成：一直在用 git（重构链 `10de600 → ce15866`、补零解耦 `bd33583 → b76339e`）；`/scratch/corr-baseline` 是改动前的 worktree，A/B 对照腿用它 |
