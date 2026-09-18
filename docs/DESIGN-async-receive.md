# 异步收包后端：40 路 > 32 核那条墙

**本文记录异步收包后端的设计、实测对比与未解决的问题。** 代码在 `corr-rs/src/async_recv/`
（`corr-async` 二进制，feature `async-recv` **默认开启**；2026-09-18 从一个独立 crate 合并进来，理由见 §2.4），
复现命令在 §7，原始数据在 `bench/<tag>/`（`results.json` / `results.csv` / 每一腿的完整日志）。

`docs/ARCHITECTURE.md` §6 把"40 路"列为三道硬墙之一，其中一道写的是
**"40 收包线程 > 32 核（需 epoll 多 socket）"**。本文就是为拆这道墙写的：
把"一路一个线程"换成"一个 worker 一个核、每个 worker 用一套 epoll 收多路"，
并把这笔交易的价钱量出来。

---

## 1. 动机：线程数是继承来的，不是选出来的

原来的收包侧是**一线程一站**（`corr-rs/src/receiver.rs`）：每个 station 一个线程、绑一个核、
`recvmmsg` 忙轮询。流数 ≤ 核数时这是最好的形状——没有多路复用的开销，抖动余量最大，
项目里所有实测都是这么跑出来的。

流数 > 核数时它就有两个代价：

1. **每线程状态 × 超订**：每站的 `recvmmsg` 批次缓冲（`recvmmsg_batch × 4 KB`）、
   到达位图、栈，乘以一个大于核数的线程数；线程之间还要 CFS 轮转。
2. **抢占点永远踩在"有包在飞"的时刻**：24 个忙轮询线程抢 8 个核，每次被抢占都留下
   一个正在收包的 socket。

异步后端的做法是**把线程数变成参数**：`--workers W`，每路是一个 task，一个 worker
用一套 epoll 收它名下的所有流。40 路可以用 8 个 worker（8 核）、4 个 worker，
或者 32 个 worker——同一份代码。

---

## 2. 设计

### 2.1 只换收包这一件事（`corr-rs/src/backend.rs`）

`corr-rs` 新增一个接缝：`ReceiveBackend`（`start` / `finish`），由 `run_receive_with`
驱动。`run_receive` 变成"线程后端"的薄包装，`corr-async` 走同一个
`run_receive_with`。于是两条腿共用：**同一份配置、同一批 socket、同一个组帧器、
同一个 staging 环、同一个行调度器、同一个 CUDA ABI、同一个 writer、同一份报告**。
一份报告里的数字能对比，前提就是这些必须相同。

新增的接缝（合并后都是 `pub(crate)`）都是加法，不改老路径的语义（`cargo test` 90 + 7 全过）：

| 新增 | 作用 |
|---|---|
| `assemble::SlotRing::probe_free` | 非阻塞地问"这个槽现在能写吗"，含窗口仍开时的 `reprime` |
| `assemble::SlotRing::poll_free` | 同上，但答"不能"时把 waker 登记进环，由下一次 `release` 唤醒 |
| `assemble::StationAssembler::try_on_packet` | `on_packet` 的非阻塞孪生：拒绝时**什么都不消费**，同一个包以后重喂 |
| `net::MulticastSocket::recv_batch_nowait` | 空 socket 报 `WouldBlock`（而不是 `Ok(0)`），这是 ET 反应堆清 readiness 的唯一信号 |
| `net::MulticastSocket::set_nonblocking` | `O_NONBLOCK`，否则空轮询会按 `SO_RCVTIMEO` 阻塞 |
| `backend::ReceiveBackend` / `RecvSetup` | 收包后端接缝 |

### 2.2 一个 worker 是什么（`corr-rs/src/async_recv/backend.rs`）

一个 OS 线程 + 绑一个核 + 一个 `current_thread` tokio runtime（自带一套 epoll、
自带任务队列、不做 work-stealing，所以 task 不会在核之间迁移）。
station 按 `idx % W` 轮转分给 worker——各路速率相同，轮转天然均衡。

### 2.3 一个 station 的 task（`corr-rs/src/async_recv/station.rs`）

```
loop {
    if stop || ring.abort -> break         // 环 abort 后不会再释放槽，直接收工
    service_request()                      // 调度器要求"把这些帧宣告为缺失"，必须在任何等待之前
    if 还有上一批没放完的包:
        try_on_packet 逐个放
          全部放完 -> 发布进度，回去继续读 socket
          SlotBusy -> 让位/等 release，回来重喂同一个包（顺序因此不会乱）
          致命错误 -> 收工
    select! { tick(5ms) => 什么都不做，回去看 request
              socket 可读 => recvmmsg_nowait，拿到一批 }
}
```

### 2.4 为什么它现在是 corr-rs 的一个模块，而不是一个 crate

它一开始是独立 crate（`corr-rs-async/`），2026-09-18 合并进 `corr-rs/src/async_recv/`，
挂在 feature `async-recv` 后面，**且该 feature 默认开启**——`cargo build --release`
就能得到三个二进制，不需要记一个 flag；只有明确不要第二条腿时才 `--no-default-features`。
这么做的理由：

* **接缝退回 crate 内部**。跨 crate 时 `probe_free` / `poll_free` / `try_on_packet` /
  `note_slot_wait` / `recv_batch_nowait` / `set_nonblocking` / `RecvOutcome` / `StagingPtr::as_ptr` /
  `backend` 模块都必须是 `pub`；现在全部 `pub(crate)`，只留 `async_recv::main()` 一个 pub 入口
  给 `src/bin/corr_async.rs`（三行）。**corr-rs 的公有面比拆开时更小。**
* **`links = "corr_cuda"` + `DEP_CORR_CUDA_LIB_DIR` + 一个 build.rs 的 rpath 补丁全部删掉**：
  它们只因为 `cargo:rustc-link-arg` 不跨依赖传递而存在；同一个 crate 里 build.rs 自然管到所有 bin。
* **两条腿不可能版本错配**：一次 `cargo test` / `clippy` / `--check` 同时盖住它们。
* **门后代码不会腐烂**：feature 默认开启，所以 `cargo build --release` / `cargo test` 都编译它。
  只有 `--no-default-features` 时不编译——那时那几个接缝项由
  `#[cfg_attr(not(feature = "async-recv"), allow(dead_code))]` 标记，而它们的单元测试在两种配置里
  都跑（`cfg(test)` 与 feature 无关）。
* 代价：tokio 进了默认依赖树（要完全排除就 `--no-default-features`）；切换 feature 会重编一次
  CUDA `.so`（cargo 的 OUT_DIR 按 feature 集哈希）。

四个关键机制，每一个都对应一次实测故障：

* **非阻塞试探**：task 不能像线程那样停在 `wait_free` 上——它一停，同 worker 上另外
  20 多路就一起停。所以问完就走，`try_on_packet` 保证"被拒绝 = 什么都没消费"。
* **等 release，不等定时器**（`SlotRing::poll_free`）：这是本works最贵的一课。
  最初用"睡 200 µs 再问"代替线程的 condvar，单 worker 实测只跑到流量的
  **1/4**：释放本来在微秒后就到，却被 tokio 的 1 ms 定时器粒度量化掉了
  （`iters=3400 / parks=6718 / consume=0.05s` 三个数一眼看出线程是空闲而不是忙）。
  改成"环释放时唤醒等待的 task"之后，同一工况满速率。**释放是事件，就必须当事件送达。**
* **公平配额**（`DRAIN_BUDGET_PACKETS=512`）：socket 一直有数据时 `readable()` 永远 Ready，
  一个 task 可以永远不让位。每放 512 个包 `yield_now()` 一次，同 worker 上的流才会交错推进。
* **静默站的 tick**（`receiver.recv_timeout_ms`，与 `SO_RCVTIMEO` 同一个配置项）：
  一条断流的 station 必须还能醒来看 `prog.requested()`，否则它会把跨站水位线冻住。
  走的是同一个配置字段——这是同一个决定，不该有第二个旋钮。

顺带修掉一个老毛病：**环 abort 之后主动检查**，于是"流先停、`--duration` 后到"这条
已知挂死路径在异步腿上不存在了（线程腿靠"下一包到达时才撞上 Aborted"才发现，站静默时会挂）。

---

## 3. 正确性证据

| 检查 | 结果 |
|---|---|
| `cargo test`（corr-rs） | **89 + 7 通过**（含 3 条为本次新增：非阻塞试探不消费、窗口仍开时试探会移窗、非阻塞 socket 立刻报 `WouldBlock`） |
| `cargo clippy --release --all-targets`（两个 crate） | **0 条代码告警** |
| `corr --config ... --check` | ABI 7、布局与编号一致（ABI 未改） |
| 同一回放下的逐站对照（dry run） | 两腿每站 `pkts/lost/zero_filled/gaps` 一致，行数一致 |
| GPU 生产路径（2 站 / 1 卡 / 96 行） | 0 丢失、0 补零，文件与 `blocks.csv` 正常落盘 |
| **`tools/phase_test.sh` 的规范相位检验**（24 站、GPU、4 卡 `time_slice`、`--delay-stride 1`，`corr-async --workers 6`） | **276/276 基线**、`max abs(lag error) = 0` 样本、斜率误差 0.0000、overlay **0.00031 rad**、残差 0.0000/0.0001 —— 与 `docs/HANDOVER.md` 里钉死的线程腿基线**逐位相同**；该跑每站 1,250,109 包、`gaps=0 lost=0 zero_filled=0` |

最后一行是最强的一条：注入延迟的相位三角形是这套东西的物理判据，
异步收包路径与线程路径给出**完全一样的数字**。

---

## 4. 性能对比

### 4.1 方法

* **交错 A/B/A/B**，每腿 25 s（这台机器负载漂移是十分钟量级，块式对比会被淹没）。
* **三层丢包分别计数**：应用侧计数缺口（`lost`）、我们自己的 socket 内核丢包
  （`/proc/net/udp` 的 `drops`）、网卡硬件（`rx_missed_errors`）。**系统级**
  `Udp: RcvbufErrors` 不能用——本机常驻租户的丢包落在同一个计数器里，
  实测每条腿稳定 ~150 万个，与本次测量无关。
* **送达率**是主判据：`received_pps / offered_pps`。回放发的是无缺口计数器，
  两头一对就知道收包侧吃下了多少。
* 回放用 `--pattern zero`：`prng` 每包要算 4096 个 splitmix64（~8 µs），
  30 kpps/站时会把**回放器**而不是相关机顶到极限。
* 两条腿给同一个核池（`cpu.recv_cpus`）：`corr` 把池按轮转分给 24 个收包线程，
  `corr-async --workers W` 把同一个池分给 W 个 worker。

### 4.2 结果（2026-09-17，本机共享负载 12–31）

| 工况 | 腿 | 送达 | 我们 socket 丢包 | µs/包 | 25 s CPU | 结论 |
|---|---|---|---|---|---|---|
| **24 路 / 8 核** (3:1) @30 kpps | threads | 100.8% | 0 | 3.15 | 57.1 s | 基线 |
| | async **8** worker | 100.3% | 0 | 4.10 | 74.1 s | 贵 +30% |
| | async **2** worker | 100.3% | 0 | **2.24** | **40.5 s** | **省 29%** |
| **24 路 / 4 核** (6:1) @30 kpps | threads | 100.8% | 0 | 2.78 | 50.5 s | 基线 |
| | async 4 worker | 100.3% | 0 | 3.29 | 59.3 s | 贵 +18% |
| **40 路 / 16 核** (2.5:1) @20 kpps | threads | 101.3% | 0 | 3.35 | 67.9 s | 基线 |
| | async **4** worker | 100.5% | 0 | **3.21** | **64.6 s** | **省 5%** |
| | async 8 worker | 100.5% | 0 | 4.12 | 82.9 s | 贵 +22% |
| | async 16 worker | 100.7% | 0 | 4.51 | 90.8 s | 贵 +34% |
| **8 路 / 8 核** (1:1) @60 kpps | threads | 100.3% | 0 | 2.91 | 35.0 s | 基线 |
| | async 8 worker | 100.0% | 0 | 4.30 | 51.6 s | 贵 +48% |

独立复核（`/usr/bin/time`，20 s，24 路/8 核 @30 kpps）：

```
threads   user=5.04 sys=41.90  -> 46.94 s CPU   30,318 pps/站
async 2   user=5.79 sys=25.64  -> 31.43 s CPU   30,121 pps/站     <- 省 33%
async 4   user=10.03 sys=37.74 -> 47.77 s CPU   30,125 pps/站
```

差值几乎全在**内核时间**（41.9 s vs 25.6 s）：24 个线程抢 8 个核的那笔调度账。

### 4.3 三条能用的结论

1. **流数 > 核数时，异步多路复用更省 CPU**——前提是 worker 数按负载定，不按核数定。
   24 路用 2 个 worker 比 24 个线程省 **29–33%**；40 路用 4 个 worker 省 **5%**。
   省下来的是内核时间：不再有"24 个忙轮询线程轮流被抢占"。
2. **异步的代价按 worker 数走，不按流数走**：同一份 40 路负载，
   W=4 → 3.21 µs/包、W=8 → 4.12、W=16 → 4.51。所以
   **W 应该取"刚好覆盖负载"的最小值**（本机一条 worker 的收包能力约 400–450 kpps），
   `--workers 核数` 反而是最差的选择。这也解释了为什么 `--workers` 默认值
   （= 计划里的收包核数）在实测里总是偏贵。
3. **流数 ≤ 核数时不要用异步**：8 路 / 8 核，一线程一核没有超订可省，
   反应堆的 epoll + 任务簿记就是纯开销，**贵 48%**。

### 4.4 补零率（`blocks.csv`，生产档，GPU + 落盘）

上面三张表都是 dry run，没有 `blocks.csv`，所以**补零率一开始没比**——这是本文第一版
的漏项。补做的方法：`--produce` 让每一腿跑真正的 GPU + writer 路径，
逐行读 `blocks.csv`（`row, station, start_frame, n_frames, valid_frames, zero_filled`），
判据用**逐行**：整行全零缺失的行数（`clean`）、补零率（补零帧 ÷ 总帧）、
最坏行、以及前半程/后半程的补零率（是否在恶化）。

| 工况 | 腿 | 行数 | 全干净行 | **补零率** | 最坏一行 | 环阻塞次数 |
|---|---|---|---|---|---|---|
| 24 路 / 8 核 @30 kpps | threads | 45 | 45 | **0.0000%** | 0 | 0 |
| | async 4 | 44 | 44 | **0.0000%** | 0 | 0 |
| 24 路 / 4 核 @40 kpps | threads | 48 | 48 | **0.0000%** | 0 | 24 / 858 |
| | async 4 | 48 | 48 | **0.0000%** | 0 | 0 |
| 24 路 / 8 核 @30 kpps，**环压到 1 个可写槽**（`block_slots=2`） | threads | 35 | 35 | **0.0000%** | 0 | 22566 / 23882 |
| | async 4 | 36 | 36 | **0.0000%** | 0 | 0 |
| 24 路 / 4 核 @60 kpps（**超载**，1.44 Mpps 打进 4 核） | threads | 67 / 68 | 26 / 27 | **13.4% / 14.7%**（前半→后半 0.94%→28.4%） | 109851 | 13737 / 12815 |
| | async 4 | 21 / 25 | 20 / 24 | **0.037% / 0.091%** | 4492 | 0 |

**能跟上的三种工况里，两条腿的补零率都是 0.0000%，干净行数也一样**——包括把环压到
"只剩一个可写槽"这种最紧的合法配置。所以在这台机器能达到的负载下，
**换收包形状不改变补零率**。这是本节能负责的那半句话。

**超载那一行不能当成"异步质量更好"来读，原因是两腿看到的流量本身不同**：

* 实测计数器前进速度：threads 55,731 / 55,921 pps/站，async 17,399 / 20,881 pps/站
  （`offered_measured`，从报告里的 `start frame` 与逐站 `progress` 反推，
  不信 `--rate` 这个请求值——回放器本身是 CPU 受限的）；
* 网卡硬件丢包 `rx_missed_errors`：threads 2.3–10.1 万/腿，async 8.3–35.2 万/腿，
  与流量差同向；
* 一个落后的接收端处理的是**更老、更连续**的流量，补零自然少——
  这正是 `docs/ARCHITECTURE.md` §5 那条"过载时应用自己的缺口计数不可信"的另一面。

要把它变成可信的对比，得在**运行中**采 `/proc/net/udp` 的 `rx_queue`（进程退出后
socket 就没了，事后采到的是 0，本轮就吃了这个亏）与 `/proc/net/softnet_stat`，
把"没收到"和"收到了但落后"分开。在那之前，超载工况下两腿的补零率都不可引用。

**超载时能确认的设计差别**（两轮都复现）：thread 腿的等待发生在**环里面**——
每腿 1.2–1.4 万次环阻塞事件；异步腿每一腿都是 **0 次**（它在 park 上睡，socket 吸收）。
thread 腿的补零率还在单次运行内恶化（前半 0.94% → 后半 28.4%）。

### 4.5 16384 信道 / 1 秒积分档的验证（24 路真实前端）

> **⚠ 本节与 §4.5.0 的绝对数字都建立在"writer 仍在管线线程上"的代码（`4410e09`）之上。**
> 另一侧的同步版已把逐行写盘 + fsync 移出管线线程（`writer-thread` 分支，`497ff7d`），
> 而这正是环被关掉的原因；带着那个修复重测，两条腿都能到 0.0000–0.005%（见 §4.7）。
> 本节保留，是因为它记录的症状（每站等环 33 s/100 s、socket 溢出、补零 12–27%）
> 就是那个 bug 的形状，也是异步腿第一次撞上它的样子。

这是用户点名的验证，也是**本轮唯一一次在生产速率下跑通异步腿**的测量。
档案就是仓库里那份：`config/correlator-24-nf16384.yaml`
（24 路真实前端、`n_freq=16384` → 8 包/帧、`n_cum=24576` = 1.0067 s、
`recv_buffer_bytes=256 MiB`、`block_slots=11`、4 卡 `time_slice`），
**生产档**（真 GPU + writer，`blocks.csv` 才有）跑 30 s，交错 A/B/A/B：

| 腿 | 行数 | 全干净行 | **补零率** | 最坏一行 | 主机侧包丢失 | 该腿 CPU |
|---|---|---|---|---|---|---|
| r1 threads（24 线程/24 核） | 28 | 3 | **17.67%** | 589824（一整行全补零） | 18.6% | 354 s |
| r1 async 24 worker | 28 | 1 | **22.89%** | 187743 | 22.5% | 357 s |
| r2 threads | 29 | 2 | **12.21%** | 589824 | 12.9% | 345 s |
| r2 async 24 worker | 29 | 0 | **26.66%** | 589824 | 26.2% | 343 s |

**答案：异步腿在这份档案上的补零率是 23%–27%**（28–29 行里只有 0–1 行是全干净的），
线程腿是 12%–18%。同一窗口内交错，CPU 几乎相同（≈11.8 核/腿）。
补零率与**主机侧丢包率**几乎一比一，说明这些补零来自包没到齐，
不是来自组帧或调度：丢包是**成串**的（否则 8 包/帧会把 L 的丢包放大成 1-(1-L)^8），
成串丢包 = socket 缓冲溢出。

**为什么会溢出：** 同档案的 dry run（不提交 GPU，20 s）把这些数指得很清楚：

| dry 腿 | 每站收到 | 计数缺口 | 内核丢包（`/proc/net/udp` drops，运行中采样） | 结束时 socket 积压 |
|---|---|---|---|---|
| threads | 194 kpps/站 | **0** | **0** | — |
| async 24 worker | 182 kpps/站 | 6.6% | **约 287 万（≈360k/s）** | 每 socket ≈400 MB（24 个合计 9.5 GB） |

也就是说：**不提交 GPU 时线程腿一个包都不丢，异步腿已经差 6%**，
差额累积成积压、积压溢成内核丢包。生产档里环的压力更大（GPU 持有 staging 槽），
两条腿都掉，异步掉得更多。前端本身没问题：dry 线程腿看到的计数器前进就是 194 kpps/站，
与设计值 195.3 一致、0 缺口。

> **文档里这份档案的参考值是 0.1050%（线程腿、300 s、满速率）。今天两条腿都差了 100 倍以上**，
> 而且 dry 线程腿能跑出 0 丢包——所以差别不在前端也不在网卡，而在**环的压力**：
> 该档案 `ring_lead` 未显式设置 → pipeline 持有 1 个槽，接收侧领先 10 个槽，
> GPU 一起作用时接收侧就会被顶到。这条不属于本文的范围，但**别把今天的 12–27% 当成档案的正常水平**。

#### 4.5.0 更多窗口，以及"今天谁都不行"的原因

同一档案又在更安静的窗口（起始 load 9.2）与另一窗口各跑一次交错 A/B，并试了不同 worker 数：

| 窗口 | threads 补零率（丢包） | async24 补零率（丢包） | async12 补零率 |
|---|---|---|---|
| 负载 20–34 | 17.7%（18.6%）/ 12.2%（12.9%） | 22.9%（22.5%）/ 26.7%（26.2%） | — |
| 起始 load 9.2 | 9.0%（8.6%）/ 12.3%（11.5%） | 15.1%（14.5%）/ 16.6%（15.8%） | — |
| 又一窗口 | 14.2%（16.0%） | 32.4%（31.1%） | **42.7%（41.5%）** |

三条读法：

1. **线程腿在那时也远不是 0.1050%**（9%–18%，比文档差 100 倍以上）。所以那**不是"同步行、异步不行"**。
   当时的归因（"H2D 每卡 4.55–4.7 GB/s 顶到链路上限、pipeline 四分之一时间在等设备释放槽"）
   **只对了症状、错了因果**：真正的因果由另一侧的会话查清并修好——
   **writer 一直在管线线程上同步写 300 个文件 + 300 次 fsync（一行 158 ms，其中 142 ms 是 fsync），
   每一行都把管线线程钉住 158 ms，环因此被关掉、接收侧被拒槽、socket 溢出。**
   H2D 偏低（4.55–4.7 而不是 4.78–4.80）是**管线被钉住的结果**，不是原因。
   同一个档案 dry run（不提交 GPU）线程腿是 0 丢包；而带着 writer 线程的修复重测，
   两条腿的"每站等环"都变成 **0.0 s/100 s**（§4.7）。
2. **异步腿在任何窗口都更差**（1.3–2.3×），而且这一点**与本文 §4.3 的结论一致**：
   这份档案是 24 路 / 24 个收包核 = **1:1**，正是"流数 ≤ 核数，别用异步"那一格；
   反应堆每读一批要多走一次 poll，没有多路可复用。
3. **worker 数不能少也不能多**：`--workers 12` 的容量 ≈ 12 × 400–450 kpps ≈ 4.8–5.4 Mpps，
   正好压在 4.66 Mpps 的负载上 → 丢包 41.5%（比 24 worker 更差）；
   `--workers 24` 容量够但每个 worker 只有一路，poll 开销最大。
   这份档案想用异步，容量和开销之间没有舒服的取值——**它本来就不需要异步**。

结论：**16384 信道 / 1 秒、24 路、满速率，用线程后端**。异步后端该用的地方是
"流数 > 核数"（40 路 / 16 核那种，见 §4.2/§4.3），而且要按负载选 W。

#### 4.5.1 这次验证抓出来的 bug（已修，`try_on_packet` 的顺序）

第一次跑这个档案时，异步腿**60 s 只出 0–1 行**（线程腿 58 行），
`stall: 21 recoveries`，计数器只前进 1.5 kframes/s——看着像"异步撑不住生产速率"，
其实是**一个新代码里的死锁**，躲在非阻塞试探里：

`try_on_packet` 原来**先探环、后声明计数跳变**。一个跳变了的包（计数器跳过一段）
属于**比已释放前沿更靠前**的槽，于是试探拒绝它；而"把跳过的帧宣告为缺失"这一步
因此没跑，这个站已发布的 progress 停在它手里的包**后面**。
调度器要等所有站跨过槽边界才提交、才释放——于是**站等环、环等水位线、水位线等这个站**，
只能靠 200 ms 的停滞恢复一轮挪一个槽（实测节奏 175 ms/slot，正好对上）。

修法：**先声明跳变（`finalize_range` + `advance_to`，后者会发布），再探环**
——也就是 `on_packet` 原本的顺序。修完之后同一档案同一参数：**58 行 / 60 s**，
与线程腿一模一样。这条与 `docs/ARCHITECTURE.md` 的不变量 1 是同一件事的另一面：
**任何"可能阻塞的等待"之前必须把已取得的进展发布出去**，包括"把包放回去等槽"这种等待。
`FeedOutcome::SlotBusy` 的契约也随之改写：它保证"这个包没有被消费"，
但**不**保证"什么都没发生"——跳变该宣告的仍然要宣告。测试
`try_on_packet_declares_a_jump_before_refusing_the_slot` 钉住这个顺序。

### 4.7 与同步版 `writer-thread` 的交叉对拍（2026-09-18 13:55–14:10）

另一侧的会话把 writer 移出管线线程（`/scratch/jhgu/head-wt` 的 `writer-thread` 分支，
`497ff7d`；只改 `corr-rs/src/writer.rs`），并留下文档 `/scratch/jhgu/head-wt/docs/baseline_test_sync.md`。
我做了三件事：

1. **独立复算他们的数字**：用他们的 `blocks.csv` 重算批次 C 与批次 E 的逐行补零率，
   §4.2 的两列、§4.3、§4.4 **逐位吻合**（如新 fsync=1 六轮 15.3172/2.2165/0.0106/0.1111/0.0000/21.9084%）。
2. **把异步腿搬到他们的代码上**：`git worktree add -b async-on-writer` 取 `writer-thread`，
   再把本会话对 `corr-rs` 的改动与当时的 `corr-rs-async/` 打进去（改动文件互不重叠；
   那时它还是独立 crate——**合并之后这件事变成一次 `git merge`**，见 §2.4），
   于是两条腿**共用同一份 writer / 管线 / 调度器**，只差收包形状。
3. **按他们的协议对拍**：三臂（`old` = `4410e09`、`new` = `writer-thread`、`async` = 本会话后端
   跑在 `writer-thread` 上，`--workers 24`）× 3 轮拉丁方，每臂 **100 s**，同一窗口，配置
   `/scratch/jhgu/cfg-ab-fsync1.yaml`（= `config/correlator-24-nf16384.yaml` 的值，未改任何字段）。

| 轮（顺序） | old 补零率 | new 补零率 | **async 补零率** | new−old | async−new |
|---|---|---|---|---|---|
| 1（old→new→async） | 2.1591%（1/98 干净） | 0.0003%（96/98） | **0.0000%（98/98）** | −2.159 | −0.0003 |
| 2（new→async→old） | 3.0823%（1/97） | 0.0000%（98/98） | **0.0012%（97/98）** | −3.082 | +0.0012 |
| 3（async→old→new） | 6.4220%（1/98） | 1.1584%（0/98） | **0.0045%（84/98）** | −5.264 | −1.154 |

结构性指标（同一批）：

| | old | new | async |
|---|---|---|---|
| 每站等 staging 槽 | 32.2–34.8 s / 100 s | **0.0** | **0.0** |
| pipeline release-wait | 16.8–19.9 s | 2.8–4.1 s | 2.1–2.3 s |
| 环阻塞事件 / 自旋 | 15.2 万 / 30.7 亿次 | 0 / 0 | 0 / 0 |
| 写线程自报 | （在管线线程上） | 106–123 ms/行 | 105–109 ms/行 |
| H2D 聚合 | 19.0 GB/s | 19.0 GB/s | 19.1 GB/s |
| CPU（user+sys） | 345–378 + 1014–1042 s | 247–251 + 983–989 s | 244–248 + 970–985 s |

**结论**：在带 writer 线程的代码上，**异步腿与同步腿已经等价**——逐轮配对差 −0.0003 / +0.0012 /
−1.15 个百分点，三条腿都出 97–98 行、都跑满 195 kpps/站，CPU 也基本相同（异步反而少一点 user）。
我在 §4.5.0 里测到的"异步差 1.3–2.3 倍"**是"writer 钉住管线 → 环持续被关 → 每站等环 30+ s"
那个工况的产物**：环一关，两条腿的"被拒之后怎么回来"就分出了高低；环从不开（new/async 这里都是
0.0 s）时，差别消失。**这份档案上异步没有任何收益，也没有代价——因为它本来就是 1:1，
收益在"流数 > 核数"那边。**

### 4.6 这条数据的边界（必须说清楚）

* 全部是 **dry run**（不提交 GPU、不落盘）：隔离收包+组帧+调度这一层。
  生产（GPU 持有 staging 槽）时环释放更慢、park 更频繁，
  那条路径本次只做了功能验证（§3 的 96 行与相位检验），**没做性能对比**。
* 单站速率 20–60 kpps，**低于生产的 195 kpps**。本机做不到满速率回放：
  195 kpps × 24 路的回放本身要把 24 个发送线程和收包线程抢核，
  这一点 `docs/HANDOVER.md` §5 早就写明了。实测 190 kpps/站 × 8 路（1.5 Mpps、
  6.2 GB/s）时**两条腿都吃不下**（送达 52–54%），所以生产速率下的结论仍然是空的。
* 判据是"送达率 + CPU + 逐行补零率"：补零率见 §4.4（生产档、`blocks.csv`；见 §4.4 与 §4.5，
  也可以用 `tools/rows_trend.py` 单独核对一份运行目录）。
* 负载区间 12–31，同一条腿两轮之间的重复性很好（如 async 单 worker 25 s 内
  consume 0.03–0.05 s），但**跨工况比较必须在同一窗口内交错**，这是老规矩。

---

## 5. 两个把结论带偏过的测量陷阱（下次别再踩）

1. **应用侧的 `lost=0` 不是"没丢包"的证据。** 一个收不过来的接收端永远在追
   缓冲区里最老的那段包，永远看不到内核开始丢包的那个前沿，于是缺口计数是 0：
   实测异步 1 worker 只吃下 15% 的流量，`lost=0`、`zero_filled=0`，
   而 `/proc/net/snmp` 的 `RcvbufErrors` 正以 ~600k/s 增长。
   **判据必须是"送达率 + 内核丢包"，不是应用自己的缺口计数。**
2. **两个静默返回 0 的指标**：`/proc/net/snmp` 的头部第一列是 `"Udp:"`（带冒号），
   按 `"Udp"` 匹配会拿到空字典，于是"内核丢包"永远是 0；
   `rx_discards_phy` 在本机网卡上**根本不存在**，`cat` 失败被吞掉，
   于是"网卡丢包"也永远是 0。本机该看的是 `rx_missed_errors`。
   两个 bug 都在 `tools/async_bench.py` 里修掉了——**每个计数器都要对着真文件验证一次**。

---

## 6. 未解决

* **本机 epoll 驱动的收包似乎让网卡 RX 环溢出得更多**，三处证据同向：
  超载腿 `rx_missed_errors` async 8.3–35.2 万 vs threads 2.3–10.1 万；
  真实前端下 thread 腿抓到 68 kpps/站而异步腿只有 5–10 kpps/站；
  两者都是"异步看到的流量更少"。如果是真的，那它抵消了异步在多路复用上省下的 CPU。
  需要按 `docs/ARCHITECTURE.md` §5 的三层计数加上 `/proc/net/softnet_stat`
  与 `/proc/net/udp` 的**运行中**采样来定性——**在那之前不要引用这条**。
* **真实前端的对照（本次没做成）**。测量窗口内前端把计数器走得很快
  （约 190 kpps/站）但送到本机 socket 的只有 17–21 kpps/站
  （`tools/mcast_rate.py`，256 MiB 缓冲，独立进程），
  而且同一批组上还有常驻租户。在这种工况下"一个接收端吃下多少"主要由
  主机的投递路径（softnet / NAPI / 中断）决定，而不是由收包代码的形状决定：
  实测线程腿 68 kpps/站、异步腿 5–10 kpps/站，两者 socket 丢包都是 0、
  网卡 `rx_missed_errors` 也只差 30%。**这个差是不是"忙轮询把 NAPI 拉得更勤"造成的，
  本次没有验证**。要验证得按 `docs/ARCHITECTURE.md` §5 的三层计数来做，
  并加上第三层：`/proc/net/softnet_stat` 的 backlog 丢弃，与 A/B/A/B 交错。
  在验证之前，真实前端下的结论**不可引用**。
* **生产档（GPU）下的对比**：需要 `config/correlator-24-replay.yaml` 那一档
  （staging 2.3 GiB、4 卡 `time_slice`）跑 A/B，本次只做了功能验证。
* ~~异步腿在生产速率下丢得更多~~ **已解释（§4.7）**：那是"writer 钉住管线线程"的产物；
  换成 `writer-thread` 后两条腿的每站等环都是 0.0 s，补零率与 CPU 等价。
  仍未验证的是 **坏窗口**（对方文档 §5：同一份代码也能跳到 22–25%）里两条腿的关系——
  我那 9 次全落在好窗口，坏窗口里没比过。
* **`--workers` 默认值该改**：按 §4.3 第 2 条，默认给"计划的收包核数"是偏贵的，
  合理默认是按负载估（≈ ceil(总 pps / 400k)），或者干脆要求显式给。
* **park 路径在生产 backpressure 下的行为**没有单独测过（dry run 里环几乎不拒）。
* 环的 waker 列表是"唤醒即清空"：24 路同时 park 时每次 release 唤醒全部等待者，
  是一次小的惊群。量级上没问题（每次唤醒只是重新探一次原子），但没有测过上限。

---

## 7. 复现

```bash
# 构建：默认就产三个二进制（corr / corr-replay / corr-async）
cd corr-rs && cargo build --release                        # 默认含 async-recv
cd corr-rs && cargo build --release --no-default-features  # 只要一线程一站那两个（不引入 tokio）
cd corr-rs && cargo test                                  # 两条腿都编译、都测

# 单条腿（回放自动起停，每条腿 25 s，A/B/A/B 共约 2 分钟）
cd ..
python3 tools/async_bench.py --tag my-24s-8c --stations 24 --pool 16-23 \
        --rate 30000 --duration 25 --rounds 2 --legs threads,async:2

# 40 路目标
python3 tools/async_bench.py --tag my-40s-16c --stations 40 --pool 16-31 \
        --rate 20000 --duration 25 --rounds 2 --legs threads,async:4

# 规范相位检验（异步腿）——把 phase_test.sh 的 CORR 换成 corr-async 并给 --workers
CFG=config/correlator-24-replay.yaml RATE=50000 DUR=25 CORR_ARGS="--workers 6" \
        CORR=./corr-rs/target/release/corr-async tools/phase_test.sh
```

`bench/<tag>/` 里留着每一腿的完整日志（报告原文）、`results.json`、`results.csv`；
`bench/` 已在 `.gitignore` 里。
