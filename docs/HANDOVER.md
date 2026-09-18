# 交接文档：从这里读起

**本文是给"下一个接手的人 / 下一个对话"的入口。** 30 秒定位、当前在哪、下一步做什么、
已知的坑。历史过程在 `HANDOFF.md`（长，是日志不是入口），结构在 `docs/ARCHITECTURE.md`。

---

## 0. 一分钟定位

RFSoC 阵列望远镜的**实时相关机**：收 24 路组播 UDP（每包 4104 B = 8 B 小端计数器 + 4096 int8），
每站做 R2C FFT，按 `j <= k` 累加 `Y_j·conj(Y_k)`，每行（1 s 积分）给每条基线落一个
`complex64[ntime][nch]` 文件。

两半：`corr-cuda/`（C ABI + CUDA，负责 staging/拷贝/FFT/相关/多卡放置）与
`corr-rs/`（Rust，负责收包/组帧/调度/落盘/判读）。

```bash
cd corr-rs && cargo build --release      # 顺带构建 ../corr-cuda 的 .so；产 corr + corr-replay + corr-async
cargo build --release --no-default-features   # 只要一线程一站那两个（不引入 tokio）
./target/release/corr --config ../config/correlator-24.yaml --check   # ABI/布局/编号一致性
cargo test                               # 90 + 7（默认含 async-recv，两条腿都编译）
cargo test --no-default-features          # 同上，但不编译异步后端
                                             # C 侧（abi_smoke / multi_device / 两个 golden）
cmake -S ../corr-cuda -B ../corr-cuda/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../corr-cuda/build -j && (cd ../corr-cuda/build && ctest)
tools/doctor.py --config ../config/correlator-24.yaml                 # 机器侧体检（先跑这个）
```

> **构建目录必须放在工作区里**：工作区文件系统的时钟比 `/tmp` 慢十几分钟（`date` 与
> `touch` 落盘的 mtime 差 13 min）。把 CMake 构建目录放到 `/tmp` 会让 make 认为目标文件
> 比源文件新，**永远不重编**，而且 `ctest` 跑的是上一版二进制 —— 这一轮就因此白跑过一次
> "新代码输出全零"的假故障。用 `corr-cuda/build/`（`.gitignore` 里有）。cargo 不受影响：
> 源和 `target/` 在同一个文件系统上。

## 1. 当前状态（交接点）

* 两个会话在同一条链上：重构链 `10de600 → ce15866`（A）→ 补零解耦 `bd33583 → b76339e`（B，
  ABI 6 → 7）→ 判读工具与文档 `477d45b`、`bf29bae`、`36e87d3`、`edfa08e`、`2b3fc7e`（B）
  → `684e827`（A：`tools/phase_test.sh`）。当前 HEAD `684e827`。
* **补零解耦的第 3、4 步已落地（`b76339e`，ABI 7）**：掩码随样本上设备、kernel 按
  掩码整帧清零、组帧器不再补零也不再为补零等环。下面这条与 §2 的实测都以它为准。`cargo test` **81 + 7 通过**、
  `clippy` **0 条代码告警**、C 侧 `ctest` **6/6**（abi_smoke / multi_device / 两个 numpy golden）。
  验收实测见 `docs/DESIGN-zero-fill-decoupling.md` §8：判据 1、3 通过（含逐字节对照），
  **判据 2 的后半与判据 4 的预期没有兑现**，原因在 §8.3/§8.4，下一步在 §8.6。
* 改动前的工作树停在 `bd33583`，保留成一个 git worktree：`/scratch/corr-baseline`
  （A/B 对照腿用的旧二进制就从那里编）。
* 实跑证据：**24 路满速率、4 卡 `time_slice`（R=1）**，每卡 H2D 4.80 GB/s（满需求）；
  5 分钟跑 **299/300 行准点**，补零率 **0.0013% – 0.0376%**（随负载 16 → 24 变化，见 §3）；
  相位–频率检验 **276/276** 基线精确复现注入延迟（`tools/phase_triangle.py`）。
* **2026-09-16 复测（ABI 7，补零解耦落地后，真实前端 24 路）**：`config/correlator-24.yaml`，
  4 卡 `time_slice`，300 s 跑 **299 行**（1.0014 s 节拍，与上面那条旧记录一致）；每卡 H2D
  **4.79 GB/s**（需求 4.80，聚合 **19.2 GB/s**），SM 利用率 ~40% —— 墙是主机链路不是算力；
  进程 **11.7 核 / 32 核**（另跑 90 s 采样所得）。补零 **0.0105%**（5 min，load 均值 21.4、
  峰 28.6）与 **0.0033%**（90 s，load 均值 17.9），**230/299 行整行零缺失**，三段
  35.0k/18.1k/20.6k 无上升趋势；0 次 stall 恢复、无停止原因、输出 2.8 GB（=9.8 MB/s）。
  相位-频率：τ=0/1/2 三档拟合 **0.0000/1.0000/2.0000** 样本、最大残差 ≤1e-4 rad
  （`tools/phase_check.py`）；24 站 delay-stride 三角形 **276/276**、lag 误差 0 样本、
  overlay 0.00027 rad（`tools/phase_triangle.py`）。图：根目录 `phase_check.png`、
  `phase_triangle.png`（残差版）、`phase_triangle_lines.png`（两线版：每格虚线是模型、
  实线是数据，384 通道窗口）。`phase_triangle_lines.png` 是 `--delay-stride 1` 基线（站 i 延迟 i 个采样点，i=0…23）：rate 20000、60 s、**0 丢包 0 补零**，276/276、lag 误差 0 样本、overlay 0.00034 rad。复现命令见 `tools/README.md`，原始日志在 `/scratch/zb/`。
* **相位检测的规范命令是 `tools/phase_test.sh`**（参数钉死：`--rate 50000 --duration 25 --delay-stride 1`，
  分析用 `phase_triangle.py --stride 1 --mosaic phase`，**窗口取默认 1536**）。2026-09-16 在 ABI 7 上
  按默认跑：**276/276、max |lag error| 0 样本、slope 误差 0.0000、overlay 0.00031 rad、残差 0.0000/0.0001**，
  全 24 站 `gaps=0 lost=0 zero_filled=0`、逐行 5/5 行整零，load 9.1/16.8/22.1 —— 与 A 的基准逐位相同。
  图：`phase_triangle_pinned.png`（脚本原样输出）与 `phase_triangle_pinned_zoom.png`（同一次运行、
  同一窗口，只是面板放大到 1.05 in/150 dpi 以便阅读）。**两线版图的"斜率看着不同"只来自
  `--phase-window`**：每通道 0.021973°（fft_len = 2·nch = 16384），一格里铺 1536 个通道还是 384 个，
  决定它看着有多陡、画几圈；拟合值不受窗口影响。
* **实时状态行（2026-09-18）**：`monitoring.interval_ms`（默认 1000，0 = 关）现在真的生效 ——
  pipeline 线程每个间隔往 stderr 打一行：行数/行率、逐行补零率（累计 + 窗口）、计数缺口、
  **接收侧等环占比**、站间进度散布（槽）、环阻塞事件增量、停滞恢复增量、本进程核数/机器 load、
  H2D 相对"需求字节率"的百分比、pipeline 等设备归还槽的占比、我们 socket 的内核丢包增量，
  末尾 `capacity:` 指出**先顶到上限的资源**（socket 丢包 → 环 → GPU 链路 → 整机 load → 本进程 CPU）。
  为此 `StationProgress` 增 `slot_wait_ns`，两条腿各自发布（线程腿 `wait_free`、异步腿 park），
  所以"这一秒有多少时间没在读 socket"两边同口径。`monitoring.stats_csv` 仍是接受但不实现的字段。
  注意：**冷启动头十几秒**（cuFFT plan、首行对齐、建 304 个文件）等环占比会明显偏高，25 s 的短跑
  不代表稳态 —— 稳态判据仍要用 300 s 的逐行 `blocks.csv`。
* **输出命名（2026-09-18）**：文件名改成
  `<file_prefix><站j><站k>-<时刻>.bin`（每个 baseline 一个，站名直连）与
  `<file_prefix><时刻>-time.bin` / `-time.txt` / `-blocks.csv` / `-metadata.yaml`；
  `<file_prefix>` 是新增配置项 `storage.file_prefix`（默认空、不含路径），`<时刻>` 是该套
  第一条行的本地时间（`storage.file_prefix_format` 默认改为 `"%Y%m%d-%H%M%S"`）。
  metadata 里新增 `file_prefix` / `stamp` 两半，并逐条列出每个 baseline 的实际文件名，
  读端不再重建命名规则；旧布局（`baseline_0000_j00_k00.bin` / `rows_ms.*`）**只读兼容**
  （`tools/corr_common.py` 认出两种布局），新二进制不会续写旧套。三个二进制都搬到
  `corr-rs/src/bin/`（`corr.rs` / `corr_async.rs` / `replay.rs`）。
* 下面这条是 2026-09-17 的原始记录（命名已被上面这条取代，其余行为不变）：
  输出按**本地日期**分套 —— `<prefix>baseline_*.bin` /
  `<prefix>rows_ms.*` / `<prefix>blocks.csv` / `<prefix>metadata.yaml`，`<prefix>` =
  该套第一条行的本地 `%Y%m%d_%H%M%S`。跨本地零点自动关旧套、开新套；`row_index` 继续全局、
  按行的**起始时刻**归属、当天重启追加当日那套。开关 `storage.daily_files`（默认**开**，
  对所有运行生效）+ `storage.file_prefix_format`；当日改了配置时默认拒绝启动，用
  `--allow-second-set` / `--set <prefix>` 显式继续。设计与实测见
  `docs/DESIGN-daily-files.md`；端到端判据 12 条全过（`tools/daily_rollover_test.sh`：
  TZ 把本地零点挪进窗口，20 kpps 与 50 kpps 各一次 —— 前者跨零点 **0 帧补零**，
  后者紧随边界的一行掉 6.3%，切套的文件工作 ~0.5 s）。
* 判读工具现在认前缀：`rows_trend.py` / `longrun_report.py` / `compare_blocks.py` /
  `compare_runs.py` / `phase_check.py` / `phase_triangle.py` 都可加 `--prefix`，
  默认取**最后一套**；不带前缀的老目录照旧能读（`tools/corr_common.py` 里的 `find_sets`）。
* **异步收包后端（2026-09-17；2026-09-18 合并进 `corr-rs/src/async_recv/`，feature `async-recv` **默认开启**）**：`docs/ARCHITECTURE.md` §6 那条
  "40 收包线程 > 32 核"的墙有了实现与实测。线程数变成参数（`--workers W`，一个 worker
  一套 epoll 收多路）；接缝是 `corr-rs/src/backend.rs` 的 `ReceiveBackend`，
  两条腿共用同一份编排/报告/writer/ABI，只换收包形状。**只加了公有 API，没改老路径语义**：
  `cargo test` 89+7、clippy 0 告警、`--check` 仍报 abi=7；GPU 生产路径端到端跑通（96 行、
  0 丢失 0 补零）；**规范相位检验 276/276、lag 误差 0 样本、overlay 0.00031 rad —— 与
  本文 §1 里钉死的线程腿基线逐位相同**。性能（交错 A/B、每腿 25 s、dry run）：
  24 路/8 核 `--workers 2` 比 24 线程**省 29–33% CPU**（送达都 100%），`--workers 8` 反而
  贵 30%；40 路/16 核 `--workers 4` 省 5%、`--workers 16` 贵 34%；8 路/8 核（流数≤核数）
  线程便宜 48%。**结论：异步的代价按 worker 数走不按流数走，W 要按负载定
  （一条 worker ≈ 400–450 kpps），流数 ≤ 核数时别用。** 设计与全部数据
  （含真实前端那次**没做成**的对照与两个把判据带偏过的测量陷阱）见
  `docs/DESIGN-async-receive.md`，台架 `tools/async_bench.py`。
  **补零率（生产档、逐行 `blocks.csv`）**：24 路/8 核 @30k、24 路/4 核 @40k、以及
  "环压到 1 个可写槽"（`block_slots=2`）@30k 三种工况下，**两条腿都是 0.0000%**
  （全干净行 45/45、48/48、35/35 与 44/44、48/48、36/36），即**能跟上的负载下换收包形状
  不改变补零率**；超载（24 路/4 核 @60k）时 thread 腿补零 13–15%（前半→后半 0.9%→28%、
  1.2–1.4 万次环阻塞）而异步腿 0.04–0.09%——但两腿看到的流量本身不同
  （实测 counter 前进 55.9k vs 20.9k pps/站，网卡丢包 2.3–10.1 万 vs 8.3–35.2 万），
  **那一行不能读成"异步质量更好"**，理由与需要补的采样见 `docs/DESIGN-async-receive.md` §4.4/§6。
  **16384 信道 / 1 秒档（真实前端、生产档、交错 A/B，30 s 腿）**：异步腿补零率
  **22.9% / 26.7%**（28–29 行里 1 / 0 行全干净），线程腿 12.2% / 17.7%；两腿 CPU 相同（≈11.8 核）。
  补零率与主机侧丢包率几乎一比一（成串丢包 = socket 溢出）：dry 对照里线程腿
  **0 丢包**，异步腿已差 6%、内核丢包 360k/s、每 socket 积压 400 MB。
  前端/网卡不是原因（dry 线程腿 194 kpps/站、0 缺口、计数器与设计值一致）。
  **⚠ 更正（2026-09-18）**：下面这句的因果是错的。另一侧会话查明并修好了真正的原因——
  **writer 一直同步跑在管线线程上**（每行 300 文件 + 300 次 fsync = 158 ms/行），
  每一行都把管线线程钉住，环因此被关、接收侧被拒槽、socket 溢出（他们的 `writer-thread`
  分支 `497ff7d`，只改 `writer.rs`；文档 `/scratch/jhgu/head-wt/docs/baseline_test_sync.md`）。
  我把异步腿搬到他们的分支上按同一协议对拍（3 臂 × 3 轮拉丁方 × 100 s，同一窗口）：
  每站等环 **old 32.2–34.8 s → new 0.0 / async 0.0**，补零率 **old 2.16/3.08/6.42%、
  new 0.0003/0.0000/1.1584%、async 0.0000/0.0012/0.0045%**（配对差 async−new =
  −0.0003/+0.0012/−1.15 个百分点），CPU 基本相同。**结论：带 writer 线程之后，这份档案上
  异步与同步等价**（它本来 1:1，异步既无收益也无代价）；我下面的旧数字是在未修代码上测的，
  已改为只作症状记录。详见 `docs/DESIGN-async-receive.md` §4.5/§4.7 与
  `/scratch/jhgu/async-side-comparison.md`（回给对方的比对报告）。
  ~~同一档案在 5 个窗口交错复测：异步腿每一窗口都更差（15–33% vs 9–18%）~~，
  且 `--workers 12`（容量正好压线）更差（42.7%）；线程腿今天也不是 0.1050%
  ——报告显示 H2D 每卡 4.55–4.7 GB/s（满需求 4.80）、pipeline 有 25% 时间在等设备释放槽、
  接收侧被拒槽最长 1 s，而**同档案 dry run 线程腿 0 丢包**，所以今天这笔账记在 GPU 链路上。
  **结论：这份档案该用线程后端**（24 路 / 24 收包核 = 1:1，正是 §4.3 说的"别用异步"那一格）。
  **这次验证抓出并修掉一个新 bug**：`try_on_packet` 原来先探环后声明计数跳变，
  导致"站等环、环等水位线"的自锁——修前 60 s 只出 0–1 行，修后 58 行（与线程腿一致）；
  详见 `docs/DESIGN-async-receive.md` §4.5.1。
* 结构：`run.rs` 639 行（编排）+ `receiver.rs` / `pipeline.rs` / `report.rs` / `stop.rs`。
  **三条不变量已结构化**：进度发布唯一入口、`block_slots - ring_lead ≥ 1` 由校验强制、
  停止原因唯一且首个胜出（见 `docs/ARCHITECTURE.md` §3）。

## 2. 下一步：别再把整槽的零搬上设备

**补零解耦的第 3、4 步已经做完（`b76339e`，ABI 6 → 7）**：掩码随样本一起上设备（每个 chunk
一次小 strided 拷贝）、`zero_invalid_frames_kernel` 在 int8→float 之前按掩码把无效帧整帧写零、
`corr_row_slot_free` 在交还槽时清空该槽掩码、组帧器停止补零且不再为补零等环（`finalize_range`
里已无 `wait_free`、无 memset，缺口一次 `advance_to` 越过）。四条判据的实测（逐行数字、
协议与 load 区间）在 `docs/DESIGN-zero-fill-decoupling.md` §8。**`--check` 现在报 abi=7。**

但它实测表明**真正吃掉管线的是"空槽照样要 H2D / FFT / 相关"**（§8.3、
§8.6）。现在掩码已经在设备侧、主机提交时也能读，于是"这一槽 24 个站一帧有效都没有"是
可判定的：跳过它的 H2D 和三个 kernel，结果不变（行最后统一除以 `n_cum`）。饱和工况下
每个站缺 ~310 个槽的帧、整轮只提交 693 个槽 —— 上限接近一半的设备工作是在算零。

顺序建议：

1. **整槽跳过**（收益最大，见 `docs/DESIGN-zero-fill-decoupling.md` §8.6）；
2. **流先停、`--duration` 后到会挂住**（新旧一样，既有问题，§8.7）；起跑前确认协议里回放
   比关联机活得久，或者把这条路径修掉；
3. `writer` 移出 pipeline 线程（短积分/高频吐数时才需要）。

**别做的事**：不要再往组帧器里加回补零。掩码是唯一判据，写零只会重新引入 §1 那条正反馈。
**判据的来源**：一律用逐行数字（`tools/rows_trend.py`、`tools/longrun_report.py`、
`tools/compare_blocks.py` 读的都是 `blocks.csv`），不要用运行报告里的每站汇总量 ——
一次 40 s 跑汇总 0.07%，而 38 行里只有 6 行是全零。load 是这些数字的自变量，同配置在
load 16–21 与 24–27 之间差一个量级，比较必须在同一窗口内交错（A/B/A/B）。
## 3. 已知的坑（照单别再踩）

* **判据用逐行数字**（`tools/rows_trend.py` / `tools/longrun_report.py`），不要用运行报告里的
  每站汇总量：一次 40 s 跑汇总 0.07%，实际 38 行里只有 6 行全零。
* **对比要在同一窗口内交错**（A/B/A/B 各 5–10 分钟）。这台共享机器的负载漂移是十分钟量级，
  同一配置两次跑实测差 3.2 倍；块式对比会被淹没。**记录 load**，它是这些数字的自变量。
* **负载是真实的自变量**：load 16–21 时补零 0.001–0.004%，load 24–27 时同配置 0.35–1.96%。
* `pkill -f corr-replay` **会杀掉执行它的 shell 自己** —— 用 `pkill -x`。
* **两个生产端会交错计数器**；起跑前用 `pgrep -x` 确认没有残留。
* 起跑前跑 `tools/doctor.py`：它会报 IRQ 亲和是否覆盖收包核、RX 环是否 1024/8192、
  `netdev_max_backlog`、`SO_BUSY_POLL` 权限、以及**是否有别的进程占着前端的组**
  （这台机器上常驻租户占着 15 个组，它出现在你每一次测量里）。
* **"没有已知告警"属于"已验证"**：带着已看到的 clippy 告警提交过两次，都被后续提交修掉。
* **C 侧测试要真跑**：`abi_smoke` 从 ABI 6（掩码区加进 staging）起就一直失败 —— 它还在按
  "每站区段 = 纯样本"算 `station_stride`，而没人重编重跑，`cargo test` 全绿掩盖了它。
  ABI 7 这轮把它修好并加上了掩码生命周期检查。改 ABI 就顺手跑一次 `ctest`。
* **`--stall` 不是丢包的站**：`corr-replay --stall i:s` 期间该站的 `n` 冻结，恢复后从原
  counter 继续 —— 它的包变成 late（被丢），而**其他 23 站在它身后被环挡住**（实测一站
  stall 3 s，其余每站丢 15.8% 的包）。要真的造"缺口"，用 `--gap-len/--gap-at`。
* **回放比 `--duration` 早结束会让关联机挂住**（停在 futex，无报错），新旧二进制都一样。
  让回放的时长盖过关联机的运行窗口。

## 4. 文档地图

| 文档 | 用途 |
|---|---|
| `docs/HANDOVER.md` | 本文：入口 |
| `docs/ARCHITECTURE.md` | 数据路径、模块地图、**五条不变量**、判读流程、未完成清单 |
| `docs/DESIGN-zero-fill-decoupling.md` | 补零解耦的设计 + **§7 分步计划与验收命令** |
| `docs/DESIGN-async-receive.md` | **异步收包后端**（流数 > 核数）：设计、正确性证据、交错 A/B 实测、测量陷阱、未解决的问题 |
| `config/README.md` | 7 个配置档案各自用途 + 三条"别再改回去"的约定 |
| `tools/README.md` | 判读/体检工具清单 + 三条使用纪律 |
| `docs/DESIGN-daily-files.md` | 按本地日期分文件集的行为、切套代价与两条被接受的风险 |
| `HANDOFF.md` | 历史日志（很长，过程与实测记录） |

## 5. 明确没做完 / 没验证的

* **整槽跳过**（§8.6）：这是补零解耦真正要兑现的收益，尚未实现。
* **判据 2 的后半、判据 4 的"更低"**：都没兑现（§8.3/§8.4），不是"没测"而是"测了没有"。
  要复现文档里 0.001–0.0376% 那种工况需要**真实前端**（195 kpps × 24），这台机器上做不到：
  回放要 24 个发送线程，和 24 个收包线程抢核，本身就把工况变成过载（load 25–33、
  15% 丢失）。饱和工况的对照结论只在那个工况里成立。
* **本机环境的四道限制**（`tools/doctor.py` 每次都报）：NIC 的 IRQ 与收包核重叠、
  RX 环只有 1024/8192、没有 `SO_BUSY_POLL` 权限、常驻租户占着 15 个组。都没有 root 可改。
* `RunStop::ZeroFillCap`（调度器侧上限）仍然没被实跑触发过 —— 收包线程自己的上限总是先到期。
* `writer` 仍在 pipeline 线程里。
* 40 路的三道墙：源只有 24 路、入口 200G < 259 Gbit/s、~~40 收包线程 > 32 核~~（**已做**：
  `corr-rs/src/async_recv/`（`corr-async`），见本文 §1 与 `docs/DESIGN-async-receive.md`；剩下的是真实前端下的对照没做成）。
