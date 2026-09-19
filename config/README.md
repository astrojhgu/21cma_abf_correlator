# 配置档案怎么选

| 文件 | 用途 | 站数 | nch | 备注 |
|---|---|---|---|---|
| `correlator.yaml` | 日常开发（真实前端）| 5 启用 | 4096 | 最省资源，改一行 `enabled` 即可放大 |
| `correlator-24.yaml` | **24 路标准实跑（推荐）** | 24 | 4096 | tslice + 4 卡，`auto_layout` 绑核 |
| `correlator-24-nf8192.yaml` | 24 路 × 8192 信道 / 1 s | 24 | 8192 | staging 2.3 GiB；输出 19.6 MB/s |
| `correlator-24-nf16384.yaml` | 24 路 × 16384 信道 / 1 s | 24 | 16384 | staging 2.0 GiB；输出 39 MB/s；**收包缓冲必须 256 MiB**（见下）|
| `correlator-24-nf16384-manual.yaml` | **手动跑用**：同上，但输出固定在 `/scratch/jhgu/corr/`（`run.name: corr` + `output_dir: /scratch/jhgu`，不再套一层运行目录）| 24 | 16384 | 配套脚本 `tools/run-24-nf16384.sh`；同一天同配置的第二次运行会续写当日那一套，另起一套用 `--allow-second-set` |
| `correlator-20-nf8192.yaml` | 20 路 × 8192 信道 / 1 s | 20 | 8192 | 这台机器的收包能力边界之内（0 丢包）|
| `correlator-24-replay.yaml` | 回放测试（相位/故障注入）| 24 | 8192 | 组走 239.1.200.x，可与观测同时跑 |
| `correlator-40-replay.yaml` | 40 路功能测试（回放）| 40 | 4096 | 需要 `auto_layout` 共享核警告；40 线程 > 32 核 |
| `correlator-test.yaml` / `correlator-mgpu-test.yaml` | 早期小规模测试 | 2 / 8 | 4096 | 保留作最小复现 |

五条约定（都是踩出来的，别再改回去）：

1. **绑核用策略，不手写核号**：`cpu.avoid_nic_interfaces: [ens2f0np0]` + `cpu.auto_layout: true`。
   手写列表在换机器时会**静默**出错（症状只是长跑里几个百分点的补零）。
2. **`recv_buffer_bytes` 与 `gpu.ring_lead` 必须配对**：socket 能积压的量 ≤ 环允许领先的量，
   否则积压只能涨不能消（见 `docs/ARCHITECTURE.md` §3.2）。
3. **输出按本地日期分文件集，默认对所有档案生效**（`storage.daily_files: true`，前缀
   `storage.file_prefix_format: "%Y%m%d-%H%M%S"` + `storage.file_prefix: ""`）：跨本地零点
   自动换一套 `<前缀><站j><站k>-<时刻>.bin` / `<前缀><时刻>-time.{bin,txt}` /
   `-blocks.csv` / `-metadata.yaml`，名字里的时刻取该套第一条行的本地时刻。要续写当日那套，两次进程必须给同一个 `run.name`（默认按时间戳命名，
   每次都是新目录）。设计、代价与两条被接受的风险见 `docs/DESIGN-daily-files.md`；
   端到端判据 `tools/daily_rollover_test.sh`。要退回老的无前缀布局：
   `storage.daily_files: false`。
4. **`recv_buffer_bytes` 要按档案定，不能跨档案抄**。nch=16384 档最初照抄了 nf8192 档的
   32 MiB（实得 64 MiB ≈ 84 ms），同一台机器、同样 300 s：每站丢 25.1 万个包（0.43%）、
   补零 0.4269%；改成 256 MiB（实得 512 MiB ≈ 640 ms）后丢 10.9 万个（0.077%）、补零
   **0.1050%**。4 倍的差别全在收包侧。反过来推不成立：把 nch=16384 的槽改小
   （frames_per_slot 256→128、block_slots 11→21，领先时间不变）反而差到 3.07%，
   所以"槽越大越需要大缓冲"不是这里的机制，**要重新定这条就重新测**。
5. **`monitoring.interval_ms` 现在会打实时状态行**（默认 1000 ms，一行到 stderr）：
   行数/行率、逐行补零率（累计与窗口）、计数缺口、**接收侧等环占比**、站间进度散布、
   本进程核数与机器 load、H2D 相对需求速率的百分比、pipeline 等设备归还槽的占比、
   我们 socket 的内核丢包增量，最后一段 `capacity:` 指出**当前先顶到上限的是谁**
   （socket 丢包 / 环 / GPU 链路 / 整机 / 本进程 CPU）。设 `0` 关闭。同段的
   `stats_csv` 仍是**已接受但未实现**的字段（配置里写了它不会报错，但也不会写文件）。
6. **`monitoring.http` 是只读状态接口**（默认 `enabled: false`，要显式打开）：
   `GET /status.json` 返回与上面那行**同一份**样本的 JSON，由 corr 的一个独立线程
   serve；页面在独立运行的 `tools/corr-web.py` 里。它不在数据通路上（bind 失败只告警、
   不停跑），编译期可以整体去掉（`--no-default-features` 或只关 `status-http`）。
   字段表、契约与"为什么可以放进生产进程"见 `docs/DESIGN-status-http.md`。
   **`monitoring` 段不参与 `config_hash`**：它是观测，改了它不会改变已有行的含义，
   所以打开接口、改打印间隔都不会让当天的套变成"配置变了、拒绝续写"。
   （其余字段照旧参与 —— 那条保护问的是"行的含义变没变"。）

4. **改完先跑 `tools/doctor.py --config <cfg>`**：它检查 IRQ 撞核、RX 环大小、backlog、
   `SO_BUSY_POLL` 权限、以及是否有别的进程占着前端的组。

7. **`storage.queue_full` 决定写盘队列满了怎么办**（默认 `drop`）：丢**整行**并继续收数，
   而不是等 writer。这不是细节：`block`（改前的行为）会让一次写盘停顿顺着
   writer→pipeline→ring→worker 传下去，最后没人读 socket。实测 2026-09-19 一次 **52 秒**
   的停顿变成 **2.4 亿包**丢弃，且那一腿再没恢复。丢行是安全的：队列的单位是一整行，
   丢弃发生在入队之前，所以一行要么进全部基线文件、要么一个都不进；行号来自帧计数器，
   所以 `blocks.csv`/`time.bin` 会在那里干净地跳跃。丢弃会记日志（区间）+ 运行时报告。
