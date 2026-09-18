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
4. **改完先跑 `tools/doctor.py --config <cfg>`**：它检查 IRQ 撞核、RX 环大小、backlog、
   `SO_BUSY_POLL` 权限、以及是否有别的进程占着前端的组。
