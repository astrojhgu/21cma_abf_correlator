# 配置档案怎么选

| 文件 | 用途 | 站数 | nch | 备注 |
|---|---|---|---|---|
| `correlator.yaml` | 日常开发（真实前端）| 5 启用 | 4096 | 最省资源，改一行 `enabled` 即可放大 |
| `correlator-24.yaml` | **24 路标准实跑（推荐）** | 24 | 4096 | tslice + 4 卡，`auto_layout` 绑核 |
| `correlator-24-nf8192.yaml` | 24 路 × 8192 信道 / 1 s | 24 | 8192 | staging 2.3 GiB；输出 19.6 MB/s |
| `correlator-20-nf8192.yaml` | 20 路 × 8192 信道 / 1 s | 20 | 8192 | 这台机器的收包能力边界之内（0 丢包）|
| `correlator-24-replay.yaml` | 回放测试（相位/故障注入）| 24 | 8192 | 组走 239.1.200.x，可与观测同时跑 |
| `correlator-40-replay.yaml` | 40 路功能测试（回放）| 40 | 4096 | 需要 `auto_layout` 共享核警告；40 线程 > 32 核 |
| `correlator-test.yaml` / `correlator-mgpu-test.yaml` | 早期小规模测试 | 2 / 8 | 4096 | 保留作最小复现 |

三条约定（都是踩出来的，别再改回去）：

1. **绑核用策略，不手写核号**：`cpu.avoid_nic_interfaces: [ens2f0np0]` + `cpu.auto_layout: true`。
   手写列表在换机器时会**静默**出错（症状只是长跑里几个百分点的补零）。
2. **`recv_buffer_bytes` 与 `gpu.ring_lead` 必须配对**：socket 能积压的量 ≤ 环允许领先的量，
   否则积压只能涨不能消（见 `docs/ARCHITECTURE.md` §3.2）。
3. **改完先跑 `tools/doctor.py --config <cfg>`**：它检查 IRQ 撞核、RX 环大小、backlog、
   `SO_BUSY_POLL` 权限、以及是否有别的进程占着前端的组。
