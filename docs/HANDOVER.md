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
cd corr-rs && cargo build --release      # 顺带构建 ../corr-cuda 的 .so
./target/release/corr --config ../config/correlator-24.yaml --check   # ABI/布局/编号一致性
cargo test                               # 80 + 7
tools/doctor.py --config ../config/correlator-24.yaml                 # 机器侧体检（先跑这个）
```

## 1. 当前状态（交接点）

* 最后提交 `ce15866`；工作树干净；`cargo test` **80 + 7 通过**；`clippy` **0 条代码告警**。
* 实跑证据：**24 路满速率、4 卡 `time_slice`（R=1）**，每卡 H2D 4.80 GB/s（满需求）；
  5 分钟跑 **299/300 行准点**，补零率 **0.0013% – 0.0376%**（随负载 16 → 24 变化，见 §3）；
  相位–频率检验 **276/276** 基线精确复现注入延迟（`tools/phase_triangle.py`）。
* 结构：`run.rs` 639 行（编排）+ `receiver.rs` / `pipeline.rs` / `report.rs` / `stop.rs`。
  **三条不变量已结构化**：进度发布唯一入口、`block_slots - ring_lead ≥ 1` 由校验强制、
  停止原因唯一且首个胜出（见 `docs/ARCHITECTURE.md` §3）。

## 2. 下一步就一件事：把补零从收包线程移交给 GPU

按 `docs/DESIGN-zero-fill-decoupling.md` **§7 的第 3 + 4 步**（必须同时落地）：

1. **掩码上设备**：今天的 slot 是用 `kernels.cu:605` 的 `cudaMemcpy2DAsync` 送过去的，
   **只含样本** ⇒ 掩码根本没到设备上。需要额外一次小 H2D（每槽每站 32 B，24 站共 768 B/槽）。
2. **kernel 清零**：装阶段按掩码把无效帧整帧清零。
3. **组帧器停止补零**（删 `zero_frame` 的调用路径）。
4. 之后才谈去掉回退开关（设计文档第 7 项）。

**为什么必须一起落**：若组帧器先停止补零而 kernel 还没读掩码，产出的是**整片全零的相关结果**，
不报错、数据看着正常 —— 对一个相关机这是最糟的失败模式。

**验收判据**（设计文档 §4）：`blocks.csv` 的 `valid_frame`/`zero_filled` 逐行与今天一致；
相位检验仍 276/276；缺口出现后 `progress` 立即越过而不是逐槽爬行；报告里"等 staging 槽"的时间下降。

第 1、2 步已完成，可依赖：ABI 6 的掩码区（`--check` 校验）与
`the_mask_marks_exactly_the_frames_that_were_not_filled`
—— 它断言**掩码标记的缺失帧 == 实际被补零的帧**，正是第 3 步唯一依赖的不变量。

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

## 4. 文档地图

| 文档 | 用途 |
|---|---|
| `docs/HANDOVER.md` | 本文：入口 |
| `docs/ARCHITECTURE.md` | 数据路径、模块地图、**五条不变量**、判读流程、未完成清单 |
| `docs/DESIGN-zero-fill-decoupling.md` | 补零解耦的设计 + **§7 分步计划与验收命令** |
| `config/README.md` | 7 个配置档案各自用途 + 三条"别再改回去"的约定 |
| `tools/README.md` | 判读/体检工具清单 + 三条使用纪律 |
| `HANDOFF.md` | 历史日志（很长，过程与实测记录） |

## 5. 明确没做完 / 没验证的

* **补零解耦的第 3、4 步**（核心行为切换）。
* `RunStop::ZeroFillCap`（调度器侧上限）是唯一**没被实跑触发过**的停止原因
  ——收包线程自己的上限总是先到期；要触发需要"线程卡死"的场景。
* 掩码改动的影响**没有做同窗口交错对照**（掩码开/关各 3 分钟），严格说
  "与负载一致"不等于"已排除掩码改动"。
* 40 路的三道墙：源只有 24 路、入口 200G < 259 Gbit/s、40 收包线程 > 32 核（需 epoll 多 socket）。
* `writer` 仍在 pipeline 线程里（短积分/高频吐数时需要独立线程）。
