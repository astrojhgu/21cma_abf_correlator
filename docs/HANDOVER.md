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
cargo test                               # 81 + 7
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

* 最后提交 `b76339e`：**补零解耦的第 3、4 步已落地**（ABI 7：掩码随样本上设备、kernel 按
  掩码整帧清零、组帧器不再补零也不再为补零等环）。`cargo test` **81 + 7 通过**、
  `clippy` **0 条代码告警**、C 侧 `ctest` **6/6**（abi_smoke / multi_device / 两个 numpy golden）。
  验收实测见 `docs/DESIGN-zero-fill-decoupling.md` §8：判据 1、3 通过（含逐字节对照），
  **判据 2 的后半与判据 4 的预期没有兑现**，原因在 §8.3/§8.4，下一步在 §8.6。
* 改动前的工作树停在 `bd33583`，保留成一个 git worktree：`/scratch/corr-baseline`
  （A/B 对照腿用的旧二进制就从那里编）。
* 实跑证据：**24 路满速率、4 卡 `time_slice`（R=1）**，每卡 H2D 4.80 GB/s（满需求）；
  5 分钟跑 **299/300 行准点**，补零率 **0.0013% – 0.0376%**（随负载 16 → 24 变化，见 §3）；
  相位–频率检验 **276/276** 基线精确复现注入延迟（`tools/phase_triangle.py`）。
* 结构：`run.rs` 639 行（编排）+ `receiver.rs` / `pipeline.rs` / `report.rs` / `stop.rs`。
  **三条不变量已结构化**：进度发布唯一入口、`block_slots - ring_lead ≥ 1` 由校验强制、
  停止原因唯一且首个胜出（见 `docs/ARCHITECTURE.md` §3）。

## 2. 下一步：别再把整槽的零搬上设备

补零解耦已经做完，但实测表明**真正吃掉管线的是"空槽照样要 H2D / FFT / 相关"**（§8.3、
§8.6）。现在掩码已经在设备侧、主机提交时也能读，于是"这一槽 24 个站一帧有效都没有"是
可判定的：跳过它的 H2D 和三个 kernel，结果不变（行最后统一除以 `n_cum`）。饱和工况下
每个站缺 ~310 个槽的帧、整轮只提交 693 个槽 —— 上限接近一半的设备工作是在算零。

顺序建议：

1. **整槽跳过**（收益最大，见 `docs/DESIGN-zero-fill-decoupling.md` §8.6）；
2. **流先停、`--duration` 后到会挂住**（新旧一样，既有问题，§8.7）；起跑前确认协议里回放
   比关联机活得久，或者把这条路径修掉；
3. `writer` 移出 pipeline 线程（短积分/高频吐数时才需要）。

**别做的事**：不要再往组帧器里加回补零。掩码是唯一判据，写零只会重新引入 §1 那条正反馈。
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
| `config/README.md` | 7 个配置档案各自用途 + 三条"别再改回去"的约定 |
| `tools/README.md` | 判读/体检工具清单 + 三条使用纪律 |
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
* 40 路的三道墙：源只有 24 路、入口 200G < 259 Gbit/s、40 收包线程 > 32 核（需 epoll 多 socket）。
