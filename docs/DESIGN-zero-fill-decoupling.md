# 设计：把补零从收包线程移交给 GPU

> **术语**：本文说的"组帧器"是 `corr-rs/src/assemble.rs`（`StationAssembler`）——
> 把 UDP 包按计数器拼成 FFT 帧、管理 staging 环、并在缺包时补零的那个 Rust 模块。
> 它是 "frame assembler"，与**汇编语言无关**（本项目没有手写汇编：只有 Rust、
> CUDA C++ 和一小段 C）。

状态：**第 1、2 步已完成**（ABI 6 的掩码区；组帧器维护掩码并有等价性测试，见 §7）；
**第 3、4 步未实现** —— 也就是「零由 GPU 补、组帧器不再补零」这个行为切换尚未发生。
本文是待执行的设计。写它的原因是这一项
需要动 ABI、kernel、组帧器三处，必须一次做完并整体验证，不适合夹在别的改动里。

## 1. 要解决的问题

某站丢包时，当前由**它自己的收包线程**把缺失的帧写成零（`assemble.rs::zero_frame`）。
这带来两个后果，第二个是长跑丢包的放大机制：

1. **补零与读包争同一个线程**：落后 → 要补零 → 更少时间读 socket → 更落后。
   nch=8192 时一帧 16 KiB，24 路实测补零带来的 memset 写流量可达 ~5 GB/s。
2. **补零被 staging 环的节奏卡住**：`finalize_range` 每推进一个槽都要
   `wait_free`（等管线释放），所以一次 150 ms 的缺口要花 ~150 ms 才能补完
   （`HANDOFF` §6.13.2 的实测：缺口 49 槽 ≈ 137 ms，期间其它 23 站全被环挡住）。

结论：**只要"零"必须由收包线程写进 staging，落后就一定会自我放大。**

## 2. 提议

让组帧器**只记录**哪些帧缺失（有效性位图，它本来就有 `arrival` 位图），零由 GPU 侧补：

- 组帧器不再写零、也不再为补零等环；`progress` 立刻越过缺口（它的语义本来就该是
  "不会再有数据到来"，而不是"数据已就位"，调度器用它的方式不变）。
- 缺失信息随槽一起到达设备，kernel 在装载该站该帧时把它当零处理。

### 2.1 掩码放在哪里（关键设计选择）

**放在 staging 里**，而不是新加一条 H2D 通道：每槽每站只需要
`frames_per_slot` 个 bit，24 站 × 256 帧 = 768 B/槽，与样本同槽同行，天然随
`corr_row_slot_free` 的生命周期走，不需要第二套同步。

- 布局：在每站每槽的区域里划出一小块掩码区（或在整块 staging 之后单独一段），
  偏移与大小写进 ABI（`corr_config` 增两个字段，`abi_version` 提到 6）。
- 组帧器写掩码 = 一次位操作（它已经在 `arrival` 里维护了同一信息）。
- 管线不需要改动：它已经把整槽 H2D 过去。

### 2.2 kernel 侧

`kernels.cu` 的 FFT 装载阶段读掩码：该帧无效则整帧写零（等价于现在的行为，
但由 GPU 做，且与读包并行）。核里已经按 `station × frame` 切块，
把它变成一个分支即可。相关核（correlation）不需要改：它只读频谱。

## 3. 触点清单（照此顺序做）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `corr-cuda/include/corr_cuda.h` | `corr_config` 增 `validity_offset` / `validity_bytes`；`CORR_ABI_VERSION` → 6 |
| 2 | `corr-rs/src/ffi.rs` | 镜像两个新字段 + 编译期断言（已有惯例） |
| 3 | `corr-cuda/src/corr_cuda.cpp` | staging 布局里预留掩码区；创建时清零 |
| 4 | `corr-rs/src/assemble.rs` | `zero_frame`/`finalize_range` 不再写零，改为写掩码；`stats.zero_filled_frames` 的语义变成"标记为缺失的帧" |
| 5 | `corr-cuda/src/kernels.cu` | 装载时按掩码把无效帧清零 |
| 6 | `corr-rs/src/report.rs` / `writer.rs` | `blocks.csv` 的 `zero_filled` 列改从掩码统计（数值应当与现在一致） |
| 7 | `corr-rs/src/config.rs` | 新增开关（默认新行为）；旧行为保留一个跑次以便对照 |

## 4. 验收判据（缺一不可）

1. **科学输出不变**：同一份回放数据、同一处 stall，`blocks.csv` 的
   `valid_frames` / `zero_filled` 逐行**与旧行为一致**；C 侧 golden 测试逐字节通过。
2. **正反馈消失**：缺口出现后，该站的 `progress` 在下一次调度检查时就越过缺口，
   而不是每槽一步地爬；报告里"等 staging 槽"的时间应显著下降。
3. **相位测试不受影响**：`tools/phase_triangle.py` 的 276/276 仍然成立。
4. **压力对照**：`corr-replay --stall` + 上限放开，在 load ≥ 20 的窗口里，
   新行为的补零率不高于旧行为（期望更低），并记录两者负载。

## 5. 风险与已知坑

- **`progress` 语义**：调度器的水位线用它判断"这一槽可以提交了"。改成"不再有数据到来"
  之后必须确认 `sched.rs` 的所有用法都仍然成立（它只做 `>=` 比较，理论上成立，
  但这是本设计里唯一可能引入死锁的地方，必须专门测）。
- **零补上限**：现在的上限由组帧器自己计数并抛 `ZeroFillCapExceeded`
  （见 `pipeline.rs` 里那段注释）。掩码方案下计数仍然由组帧器做（它知道缺了多少帧），
  所以上限语义可以保留；不要顺手把它删掉。
- **`frames_per_slot` 变化**：掩码大小必须随它缩放，别忘了配置校验。
- **回退**：第 7 项那个开关就是回退路径；实现期间保持旧路径可用，直到判据 1 通过。

## 6. 这项工作之后还剩什么

- `writer` 移出 pipeline 线程（短积分/高频吐数时才需要）
- `2·nch` 换算在工具里只留一处（已在三处独立出现过）
- 40 路的三道墙：源只有 24 路、入口 200G < 259 Gbit/s、40 收包线程 > 32 核（需 epoll 多 socket）

## 7. 分步执行（每步都可单独提交、单独验证）

不要一次改完再验：第 4 步（组帧器不再写零）与第 5 步（kernel 读掩码）必须**同一步**落地，
否则中间状态会产出全零数据 ✗。其余步骤都可以独立提交。

| 步 | 内容 | 验证命令与判据 |
|---|---|---|
| **0** | 记录基线（今天就能做）| 用第 4 节的 stall 场景跑一次，保存 `blocks.csv`、`rows_trend.py` 的三段趋势、以及报告里"等 staging 槽"的时间。**这就是修复要改善的对照** |
| 1 | ABI 加 `validity_offset` / `validity_bytes_per_slot` / `validity_total_bytes`；staging 分配加上掩码区并 `memset 0xFF`（全有效）；Rust 侧镜像 + 编译期断言 | `corr --check` 通过、`abi_smoke` 通过、`--print-plan` 里 staging 大小含掩码区 |
| 2 | 组帧器**在继续补零的同时**写掩码；加一个测试断言"掩码标记的缺失帧 == 实际被补零的帧" | `cargo test`；行为未变（golden 测试逐字节一致）|
| 3 | kernel 读掩码把无效帧清零；组帧器**停止**补零 | 第 4 节判据 1（`blocks.csv` 逐行一致）+ 判据 3（相位 276/276）|
| 4 | 去掉第 7 项的开关（或反过来，用它做回退）| 判据 2、4（正反馈消失、压力对照）|

**第 0 步的 stall 场景**（注意 `--stall-at` 必须让该站先被预排空看到，否则只会触发预排空检查）：

```bash
pkill -x corr-replay                       # 不是 -f：-f 会杀掉执行它的 shell
python3 -c "import re;s=open('config/correlator-24-replay.yaml').read();\
open('/tmp/cfg-stall.yaml','w').write(re.sub(r'zero_fill_limit_frames: \d+','zero_fill_limit_frames: 50000000',s))"
nohup ./corr-rs/target/release/corr-replay --config /tmp/cfg-stall.yaml --pattern prng \
      --rate 50000 --stall 3:30 --stall-at 8 --duration 120 > /tmp/replay-stall.log 2>&1 &
sleep 6
./corr-rs/target/release/corr --config /tmp/cfg-stall.yaml --duration 60 --output-dir /scratch/feedback-before
python3 tools/rows_trend.py  /scratch/feedback-before/*/ --step 20
grep -E "receive threads|per second" /tmp/feedback-before-run.log
```

判据（修复后应当同时成立）：补零率不高于基线、`blocks.csv` 逐行一致、
**缺口的 `progress` 越过不再逐槽爬行**、报告里"等 staging 槽"的时间显著下降。
