# DESIGN-no-recvmmsg.md —— `recvmmsg` 到底有没有必要

一句话结论：**在 24 路 / 4.66 Mpps / 200GbE 77% 的负载下，`recvmmsg` 对正确性不是必需的**
（换成单包 `recvfrom` 零丢包、行率不变），但它值 **约 1.1 个核**；而这 1.1 个核里
**只有 0.23 个是 syscall 本身，0.91 个是它顺带批处理掉的"调用方每轮记账"**。

这个分解是本文档的主要产出，因为它决定了重构该往哪里下手：把 `recvmmsg` 删掉只解决
1/5 的问题，真正可疑的是收包循环每轮那 7 次原子写。

---

## 1. 起因与要回答的问题

收包层用 `libc::recvmmsg`（`corr-rs/src/net.rs`），注释给的理由是"195 kpps 下每包一次
syscall 会在进出内核上吃掉可观的一个核的份额"。这个论证是**为 1500 字节帧写的**：现在
前端发的是 4104 字节的 jumbo frame，同一个 syscall 摊到的字节数是当时的 2.7 倍。

问题因此拆成两个，答案不一样，必须分开测：

1. **批量化 syscall 是不是必需的？** 单包 `recvfrom` 能不能扛住，代价多少？
2. 如果代价不小，**那是 syscall 的代价，还是批量化顺手摊薄掉的别的东西？**

---

## 2. 环境与被测负载（全部实测，不是配置抄写）

| 项 | 值 | 来源 |
|---|---|---|
| 主机 | g05，32 核 / 2 NUMA，4× L20 | `nproc` / 配置注释 |
| 网卡 | ens2f0np0，200GbE | `ip addr` |
| 前端 | **24 路全部在发**，每路 195,312.5 包/s | `tools/mcast_rate.py --include-disabled` |
| 链路实测 | **4,655,370 pps、19.30 GB/s、帧长 4145 B** | join 24 组期间取 NIC 计数器差分 |
| 链路占空 | 19.30 GB/s ≈ 200GbE 的 **77%** | 同上 |
| 配置 | `config/correlator-24-nf16384-g05.yaml`，`receiver: threads`（24 个收包线程） | 运行日志 `receiver : threads (24 station(s)…)` |
| 运行模式 | `--dry-run`：收包、拼帧、走水位线；**不碰 GPU、不落盘** | `corr-rs/src/bin/corr.rs` `Mode::DryRun` |
| 构建 | cargo / rustc 1.98.1，`--release`，默认 feature | `--version` |

> 注意一个坑：**没 join 组播组时 NIC 的计数器不成立**（多播过滤会先把包丢掉）。
> 不 join 直接量 `rx_packets`，得到的是 2.93 Mpps，比真实负载少 37%，
> 会让人以为链路有余量。必须先让 socket join 上去再量。

`--dry-run` 是刻意的：要问的是收包 syscall 的代价，把管线、四张卡和 `/scratch` 拉进来
只会引入三个与问题无关的方差源。

---

## 3. 三个臂

| 臂 | 收包系统调用 | 每站 syscall/s | 每站迭代/s | 每次拿到的包数 |
|---|---|---|---|---|
| `base` | `recvmmsg`，batch 32，**不带 `MSG_WAITFORONE`** | ≈ 6,135 | ≈ 6,135 | **31.9999** |
| `looped` | 循环 `recvfrom`，一次调用最多填 32 个 | ≈ 203,000 | ≈ 6,816 | 28.81 |
| `single` | `recvfrom`，一次调用 1 个 | ≈ 196,300 | ≈ 196,313 | 1 |

`looped` 这个臂是为了把上面两个问题分开才存在的：它交给调用方的**批次形状和 `base` 一样**
（所以调用方的每轮记账仍然是批处理的），但**每包一次 syscall，和 `single` 一样**。于是：

- `base → looped` = **纯粹是 syscall 次数**（1 : 32），迭代次数几乎不变（6,135 : 6,816）
- `looped → single` = **纯粹是批次形状**（迭代次数 1 : 29），syscall 次数几乎不变

分支：`no_recvmmsg`（`single`，提交 `f2b927c`）、`no_recvmmsg-looped`（`looped`）。
`base` 用 `fdd8e15` 的发布二进制。

调用次数从哪来：`report.rs` 里本来就有 `recv_batches` / `recv_packets` 两个字段，只是从没
打印过。为了取"平均批次大小"这个数，在一个一次性构建里把它们打了出来
（`bench/instrument-recv-calls.patch`，未提交）。

---

## 4. 结果

### 4.1 正确性：三个臂完全一样

| 指标 | base | looped | single | 判据 |
|---|---|---|---|---|
| 稳态 socket 丢弃/s | 0 | 0 | 0 | 6/6 轮 × 3 臂全为 0 |
| 60 s 内丢包 | 0–1 | 0–4 | 0–3 | 每臂总收包 282.7 M，差异是噪声量级 |
| 行率 | 0.982–1.000 行/s | 同 | 同 | 配对差值为 0 或 ±0.018 |
| ≥20 ms 读停顿 | 0 | 0 | 0 | — |
| NIC `rx_out_of_buffer` 增量 | 0 | 0 | 0 | 三臂共享的地板 |

**结论 1：24 路满负载下，`recvmmsg` 不是正确性所必需的。**

### 4.2 代价：约 1.1 个核，且可加分解

6 轮 × 60 s，循环轮转顺序，`tools/ab_report.py` 读配对差值。
核数取监视器每秒行的均值（`cpu X/32c`）。

| 对比 | Δ 核数 | Δ CPU/60 s | Δ user | Δ sys | 符号一致性 |
|---|---|---|---|---|---|
| `base → looped` | **+0.23** | +13.5 s | +2.8 s | **+10.8 s** | 6/6 |
| `looped → single` | **+0.91** | +54.6 s | **+57.8 s** | −2.2 s | 6/6 |
| `base → single` | **+1.12** | +67.1 s | +60.7 s | +7.6 s | 6/6 |

绝对核数：`base` 13.09、`looped` 13.29、`single` 14.14（共 32 核）。

0.23 + 0.91 = 1.14 ≈ 1.12 —— **两个效应基本可加**，而且**记账分离得很干净**：

- **syscall 的代价记在 sys time**（+10.8 s / 60 s）
- **调用方每轮记账的代价记在 user time**（+57.8 s / 60 s）

### 4.3 为什么 `base` 能把记账也摊薄：`recvmmsg` 的真实语义

这一条是本次最重要的发现，而且它推翻了原来的注释。

`tools/recvmmsg-semantics.c`（3 个数据报已入队，`vlen=8`，`SO_RCVTIMEO` 300 ms）：

```
plain recvmmsg : n=3 in   306.9 ms   (3 个数据报已入队, vlen=8)
MSG_WAITFORONE : n=3 in     0.0 ms
```

**不带 `MSG_WAITFORONE` 的 `recvmmsg` 会一直阻塞到凑满整批**，只有超时能提前结束它。
`man 2 recvmmsg` 的措辞也印证了这一点：`MSG_WAITFORONE` 的作用是"收到第一条消息**之后**
打开 `MSG_DONTWAIT`"——反过来说，不带它时后续消息仍然阻塞。

在真实运行里表现为三个数字：

- `base` 每次调用恰好 **31.9999** 个包（`single` 是 1，`looped` 是 28.81）
- 60 s、24 站合计只有 **1 次空轮询**——socket 从来没被排空过
- 每次调用内耗时 **154 µs**，恰好等于 32 个包在 195 kpps 下的到达时间（164 µs）

所以 `base` 的收包线程不是"一次 syscall 把队列排空"，而是**每 32 个包醒一次、睡约 164 µs**。
它省下的 syscall 是次要的，**真正省下的是调用方每轮那点固定工作——因为那些工作顺带被
32 个包摊薄了**。这是那个 `recvmmsg` 论证里没写出来的收益。

### 4.4 那 0.91 个核具体是什么

`receiver.rs` 的收包循环里，**每个迭代**（不管拿到几个包）都要做：

```
a.prog.requested()                      原子读 (Acquire)  advance_to
a.prog.mark_packet()                    时钟读 + 原子写 (Release)  last_packet_ns
a.prog.add_packets(n)                   原子 fetch_add (RMW)  packets
a.prog.set(asm.progress())              原子写 (Release)  progress
a.prog.set_zero_filled_frames(…)        原子写 (Release)  zero_filled_frames
a.prog.set_lost_packets(…)              原子写 (Release)  lost_packets
a.prog.set_slot_wait_ns(…)              原子写 (Release)  slot_wait_ns
2 × Instant::now()                      时钟读（ns_recv / ns_assemble 计时）
```

约 **7 次原子操作（其中一次是 locked RMW）+ 3 次时钟读**。而 `StationProgress` 的 7 个
`AtomicU64` **没有任何 cache line padding**：在 `Arc<StationProgress>` 的 `ArcInner` 里它们
占偏移 16..72，也就是前 6 个挤在同一条 64 B cache line 上，而 `progress` 正被管线线程的
水位线循环持续读——这是教科书式的 false sharing。

数量核对：多出的迭代数是 272.9 M / 60 s，多花 57.8 s user CPU，即 **≈ 212 ns/迭代**。
对这个工作量来说是很合理的数。

---

## 5. 结论与建议

1. **`recvmmsg` 在正确性上不必要**，在本机当前负载下换掉它不会丢数据。
2. **但它不是免费的**：约 1.1 个核 = 32 核机器的 3.5%，或进程自身 CPU 的 +7.5%
   （13.09 → 14.14 核）。
3. **代价的 4/5 不在 syscall**，而在收包循环每轮的固定记账上——`recvmmsg` 的
   "阻塞到凑满整批"行为**顺带**把这个开销摊薄了 32 倍。
4. 因此，如果目标是"精减结构、消除不必要的优化"，正确的做法**不是**只把
   `recvmmsg` 换成 `recvfrom`，而是：
   - 换掉 `recvmmsg`（它带来的语义是"等满一批才返回"，这本身是个隐式延迟，不是谁都想要）；
   - **同时把每轮记账降下来**：`StationProgress` 加 cache line padding、把
     `zero_filled_frames` / `lost_packets` / `slot_wait_ns` 这类只在拼帧状态真的变了才需要
     更新的量按需发布、把 `Instant::now()` 的计时移出热路径。
   两件事一起做，1.1 个核里的大部分就回来了，而且结构比现在简单。
5. 顺带：`RecvStats` 里的 `batches` 和 `packets` 两个字段在 `net.rs` 之外无人读取，是死字段；
   `unsafe impl Send for MulticastSocket` 在去掉 iovec/mmsghdr 后也不再需要（自动成立）。

### 5.1 分支现状

- `no_recvmmsg`（`f2b927c`）：只换收包系统调用，**其余结构未动**，所以它是干净的 A/B 变体，
  但**不是**最终该合入的形态。
- `receiver.recvmmsg_batch` 在 `no_recvmmsg` 上被保留了：所有 `config/*.yaml` 都带这个键，
  而 `ReceiverCfg` 是 `deny_unknown_fields`，删字段会让所有配置解析失败。等到 §5 的第 4
  条一起做时，字段和 YAML 一起删。
- `no_recvmmsg-looped`：**只用于本次分解实验**，不该合入（它把批处理形状和 32 次 syscall
  拼在一起，两头都不占优）。

---

## 6. 没测的（下一个该测的）

| 项 | 为什么重要 |
|---|---|
| **async 后端** | **最该接着测的一腿。** 生产默认是 `corr-async --workers 24`，它的 `recv_batch_nowait` 每个 reactor 就绪事件只取一批；改成每次 1 个包后，reactor 轮转次数会从"每 32 包一次"变成"每包一次"，代价很可能比同步后端更贵——可能不止 1 个核。 |
| 40 路目标 | 本机前端只有 24 路在发，40 路（设计目标）没有可用的真实负载。1.1 个核大致按站数线性放大，但没测。 |
| 生产模式（带 GPU + 落盘） | dry_run 隔离了 syscall 问题，但生产里收包线程还会被拒槽、socket 缓冲会成为最后一道余量。本机 dry_run 下"等 slot"是 0.0 s/站，三臂都一样，所以这个问题在本次实验里没被触及。 |
| 包更小的情况 | 本次全是 4104 B 的 jumbo frame。若将来帧长变小，每包 syscall 的相对代价会上升。 |

---

## 7. 复现

```bash
# 1. 确认前端在发、且 24 组都活着（必须先 join，否则 NIC 计数器不成立）
tools/mcast_rate.py config/correlator-24-nf16384-g05.yaml --seconds 4 --include-disabled

# 2. 基线二进制（fdd8e15）。变体二进制在各自分支上 cargo build --release 后拷进 bench/
mkdir -p bench/baseline-bin
cp -p corr-rs/target/release/corr bench/baseline-bin/

# 3. 三臂轮转 A/B（每轮 60 s，顺序循环轮转；rounds 取 3 的倍数才位置平衡）
tools/recvmmsg_ab.sh 6 60

# 4. 判读（看配对差值的符号，不要看单次绝对值）
tools/ab_report.py bench/recvmmsg-ab/<时间戳>

# 5. 单独证明 recvmmsg 的阻塞语义
gcc -O2 -o /tmp/ms tools/recvmmsg-semantics.c && /tmp/ms

# 6. 需要看每线程 user/sys 与 socket 积压时
tools/recv_probe.sh bench/baseline-bin/corr
```

原始日志在 `bench/recvmmsg-ab/` 与 `bench/no-recvmmsg/`（`bench/` 被 .gitignore 忽略——
按仓库既有约定，那里只放产物，不放脚本）。
