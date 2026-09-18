# baseline_test_sync.md —— 同步版（writer 独立线程）16384/1s baseline 测试记录

写给**异步收包那一侧的会话**：这是同步版本在 `nch=16384 / 1 s 积分 / 24 路真实数据` 下的
完整测试过程、原始结果、复现命令与干扰项清单，用来做交叉比对。

- 测试者：本仓 `writer-thread` 分支（工作树 `/scratch/jhgu/head-wt`）
- 时间：2026-09-17 20:39 – 22:31，2026-09-18 13:03 – 13:47（本地时间，机器重启前后各一段）
- 被测代码：`4410e09`（基线，writer 在管线线程上）+ 本项目新增的 writer 独立线程
- 结论一句话：**安静窗口下 16384/1s 已经可用（逐行补零 0.0001%–0.0155%）；坏窗口下任何版本都会到
  20–25%，那不是 writer 的问题，也还没有查清。**

> 重要：本文件只描述**同步版**。异步版请用同一套口径跑一遍再比；第 9 节是我希望你们回的字段。
> 设计动机与同样的数字另见 `docs/DESIGN-writer-thread.md`（在第 1 节的分支上）。

---

## 1. 被测对象

| 项 | 值 |
|---|---|
| 基线提交 | `4410e09`（= 当前 master） |
| 本次改动 | 分支 `writer-thread`：`cf4d153` + `752688e` |
| 工作树 | `/scratch/jhgu/head-wt`（`git worktree add … HEAD` 出来的，与主树共用 object store） |
| 基线二进制 | `/scratch/jhgu/base-wt/corr-rs/target/release/corr`（另一个 worktree，detached at `4410e09`） |
| 新二进制 | `/scratch/jhgu/head-wt/corr-rs/target/release/corr`（`752688e`） |
| 改动范围 | **只有 `corr-rs/src/writer.rs`**（+ 文档 + `tools/wt_compare.py`）；`run.rs` / `pipeline.rs` / `report.rs` / `config/` 一行未改，ABI 未变 |
| 配置 | `config/correlator-24-nf16384.yaml` 原样：24 站 → 300 baseline，`n_freq=16384`，row 24576 帧（1.007 s），staging 2112 MiB，输出 39.062 MB/s，**`fsync_every_rows: 1`（没有改，理由见 §4.3）** |

`A/B` 用的配置是这份文件的副本，只差 `fsync_every_rows` 与注释（`config_hash` 是对
serde 序列化后的字段取 FNV，注释不参与）：`/scratch/jhgu/cfg-ab-fsync1.yaml` 与原配置**值完全相同**，
`/scratch/jhgu/cfg-ab-fsync10.yaml` 只有那一行是 10。

改动做了什么（一句话）：`write_row` 不再自己写盘/fsync，而是把这一行拷进池化缓冲投进有界队列，
由 `corr-writer` 线程持有文件并负责写、fsync、每日切套；队列深度 =
`clamp(storage.queue_bytes / row_bytes, 2, 8)`（16384 档 = 8 行 ≈ 314 MB）。签名未变，新增
`Writer::sync()`（给"不停止线程但要看已交付的行"用，只有测试/监控需要）。

---

## 2. 测量口径（必须按这个口径比，否则数字不可比）

### 2.1 指标定义

| 指标 | 来源 | 定义 |
|---|---|---|
| **补零率（逐行）** | `<setdir>/blocks.csv`（第 5 列 valid、第 6 列 zero_filled） | `Σ zero_filled / Σ(valid+zero_filled)`，跨 24 站与所有行求和。**不要用运行报告里的 per-station 汇总**，它会把"整行整行被关掉"和"每行涓流丢 0.1%"混成一个数 |
| 完全干净的行 | 同上 | 该行 24 站 `zero_filled` 全为 0 的行数 / 总行数 |
| **每站等环** | 运行日志 `receive threads: … assembling them (of which Y s waiting for a staging slot)` | 收包线程在"往 staging 环写"这一步被环满挡住的累计秒数（每站平均）。**0 = 环从未满过**；只有它保证"没有丢帧的结构性原因" |
| release-wait | 运行日志 `pipeline thread: … release-wait X s` | 管线线程等 GPU 交还 staging 槽的累计秒数 |
| 管线线程库内时间 | 同上 | `queue + release-wait + rows-wait + fetch`；**不含**写盘（writer 的耗时不在这个数里，见 §4.5） |
| 写线程自报 | 运行日志 `writer: thread wrote N rows, X ms/row of its own time` | 写线程每行真正忙的时间；行周期 1007 ms，接近它就是写线程成为瓶颈 |
| 队列最深 | 运行日志 `writer: N rows handed over, deepest queue M of K rows` | 交付出去但还没写完的行数峰值 |
| 机器状态 | `/scratch/jhgu/zf-sampler.py` 每 5 s 一行 | `load=`、`group_users=`、`groups_with_2plus=`、四卡 `gpu=`、`kworkers=` |

工具：`tools/longrun_report.py <setdir…>`（补零率/干净行/三等分）、`tools/rows_trend.py <setdir>`
（逐行明细，`--worst`）、`tools/wt_compare.py <rundir> [arms]`（本分支新增：把上面所有指标按轮次
并排列出，直接读产物与日志，不读运行报告的自我评价）。

### 2.2 为什么必须"轮内轮转"（这一条是本次最重要的方法学结论）

这台机器每 3–7 分钟在"环不关"与"环关掉 33–43 s/100 s"两个状态之间跳，两者补零率相差 20 多个
百分点。因此：

- **两次相隔 8 分钟的 300 s 对比是无效的**（我早期就是这么做错了一次，见 §4.3）；
- 正确做法：**把要比的臂放进同一个窗口**，每个臂跑 100–120 s，顺序每轮轮转，跑 4–6 轮。这样
  每个臂在每个位置上出现次数相同，位置偏差（实测第一位偏好：位均 6.3% / 14.1% / 16.0%）被抵消。
- 判断只看**配对差值的符号是否一致**，不要看单次的绝对值。

---

## 3. 测试过程（5 个批次）

| 批次 | 时间（本地） | 每次时长 | 臂 | 目录 |
|---|---|---|---|---|
| A | 09-17 20:39–21:02 | 300 s ×2、150 s ×3 | 旧二进制，fsync 1 / 10 / 100，另有 nf4096 | `/scratch/jhgu/corr-zf{,2,3,4,5}` |
| B | 09-17 21:25–21:41 | 300 s ×4 | 旧/新 × fsync 1/10 | `/scratch/jhgu/wt-ab` |
| C | 09-17 21:46–22:31 | 100 s ×18 | 旧fsync1 / 新fsync1 / 新fsync10 轮转 6 轮，再旧fsync1/旧fsync10 对拍 4 轮 | `/scratch/jhgu/wt-ab2` |
| D | 09-18 13:03–13:09 | 300 s ×1 | 新，fsync=1 | `/scratch/jhgu/wt-now` |
| E | 09-18 13:38–13:47（**重启后**） | 120 s ×4 | 旧/新，fsync=1，顺序对调（旧→新，新→旧） | `/scratch/jhgu/wt-rb` |

每次运行的命令行（批次 C/E 用的就是这个）：

```bash
cd /scratch/jhgu/head-wt
python3 /scratch/jhgu/zf-sampler.py /scratch/jhgu/wt-rb/<tag>-samples.txt 135 &
./corr-rs/target/release/corr --config /scratch/jhgu/cfg-ab-fsync1.yaml \
    --output-dir /scratch/jhgu/wt-rb --run-name <tag> --duration 120 2>&1 | tee /scratch/jhgu/wt-rb/<tag>-run.log
```

前置检查每次先做：`pgrep -x corr`（必须为空，**不要**用 `pkill -f`）、`python3 tools/doctor.py`、
`nvidia-smi --query-compute-apps`（确认没有别的 CUDA 进程）。

---

## 4. 结果

### 4.1 300 s 背靠背，fsync=1，同一窗口（21:25 vs 21:30，load 27.3 → 27.3）

| | 旧（`4410e09`） | 新（writer 线程） |
|---|---|---|
| 补零率（逐行） | **29.0060%**（297 行） | **0.0155%**（297 行） |
| 完全干净的行 | 1 / 297 | 213 / 297 |
| 管线线程库内时间 | 98.6 s | 25.7 s |
| ├ release-wait | 87.6 s | 13.3 s |
| └ 每站等环 | 122.5 s | 0.1 s |
| 写线程自报 | （在管线线程上） | 128 ms/行；队列最深 1/8 |

同批另两次：旧 21:35 = **20.4369%**（每站等环 122.4 s）；新 fsync=10 21:41 = **23.8623%**
（每站等环 103.4 s）——**同样是新代码，21:30 是 0.0155%、21:41 是 23.86%**，这一对就是"窗口
支配绝对值"的最好证据。

### 4.2 100 s × 6 轮 × 3 臂，轮内轮转（拉丁方，批次 C）

每轮顺序轮转（1: 前,后1,后10 → 2: 后1,后10,前 → 3: 后10,前,后1 → 4 同 1 → 5 同 2 → 6 同 3），
6 轮后每个臂在每个位置各出现两次。

| 轮 | 旧 fsync=1 | 新 fsync=1 | 新 fsync=10 |
|---|---|---|---|
| 1 | 28.0899% | 15.3172% | 0.0000% |
| 2 | 20.4843% | 2.2165% | 8.9726% |
| 3 | 10.0307% | 0.0106% | 0.1345% |
| 4 | 7.0582% | 0.1111% | 24.6741% |
| 5 | 28.9530% | 0.0000% | 22.2753% |
| 6 | 27.5689% | 21.9084% | 0.0135% |
| **均值 / 中位数** | **20.36% / 24.03%** | **6.59% / 1.16%** | **9.35% / 4.55%** |

**旧 vs 新（同为 fsync=1）：6/6 轮都是旧的更差**，逐轮差值 +12.8 / +18.3 / +10.0 / +6.9 /
+29.0 / +5.7 个百分点（中位数 +11.4）。

收包侧（每站等环，100 s 内，单位 s）：旧 39.7 / 40.8 / 34.8 / 43.4 / 41.0 / 39.7（**从不低于
34.8**）；新 23.0 / 33.7 / 0.0 / 0.1 / 0.1 / 33.0。管线线程 release-wait：旧 29.2 / 22.2 /
26.1 / 11.4 / 29.2 / 29.2；新 20.1 / 7.8 / 4.5 / 4.3 / 2.5 / 27.8。写线程自报：新 fsync=1 为
133–140 ms/行，fsync=10 为 58–69 ms/行（行周期 1007 ms，队列最深 1 行，5 个批次里从未出现 >1）。

### 4.3 fsync 这个旋钮：同窗口对拍（批次 C 后半，**旧二进制**）

| 轮 | fsync=1 | fsync=10 |
|---|---|---|
| 7 | 10.4026% | 9.7610% |
| 8 | 12.6802% | 15.7060% |
| 9（坏窗口） | 29.8336% | 28.6014% |
| 10 | 9.2375% | 7.9979% |

**没有差别**。原因：0.47 ms 是**按次**计费的（同样 39 MB 写成 1 个文件只要 16.8 ms/次），
`fsync_every_rows: 10` 只是把 300 次摊成 30 次、变成每 10 行一次 ~1.4 s 的突发，平均每行仍 ~14 ms。

> ⚠️ 这条推翻了本会话早期的一个结论：当时相隔 8 分钟测到"fsync=1 → 28–29% / fsync=10 → 7.26%"，
> 那是窗口漂移。现在**原配置保持 `fsync_every_rows: 1`（崩溃最多丢 0 行），没有改配置**。
> 如果异步版也基于那条旧结论调过这个旋钮，建议按本节重新评估。

### 4.4 重启后 120 s × 2 对（批次 E，顺序对调）

| 对（顺序） | 旧 fsync=1 | 新 fsync=1 |
|---|---|---|
| 1（旧 → 新） | 0.0084%（118 行，95 行干净；每站等环 6.0 s） | **0.0005%**（119 行，117 行干净；每站等环 0.0 s） |
| 2（新 → 旧） | 0.0349%（118 行，109 行干净；每站等环 7.0 s） | **0.0001%**（118 行，117 行干净；每站等环 0.0 s） |

重启后机器极静（起步 load 1.25、无其它 CUDA 进程、运行中 13–17，而 09-17 晚上是 26–27）。
管线线程库内时间：旧 18.7 / 19.8 s、release-wait 14.5 / 15.0 s；新 8.1 / 7.0 s、release-wait
3.6 / 2.6 s。写线程自报 97 / 104 ms/行。

**读法**：安静时两个版本都"够好"，但旧的**结构性代价仍然在**——120 s 里每站必须等环 6–7 s
（**从不为 0**），新的始终 0.0 s。也就是说旧版本只是"机器有余量时没把它变成丢数据"。

### 4.5 结构性指标（不依赖窗口，可用它判断"谁在挡路"）

| | 旧 | 新 |
|---|---|---|
| 写盘发生在线程 | 管线线程（每行 300 文件 + 300 次 fsync ≈ 142 ms/行） | `corr-writer`（实测 97–140 ms/行，行周期 1007 ms → **8 倍余量**） |
| 队列最深 | 无 | 1 / 8（314 MB 上限） |
| 管线线程库内时间 | 98.6 s / 300 s（33%） | 25.7 s / 300 s（9%） |

判定"写盘是不是瓶颈"从此可以只看日志：写线程每行耗时 vs 行周期，以及 `deepest queue`。
**注意**：批次 B（`/scratch/jhgu/wt-ab/*-run.log`）里的 `deepest queue 297 of 8` 是**统计 bug**
（按缓冲计数、而缓冲在两处取用、只在一处归还），`752688e` 修好了；这些 run 的补零率不受影响，
但引用队列数字时请只引用批次 C/E 之后的。

### 4.6 其它档位与更早的记录（**不可作为比对基线**）

| 配置 | 二进制 | 时长/行数 | 补零率 | 每站等环 |
|---|---|---|---|---|
| nf16384 fsync=100 | 旧 | 150 s / 148 | 8.0706% | 47.7 s |
| nf16384 fsync=10 | 旧 | 150 s / 148 | 8.7607% | 63.6 s |
| **nf4096** fsync=1（`config/correlator-24.yaml`） | 旧 | 150 s / 148 | 25.1254% | 67.0 s |
| nf16384 fsync=1，09-17 下午安静窗口 | 旧 | ~300 s | 记录到 0.0105% 与 0.105% / 0.131% 两档 | — |

最后一行说明这台机器在安静时**旧代码也能到 10⁻⁴ 量级**；但那两批产物在 `/scratch` 清理时已删除，
无法复算，**所以不要拿它当基线**，只用来说明"绝对值由窗口决定"这一点。

---

## 5. 结论与置信度（请按这个强度引用）

**可以下的结论（强）**

1. 写盘（含 fsync）已不在管线线程上；写线程有 8 倍余量，队列常态为空。这是结构性的，与窗口无关。
2. 同一个窗口内，新版本 6/6 轮优于旧版本（差值中位数 +11.4 个百分点），收包侧"每站等环"由
   "从不低于 35 s/100 s"变为"可以是 0"。
3. `fsync_every_rows` 1 vs 10 在同一窗口内对补零率**无效**（4/4 轮无差别），配置保持 1。
4. 安静窗口下 16384/1s 逐行补零 0.0001%–0.0155%（117/119 到 219/297 行完全干净），
   即"可用"。

**不能下的结论（弱/未知）**

- 不能说"新版本在任何机器状态下都 ≤0.01%"：坏窗口里新版本也到 **21.9% / 24.7%**（第 4.2 表
  第 4、6 轮），此时每站等环 33–35 s/100 s。**环被关闭还有第二个原因，与 writer 无关。**
- 坏窗口的成因未查清（见 §6）。
- 跨会话、跨机器、跨天的绝对值不可直接比；只能比同一窗口内的配对差。

---

## 6. 干扰项与环境（比对时必须一起报告）

1. **机器重启**：09-18 13:25 前后重启（13:38 时 uptime 13 分钟）。重启前 load 26–27（含另一个
   会话的测试），重启后 1.2。重启前后分别是批次 C 与批次 E。
2. **机器时钟跳变**：09-17 22:31 → 09-18 13:03（本地）之间，机器时钟向前跳了约 14.5 小时，而
   我的实际耗时只有约半小时。副作用：那之后的运行按"每日文件"规则开了新的一天
   （`20260918_130346`），日志里会有 `local day changed … closed set … opened …`。
   另外 `/scratch` 的 NFS mtime 一直比 wall clock 慢 13–14 分钟 → **永远不要把文件 mtime 和
   `time.time()` 混用**（早期一次"13 分钟停摆"就是这么误判出来的）。
3. **15 个前端组被"非本次运行"的 socket 加入**（`tools/doctor.py` 一直在报，重启后依然）：
   内核为每个包多做一次拷贝。这是长期存在的邻居，不是我或你们的进程。
4. **另一个会话（你们）的测试目录**：`/scratch/nf16384-verify`、`nf16384-pool`、`nf16384-quiet`、
   `regress-threads`、`phase-async`、`async-smoke`（含 `threads` / `async12` / `async24` 臂），
   时间集中在 09-17 18:50–20:47。**这批与我在 20:39–21:02 看到的坏窗口重叠**；21:41–22:31 的
   坏窗口没有对应目录，解释不了全部。请你们核对自己的时序：
   如果你们在我这些坏窗口里也在跑，请在比对报告里标出来。
5. **不是 GPU 降频**：`/scratch/jhgu/wt-ab2/throttle.txt` 显示 4 卡时钟 2520/2520 MHz、
   `throttle_reasons=0x0`、~115 W、43–45 °C；`procs.txt` 显示运行期间 `--query-compute-apps`
   只有本次 corr 自己。
6. `tools/doctor.py` 重启后仍是同样的 5 条待修（网卡 IRQ 与收包核 [0,1,2,3] 重叠、RX ring
   1024/8192、`netdev_max_backlog=5000`、`SO_BUSY_POLL` EPERM、收包核在网卡自己的 NUMA 节点）。
   这些在所有批次里都是常量，不解释批次间差异。

---

## 7. 产物清单

**还在**（每个 run 目录都保留 `blocks.csv`、`rows_ms.bin`、`metadata.yaml`、`*-run.log`、
`*-samples.txt`）：

- `/scratch/jhgu/wt-rb/`（批次 E，120 s ×4 + `driver.log` + 采样）
- `/scratch/jhgu/wt-now/`（批次 D，300 s）
- `/scratch/jhgu/wt-ab2/`（批次 C，18 个 100 s run + `driver.log`/`driver2.log` + 采样 +
  `context.txt`/`procs.txt`/`throttle.txt`）
- `/scratch/jhgu/wt-ab/`（批次 B，4 个 300 s run）
- `/scratch/jhgu/corr-zf{,_2,_3,_4,_5}/`（批次 A）
- `/scratch/jhgu/daily-rollover-wt-223210/`（切套测试，切套对管线线程代价 3 ms）
- 辅助：`/scratch/jhgu/cfg-ab-fsync1.yaml`、`cfg-ab-fsync10.yaml`、`zf-sampler.py`、
  `wt-ab-driver.sh`、`wt-paired-driver.sh`、`wt-paired2-driver.sh`、`wt-postreboot.sh`

**已删除**：所有 `*baseline_*.bin`（波形本体，约 180 GB，为给 `/scratch` 腾地方）。
→ **逐行补零统计、行时刻、日志全部可复算；但无法再对帧级数据重做 auto/cross 谱。**

**未保存**：§4.6 最后一行那两批下午的数据（产物已随 `/scratch` 清理消失）。

---

## 8. 复现（最短路径）

```bash
# 1) 拿代码：分支 writer-thread（工作树已在 /scratch/jhgu/head-wt）
cd /scratch/jhgu/head-wt && git log --oneline -2      # 752688e / cf4d153

# 2) 前置检查（必须为空/无别的 CUDA 进程）
pgrep -x corr; python3 tools/doctor.py; nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 3) 一轮配对（120 s × 2 对，顺序对调；tags 用 b-r<N>/n1-r<N> 便于下面的工具直接读）
OUT=/scratch/jhgu/<your-out>; mkdir -p $OUT
for tag in b-r1 n1-r1 n1-r2 b-r2; do
  case $tag in
    b-*) BIN=/scratch/jhgu/base-wt/corr-rs/target/release/corr;;
    n1-*) BIN=/scratch/jhgu/head-wt/corr-rs/target/release/corr;;
  esac
  python3 /scratch/jhgu/zf-sampler.py $OUT/$tag-samples.txt 135 &
  $BIN --config /scratch/jhgu/cfg-ab-fsync1.yaml --output-dir $OUT --run-name $tag --duration 120 \
       2>&1 | tee $OUT/$tag-run.log | grep -E "^  output|pipeline thread|staging slot|writer:"
  kill %1 2>/dev/null
done

# 4) 看结果（都是读产物，不读运行报告的自我评价）
python3 tools/longrun_report.py $OUT/*/<tag>      # 逐行补零率、干净行数
python3 tools/rows_trend.py   $OUT/b-r1/b-r1      # 逐行明细、最差行
python3 tools/wt_compare.py   $OUT b,n1           # 各臂并排：补零率/管线账/等环/写线程
```

`tools/wt_compare.py` 目前只在我的分支上（合并后即进主树）；它要求目录名形如
`<臂>-r<轮次>` 且运行日志与目录同级，否则对应的列会显示 `--`。

---

## 9. 请异步版本回什么（比对清单）

请用**同一口径**做一遍，并在报告里给出：

1. **代码身份**：commit / 工作树 / 二进制路径 / 与 `4410e09` 的差异文件清单；配置是否改过
   （`fsync_every_rows` 保持 1？`storage.*` 其它项有没有动？）。
2. **同窗口配对**：至少 2 对（4 次）100–120 s 的短跑，顺序对调；若可能，做 4–6 轮轮转。
   报告每轮的**逐行补零率、完全干净行数**，以及配对数与方向是否一致。
3. **结构性指标**：管线线程 `queue / release-wait / rows-wait / fetch`；**你的等价于"每站等环"
   的指标**（异步收包改了这一步，请说明现在是在哪里等：staging 环？描述符池？socket？），
   以及写线程自报（若已合并 `writer-thread`）。
4. **机器状态**：每次运行的起止时刻、`*-samples.txt` 里的 load/GPU 区间、doctor 结论、
   有无其它 CUDA 进程；以及**你们自己那批测试的运行时段**（见 §6.4）。
5. **坏窗口单独列出**：任何补零率 ≥5% 的 run，请把完整运行日志（含每站 pkts/gaps/lost/
   zero_filled 与 per-second 段）留档并给出路径，我想和我的时间线对齐。
6. **不要**用 per-station 汇总下结论（会把逐行涓流藏起来），也**不要**用相隔几分钟的两次运行
   比较不同版本。

我这边会回：同一窗口下的配对表、坏窗口时间线、以及（如果你们愿意）把 `writer-thread`
合进你们的树后在你们那套异步收包上的同一协议结果。

---

## 10. 未解决 / 待办

1. **坏窗口**（每站等环 33–43 s/100 s、补零 20–25%）成因未查清；已知与 fsync 无关、与写线程
   无关、与 GPU 降频无关；与另一个会话的测试只对上一部分时间。这是 16384/1s 目前唯一的
   可用性风险。
2. **涓流丢帧没有任何告警**：运行中只有"某站 200 ms 完全无包"才算 stall（上限 5000 万帧），
   慢性丢 20% 会一路跑完并报成功。要不要加一个"逐行补零比例 + 每站等环"的看门狗（超阈值停或
   告警），待定。
3. **零余量是配置属性**：16384 档 H2D 需求 19.1 GB/s ≈ 实测上限 19.2 GB/s（4 卡 ×4.80）。
   任何扰动都会变成丢帧；这是选档位时要一起考虑的，不是代码缺陷。
4. `storage.writers` 仍是未使用字段（写作线程只有 1 个，理由见 `docs/DESIGN-writer-thread.md` §3）。
