# 状态接口：`GET /status.json`

一份只读 JSON，由 corr 进程的一个独立线程 serve；页面、历史、曲线在独立的监测
程序里（`tools/corr-web.py`）。默认**关**，在配置里显式打开：

```yaml
monitoring:
  interval_ms: 1000          # 已有的实时统计行（stderr）
  http:
    enabled: true
    bind: "127.0.0.1:3081"   # 只监听回环；3080 已被本机 DSH web GUI 占用
```

## 1. 为什么可以放进生产进程里

把网络服务放进一个实时接收进程里，本能上是该犹豫的。这里之所以可以，是因为三条
边界每条都可以被指出来：

**它不在数据通路上。** 采样器（`corr-rs/src/monitor.rs`）本来每秒就要算这些量
（实时统计行），现在只是把算完的那一份再往一个槽里放一次：

| 层 | 模块 | 知道什么 | 不知道什么 |
|---|---|---|---|
| 采样 | `monitor.rs` | 哪些计数器存在、怎么读、怎么差分、谁是瓶颈 | 不知道 HTTP、JSON、端口 |
| 传输 | `status.rs` | 一个监听 socket、一个线程、HTTP 与 JSON | 不知道 sched/gpu/ring 的存在 |
| 数据 | `monitor::Snapshot` | 纯字段，`serde::Serialize`，没有方法 | —— |
| 接线 | `run.rs` | 一处 `Cell::new()` + 一处 `status::serve()` | —— |

接收、拼帧、调度、写入四条路径里，没有任何一行引用 `status`；`pipeline.rs` 唯一的
改动是调用 `monitor.tick(..., set_tag, ...)`，而它本来就在调用。

**它的成本是一次上锁。** `Cell` 是一个 `Mutex<Option<Arc<Snapshot>>>`：pipeline
线程每秒把 `Arc` 换掉一次（纳秒级、无竞争的锁），HTTP 线程克隆 `Arc` 之后才做
序列化，所以 JSON 渲染永远不会挡在采样器前面。读者看到的永远是完整的一份快照，
不会读到写了一半的东西。

**它挂掉不会带走运行。** release 档是 `panic = "abort"`，这个线程里一次 panic 就会
abort 整个进程 —— 所以 `status.rs` 里没有任何可达的 `unwrap`/`expect`/切片索引，
每条错误路径都是"丢掉这一个连接"。启动时 `bind` 失败（端口被占、没权限）只打印一行
告警然后继续跑：**收数是目的，监测不是**。反过来，配置里 `bind` 写错（不是 `IP:port`）
是启动失败 —— 配置错误该早失败，环境问题不该。

编译期也可以整体去掉：`status-http` feature（默认开）关掉后，`status.rs`、HTTP 与
`serde_json` 都不进构建，与 `async-recv` 关掉 tokio 是同一套做法。

## 2. 接口契约

| 方法 | 路径 | 回答 |
|---|---|---|
| GET | `/status.json` | 当前快照；`ready: false` 表示还没有第一份样本 |
| GET | `/` | 端点清单（仍是 JSON，这个接口不 serve HTML） |
| 其它 | 任何 | `404` + JSON |

* 只有 GET；没有请求体、没有查询参数（查询串被忽略，方便加 cache buster）；
* 无 keep-alive（每个请求一个连接，`Connection: close`）；
* 无认证、无 TLS —— 默认只绑回环，要暴露到内网请自己权衡（`bind: 0.0.0.0:3081`）；
* `schema` 是接口版本：**加字段向后兼容，改字段含义必须加版本号**。读者应当拒绝
  自己不认识的 `schema`。

### 字段

顶层：`schema`、`ready`、`ts_unix_ms`、`pid`、`started_unix_ms`、`interval_ms`、
`run`、`cumulative`、`window`、`recv`、`gpu`、`host`、`capacity`、`stations`。

| 分组 | 字段 | 含义 |
|---|---|---|
| `run` | `name`、`output_dir`、`set_tag` | 哪个 run；`set_tag` = `<file_prefix><stamp>`，跨本地日翻转时会变 |
| | `n_station`、`n_baseline`、`n_freq`、`n_cum`、`integration_ms` | 几何与积分长度 |
| | `frames_per_slot`、`block_slots` | 槽大小与环的槽数（领先量的分母） |
| `cumulative` | `rows`、`frames`、`zero_filled_frames`、`zero_fill_pct` | 自起跑累计 |
| | `lost_packets`、`stall_recoveries`、`socket_drops`、`ring_block_events` | 三类丢数 + 水位线脱困次数 |
| `window` | `dt_s`、`rows`、`rows_per_s` | 上一个采样区间；`dt_s == 0` 表示"还没有区间"（第一份样本） |
| | `zero_fill_pct`、`lost_packets`、`socket_drops`、… | 同一批量的**增量** |
| `recv` | `wait_pct` | 站·秒里"等环"占的比例；等待不是丢数，但没有余量了 |
| | `spread_slots`、`blocked_now` | 站间进度差（槽）、此刻被环挡住的站数 |
| `gpu` | `h2d_gbs`、`required_gbs`、`link_pct`、`slot_free_pct` | 链路实际/需求、pipeline 等槽释放的比例 |
| `host` | `cpu_cores`、`cores_online`、`load1` | 本进程核数 / 在线核数 / 整机 load |
| `capacity` | `bottleneck` | `ok` / `socket_drops` / `ring` / `gpu_link` / `machine_load` / `process_cpu` |
| | `detail`、`limits` | 与 stderr 那行**同一份**结论与门限（同一函数产出，不会两处打架） |
| `stations[]` | `name`、`progress_frames`、`zero_filled_frames`、`lost_packets`、`slot_wait_ms`、`silence_s` | 每站明细：哪一路在丢、在等、在静默 |

`capacity.bottleneck` 的优先级是刻意的：**已经丢数的最优先**（socket 溢出 → 环 →
GPU 链路 → 整机 load → 本进程 CPU → 有余量）。

### 读数时的三条纪律

1. **看 `window`，不只看 `cumulative`。** 累计补零率会把"第 30 秒的一阵丢包"稀释掉，
   窗口值才是"现在怎么样"。
2. **判损失按三处一起看**：`lost_packets`（前端计数器的缺口，包在上游就没了）、
   `socket_drops`（内核在我们自己的 socket 上丢的）、网卡计数（`ethtool -S`）。
   只报一处就会得出错误结论 —— 详见 `docs/HANDOVER.md` 里的两个测量陷阱。
3. **`wait_pct` 高不等于丢数**，但等于"没有抖动余量"：它一高，下一次抢占就直接丢包。

## 3. 配套的监测界面

`tools/corr-web.py`：只用 Python 标准库，每秒取一次 `/status.json`，**自己攒历史**
（接口只给当前快照），页面在 `http://127.0.0.1:8080/`。

```bash
tools/corr-web.py                      # 默认取 127.0.0.1:3081
tools/corr-web.py --interval 0.5 --history 7200
```

corr 停了之后界面不会崩：它显示"连接断开 N 秒"并继续显示最后一份快照，所以跑完
一腿仍然能回看。历史只在这个程序的内存里，重启它就没了 —— 要留档请连 corr 的
stderr 日志（`run-24-nf16384.sh` 会把整段 tee 到 `<output_dir>/corr/<mode>-<stamp>.log`）
一起留。

## 4. 没有做，以及为什么

* **不 serve HTML**：接口只给数据，页面是看的人的事（换掉界面不用动 corr）。
* **不给历史**：历史需要进程内多一份环形缓冲；监测程序自己攒同样容易，还不需要
  在接收进程里多一份常驻内存。
* **不做认证**：默认只绑回环。要跨机看就先解决"谁可以看"这个问题，而不是把一个
  token 塞进这个接口。
* **不写 `stats.csv`**：`monitoring.stats_csv` 仍然是"接受并忽略"的字段（见
  `config/README.md`）；需要落盘的曲线交给监测程序。
