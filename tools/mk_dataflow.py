#!/usr/bin/env python3
"""Draw the correlator's data flow as SVG.

Every number in the picture is one that was measured or that comes straight out
of the shipped configuration; the measurement window and the exact figures are in
HANDOFF.md 6.5 / 6.10 / 6.11 / 6.12. The point of the picture is to show *where*
the bytes go and *who waits for whom*, because that is what the multi-card design
turns on.

usage: mk_dataflow.py > dataflow.svg
"""

W, H = 1600, 1310

# palette
C_NET = ("#e8f0fe", "#1967d2")
C_RX = ("#e6f4ea", "#137333")
C_STAGE = ("#fef7e0", "#b06000")
C_GPU = ("#f3e8fd", "#7b1fa2")
C_SCHED = ("#e8eaed", "#3c4043")
C_OUT = ("#fce8e6", "#c5221f")
C_CTRL = "#d93025"
INK = "#202124"
MUTED = "#5f6368"

out = []
add = out.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, title, lines, colours, rx=10, tsize=15, lsize=12.5, dash=None):
    fill, stroke = colours
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.6"{d}/>')
    ty = y + 21
    if title:
        add(f'<text x="{x+12}" y="{ty}" font-size="{tsize}" font-weight="600" '
            f'fill="{stroke}">{esc(title)}</text>')
        ty += 19
    for ln in lines:
        add(f'<text x="{x+12}" y="{ty}" font-size="{lsize}" fill="{INK}">{esc(ln)}</text>')
        ty += 16.5
    return ty


def arrow(x1, y1, x2, y2, colour=INK, dash=None, width=1.6, marker="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
        f'stroke-width="{width}"{d} marker-end="url(#{marker})"/>')


def label(x, y, text, size=12, colour=MUTED, weight=None, anchor="start"):
    w = f' font-weight="{weight}"' if weight else ""
    add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{colour}"{w} '
        f'text-anchor="{anchor}">{esc(text)}</text>')


def band(y, n, name, h):
    add(f'<rect x="14" y="{y}" width="30" height="{h}" rx="8" fill="#f1f3f4" '
        f'stroke="#dadce0"/>')
    add(f'<text x="29" y="{y+h/2}" font-size="14" font-weight="700" fill="{MUTED}" '
        f'text-anchor="middle" transform="rotate(-90 29 {y+h/2})">{esc(n)} {esc(name)}</text>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="Noto Sans CJK SC, Noto Sans CJK JP, sans-serif">')
add('<defs>'
    '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#202124"/></marker>'
    '<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#d93025"/></marker>'
    '<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7b1fa2"/></marker>'
    '</defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
add(f'<text x="60" y="30" font-size="21" font-weight="700" fill="{INK}">'
    f'相关机数据流（当前实现）</text>')
label(430, 30, "1 s 积分 · 24 路 · 每站 195,312.5 包/s · 4104 B/包(8 B counter + 4096 int8) "
               "· 帧 = 8192 样本 @ 97,656.25 Hz · 行 = 97,792 帧 = 1.0014 s", 12.5)

# ---------------------------------------------------------------- ① 前端 / 网卡
band(48, "①", "入口", 104)
box(60, 48, 500, 104, "RFSoC 前端（3 块板 × 24 站）",
    ["每站一个多播组 239.1.13x/16x:100xx，counter 是三块板共享的全局包计数",
     "每包 4104 B：8 B little-endian counter + 4096 个 int8 样本",
     "合计 4.69 Mpps / 155 Gbit/s（24 路）；40 路将是 7.81 Mpps / 259 Gbit/s"],
    C_NET)
box(580, 48, 420, 104, "ens2f0np0（200 Gb/s，NUMA node 0）",
    ["每组一个 socket，SO_RCVBUF 实测授予 512 MiB/组",
     "recvmmsg 批量 32；SO_BUSY_POLL 返回 EPERM（无 root）",
     "40 路时入口需 ≥400G 或 2×200G bond（这块网卡是 24 路的量级）"],
    C_NET)
box(1020, 48, 540, 104, "入口实测（本轮）",
    ["dry-run（只收包，无 GPU）：196 kpps/站 · 4.69 Mpps 总 · 丢包 0 · 0.5 s 内无缺口",
     "⇒ 收包路径本身不丢包，也没有被其他用户的负载拖垮",
     "produce 模式下收到的速率上限 = 管线的消费速率（见 ④）"],
    ("#f8f9fa", MUTED))
arrow(560, 100, 578, 100)
arrow(1000, 100, 1018, 100)

# ---------------------------------------------------------------- ② 收包线程
band(172, "②", "拼帧", 158)
box(60, 172, 1500, 158, "收包线程 ×24（每站 1 个，绑核 0..23）",
    [], C_RX)
steps = [
    ("recvmmsg 批量取包", "32 个/次，非阻塞轮询 + 5 ms 超时"),
    ("解析 counter", "frame_id = counter / P（P = n_freq/2048 = 2）"),
    ("定位 staging", "slot_seq = frame_id/256；slot = slot_seq % 3；帧内偏移 = frame_id % 256"),
    ("写整帧", "2 个包拼成 8192 B，落进该站自己的 staging 区域；缺包的帧整帧补零"),
    ("发布进度", "progress = 第一个未 final 的帧；last_packet_ns = 现在"),
]
x = 76
for i, (t, d) in enumerate(steps):
    w = 258 if i != 3 else 300
    add(f'<rect x="{x}" y="{200}" width="{w}" height="66" rx="8" fill="#ffffff" '
        f'stroke="{C_RX[1]}" stroke-width="1.3"/>')
    add(f'<text x="{x+10}" y="{221}" font-size="13.5" font-weight="600" fill="{C_RX[1]}">'
        f'{esc(t)}</text>')
    # wrap d
    lines = []
    line = ""
    for tok in d.split(" "):
        cand = tok if not line else line + " " + tok
        if len(cand) > 30 and line:
            lines.append(line)
            line = tok
        else:
            line = cand
    if line:
        lines.append(line)
    for j, ln in enumerate(lines[:3]):
        add(f'<text x="{x+10}" y="{238+j*15}" font-size="11.5" fill="{INK}">{esc(ln)}</text>')
    if i < len(steps) - 1:
        arrow(x + w + 2, 233, x + w + 16, 233, C_RX[1], width=1.3)
    x += w + 18
label(76, 292, "实测：196 kpps/站、4.69 Mpps、丢包 0（dry-run）；自适应等待下进程 CPU 6.1 核"
               "（纯自旋为 21.5 核）——收包路径不是瓶颈", 12.5)
label(76, 310, "补零上限：单站累计补零 > 488,281 帧 → 干净停止并报出是哪一路（防止无限产零）",
      12.5, MUTED)

# ---------------------------------------------------------------- ③ 共享 staging
band(350, "③", "staging", 176)
box(60, 350, 1500, 176, "共享 pinned staging：24 站 × 3 槽 × 256 帧 × 8192 B = 144 MiB"
                        "（cudaHostAllocPortable，收包线程写一次，多张卡同时读）", [], C_STAGE)
# layout strip: 站 0 的三个槽，然后"…×23"
sx, sy = 96, 408
label(sx - 20, sy - 12, "布局 [站][槽][帧][样本]：", 12.5, INK, "600")
for slot in range(3):
    x0 = sx + slot * 168
    add(f'<rect x="{x0}" y="{sy}" width="150" height="52" rx="6" fill="#ffe9b8" '
        f'stroke="{C_STAGE[1]}" stroke-width="1.2"/>')
    add(f'<text x="{x0+8}" y="{sy+18}" font-size="12" font-weight="600" fill="#8a4b00">'
        f'槽 {slot}：256 帧</text>')
    for f in range(4):
        add(f'<rect x="{x0+8+f*34}" y="{sy+24}" width="30" height="18" rx="3" fill="#fff7e0" '
            f'stroke="{C_STAGE[1]}" stroke-width="0.8"/>')
        add(f'<text x="{x0+23+f*34}" y="{sy+37}" font-size="10" fill="{MUTED}" '
            f'text-anchor="middle">帧</text>')
    add(f'<text x="{x0+75}" y="{sy+37}" font-size="10" fill="{MUTED}" text-anchor="middle">'
        f'… 256 帧 = 2 MiB</text>')
add(f'<text x="{sx+3*168+8}" y="{sy+34}" font-size="13" fill="{INK}" font-weight="600">'
    f'… 站 1 … 站 23（每站 3 槽 = 6 MiB，24 站合计 144 MiB）</text>')
label(76, 462, "① 一张卡的 resident 站集合是分散的 → 拷贝按“站”发（每站一段连续内存，1D 拷贝）；"
               "resident 恰好是全部站时退化成一条 2D 描述符", 12.5, INK)
label(76, 480, "② slot 释放 = 所有持有该站副本的卡都拷完（AND）——多读者，R 越大抖动越大", 12.5, INK)
label(76, 498, "③ 起始帧是按嗅探 counter 预测的；第一包若落在窗口之外，ring 会前移（启动时允许）或"
               "报 StartupGap 干净停止", 12.5, INK)

# ---------------------------------------------------------------- ④ 多卡
band(548, "④", "计算", 330)
label(76, 560, "每张卡 = 一个 CUDA context（一条 copy stream + 一条 compute stream + 一条 fetch stream，事件串联）",
      13, INK, "600")
cards = [
    ("GPU 0", "node 2", "16 站", "基线 0..73（73 条）", "10.2~12.0 GB/s", [0,1,2,3,4,5,7,8,9,10,11,12,14,16,18,20]),
    ("GPU 1", "node 2", "16 站", "基线 1..79（77 条）", "10.2~12.0 GB/s", [0,2,3,5,6,7,8,9,11,13,14,15,17,19,21,22]),
    ("GPU 2", "node 7", "15 站", "基线 2..76（75 条）", "9.6~11.2 GB/s", [1,2,4,5,6,8,10,11,12,13,16,17,18,19,20,23]),
    ("GPU 3", "node 7", "15 站", "基线 3..80（75 条）", "9.6~11.2 GB/s", [0,1,4,6,7,9,10,12,13,15,16,17,20,21,22,23]),
]
cwd, gap = 356, 12
for ci, (name, node, res, base, rate, residents) in enumerate(cards):
    x = 60 + ci * (cwd + gap)
    add(f'<rect x="{x}" y="580" width="{cwd}" height="288" rx="10" fill="{C_GPU[0]}" '
        f'stroke="{C_GPU[1]}" stroke-width="1.8"/>')
    add(f'<text x="{x+12}" y="602" font-size="15" font-weight="700" fill="{C_GPU[1]}">'
        f'{esc(name)}  L20 · {esc(node)} · Gen4 x16</text>')
    add(f'<text x="{x+12}" y="620" font-size="12.5" fill="{INK}">'
        f'resident {esc(res)} · {esc(base)} · 实测 {esc(rate)}</text>')
    # residency strip (24 small squares)
    add(f'<text x="{x+12}" y="638" font-size="11" fill="{MUTED}">'
        f'resident 集合（0..23）：</text>')
    for st in range(24):
        fill = C_GPU[1] if st in residents else "#ffffff"
        add(f'<rect x="{x+12+st*10}" y="644" width="9" height="9" rx="1.5" fill="{fill}" '
            f'stroke="{C_GPU[1]}" stroke-width="0.8"/>')
    chain = [
        "H2D 逐站拷贝（copy stream，异步）",
        "int8 → float32（convert kernel）",
        "cuFFT R2C，长度 8192，批 = 本卡站数×256",
        "相关 kernel（本卡 pair 表，j ≤ k）",
        "行累加器 +=  Y_j · conj(Y_k)（显存常驻）",
        "scale ÷ n_cum → 行栈 rows_per_fetch",
        "D2H：本卡基线 slice → host 行缓冲",
    ]
    y = 668
    for i, t in enumerate(chain):
        add(f'<rect x="{x+12}" y="{y}" width="{cwd-24}" height="22" rx="5" fill="#ffffff" '
            f'stroke="{C_GPU[1]}" stroke-width="1"/>')
        add(f'<text x="{x+22}" y="{y+15.5}" font-size="12" fill="{INK}">{esc(t)}</text>')
        if i < len(chain) - 1:
            arrow(x + cwd / 2, y + 22, x + cwd / 2, y + 27, C_GPU[1], width=1.1, marker="ahg")
        y += 27
    if ci == 0:
        label(x + 12, 872, "R = Σ resident / 站数 = 62/24 = 2.58 → 每卡需求 12.4 GB/s；"
                           "纯拷贝能力 20.3 GB/s/卡", 11.5, MUTED)

# staging -> cards arrows
for ci in range(4):
    x = 60 + ci * (cwd + gap) + cwd / 2
    arrow(x, 528, x, 578, C_GPU[1], width=1.8, marker="ahg")
label(760, 546, "H2D：按卡自己的 resident 集合搬（同一站的字节被 R 张卡各搬一次）", 12, C_GPU[1], anchor="middle")

# cards -> row buffer
for ci in range(4):
    x = 60 + ci * (cwd + gap) + cwd / 2
    arrow(x, 870, x, 906, C_GPU[1], width=1.8, marker="ahg")

# ---------------------------------------------------------------- ⑤ 调度
band(908, "⑤", "调度", 150)
box(60, 908, 1000, 150, "管道线程（1 个，绑核 24）——它只排队和等待，不算数", [], C_SCHED)
sched = [
    ("水位线", "min(各站 progress)"),
    ("提交槽", "corr_row_accumulate"),
    ("释放槽", "corr_row_slot_free（等所有卡）"),
    ("关行", "corr_row_end（÷n_cum）"),
    ("取行", "rows_wait + rows_fetch"),
]
x = 76
for i, (t, d) in enumerate(sched):
    w = 178
    add(f'<rect x="{x}" y="{938}" width="{w}" height="52" rx="7" fill="#ffffff" '
        f'stroke="{C_SCHED[1]}" stroke-width="1.2"/>')
    add(f'<text x="{x+10}" y="{957}" font-size="12.5" font-weight="600" fill="{C_SCHED[1]}">'
        f'{esc(t)}</text>')
    add(f'<text x="{x+10}" y="{974}" font-size="11" fill="{INK}">{esc(d)}</text>')
    if i < len(sched) - 1:
        arrow(x + w + 2, 964, x + w + 14, 964, C_SCHED[1], width=1.2)
    x += w + 16
label(76, 1010, "实测时间分解（24 路 4 卡 covering）：排队 0.6 s、等释放 10.0 s（占 71%）、"
                "取行 ~0 ⇒ 它卡在“等拷贝”上", 12.5, INK, "600")
label(76, 1028, "补零/脱困：某站 200 ms 无任何包 → 调度器要求它补零到当前槽末（受 488,281 帧上限约束）；"
                "调度器只对有包到达的站判活", 11.5, MUTED)
box(1076, 908, 484, 150, "控制面（虚线）",
    ["水位线 ← 各站 progress（原子量）",
     "补零请求 → 收包线程 advance_to",
     "释放 → ring 唤醒等待者（自适应等待）",
     "停止：SIGINT / 补零超限 / StartupGap / 收包致命错"],
    ("#f1f3f4", C_CTRL), dash="5 4")

# ---------------------------------------------------------------- ⑥ 输出
band(1078, "⑥", "输出", 152)
box(60, 1078, 900, 152, "行拼装 + writer（在管道线程里同步执行）",
    ["每行 = 300 baseline × 4096 bin × complex64 = 9.83 MB",
     "covering 下每卡的基线在全局编号里散开 → 库内先取到 scratch，再按全局基线号逐个放置",
     "基线归属由 corr_device_baselines 可查；拼好后写盘：",
     "  每个 baseline 一个 <file_prefix><站j><站k>-<时刻>.bin（O_APPEND 语义但用 seek，可续写）",
     "  <file_prefix><时刻>-time.bin / -time.txt（每行首帧 final 的时刻；.bin 是 Unix ms，"
     ".txt 是本地 YYYY-mm-DD HH:MM:SS.mmm）",
     "  <file_prefix><时刻>-blocks.csv（每站每行：first_frame_id, n_cum, valid, zero_filled）",
     "  <file_prefix><时刻>-metadata.yaml（几何 + config_hash + 每个 baseline 的文件名）"],
    C_OUT)
box(980, 1078, 580, 152, "输出速率与续写",
    ["1 s 积分：9.82 MB/s、300 文件写/s、fsync 300 次/行（fsync_every_rows=1）",
     "10 ms 积分：937 MB/s、28,610 fsync/s ⇒ 必须先实现 storage.writers（见 HANDOFF 6.9）",
     "/scratch：实测单线程缓冲写 837 MB/s；全局 sync 会卡数分钟，别用它计时",
     "续写：config_hash 一致才允许；四个文件按行数取最小者对齐，可随时 Ctrl-C"],
    C_OUT)
arrow(880, 1060, 880, 1076, C_OUT[1], width=1.8)
arrow(980, 1154, 1000, 1154, C_OUT[1], width=1.2)

# ---------------------------------------------------------------- 结论条
add(f'<rect x="60" y="1244" width="1500" height="48" rx="8" fill="#f8f9fa" '
    f'stroke="#dadce0"/>')
add(f'<text x="76" y="1263" font-size="12.5" fill="{INK}">'
    f'<tspan font-weight="700">R（复制因子）= 图中“同一站的字节被搬几次”的倍数。</tspan>'
    f'约束：R × 32 GB/s ≤ 4 × 每卡可达 H2D。实测 4 卡纯拷贝 81~97 GB/s（20.3/卡）、'
    f'管线 60 GB/s（15/卡）；40 路在 R=2.10 时需 67 GB/s（每卡 16.8）。</text>')
add(f'<text x="76" y="1281" font-size="12.5" fill="{INK}">'
    f'<tspan font-weight="700">⇒ 差 10~15%：需要三条乘法杠杆之一或组合 —— '
    f'①R 从 2.58 压到 2.10　②每卡能力 15→20 GB/s　③每站字节减半（4 bit 打包）。</tspan>'
    f'</text>')
add('</svg>')

print("\n".join(out))
