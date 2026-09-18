#!/usr/bin/env bash
# tools/daily_rollover_test.sh -- does the output really change files at local midnight?
#
# The one thing a daily-file feature has to get right is a boundary that happens
# once a day, so waiting for it is not a test. This moves local midnight into the
# test window instead, with a POSIX TZ string: "TZ=CORR-14:57" means UTC+14:57, and
# any offset is expressible to the second, so the run can be started at local
# 23:58:xx on purpose.
#
# What it checks, all of it on the files the run left behind (docs/DESIGN-daily-files.md §5):
#
#   1. a second file set appears, with a different local day in its prefix
#   2. the prefix of each set equals the local time of that set's own first row
#   3. the two sets' rows add up to the rows the run reported, with no gap and no
#      overlap in the global row_index
#   4. the frame grid is continuous across the boundary (next first_frame_id =
#      previous last + n_cum)
#   5. rows_ms is strictly increasing when the sets are read in order
#   6. the boundary row is attributed by its *start*: the last row of the old set
#      started before local midnight, the first row of the new set after it
#   7. what the switch cost: the gap between those two rows' timestamps against the
#      nominal row period, and the per-row zero fill on both sides
#
# Usage: tools/daily_rollover_test.sh [RATE] [DURATION_S] [MINUTES_BEFORE_MIDNIGHT]
set -euo pipefail
cd "$(dirname "$0")/.."

RATE=${1:-20000}
DUR=${2:-240}
LEAD_MIN=${3:-2}          # start this many minutes before local midnight
CFG=config/correlator-24-replay.yaml
OUT=${OUT:-/scratch/daily-rollover-$(date +%H%M%S)}
# Never judge a directory that already holds a run: an earlier attempt's sets would
# be read as this run's, which fails every count check for a reason that has nothing
# to do with the code. A second run into the same path gets its own subdirectory.
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  OUT="$OUT/$(date +%Y%m%d-%H%M%S)"
  echo "note       : that output path was not empty; using $OUT"
fi

# --- put local midnight LEAD_MIN minutes from now ---
read -r TZSTR LOCALNOW <<<"$(python3 - "$LEAD_MIN" <<'PY'
import sys, time
lead = float(sys.argv[1]) * 60.0
now = time.time()
# local time we want right now: 24:00 minus `lead`
tod = (now + time.timezone - time.daylight * 3600) % 86400  # placeholder, replaced below
# Solve for the POSIX offset instead: local = UTC + offset, and we want
# local_time_of_day == (86400 - lead) mod 86400.
want = (86400.0 - lead) % 86400.0
utc_tod = now % 86400.0
off = (want - utc_tod) % 86400.0          # seconds east of UTC
if off > 43200:
    off -= 86400.0                        # keep it in (-12h, +12h] for sanity
sign = "-" if off >= 0 else "+"           # POSIX TZ is inverted: west is positive
a = abs(off)
h, rem = divmod(int(a), 3600)
m, s = divmod(rem, 60)
tz = f"CORR{sign}{h:02d}:{m:02d}:{s:02d}"
import os
os.environ["TZ"] = tz
time.tzset()
print(tz, time.strftime("%Y-%m-%d %H:%M:%S %z"))
PY
)"
export TZ="$TZSTR"
echo "daily rollover test"
echo "  local now  : $LOCALNOW   (TZ=$TZSTR, UTC $(date -u +%H:%M:%S))"
echo "  midnight in: ${LEAD_MIN} min; run: rate=$RATE duration=${DUR}s"
echo "  out        : $OUT"
mkdir -p "$OUT"

pkill -x corr-replay 2>/dev/null || true
sleep 1
if pgrep -x corr-replay >/dev/null; then
  echo "a corr-replay is still running; refusing to start a second producer" >&2
  exit 1
fi

# The replay must outlive the run: a stream that stops first hangs the correlator
# (docs/DESIGN-daily-files.md, and the handover's list of traps).
REPLAY_SECS=$(( DUR + 30 ))
( for i in $(seq 1 $(( (DUR + LEAD_MIN * 60) / 3 + 20 )) ); do
    echo "load=$(cut -d' ' -f1-3 /proc/loadavg)"
    sleep 3
  done ) > "$OUT/load.txt" &

./corr-rs/target/release/corr-replay --config "$CFG" --pattern prng \
    --rate "$RATE" --duration "$REPLAY_SECS" > "$OUT/replay.log" 2>&1 &
REPLAY_PID=$!
sleep 6

# Watch the directory while the run goes, so the *cost* of the switch can be
# measured rather than argued about. `st_mtime` on its own cannot do it: the old
# set's last write is visible, but the new set's file keeps being written, so its
# mtime is the last row of the test rather than the first. Watching "when did the
# last of the new set's 304 files appear" against "when was the old set last
# written" gives the window the pipeline thread spent closing one set and creating
# the next.
RUN_ROOT="$OUT"
export RUN_ROOT
python3 - > "$OUT/switch-times.tsv" <<'POLL' &
import os, re, time
parent = os.environ["RUN_ROOT"]
deadline = time.time() + float(os.environ.get("POLL_SECS", "900"))
root = parent
seen, ready, last_change, lastname, last_mtime = {}, {}, {}, {}, {}
while time.time() < deadline:
    if os.path.exists(os.path.join(parent, "POLL-STOP")):
        break
    subdirs = sorted(
        d for d in os.listdir(parent)
        if d.startswith("corr_") and os.path.isdir(os.path.join(parent, d))
    )
    if subdirs:
        root = os.path.join(parent, subdirs[-1])
        for name in os.listdir(root):
            # A set announces itself through its metadata; the tag is the name
            # without that suffix. The sidecar that grows every row is time.bin,
            # so its mtime is what says the set is still being written.
            if not name.endswith("-metadata.yaml"):
                continue
            prefix = name[: -len("-metadata.yaml")]
            path = os.path.join(root, prefix + "-time.bin")
            # Everything is timestamped with *this* process's clock. The file
            # system's own mtime is not usable here: the shared /scratch reports
            # mtimes about thirteen minutes behind the wall clock, so comparing one
            # against time.time() invents a stall of that size.
            try:
                m = os.stat(path).st_mtime
            except FileNotFoundError:
                continue
            if prefix not in seen:
                seen[prefix] = time.time()
            if last_mtime.get(prefix) != m:
                last_mtime[prefix] = m
                last_change[prefix] = time.time()
            if prefix not in lastname:
                try:
                    txt = open(os.path.join(root, prefix + "-metadata.yaml")).read()
                    # The set's metadata lists every baseline's file name; the last
                    # entry is the last file it creates, so its arrival means the
                    # set is complete.
                    files = re.findall(r"file: (\S+?)\}", txt)
                    if files:
                        lastname[prefix] = files[-1]
                except Exception:
                    pass
            if prefix in lastname and prefix not in ready:
                if os.path.exists(os.path.join(root, prefix + lastname[prefix])):
                    ready[prefix] = time.time()
    time.sleep(0.01)
for p in sorted(seen):
    print(f"{p}\t{seen[p]:.3f}\t{ready.get(p, 0):.3f}\t{last_change.get(p, 0):.3f}")
POLL
POLLER=$!

./corr-rs/target/release/corr --config "$CFG" --duration "$DUR" --output-dir "$OUT" \
    2>&1 | tee "$OUT/corr.log" | grep -E "^daily files|^set |^start frame|^  output:|STOPPED" || true
touch "$OUT/POLL-STOP"
wait $POLLER 2>/dev/null || true
pkill -x corr-replay 2>/dev/null || true
wait $REPLAY_PID 2>/dev/null || true

RUN_DIR=$(ls -d "$OUT"/corr_*/ | head -1)
echo
echo "checking $RUN_DIR"

FILE_PREFIX_FORMAT=$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['storage'].get('file_prefix_format','%Y%m%d-%H%M%S'))")
export FILE_PREFIX_FORMAT
python3 - "$RUN_DIR" <<'PY'
import os, re, struct, sys, time
sys.path.insert(0, "tools")   # corr_common: the same set discovery the other tools use
import corr_common as cc

run_dir = sys.argv[1].rstrip("/")
sets = [s for s in cc.find_sets(run_dir, require="metadata.yaml")]
if not sets:
    sys.exit("FAIL: no file set (no <prefix>metadata.yaml) was written")

def rows_of(path):
    n = os.path.getsize(path) // 8
    with open(path, "rb") as fh:
        return list(struct.unpack(f"<{n}q", fh.read(n * 8)))

def blocks_of(path, n_station):
    out = []
    for line in open(path):
        f = line.strip().split(",")
        if len(f) >= 6 and f[0].isdigit():
            out.append((int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[5])))
    return out

fail = []
def check(ok, what, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fail.append(what)

print(f"sets found: {len(sets)}")
info = []
for s in sets:
    day = cc.metadata_field(s, "set_day")
    kind = cc.metadata_field(s, "set_kind")
    n_cum = int(cc.metadata_field(s, "n_cum") or 0)
    n_station = int(cc.metadata_field(s, "n_station") or 1)
    ms = rows_of(s["rows_ms"])
    bl = blocks_of(s["blocks"], n_station)
    first_rows = [b for b in bl if b[1] == 0]  # one line per station per row
    info.append(dict(s=s, day=day, kind=kind, n_cum=n_cum, n_station=n_station, ms=ms,
                     n_rows=len(ms), blocks_rows=len(first_rows),
                     first_frame=first_rows[0][2] if first_rows else None,
                     last_frame=first_rows[-1][2] if first_rows else None,
                     row_index_first=first_rows[0][0] if first_rows else None,
                     row_index_last=first_rows[-1][0] if first_rows else None,
                     zf=[b[4] for b in first_rows]))
    print(f"  {s['prefix']!r} day={day} kind={kind} rows={len(ms)} "
          f"row_index={info[-1]['row_index_first']}..{info[-1]['row_index_last']} "
          f"frames={info[-1]['first_frame']}..{info[-1]['last_frame']}")

check(len(sets) >= 2, "a second file set appeared (the day changed under the run)",
      f"{len(sets)} sets")
if len(sets) < 2:
    sys.exit(1)

# 2. the set's stamp is the local time of its own first row, and its tag is the
#    configured prefix followed by that stamp
fmt = os.environ.get("FILE_PREFIX_FORMAT") or "%Y%m%d-%H%M%S"
for i in info:
    want = time.strftime(fmt, time.localtime(i["ms"][0] / 1000.0))
    fp = cc.metadata_field(i["s"], "file_prefix") or ""
    stamp = cc.metadata_field(i["s"], "stamp") or ""
    check(stamp == want, f"stamp names its own first row ({stamp!r} vs {want!r})")
    check(i["s"]["prefix"] == fp + stamp,
          f"tag is the prefix plus the stamp ({i['s']['prefix']!r} vs {fp + stamp!r})")

# 1+6. different days, and the boundary row attributed by its start
n_cum = info[0]["n_cum"]
for prev, nxt in zip(info, info[1:]):
    check(prev["day"] != nxt["day"], f"day differs across the boundary ({prev['day']} -> {nxt['day']})")
    check(nxt["day"] > prev["day"], "the new day is later, not earlier")
    # every row of a set belongs to that set's local day
    same = all(time.strftime("%Y%m%d", time.localtime(m / 1000.0)) == prev["day"] for m in prev["ms"])
    check(same, f"every row of {prev['s']['prefix']!r} is on its own local day")
    # 3. contiguous global row_index
    check(nxt["row_index_first"] == prev["row_index_last"] + 1,
          "row_index continues across the boundary",
          f"{prev['row_index_last']} -> {nxt['row_index_first']}")
    # 4. contiguous frame grid
    check(nxt["first_frame"] == prev["last_frame"] + n_cum,
          "first_frame_id continues (previous last + n_cum)",
          f"{prev['last_frame']} + {n_cum} vs {nxt['first_frame']}")
    # 5. monotone timestamps across the boundary
    check(nxt["ms"][0] > prev["ms"][-1], "the new set's first row is later in time",
          f"{prev['ms'][-1]} -> {nxt['ms'][0]}")

# 5b. strictly increasing within every set
for i in info:
    inc = all(b > a for a, b in zip(i["ms"], i["ms"][1:]))
    check(inc, f"rows_ms strictly increasing inside {i['s']['prefix']!r}")

# 7. what the switch cost.
#
# Not from rows_ms: those are the *grid* time of each row's first frame, handed
# down from the front end, so they cannot show a stall in the writer at all. The
# cost lands on the pipeline thread -- close+fsync the old set, create 304 files of
# the new one -- and it shows up between the old set's last write and the new set's
# first one. Second resolution is too coarse for that, so this is in milliseconds.
def mtime(path):
    try:
        return os.stat(path).st_mtime
    except FileNotFoundError:
        return None

def baseline_zero(d, prefix):
    # corr_common reads the names out of the set's metadata, so this works for
    # both layouts and does not repeat the naming rule here.
    b = cc.set_baselines(d, prefix=prefix)
    return b[0][3] if b else os.path.join(d, "missing-baseline")

periods = []
for i in info:
    periods += [b - a for a, b in zip(i["ms"], i["ms"][1:])]
nominal = sorted(periods)[len(periods) // 2] if periods else 0
print(f"  nominal row period inside a set: {nominal/1000:.2f} s "
      f"(the injector's rate, not the front end's 195 kpps)")
for prev, nxt in zip(info, info[1:]):
    gap = nxt["ms"][0] - prev["ms"][-1]
    print(f"  boundary in grid time: {gap/1000:.2f} s between the two rows "
          f"({(gap - nominal)/1000:+.2f} s against the nominal period; grid time, so not a stall)")
    # Measured live by the poller the script started beside the run.
    try:
        poll = {}
        with open(os.path.join(os.path.dirname(run_dir.rstrip("/")), "switch-times.tsv")) as fh:
            for line in fh:
                f = line.split()
                if len(f) == 4:
                    poll[f[0]] = (float(f[1]), float(f[2]), float(f[3]))
        a = poll.get(prev["s"]["prefix"], (0, 0, 0))[2]      # its last row was written
        b = poll.get(nxt["s"]["prefix"], (0, 0, 0))[1]       # its 304 files all existed
        c = poll.get(nxt["s"]["prefix"], (0, 0, 0))[0]       # its first file existed
        if a and b and c:
            # Two things separate those events: the row period (the next row of the
            # new day cannot start being written before it exists) and the switch
            # itself (close+fsync the old set, create 304 files). Only the second is
            # the cost of this feature, so the period is subtracted using the
            # *injector's* period measured above -- both clocks are CLOCK_REALTIME,
            # unlike the file system's mtime, which on this shared /scratch runs
            # about thirteen minutes behind.
            period = nominal / 1000.0
            print(f"  switch cost: {1000*(c-a) - 1000*period:.0f} ms of extra pipeline time "
                  f"(first new file {1000*(c-a):.0f} ms after the old set's last write, minus the "
                  f"{period:.2f} s row period)")
            print(f"  the new set's 304 files were all there {1000*(b-c):.0f} ms after its first")
    except FileNotFoundError:
        pass
    print(f"  zero fill: last 3 rows of {prev['s']['prefix']!r} {prev['zf'][-3:]} frames, "
          f"first 3 rows of {nxt['s']['prefix']!r} {nxt['zf'][:3]} frames")

total_rows = sum(i["n_rows"] for i in info)
reported = None
for line in open(os.path.join(os.path.dirname(run_dir.rstrip('/')), "corr.log"), errors="replace"):
    m = re.search(r"\|\s*(\d+) rows\s*\|", line)
    if m:
        reported = int(m.group(1))
print(f"  rows: {total_rows} in the sets, {reported} reported by the run")
check(reported is None or total_rows == reported, "the sets' rows add up to the run's rows")

tw = sum(sum(i["zf"]) for i in info)
tt = sum(i["n_rows"] * i["n_station"] * i["n_cum"] for i in info)
print(f"  zero fill over the whole test: {tw} frames of {tt} ({100.0*tw/max(tt,1):.4f}%)")

print()
if fail:
    print(f"FAILED: {len(fail)} check(s): {fail}")
    sys.exit(1)
print("all checks passed")
PY

echo
echo "artifacts in $OUT"
