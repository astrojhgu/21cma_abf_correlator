"""Conversions shared by the judging tools, in one place.

Every tool here reads one of two products: a `blocks.csv` from a run, or a
`baseline_*.bin` from a run. Both are shaped by one arithmetic detail that has
already caused three separate wrong answers:

    the channel axis is the **retained half** of a real-to-complex transform

so a spectrum file with `nch` channels came from an FFT of `2 * nch` samples. A
delay of `d` samples multiplies channel `k` by `exp(-2*pi*i*k*d/fft_len)`, and using
`nch` in place of `fft_len` reports exactly **half** the delay -- a plausible number,
which is why it survived two rounds of review in `phase_triangle.py` (once in the
estimator, once in the line drawn over the data) and once more in a throwaway
script. Anything that converts between channels, delay and phase lives here now.
"""

from __future__ import annotations


def fft_len(nch: int) -> int:
    """Samples per FFT frame, given the number of retained channels."""
    return 2 * nch


def delay_per_channel_rad(delay: float, nch: int) -> float:
    """Phase advance per channel, in radians, for a delay in samples."""
    return -2.0 * 3.141592653589793 * delay / fft_len(nch)


def delay_from_slope(slope_rad_per_channel: float, nch: int) -> float:
    """Delay in samples, from a fitted phase slope in radians per channel."""
    return -slope_rad_per_channel * fft_len(nch) / (2.0 * 3.141592653589793)


def wrap_deg(v):
    """Fold degrees into [-180, 180). The axis convention the plots use."""
    import numpy as np

    return (np.asarray(v) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# File sets
# ---------------------------------------------------------------------------
#
# A run directory holds one *set* per local day (docs/DESIGN-daily-files.md).
# Since 2026-09-18 a set looks like
#
#   <file_prefix><station_j><station_k>-<stamp>.bin   one per baseline
#   <tag>-time.bin / <tag>-time.txt                   per-row wall clock
#   <tag>-blocks.csv                                  per-row validity
#   <tag>-metadata.yaml                               how to read the above
#
# with `<tag>` = `<file_prefix>-<stamp>` (just `<stamp>` when no prefix is
# configured) and `<stamp>` the local time of the set's first row. Sets written
# before that -- `<prefix>baseline_0000_j00_k00.bin`, `<prefix>rows_ms.*` -- are
# still read; `find_sets` tags each set with the layout it found. Reading a day
# means naming the set; reading "the run" means reading them in order.

import os
import re
import sys

BASELINE_RE = re.compile(r"^baseline_(\d{4})_j(\d{2})_k(\d{2})\.bin$")


def find_sets(run_dir, require=None):
    """Every file set in `run_dir`, sorted by tag (oldest first).

    Two layouts are recognised, because the file names changed once (2026-09-18)
    and the archives from before it are still read:

    * **named** -- `<tag>-blocks.csv` plus `<tag>-time.bin`, `<tag>-time.txt` and
      `<tag>-metadata.yaml`, with one `<file_prefix><station_j><station_k>-<stamp>.bin`
      per baseline. `tag` is the set's identity and is what `--set` takes.
    * **legacy** -- `<prefix>blocks.csv` plus `<prefix>rows_ms.bin`,
      `<prefix>rows_ms.txt`, `<prefix>metadata.yaml` and
      `<prefix>baseline_0000_j00_k00.bin`.

    Every set is tagged with the layout it was found in, so callers that need a
    file the two layouts name differently (`rows_ms` below) do not have to care.
    """
    out = []
    try:
        names = os.listdir(run_dir)
    except FileNotFoundError:
        sys.exit(f"{run_dir}: no such directory")
    for name in names:
        if name.endswith("-blocks.csv"):
            tag = name[: -len("-blocks.csv")]
            layout = "named"
            files = {
                "blocks": os.path.join(run_dir, name),
                "rows_ms": os.path.join(run_dir, f"{tag}-time.bin"),
                "rows_ms_txt": os.path.join(run_dir, f"{tag}-time.txt"),
                "metadata": os.path.join(run_dir, f"{tag}-metadata.yaml"),
            }
        elif name.endswith("blocks.csv"):
            tag = name[: -len("blocks.csv")]
            layout = "legacy"
            files = {
                "blocks": os.path.join(run_dir, name),
                "rows_ms": os.path.join(run_dir, tag + "rows_ms.bin"),
                "rows_ms_txt": os.path.join(run_dir, tag + "rows_ms.txt"),
                "metadata": os.path.join(run_dir, tag + "metadata.yaml"),
            }
        else:
            continue
        if require and not os.path.exists(files["metadata"]):
            continue
        out.append(dict(dir=run_dir, prefix=tag, tag=tag, layout=layout, **files))
    if not out and not require:
        # A directory with no set at all: report one empty set so the caller's
        # error message names the path rather than an empty list.
        out.append(
            {
                "dir": run_dir,
                "prefix": "",
                "tag": "",
                "layout": "named",
                "blocks": os.path.join(run_dir, "blocks.csv"),
                "rows_ms": os.path.join(run_dir, "time.bin"),
                "rows_ms_txt": os.path.join(run_dir, "time.txt"),
                "metadata": os.path.join(run_dir, "metadata.yaml"),
            }
        )
    return sorted(out, key=lambda s: s["prefix"])


def pick_set(run_dir, prefix=None):
    """The set to read: the named one, or the newest (highest prefix).

    Newest by default because that is the day a person asking about "the run" is
    almost always asking about, and because prefixes sort chronologically.
    """
    sets = find_sets(run_dir, require="metadata.yaml")
    if prefix is not None:
        for s in sets:
            if s["prefix"] == prefix:
                return s
        sys.exit(
            f"{run_dir}: no set with prefix {prefix!r}; there are "
            + ", ".join(repr(s["prefix"]) for s in sets)
        )
    if not sets:
        sys.exit(f"{run_dir}: no file set (no *-metadata.yaml and no *metadata.yaml)")
    return sets[-1]


def baseline_files(set_info):
    """`[(index, j, k, filename)]` from the set's metadata.

    The writer records every baseline's file name, so a reader never has to
    reconstruct the naming rule -- which changed once already (the station pair
    and the stamp moved into the name on 2026-09-18) and is not worth
    reconstructing twice. A set written before that has `file:` values in the old
    shape and is read the same way.
    """
    out = []
    try:
        fh = open(set_info["metadata"])
    except OSError:
        return out
    with fh:
        inside = False
        for line in fh:
            if line.startswith("baselines:"):
                inside = True
                continue
            if not inside:
                continue
            if not line.startswith("  - "):
                break
            fields = dict(re.findall(r"(\w+): ([^,}]+)", line))
            if "file" in fields:
                out.append(
                    (
                        int(fields["index"]),
                        int(fields["j"]),
                        int(fields["k"]),
                        fields["file"].strip(),
                    )
                )
    return out


def set_baselines(run_dir, plan_files=None, prefix=None):
    """`(index, j, k, path)` for every baseline file of the chosen set.

    The names come from the set's own metadata, so both layouts are handled the
    same way; the regex below is only a fallback for a set whose metadata predates
    the `baselines:` list.
    """
    s = pick_set(run_dir, prefix)
    out = []
    for idx, j, k, fname in baseline_files(s):
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            out.append((idx, j, k, p))
    if out:
        return sorted(out)
    for name in os.listdir(run_dir):
        if not name.startswith(s["prefix"]):
            continue
        m = BASELINE_RE.match(name[len(s["prefix"]) :])
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), os.path.join(run_dir, name)))
    out.sort()
    return out


def metadata_field(set_info, key):
    """One `key: value` line out of a set's metadata.yaml, or None."""
    try:
        with open(set_info["metadata"]) as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return None
    return None


def rows_ms(set_info):
    """Every row's wall clock in the set, as ints, in file order."""
    import struct

    with open(set_info["rows_ms"], "rb") as fh:
        raw = fh.read()
    n = len(raw) // 8
    return list(struct.unpack(f"<{n}q", raw[: n * 8]))
