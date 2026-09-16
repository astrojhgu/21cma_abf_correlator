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
# A run directory holds one *set* per local day: `<prefix>baseline_*.bin`,
# `<prefix>rows_ms.bin`, `<prefix>rows_ms.txt`, `<prefix>blocks.csv`,
# `<prefix>metadata.yaml` (docs/DESIGN-daily-files.md). With
# `storage.daily_files: false` the prefix is empty and there is exactly one set,
# which is what every tool here used to assume. Reading a day therefore means
# naming the set; reading "the run" means reading them in order.

import os
import re
import sys

BASELINE_RE = re.compile(r"^baseline_(\d{4})_j(\d{2})_k(\d{2})\.bin$")


def find_sets(run_dir, require=None):
    """Every file set in `run_dir`, sorted by prefix (oldest first).

    A set is recognised by its `blocks.csv`; the prefix is whatever precedes that
    name. `require` is an optional filename to insist each set contains (e.g.
    `metadata.yaml`), which also filters out half-created sets.
    """
    out = []
    try:
        names = os.listdir(run_dir)
    except FileNotFoundError:
        sys.exit(f"{run_dir}: no such directory")
    for name in names:
        if not name.endswith("blocks.csv"):
            continue
        prefix = name[: -len("blocks.csv")]
        if require and not os.path.exists(os.path.join(run_dir, prefix + require)):
            continue
        out.append(
            {
                "dir": run_dir,
                "prefix": prefix,
                "blocks": os.path.join(run_dir, name),
                "rows_ms": os.path.join(run_dir, prefix + "rows_ms.bin"),
                "rows_ms_txt": os.path.join(run_dir, prefix + "rows_ms.txt"),
                "metadata": os.path.join(run_dir, prefix + "metadata.yaml"),
            }
        )
    if not out and not require:
        # A directory with no blocks.csv at all: report it as one empty set so the
        # caller's error message names the path rather than an empty list.
        out.append(
            {
                "dir": run_dir,
                "prefix": "",
                "blocks": os.path.join(run_dir, "blocks.csv"),
                "rows_ms": os.path.join(run_dir, "rows_ms.bin"),
                "rows_ms_txt": os.path.join(run_dir, "rows_ms.txt"),
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
        sys.exit(f"{run_dir}: no file set (no <prefix>metadata.yaml)")
    return sets[-1]


def set_baselines(run_dir, plan_files=None, prefix=None):
    """`(index, j, k, path)` for every baseline file of the chosen set.

    `plan_files` is ignored; it exists so a caller that already listed the
    directory can pass what it has and still get the unprefixed layout right.
    """
    s = pick_set(run_dir, prefix)
    out = []
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
