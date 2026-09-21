#!/usr/bin/env python3
"""Report per-level medians across repeated metered passes.

A single metered pass at 128 tokens swings by up to 18% on this rig, so one pass
cannot support an A/B at that scale. Repeating and taking the median cuts the
spread to a few percent; this reads the `c<N>` lines the meter writes and
prints the median aggregate tok/s per level.

  median_levels.py <tag> <passes> [levels...]
"""

import re
import statistics
import sys
from pathlib import Path

DRIVER = Path.home() / "vllm-spark-0731" / "outputs" / "driver"
AGG = re.compile(r"agg=\s*([\d.]+)")
WALL = re.compile(r"wall=\s*([\d.]+)s")
TOKS = re.compile(r"toks=(\d+)")


def read_pass(tag: str, pass_no: int) -> dict[int, tuple[float, float]]:
    """level -> (agg tok/s, tokens/wall recomputed)."""
    path = DRIVER / f"{tag}-{pass_no}.meter.log"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("c") or not line[1:2].isdigit():
            continue
        level = int(line.split()[0][1:])
        agg = AGG.search(line)
        wall = WALL.search(line)
        toks = TOKS.search(line)
        if not (agg and wall and toks):
            continue
        wall_s = float(wall.group(1))
        out[level] = (float(agg.group(1)), int(toks.group(1)) / wall_s if wall_s else 0.0)
    return out


def main() -> None:
    tag = sys.argv[1]
    passes = int(sys.argv[2])
    levels = [int(x) for x in sys.argv[3:]] or [1, 3, 5, 6]

    data = {level: [] for level in levels}
    for p in range(1, passes + 1):
        got = read_pass(tag, p)
        if not got:
            print(f"warning: no results for pass {p} ({DRIVER}/{tag}-{p}.meter.log)")
        for level, value in got.items():
            data.setdefault(level, []).append(value)

    print(f"tag {tag}, {passes} passes")
    print(f"{'level':>6} {'n':>3} {'median agg tok/s':>17} {'spread':>9}   values")
    for level in sorted(data):
        vals = [v[0] for v in data[level]]
        if not vals:
            print(f"{level:6d}   0 {'-':>17} {'-':>9}")
            continue
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) / med * 100 if med else 0.0
        shown = " ".join(f"{v:.1f}" for v in vals)
        print(f"{level:6d} {len(vals):3d} {med:17.1f} {spread:8.1f}%   {shown}")


if __name__ == "__main__":
    main()
