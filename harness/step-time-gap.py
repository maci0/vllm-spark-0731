#!/usr/bin/env python3
"""Derive engine step time from the drive-meter artefacts and compare two arms.

`drive-meter.sh` reports wall time and `drafts_per_req`, but `drafts_per_req` is a
**sum over the concurrent requests**, so the engine step count is::

    steps = drafts_per_req / concurrency

confirmed exactly at c6: 640 drafts / 6 = 106.7 steps, and 512 tokens /
4.800 tokens_per_step = 106.7. So::

    step_ms = wall_s / steps * 1000

That conversion is the whole point. Aggregate tok/s alone cannot separate "our step
returns fewer tokens" from "our step takes longer", and the two need opposite fixes.

Usage:
    step-time-gap.py <ours-tag> <ref-tag> [passes...]      # default 1 2 3
"""
from __future__ import annotations

import pathlib
import re
import statistics
import sys

LEVELS = (1, 3, 5, 6)
DRIVER = pathlib.Path("outputs/driver")


def parse(tag: str, passes: tuple[int, ...]) -> dict[int, list[tuple[float, float, float]]]:
    """level -> [(step_ms, tokens_per_step, accept_rate)] over the passes."""
    out: dict[int, list[tuple[float, float, float]]] = {L: [] for L in LEVELS}
    for p in passes:
        path = DRIVER / f"{tag}-{p}.meter.txt"
        if not path.exists():
            continue
        blocks = re.split(r"-- level (\d+)\n", path.read_text())
        for i in range(1, len(blocks), 2):
            level = int(blocks[i])
            body = blocks[i + 1]
            wall = re.search(r"wall_s\s+([0-9.]+)", body)
            drafts = re.search(r"drafts_per_req\s+(\d+)", body)
            tps = re.search(r"tokens_per_step\s+([0-9.]+)", body)
            acc = re.search(r"accept_rate\s+([0-9.]+)", body)
            if not (wall and drafts and tps) or level not in out:
                continue
            steps = float(drafts.group(1)) / level
            out[level].append(
                (
                    float(wall.group(1)) / steps * 1000.0,
                    float(tps.group(1)),
                    float(acc.group(1)) if acc else 0.0,
                )
            )
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ours_tag, ref_tag = sys.argv[1], sys.argv[2]
    passes = tuple(int(a) for a in sys.argv[3:]) or (1, 2, 3)
    ours, ref = parse(ours_tag, passes), parse(ref_tag, passes)
    if not any(ours.values()) or not any(ref.values()):
        print("no meter artefacts found; check the tags and outputs/driver/")
        return 1

    print(f"{'lvl':>4} {'ours step_ms':>13} {'ref step_ms':>12} {'delta':>8} "
          f"{'ours t/step':>12} {'ref t/step':>11} {'ours acc':>9} {'ref acc':>8} {'n':>3}")
    deltas = []
    for level in LEVELS:
        if not ours[level] or not ref[level]:
            continue
        med = statistics.median
        o_step = med([x[0] for x in ours[level]])
        r_step = med([x[0] for x in ref[level]])
        o_tps = med([x[1] for x in ours[level]])
        r_tps = med([x[1] for x in ref[level]])
        o_acc = med([x[2] for x in ours[level]])
        r_acc = med([x[2] for x in ref[level]])
        deltas.append(o_step - r_step)
        print(f"{level:>4} {o_step:>13.1f} {r_step:>12.1f} {o_step - r_step:>+8.1f} "
              f"{o_tps:>12.3f} {r_tps:>11.3f} {o_acc:>8.1f}% {r_acc:>7.1f}% "
              f"{min(len(ours[level]), len(ref[level])):>3}")

    if deltas:
        lo, hi = min(deltas), max(deltas)
        print(f"\nstep-time delta: median {statistics.median(deltas):+.1f} ms, "
              f"range {lo:+.1f} .. {hi:+.1f} ms")
        print("A delta that stays flat across levels is a FIXED per-step cost "
              "(launch/python/eager glue); a delta that grows with batch is "
              "per-row work. They need different fixes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
