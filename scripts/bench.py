#!/usr/bin/env python3
"""Decode-throughput benchmark for the live head.

Usage (run on spark1):
    python3 scripts/bench.py [--runs-1way N] [--runs-8way N] [--base URL]

Measures greedy decode (temperature=0, max_tokens=128) for:
  - 1-way: N sequential single-request runs -> per-run and median tok/s.
  - 8-way: N runs of 8 concurrent requests   -> aggregate wall tok/s (1024/elapsed).

Stdlib only. Match HANDOFF.md pins: 1-way median ~26.90, 8-way ~85.98.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8000/v1/completions"
MODEL = "deepseek-v4-flash"
MAX_TOKENS = 128
PROMPT = "The capital of France is"


def decode_once(base: str) -> tuple[float, int]:
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        base, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    dt = time.perf_counter() - t0
    ntok = data.get("usage", {}).get("completion_tokens", MAX_TOKENS)
    return dt, ntok


def bench_1way(base: str, runs: int) -> list[float]:
    rates: list[float] = []
    for _ in range(runs):
        dt, ntok = decode_once(base)
        rates.append(ntok / dt)
    return rates


def bench_8way(base: str, runs: int) -> list[float]:
    walls: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: decode_once(base), range(8)))
        dt = time.perf_counter() - t0
        walls.append(8 * MAX_TOKENS / dt)
    return walls


def fmt(rates: list[float]) -> str:
    if not rates:
        return "n/a"
    med = statistics.median(rates)
    lo = min(rates)
    hi = max(rates)
    return f"median {med:.2f}  (min {lo:.2f} / max {hi:.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--runs-1way", type=int, default=9)
    ap.add_argument("--runs-8way", type=int, default=5)
    args = ap.parse_args()

    # warmup
    decode_once(args.base)

    one = bench_1way(args.base, args.runs_1way)
    print(f"1-way 128: {fmt(one)} tok/s  ({len(one)} runs)")

    eight = bench_8way(args.base, args.runs_8way)
    print(f"8-way 128: {fmt(eight)} tok/s  ({len(eight)} runs)")


if __name__ == "__main__":
    main()
