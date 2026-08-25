#!/usr/bin/env python3
"""Concurrency benchmark for the DeepSeek-V4-Flash-0731 serve. Stdlib only.

Reports per-stream and aggregate decode tok/s at each concurrency level,
one warm pass first. Mirrors the field-notes harness
(dgx-spark-deepseek-v4-flash-0731/scripts/bench.py) without aiohttp.

Two harnesses:

  completions (default, our standard): France greedy, temp 0.
    python3 bench-concurrency.py --levels 1 6 16 32

  chat (golden / GOLDEN.md numbers): one shared BST coding prompt, temp 0.7.
    python3 bench-concurrency.py --chat --max-tokens 128 --levels 1 3 5 6

Run on spark1 against 127.0.0.1:8000 (never from the laptop; latency skews).
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

GOLDEN_PROMPT = (
    "Write a Python binary search tree with insert, delete, and inorder "
    "traversal; explain each method."
)
FRANCE_PROMPT = "The capital of France is"


def one_request(base: str, model: str, chat: bool, prompt: str,
                max_tokens: int, temp: float):
    if chat:
        body = {"model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temp}
        url = f"{base}/chat/completions"
    else:
        body = {"model": model, "prompt": prompt,
                "max_tokens": max_tokens, "temperature": temp}
        url = f"{base}/completions"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    toks = d.get("usage", {}).get("completion_tokens", 0)
    return toks, time.time() - t0


def sweep(base, model, chat, prompt, max_tokens, temp, c):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(
            lambda _: one_request(base, model, chat, prompt, max_tokens, temp),
            range(c)))
    wall = time.time() - t0
    toks = sum(x for x, _ in res)
    per = toks / wall / c if c else 0.0
    print(f"c{c:<3} per-stream={per:6.1f} tok/s  agg={toks / wall:7.1f}  "
          f"wall={wall:5.1f}s  toks={toks}", flush=True)
    return toks / wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--chat", action="store_true",
                    help="golden harness: chat completions + BST coding prompt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temp", type=float, default=None,
                    help="chat harness: default 0.7 (golden); completions: "
                         "always 0.0")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 6, 16, 32])
    ap.add_argument("--no-warm", action="store_true")
    args = ap.parse_args()

    prompt = args.prompt or (GOLDEN_PROMPT if args.chat else FRANCE_PROMPT)
    # Chat harness defaults to the golden temperature 0.7; completions is
    # always greedy (temp 0) regardless of --temp.
    temp = (args.temp if args.temp is not None else 0.7) if args.chat else 0.0
    print(f"mode={'chat' if args.chat else 'completions'} prompt={prompt[:60]!r} "
          f"max_tokens={args.max_tokens} temp={temp}", flush=True)
    if not args.no_warm:
        sweep(args.base, args.model, args.chat, prompt,
              args.max_tokens, temp, 1)
    for c in args.levels:
        sweep(args.base, args.model, args.chat, prompt,
              args.max_tokens, temp, c)


if __name__ == "__main__":
    main()
