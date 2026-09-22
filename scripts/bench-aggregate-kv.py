#!/usr/bin/env python3
"""2M-token aggregate load: N concurrent streams, each prefilling P tokens
with N*P = 2M, then a short decode. Proves the pool serves 2M live KV with
gates passing — no request over the 1M native per-request cap.

Usage (on spark1, engine healthy):
    python3 scripts/bench-aggregate-kv.py --streams 4 --prompt-tokens 524288 --decode-tokens 64

Long prompts are built by repeating a neutral paragraph to the target token
count (measured by the served tokenizer length endpoint is overkill; estimate
4 chars/token and verify usage.prompt_tokens afterwards).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PARA = ("Write a Python binary search tree with insert, delete, and inorder "
        "traversal; explain each method. ")


def one_stream(base, model, prompt, decode_tokens, seed, out):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": decode_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False}, "seed": seed}
    req = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read().decode())
        u = d.get("usage", {})
        out.append((u.get("prompt_tokens", -1), u.get("completion_tokens", -1),
                    time.time() - t0, "ok"))
    except Exception as e:
        out.append((-1, -1, time.time() - t0, f"{type(e).__name__}: {e}"[:120]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=524288)
    ap.add_argument("--decode-tokens", type=int, default=64)
    args = ap.parse_args()

    # ~4 chars per token for English prose.
    reps = args.prompt_tokens * 4 // len(PARA) + 1
    prompt = (PARA * reps)[:args.prompt_tokens * 4]
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.streams) as ex:
        list(ex.map(lambda i: one_stream(args.base, args.model, prompt,
                                         args.decode_tokens, 1234 + i,
                                         results),
                    range(args.streams)))
    wall = time.time() - t0
    live = sum(p for p, _, _, s in results if s == "ok")
    print(f"streams={args.streams} live_prompt_tokens={live} "
          f"target={args.streams * args.prompt_tokens} wall={wall:.0f}s")
    for i, (p, c, w, s) in enumerate(results):
        print(f"  s{i}: prompt={p} completion={c} wall={w:.0f}s status={s}")


if __name__ == "__main__":
    main()
