#!/usr/bin/env python3
"""2x1M multi-user probe: two 1M-token requests, SEQUENTIAL admission, then
concurrent short decode with gate check.

Why sequential: 4x512k concurrent prefill OOM-rebooted spark1 twice. Prefill
scratch (~3-4 GiB per 512k stream, ~6-8 GiB per 1M stream implied) is the
wall, not pool residency. Admit stream 0 fully, wait for completion, admit
stream 1, then decode a short prompt on both concurrently and check gates.

Usage (on spark1, engine healthy):
    python3 scripts/bench-2x1m.py --prompt-tokens 1048576 --decode-tokens 64
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

PARA = ("Write a Python binary search tree with insert, delete, and inorder "
        "traversal; explain each method. ")
FRANCE = "The capital of France is"
MUL = "9x8="


def post(base, model, prompt, decode_tokens, seed, temp=0.0):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": decode_tokens, "temperature": temp,
            "chat_template_kwargs": {"thinking": False}, "seed": seed}
    req = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read().decode())
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--prompt-tokens", type=int, default=1048576)
    ap.add_argument("--decode-tokens", type=int, default=64)
    args = ap.parse_args()

    reps = args.prompt_tokens * 4 // len(PARA) + 1
    prompt = (PARA * reps)[:args.prompt_tokens * 4]
    print(f"prompt chars={len(prompt)} (~{len(prompt)//4} tokens)", flush=True)

    for i in range(2):
        print(f"--- admitting stream {i} ---", flush=True)
        t0 = time.time()
        try:
            d, w = post(args.base, args.model, prompt, 16, 1234 + i)
            u = d.get("usage", {})
            print(f"stream {i} admitted: prompt={u.get('prompt_tokens')} "
                  f"completion={u.get('completion_tokens')} wall={w:.0f}s",
                  flush=True)
        except Exception as e:
            print(f"stream {i} FAILED: {type(e).__name__}: {e}"[:200],
                  flush=True)
            return

    print("--- concurrent decode + gates on both streams ---", flush=True)
    for i, gate, want in [(0, FRANCE, "Paris"), (1, MUL, "72")]:
        try:
            d, w = post(args.base, args.model, gate, 16, 4321 + i)
            txt = "".join(
                c.get("message", {}).get("content", "")
                for c in d.get("choices", []))
            ok = want in txt
            print(f"stream {i} gate {'PASS' if ok else 'FAIL'}: "
                  f"{txt[:60]!r} wall={w:.0f}s", flush=True)
        except Exception as e:
            print(f"stream {i} gate ERROR: {type(e).__name__}: {e}"[:160],
                  flush=True)


if __name__ == "__main__":
    main()
