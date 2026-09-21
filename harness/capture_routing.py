#!/usr/bin/env python3
"""Capture the router's expert choices from a live arm.

The expert-bytes attribution says we stream ~1.202 GB per layer where the reference
streams ~0.916 GB. Both mechanisms that were cheap to test are closed: the kernel
does skip untouched experts and runs at 88 % of the streaming bound, and both arms
run EP = 1 so expert parallelism is not it. What is left is *which* experts the
router picks and how many rows land per step - unmeasured until now.

vLLM can report it. `--enable-return-routed-experts` (an EngineArgs/ModelConfig
field, off by default) makes the chat-completion response carry
`routed_experts: str`, a base64-encoded numpy array, documented at
`vllm/entrypoints/openai/chat_completion/protocol.py:119-124`:

    Decode:
      np.load(io.BytesIO(base64.b64decode(s)))

So: serve with the flag, send one request at a level from the protocol, and count
the distinct experts the router actually touched. That turns
experts-touched-per-rank from an inference into a measurement.

Usage, against a served arm:

    python3 capture_routing.py [--base http://127.0.0.1:8000] [--seed 1234]
                              [--max-tokens 512] [--concurrency 1]

Writes JSON lines to stdout.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request
from collections import Counter

PROMPT = (
    "Write a Python binary search tree with insert, delete, and in-order traversal. "
    "Include type hints and a short docstring for each method."
)


def post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def decode_routed(encoded: str):
    import numpy as np

    raw = base64.b64decode(encoded)
    return np.load(io.BytesIO(raw))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "seed": args.seed,
        "chat_template_kwargs": {"thinking": False},
        # one request per sequence, so a level-c run has c sequences in flight
        "n": 1,
    }

    try:
        resp = post(f"{args.base}/v1/chat/completions", payload)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"event": "error", "type": type(exc).__name__,
                          "message": str(exc)[:400]}), flush=True)
        return 1

    choices = resp.get("choices", [])
    if not choices:
        print(json.dumps({"event": "error", "message": "no choices in response",
                          "keys": sorted(resp.keys())}), flush=True)
        return 1

    encoded = choices[0].get("routed_experts")
    if encoded is None:
        print(json.dumps({
            "event": "no_routing_data",
            "note": "field absent, so either the server did not start with "
                    "--enable-return-routed-experts, or the request aborted before a forward pass",
            "choice_keys": sorted(choices[0].keys()),
        }), flush=True)
        return 1

    arr = decode_routed(encoded)
    flat = arr.reshape(-1) if hasattr(arr, "reshape") else arr
    counts = Counter(int(x) for x in flat.tolist())
    total = sum(counts.values())
    distinct = len(counts)
    top = counts.most_common(10)
    print(json.dumps({
        "event": "routing",
        "array_shape": list(getattr(arr, "shape", [])),
        "array_dtype": str(getattr(arr, "dtype", "?")),
        "total_expert_slots": total,
        "distinct_experts_touched": distinct,
        "max_expert_load": top[0][1] if top else 0,
        "top_experts": top,
        "singleton_experts": sum(1 for _, c in counts.items() if c == 1),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
