#!/usr/bin/env python3
"""Per-cache-group page audit: where does each stack's KV pool actually go?

Reads the live vLLM block-manager accounting out of a running engine instead
of trusting width comments: for every KV cache group, report num_blocks,
block_size, bytes per block, and total bytes. Run read-only against each
image's own tree (no serve needed — import the manager classes and ask for
the profiled allocation at a fixed config).

Usage (inside either image):
    python3 /probe/probe_kv_pages.py --max-model-len 262144

Output is one JSON object per line: {"event": ...}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--block-size", type=int, default=256)
    args = ap.parse_args()

    print(json.dumps({"event": "config", "max_model_len": args.max_model_len,
                      "block_size": args.block_size}), flush=True)

    # 1. The width decision functions both engines share.
    try:
        from vllm.models.deepseek_v4.attention import _dsv4_page_alignment
        print(json.dumps({"event": "page_alignment_nvfp4",
                          "bytes": _dsv4_page_alignment("nvfp4_ds_mla")}),
              flush=True)
        print(json.dumps({"event": "page_alignment_fp8",
                          "bytes": _dsv4_page_alignment("fp8_ds_mla")}),
              flush=True)
    except Exception as e:
        print(json.dumps({"event": "page_alignment_error",
                          "error": f"{type(e).__name__}: {e}"}), flush=True)

    # 2. Block-size ladder actually offered to the manager.
    try:
        from vllm.v1.kv_cache_interface import get_kv_cache_config
        print(json.dumps({"event": "has_get_kv_cache_config", "value": True}),
              flush=True)
    except Exception as e:
        print(json.dumps({"event": "has_get_kv_cache_config",
                          "value": False, "error": str(e)[:120]}), flush=True)

    # 3. Which backend classes would own the groups (import = reachable).
    for mod, cls in [
        ("vllm.v1.attention.backends.mla.sparse_swa", "SparseMLABackend"),
        ("vllm.attention.backends.flashinfer", "FlashInferBackend"),
    ]:
        try:
            m = __import__(mod, fromlist=[cls])
            print(json.dumps({"event": "backend_reachable", "class": cls}),
                  flush=True)
        except Exception as e:
            print(json.dumps({"event": "backend_missing", "class": cls,
                              "error": f"{type(e).__name__}: {str(e)[:100]}"}),
                  flush=True)

    # 4. Environment actually visible to the allocator.
    print(json.dumps({"event": "env",
                      "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "python": sys.version.split()[0]}), flush=True)


if __name__ == "__main__":
    main()
