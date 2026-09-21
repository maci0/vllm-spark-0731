#!/usr/bin/env python3
"""What token width does each engine's `nvfp4_ds_mla` actually allocate?

Goal item 1 assumes our `nvfp4_ds_mla` is an envelope alias for the fp8 page and
that the reference has a real NVFP4 writer worth porting. This reads the answer
out of the reference image's own source instead of assuming it: the width
functions that decide the KV block size for each dtype.

Read-only against the reference image:

    docker run --rm --entrypoint python3 ghcr.io/anemll/dspark-vllm-gx10:0.1.1 \
        /probe/probe_nvfp4_kv_width.py

    docker run --rm --entrypoint python3 -v ~/bench:/probe \
        ghcr.io/anemll/dspark-vllm-gx10:0.1.1 /probe/probe_nvfp4_kv_width.py
"""

from __future__ import annotations

import glob
import json
import os
import re

ROOTS = (
    "/usr/local/lib/python3.12/dist-packages/vllm",
    "/vllm",
    "/opt/vllm",
)
# Lines that decide a KV token width: a returned byte count or a literal width.
WIDTH_RE = re.compile(r"(return\s+.*\b(\d{3,4})\b|=\s*(\d{3,4})\b)")
WANT = ("nvfp4_ds_mla", "fp8_ds_mla")


def main():
    root = next((r for r in ROOTS if os.path.isdir(r)), None)
    if root is None:
        raise SystemExit("no vllm package found")
    print(json.dumps({"event": "root", "path": root}), flush=True)

    files = []
    for path in glob.glob(f"{root}/**/*.py", recursive=True):
        try:
            text = open(path, errors="ignore").read()
        except OSError:
            continue
        if all(w in text for w in WANT):
            files.append(path)

    print(json.dumps({"event": "files_naming_both_dtypes",
                      "count": len(files),
                      "paths": [p.replace(root, "vllm") for p in sorted(files)]}),
          flush=True)

    widths = {}
    for path in sorted(files):
        lines = open(path, errors="ignore").read().splitlines()
        rel = path.replace(root, "vllm")
        for i, line in enumerate(lines):
            if not any(w in line for w in WANT):
                continue
            # a width decision usually sits within the next few lines
            block = "\n".join(lines[i:i + 6])
            m = WIDTH_RE.search(block)
            if not m:
                continue
            value = int(next(g for g in m.groups()[1:] if g))
            if not 100 <= value <= 2000:
                continue
            comment = ""
            for back in range(1, 4):
                if i - back >= 0 and "#" in lines[i - back]:
                    comment = lines[i - back].strip()
                    break
            key = f"{rel}:{i + 1}"
            widths[key] = {"value": value, "comment": comment[:160]}
            break

    print(json.dumps({"event": "token_widths", "found": widths}, indent=1), flush=True)

    # The arithmetic the reference states for its own envelope.
    print(json.dumps({"event": "envelope_arithmetic",
                      "stated": "448 NoPE + 128 RoPE + 8 fp8 scale",
                      "sum": 448 + 128 + 8}), flush=True)


if __name__ == "__main__":
    main()
