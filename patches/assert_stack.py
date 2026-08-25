#!/usr/bin/env python3
"""Refuse illegal KV/attention mixes. DSpark is required on every 0.28 stack."""

from __future__ import annotations

import argparse
import os
import sys

DSPARK_K = 5
ALLOWED_KV = ("fp8_ds_mla", "nvfp4_ds_mla")
ALLOWED_MOE = ("b12x", "flashinfer_b12x")
B12X_ATTN = "B12X_MLA_SPARSE"
SM120_ATTN = "FLASHINFER_MLA_SPARSE_DSV4"


def check(
    kv_cache_dtype: str,
    attention_backend: str,
    moe_backend: str,
    spec_method: str = "dspark",
    num_speculative_tokens: int | str = DSPARK_K,
) -> str:
    kv = (kv_cache_dtype or "").strip()
    attn = (attention_backend or "").strip()
    moe = (moe_backend or "").strip()
    spec = (spec_method or "").strip()
    try:
        k = int(num_speculative_tokens)
    except (TypeError, ValueError):
        raise SystemExit(
            f"num_speculative_tokens={num_speculative_tokens!r} is not an int"
        ) from None

    if spec != "dspark":
        raise SystemExit(
            f"refusing spec_method={spec!r}: this recipe serves DSpark only "
            "(method=dspark, k=5, locked to 0731 n_predict=5)"
        )
    if k != DSPARK_K:
        raise SystemExit(
            f"refusing num_speculative_tokens={k}: 0731 DSpark k is locked at {DSPARK_K}"
        )

    if kv not in ALLOWED_KV:
        raise SystemExit(
            f"unknown kv={kv!r}. expected one of {ALLOWED_KV}"
        )

    if attn == B12X_ATTN:
        # Overlay B12X_MLA_SPARSE is the DSV4 584-byte kernel path, so
        # nvfp4_ds_mla (same envelope) is legal. Real 432/368 NVFP4 is not.
        return "nvfp4-b12x" if kv == "nvfp4_ds_mla" else "fp8-b12x"

    if attn == SM120_ATTN:
        return "nvfp4" if kv == "nvfp4_ds_mla" else "fp8"

    if attn:
        raise SystemExit(
            f"refusing attention_backend={attn!r}. "
            f"Use {B12X_ATTN} or {SM120_ATTN} (SM12x), or leave empty."
        )

    if moe and moe not in ALLOWED_MOE:
        print(f"warn: moe_backend={moe!r}", file=sys.stderr)

    if kv == "nvfp4_ds_mla":
        return "nvfp4"
    return "fp8"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", default=os.environ.get("KV_CACHE_DTYPE", ""))
    parser.add_argument("--attn", default=os.environ.get("ATTENTION_BACKEND", ""))
    parser.add_argument("--moe", default=os.environ.get("MOE_BACKEND", ""))
    parser.add_argument("--spec", default=os.environ.get("SPEC_METHOD", "dspark"))
    parser.add_argument(
        "--k",
        default=os.environ.get("NUM_SPECULATIVE_TOKENS", str(DSPARK_K)),
    )
    args = parser.parse_args()
    stack = check(args.kv, args.attn, args.moe, args.spec, args.k)
    print(stack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
