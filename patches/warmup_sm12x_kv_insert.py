#!/usr/bin/env python3
"""Compile SM12x Triton KV-insert kernels while the GPU is empty.

First serve call happens after 79 GiB weights + ~24 GiB KV. spark2 has no
swap, so Triton JIT at that peak is SIGKILL (container exit 137).
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
    xpu_qnorm_rope_kv_fp8_insert,
)


def main() -> int:
    torch.cuda.init()
    q = torch.randn(4, 32, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(8, 64, 584, dtype=torch.uint8, device="cuda")
    slot = torch.arange(4, dtype=torch.int64, device="cuda")
    pos = torch.arange(4, dtype=torch.int64, device="cuda")
    cos_sin = torch.randn(4096, 64, dtype=torch.float32, device="cuda")
    xpu_qnorm_rope_kv_fp8_insert(q, kv, cache, slot, pos, cos_sin, 1e-6, 64)
    torch.cuda.synchronize()
    print("ok sm12x triton kv-insert compiled for block_size=64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
