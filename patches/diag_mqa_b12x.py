#!/usr/bin/env python3
"""Cosine B12x contiguous MQA vs Python ReLU for a France-sized prefill."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def python_relu_mqa(q_fp8, w, k_fp8, k_sf, ks, ke):
    q_f = q_fp8.to(torch.float32)
    k_f = k_fp8.to(torch.float32)
    M, H, D = q_f.shape
    N = k_f.shape[0]
    scores = torch.einsum("mhd,nd->mhn", q_f, k_f)
    scores = (scores * k_sf).relu() * w.unsqueeze(-1)
    logits = scores.sum(dim=1)
    pos = torch.arange(N, device=logits.device).unsqueeze(0)
    mask = (pos >= ks.unsqueeze(1)) & (pos < ke.unsqueeze(1))
    return logits.masked_fill(~mask, float("-inf"))


def main() -> int:
    from b12x.attention.nsa_indexer._impl import (
        IndexerContiguousMetadata,
        contiguous_logits,
        supports_contiguous_logits_kernel,
    )

    torch.manual_seed(0)
    device = "cuda"
    M, H, D, N = 5, 64, 128, 5
    q = torch.randn(M, H, D, device=device).to(torch.float8_e4m3fn)
    k = torch.randn(N, D, device=device).to(torch.float8_e4m3fn)
    k_sf = torch.rand(N, device=device, dtype=torch.float32) * 0.05 + 0.01
    w = torch.rand(M, H, device=device, dtype=torch.float32)
    ks = torch.zeros(M, device=device, dtype=torch.int32)
    ke = torch.arange(1, M + 1, device=device, dtype=torch.int32)
    print(
        "supports",
        supports_contiguous_logits_kernel(
            q_fp8=q, weights=w, k_quant=k, k_scale=k_sf, k_start=ks, k_end=ke
        ),
        flush=True,
    )
    ref = python_relu_mqa(q, w, k, k_sf, ks, ke)
    for mode in (0, 1, 2):
        try:
            out = contiguous_logits(
                q_fp8=q,
                weights=w,
                kv_fp8=(k, k_sf),
                metadata=IndexerContiguousMetadata(k_start=ks, k_end=ke),
                score_mode=mode,
            )
        except Exception as e:
            print(f"mode {mode} err {type(e).__name__} {e}", flush=True)
            continue
        af, bf = out.float().flatten(), ref.float().flatten()
        fin = torch.isfinite(af)
        cos = F.cosine_similarity(af[fin], bf[fin], dim=0).item() if fin.any() else float("nan")
        print(
            f"mode {mode}: cosine={cos:.8f} max_abs={(af-bf).abs().max().item():.6g} "
            f"out_finite={int(fin.sum())}/{af.numel()} ref_last_row={ref[-1].tolist()} "
            f"out_last_row={out[-1].tolist()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
