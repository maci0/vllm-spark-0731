#!/usr/bin/env python3
"""Cosine TileLang hc_head / mHC post / small-FMA fused path vs torch.

France prefill is 5 tokens (TileLang use_small_fma) and already has a 96-way
tied logit plateau, so this path is on the live residual stack.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


HIDDEN = 4096
HC_MULT = 4
HC_DIM = HC_MULT * HIDDEN
HC_MULT3 = HC_MULT * 2 + HC_MULT * HC_MULT  # 24
RMS_EPS = 1e-6
HC_EPS = 1e-6
SINKHORN_EPS = 1e-6
POST_ALPHA = 2.0
SINKHORN_ITERS = 20


def cosine(a: torch.Tensor, b: torch.Tensor, name: str) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    cos = F.cosine_similarity(af, bf, dim=0).item()
    mad = (af - bf).abs().max().item()
    a_nan = torch.isnan(af).any().item()
    b_nan = torch.isnan(bf).any().item()
    print(
        f"{name}: cosine={cos:.8f} max_abs={mad:.6g} "
        f"a_rms={af.pow(2).mean().sqrt().item():.5g} "
        f"b_rms={bf.pow(2).mean().sqrt().item():.5g} nan={a_nan}/{b_nan}",
        flush=True,
    )
    return cos


def hc_head_torch(hs, fn, scale, base, rms_eps, hc_eps):
    t, c, h = hs.shape
    x = hs.reshape(t, c * h).float()
    mixes = x @ fn.t()
    rsqrt = torch.rsqrt(x.square().sum(-1, keepdim=True) / (c * h) + rms_eps)
    pre = torch.sigmoid(mixes * rsqrt * scale[0] + base) + hc_eps
    out = (pre.unsqueeze(-1) * hs.float()).sum(1)
    return out.to(torch.bfloat16), pre


def test_hc_head() -> None:
    from vllm.model_executor.kernels.mhc.tilelang import (
        hc_head_fused_kernel_tilelang,
    )

    torch.manual_seed(0)
    device = "cuda"
    fn = torch.randn(HC_MULT, HC_DIM, device=device, dtype=torch.float32) * 0.02
    scale = torch.tensor([0.7], device=device, dtype=torch.float32)
    base = torch.randn(HC_MULT, device=device, dtype=torch.float32) * 0.1

    print("=== hc_head TileLang vs torch ===", flush=True)
    for t in (1, 5, 32):
        hs = torch.randn(t, HC_MULT, HIDDEN, device=device, dtype=torch.bfloat16)
        out_tl = hc_head_fused_kernel_tilelang(hs, fn, scale, base, RMS_EPS, HC_EPS)
        out_th, pre = hc_head_torch(hs, fn, scale, base, RMS_EPS, HC_EPS)
        cosine(out_tl, out_th, f"hc_head T={t}")
        print(
            f"  T={t} pre_mix mean={pre.mean().item():.5g} "
            f"tl_out_rms={out_tl.float().pow(2).mean().sqrt().item():.5g}",
            flush=True,
        )

    try:
        from vllm.model_executor.kernels.mhc.triton import hc_head_reduce_triton_kernel

        hs = torch.randn(5, HC_MULT, HIDDEN, device=device, dtype=torch.bfloat16)
        out_tr = torch.empty(5, HIDDEN, device=device, dtype=torch.bfloat16)
        hc_head_reduce_triton_kernel(
            hs, fn, scale, base, out_tr, HIDDEN, RMS_EPS, HC_EPS, HC_MULT
        )
        out_th, _ = hc_head_torch(hs, fn, scale, base, RMS_EPS, HC_EPS)
        cosine(out_tr, out_th, "hc_head triton T=5")
    except Exception as e:
        print("triton hc_head skip", type(e).__name__, e, flush=True)


def test_mhc_post() -> None:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    torch.manual_seed(1)
    device = "cuda"
    print("=== mhc_post TileLang vs torch ===", flush=True)
    for t in (1, 5, 32):
        residual = torch.randn(t, HC_MULT, HIDDEN, device=device, dtype=torch.bfloat16)
        x = torch.randn(t, HIDDEN, device=device, dtype=torch.bfloat16)
        post = torch.rand(t, HC_MULT, 1, device=device, dtype=torch.float32)
        comb = torch.softmax(
            torch.randn(t, HC_MULT, HC_MULT, device=device, dtype=torch.float32),
            dim=-1,
        )
        out_tl = mhc_post_tilelang(x, residual, post, comb)
        out_th = mhc_post_torch(x, residual, post, comb)
        cosine(out_tl, out_th, f"mhc_post T={t}")


def test_fused_post_pre() -> None:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

    torch.manual_seed(2)
    device = "cuda"
    fn = torch.randn(HC_MULT3, HC_DIM, device=device, dtype=torch.float32) * 0.01
    hc_scale = torch.tensor([0.5, 0.25, 0.1], device=device, dtype=torch.float32)
    hc_base = torch.randn(HC_MULT3, device=device, dtype=torch.float32) * 0.1
    nw = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)

    print("=== mhc_fused_post_pre TileLang vs torch post+pre ===", flush=True)
    for t in (1, 5, 8, 16, 32):
        residual = torch.randn(t, HC_MULT, HIDDEN, device=device, dtype=torch.bfloat16)
        x = torch.randn(t, HIDDEN, device=device, dtype=torch.bfloat16)
        post = torch.rand(t, HC_MULT, 1, device=device, dtype=torch.float32)
        comb = torch.softmax(
            torch.randn(t, HC_MULT, HC_MULT, device=device, dtype=torch.float32),
            dim=-1,
        )
        res_tl, post_tl, comb_tl, lin_tl = mhc_fused_post_pre_tilelang(
            x,
            residual,
            post,
            comb,
            fn,
            hc_scale,
            hc_base,
            RMS_EPS,
            HC_EPS,
            SINKHORN_EPS,
            POST_ALPHA,
            SINKHORN_ITERS,
            norm_weight=nw,
            norm_eps=RMS_EPS,
        )
        res_th = mhc_post_torch(x, residual, post, comb)
        post_th, comb_th, lin_th = mhc_pre_torch(
            res_th,
            fn,
            hc_scale,
            hc_base,
            RMS_EPS,
            HC_EPS,
            SINKHORN_EPS,
            POST_ALPHA,
            SINKHORN_ITERS,
        )
        xf = lin_th.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        lin_ref = (xf * rms * nw.float()).to(torch.bfloat16)
        path = "small_fma" if t <= 16 else "large"
        print(f"--- T={t} ({path}) ---", flush=True)
        cosine(res_tl, res_th, "residual")
        cosine(post_tl, post_th, "post_mix")
        cosine(comb_tl, comb_th, "comb_mix")
        cosine(lin_tl, lin_ref, "layer_input_fused_norm")


def main() -> int:
    torch.cuda.init()
    print("device", torch.cuda.get_device_name(0), flush=True)
    from vllm.platforms import current_platform
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tlk

    print(
        "capability",
        current_platform.get_device_capability(),
        "ENABLE_PDL",
        tlk.ENABLE_PDL,
        flush=True,
    )
    test_hc_head()
    test_mhc_post()
    test_fused_post_pre()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
