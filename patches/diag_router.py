#!/usr/bin/env python3
"""Cosine DeepSeek-V4 routers on SM12x vs torch sqrtsoftplus+hash."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_ids(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    af = a.detach().cpu()
    bf = b.detach().cpu()
    match = (af.to(torch.int64) == bf.to(torch.int64)).float().mean().item()
    print(f"{name}: id_match={match:.4f} a={af[0].tolist() if af.ndim==2 else af.tolist()[:8]}", flush=True)


def cosine_w(a: torch.Tensor, b: torch.Tensor, name: str) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    if af.numel() == 0:
        print(f"{name}: empty", flush=True)
        return 1.0
    cos = F.cosine_similarity(af, bf, dim=0).item()
    mad = (af - bf).abs().max().item()
    print(f"{name}: cosine={cos:.8f} max_abs={mad:.6g}", flush=True)
    return cos


def torch_topk_softplus_sqrt(
    gating_output,
    topk,
    renormalize,
    routed_scaling_factor,
    e_score_correction_bias=None,
    input_ids=None,
    hash_indices_table=None,
):
    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores
    if hash_indices_table is not None:
        topk_ids = hash_indices_table[input_ids.long()]
    else:
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)[1]
    topk_weights = original_scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def main() -> int:
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import (
        can_use_dsv4_topk,
        dsv4_topk,
    )
    import torch as _

    print("device", torch.cuda.get_device_name(0), flush=True)
    print("pdl", current_platform.is_arch_support_pdl(), flush=True)
    import torch as t2
    print("moe ops", [x for x in dir(t2.ops._moe_C) if "topk" in x.lower() or "soft" in x.lower()], flush=True)

    torch.manual_seed(0)
    device = "cuda"
    T, E, K = 5, 256, 6
    hidden = torch.randn(T, 4096, device=device, dtype=torch.bfloat16)
    gate = torch.randn(T, E, device=device, dtype=torch.float32)
    bias = torch.randn(E, device=device, dtype=torch.float32)

    print("=== dsv4_topk (layers 3+) ===", flush=True)
    print(
        "can_use",
        can_use_dsv4_topk(gate, bias, K, True, torch.int32),
        flush=True,
    )
    w_ref, id_ref = torch_topk_softplus_sqrt(gate, K, True, 1.5, bias)
    w, ids = dsv4_topk(gate, bias, torch.int32, 1.5)
    cosine_ids(ids, id_ref, "dsv4 ids")
    cosine_w(w, w_ref, "dsv4 weights")

    print("=== fused_topk_bias with bias (should take dsv4) ===", flush=True)
    w2, ids2 = fused_topk_bias(
        hidden,
        gate,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=bias,
        topk=K,
        renormalize=True,
        routed_scaling_factor=1.5,
    )
    cosine_ids(ids2, id_ref, "fused-bias ids")
    cosine_w(w2, w_ref, "fused-bias weights")

    print("=== hash MoE (layers 0-2) ===", flush=True)
    vocab = 1024
    table = torch.stack(
        [torch.randperm(E, device=device)[:K] for _ in range(vocab)]
    ).to(dtype=torch.int32)
    input_ids = torch.tensor([6102, 294, 8760, 344, 12], device=device, dtype=torch.int32) % vocab
    w_href, id_href = torch_topk_softplus_sqrt(
        gate, K, True, 1.5, None, input_ids, table
    )
    w_h, id_h = fused_topk_bias(
        hidden,
        gate,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        topk=K,
        renormalize=True,
        input_tokens=input_ids,
        hash_indices_table=table,
        routed_scaling_factor=1.5,
    )
    cosine_ids(id_h, id_href, "hash ids")
    cosine_w(w_h, w_href, "hash weights")
    print("hash ids live", id_h.tolist(), "ref", id_href.tolist(), flush=True)
    print("hash w live", w_h[0].tolist(), "ref", w_href[0].tolist(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
