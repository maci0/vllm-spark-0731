#!/usr/bin/env python3
"""Load only embed + lm_head and score 'The capital of France is' without a transformer."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = Path("/models/ds4-flash-0731")
INDEX = json.loads((MODEL / "model.safetensors.index.json").read_text())
WMAP = INDEX["weight_map"]


def shard_path(key: str) -> Path:
    return MODEL / WMAP[key]


def load(key: str, device: str = "cpu") -> torch.Tensor:
    path = shard_path(key)
    with safe_open(str(path), framework="pt", device=device) as f:
        return f.get_tensor(key)


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 128x128 block scale: scale byte e -> 2^(e-127), 0 -> 0."""
    w = weight.view(torch.float8_e4m3fn) if weight.dtype == torch.uint8 else weight
    if scale.dtype == torch.uint8:
        s = torch.where(
            scale == 0,
            torch.zeros((), dtype=torch.float32, device=scale.device),
            torch.exp2(scale.float() - 127),
        )
    else:
        s = scale.float()
    block = 128
    n, k = w.shape
    w_blocks = w.float().view(n // block, block, k // block, block)
    s = s.view(n // block, 1, k // block, 1)
    return (w_blocks * s).reshape(n, k)


def main() -> int:
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    ids = tok.encode("The capital of France is", add_special_tokens=False)
    print("prompt ids", ids, [tok.decode([i]) for i in ids], flush=True)

    head_keys = [k for k in WMAP if k.startswith("head.")]
    embed_keys = [k for k in WMAP if k.startswith("embed.")]
    print("head keys", head_keys, flush=True)
    print("embed keys", embed_keys, flush=True)

    device = "cuda"
    embed = load("embed.weight", "cpu")
    head = load("head.weight", "cpu")
    print("embed", tuple(embed.shape), embed.dtype, flush=True)
    print("head", tuple(head.shape), head.dtype, flush=True)

    scale_e = load("embed.weight_scale_inv", "cpu") if "embed.weight_scale_inv" in WMAP else None
    scale_h = load("head.weight_scale_inv", "cpu") if "head.weight_scale_inv" in WMAP else None
    # some dumps use weight_scale
    for alt in ("embed.weight_scale", "embed.weight_scale_inv"):
        if alt in WMAP:
            print("found", alt, flush=True)
    for alt in ("head.weight_scale", "head.weight_scale_inv"):
        if alt in WMAP:
            print("found", alt, flush=True)
            scale_h = load(alt, "cpu")
            print("head scale", tuple(scale_h.shape), scale_h.dtype, flush=True)
    for alt in ("embed.weight_scale", "embed.weight_scale_inv"):
        if alt in WMAP:
            scale_e = load(alt, "cpu")
            print("embed scale", tuple(scale_e.shape), scale_e.dtype, flush=True)

    if scale_e is not None:
        embed_f = dequant_fp8(embed, scale_e)
    elif embed.dtype == torch.bfloat16:
        embed_f = embed.float()
    else:
        embed_f = embed.float()

    if scale_h is not None:
        head_f = dequant_fp8(head, scale_h)
    elif head.dtype == torch.bfloat16:
        head_f = head.float()
    else:
        head_f = head.float()

    print("embed_f", tuple(embed_f.shape), embed_f.dtype, "rms", embed_f.pow(2).mean().sqrt().item(), flush=True)
    print("head_f", tuple(head_f.shape), head_f.dtype, "rms", head_f.pow(2).mean().sqrt().item(), flush=True)

    hidden = embed_f[ids[-1]]
    logits = head_f @ hidden
    logp = F.log_softmax(logits, dim=0)
    topv, topi = torch.topk(logp, 16)
    print("last-token embed @ lm_head (no transformer):", flush=True)
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  {i:6d} {v:8.4f} {tok.decode([i])!r}", flush=True)

    # how many tokens share the max within 1e-5
    mx = logp.max()
    n_tie = int((logp > mx - 1e-4).sum().item())
    print("n_near_max", n_tie, "max", mx.item(), "min", logp.min().item(), flush=True)
    print("logp at 82357 的超", logp[82357].item(), "at 11111 Paris", logp[11111].item(), "at 344 is", logp[344].item(), flush=True)

    # also try mean-pool of prompt embeds
    pooled = embed_f[ids].mean(0)
    logits2 = head_f @ pooled
    logp2 = F.log_softmax(logits2, dim=0)
    topv, topi = torch.topk(logp2, 8)
    print("mean-pool prompt embeds @ lm_head:", flush=True)
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  {i:6d} {v:8.4f} {tok.decode([i])!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
