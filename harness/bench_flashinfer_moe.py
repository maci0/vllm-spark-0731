#!/usr/bin/env python3
"""FlashInfer `b12x_fused_moe` alone, so the two wheels can be compared directly.

The MoE head-to-head measured our image's FlashInfer 0.7.0 build of
`b12x_fused_moe` at ~11x our own b12x MoE, across 64/128/256 local experts, which
refutes the sharding explanation but leaves a contradiction: that cost cannot fit
inside the reference's 38.65 ms step for 43 layers.

The reference image runs FlashInfer **0.6.15**, ours **0.7.0**. If the same-named
kernel differs between those wheels, the head-to-head measured something the
comparator does not run. This script depends only on FlashInfer, so it runs in
both images unchanged.

    docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench \\
        vllm-spark-0731:main-029-proto-ccompile /bench/bench_flashinfer_moe.py
    docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench \\
        ghcr.io/anemll/dspark-vllm-gx10:0.1.1 /bench/bench_flashinfer_moe.py
"""

from __future__ import annotations

import json
import sys

import torch

TOPK = 6
HIDDEN = 4096
INTER = 2048
TOKENS = 48
SF_VEC = 16
E_LOCALS = (256, 128, 64)
REPS = 3
ITERS = 30
WARMUP = 3


def build(num_experts: int, device: str):
    from flashinfer import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    torch.manual_seed(0)
    one = torch.ones(1, dtype=torch.float32, device=device)

    def pack(rows: int, cols: int):
        packed, scales = [], []
        for _ in range(num_experts):
            w = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
            w = w / w.abs().amax().clamp_min(1e-6)
            q, sf = fp4_quantize(w, global_scale=one, sf_vec_size=SF_VEC,
                                 sf_use_ue8m0=False)
            packed.append(q)
            scales.append(sf)
        p = torch.stack(packed)
        s = torch.stack(scales)
        e, n, k_sf = s.shape
        mma = convert_sf_to_mma_layout(
            s.reshape(e * n, k_sf), m=n, k=k_sf * SF_VEC, num_groups=e
        )
        return p, mma.contiguous()

    w1, s1 = pack(2 * INTER, HIDDEN)
    w2, s2 = pack(HIDDEN, INTER)
    return w1, s1, w2, s2


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    import flashinfer
    from flashinfer.fused_moe import b12x_fused_moe

    print(json.dumps({"event": "version", "flashinfer": flashinfer.__version__,
                      "kernel": getattr(b12x_fused_moe, "__module__", "?")}), flush=True)

    for num_experts in E_LOCALS:
        w1, s1, w2, s2 = build(num_experts, "cuda")
        torch.manual_seed(1234)
        ids = torch.stack([
            torch.randperm(num_experts, device="cuda")[:TOPK] for _ in range(TOKENS)
        ]).to(torch.int32)
        scales = torch.rand(TOKENS, TOPK, dtype=torch.float32, device="cuda")
        scales = scales / scales.sum(dim=-1, keepdim=True)
        x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
        alpha = torch.ones(num_experts, dtype=torch.float32, device="cuda")
        kw = dict(
            x=x, w1_weight=w1, w1_weight_sf=s1, w2_weight=w2, w2_weight_sf=s2,
            token_selected_experts=ids, token_final_scales=scales,
            num_experts=num_experts, top_k=TOPK, w1_alpha=alpha,
            w2_alpha=alpha.clone(),
            fc2_input_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
            input_global_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
            quant_mode="nvfp4", activation="silu",
        )
        # The signature moved between wheels (0.6.15 has no input_global_scale),
        # so pass only what this build accepts.
        import inspect as _inspect
        accepted = set(_inspect.signature(b12x_fused_moe).parameters)
        dropped = sorted(set(kw) - accepted)
        if dropped:
            print(json.dumps({"event": "dropped_kwargs", "e_local": num_experts,
                              "keys": dropped}), flush=True)
        kw = {k: v for k, v in kw.items() if k in accepted}

        for rep in range(REPS):
            try:
                out = b12x_fused_moe(**kw)
                torch.cuda.synchronize()
                for _ in range(WARMUP):
                    b12x_fused_moe(**kw)
                torch.cuda.synchronize()
                samples = []
                for _ in range(ITERS):
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record()
                    b12x_fused_moe(**kw)
                    b.record()
                    torch.cuda.synchronize()
                    samples.append(a.elapsed_time(b))
                samples.sort()
                med = samples[len(samples) // 2]
                print(json.dumps({"event": "result", "rep": rep,
                                  "e_local": num_experts,
                                  "median_ms": round(med, 4),
                                  "min_ms": round(samples[0], 4),
                                  "finite": bool(torch.isfinite(out.float()).all()),
                                  "mean_abs": round(float(out.float().abs().mean()), 4)}),
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "error", "rep": rep,
                                  "e_local": num_experts,
                                  "type": type(exc).__name__,
                                  "message": str(exc)[:300]}), flush=True)
                break
        del w1, s1, w2, s2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
