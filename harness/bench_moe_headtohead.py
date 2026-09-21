#!/usr/bin/env python3
"""MoE kernel head-to-head: our b12x MoE against FlashInfer's fused `b12x_fused_moe`.

The comparator's recipe runs `--moe-backend flashinfer_b12x`, which dispatches to
`FlashInferB12xExperts` and its `b12x_fused_moe` kernel (dispatch, both GEMMs,
SwiGLU and topk reduction in one call). Our measured arms run `MOE_BACKEND=b12x`,
the b12x package's own MoE. ffn/MoE is where the profiler puts 65 % of a target
layer, so this is the question the remaining gap hangs on.

A first run at 256 experts on one GPU gave ours 10.07 ms against FlashInfer's
127.7 ms. That 13x cannot be believed as a statement about the comparator's
engine, because 127.7 ms for one MoE layer does not fit inside a 43-layer step
measured at 38.65 ms. The prime suspect is sharding: the served engine runs TP=2
with experts split across ranks, so each rank holds fewer experts and each held
expert owns more routed rows, while a single-GPU run with all 256 experts and 288
rows pads every expert to the static kernel's m48 tile.

So this sweeps the local expert count instead of assuming it. 288 routed rows
(48 tokens x topk 6) are spread over `E_local` experts, which is what a rank sees
under expert-parallel sharding.

**What this is and is not.** It compares throughput at equal shapes, routing and
local expert count. It is *not* a correctness comparison: the two kernels take
different scale recipes (FlashInfer's NVFP4 path uses vec-16 E4M3 scales in the
6-D MMA layout, ours uses the e8m0 K/32 grid the DSV4 checkpoint carries), so
their outputs are not expected to match. Each is checked only for running and
producing finite output of a sane magnitude.

Run inside `vllm-spark-0731:main-029-proto-ccompile`:

    docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench \\
        vllm-spark-0731:main-029-proto-ccompile /bench/bench_moe_headtohead.py
"""

from __future__ import annotations

import json
import sys

import torch

TOPK = 6
HIDDEN = 4096
INTER = 2048
TOKENS = 48          # c6 decode step at k=7, six sequences
ROUTED_ROWS = TOKENS * TOPK
SF_VEC = 16          # NVFP4 scale block; the working path is E4M3, not E8M0
E_LOCALS = (256, 128, 64)   # local experts per rank: TP=1, TP=2, TP=4 sharding
REPS = 3
ITERS = 30
WARMUP = 3


def make_weights(num_experts: int, device: str):
    """bf16 expert stack; fp4_quantize accepts fp16/bf16/e4m3 only."""
    torch.manual_seed(0)
    w13, w2 = [], []
    for _ in range(num_experts):
        a = torch.randn(2 * INTER, HIDDEN, device=device, dtype=torch.bfloat16)
        w13.append(a / a.abs().amax().clamp_min(1e-6))
        b = torch.randn(HIDDEN, INTER, device=device, dtype=torch.bfloat16)
        w2.append(b / b.abs().amax().clamp_min(1e-6))
    return w13, w2


def flashinfer_weights(w13, w2):
    """Mirrors FlashInferB12xExperts.process_weights_after_loading.

    NVFP4 vec-16 E4M3 scales; the scale stack reshaped to [E*N, K/16] and
    converted once with num_groups=E, giving the 6-D MMA tensor whose last axis
    is the group (expert) axis.
    """
    from flashinfer import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    one = torch.ones(1, dtype=torch.float32, device="cuda")

    def pack(stack):
        packed, scales = [], []
        for w2d in stack:
            q, sf = fp4_quantize(
                w2d, global_scale=one, sf_vec_size=SF_VEC, sf_use_ue8m0=False
            )
            packed.append(q)
            scales.append(sf)
        p = torch.stack(packed)
        s = torch.stack(scales)
        e, n, k_sf = s.shape
        mma = convert_sf_to_mma_layout(
            s.reshape(e * n, k_sf), m=n, k=k_sf * SF_VEC, num_groups=e
        )
        return p, mma.contiguous()

    p1, s1 = pack(w13)
    p2, s2 = pack(w2)
    return p1, s1, p2, s2


def ours_weights(w13, w2):
    """Our e8m0 K/32 grid, the contract bench_moe_clusters.py established."""
    sys.path.insert(0, "/bench")
    import bench_moe_clusters as bench

    p1, s1, p2, s2 = [], [], [], []
    for a, b in zip(w13, w2):
        q, s = bench.quantize_k32(a)
        p1.append(q)
        s1.append(s)
        q, s = bench.quantize_k32(b)
        p2.append(q)
        s2.append(s)
    return torch.stack(p1), torch.stack(s1), torch.stack(p2), torch.stack(s2)


def routing(num_experts: int, device: str):
    torch.manual_seed(1234)
    ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:TOPK] for _ in range(TOKENS)]
    ).to(torch.int32)
    w = torch.rand(TOKENS, TOPK, dtype=torch.float32, device=device)
    return ids, w / w.sum(dim=-1, keepdim=True)


def run_flashinfer(packed, ids, scales, num_experts: int, device: str):
    from flashinfer.fused_moe import b12x_fused_moe

    p1, s1, p2, s2 = packed
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
    alpha = torch.ones(num_experts, dtype=torch.float32, device=device)
    kw = dict(
        x=x, w1_weight=p1, w1_weight_sf=s1, w2_weight=p2, w2_weight_sf=s2,
        token_selected_experts=ids, token_final_scales=scales,
        num_experts=num_experts, top_k=TOPK,
        w1_alpha=alpha, w2_alpha=alpha.clone(),
        fc2_input_scale=torch.ones(1, dtype=torch.float32, device=device),
        input_global_scale=torch.ones(1, dtype=torch.float32, device=device),
        quant_mode="nvfp4", activation="silu",
    )
    out = b12x_fused_moe(**kw)
    torch.cuda.synchronize()
    return (lambda: b12x_fused_moe(**kw)), out


def run_ours(packed, ids, scales, num_experts: int, device: str):
    from b12x.moe import fused_moe as fm

    p1, s1, p2, s2 = packed
    src = fm.PackedSource(
        format=fm.PackedSourceFormat.MXFP4_E8M0_K32, w13_layout=fm.W13Layout.W31
    )
    act = fm.ActivationSpec(
        mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16
    )
    geo = fm.MoEGeometry(
        num_experts=num_experts, hidden_size=HIDDEN, intermediate_size=INTER
    )
    plan = fm.plan_weights(source=src, activation=act, geometry=geo)
    ones = torch.ones(num_experts, dtype=torch.float32, device=device)
    experts = fm.prepare_weights(
        plan=plan,
        weights=fm.PackedWeights(
            w13=p1, w2=p2, w13_block_scales=s1, w2_block_scales=s2,
            w13_global_scales=ones, w2_global_scales=ones.clone(),
        ),
    )
    execution = fm.plan_execution(
        experts=experts, capacity=fm.ExecutionCapacity(max_tokens=TOKENS, top_k=TOPK)
    )
    fm.prewarm(execution)
    spec = execution.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
    binding = fm.bind(
        execution, scratch=scratch, a=x, experts=experts,
        topk_weights=scales, topk_ids=ids,
    )
    out = fm.run(binding=binding)
    torch.cuda.synchronize()
    return (lambda: fm.run(binding=binding)), out


def timeit(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        samples.append(a.elapsed_time(b))
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def sanity(out: torch.Tensor) -> str:
    f = out.float()
    if not torch.isfinite(f).all():
        return "NOT FINITE"
    return f"finite, mean|out|={float(f.abs().mean()):.4g}"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(json.dumps({"event": "geometry", "hidden": HIDDEN, "inter": INTER,
                      "tokens": TOKENS, "topk": TOPK, "routed_rows": ROUTED_ROWS,
                      "e_locals": list(E_LOCALS)}), flush=True)

    quantizers = {"b12x_ours": ours_weights, "flashinfer_b12x": flashinfer_weights}
    runners = {"b12x_ours": run_ours, "flashinfer_b12x": run_flashinfer}

    for num_experts in E_LOCALS:
        bf_w13, bf_w2 = make_weights(num_experts, "cuda")
        ids, scales = routing(num_experts, "cuda")
        print(json.dumps({"event": "case", "e_local": num_experts,
                          "rows_per_expert": round(ROUTED_ROWS / num_experts, 2)}),
              flush=True)

        packed = {}
        for label, quant in quantizers.items():
            try:
                packed[label] = quant(bf_w13, bf_w2)
                print(json.dumps({"event": "weights_ok", "lib": label,
                                  "e_local": num_experts,
                                  "w1": list(packed[label][0].shape),
                                  "w1_sf": list(packed[label][1].shape)}), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "weights_error", "lib": label,
                                  "e_local": num_experts,
                                  "type": type(exc).__name__,
                                  "message": str(exc)[:300]}), flush=True)

        for rep in range(REPS):
            for label in ("b12x_ours", "flashinfer_b12x"):
                if label not in packed:
                    continue
                try:
                    # both paths mutate their inputs: prepare_weights repacks in
                    # place, so every rep gets its own copy
                    fresh = tuple(t.clone() for t in packed[label])
                    run, out = runners[label](fresh, ids, scales, num_experts, "cuda")
                except Exception as exc:  # noqa: BLE001
                    print(json.dumps({"event": "error", "rep": rep, "lib": label,
                                      "e_local": num_experts,
                                      "type": type(exc).__name__,
                                      "message": str(exc)[:300]}), flush=True)
                    packed.pop(label, None)
                    continue
                med, best = timeit(run)
                print(json.dumps({"event": "result", "rep": rep, "lib": label,
                                  "e_local": num_experts, "median_ms": round(med, 4),
                                  "min_ms": round(best, 4), "sanity": sanity(out)}),
                      flush=True)
        del bf_w13, bf_w2, packed
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
