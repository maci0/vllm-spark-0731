#!/usr/bin/env python3
"""Does our MoE skip experts that receive no tokens?

The expert-bytes attribution says we stream ~1.202 GB per layer where the
reference streams 0.916 GB, and that difference is the whole gap. One of the two
candidate mechanisms is a failure to skip experts with no rows: if our kernel
reads every resident expert regardless of routing, then concentrating the routing
on fewer experts will not make it faster, while a kernel that skips will speed up
roughly in proportion to the experts actually touched.

So this holds the weights, the shapes and the routed-row count fixed and varies
only how many distinct experts the routing targets:

    active experts in (256, 128, 64, 32, 8)

288 routed rows (48 tokens x topk 6) are drawn from the active set. If time is
flat across the column, the kernel reads all resident experts; if it falls, it
skips.

Three reps, first discarded: a concurrent CPU-bound build inflates the first.

    docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench \\
        vllm-spark-0731:main-029-proto-ccompile /bench/bench_moe_skipping.py
"""

from __future__ import annotations

import json
import sys

import torch

NUM_EXPERTS = 256
TOPK = 6
HIDDEN = 4096
INTER = 2048
TOKENS = 48
ROUTED_ROWS = TOKENS * TOPK
ACTIVE = (256, 128, 64, 32, 8)
REPS = 3
ITERS = 30
WARMUP = 3


def build_weights(device: str):
    sys.path.insert(0, "/bench")
    import bench_moe_clusters as bench

    torch.manual_seed(0)
    p1, s1, p2, s2 = [], [], [], []
    for _ in range(NUM_EXPERTS):
        a = torch.randn(2 * INTER, HIDDEN, device=device)
        a = a / a.abs().amax().clamp_min(1e-6)
        q, s = bench.quantize_k32(a)
        p1.append(q)
        s1.append(s)
        del a
        b = torch.randn(HIDDEN, INTER, device=device)
        b = b / b.abs().amax().clamp_min(1e-6)
        q, s = bench.quantize_k32(b)
        p2.append(q)
        s2.append(s)
        del b
    return torch.stack(p1), torch.stack(s1), torch.stack(p2), torch.stack(s2)


def make_case(weights, active: int, device: str):
    from b12x.moe import fused_moe as fm

    p1, s1, p2, s2 = weights
    src = fm.PackedSource(
        format=fm.PackedSourceFormat.MXFP4_E8M0_K32, w13_layout=fm.W13Layout.W31
    )
    act = fm.ActivationSpec(
        mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16
    )
    geo = fm.MoEGeometry(
        num_experts=NUM_EXPERTS, hidden_size=HIDDEN, intermediate_size=INTER
    )
    plan = fm.plan_weights(source=src, activation=act, geometry=geo)
    ones = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=device)
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

    # Only the routing changes: expert ids come from the first `active` experts.
    # Draw from `active` distinct experts, allowing repeats across the batch, so
    # the number of experts actually touched is `active` and only that varies.
    torch.manual_seed(1234)
    pool = torch.randperm(NUM_EXPERTS, device=device)[:active]
    ids = pool[torch.randint(0, active, (TOKENS, TOPK), device=device)].to(torch.int32)
    w = torch.rand(TOKENS, TOPK, dtype=torch.float32, device=device)
    w = w / w.sum(dim=-1, keepdim=True)
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1

    binding = fm.bind(
        execution, scratch=scratch, a=x, experts=experts,
        topk_weights=w, topk_ids=ids,
    )
    out = fm.run(binding=binding)
    torch.cuda.synchronize()
    return (lambda: fm.run(binding=binding)), out, ids


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


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(json.dumps({"event": "geometry", "experts_resident": NUM_EXPERTS,
                      "routed_rows": ROUTED_ROWS, "active_sweep": list(ACTIVE)}),
          flush=True)
    weights = build_weights("cuda")
    print(json.dumps({"event": "weights_ok", "w1": list(weights[0].shape)}), flush=True)

    for active in ACTIVE:
        meds = []
        for rep in range(REPS):
            fresh = tuple(t.clone() for t in weights)
            try:
                run, out, ids = make_case(fresh, active, "cuda")
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "error", "active": active, "rep": rep,
                                  "type": type(exc).__name__,
                                  "message": str(exc)[:300]}), flush=True)
                break
            med, best = timeit(run)
            meds.append(med)
            print(json.dumps({"event": "result", "active": active, "rep": rep,
                              "distinct_ids": int(ids.unique().numel()),
                              "median_ms": round(med, 4), "min_ms": round(best, 4),
                              "finite": bool(torch.isfinite(out.float()).all())}),
                  flush=True)
        if meds:
            print(json.dumps({"event": "summary", "active": active,
                              "medians": [round(m, 4) for m in meds],
                              "clean_median_ms": round(sum(meds[1:]) / max(len(meds) - 1, 1), 4)}),
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
