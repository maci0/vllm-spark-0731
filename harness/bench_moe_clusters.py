#!/usr/bin/env python3
"""DSV4-Flash decode MoE on our b12x: does the mid-regime cluster cap matter?

``harness/probe_moe_tuning.py`` showed our generation has no ``static`` MoE
backend and runs a flat ``max_active_clusters`` cap of 188 for every routed-row
count up to 640, while the reference ships a different kernel tuned per row
count. This converts that structural finding into milliseconds on our side
alone: same kernel, same weights, same routing, only the cap moves.

That is the lazy half of a head-to-head, and the actionable half. It needs no
reference kernel and no reference weight packer, because the question is whether
our own configuration leaves time on the table in the band every protocol level
runs in.

Weight contract, read off ``fused_moe._impl._prepare_w4a8_from_mxfp4_source``:
``w1`` is uint8 ``[E, 2*inter, hidden/2]``, ``w2`` is uint8 ``[E, hidden,
inter/2]``, both packed two FP4 nibbles per byte along their last (K)
dimension, with E8M0 scale grids ``[E, rows, K//32]`` passed plain, not
swizzled. Activation is A8 with unit global scales, which is what the served
DSV4 path uses.

Run inside ``vllm-spark-0731:main-029-proto-b12xref``:

    docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench \
        vllm-spark-0731:main-029-proto-b12xref /bench/bench_moe_clusters.py
"""

from __future__ import annotations

import json
import sys

import torch

# DeepSeek-V4-Flash-0731, docs/knowledge/02-model.md: 256 routed experts,
# 6 per token, hidden 4096, moe_intermediate 2048.
NUM_EXPERTS = 256
TOPK = 6
HIDDEN = 4096
INTER = 2048
TOKENS = 48          # c6 decode step at k=7, six sequences
KBLOCK = 32          # e8m0 K32 scale grid, as the validator requires

# Our untuned default first, then the reference's tuned static caps for this band.
CAPS = (188, 175, 149, 141, 130, 96)
ITERS = 30
WARMUP = 3
REPS = 3


def quantize_k32(w_bf16: torch.Tensor):
    """FP4 E2M1 packed to uint8 plus an E8M0 per-K/32 scale byte grid.

    ``w_bf16`` is ``[rows, cols]`` with ``cols`` the K dimension.
    """
    from b12x._lib.intrinsics import fp4_quantize_values_torch, pack_grouped_fp4_values

    rows, cols = w_bf16.shape
    assert cols % KBLOCK == 0, cols
    blocked = w_bf16.float().reshape(rows, cols // KBLOCK, KBLOCK)
    bmax = blocked.abs().amax(dim=-1)
    # E8M0 holds powers of two; pick the smallest exponent that keeps the block
    # inside the FP4 range at |value| <= 6.
    exp = torch.ceil(torch.log2((bmax / 6.0).clamp_min(2.0 ** -120)))
    byte = (exp + 127).clamp(0, 255).to(torch.uint8)
    scale = torch.pow(torch.tensor(2.0), exp)
    scaled = (blocked / scale.unsqueeze(-1)).reshape(rows, cols)
    values = fp4_quantize_values_torch(scaled)
    packed = pack_grouped_fp4_values(values.unsqueeze(0)).squeeze(-1)
    return packed.contiguous(), byte.contiguous()


def build_weights(device: str):
    """One synthetic expert stack in the kernel's expected layout."""
    torch.manual_seed(0)
    w1_p, w1_s, w2_p, w2_s = [], [], [], []
    for _ in range(NUM_EXPERTS):
        w1 = torch.randn(2 * INTER, HIDDEN, device=device)
        w1 = w1 / w1.abs().amax().clamp_min(1e-6)
        p, s = quantize_k32(w1)
        w1_p.append(p)
        w1_s.append(s)
        del w1

        w2 = torch.randn(HIDDEN, INTER, device=device)
        w2 = w2 / w2.abs().amax().clamp_min(1e-6)
        p, s = quantize_k32(w2)
        w2_p.append(p)
        w2_s.append(s)
        del w2

    ones = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=device)
    return (
        torch.stack(w1_p), torch.stack(w2_p),
        torch.stack(w1_s), torch.stack(w2_s),
        ones, ones.clone(),
    )


def set_dynamic_cap(cap: int) -> None:
    """Pin the decode/dynamic ladder to a single flat cap."""
    import b12x.moe._shared.tuning as tuning

    tuning.MAX_ACTIVE_CLUSTERS_POLICY[("decode", "dynamic")] = (
        tuning.MaxActiveClustersPolicy(ladder=((1 << 30, int(cap)),))
    )


def make_case(weights, device: str):
    from b12x.moe import fused_moe as fm

    # Same activations and same routing for every cap, so the caps are compared
    # on identical work rather than on a fresh draw each time.
    torch.manual_seed(1234)

    w1, w2, w1_s, w2_s, g1, g2 = weights
    sources = fm.PackedSource(
        format=fm.PackedSourceFormat.MXFP4_E8M0_K32, w13_layout=fm.W13Layout.W31
    )
    activation = fm.ActivationSpec(
        mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16
    )
    geometry = fm.MoEGeometry(
        num_experts=NUM_EXPERTS, hidden_size=HIDDEN, intermediate_size=INTER
    )
    weight_plan = fm.plan_weights(
        source=sources, activation=activation, geometry=geometry
    )
    experts = fm.prepare_weights(
        plan=weight_plan,
        weights=fm.PackedWeights(
            w13=w1, w2=w2,
            w13_block_scales=w1_s, w2_block_scales=w2_s,
            w13_global_scales=g1, w2_global_scales=g2,
        ),
    )
    execution = fm.plan_execution(
        experts=experts,
        capacity=fm.ExecutionCapacity(max_tokens=TOKENS, top_k=TOPK),
    )
    fm.prewarm(execution)
    spec = execution.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.rand(TOKENS, TOPK, dtype=torch.float32, device=device)
    w = w / w.sum(dim=-1, keepdim=True)
    ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, device=device)[:TOPK] for _ in range(TOKENS)]
    ).to(torch.int32)
    binding = fm.bind(
        execution, scratch=scratch, a=x, experts=experts,
        topk_weights=w, topk_ids=ids,
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


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(json.dumps({"event": "geometry", "experts": NUM_EXPERTS, "topk": TOPK,
                      "hidden": HIDDEN, "inter": INTER, "tokens": TOKENS,
                      "routed_rows": TOKENS * TOPK}), flush=True)

    weights = build_weights("cuda")
    print(json.dumps({"event": "weights_ok",
                      "w1": list(weights[0].shape), "w2": list(weights[1].shape),
                      "w1_sf": list(weights[2].shape), "w2_sf": list(weights[3].shape),
                      "dtypes": [str(weights[0].dtype), str(weights[2].dtype)]}), flush=True)

    baseline = None
    for rep in range(REPS):
      for cap in CAPS:
          set_dynamic_cap(cap)
          # `prepare_weights` repacks the packed tensors in place, so every case
          # must get its own copy or builds after the first consume mutated input.
          case_weights = tuple(w.clone() for w in weights)
          try:
              run, out = make_case(case_weights, "cuda")
          except Exception as exc:  # noqa: BLE001
              print(json.dumps({"event": "error", "cap": cap,
                                "type": type(exc).__name__,
                                "message": str(exc)[:400]}), flush=True)
              continue
          med, best = timeit(run)
          if baseline is None:
              baseline = med
          print(json.dumps({"event": "result", "rep": rep, "cap": cap,
                            "median_ms": round(med, 4), "min_ms": round(best, 4),
                            "vs_cap188": round(med / baseline, 4),
                            "out_abssum": round(float(out.float().abs().sum()), 1)}),
                flush=True)


if __name__ == "__main__":
    sys.exit(main())
