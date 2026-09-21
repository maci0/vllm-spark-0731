#!/usr/bin/env python3
"""WO-projection head-to-head: our b12x generation vs the reference's b12x_ref.

The reference image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`) carries b12x 0.15.3.
`vllm-spark-0731:main-029-proto-b12xref` keeps our b12x (1.2.6/1.3.0 code) under
`b12x` and installs 0.15.3 alongside as `b12x_ref`. This measures both native
WO-A/WO-B kernels at the DeepSeek-V4-Flash-0731 shapes on the GPU, which is the
premise HANDOVER's next step asks for before any call site is ported.

Run inside that image on spark1:

    docker run --rm --gpus all -v ~/bench:/bench vllm-spark-0731:main-029-proto-b12xref \
        python3 /bench/bench_b12x_wo.py

Prints one JSON line per (library, tokens). No serving, no model weights.
"""

from __future__ import annotations

import json
import sys

import torch

# DeepSeek-V4-Flash-0731, TP=1 (single-GPU probe): docs/knowledge/02-model.md.
HIDDEN = 4096
N_GROUPS = 8
HEADS_PER_GROUP = 8
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
RANK = 1024  # o_lora_rank
GROUP_WIDTH = HEADS_PER_GROUP * HEAD_DIM  # 4096

TOKENS = (1, 2, 4, 8, 16, 32, 48, 64)
ITERS = 50
WARMUP = 5


def make_weights(device):
    """FP8 block-scaled WO-A/WO-B in checkpoint layout.

    WO-A [groups*rank, group_width], WO-B [hidden, groups*rank]; 128x128 block
    scales, which is what both pack functions requantize from.
    """
    wa = (torch.randn(N_GROUPS * RANK, GROUP_WIDTH, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    wb = (torch.randn(HIDDEN, N_GROUPS * RANK, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    sa = torch.ones(N_GROUPS * RANK // 128, GROUP_WIDTH // 128, device=device)
    sb = torch.ones(HIDDEN // 128, N_GROUPS * RANK // 128, device=device)
    return wa, sa, wb, sb


def timeit(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def bench_ours(wa, sa, wb, sb, tokens):
    from b12x.gemm import wo_projection as wo

    caps = wo.Caps(
        device="cuda",
        max_tokens=tokens,
        groups=N_GROUPS,
        group_width=GROUP_WIDTH,
        rank=RANK,
        hidden=HIDDEN,
    )
    plan = wo.plan(caps)
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    weights = wo.pack_weights(
        wa, sa, wb, sb, groups=N_GROUPS, group_width=GROUP_WIDTH, rank=RANK, hidden=HIDDEN
    )
    x = torch.randn(tokens, N_GROUPS, GROUP_WIDTH, dtype=torch.bfloat16, device="cuda")
    binding = wo.bind(plan, scratch=scratch, source_tgd=x, weights=weights, expected_m=tokens)
    out = wo.run(binding=binding)
    torch.cuda.synchronize()
    return (lambda: wo.run(binding=binding)), out


def bench_ref(wa, sa, wb, sb, tokens):
    import b12x_ref.gemm.wo_projection as wor

    caps = wor.WOProjectionScratchCaps(
        device="cuda",
        max_tokens=tokens,
        groups=N_GROUPS,
        group_width=GROUP_WIDTH,
        rank=RANK,
        hidden=HIDDEN,
    )
    plan = wor.plan_wo_projection_scratch(caps)
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    weights = wor.pack_wo_projection_fp8_block_scaled_weights_mxfp8(
        wa, sa, wb, sb, groups=N_GROUPS, group_width=GROUP_WIDTH, rank=RANK, hidden=HIDDEN
    )
    x = torch.randn(tokens, N_GROUPS, GROUP_WIDTH, dtype=torch.bfloat16, device="cuda")
    binding = plan.bind(scratch=scratch, source_tgd=x, weights=weights)
    out = wor.wo_projection_mxfp8(binding=binding)
    torch.cuda.synchronize()
    return (lambda: wor.wo_projection_mxfp8(binding=binding)), out


def check_agreement(wa, sa, wb, sb, tokens):
    """Same weights and same input through both kernels, relative difference.

    Without this the timings could be measuring different work.
    """
    import b12x_ref.gemm.wo_projection as wor
    from b12x.gemm import wo_projection as wo

    x = torch.randn(tokens, N_GROUPS, GROUP_WIDTH, dtype=torch.bfloat16, device="cuda")

    caps = wo.Caps(
        device="cuda", max_tokens=tokens, groups=N_GROUPS,
        group_width=GROUP_WIDTH, rank=RANK, hidden=HIDDEN,
    )
    plan = wo.plan(caps)
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    w = wo.pack_weights(
        wa, sa, wb, sb, groups=N_GROUPS, group_width=GROUP_WIDTH, rank=RANK, hidden=HIDDEN
    )
    ours = wo.run(
        binding=wo.bind(plan, scratch=scratch, source_tgd=x, weights=w, expected_m=tokens)
    ).float()

    rcaps = wor.WOProjectionScratchCaps(
        device="cuda", max_tokens=tokens, groups=N_GROUPS,
        group_width=GROUP_WIDTH, rank=RANK, hidden=HIDDEN,
    )
    rplan = wor.plan_wo_projection_scratch(rcaps)
    rspec = rplan.scratch_specs()[0]
    rscratch = torch.empty(rspec.shape, dtype=rspec.dtype, device=rspec.device)
    rw = wor.pack_wo_projection_fp8_block_scaled_weights_mxfp8(
        wa, sa, wb, sb, groups=N_GROUPS, group_width=GROUP_WIDTH, rank=RANK, hidden=HIDDEN
    )
    ref = wor.wo_projection_mxfp8(
        binding=rplan.bind(scratch=rscratch, source_tgd=x, weights=rw)
    ).float()

    denom = ours.abs().mean().clamp_min(1e-6)
    rel = float((ours - ref).abs().mean() / denom)
    return rel


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(
        json.dumps(
            {
                "event": "device",
                "name": torch.cuda.get_device_name(0),
                "shapes": {
                    "hidden": HIDDEN,
                    "groups": N_GROUPS,
                    "group_width": GROUP_WIDTH,
                    "rank": RANK,
                },
            }
        ),
        flush=True,
    )
    wa, sa, wb, sb = make_weights("cuda")

    for tokens in TOKENS:
        for label, build in (("b12x_ours", bench_ours), ("b12x_ref", bench_ref)):
            try:
                run, out = build(wa, sa, wb, sb, tokens)
                med, best = timeit(run)
                print(
                    json.dumps(
                        {
                            "event": "result",
                            "lib": label,
                            "tokens": tokens,
                            "median_ms": round(med, 4),
                            "min_ms": round(best, 4),
                            "out_shape": list(out.shape),
                            "out_sum": round(float(out.float().sum()), 3),
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    json.dumps(
                        {
                            "event": "error",
                            "lib": label,
                            "tokens": tokens,
                            "type": type(exc).__name__,
                            "message": str(exc)[:300],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    sys.exit(main())
