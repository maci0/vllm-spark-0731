#!/usr/bin/env python3
"""Why did the MoE cluster sweep's output sums differ by 35 % across caps?

``bench_moe_clusters.py`` reported |sum| of 516316 at cap 188 against 795297 at
cap 130 with the same activations and the same routing. A correct kernel should
not do that, and until it is explained no number from that bench can be used.

This isolates the cause by asking three questions in one process:

1. Is the output stable across repeated `run` calls at a fixed cap? That
   separates a race from a configuration difference.
2. Is it stable across two rebuilds of the same case at a fixed cap? That
   catches buffer aliasing between plans.
3. Do the buffers actually move between caps? That catches a stale view.

Run inside ``vllm-spark-0731:main-029-proto-b12xref``:

    docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench \
        vllm-spark-0731:main-029-proto-b12xref /bench/diag_moe_divergence.py
"""

from __future__ import annotations

import json
import sys

import torch


def fingerprint(out: torch.Tensor) -> dict:
    f = out.float()
    return {
        "abssum": round(float(f.abs().sum()), 1),
        "sum": round(float(f.sum()), 1),
        "ptr": out.data_ptr(),
        "shape": list(out.shape),
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    sys.path.insert(0, "/bench")
    import bench_moe_clusters as bench

    weights = bench.build_weights("cuda")
    print(json.dumps({"event": "weights_ok"}), flush=True)

    for cap in (188, 130):
        bench.set_dynamic_cap(cap)

        run, out = bench.make_case(weights, "cuda")
        print(json.dumps({"event": "case_built", "cap": cap, **fingerprint(out)}), flush=True)

        # 1. same binding, repeated calls
        for i in range(4):
            run()
            torch.cuda.synchronize()
            print(json.dumps({"event": "repeat_run", "cap": cap, "i": i,
                              **fingerprint(out)}), flush=True)

        # 2. rebuild the same case at the same cap
        run2, out2 = bench.make_case(weights, "cuda")
        print(json.dumps({"event": "rebuilt", "cap": cap, **fingerprint(out2)}), flush=True)

        # 3. does the first binding still read the same place?
        run()
        torch.cuda.synchronize()
        print(json.dumps({"event": "first_binding_after_rebuild", "cap": cap,
                          **fingerprint(out)}), flush=True)

    # Inputs must be identical for the comparison to mean anything.
    bench.set_dynamic_cap(188)
    torch.manual_seed(1234)
    x1 = torch.randn(bench.TOKENS, bench.HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
    ids1 = torch.stack([
        torch.randperm(bench.NUM_EXPERTS, device="cuda")[: bench.TOPK]
        for _ in range(bench.TOKENS)
    ]).to(torch.int32)
    torch.manual_seed(1234)
    x2 = torch.randn(bench.TOKENS, bench.HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
    ids2 = torch.stack([
        torch.randperm(bench.NUM_EXPERTS, device="cuda")[: bench.TOPK]
        for _ in range(bench.TOKENS)
    ]).to(torch.int32)
    print(json.dumps({
        "event": "input_reproducibility",
        "x_equal": bool(torch.equal(x1, x2)),
        "ids_equal": bool(torch.equal(ids1, ids2)),
    }), flush=True)


if __name__ == "__main__":
    main()
