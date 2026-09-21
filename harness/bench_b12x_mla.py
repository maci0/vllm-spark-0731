#!/usr/bin/env python3
"""DSV4 compressed sparse MLA head-to-head: our b12x against the reference's b12x_ref.

Attention is the region the profiler prices at 0.94 ms of a 2.83 ms DSV4-Flash target
layer, larger than WO's 0.60 ms, so the generation question is worth this measurement.
Same discipline as ``bench_b12x_wo.py``: one contract, both kernels, identical inputs,
agreement checked before any timing is believed.

Inputs are built with the reference package's own packer
(``compressed_reference.pack_compressed_mla_kv_cache_reference``) and validated against
its pure-torch implementation, so the 584 B/token page layout is the real one rather
than an invented one.

Run inside ``vllm-spark-0731:main-029-proto-b12xref`` on spark1:

    docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench \
        vllm-spark-0731:main-029-proto-b12xref /bench/bench_b12x_mla.py
"""

from __future__ import annotations

import json
import sys

import torch

# TP=2, so a rank's kernel sees the local head count.
NUM_Q_HEADS = 32
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
SWA_WIDTH = 128
INDEX_TOPK = 512
KV_PAGE_SIZE = 64
BATCH = 6
NEXT_N = 8
INDEXED_TOKENS = INDEX_TOPK * KV_PAGE_SIZE  # a full topk of indexed context
ITERS = 30
WARMUP = 3
SM_SCALE = HEAD_DIM ** -0.5


def geometry():
    import b12x_ref.attention.mla as ref

    r = ref.compressed_reference
    return {
        "bytes_per_token": r.COMPRESSED_MLA_BYTES_PER_TOKEN,
        "nope_dim": r.COMPRESSED_MLA_NOPE_DIM,
        "rope_dim": r.COMPRESSED_MLA_ROPE_DIM,
        "head_dim": r.COMPRESSED_MLA_HEAD_DIM,
        "topk": r.COMPRESSED_MLA_INDEX_TOPK,
        "swa_tokens": r.COMPRESSED_MLA_SWA_TOKENS,
        "page_nbytes": r.compressed_mla_page_nbytes(KV_PAGE_SIZE),
    }


def pack(tokens: int, device: str):
    import b12x_ref.attention.mla as ref

    r = ref.compressed_reference
    k_nope = torch.randn(tokens, NOPE_DIM, device=device) * 0.1
    k_rope = torch.randn(tokens, ROPE_DIM, device=device) * 0.1
    num_pages = (tokens + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE
    return r.pack_compressed_mla_kv_cache_reference(
        k_nope, k_rope, page_size=KV_PAGE_SIZE, num_pages=num_pages
    ), num_pages


def build_case(rows: int, device: str):
    """SWA window of 128 tokens and a full 512-page indexed history, per row."""
    swa_cache, swa_pages = pack(rows + SWA_WIDTH, device)
    idx_cache, idx_pages = pack(INDEXED_TOKENS, device)

    q = torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.1

    # SWA: row r looks back SWA_WIDTH tokens; page ids of that window.
    win = torch.arange(SWA_WIDTH, device=device)
    swa_indices = (win // KV_PAGE_SIZE)[None, :].expand(rows, -1).contiguous().to(torch.int32)
    swa_lengths = torch.full((rows,), SWA_WIDTH, dtype=torch.int32, device=device)

    # Indexed: every row attends the full topk history through its own page table.
    topk = min(INDEX_TOPK, idx_pages)
    idx_ids = torch.arange(topk, device=device)
    indexed_indices = idx_ids[None, :].expand(rows, -1).contiguous().to(torch.int32)
    indexed_lengths = torch.full((rows,), topk, dtype=torch.int32, device=device)
    indexed_page_table = idx_ids[None, :].expand(rows, -1).contiguous().to(torch.int32)

    return {
        "q": q,
        "swa_k_cache": swa_cache,
        "swa_indices": swa_indices,
        "swa_lengths": swa_lengths,
        "indexed_k_cache": idx_cache,
        "indexed_indices": indexed_indices,
        "indexed_lengths": indexed_lengths,
        "indexed_page_table": indexed_page_table,
        "idx_pages": idx_pages,
        "swa_pages": swa_pages,
    }


def reference_out(case):
    import b12x_ref.attention.mla as ref

    return ref.compressed_reference.compressed_sparse_mla_reference(
        case["q"],
        case["swa_k_cache"],
        case["swa_indices"],
        case["swa_lengths"],
        sm_scale=SM_SCALE,
        extra_k_cache=case["indexed_k_cache"],
        extra_indices=case["indexed_indices"],
        extra_topk_lengths=case["indexed_lengths"],
        swa_page_size=KV_PAGE_SIZE,
        extra_page_size=KV_PAGE_SIZE,
    )


def forward_kwargs(case):
    return dict(
        q_all=case["q"],
        swa_k_cache=case["swa_k_cache"],
        swa_indices=case["swa_indices"],
        swa_topk_lengths=case["swa_lengths"],
        sm_scale=SM_SCALE,
        swa_page_size=KV_PAGE_SIZE,
        indexed_k_cache=case["indexed_k_cache"],
        indexed_indices=case["indexed_indices"],
        indexed_topk_lengths=case["indexed_lengths"],
        indexed_page_size=KV_PAGE_SIZE,
        # The unified SM12x backend takes raw slot ids only; a mapped page table is
        # rejected on both sides, so the indexed cache is keyed directly.
        indexed_page_table=None,
    )


def run_ours(case):
    import b12x.attention.compressed_sparse_mla as mine

    rows = case["q"].shape[0]
    caps = mine.Caps(
        device="cuda",
        num_q_heads=NUM_Q_HEADS,
        max_q_rows=rows,
        max_width=SWA_WIDTH + INDEX_TOPK,
        max_page_table_width=INDEX_TOPK,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        kv_dtype=torch.uint8,
        page_size=KV_PAGE_SIZE,
    )
    plan = mine.plan(caps)
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = plan.bind(
        scratch=scratch,
        q=case["q"],
        swa_indices=case["swa_indices"],
        swa_lengths=case["swa_lengths"],
        indexed_indices=case["indexed_indices"],
        indexed_lengths=case["indexed_lengths"],
        indexed_page_table=None,
    )
    # The binding owns q and the index tensors; only the caches and the scalars
    # are passed alongside it.
    kw = dict(
        binding=binding,
        swa_k_cache=case["swa_k_cache"],
        indexed_k_cache=case["indexed_k_cache"],
        sm_scale=SM_SCALE,
        swa_page_size=KV_PAGE_SIZE,
        indexed_page_size=KV_PAGE_SIZE,
    )
    out = mine.run(**kw)
    torch.cuda.synchronize()
    return (lambda: mine.run(**kw)), out


def run_ref(case):
    import b12x_ref.attention.mla as ref
    from b12x_ref.attention.workspace import (
        B12XAttentionArena,
        B12XAttentionArenaCaps,
        B12XAttentionWorkspaceContract,
    )

    rows = case["q"].shape[0]
    kv_rows = case["idx_pages"] * KV_PAGE_SIZE
    arena_caps = B12XAttentionArenaCaps(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=NUM_Q_HEADS,
        indexer_num_q_heads=NUM_Q_HEADS,
        head_dim=HEAD_DIM,
        max_v_head_dim=HEAD_DIM,
        topk=SWA_WIDTH,
        max_page_table_width=INDEX_TOPK,
        extend_max_total_q=rows,
        extend_max_batch=BATCH,
        extend_max_kv_rows=kv_rows,
        paged_max_q_rows=rows,
        paged_max_batch=BATCH,
        reserve_compressed_mla_staging=True,
    )
    arena = B12XAttentionArena.allocate(arena_caps)
    contract = B12XAttentionWorkspaceContract(
        mode="decode",
        max_total_q=rows,
        max_batch=BATCH,
        max_paged_q_rows=rows,
        max_kv_rows=kv_rows,
        v_head_dim=HEAD_DIM,
        indexer_num_q_heads=NUM_Q_HEADS,
        max_page_table_width=INDEX_TOPK,
        topk=SWA_WIDTH,
    )
    workspace = arena.make_workspace(contract)
    kw = forward_kwargs(case)
    out = ref.compressed_mla_decode_forward(workspace=workspace, **kw)
    torch.cuda.synchronize()
    return (lambda: ref.compressed_mla_decode_forward(workspace=workspace, **kw)), out


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


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).abs().mean() / a.abs().mean().clamp_min(1e-6))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print(json.dumps({"event": "geometry", **geometry()}), flush=True)

    rows = BATCH * NEXT_N
    case = build_case(rows, "cuda")
    print(
        json.dumps(
            {
                "event": "case",
                "rows": rows,
                "q": list(case["q"].shape),
                "swa_cache": list(case["swa_k_cache"].shape),
                "indexed_cache": list(case["indexed_k_cache"].shape),
                "swa_indices": list(case["swa_indices"].shape),
                "indexed_indices": list(case["indexed_indices"].shape),
                "indexed_page_table": list(case["indexed_page_table"].shape),
            }
        ),
        flush=True,
    )

    truth = None
    try:
        truth = reference_out(case)
        torch.cuda.synchronize()
        print(json.dumps({"event": "reference_ok", "shape": list(truth.shape)}), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"event": "reference_error", "type": type(exc).__name__,
                          "message": str(exc)[:400]}), flush=True)

    outs = {}
    for label, fn in (("b12x_ours", run_ours), ("b12x_ref", run_ref)):
        try:
            runner, out = fn(case)
            med, best = timeit(runner)
            entry = {
                "event": "result",
                "lib": label,
                "median_ms": round(med, 4),
                "min_ms": round(best, 4),
                "out_shape": list(out.shape),
            }
            if truth is not None:
                entry["rel_vs_reference"] = round(rel(out, truth), 6)
            outs[label] = out
            print(json.dumps(entry), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"event": "error", "lib": label,
                              "type": type(exc).__name__,
                              "message": str(exc)[:400]}), flush=True)

    if len(outs) == 2:
        print(json.dumps({"event": "cross",
                          "rel_ours_vs_ref": round(rel(outs["b12x_ours"], outs["b12x_ref"]), 6)}),
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
