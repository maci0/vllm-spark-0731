[← Index](00-index.md) · [Glossary](glossary.md)

# Kernel Stack & Attention Backends

## The Kernel Stack on SM12x

SM12x (sm_121a, family 120) runs DeepGEMM nv_dev (`is_deep_gemm_supported()` is True), but stock DeepGEMM shapes (such as 2-state MQA pages or mHC broadcast) and **CUTLASS block-FP8** (SM90/SM100) cannot execute natively. Every unsupported op must use **b12x** or a **PyTorch/TileLang fallback**.

| Operation | Kernel | Notes |
|-----------|--------|-------|
| Linear FP8 | `B12xFp8BlockScaledMM` | `--linear-backend b12x`; wq_a cosine 0.9999986 vs torch |
| MoE MXFP4 | `B12X_MXFP4_MXFP8` | `--moe-backend b12x` |
| Attention MLA | `B12X_MLA_SPARSE` (b12x compressed MLA) | 584 B DSV4 page |
| MQA logits (prefill) | PyTorch dequant + ReLU | `_sm12x_fp8_mqa_logits` fallback |
| MQA logits (decode) | b12x paged, page_size 64 | packed-at-store |
| fp8_einsum | PyTorch dequant fallback | |
| mHC prenorm | TileLang | fallback when DeepGEMM mHC is unsupported |
| Speculative decode | DSpark k=5 | FlashInfer TOPK=192 dispatch |

### DeepGEMM & CUTLASS Compatibility on SM12x

- **DeepGEMM nv_dev** is compiled for SM12x on `main-b12x` (`is_deep_gemm_supported()` returns True), but stock DeepGEMM kernels assert Hopper/Blackwell page layouts (32 or 64 states, whereas DSV4 compress-128 pages have 2 states) or unsupported broadcast shapes. These paths require guards (e.g. `_should_build_paged_mqa_logits_metadata`) and PyTorch/b12x fallbacks.
- **Pure-FP8 Linear GEMM Aliasing Bug**: In `nv_dev 8b1392b`, `csrc/apis/gemm.hpp:851` aliases pure-FP8 `fp8_gemm_nt` to `fp8_fp4_gemm_nt`. On SM12x (`arch_major == 12`), that dispatcher unconditionally executes `sm120_fp8_fp4_gemm_1d1d`, treating FP8 weights as FP4 and producing silent output corruption. Never use `LINEAR_BACKEND=deep_gemm` on SM12x until upstream separates the 1d1d pure-FP8 dispatch; keep `--linear-backend b12x` pinned (see [09-golden-deepgemm.md](09-golden-deepgemm.md)).
- **CUTLASS block-FP8** requires SM90/SM100 TMA architectures and does not run on SM12x, falling back to PyTorch/Triton/TileLang.

SM12x kernels live in:
- **DeepGEMM nv_dev** (not main) — SM12x MQA only in nv_dev branch
- **b12x** — purpose-built for SM120/SM121
- **FlashInfer** — SM120 support for DSV4 MLA (TOPK 192 in #4380)

---

## Attention Backends

### `B12X_MLA_SPARSE` (Live: main-b12x)

- **b12x compressed MLA** on the 584 B DSV4 page
- **Stock vLLM main has no such enum** — registered via `patches/files/dsv4_b12x_sparse.py`
- Used for both target and DSpark draft
- Implements the packed-at-store paged indexer correctly

### `FLASHINFER_MLA_SPARSE_DSV4` (Overlay Fallback)

- FlashInfer DSV4 sparse MLA, same 584 B page
- Different kernel implementation
- Requires overlay patch `patch_dsv4_nvfp4_attn` or SM12x rejects the dtype
- Has the **packed-at-store gather bug** (see below)

### Backend parity, measured 2026-08-24

After fixing the eidx contiguity bug (below), the two backends are roughly
at parity on the matched-main image:

| backend | c1 | c32 |
|---|---|---|
| B12X_MLA_SPARSE | **25.8** | 172 |
| FLASHINFER_MLA_SPARSE_DSV4 | 22.6 | **179** |

The ~2× gap to the anemll/eugr images is a **whole-stack difference** (older
vLLM, real NVFP4 KV writer, no-spec options), not the attention backend.
Live pin stays `B12X_MLA_SPARSE` (better single-stream).

### FlashInfer eidx contiguity bug (fixed)

`flashinfer_sparse.py:_forward_decode` passes `extra_sparse_indices` that can
be non-contiguous — C4A uses `global_indices.view(num_decode_tokens, 1, -1)`,
C128A a non-contiguous metadata tensor. The FlashInfer SM120 C++
(`sparse_mla_sm120.cu`) checks `eidx.IsContiguous()` and dies with
"eidx must be contiguous" during warmup, so `FLASHINFER_MLA_SPARSE_DSV4`
would not boot. Fix: `.contiguous()` on both paths (overlay
`flashinfer-eidx-contig`). Upstream PR candidate.

---

## The Packed-at-Store Paged Indexer (and Its Gather Bug)

### Storage Layout

The sparse indexer K is stored **packed K-then-scale** per 64-token page:
```
[page][8192 bytes K][256 bytes scale]
```
= 128 B K + 4 B scale per token, **separated** (not interleaved).

This skips the unused page64 workspace: **KV 94,516 → 97,737 tokens**.

### The Bug

| Kernel | Layout Expected | Layout Actual | Result |
|--------|-----------------|---------------|--------|
| b12x paged (`try_paged_mqa_logits`) | Packed K-then-scale | Packed K-then-scale | ✅ Correct |
| FlashInfer gather (`fp8_fp4_paged_mqa_logits`) | 132-byte interleaved | Packed K-then-scale | ❌ Numerically wrong |

**Symptoms**: France still prints "Paris" but DSpark acceptance collapses 38–70% vs ~73%.

### Performance Impact

| Path | 1-way | 8-way |
|------|-------|-------|
| b12x paged (correct) | 26.90 tok/s | 85.98 tok/s |
| FlashInfer gather (buggy) | 29.88 tok/s | 89.34 tok/s |

The gather path is **~11% faster** but numerically wrong. Fix = make gather read packed layout.

---

## b12x Kernel Details (from `patches/files/sm12x_b12x_kernels.py`)

### Key Constants
```python
_INDEX_HEAD_DIM = 128
_PAGE_SIZE = 64
_SCALE_BYTES = 4
_PACKED_PAGE_BYTES = _PAGE_SIZE * (_INDEX_HEAD_DIM + _SCALE_BYTES)  # 8448
_TOKEN_BYTES = _INDEX_HEAD_DIM + _SCALE_BYTES  # 132
```

### Critical Functions

- `expand_block_table_to_page64()` — 256-token manager blocks → 4×64 kernel pages
- `trim_page_table_skip_schedule()` — drops last page at schedule threshold (≥1024 pages)
- `try_paged_mqa_logits()` — b12x paged kernel consuming packed layout
- `try_b12x_wo_proj()` — WO projection (o_proj) on SM12x, otherwise PyTorch einsum dequant

### Scheduled Paged Scorer

b12x uses the scheduled paged scorer when `max_pages >= 1024 and q_rows <= 8`.
The **1-row kernel is the only scheduled path that works** — do not feed 1-row scheduled into 8-row decode.

---

## FlashInfer DSV4 TOPK 192

DSpark k=5 requires dispatch TOPK = `ceil(133/64)*64 = 192`.
Stock FlashInfer has {128, 512, 1024, 2048}.
**PR #4380** (merged 2026-08-08) added 192 and 256.
**Must use FlashInfer git main** (not 0.6.16.post3 from v0.27.1).

---

## Related Docs

- [01-hardware.md](01-hardware.md) — Why SM12x forces this kernel stack
- [02-model.md](02-model.md) — MLA and DSpark requirements
- [04-quantization-kv.md](04-quantization-kv.md) — KV cache dtypes and page layouts
- [05-performance.md](05-performance.md) — Benchmark impact of kernel choices
- [08-upstream.md](08-upstream.md) — Upstream PRs for SM12x support
- [09-golden-deepgemm.md](09-golden-deepgemm.md) — DeepGEMM regression and pin-back

### Raw evidence (field notes)

- [`../field-notes/nvfp4/DEEPGEMM_CALL_SITES.md`](../field-notes/nvfp4/DEEPGEMM_CALL_SITES.md) — every unguarded DeepGEMM call site on SM120
- [`../field-notes/nvfp4/MHC_DEEPGEMM_SM121.md`](../field-notes/nvfp4/MHC_DEEPGEMM_SM121.md) — mHC pre-broadcast assertion and the #50645 guard

---

**[← Prev](02-model.md) · [Glossary](glossary.md) · [Next](04-quantization-kv.md) →**
