[← Back to Knowledge Index](00-index.md)

# Upstream Gaps & PRs

Tracked in `docs/UPSTREAM.md`. **Do not open duplicate PRs** — comment with Spark evidence if a PR already covers it.

---

## Pins (From `docs/UPSTREAM.md`)

| Tree | ID | When |
|------|-----|------|
| vLLM matched-main (live image) | `e25c586b9` (`v0.1.dev1+ge25c586b9.d20260823`) | 2026-08-23 |
| vLLM v0.28.0rc2 (overlay fallback) | `74a6576b9b58` | 2026-08-21 06:47 UTC |
| vLLM main (PR check) | default branch | 2026-08-24 afternoon |
| DeepGEMM in v0.27.1 / overlay `.so` | `e21c821f39a2` (DeepGEMM **main**, ~SM90/SM100) | 2026-08-04 |
| DeepGEMM in v0.28.0rc2, vLLM main cmake, matched-main | `8b1392b978f5` (**nv_dev** HEAD) | 2026-08-11 |
| DeepGEMM in eugr Dockerfile | `a6b593d28267` (nv_dev, frozen) | 2026-06-29 |
| FlashInfer overlay image | `0.6.16.post3` | from v0.27.1; overlay adds TOPK=192 |
| FlashInfer matched-main | git **main** (192 present) | image build |
| flashinfer-ai main | has 192 and 256 | #4380 merged 2026-08-08 |

---

## Backport (Fix Exists — Apply as Git-Diff Patch)

| Gap | Fix | Status | Backport |
|-----|-----|--------|----------|
| MoE `--moe-backend b12x` | #52018 | **merged** 2026-08-21 | overlay `copy_new_modules` / `patch_moe_backend` / etc. (already in `apply_overlays.py`) |
| KVBlockZeroer non-uniform pages | #49704 | **merged** 2026-07-24 | overlay `patch_kv_zeroer_skip` (ratio=1); do not re-PR |
| FlashInfer DSV4 TOPK=192 (DSpark k=5) | #4380 | **merged** flashinfer-ai | already in git-main FlashInfer |
| FlashInfer C128A eidx contiguity (DSV4 spec-decode boot crash) | #53574 | **open** 2026-08-24 | `pr-53574.diff` + overlay `flashinfer-eidx-contig` (consumer, defense-in-depth) |
| Triton block-scaled MM E8M0 upcast (`KeyError: float8_e8m0fnu` on SM12x) | #47988 | **open** 2026-08-18 | `pr-47988.diff` (source hunks) + overlay `triton-e8m0-sm12x` (skips when #47988 form present) |
| SM12x einsum recipe (SM90 `(1,128,128)` vs SM100 packed INT32) | #53521 | **open** | `pr-53521.diff` + overlay `einsum-sm12x` |
| DeepGEMM SM120/SM121 SF layout (`csrc/apis/layout.hpp`) | DeepGEMM #403 | **merged** `nv_dev` | `deepgemm-pr-403.diff` (applied idempotently in `Dockerfile.main`) |
| DeepGEMM SM120 pure-FP8 GEMM port (`sm100_fp8_gemm_1d1d` to `nv_dev`) | anemll 2.5.0 port | **staged local port** | `deepgemm-fp8-1d1d-port.diff` (applied in `Dockerfile.main`) |

---

## Open PRs (Comment / Contribute)

| Gap | PR | Notes |
|-----|-----|-------|
| `mhc_pre_broadcast_tilelang` unguarded `tf32_hc_prenorm_gemm` | #53055 (also #50645) | Same PR covers CUTLASS FP8 `is_supported()` SM12x |
| CUTLASS FP8 `is_supported()` ignores SM12x | #53055 | Overlay `patch_cutlass_sm12x_guard`; backport `pr-53055.diff` |
| DSV4 sparse-MLA spec-decode query shapes | #52499 | Backport `pr-52499.diff` (`flashinfer_sparse.py` decode next_n shapes) |
| DSV4 kernel block 64 | #53425 | Backport `pr-53425.diff` |
| Indexer DeepGEMM gate | #53522 | Backport `pr-53522.diff` |

---

## Triaged 2026-08-24 against vLLM main (`6648eb1`) — no new PRs needed

Every gap below either has an open PR (backported as `patches/upstream/pr-*.diff`)
or was verified not-a-bug. Do not open duplicate PRs.

1. **FlashInfer eidx contiguity** — covered by **#53574** (C128A builder fix).
   - Root cause confirmed on our pair: `_build_c128a_metadata` publishes a
     width-narrowed slice of the persistent `global_decode_buffer`; DSpark
     verification batches (`num_decodes*(1+K) > 64` tokens) hit the paged
     orchestrator's `eidx.IsContiguous()` check and crash at boot.
   - The 0731 checkpoint alternates `compress_ratios` `4, 128, ...`, so both
     branches run. The C4A branch is verified **contiguous**
     (`empty_like` of the contiguous `topk_indices_buffer` row slice in
     `dspark.py`) — no C4A bug; the consumer `.contiguous()` is a no-op there.
   - Backport `pr-53574.diff`; overlay `flashinfer-eidx-contig` kept as
     defense-in-depth. Evidence comment posted on #53574.

2. **Triton block-scaled MM E8M0 upcast** — covered by **#47988**
   (unconditional E8M0→fp32 upcast in `w8a8_triton_block_scaled_mm`).
   - Confirmed `KeyError: 'float8_e8m0fnu'` on SM121a with
     `LINEAR_BACKEND=triton` (rocm/xpu-only gate was the cause).
   - Backport `pr-47988.diff` (source hunks: `cutlass.py` + `fp8_utils.py`).
     Overlay `triton-e8m0-sm12x` now skips when #47988's form is present.
   - Evidence comment posted on #47988 (Triton boots but is slower than b12x
     on this pair — c32 144 vs 172 tok/s — correctness fix, not a speed win).

3. **`compute_fp8_einsum_recipe` SM12x** — **#53521** OPEN.
   - `major >= 10` routes family 120 to the SM100 packed-INT32 TMA recipe.
   - Backported (`pr-53521.diff`); production evidence comment posted
     (o_proj noise → coherent France with the SM90 `(1,128,128)` recipe).

4. **b12x packed-at-store indexer gather** (HANDOFF item 18) — FIXED 2026-08-24.
   - `packed_gather_mqa_logits` reads the packed K-then-scale offsets
     (`patches/files/sm12x_b12x_kernels.py`); unit test passes with 0.0 diff
     (verified inside the production image). Local layout mismatch — stays
     a local patch; the paged kernel remains the live fast path.

5. **Real DSV4 NVFP4 writer** (584 B)
   - GLM 432/368 `scale_format=2` is the only upstream writer
   - The DSV4 584 B envelope is community (the anemll golden image ships the
     only known real-NVFP4 writer; porting it is a kernel port, not a config)

---

## Do NOT Send as "Fixes" (Measured Worse on This Pair)

- CUDA graph size 6
- Extra capture sizes
- `preinitialize_invalid_logits=False`
- Multi-row scheduled paged scorer
- Gather of packed-at-store with the interleaved FlashInfer gather (fixed: use `packed_gather_mqa_logits`, which reads the packed K-then-scale offsets)
- Expand page64 tables ×4
- Feed 1-row scheduled kernel into 8-row decode

---

## GDS / LMCache / KV offload (triaged 2026-08-24)

- **Stock CPU offload crashes on DSV4.** `kv_offload/cpu/gpu_worker.py` assumes
  flat int8 rows (`assert gpu_tensor.dtype == torch.int8; ndim == 2`) and one
  uniform `worker_kv_bytes_per_block` (`kv_offload/cpu/spec.py`); DSV4's cache
  is padded uint8 multi-group pages (576/584 B alignment + indexer group).
  Pointer drift → `cudaErrorIllegalAccess`. No open PR; upstream issue filed:
  [vllm-project/vllm#53607](https://github.com/vllm-project/vllm/issues/53607)
  (root cause + fix direction: per-group page sizes, uint8, bounds-checked
  pointers).
- **LMCache lacks `SupportsHMA`.** `LMCacheConnectorV1` doesn't implement it,
  so `--kv-transfer-config` disables the hybrid KV manager and DSV4
  sparse-MLA decode fails (`sparse_mla_sm120_decode_dsv4: num_tokens>64`).
  [LMCache#3261](https://github.com/LMCache/LMCache/issues/3261) was closed
  without an implementation. Marking SupportsHMA alone would just move the
  failure into the transfer-layout crash above.
- **One root fix unblocks both** (OffloadingConnector and LMCache): make the
  offload transfer path understand padded multi-group pages. Staged after the
  speed goal. Stock alternative that *works* with HMA: `OffloadingConnector`
  (implements `SupportsHMA`) — but it still hits the transfer crash on DSV4.

---

## DeepGEMM Pin Discipline (Not Interchangeable)

| Image | DeepGEMM Commit | Branch | Notes |
|-------|-----------------|--------|-------|
| v0.27.1 / overlay `.so` | `e21c821` | main | No SM12x MQA |
| v0.28.0rc2 / vLLM main cmake / matched-main | `8b1392b` | **nv_dev** | Has SM12x MQA |
| eugr Dockerfile | `a6b593d` | nv_dev (frozen) | MXFP4 grouped-scale regression at `f8e8fb5` / PR #384 |

**Do not mix these.** The nv_dev branch has SM12x MQA; main does not.

---

## Related Docs

- [00-index.md](00-index.md) — Quick links
- [01-hardware.md](01-hardware.md) — Why SM12x needs nv_dev
- [03-kernels-attention.md](03-kernels-attention.md) — DeepGEMM/CUTLASS gaps
- [06-deployment.md](06-deployment.md) — Build pins
- [07-gotchas.md](07-gotchas.md) — "Do not send as fixes"