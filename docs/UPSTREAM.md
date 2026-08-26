# Upstream tracker

Last verified: 2026-08-24 (afternoon). Open PRs below were re-checked
`OPEN` at that time. None merged since morning.

Live runtime is **matched vLLM main**, not the rc2 overlay. Image
`vllm-spark-0731:main-b12x` (`v0.1.dev1+ge25c586b9.d20260823`, CUDA 13.3.1,
torch 2.14 `12.1a`). Canonical Architecture & Codebase Audit: [outputs/vllm-spark-0731-docs-audit.md](../outputs/vllm-spark-0731-docs-audit.md) ([Plan](../outputs/.plans/vllm-spark-0731-docs.md)). Ops and measured numbers: [HANDOFF.md](../HANDOFF.md).
Build: [PLAN-MAIN.md](PLAN-MAIN.md).

The rc2 overlay (`vllm/vllm-openai:v0.27.1` arm64 runtime + v0.28.0rc2
Python `74a6576` + `patches/apply_overlays.py`) is the fallback if
main-b12x is down. v0.28.0rc2 has no arm64 image. "In rc2" means the
**Python tag**, not the v0.27.1 `.so` / FlashInfer wheel.

Do not open duplicate PRs. Comment with Spark evidence if a PR already
covers it. Do not upstream Spark measurements that failed on this pair
(capture size 6, gather of packed-at-store, `preinitialize_invalid_logits=False`,
multi-row scheduled paged scorer).

For the full backport patch registry, including active upstream PR backports (`pr-*.diff`), DeepGEMM backports (`deepgemm-*.diff`), and historical donor diffs (`0001*`, `0002*`, `0003*`, `b12x-utils-main.py`), see [patches/upstream/README.md](../patches/upstream/README.md).

## Pins

| Tree | ID | When |
|------|-----|------|
| vLLM matched-main (live image) | `e25c586b9` (`v0.1.dev1+ge25c586b9.d20260823`) | 2026-08-23 |
| vLLM v0.28.0rc2 (overlay fallback) | `74a6576b9b58` | 2026-08-21 06:47 UTC |
| vLLM main (PR check) | default branch | 2026-08-24 afternoon; #53425 #53521 #53522 #53055 #52499 #41834 #52708 #53574 #47988 still OPEN |
| DeepGEMM in v0.27.1 / overlay `.so` | `e21c821f39a2` (DeepGEMM **main**, ~SM90/SM100) | 2026-08-04 |
| DeepGEMM in v0.28.0rc2, vLLM main cmake, and matched-main | `8b1392b978f5` (**nv_dev** HEAD) | 2026-08-11 |
| DeepGEMM in eugr Dockerfile | `a6b593d28267` (nv_dev, frozen) | 2026-06-29 |
| FlashInfer overlay image | `0.6.16.post3` | from v0.27.1; overlay adds TOPK=192 |
| FlashInfer matched-main | git **main** (192 present) | image build |
| flashinfer-ai main | has 192 and 256 | #4380 merged 2026-08-08 |

## Matched-main live (2026-08-24)

`vllm-spark-0731:main-b12x` already left the 0.27.1 overlay behind. Remaining
gaps vs stock vLLM main are overlays and local helpers, not "switch to nightly".

Measured on 2x Spark TP=2, util 0.8, DSpark k=5, `B12X_MLA_SPARSE` +
`nvfp4_ds_mla`: France green; last 1-way median 26.90 tok/s (gather pin
~30.6); last 8-way 85.98; KV 97,737 after skipping unused page64 workspace.
PIECEWISE 11/11, FULL 7/7, DSpark backbone FULL 6/6 (sample eager).

### Keep as local overlays (do not PR as-is)

| Overlay / helper | Why skip a vLLM PR | `--only` |
|------------------|--------------------|----------|
| `B12X_MLA_SPARSE` on the 584 B DSV4 page | Stock main has no enum. eugr has a different writer mix. GLM NVFP4 is 432/368 `scale_format=2`. | `b12x-sparse` (`files/dsv4_b12x_sparse.py`) |
| Packed-at-store indexer K, page 64 | Gather of that layout is numerically wrong (France still Paris, DSpark accept 38–70%, 1-way hitch). Dual packed+interleaved sidecar OOM spark2 (~3.7 GiB). | `indexer-store-page64` |
| `plan_paged_schedule` only when `q_rows==1` | Multi-row scheduled scorer (`q_rows` 2–8) was slower than unscheduled 1023-page on this pin (1-way 25.36). Not a general rule without other-GPU data. Planning inside CUDA graphs freezes warmup seqlens. | `indexer-b12x-schedule` + helper `_B12X_SCHEDULE_MAX_Q_ROWS = 1` |
| Skip `page64_block_table_buffer` when tables are already 1024-wide | Spark-specific table width (`width*64 >= max_model_len`). KV 94,516 → 97,737. | same overlay |
| WO `torch.bmm` after fused inv-RoPE dequant | MXFP8 `wo_proj.run()` France-loops on this pair. | `o-proj-b12x` |
| DSpark backbone FULL, `_sample_sequential` eager | Graphing sample dropped accept 66.7% → 57.4% (shared `lm_head`). | `dspark-backbone-none` |
| In-graph TP all-reduce | Overlay needed clone-off-pool eager-break. Matched-main keeps France with this + `FULL_AND_PIECEWISE`. Not a one-liner for all NCCL topologies. | `ar-piecewise-ws` |

### Do not send as "fixes" (measured worse)

- CUDA graph size 6 so DSpark 1+5 does not pad to 8: 1-way 23.98, 8-way 71.52.
- Extra capture sizes `[1,2,3,...,8,...]`: KV 97k→92k, no 1-way win.
- `preinitialize_invalid_logits=False`: logprob -0.335, 1-way 15–19.
- Skip paged indexer for `m_rows<=8` and gather packed-at-store: accept collapse.
- Feed the 1-row scheduled kernel into 8-row decode: 8-way 16.29.
- Expand already-1024 page64 tables `*4`: garbage page ids.

The 1-way hole (~4 tok/s vs interleaved gather) is still a kernel/grid
question, not a missing vLLM flag. Do not gather packed storage to close it.

## Already in v0.28.0rc2

No overlay needed for these **source** pieces. SM12x still needs other overlays
(DeepGEMM `.so`, CUTLASS `.so`, FlashInfer wheel, dtype list).

| Piece | Where in rc2 | Upstream PR | Notes |
|-------|--------------|-------------|-------|
| DSpark `method=dspark` | `config/speculative.py`, `models/deepseek_v4/nvidia/dspark.py`, `v1/worker/gpu/spec_decode/dspark/` | several, including #51538 (2026-08-15), #52288 (2026-08-15) | 0731 still locks **k=5** |
| `FLASHINFER_MLA_SPARSE_DSV4` | `models/deepseek_v4/nvidia/flashinfer_sparse.py` | #51538 | SM12x accepts `fp8` / `fp8_e4m3` / `fp8_ds_mla` only. Not `nvfp4_ds_mla`. Kernel block `[256]`. |
| Linear `--linear-backend b12x` | `LinearBackend` includes `"b12x"`; `warmup/b12x_warmup.py` warms FP8/MXFP4/NVFP4 linears | #52016 merged **2026-08-14** (before rc2) | Live kernel `B12xFp8BlockScaledMMKernel` |
| Linear `flashinfer_b12x` | same `LinearBackend` | rc2 | CuteDSL NVFP4 GEMM. Different from MoE `b12x`. |
| MoE `flashinfer_b12x` | `MoEBackend` | rc2 | CuteDSL fused MoE. **Not** MXFP4 `B12xExperts`. |
| mHC siblings TileLang fallback | `mhc_pre_tilelang`, `mhc_fused_post_pre_tilelang` use `is_deep_gemm_supported()` | already in rc2 | **`mhc_pre_broadcast_tilelang` is not guarded** (see below) |
| `support_deep_gemm()` includes family 120 | `platforms/cuda.py` | rc2 | Matches rc2 cmake (`nv_dev` SM12x). This image still ships the v0.27.1 **main** DeepGEMM `.so`, so the gate is a footgun here. |
| KVBlockZeroer non-uniform pages | `v1/worker/utils.py` | #49704 merged 2026-07-24 | rc2 still **`assert shape[block_dim] % ratio == 0`**. SM12x 64-vs-256 hits that. |

## Not in rc2, merged on vLLM main after the tag

Cherry-pick / overlay until the next release that contains the merge.

| Piece | Merged | PR | Overlay |
|-------|--------|-----|---------|
| MoE `--moe-backend b12x` (`MoEBackend`, `fused_moe/b12x.py`, `quantization/utils/b12x_moe.py`, mxfp4 oracle `B12X_MXFP4_*`, `get_b12x_fused_moe`, `B12xWarmupUnit`) | 2026-08-21 15:04 UTC, **8h after rc2** | [#52018](https://github.com/vllm-project/vllm/pull/52018) | `copy_new_modules`, `patch_moe_backend`, `patch_envs`, `patch_utils_b12x`, `patch_mxfp4_oracle`, `patch_mxfp4_process_weights` |
| Newer KVBlockZeroer (no unaligned assert) | after rc2 (main rewrite) | evolved past #49704 | `patch_kv_zeroer_skip` (ratio=1). Do not re-PR #49704. |

## Not in rc2, still open on vLLM main

Same bug in the tag **and** on main today. Comment or small PR. Do not duplicate.

| Piece | rc2 | main 2026-08-24 | Action |
|-------|-----|-----------------|--------|
| `mhc_pre_broadcast_tilelang` unguarded `tf32_hc_prenorm_gemm` | unguarded | still unguarded | [#53055](https://github.com/vllm-project/vllm/pull/53055) (also CUTLASS + sm121 carve-out). Older [#50645](https://github.com/vllm-project/vllm/pull/50645) needs-rebase. Backport: `pr-53055.diff`; Overlay: `patch_mhc`. Comment only; do not duplicate. |
| CUTLASS FP8 `is_supported()` ignores SM12x | `CutlassFp8BlockScaledMMKernel` returns True if `CUTLASS_BLOCK_FP8_SUPPORTED` | #53055 still open | Backport: `pr-53055.diff`; Overlay: `patch_cutlass_sm12x_guard`. Same PR as mHC. |
| `compute_fp8_einsum_recipe`: `major >= 10` → SM100 packed INT32 | yes | **still yes** | [#52357](https://github.com/vllm-project/vllm/pull/52357) **closed** 2026-08-23 (Triton slice abandoned). Recipe-only follow-up: [#53521](https://github.com/vllm-project/vllm/pull/53521). Backport: `pr-53521.diff`; Overlay: `patch_einsum_sm12x_recipe`. |
| DSV4 kernel block `[256]` on SM12x | `[256]` on sparse MLA, FlashInfer DSV4, V4 indexer | **still `[256]`** | [#53425](https://github.com/vllm-project/vllm/pull/53425) OPEN (rebased; DCO passes). Backport: `pr-53425.diff`; Overlay: `patch_dsv4_sm12x_block_size`. |
| Indexer paged MQA metadata uses `has_deep_gemm()` not `is_deep_gemm_supported()` | yes | still that pattern | Not in #41834 / #53055. Opened [#53522](https://github.com/vllm-project/vllm/pull/53522) (`is_deep_gemm_supported()` + `num_states in (32, 64)`). Backport: `pr-53522.diff`; Overlay: `patch_indexer_deepgemm_guard`. |
| DSpark SM120 spec-decode query rank / `num_tokens > 64` | #51538 is in rc2 (backend + top-k selection). Flat 3-D spec query may remain. | [#52499](https://github.com/vllm-project/vllm/pull/52499) open | Backport: `pr-52499.diff`; Comment only. We did not need this after TOPK=192. |
| FlashInfer eidx contiguity (C128A builder) | `_build_c128a_metadata` view of a width-sliced `global_decode_buffer`; DSpark batches >64 tokens crash at boot | **still unpatched** 2026-08-24; [#53574](https://github.com/vllm-project/vllm/pull/53574) OPEN | Backport: `pr-53574.diff`; Overlay: `flashinfer-eidx-contig`. C4A branch verified contiguous — no C4A bug. |
| Triton E8M0 upcast gated on rocm/xpu | `KeyError: 'float8_e8m0fnu'` on SM12x | **still gated** 2026-08-24; [#47988](https://github.com/vllm-project/vllm/pull/47988) OPEN (unconditional upcast) | Backport: `pr-47988.diff`; Overlay: `triton-e8m0-sm12x` skips when #47988 form present. |
| SM12x DSv4 umbrella | partial (backend exists) | [#41834](https://github.com/vllm-project/vllm/pull/41834) needs-rebase | Comment only. `sm12x_mqa.py` lives there. Pointed at the focused PRs. |
| `fp8_einsum` SM12x Python dequant | rc2 is DeepGEMM-or-missing | #52357 Triton path closed | Overlay: `patch_fp8_einsum_fallback`. Not upstreamed; `float8_e8m0fnu` dies on this image. Recipe is #53521. |

## FlashInfer (not vLLM rc2)

vLLM does not vendor `_DECODE_DSV4_DISPATCH`. The table lives in the **wheel**.

| Piece | flashinfer-ai main | Overlay image (`0.6.16.post3`) | Overlay |
|-------|--------------------|-----------------------------|---------|
| TOPK 192 (DSpark k=5, window 128 → `ceil(133/64)*64`) | [#4380](https://github.com/flashinfer-ai/flashinfer/pull/4380) merged 2026-08-08 (192 **and** 256) | Stock wheel had no 192. Overlay has 192 after patch, still no 256. Matched-main FlashInfer is git main (192 present). **v0.6.17 (vLLM nightly pin) still `{128,512,1024}`.** | `patch_flashinfer_dsv4_dispatch` + `patch_flashinfer_dsv4_cu_dispatch` |
| Page block 64 | `_DECODE_DSV4_PAGE_BLOCK_SIZE = 64` on main | SM120 decode is 64-token pages | vLLM must advertise 64: #53425 |

Do not open another FlashInfer PR.

## Local / skip (not upstream as-is)

| Overlay | Why skip |
|---------|----------|
| Blanket `is_deep_gemm_supported()` False on family 120 | Needed while the overlay image keeps the v0.27.1 DeepGEMM **main** `.so`. rc2 cmake already wants nv_dev. Matched-main compiles nv_dev. Do not upstream the kill. |
| `nvfp4_ds_mla` 584 B alias + MLA guard exact `"nvfp4"` | rc2 `startswith("nvfp4")` would reject it. Main has `nvfp4` / `nvfp4_4over6` only. No NVFP4 CUDA writer in this image. |
| DSV4 `supports_combination` +`nvfp4_ds_mla` | Lets FLASHINFER select the envelope name. Same skip. |
| MQA ReLU / no `.item()` / b12x MQA / Triton SWA insert | SM12x fallbacks. #41834 `sm12x_mqa.py` is the landing zone. |
| DSpark skip CUDA graphs, `lm_head` restore, logit dump | Overlay diagnostics. Matched-main graphs the backbone and leaves sample eager. |
| TP all-reduce eager-break + clone off graph pool | Overlay PIECEWISE France. Matched-main uses in-graph AR instead (`ar-piecewise-ws`). |
| Packed-at-store indexer, `q_rows==1` schedule gate, WO bmm, DSpark sample eager | Spark measurements. See Matched-main live above. |
| CUDA graph size 6 / gather packed pages / `preinitialize_invalid_logits=False` | Measured worse. Do not PR. |

## DeepGEMM (eugr Dockerfile vs vLLM pins)

eugr **rebuilds** DeepGEMM (`DEEPGEMM_SRC_DIR`) for SM12x. This overlay image
does **not**. Runtime `.so` is whatever `vllm/vllm-openai:v0.27.1` compiled.

Three different trees:

| Tree | SHA | Branch | SM12x MQA in `attention.hpp` |
|------|-----|--------|------------------------------|
| This image / v0.27.1 cmake | `e21c821f39a2` | DeepGEMM **main** (~2 commits off) | No. `DG_HOST_ASSERT` arch 9/10 only. Live crash: `attention.hpp:122`. |
| eugr Dockerfile | `a6b593d28267` | **nv_dev**, frozen | Yes (`arch_major == 12`). |
| v0.28.0rc2 and vLLM main cmake | `8b1392b978f5` | **nv_dev** HEAD | Yes. Comment: "Pinned to the tip of the nv_dev branch (SM120 support)." |

**2026-08-25: `8b1392b` is a REGRESSION for SM12x fp8 linear** (see
[docs/knowledge/09](knowledge/09-golden-deepgemm.md)): it dropped the
pure-fp8 1d1d kernels and aliased `fp8_gemm_nt = fp8_fp4_gemm_nt`
(`gemm.hpp:851`), so fp8 weights feed the fp4 kernel = silent corruption on
GB10. Matched-main is now pinned back to `a6b593d` (the eugr/golden freeze);
a vLLM PR to move the cmake tag back to `a6b593d` is prepared
(branch `fix-deepgemm-sm12x-fp8-regression` on maci0/vllm, PR opened after
cluster validation).

SM12x kernels live on DeepGEMM `nv_dev` ([PR #324](https://github.com/deepseek-ai/DeepGEMM/pull/324) / issue #324). They are **not** on DeepGEMM `main`.

### What eugr's Dockerfile actually pins

Source: [eugr/spark-vllm-docker `Dockerfile`](https://github.com/eugr/spark-vllm-docker/blob/main/Dockerfile)
(lines ~31, ~284-288, ~637). Pin commit:
[3fba416](https://github.com/eugr/spark-vllm-docker/commit/3fba416a3560e35e92fa9711a11b635ab8716d88)
(2026-07-20, "Pin DeepGEMM to avoid regression").

| Item | eugr | In v0.28.0rc2? | Status 2026-08-23 |
|------|------|----------------|-------------------|
| Rebuild DeepGEMM from source as `DEEPGEMM_SRC_DIR` | yes, `TORCH_CUDA_ARCH_LIST=12.1a` | cmake **can** (`DEEPGEMM_SRC_DIR` or FetchContent). Official rc2 Python expects nv_dev `8b1392b`. No arm64 rc2 image. | We skip rebuild. Overlay: `patch_deep_gemm_sm12x_guard`. |
| Freeze at `a6b593d` instead of nv_dev tip | "SM121 DeepSeek-V4 MXFP4 grouped scale-factor regression first observed at nv_dev `f8e8fb5` (PR #384); last known good" | **No.** rc2/main FetchContent tag is `8b1392b` (PR #396 SiTU, 2026-08-11). | `8b1392b` is only 3 SiTU commits after `f8e8fb5`. It does **not** claim to fix grouped MXFP4 scales. eugr still frozen. Do not bump blindly. |
| `DG_JIT_USE_NVRTC=0` | build + runner. "disable for conflicts with DeepGEMM" / "compatibility with DeepGEMM changes" | Not a vLLM source pin. nv_dev JIT vs NVRTC. | Irrelevant here while DeepGEMM is forced off. Keep if anyone rebuilds nv_dev. |
| `transform_sf_into_required_layout` missing `arch_major=12` for `(gran_mn=1, gran_k=32)` | nv_dev pin already has `arch_major == 12` | DeepGEMM **main** still SM100-only | [deepseek-ai/DeepGEMM#372](https://github.com/deepseek-ai/DeepGEMM/issues/372) / [#403](https://github.com/deepseek-ai/DeepGEMM/pull/403). Merged in `nv_dev 8b1392b`. Backported as `deepgemm-pr-403.diff` for clean builds. |
| DeepGEMM SM120 pure-FP8 GEMM port (`sm100_fp8_gemm_1d1d` kernel to `nv_dev`) | anemll 2.5.0 port | staged local port | Tracked as `deepgemm-fp8-1d1d-port.diff` (applied in `docker/Dockerfile.main` before compilation). |
| CUDA 13 `CUDA_SUPPORTED_ARCHS` drops 12.1 | opt-in `patch_vllm_preserve_sm12x_target.py` | rc2 CUDA 13 DeepGEMM list uses family `12.0f`, not `12.1a` | Already in [#52708](https://github.com/vllm-project/vllm/pull/52708) (and older [#38484](https://github.com/vllm-project/vllm/pull/38484)). Comment only; do not duplicate. Not this overlay image. |

**Do not** open another DeepGEMM SM12x-enable PR. #372 covers main. nv_dev already has the kernels. vLLM #41062 (extend DeepGEMM MoE gates to SM12x) is closed, not merged; cmake moved to nv_dev instead.

**Do not** rebuild DeepGEMM into this overlay image without leaving the v0.27.1 `.so` model. If that happens, re-measure eugr's grouped MXFP4 pin (`a6b593d` vs `8b1392b`) on 0731 experts before trusting rc2's FetchContent tag.

### Other live patches in eugr's Dockerfile (not DeepGEMM)

Nightly source-build workarounds. Most cited vLLM PRs are already **merged**. The patch remains because eugr tracks `main`, not rc2.

| Patch | Cited PR / issue | Upstream now | Our image |
|-------|------------------|--------------|-----------|
| SM12x `cooperative_topk` → `persistent_topk` ("invalid argument") | vLLM #43008 merged 2026-06-23 | Kernel still in rc2 (`csrc/libtorch_stable/cooperative_topk.cu`) | Not applied. MoE is b12x. |
| Gemma4 MTP embedding share | #43957 / issue #47794 **closed** | fix landed | N/A (0731) |
| DiffusionGemma tensor causal | #47914 merged | still patched on their nightly | N/A |
| AutoGPTQ symmetric MoE qzeros | #43409 merged | still patched on their nightly | N/A |
| MiniMax QK RMSNorm IPC off | #43410 merged | still patched | N/A |
| RoutedExperts `weight_shape` scalar | #43362 merged | still patched | N/A |
| `topk_softplus_sqrt` XPU no-op | #49408 / fix #49452 merged 2026-07-22 | **in rc2** | skip |
| fastsafetensors sort | issue #34180 closed | commented out in Dockerfile | skip |
| FlashInfer autotune revert #41524 | merged, revert commented out | skip | skip |

## If we switched to vLLM nightly

Research date: 2026-08-23. Historical. Matched-main is already live as
`vllm-spark-0731:main-b12x` (see HANDOFF). Remaining keep/add overlays
are the Matched-main live table, not a Hub `nightly` pull.

**Arm64 images exist.** `vllm/vllm-openai:nightly` and `nightly-aarch64` were pushed
2026-08-22 (`e9d1398`, `[Bugfix][Kimi K3] #53327`). That is newer than the
v0.27.1 **release** this recipe uses. There is still no v0.28.0rc2 arm64 tag.

**Default `nightly` is CUDA 12.9.** Docs: wheels.vllm.ai/nightly is cu129.
Hub pairs `nightly` with `cu129-nightly` the same day. `cu130-nightly` last
moved **2026-04-23** (stale). CUDA 13 arm64 is
`nightly-dev-arm64-cu13.0.1-<sha>` (last seen `728d3ad`, 2026-08-19), not
the `nightly` tag. Overlay fallback is PyTorch **cu130**. Matched-main is
CUDA 13.3.1 + torch 2.14 `12.1a`. Do not pull `cu130-nightly`.

**wheels.vllm.ai nightly has no aarch64 wheel** (x86_64 only). ARM means
Docker, or a from-source build on the Spark.

**Official arch list is `12.0`, not `12.1a`.**
vLLM's `docker/versions.json` `TORCH_CUDA_ARCH_LIST` default:
`7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0`. CUDA 13 CMake
`CUDA_SUPPORTED_ARCHS` also drops 12.1 (family 12.0). eugr compiles
`12.1a` on purpose. Family `12.0f` is supposed to run on GB10; that is
unproven on this pair.

### What nightly would actually drop vs keep

Assume a **matched** image (Python + `.so` from the same main commit), plus
`b12x==1.2.6` still installed. Not "overlay nightly Python onto v0.27.1".

| Overlay / gap | On main / nightly 2026-08-22 | Keep? |
|---------------|------------------------------|-------|
| MoE `--moe-backend b12x` (#52018) | **in tree** | drop cherry-pick |
| DeepGEMM pin | cmake `8b1392b` **nv_dev**, compiled with the image | drop `patch_deep_gemm_sm12x_guard` **only after** MQA/mHC actually run. eugr still frozen at `a6b593d` for MXFP4 grouped scales |
| KVBlockZeroer unaligned assert | rewritten on main | likely drop |
| FlashInfer TOPK 192 | **no.** docker pin `FLASHINFER_VERSION=0.6.17`. Tag v0.6.17 dispatch is still `{128,512,1024}`. 192/256 are on flashinfer **main** (#4380), not in 0.6.17 | **keep** overlay or bump FlashInfer to main / 0.6.18 nightly |
| DSV4 kernel block 64 | #53425 still OPEN (DCO fixed 2026-08-24) | **keep** |
| einsum SM12x recipe | #52357 closed; #53521 OPEN | **keep** (wrong recipe + live DeepGEMM is worse than the PyTorch fallback) |
| Indexer paged MQA DeepGEMM gate | #53522 OPEN | **keep** until merged |
| mHC broadcast / CUTLASS SM12x | #53055 still OPEN | keep CUTLASS guard. mHC might use nv_dev `sm120_tf32_hc_prenorm_gemm` if the `.so` is real; unproven |
| MQA ReLU / graph-safe / b12x MQA | #41834 still OPEN | **keep** unless DeepGEMM SM12x MQA is measured equal to `fp8_mqa_logits_torch` |
| `nvfp4_ds_mla` | still not a cache dtype | **keep** if we stay on that name |
| TP all-reduce clone / DSpark skip graphs | local | **keep** for PIECEWISE France |
| `cooperative_topk` SM12x invalid argument | kernel still in tree (#43008) | b12x MoE avoids it; Triton/FI MoE may not |

### What would not help

- Overlaying nightly Python onto the v0.27.1 `.so`. Larger ABI drift than rc2-on-0.27.1.
- Expecting official nightly to be Spark-tuned. eugr's image is already "main + 12.1a + rebuilt DeepGEMM + Spark patches". That is a different product (`B12X_MLA_SPARSE`, `pin.eugr-b12x.env`).
- Enabling DeepGEMM MoE on 0731 MXFP4 experts at pin `8b1392b` without re-measuring eugr's grouped-scale regression.

### If leaving 0.27.1 anyway

Build on the Spark from `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04`.
Compile NCCL (`sm_121`) and PyTorch `release/2.14` (`12.1a`), then
`use_existing_torch.py` + vLLM main. Do not pip cu132 wheels. Do not use
NGC pytorch or `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0).
`torch_cuda_arch_list='12.1a'`, decide DeepGEMM `a6b593d` vs `8b1392b` on
0731 experts, install b12x / FlashInfer / InstantTensor / fastsafetensors /
LMCache from git (cutlass metadata rewritten to 4.7.0), override vLLM
cuda.txt 0.6.17 / cutlass 4.6.2 / tilelang 0.1.12, keep the quality overlays
that are still open.
Pulling Hub `nightly` (cu129, arch 12.0) is the cheaper experiment, not
the production path. Full pin: [PLAN-MAIN.md](PLAN-MAIN.md) sections 4.1–4.4.

## eugr/spark-vllm-docker (recipes / issues)

Different stack (`B12X_MLA_SPARSE`, their nightly + rebuilt nv_dev DeepGEMM). Not an rc2 gap.

| Item | Status | Link |
|------|--------|------|
| 0731 recipe on their main | already there | `recipes/deepseek-v4-flash-0731.yaml` |
| DSpark topk round 256→128 | obsolete vs FlashInfer #4380; 0731 needs 192 | [PR #319](https://github.com/eugr/spark-vllm-docker/pull/319) |
| Graph IMA / 2-node hang | their B12X GEMM vs our PYNCCL; earlyoom at util 0.85 | [#349](https://github.com/eugr/spark-vllm-docker/issues/349), [#352](https://github.com/eugr/spark-vllm-docker/issues/352), [#348](https://github.com/eugr/spark-vllm-docker/issues/348) |

## Comments / PRs we posted

| Date | Target | What | URL |
|------|--------|------|-----|
| 2026-08-23 | vllm #52357 | SM12x einsum recipe | https://github.com/vllm-project/vllm/pull/52357#issuecomment-5383962367 |
| 2026-08-23 | vllm #53055 | mHC + CUTLASS + sm121 | https://github.com/vllm-project/vllm/pull/53055#issuecomment-5383962429 |
| 2026-08-23 | vllm #50645 | point at #53055 | https://github.com/vllm-project/vllm/pull/50645#issuecomment-5383962502 |
| 2026-08-23 | vllm #41834 | 2-node 0731 field report | https://github.com/vllm-project/vllm/pull/41834#issuecomment-5383963377 |
| 2026-08-23 | vllm #52499 | DSpark k=5 / TOPK 192 | https://github.com/vllm-project/vllm/pull/52499#issuecomment-5383963551 |
| 2026-08-23 | vllm #53425 | **opened** SM12x DSV4 kernel block 64 | https://github.com/vllm-project/vllm/pull/53425 |
| 2026-08-24 | vllm #53425 | rebase onto main; DCO sign-off | https://github.com/vllm-project/vllm/pull/53425 |
| 2026-08-24 | vllm #53521 | **opened** SM12x Hopper fp8_einsum recipe | https://github.com/vllm-project/vllm/pull/53521 |
| 2026-08-24 | vllm #53522 | **opened** indexer paged MQA DeepGEMM gate | https://github.com/vllm-project/vllm/pull/53522 |
| 2026-08-24 | vllm #52357 | closed Triton path; point at #53521 | https://github.com/vllm-project/vllm/pull/52357#issuecomment-5390775349 |
| 2026-08-24 | vllm #52708 | CUDA 13 `12.1` already in this PR; no duplicate | https://github.com/vllm-project/vllm/pull/52708#issuecomment-5390775145 |
| 2026-08-24 | vllm #41834 | point at focused PRs | https://github.com/vllm-project/vllm/pull/41834#issuecomment-5390775505 |
| 2026-08-24 | vllm #53055 | indexer gate is #53522, not a duplicate mHC PR | https://github.com/vllm-project/vllm/pull/53055#issuecomment-5390775697 |
| 2026-08-23 | eugr #319 | FlashInfer #4380; 0731 needs 192 | https://github.com/eugr/spark-vllm-docker/pull/319#issuecomment-5383963577 |
| 2026-08-23 | eugr #349 | earlyoom + graphs | https://github.com/eugr/spark-vllm-docker/issues/349#issuecomment-5383964228 |
| 2026-08-23 | eugr #352 | shm_broadcast hang | https://github.com/eugr/spark-vllm-docker/issues/352#issuecomment-5383964230 |
| 2026-08-23 | eugr #348 | pointer to #352 | https://github.com/eugr/spark-vllm-docker/issues/348#issuecomment-5383970324 |
| 2026-08-24 | vllm #53574 | C128A eidx root-cause confirmation + C4A-branch-contiguous finding | https://github.com/vllm-project/vllm/pull/53574#issuecomment-5398599445 |
| 2026-08-24 | vllm #47988 | SM121a `KeyError: float8_e8m0fnu` confirmation; backporting the unconditional upcast | https://github.com/vllm-project/vllm/pull/47988#issuecomment-5398601080 |
| 2026-08-24 | vllm #53521 | production confirmation: o_proj noise → coherent with SM90 recipe | https://github.com/vllm-project/vllm/pull/53521#issuecomment-5398604045 |
| 2026-08-25 | vllm #53680 | **opened** DeepGEMM pin-back to a6b593d (SM12x fp8 regression in 8b1392b) | https://github.com/vllm-project/vllm/pull/53680 |
| 2026-08-25 | DeepGEMM #417 | **opened** regression issue (removed kernels + fp4 alias) | https://github.com/deepseek-ai/DeepGEMM/issues/417 |
| 2026-08-24 | vllm #53607 | **opened** DSV4 CPU KV-offload flat-layout root-cause issue (GDS/LMCache track) | https://github.com/vllm-project/vllm/issues/53607 |
