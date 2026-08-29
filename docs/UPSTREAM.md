# Upstream tracker

Last verified: **2026-08-29**. Recipe repo (public, reproducible):
**https://github.com/maci0/vllm-spark-0731** — all overlays, backport diffs,
knowledge docs, and measured numbers referenced below live there.

**v0.28.1rc0 checked 2026-08-29 (tag 2026-08-27, commit `79651d60`): none
of our PRs merged** (all 12 still OPEN: #53680 #53522 #53425 #53271 #46716
#52941 #53574 #47988 #53055 #41834 #52499 #52708; DeepGEMM #419 #337 #403
OPEN). Relevant new commits in the range (other people's work): #53649
(Blackwell triton batch-invariance, 33.6% E2E), #52823 (DSV4 adaptive topk
width), #53040 (DSV4 shared experts -> MegaMoE), #52809 (DSpark inheritance
scoped to DSV4), #52795/#52783 (DSV4 adaptive verification), #53326 (b12x
modules before Dynamo tracing). None replaces an overlay; **stay on
v0.28.0** (rebasing adds conflict risk in the DSV4/MoE/indexer files we
patch). PR triage 2026-08-29: #53574 lucifer1004 implemented the SM120
full-width-decode fix + `/ci run`; failing step is an H200 infra flake
(MOSS-Audio timeout), awaiting maintainer retry — no action needed.
#47988 kitch2400 independently confirmed required on GB10 (still red
mergify conflict — author @waynehacking8 must rebase). #53055 author
force-pushed `e1d67cc` (DCO cleared, tests pass), awaiting review label.

Live runtime (2026-08-28): **v0.28.0 release** (`2cf0a6915ce5`, "DeepSeek V4:
sparse MLA works end-to-end for plain decode, MTP, and DSpark speculative
decoding (#51538)") as image `vllm-spark-0731:main-b12x-028-rdma` =
`main-b12x-028-p1` (full CUDA 13.3.1 / torch 2.14 `12.1a` build) + rdma-core
v54 libmlx5 overlay (NCCL RoCE; see docs/knowledge/05-performance.md) +
`patches/files` donors + warmup ext. Canonical audit:
[outputs/vllm-spark-0731-docs-audit.md](../outputs/vllm-spark-0731-docs-audit.md).
Ops and measured numbers: [HANDOFF.md](../HANDOFF.md). Build:
[PLAN-MAIN.md](PLAN-MAIN.md).

Measured 2026-08-28 (2x GB10, TP=2, DSpark k=5, `B12X_MLA_SPARSE`,
`nvfp4_ds_mla`, util 0.8): **c1 steady-state 40.2-43.5 tok/s, c8 117, c16
183, c24 261, c32 306.8 agg** (SM util 95%). France greedy
`' Paris...'` logprob -0.254 (matches golden einsum exactly); o_proj decode
bmm active with **0 fallbacks**.

The rc2 overlay (`vllm/vllm-openai:v0.27.1` arm64 runtime + v0.28.0rc2
Python `74a6576` + `patches/apply_overlays.py`) is the historical fallback;
the v0.28.0 release stack superseded it. "In v0.28.0" means the **release
Python tag**, not the v0.27.1 `.so` / FlashInfer wheel.

Do not open duplicate PRs. Comment with Spark evidence if a PR already
covers it. Do not upstream Spark measurements that failed on this pair
(capture size 6, gather of packed-at-store, `preinitialize_invalid_logits=False`,
multi-row scheduled paged scorer).

For the full backport patch registry, including active upstream PR backports (`pr-*.diff`), DeepGEMM backports (`deepgemm-*.diff`), and historical donor diffs (`0001*`, `0002*`, `0003*`, `b12x-utils-main.py`), see [patches/upstream/README.md](../patches/upstream/README.md).


## Pins

| Tree | ID | When |
|------|-----|------|
| vLLM **v0.28.0 release** (live image `main-b12x-028-rdma`) | `2cf0a6915ce5` | 2026-08-27 (rebase) |
| vLLM v0.28.0rc2 (historical overlay fallback) | `74a6576b9b58` | 2026-08-21 06:47 UTC |
| vLLM main (PR check) | default branch | 2026-08-28: #53425 #53522 #53680 #53055 #52499 #41834 #52708 #53574 #47988 still OPEN; #53521 #53898 CLOSED 2026-08-27 (einsum misread resolved as misdiagnosis — kernel correct, see table); **#46716 rebased 2026-08-28** |
| DeepGEMM in v0.27.1 / overlay `.so` | `e21c821f39a2` (DeepGEMM **main**, ~SM90/SM100) | 2026-08-04 |
| DeepGEMM in v0.28.0rc2, vLLM main cmake, and matched-main | `8b1392b978f5` (**nv_dev** HEAD) | 2026-08-11 |
| DeepGEMM in eugr Dockerfile | `a6b593d28267` (nv_dev, frozen) | 2026-06-29 |
| FlashInfer overlay image | `0.6.16.post3` | from v0.27.1; overlay adds TOPK=192 |
| FlashInfer matched-main | git **main** (192 present) | image build |
| flashinfer-ai main | has 192 and 256 | #4380 merged 2026-08-08 |

## Matched-main live (2026-08-24, historical)

> Superseded 2026-08-27/28 by the **v0.28.0 release** stack (see header). The
> overlays below still apply (they are the same `apply_main` set the v0.28.0
> image uses); the numbers are the pre-v0.28.0 baseline.

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

## Already in v0.28.0 release

No overlay needed for these **source** pieces (the v0.28.0 changelog: "DeepSeek
V4 sparse MLA works end-to-end for plain decode, MTP, and DSpark speculative
decoding (#51538), AMD Quark NVFP4 (#47972), sparse top-k metadata kernel
optimizations (#52084, #51967), narrowed eager CUDA graph regions (#51430,
#52401)"). SM12x still needs other overlays (DeepGEMM `.so`, CUTLASS `.so`,
FlashInfer wheel, dtype list).

| Piece | Where in v0.28.0 | Upstream PR | Notes |
|-------|--------------|-------------|-------|
| DSpark `method=dspark` | `config/speculative.py`, `models/deepseek_v4/nvidia/dspark.py`, `v1/worker/gpu/spec_decode/dspark/` | several, including #51538 (2026-08-15), #52288 (2026-08-15) | 0731 still locks **k=5** |
| `FLASHINFER_MLA_SPARSE_DSV4` | `models/deepseek_v4/nvidia/flashinfer_sparse.py` | #51538 | SM12x accepts `fp8` / `fp8_e4m3` / `fp8_ds_mla` only. Not `nvfp4_ds_mla`. Kernel block `[256]`. |
| MoE `--moe-backend b12x` + linear b12x + `b12x_warmup.py` | `MoEBackend`, `fused_moe/b12x.py`, `LinearBackend`, `warmup/b12x_warmup.py` | #52018 (merged 2026-08-21, **8h after the rc2 tag**; in the release) | Overlay `patch_moe_backend` / `patch_utils_b12x` / `patch_mxfp4_oracle` **auto-skip** on v0.28.0 ("already applied"). Live kernel `B12xFp8BlockScaledMMKernel` / b12x MoE. |
| mHC siblings TileLang fallback | `mhc_pre_tilelang`, `mhc_fused_post_pre_tilelang` use `is_deep_gemm_supported()` | already in release | **`mhc_pre_broadcast_tilelang` is not guarded** (see below); not exercised by the nvidia 0731 model (embed pre-broadcasts to 3D). |
| `support_deep_gemm()` includes family 120 | `platforms/cuda.py` | in release | Matches release cmake (`nv_dev` SM12x). This image ships the v0.27.1 **main** DeepGEMM `.so`, so the gate is a footgun here (`patch_deep_gemm_sm12x_guard`). |
| KVBlockZeroer rewrite (no unaligned assert) | `v1/worker/utils.py` (address-table zeroer) | evolved past #49704 | `patch_kv_zeroer_skip` still applied defensively. |

## 2026-08-28 session findings (new)

1. **o_proj decode bmm — root cause + correct dequant (perf, not
   correctness).** vLLM's DSV4 `wo_a` post-loads via
   `deepgemm_post_process_fp8_weight_block` with `is_bmm=True` into **3D**
   `[G, R, D]` (TP=2: `[4, 1024, 4096]`) and the scale becomes DeepGEMM's
   **MN-major TMA-aligned packed UE8M0** `[G, R, D/512]` int32 (4 ue8m0
   exponents per int32, byte `j` = k-block `4i+j`, rows per-gran-block
   broadcast). The decode bmm dequant must unpack that layout; multiplying
   by the raw int32 produces garbage (measured). Fixed + validated
   numerically against the reference block-scale expansion. Local overlay
   `o-proj-b12x` (`patches/files/sm12x_b12x_kernels.py`); not upstreamable
   as-is (the upstream einsum path is correct — this only avoids the slow
   einsum at decode). Measured: c1 29.8 → 33-38 (TTFT-incl.) /
   40.2-43.5 steady-state.
2. **`deepseek_v4_mhc_warmup` is a silent no-op on the NVIDIA DSV4 layer —
   OPEN UPSTREAM BUG (PR opened 2026-08-28, see Comments table).** The
   v0.28.0/main nvidia `DeepseekV4DecoderLayer` has no `hc_pre`/`hc_post`
   methods (it calls `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang`
   directly), so the warmup's layer gate never matches and every boot's
   first request pays the TileLang JIT (~30-120 s; the c16 collapse).
   Local fix: `patches/files/dsv4_warmup_ext.py` (drives the real layer
   calls + `hc_head_op` + DSpark gumbel sampler) — c16 44.5 → 183.0,
   c32 306.8 agg.
3. **c16 collapse + 300+ agg both resolved** — see header numbers. SM util
   went 47% → 95% (compute-bound now).
4. **Recipe repo is public**: https://github.com/maci0/vllm-spark-0731
   (pushed 2026-08-28; all PRs above link it as the reproducible source).



## Not in v0.28.0, still open on vLLM main

Same bug in the release **and** on main today. Comment or small PR. Do not duplicate.

| Piece | v0.28.0 | main 2026-08-28 | Action |
|-------|-----|-----------------|--------|
| `mhc_pre_broadcast_tilelang` unguarded `tf32_hc_prenorm_gemm` | unguarded | still unguarded | [#53055](https://github.com/vllm-project/vllm/pull/53055) (also CUTLASS + sm121 carve-out). Older [#50645](https://github.com/vllm-project/vllm/pull/50645) needs-rebase. Backport: `pr-53055.diff`; Overlay: `patch_mhc`. Comment only; do not duplicate. Not exercised by the 0731 nvidia model (3D pre-broadcast), kept defensively. |
| CUTLASS FP8 `is_supported()` ignores SM12x | `CutlassFp8BlockScaledMMKernel` returns True if `CUTLASS_BLOCK_FP8_SUPPORTED` | #53055 still open | Backport: `pr-53055.diff`; Overlay: `patch_cutlass_sm12x_guard`. Same PR as mHC. |
| `compute_fp8_einsum_recipe`: `major >= 10` → SM100 packed INT32 | yes | stock config is correct | [#53521](https://github.com/vllm-project/vllm/pull/53521) **CLOSED 2026-08-27** (not needed). Stock SM12x `(1,1,128)` + `tma_aligned_scales=True` verified numerically correct on GB10 (einsum mean_rel 0.000000 vs bf16 ref; E2E France coherent). |
| `fp8_einsum` on SM12x | yes | **stock kernel path is CORRECT** with packed E8M0 scales + `(1,1,128)` (mean_rel 0.000000 on GB10) | [#53898](https://github.com/vllm-project/vllm/pull/53898) **CLOSED 2026-08-27** (fallback not needed). Real fix upstream: deepseek-ai/DeepGEMM **#337**. |
| DSV4 kernel block `[256]` on SM12x | `[256]` on sparse MLA, FlashInfer DSV4, V4 indexer | **still `[256]`** | [#53425](https://github.com/vllm-project/vllm/pull/53425) OPEN — fixed 2026-08-26 (ed71de5): `indexer → vllm.models.deepseek_v4.sparse_mla` module-level import broke `vllm._aiter_ops` cold start (kitch2400 report); lazy import inside `get_supported_kernel_block_sizes()`. Backport: `pr-53425.diff`; Overlay: `patch_dsv4_sm12x_block_size`. |
| Indexer paged MQA metadata uses `has_deep_gemm()` not `is_deep_gemm_supported()` | yes | still that pattern | [#53522](https://github.com/vllm-project/vllm/pull/53522) OPEN (`is_deep_gemm_supported()` + `num_states in (32, 64)`). **ivanusto reviewed 2026-08-24: test passed, gate scoped correctly**. Backport: `pr-53522.diff`; Overlay: `patch_indexer_deepgemm_guard`. |
| DSpark SM120 spec-decode query rank / `num_tokens > 64` | #51538 in release (backend + top-k). Flat 3-D spec query may remain. | [#52499](https://github.com/vllm-project/vllm/pull/52499) open | Comment only. Not needed after TOPK=192. |
| FlashInfer eidx contiguity (C128A builder) | `_build_c128a_metadata` view of a width-sliced `global_decode_buffer`; DSpark batches >64 tokens crash at boot | **still unpatched** 2026-08-24; [#53574](https://github.com/vllm-project/vllm/pull/53574) OPEN | Backport: `pr-53574.diff`; Overlay: `flashinfer-eidx-contig`. C4A branch verified contiguous — no C4A bug. |
| Triton E8M0 upcast gated on rocm/xpu | `KeyError: 'float8_e8m0fnu'` on SM12x | **still gated** 2026-08-24; [#47988](https://github.com/vllm-project/vllm/pull/47988) OPEN | Backport: `pr-47988.diff`; Overlay: `triton-e8m0-sm12x`. |
| SM12x DSv4 umbrella | partial (backend exists) | [#41834](https://github.com/vllm-project/vllm/pull/41834) needs-rebase | Comment only. Pointed at the focused PRs. |
| **DSv4 mHC TileLang warmup no-ops on the NVIDIA layer** | `deepseek_v4_mhc_warmup` gates on `layer.hc_pre`/`hc_post` which the nvidia layer lacks (it calls `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang` directly; AMD/XPU layers do have the CustomOps) | **still broken on main** | [#52941](https://github.com/vllm-project/vllm/pull/52941) OPEN (same fix + tests; older attempts #51802, #49707). **Evidence commented 2026-08-28** (c16 44.5 → 183.0, c32 306.8 agg; AMD/XPU keep-path note). Local equivalent: `patches/files/dsv4_warmup_ext.py`. Do not open a duplicate. |
| `fp8_einsum` SM12x Python dequant | release is DeepGEMM-or-missing | #52357 Triton path closed | Overlay: `patch_fp8_einsum_fallback` — **REMOVED in v0.28.0 stack** (fallback was the E2E-garbage source; stock kernel path verified correct). |


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

**2026-08-25 (corrected 2026-08-26): `8b1392b` is a REGRESSION for SM12x fp8 linear**
(see [docs/knowledge/09](knowledge/09-golden-deepgemm.md)). Verified diff
`a6b593d...8b1392b` (nv_dev):

- `fp8_gemm_nt = fp8_fp4_gemm_nt` is **not** new — the alias exists in
  `a6b593d` already (`gemm.hpp:792`). The regression is the **removal of the
  pure-fp8 1d1d kernels**: `csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp`
  (−416) and `deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh`
  (−567), plus an `fp8_fp4_mqa_logits` dispatch rewrite
  (`smxx_fp8_mqa_logits` → per-arch `sm90/sm100/sm120_mqa_logits`).
- On SM12x, pure fp8xfp8 inputs route to the combined `sm120_fp8_fp4_gemm_1d1d`
  kernel, which misreads fp8 weights as fp4 — silent corruption (France
  `' Septy Septy…'`, ~25.8 → 4.4 tok/s). Our local port
  (`deepgemm-fp8-1d1d-port.diff`) re-adds the 1d1d kernel + the fp8xfp8 branch.
  A/B of the a6 wheel vs 8b wheel is in progress to pin the exact surface.

Matched-main was pinned back to `a6b593d` (the eugr/golden freeze);
vLLM PR [#53680](https://github.com/vllm-project/vllm/pull/53680) moves the
cmake tag back to `a6b593d`; DeepGEMM issue
[#417](https://github.com/deepseek-ai/DeepGEMM/issues/417) tracks the upstream
restore — **landed 2026-08-26 as [DeepGEMM#419](https://github.com/deepseek-ai/DeepGEMM/pull/419)**
(restore + TU header fixes; driver-JIT PTX route tested and blocked, see
[docs/knowledge/09](knowledge/09-golden-deepgemm.md)). **2026-08-27/28
status: all ds-review-bot criticals/warnings addressed in `44d9d2e` (no
AB-swap for pure fp8, `allow_swap_ab` filter in SM100 heuristics, arch-10
fp4_A×fp8_B routing, epilogue_type, math.cuh include); lucifer1004's SMEM
capacity note fixed in `54d5a3e` (stages sized from the device's actual
`sharedMemPerBlockOptin`). Mergeable; no new findings.**

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
| DeepGEMM SM12x pure-FP8 route (`sm100_fp8_gemm_1d1d` kernel + fp8xfp8 branch) | present in `a6b593d`; **deleted in `8b1392b`** | staged local port | Tracked as `deepgemm-fp8-1d1d-port.diff` (applied in `docker/Dockerfile.main` before compilation): re-adds the 1d1d kernel and routes fp8xfp8 away from the combined `sm120_fp8_fp4_gemm_1d1d` kernel. |
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
| DeepGEMM pin | cmake `8b1392b` **nv_dev**, compiled with the image | drop `patch_deep_gemm_sm12x_guard` **only after** MQA/mHC actually run. eugr still frozen at `a6b593d` for MXFP4 grouped scales. **2026-08-26:** [#53680](https://github.com/vllm-project/vllm/pull/53680) re-pins cmake to `a6b593d` (8b removed the pure-fp8 1d1d kernel); port `deepgemm-fp8-1d1d-port.diff` covers 8b-era builds |
| KVBlockZeroer unaligned assert | rewritten on main | likely drop |
| FlashInfer TOPK 192 | **no.** docker pin `FLASHINFER_VERSION=0.6.17`. Tag v0.6.17 dispatch is still `{128,512,1024}`. 192/256 are on flashinfer **main** (#4380), not in 0.6.17 | **keep** overlay or bump FlashInfer to main / 0.6.18 nightly |
| DSV4 kernel block 64 | #53425 OPEN; import-cycle fix ed71de5 (2026-08-26) | **keep** |
| einsum SM12x recipe |  **keep** until merged; backport refresh pending| **keep** until merged; backport refresh pending |
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
| 2026-08-26 | DeepGEMM #419 | all review criticals addressed in `44d9d2e` | https://github.com/deepseek-ai/DeepGEMM/pull/419#issuecomment-… |
| 2026-08-26 | vllm #53522 | ivanusto review + test pass | https://github.com/vllm-project/vllm/pull/53522#issuecomment-… |
| 2026-08-26 | vllm #53425 | import-cycle fixed in `ed71de5` (kitch2400 trace) | https://github.com/vllm-project/vllm/pull/53425#issuecomment-… |
| 2026-08-26/27 | vllm #53680 | kitch2400 a6b593d validation; relationship note | https://github.com/vllm-project/vllm/pull/53680#issuecomment-… |
| 2026-08-27 | DeepGEMM #419 | lucifer1004 SMEM capacity; fixed `54d5a3e` | https://github.com/deepseek-ai/DeepGEMM/pull/419#issuecomment-… |
| 2026-08-28 | vllm #46716 | **rebased** onto current main (clean, +9/-1), bug still present upstream | https://github.com/vllm-project/vllm/pull/46716#issuecomment-5451316380 |
| 2026-08-28 | all 6 PRs | **recipe repo linked** (maci0/vllm-spark-0731) as reproducible source | see each PR |
| 2026-08-28 | vllm #52941 | DSv4 mHC warmup no-op: 2x GB10 evidence (c16 44.5→183, c32 306.8 agg) + AMD/XPU keep-path note; no duplicate opened | https://github.com/vllm-project/vllm/pull/52941#issuecomment-5453460872 |
| 2026-08-28 | vllm #53574 | lucifer1004 implemented the SM120 eidx fix (`full_width_decode` on SM120); we confirmed the C4A-branch finding earlier | https://github.com/vllm-project/vllm/pull/53574#issuecomment-… |
| 2026-08-28 | vllm #47988 | kitch2400 independent GB10 confirmation (required, still red mergify — author rebase needed) | https://github.com/vllm-project/vllm/pull/47988#issuecomment-… |
| 2026-08-28 | vllm #53055 | author force-pushed `e1d67cc` (DCO cleared, tests pass), awaiting review | https://github.com/vllm-project/vllm/pull/53055#issuecomment-… |
| 2026-08-29 | v0.28.1rc0 | verified: none of our PRs merged; relevant new commits listed in the header | https://github.com/vllm-project/vllm/releases/tag/v0.28.1rc0 |

## Patch necessity verdict (2026-08-28 audit, `patches/apply_overlays.py`)

54 overlay functions; **38 applied by `apply_main`** (the v0.28.0 stack) —
each maps to an open upstream PR backport, a local SM12x fallback, or a
Spark-specific workaround (tables above). The remaining ~16 are **defined
but not applied** (history/experiments, kept for `--only` runs):
`patch_fp8_einsum_fallback` / `patch_einsum_sm12x_recipe` /
`patch_einsum_sm12x_scale_upcast` (superseded — #53898/#53521 CLOSED),
`patch_logit_dump` (diagnostic), `patch_lm_head_restore_after_graphs`,
`patch_dspark_hidden_fix`, `patch_dspark_disable_graphs`,
`patch_dspark_backbone_none`, `patch_dspark_fullstep_graph`,
`patch_dspark_fullstep_revert` (DSpark graph-mode experiments),
`patch_indexer_packed_insert_revert` (history), `patch_tp_allreduce_piecewise_workspace`
(alternative mode). `patch_mhc`'s broadcast guard is not exercised by the
0731 nvidia model (embed pre-broadcasts to 3D) but kept for #53055 + other
DSV4 variants. On v0.28.0 the #52018 MoE-b12x overlays auto-skip (already
in the release).

