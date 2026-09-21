# Upstream tracker

> **This file is a chronological log, not current state.** It is append-only and grows over 2854
> lines: entries near the bottom are the most recent, and earlier entries contain claims that later
> work corrected, including some written as settled that are not. Read it for provenance and for why
> a decision was made. For the current summary read
> [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) (what was measured, and whether each change was kept) and
> [`HANDOVER.md`](../HANDOVER.md) (where the project stands and what is next). The
> [docs index](README.md) maps the rest.
>
> Two entries below also name `.pre-reanchor` and `.pre-proto` snapshot files. Those snapshots were
> deleted on 2026-09-13 as redundant, which is expected of a log that is not edited after the fact.

Last verified: **2026-09-11**. Recipe repo (public, reproducible):
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
patch). **08-30 check**: all 12 PRs still OPEN (no merges); #47988 author rebased -> MERGEABLE;
#53055 active review (liulanze, author fixed the import); #403 maintainers shepherding
(bvolpato->zheanxu). New main merges since 08-28: **#54048** (cuBLAS out_dtype router
GEMM on family-120, fixes GB10 bf16-rounded router logits, merged 08-30) — **backported
as `patch_router_gemm_cublas_sm12x` (--only router-gemm-cublas, also in apply_main)**,
commit a771e1a; #54277 (FlashInfer MLA for DSpark drafting, DCP-only — N/A).
PR triage 2026-08-29: #53574 lucifer1004 implemented the SM120
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



## 2026-09-11 session findings (new)

1. **The 2.7x decode gap was a full-KV-cache copy inside our own indexer
   sidecar insert. FIXED, opt-in, and not an upstream item.**
   `sync_packed_indexer_k` in `patches/files/sm12x_b12x_kernels.py` (donor for
   `vllm/utils/sm12x_b12x_kernels.py`) picked the newly inserted tokens with

       raw = kv_cache[..., :_TOKEN_BYTES]
       flat_tokens = raw.reshape(-1, _TOKEN_BYTES)
       tok = flat_tokens[tok_idx]

   The slice is non-contiguous, so `reshape` materialised the whole KV cache on
   every layer of every decode step, only to gather the 1 to 48 rows just
   inserted. Attribution was exact: the torch profiler links a kernel to its
   launching op through `args["External id"]`, and matching on it gives **336
   `aten::copy_` calls, 3,320,389,632 elements, 57.5 ms** over a 6-step trace,
   with the host frames naming `sync_packed_indexer_k` at
   `sm12x_b12x_kernels.py:234` called from `compressor.py:309`. Those copies are
   eager (`graph id` 0) on side stream 29 via `multi_stream_utils.execute_in_parallel`.
   No upstream PR is possible: main vLLM has no
   `vllm/utils/sm12x_b12x_kernels.py` (verified 404 on raw main) and
   `copy_sm12x_b12x_kernels()` installs it from our donors.
   Measured, no-spec c1, same recipe otherwise: **10.1/10.2 to 25.4/25.6 tok/s**
   (anemll 26.9/27.3), coherence intact (France gives ` Paris.`, `9x8` gives
   `72`).
   **The k=7 win is unproven.** A same-boot A/B/A on the D1 recipe, with the
   toggle verified visible inside both containers, gave c1 43.0/32.7 (off),
   34.6/39.1 (on), 41.9/36.2 (off): no gain. Acceptance was 51.3% (off), 37.5%
   (on), 45.1% (off), and the two OFF phases alone span 41.1-51.3%, so the flag
   does not demonstrably change acceptance either. An earlier A/B in this
   session claimed the patch cost acceptance (60% down to 45%) and marked it
   "do not ship"; that was boot-to-boot variance and this corrects it.
   Acceptance on this recipe is unstable across 37-64% with or without the
   flag, against the anemll bar 62.4%, which itself was never measured on this
   harness.
   It therefore ships **off** as `VLLM_B12X_INDEXER_DIRECT_GATHER` (default
   `0`), with the original flatten+gather as the default, and is not
   upstreamable either way. The toggle also honours a marker file under
   `VLLM_SKIP_FLAG_DIR`, so a running server can be switched without rebooting;
   toggle **both** nodes or the pair desynchronises at every all-reduce and the
   measurement silently cancels.
   The check was run: the copies **are** present on k=7 (672 large ones, 168
   attributed `aten::copy_`, 2.0e9 elements, 137 ms of the 463 ms GPU-busy time),
   so the fix has work to remove there. It still does not pay, because the k=7 arm
   runs only 66.1% GPU-busy against 96.1% on no-spec, so the removed GPU work is
   absorbed by 238 ms of GPU idle.
8. **The k=7 arm is host-bound on our own MoE launch path.** Of the >=1 ms GPU
   stalls, 46 of 48 sit in `run_w4a16_moe`
   (`b12x/moe/_shared/kernels/w4a16/kernel.py:11578`), 56.8 ms of 59.0 ms of
   stalled time, against only 5 gaps / 6.4 ms on the no-spec arm for the same
   host path. Per layer call the host spends ~2.1-2.6 ms: 252 us in
   `run_w4a16_moe` itself, 192 us in four per-call constructors, 133 us scanning
   16 tile candidates (`_determine_blocks_per_sm` + `_candidate_tile_fits`), 90 us
   in 32 `_shared_memory_footprint` calls, 73 us in two `compile_w4a16_fused_moe`
   lookups, 40 us in 17 `os.__getitem__` env lookups. 43 layers x ~2.2 ms is
   ~95 ms of host per forward against ~101 ms of GPU per forward, which is why
   k=7 starves and no-spec does not. Measured against anemll on the same request:
   our `run_w4a16_moe` is **1690 us per layer inclusive against their 581 us, 2.9x**,
   with a second `compile_w4a16_fused_moe` per layer (371 us against their 142 us),
   a 16-candidate tile scan against their 8, 261 us of per-call constructors
   against 80 us, and an extra 424 us `bind` layer. Per forward that is
   ~48 ms of excess host, and since the k=7 window holds ~3 forwards it accounts
   for ~150 ms of the 224 ms idle difference. Their arm has zero GPU gaps >= 1 ms;
   ours has 48. The obvious target is memoising everything shape-invariant in that
   path (tile config, shared-memory footprint, env lookups, the duplicate compile)
   so host per forward drops below GPU per forward; the indexer copy fix should
   then pay on k=7. This code is in the b12x package, not `patches/files` under
   vllm, so it needs a donor that overwrites the installed b12x module, the
   pattern `docker/Dockerfile.ov-rdma` already uses for vllm paths. This also
9. **MoE host memo patch: implemented, verified, no throughput conversion.** b12x
   `compile_w4a16_fused_moe` is memoised on its raw keyword arguments (original
   body renamed `_compile_w4a16_fused_moe_impl`, new `_FUSED_RAW_CACHE` cleared by
   `clear_w4a16_kernel_cache`), because `_FUSED_CACHE` cannot be consulted until
   the tile selection has already re-scanned every candidate. Patch script
   `.scratch/patch_b12x_memo.py`; image `vllm-spark-0731:main-029-fast2`. Verified:
   k=7 GPU gaps >= 1 ms fall from 48 gaps / 59.0 ms to 5 / 15.3 ms and
   `run_w4a16_moe` leaves the stall list. But the same-boot A/B/A shows no c1
   change from either this or the gather (off 46.0/45.3, on 46.0/43.5, off
   40.6/43.3), and acceptance drifted 44.3% -> 51.4% -> 51.9% with the toggle
   alternating, confirming again that the flag does not affect acceptance. The k=7
   deficit is therefore not explained by the two fixes so far. Next leads: our k=7
   window has 8357 GPU spans against anemll's 4381 for a similar number of
   forwards (1.8x the launches, 840 of them copies), and acceptance remains 44-52%
   against the bar 62.4%. This also corrects HANDOFF section 19, where the
   per-step MoE cost was read out of the code and judged not to matter.
   Note `stop_profile` kills the engine on the DSpark arm (also before this
   patch), so k=7 allows only one profile per boot.
10. **k=7 is launch-granularity-bound, and the linear path is the sharpest
   structural difference.** Per-stream counts over the same window: our main
   stream runs 4488 kernels for 140.6 ms where anemll's runs 1886 for 208.1 ms,
   and the union of busy spans is 8357 in 463 ms against 4381 in 410 ms, so we
   issue ~1.9x the spans at ~1.7x shorter each. 238 ms of idle spread over ~14.7k
   launches is ~16 us of gap per launch: the GPU finishes each of our kernels
   before the host has the next ready, while anemll's 94 us spans never starve,
   which is why they show zero gaps >= 1 ms. Reducing GPU time without reducing
   launch count therefore does not convert, which is what the copy fix showed.
   Sharpest count difference, the linear path: we run 342
   `per_token_group_quant_8bit_kernel` on the main stream (8 per layer) plus 546
   GEMM launches, where anemll fuses quantisation into ~216
   `deep_gemm sm120_fp8_fp4_gemm_1d1d_impl` calls. Our `--linear-backend b12x` is
   required for correctness, and our image deliberately disables DeepGEMM on
   family 120 (`patch_deep_gemm_sm12x_guard`, v0.27.1 main `.so`), so matching
   anemll means making the SM12x DeepGEMM path correct again
   (`deepgemm-fp8-1d1d-port.diff` plus an nv_dev build), a larger workstream than
   the fixes landed so far. Cheaper first: the 657 copy kernels, 307
   `Memcpy DtoD` and 628 small `vectorized_elementwise_kernel` on the main stream,
   and the apparent duplication (55 MoE launches for 43 layers, 159 all-reduce for
   ~86).
2. **NCCL library exonerated, the in-model all-reduce is still 7x.** New
   two-node all-reduce microbenchmark (`.scratch/nccl_bench.sh` +
   `nccl_ar_bench.py`, 64 KB bf16, sparkrun NCCL env, `--privileged` so
   `ibv_open_device` succeeds; without the verbs devices NCCL reports "Failed to
   initialize any NET plugin"): ours `libnccl 2.31.2` p50 **42.0 us**, anemll
   `2.30.7` p50 **41.5 us**, so no library regression at this size. In the model
   ours still averages 390 us/call with 62 of 528 above 2 ms, against anemll
   55 us/call and none above 1 ms. Our kernel is
   `ncclDevKernel_AllReduce_bf16_RING`, theirs `..._RING_LL`. **Open lead.**
3. **Correction: `VLLM_PROFILE_SKIP_INDEXER` does not exist.** It appears
   nowhere under `patches/` (grep count 0), so the "INDEXER RULED OUT" entry in
   HANDOFF section 15 rested on a run that changed nothing. The indexer and
   compressor path was in fact the whole residual.
4. **Recipe-level requirements reconfirmed, not overlay changes.**
   `--linear-backend b12x` is required for correctness (auto picks DeepGemm FP8,
   numerically wrong, garbage text) and `VLLM_B12X_MOE_FP4_FORCE_A16=1` is
   required for acceptance. **A16 is not a perf lever:** the default
   `B12X_MXFP4_MXFP8` recipe (w4a8_mx) measured 9.9/10.0 tok/s against 10.1/10.2
   for A16 (w4a16), so the flag costs ~0.2 tok/s and stays on.
   `VLLM_B12X_MOE_FP4_FORCE_A16` maps in `patches/files/fused_moe_b12x.py`
   `_B12X_MOE_MODES`, where `("mxfp4", None)` is the w4a16 recipe.
5. **b12x lineage check.** The anemll image's b12x 0.15.3 is pure
   Python/CuTe-DSL with no `.so` and copies straight out of the image. It is the
   same lineage as our 1.2.6 refactor (ours is the newer one and carries more
   kernels), with identical `decode_max_active_clusters` ladders. The
   "different kernel family / not obtainable" conclusion in HANDOFF sections
   22-25 is wrong.
6. **Canonical image still owes the rebuild.** The measured artifact is the
   derived `vllm-spark-0731:main-029-idxfix`, built from the image's own
   `sm12x_b12x_kernels.py` plus the gather change only. The tracked donor has
   that change gated and is otherwise ahead of the image (the `b12x_profile_*`
   helpers `apply_overlays.py` imports), so an overlay rebuild picks up both.
   Re-measure after the rebuild.
7. **Test status after this change** (`tests/test_sm12x_b12x_kernels.py`,
   run in-image because the host has no torch):
   `test_sync_matches_full_pack_at_written_slots`, the sidecar equivalence test
   for this exact function, passes with `VLLM_B12X_INDEXER_DIRECT_GATHER` set to
   both `0` and `1`, so the gather mapping is equivalent. Two failures are
   pre-existing and untouched by this change: `test_only_single_row_uses_schedule`
   fails identically with the original pre-fix donor, and
   `test_assert_stack.test_refuse_wrong_k` fails on the host
   (`test_assert_stack.py` has no reference to this module).

## Not in v0.28.0, still open on vLLM main

Same bug in the release **and** on main today. Comment or small PR. Do not duplicate.

| Piece | v0.28.0 | main 2026-08-28 | Action |
|-------|-----|-----------------|--------|
| `mhc_pre_broadcast_tilelang` unguarded `tf32_hc_prenorm_gemm` | unguarded | still unguarded | [#53055](https://github.com/vllm-project/vllm/pull/53055) (also CUTLASS + sm121 carve-out). Older [#50645](https://github.com/vllm-project/vllm/pull/50645) needs-rebase. Backport: `pr-53055.diff`; Overlay: `patch_mhc`. Comment only; do not duplicate. Not exercised by the 0731 nvidia model (3D pre-broadcast), kept defensively. |
| CUTLASS FP8 `is_supported()` ignores SM12x | `CutlassFp8BlockScaledMMKernel` returns True if `CUTLASS_BLOCK_FP8_SUPPORTED` | #53055 still open | Backport: `pr-53055.diff`; Overlay: `patch_cutlass_sm12x_guard`. Same PR as mHC. |
| `compute_fp8_einsum_recipe`: `major >= 10` → SM100 packed INT32 | yes | stock config is correct | [#53521](https://github.com/vllm-project/vllm/pull/53521) **CLOSED 2026-08-27** (not needed). Stock SM12x `(1,1,128)` + `tma_aligned_scales=True` verified numerically correct on GB10 (einsum mean_rel 0.000000 vs bf16 ref; E2E France coherent). |
| `fp8_einsum` on SM12x | yes | **stock kernel path is CORRECT** with packed E8M0 scales + `(1,1,128)` (mean_rel 0.000000 on GB10) | [#53898](https://github.com/vllm-project/vllm/pull/53898) **CLOSED 2026-08-27** (fallback not needed). Real fix upstream: deepseek-ai/DeepGEMM **#337**. |
| `compute_fp8_einsum_recipe` on SM12x when `VLLM_USE_DEEP_GEMM_E8M0=1` | flag off by default | not filed | Not a defect in the default path, so the two entries above stand. With E8M0 forced on, the loader emits the int32-packed scale (`gran_mn = 1`) while `compute_fp8_einsum_recipe` still returns `(1, 128, 128)` for `cap.major == 12`, and the einsum asserts at `layout.hpp:97`. Returning `(1, 1, 128), False` from the helper when `is_deep_gemm_e8m0_used()` boots and serves (verified through an injected `sitecustomize` patch, never landed). Recorded rather than filed because E8M0 measures slower at every concurrency level on this pair, so the flag is not what ships; this is here so that switching it on for another reason does not start a fresh layout investigation. |
| DSV4 kernel block `[256]` on SM12x | `[256]` on sparse MLA, FlashInfer DSV4, V4 indexer | **still `[256]`** | [#53425](https://github.com/vllm-project/vllm/pull/53425) OPEN — fixed 2026-08-26 (ed71de5): `indexer → vllm.models.deepseek_v4.sparse_mla` module-level import broke `vllm._aiter_ops` cold start (kitch2400 report); lazy import inside `get_supported_kernel_block_sizes()`. Backport: `pr-53425.diff`; Overlay: `patch_dsv4_sm12x_block_size`. |
| Indexer paged MQA metadata uses `has_deep_gemm()` not `is_deep_gemm_supported()` | yes | still that pattern | [#53522](https://github.com/vllm-project/vllm/pull/53522) OPEN (`is_deep_gemm_supported()` + `num_states in (32, 64)`). **ivanusto reviewed 2026-08-24: test passed, gate scoped correctly**. Backport: `pr-53522.diff`; Overlay: `patch_indexer_deepgemm_guard`. |
| DSpark SM120 spec-decode query rank / `num_tokens > 64` | #51538 in release (backend + top-k). Flat 3-D spec query may remain. | [#52499](https://github.com/vllm-project/vllm/pull/52499) open | Comment only. Not needed after TOPK=192. |
| FlashInfer eidx contiguity (C128A builder) | `_build_c128a_metadata` view of a width-sliced `global_decode_buffer`; DSpark batches >64 tokens crash at boot | **MERGED 2026-08-31** (`699e180df4`, ancestor of the pin); [#53574](https://github.com/vllm-project/vllm/pull/53574) | Fixed in the base: `build_c128a_topk_metadata` returns the full-width buffer slice on family 120, so the decode view is contiguous. `pr-53574.diff` and the `flashinfer-eidx-contig` overlay **retired 2026-09-14** (overlay alone was a no-op; C4A is `empty_like`-contiguous). |
| Triton E8M0 upcast gated on rocm/xpu | `KeyError: 'float8_e8m0fnu'` on SM12x | **still gated in the pin**; [#47988](https://github.com/vllm-project/vllm/pull/47988) OPEN, and the same upcast merged separately in [#56214](https://github.com/vllm-project/vllm/pull/56214) (`e77daef89e`, five commits after the pin) | Backport: `pr-47988.diff`, **refreshed 2026-09-14** against head `e1dbe81c`; it keeps the Triton hunk (the pin predates `e77daef89e`) and deliberately omits the head's `_upcast_e8m0_to_fp32` rewrite. Overlay: `triton-e8m0-sm12x`. |
| SM12x DSv4 umbrella | partial (backend exists) | [#41834](https://github.com/vllm-project/vllm/pull/41834) needs-rebase | Comment only. Pointed at the focused PRs. |
| **DSv4 mHC TileLang warmup no-ops on the NVIDIA layer** | `deepseek_v4_mhc_warmup` gates on `layer.hc_pre`/`hc_post` which the nvidia layer lacks (it calls `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang` directly; AMD/XPU layers do have the CustomOps) | **still broken on main** | [#52941](https://github.com/vllm-project/vllm/pull/52941) **CLOSED 2026-09-13**, author says superseded by [#50178](https://github.com/vllm-project/vllm/pull/50178) (MHC TileLang warmup moved to a `VllmJitKernel`) (older attempts #51802, #49707). **Evidence commented 2026-08-28** (c16 44.5 → 183.0, c32 306.8 agg; AMD/XPU keep-path note). Local equivalent: `patches/files/dsv4_warmup_ext.py`. Whether #50178 covers the nvidia dispatch is not checked. |
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
| `VLLM_B12X_INDEXER_DIRECT_GATHER` direct sidecar gather in `sync_packed_indexer_k` | Ours, not upstream (main has no `sm12x_b12x_kernels.py`). The reshape it removes is the single `reshape(-1, _TOKEN_BYTES)` in the image's tree, and the strided page view makes it clone the whole indexer page view: nsys measures it at 9.471 s of a 62.9 s device budget, 15.05 percent, over 54390 launches at 174 us, and it is absent from the capture taken with the gather enabled. 2.5x on no-spec c1. **No k=7 gain**: a within-boot A/B (off / on / off, three passes then five passes, marker file `~/.cache/huggingface/flags/indexer-fast`) puts c1 at 40.6 against 39.0 and 39.7 and c6 at 124.3 against 123.6 and 125.0, so the arms are indistinguishable and the copy is overlapped rather than on the serial path. Acceptance unchanged. Stays default `0`. Do not upstream. An earlier cross-boot comparison of this flag was invalid: the two boots differed by 3.8x in NCCL all-reduce per call (127 us against 479 us, 10.5 s of a 62.9 s budget), which is larger than the effect and opposite in sign. |

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
| DeepGEMM SM12x pure-FP8 route (`sm100_fp8_gemm_1d1d` kernel + fp8xfp8 branch) | present in `a6b593d`; **deleted in `8b1392b`** | in the pin | The staged port (`deepgemm-fp8-1d1d-port.diff`) was skipped on every `a6b593d` pin and was **deleted 2026-09-15**, together with its `docker/Dockerfile.main` step; the pin-back is what keeps the kernel present, so nothing needs staging. |
| CUDA 13 `CUDA_SUPPORTED_ARCHS` drops 12.1 | **dropped** 2026-09-15 (`patch_vllm_preserve_sm12x_target.py` and `VLLM_PRESERVE_SM12X_TARGET` removed) | rc2 CUDA 13 DeepGEMM list uses family `12.0f`, not `12.1a` | [#52708](https://github.com/vllm-project/vllm/pull/52708) **CLOSED 2026-09-14 unmerged** (upstream: 12.0/12.0f are compatible with sm_121, no separate SM121 build motivated), older [#38484](https://github.com/vllm-project/vllm/pull/38484). `v0.29.1rc0` still omits 12.1 in its `CUDA >= 13.0` branch, but the 2026-09-15 measurement showed the local patch bought no codegen, so vLLM now builds for `12.0f` and the recipe stays on the supported path. See the 2026-09-15 entry. |

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


## 2026-09-11 evening: profiling method, and two engine-level candidates (not yet filed)

Nothing in this section is PR-ready. It records what the throughput gap turned out to be, so
the candidates below can be checked against the tracker before anything is written up. No issue
or PR text was produced for any of them.

### Measured gap against the reference image, on this rig

Both arms on the same two nodes, same checkpoint, golden chat harness
(`bench-concurrency.py --chat --max-tokens 128 --levels 1 3 5 6`, temperature 0.7):

| arm | c1 | c3 | c5 | c6 | accepted/drafted | mean acceptance length |
|---|---|---|---|---|---|---|
| `vllm-spark-0731:main-029` | 23 to 28 | 47 to 56 | 68 to 73 | 75 to 88 | 30.7% | 3.15 |
| `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | 52 to 63 | 96 to 104 | 135 to 141 | 143 to 157 | 45.7% | 4.20 |

1.8x, split roughly evenly: 1.33x from tokens per step (acceptance) and 1.36x from step latency
at c6. The reference image is vLLM 0.25.2.dev0 with b12x 0.15.3 and **no DeepGEMM**, so it is a
different lineage rather than a patched 0.29 and there is no 0.29 delta to port.

### 1. Profiling note: nsys hides CUDA-graph kernels by default

nsys defaults to `--cuda-graph-trace=graph`, which reports a replayed graph as one activity row
and drops the kernels inside it. Every DSV4 decode step's 43-layer target forward is one such
graph, so a default-level trace shows a 79 ms step as an 11.5 ms eager preamble plus 63 ms of
apparent idle. `--cuda-graph-trace=node` emits the kernels with `graphNodeId` set. Worth
knowing for anyone profiling vLLM; not an upstream change.

### 2. Candidate: DeepGEMM `fp8_einsum` scale-layout assertion on sm121 with UE8M0

With `VLLM_USE_DEEP_GEMM_E8M0=1` and `--linear-backend b12x`, startup fails in the dummy
`profile_run` forward:

```
attention.py:417 _o_proj
flashinfer_sparse.py:548 deep_gemm_fp8_o_proj
ops/o_proj.py:107 fp8_einsum
utils/deep_gemm.py:471 _fp8_einsum_impl
RuntimeError: Assertion error (.../DeepGEMM/csrc/utils/layout.hpp:97):
              sf.size(-2) == ceil_div(mn, gran_mn)
```

`o_proj.py` prepares the weight with `deepgemm_post_process_fp8_weight_block(..., use_e8m0=True,
is_bmm=True, bmm_batch_size=n_groups)` and then calls
`fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale), (weight, weight_scale), z, recipe=einsum_recipe)`.
One of those two scale tensors does not have the shape the chosen DeepGEMM heuristic expects.
Our `/opt/DeepGEMM` is at `a6b593d`; `patches/upstream/deepgemm-fp8-1d1d-port.diff` was written
for a different revision (it fails to apply: `heuristics/config.hpp` conflicts and
`impls/sm100_fp8_gemm_1d1d.hpp` already exists here), so the local patch is not the fix as
written.

Our image works around this by setting `VLLM_USE_DEEP_GEMM_E8M0=0` in the image ENV. That makes
`DeepGemmQuantScaleFMT.from_oracle()` return `FLOAT32` while the checkpoint declares
`scale_fmt=ue8m0`, and the engine logs `Model config requests UE8M0 ... but
VLLM_USE_DEEP_GEMM_E8M0=0 is set`. A `_ENV` workaround like that is not upstreamable; the
upstream-relevant question is whether the o_proj scale preparation and the DeepGEMM heuristic
disagree about the scale layout.

### 3. Candidate: `--linear-backend auto` produces garbage for this model when DeepGEMM is off

`--linear-backend b12x` is required for the model to work at all here. Removing it (keeping
`--attention-backend B12X_MLA_SPARSE`) serves but returns `?carecarecarecarecare` for both
gates, with 0% draft acceptance and 1.008 tokens per step. The auto priority order puts
FlashInfer FP8 and Cutlass FP8 ahead of `B12xTensorFP8ScaledMMLinearKernel`, and with
`VLLM_USE_DEEP_GEMM_E8M0=0` those kernels are handed float32 scales where the checkpoint wants
packed UE8M0. Whether that is a bug or the documented consequence of disabling UE8M0 needs
checking before anything is filed; the practical result is that our engine can only run this
model on b12x, which is a large part of the step-latency gap.

### 4. Local overlay bug, fixed here, not an upstream item

`patches/files/dsv4_warmup_ext.py` called `gumbel_sample` with nine positional arguments against
a signature that had gained `is_drafting` as its seventh parameter, so every later argument
shifted by one and the call raised `AttributeError: 'bool' object has no attribute
'contiguous'`. The warmup swallows exceptions and only logs
`DSpark gumbel warmup failed (serving without it)`, so the visible symptom was the DSpark gumbel
triton kernels JIT-compiling during inference. Fixed by binding by keyword and by sweeping
`is_drafting` as well; verified for all 16 combinations on GB10. Needs an image rebuild, which
is owed regardless.

Artifacts: `HANDOFF.md` sections 64 to 70, recipes `d1-plain.yaml`, `d1-auto.yaml`,
`d1-linearauto.yaml`, `d1-fp32dl.yaml`, `d1-e8m0.yaml`, `anemll-plain.yaml`, results under
`outputs/driver/`.

## 2026-09-11 late: upstream tracker refresh, and why nothing from this session is a PR

### Upstream state read from GitHub (2026-09-11)

| PR | state | what it is | consequence for us |
|---|---|---|---|
| [#53574](https://github.com/vllm-project/vllm/pull/53574) | **MERGED 2026-08-31** | SM120 DSv4 contiguous C128A decode topk | our local `full_width_decode` SM120 overlay may now be redundant on a later pin. Check before the next overlay audit |
| [#47988](https://github.com/vllm-project/vllm/pull/47988) | OPEN | Handle E8M0 block scales in CUTLASS and Triton FP8 paths | likely the tracker entry for our `--linear-backend auto` garbage output. No new issue needed, and a fix may already be coming |
| [#53055](https://github.com/vllm-project/vllm/pull/53055) | OPEN | fallback `mhc_pre_broadcast` to TileLang | unchanged |
| [#53425](https://github.com/vllm-project/vllm/pull/53425) | OPEN | SM12x FlashInfer sparse MLA block size | unchanged |
| [#53522](https://github.com/vllm-project/vllm/pull/53522) | OPEN | gate indexer paged MQA metadata on DeepGEMM | unchanged |
| [#52018](https://github.com/vllm-project/vllm/pull/52018) | MERGED 2026-08-21 | b12x FP4 MoE backend | shipped, our overlays auto-skip |
| [#53898](https://github.com/vllm-project/vllm/pull/53898) | **CLOSED** | `fp8_einsum` dequant fallback + packed scales | not merged, so our local `patch_fp8_einsum_fallback` remains the only route. Keep |
| [#53521](https://github.com/vllm-project/vllm/pull/53521) | **CLOSED** | Hopper `fp8_einsum` recipe on SM12x | not merged, so our local `patch_einsum_sm12x_recipe` remains the only route. Keep |
| [#52941](https://github.com/vllm-project/vllm/pull/52941) | OPEN | DSv4 mHC TileLang warmup for the nvidia layer | our `dsv4_warmup_ext.py` is the local equivalent. Evidence already commented, do not open a duplicate |

Reading #47988 against our symptom is worth recording: with the E8M0 weight-scale handling in the
CUTLASS and Triton FP8 paths incomplete, those kernels are handed block scales they misread, which
is what the garbage output from `--linear-backend auto` looks like. That is a hypothesis from the
title, not verified against the patch contents. It is not a new upstream item either way.

### No PR-ready patch came out of this session

The only bug fixed in this session is in `patches/files/dsv4_warmup_ext.py`, which is **our own
overlay file**, not an upstream file. The vLLM 0.25.2 tree in the reference image has no
`model_executor/warmup/dsv4_warmup_ext.py` at all, and the file's own docstring describes it as a
local extension driven from our `kernel_warmup` overlay. There is nothing upstream to send for it.

The two engine-level items raised this session are candidates, not fixes, and both now map onto
existing tracker entries rather than needing new ones: the auto-FP8 garbage maps onto #47988, and
the DeepGEMM E8M0 assertion is the same area as the two CLOSED fp8_einsum PRs, where our local
overlays are the standing answer. Nothing was filed, and no issue or PR text was written.

### Landed in this repo

- `patches/files/dsv4_warmup_ext.py`: the DSpark gumbel warmup now binds `gumbel_sample` by
  keyword and sweeps `is_drafting`. Review diff at
  `.scratch/warmupfix/warmup-gumbel-signature.patch` against the copy the current image ships.
  Provenance correction: the fix was first verified from a scratch copy and only installed into
  `patches/files/` afterwards, so the "fixed" claim in the section above was written slightly
  ahead of the install. Both are true as of now.
- `tests/test_dspark_warmup_call.py` (new): parses the warmup and asserts the call binds by
  keyword, then resolves those keywords against the real `gumbel_sample` signature when a vLLM
  tree is reachable (`VLLM_SRC`, or `/opt/vllm`). Validated three ways: passes on the fixed
  overlay, fails on the file the current image ships, fails on a synthetic renamed keyword. This
  is the check that would have caught the drift when the overlay was written.
- `tests/test_assert_stack.py`: `test_refuse_wrong_k` asserted that k=7 is refused, but
  `patches/assert_stack.py` enforces a *minimum* (`DSPARK_MIN_K = 5`) and the D1 recipes serve
  k=7, so the test contradicted the code it tests and the suite was red before this session
  touched anything. Renamed to `test_refuse_k_below_min`, now checks k=4 is refused and k=7 is
  accepted. Full suite after the change: `Ran 36 tests ... OK (skipped=12)`.

None of this is upstream material. All of it is normal repo maintenance for
`maci0/vllm-spark-0731`, and the commits are the account owner's to make.

## 2026-09-13 PR re-check

Every PR re-read with `gh`. Nothing was pushed, committed or commented in this pass.

### Ours, vllm-project/vllm

| PR | subject | state | needs |
|---|---|---|---|
| #53425 | SM12x sparse MLA kernel block size 64 | OPEN, was CONFLICTING/DIRTY | rebase refreshed onto current main (see below); needs the push, then CI approval |
| #53680 | pin DeepGEMM to a6b593d | OPEN, MERGEABLE/BLOCKED | CI not approved to run (3 review requests) |
| #53522 | gate indexer paged MQA on DeepGEMM | OPEN, MERGEABLE/BLOCKED | same (1 review request) |
| #53271 | KV offload: validate device pointers | OPEN, MERGEABLE/BLOCKED | same (2 review requests) |
| #46716 | CPU shm all-reduce deadlock | OPEN, MERGEABLE/BLOCKED | same, and it has no review request assigned |
| #53898, #53521 | fp8_einsum / recipe | CLOSED | nothing |

All four BLOCKED rows show `pre-run-check=FAILURE` with every other check `SKIPPED`, which is
"CI has not been approved to run", not a failing test.

### #53425 rebase, refreshed 2026-09-13

`~/rebase-53425` on spark1, branch `rebase/53425`.

    base    6b153463a8  (current main; the worktree's remote-tracking ref was stale at
                         a0844fa6c6, which is why the first rebase attempt reported
                         "up to date" - a forced fetch of the main ref was needed first)
    tip     bb5b8cf67f  = 1f119a9f2c + bb5b8cf67f
    files   4 changed, +58/-6, py_compile clean on all four
    authors unchanged; committer set to Marcel W. Wysocki <maci.stgn@gmail.com> to match
    the PR's existing commits (an earlier attempt stamped maci@spark1.lan, a hostname
    that must not enter public history)
    applied cleanly, no conflicts, and the lazy-import resolution from the first attempt
    is intact

Push, once approved (the remote head is still `9637a0ca0b`):

    git push https://github.com/maci0/vllm.git \
      rebase/53425:sm12x-dsv4-kernel-block-64 \
      --force-with-lease=sm12x-dsv4-kernel-block-64:9637a0ca0b

The PR's own test `tests/v1/attention/test_dsv4_kernel_block_size.py` still cannot be run in the
serving image: collection fails with `ImportError: cannot import name 'kernel_launcher' from
'vllm.model_executor.warmup.jit_warmup'`, and `grep -rn kernel_launcher` over the image's tree
returns nothing, so the failing chain is an overlay/version mismatch in our image rather than
anything the PR touches. `tblib` is missing too. It needs a dev checkout or CI.

### Ours, ggml-org/llama.cpp

| PR | subject | state | needs |
|---|---|---|---|
| #28003 | RDNA3 single-token MMVQ dispatch | OPEN, **draft**, +11/-0, 1 comment | the template filled and the draft cleared, which are the owner's to do |
| #28002 | cache quantized src1 across projections | CLOSED | nothing |

The local guard is uncommitted: `.scratch/wt-28003`, branch `fix/28003-rdna3-guard` at `41686b31`,
one line adding `&& nsamples_dst == 1` to the RDNA3 fast path in `ggml/src/ggml-cuda/mmvq.cu`.
Not built or run, since the target is RDNA3 and this box is GB10.

### Tracked upstream, not ours

| PR | subject | state | note |
|---|---|---|---|
| #53574 | DSv4 SM120 C128A contiguous topk indices | **MERGED 2026-08-31** | our pin postdates it; retire the overlay and diff |
| #47988 | E8M0 block scales in CUTLASS and Triton | OPEN, MERGEABLE | moved 2026-09-12 to `e1dbe81c`; our backport is stale |
| #53055 | mhc_pre_broadcast fallback to TileLang | OPEN, MERGEABLE | updated 2026-09-11; the live version of this fix |
| #50645 | guard mhc_pre_broadcast_tilelang | OPEN, CONFLICTING | superseded by #53055 |
| #52499 | DSv4 spec-decode shapes, SM120 FlashInfer | OPEN, MERGEABLE | unclaimed since 2026-09-06 |
| #52018, #51538 | b12x FP4 MoE backend; DSv4 sparse MLA end to end | MERGED | both in our image |

## 2026-09-13 base and dependency audit

Requested: rebase the image onto `vllm-project/vllm` tag `proto-v0.1.0`, drop patches no longer
needed, and update dependencies whose fixes have landed upstream. Findings, all read-only checks.

### The requested tag is not a release, and the image is already on v0.29.0

- `https://github.com/vllm-project/vllm/releases/tag/proto-v0.1.0` returns **404**: there is no such
  release.
- It exists as an annotated **tag** on commit `69db1c26b4` (2026-09-11), whose subject is
  `[CI/Build][Rust Frontend] Publish vllm-proto on crates.io (#56365)`. It marks the `vllm-proto`
  crate publication, not a serving release.
- The actual latest release is **v0.29.0**, published 2026-09-09, and its commit is
  **`98dff2a81`** — which is exactly our current pin. **The image already sits on v0.29.0.**

### No patch is unlocked by moving to that tag, or to main

`compare/98dff2a81...69db1c26b4` contains **521 commits**, and none of the PRs we backport or track
is among them (`#53425`, `#53680`, `#53522`, `#53271`, `#46716`, `#53055`, `#50645`, `#52499`,
`#47988`, `#54631` are all still OPEN; `#53574`, `#52018`, `#51538` merged before our pin). So the
rebase moves the base and drops nothing.

### What is droppable now, without any rebase

| item | evidence | action |
|---|---|---|
| `flashinfer-eidx-contig` overlay + `patches/upstream/pr-53574.diff` | #53574 merged 2026-08-31 (`699e180df4`); our pin is 2026-09-08 | retire both, and drop the overlay block from `apply_overlays.py` |
| `patches/upstream/0001-pr-52018-b12x-moe-v0.27.1.diff` and `b12x-moe-52018-vllm-only.diff` | #52018 merged 2026-08-21, before our pin; the `v0.27.1` in the name marks the lineage it was cut for, and the overlay auto-skips on v0.28+ | dead weight for this image; deleting them only touches the historical v0.27.1 path |

Every other patch corresponds to a PR that is still OPEN upstream, so none of them can be dropped:
`pr-53055.diff`, `pr-53425.diff`, `pr-53522.diff`, `pr-47988.diff`, `pr47988.diff`, `pr54631.diff`,
`0002-pr-50645-mhc-tilelang.diff`, `kv-offload-bounds-check.patch`, `0003-nvfp4-ds-mla-v0.27.1.patch`.

### Dependencies

| dependency | ours | latest | verdict |
|---|---|---|---|
| b12x | 1.2.6 (`B12X_VERSION`) | **1.3.0** on PyPI | do **not** bump: 1.3.0 was measured earlier and is not an improvement over 1.2.6 on this pair |
| flashinfer-python | 0.6.16.post3 | **0.6.18.post1** | candidate: our overlay adds TOPK 192/256, and flashinfer-ai #4380 (2026-08-08) added both upstream, so a bump may retire the overlay. Needs the overlay rework, a rebuild and the full protocol, because a new FlashInfer changes the kernels behind measured numbers |
| DeepGEMM | `DEEPGEMM_COMMIT` (a6b593d) | not checked | open: #53680 pins it to restore pure-fp8 1d1d kernels "until DeepGEMM#419"; if #419 or #337 has landed the pin can move and that PR can be revisited |

Nothing above has been changed in the tree: no base move, no patch deleted, no dependency bumped.
A patch deletion only takes effect at the next build, so each drop has to ship with a rebuild and the
gate plus trusted-protocol check.

### Dependency decisions, 2026-09-13

**The reference engine uses b12x too.** `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` ships `b12x 0.15.3`
(with `vllm 0.25.2.dev0+g752a3a504`), and its `b12x/attention/` holds `indexer`, `paged`,
`contiguous`, `mla`, `_cute` where ours holds `dsa_indexer`, `sparse_mla`,
`compressed_sparse_mla`, `dense_mla`, `varlen`. So b12x cannot be dropped: it is the MoE, the
sparse MLA attention and the FP8 linear kernels for this model on SM12x. It can only be versioned,
and the two generations are not interchangeable, because vLLM v0.29's b12x integration is written
against the 1.x API.

Bumped in `configs/pin.main-029.env`, with no upstream constraint and no measurement against them:
`TILELANG_VERSION` 0.1.13 -> 0.1.14, `QUACK_KERNELS_VERSION` 0.6.4 -> 0.6.5,
`APACHE_TVM_FFI_VERSION` 0.1.12 -> 0.1.13.post3, `TOKENSPEED_MLA_VERSION` 0.2.5 -> 0.2.8.

Held, each for a stated reason:

| pin | newer available | why held |
|---|---|---|
| `TRITON_VERSION=3.7.1` | 3.8.0 | vLLM v0.29 pins triton itself; moving it alone risks the kernel compilation vLLM does against its own pin |
| `NVIDIA_CUDA_NVDISASM_VERSION=13.3.73` | 13.4.49 | a different CUDA minor than this toolkit (13.3.1) |
| `DEEPGEMM_COMMIT=a6b593d` | 66081d4c9c (2026-09-10) | newer reintroduces the fp8 1d1d regression this pin exists to avoid; DeepGEMM#419 and #337 are both still open |
| `B12X_VERSION=1.2.6` | 1.3.0 | 1.3.0 dropped the `dsa_indexer` names the SM12x overlays import, and measured no faster than 1.2.6 on this pair |
| `TORCH_REF=release/2.14` | none | `release/2.14` is the highest `release/2.x` branch on pytorch/pytorch |
| `FLASHINFER_REF`, `DEEPEP_REF`, `INSTANTTENSOR_REF`, `FASTSAFETENSORS_REF` (main), `LMCACHE_REF` (dev) | n/a | these already track upstream heads at build time |

The shape of this matters: the newer versions available are not the faster ones on this pair. The
reference beats us with an older vLLM and an older b12x, so a version bump is not where the 42 ms
per step lives.

### b12x identity, verified 2026-09-13

`B12X_REF=3a437ab5168060e4d625f05e1625c04089f1ba37` is **the state PyPI published as b12x
1.3.0**, not a 1.2.6 commit:

| check | result |
|---|---|
| PyPI `b12x` 1.3.0 upload | 2026-08-25T17:20:50Z |
| newest commit before it | `3a437ab5168060e4d625f05e1625c04089f1ba37`, 2026-08-25T16:52:11Z, "fix(gemm): restore vLLM FP4 recipe entry points" - our `B12X_REF` |
| `attention/dsa_indexer/__init__.py` at that commit | sha256 `f60e8f9d...`, byte-identical to the 1.3.0 wheel's copy |
| `pyproject.toml` at that commit | `version = "1.2.6"` |
| the `1.2.6` git tag | a **different** commit, `2a4ed9ef23` |
| 1.3.0 git tag | does not exist |

So the image already runs 1.3.0's code while its own METADATA says 1.2.6, and **"bump b12x to
1.3.0" has nothing to apply**. The label comes from upstream's static pyproject, so
`B12X_VERSION` should stay 1.2.6 rather than be forced to 1.3.0, which would disagree with the
installed package. The earlier note in this file that the 1.3.0 release "dropped the dsa_indexer
names the overlays import" was wrong for the release and right for master (`081b235931`), and
has been corrected in both places it lived.

## proto-v0.1.0 base port (2026-09-12)

The image base moved from `v0.29.0` (`98dff2a81`) to the `proto-v0.1.0` tag commit
`69db1c26b4fe4474ab4c9df1c9701efac8bedde1` (main as of 2026-09-11). That tag is not a
release; it marks the `vllm-proto` Rust crate publication (#56365). Build tag is
`vllm-spark-0731:main-029-proto`, so the v0.29.0 image keeps its own tag.

### Why the earlier overlay gate was wrong

The previous gate applied each overlay with `--only <name>` against one accumulating
tree, which is wrong twice over:

- it skips every overlay that has no `--only` name. `patch_mqa_logits_sm12x_fallback`
  is one, and six other overlays patch the helpers it inserts, so all six looked dead;
- each `--only` run leaves its edits behind, so later runs see a partly patched tree.

`scripts/port_scan.py` replaces it: it imports `apply_overlays`, wraps every `patch_*`/`copy_*`
callable to capture its stdout, and calls `apply_main` once on a clean tree. One pass
gives the ordered result set. `applied`/`no-op`/`fail` is read from the overlay's own
`ok `/`skip ` lines, so a silent early return is visible. `--with-upstream-patches`
applies `patches/upstream/pr-*.diff` first, which is the image build's order and the
mode that exposes needled broken by a backport.

The true count at the new base was **ok=29 skip=9 fail=9**, not the 11 failures the
`--only` gate reported.

### Eight overlays needed work

| overlay | what broke | fix |
|---|---|---|
| `patch_einsum_sm12x_recipe` | `compute_fp8_einsum_recipe` gained a `block_size` parameter; the recipe is now `(1, 1, block_size)` | needle takes `block_size`, in both the needle and the replacement |
| `patch_o_proj_b12x` | `deep_gemm_fp8_o_proj` renamed `o_fp8` to `o_proj_input`, rewrote its docstring, and added `use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn` | needle rebuilt from the new body; the b12x early return stays ahead of the quant call |
| `patch_o_proj_einsum_e8m0` | the einsum moved inside `if use_fp8:`, so the block is at 8 spaces and uses `o_proj_input` | re-indented, same `deepgemm_post_process_fp8_weight_block` mirror |
| `patch_mqa_logits_sm12x_fallback` | `fp8_fp4_paged_mqa_logits` gained the `block_tables` stride clone upstream | clone hunk added to both the needle and the replacement |
| `patch_kv_kernel_split_padded_blhnc` | `create_kv_cache_views` gained an `MLAAttentionSpec.storage_block_size` override | upstream lines added to the needle; the `tokens_per_state` override still wins for DSV4 |
| `patch_cutlass_sm12x_guard` | `pr-53055` had to run first | see the ordering note below |
| `patch_nvfp4_ds_mla` | upstream now exempts `nvfp4_ds_mla` itself via `not ...endswith("_ds_mla")` | the `config/vllm.py` sub-edit is deleted, not re-anchored |
| `patch_b12x_moe_weight_prep_v028` | both needles gone | deleted; the branch it injects cannot be reached on this base |

`patch_b12x_moe_weight_prep_v028` is dropped because `UnquantizedMoeBackend` has no
`B12X` member on either base, and `B12xExperts.__init__` rejects any config whose
`weight_quant_dtype` is not `mxfp4`/`nvfp4`. On this base b12x MoE reaches its weight
prep through the MXFP4/NVFP4 oracles, which already call
`moe_kernel.fused_experts.process_weights_after_loading(layer)`. The injected
`or isinstance(self.moe_kernel.fused_experts, B12xExperts)` test was therefore dead
code, and its removal deletes nothing that ran.

### Nine overlays now skip themselves as already upstream

`patch_envs`, `patch_moe_backend`, `patch_utils_b12x`, `patch_mxfp4_oracle`,
`patch_mxfp4_process_weights`, `patch_mhc`, `patch_mqa_paged_cudagraph_safe`, and both
FlashInfer dispatch overlays (which skip for a separate reason: no `flashinfer` tree
beside the vllm package in the scan environment, so they may still apply in the image).

`patch_mqa_paged_cudagraph_safe` is the interesting one: the new base's
`fp8_fp4_paged_mqa_logits` now clones `block_tables` to force `stride(-1)==1`, which is
upstream's own fix for the CUDA-graph-safe gather. We keep the overlay because the
replacement is still needed if the fallback's first edit short-circuits.

### Overlay needles must key on post-patch text

`docker/Dockerfile.main-overlays` applies `patches/upstream/pr-*.diff` **before**
`apply_overlays.py`. Two of those patches edit
`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`:

- `pr-47988.diff` adds the SM12x `weight_shape[0] % 128 != 0` guard to
  `CutlassFp8BlockScaledMMKernel.can_implement`;
- `pr-53055.diff` rewrites that class's `is_supported` to the typed multi-line
  signature and the message `"CUTLASS block FP8 is not supported."`.

So the post-patch text of `is_supported` is what `patch_cutlass_sm12x_guard` must match,
and that is exactly the needle this repo already had. Re-anchoring it onto the pristine
base text (as the first pass did) breaks the build with
`CUTLASS FP8 SM12x exclusion: missing needle`. The needle is reverted; the check is to
run the patches and the overlays in build order, not either one alone.

### `pr-53425.diff` re-anchored

Its `flashinfer_sparse.py` hunk #1 dropped `from vllm.v1.attention.backend import
MultipleOf`. Upstream added `AttentionCGSupport` to that same line
(`AttentionCGSupport, MultipleOf`), so the deletion no longer matched and `patch`
reported the file failed, even though the remaining four hunks applied. `MultipleOf` is
used only by the `get_supported_kernel_block_sizes` override that hunk #2 deletes, so
the import still has to go, now as a modification. Section rebuilt against the new base;
backup at `patches/upstream/pr-53425.diff.pre-reanchor`.

### Two patch files the image never applies

The build globs `patches/upstream/pr-*.diff`. `pr54631.diff` and `pr47988.diff` have no
hyphen, so the glob misses them, and both touch only `tests/`, which the installed tree
does not need. That looks deliberate; the naming is the only thing keeping it that way.

### vllm-project/vllm#52708 audited, nothing to port

PR #52708 adds `12.1` to the published `TORCH_CUDA_ARCH_LIST`, and pins the SM120
CUTLASS and NVFP4/QuTLASS arch lists to `12.0a;12.1a` and `12.0` instead of the `12.0f`
family target. It is still open; its `pre-run-check` failure is a policy gate
(`PR must have the 'verified', 'ready', or 'ready-run-all-tests' label to run
pre-commit, or the author must have at least 4 merged PRs (found 0)`), not a defect in
the change.

It does not affect this image, for a reason that is checkable rather than argued:

- this build sets `TORCH_CUDA_ARCH_LIST=12.1a` (`docker/Dockerfile.main:43`,
  `configs/pin.main-029.env:70`), and the built image reports
  `torch.cuda.get_arch_list() == ['sm_121a']`. A single arch-specific target has the
  `a`-qualified MMA codegen that the PR says the `12.0f` family cubin lacks, so the
  launch abort it fixes cannot occur here;
- `cmake/utils.cmake:414-418` in both bases documents the same resolution: a `12.0f`
  source entry that does not match exactly falls back to a major-version match and
  "the output uses TGT's value to preserve the user's compilation flags", so
  `SCALED_MM_ARCHS "12.0f"` against `CUDA_ARCHS=12.1a` yields `12.1a`.

The PR's other half is a real CUDA 13.0 ptxas limit (`cvt with .e2m1x2` not supported
for `.target sm_121`), which is why its NVFP4 and QuTLASS lists stay on plain `12.0`.
That is the one part worth re-checking if this project ever builds both 12.0 and 12.1.

### State

Patches and overlays both apply clean on the new base:
`patch -p1 --forward --dry-run` succeeds for all four `pr-*.diff`, and
`apply_overlays.py --stack main` reaches rc=0 with 87 `ok` edits after them.
The image rebuild is running; **no number in this file comes from the new base yet**.

### Follow-ups from the port (2026-09-12), not acted on

**`pr-47988.diff` has drifted from its PR.** Our pinned copy dates from 2026-08-25;
#47988 was updated 2026-09-12T07:41 (`+158/-10`, 3 files). Comparing the PR's current
source hunks with ours:

| | ours | #47988 today |
|---|---|---|
| `scaled_mm/cutlass.py` | `can_implement` SM12x `weight_shape[0] % 128 != 0` guard, plus two explanatory comment blocks | same code, comments trimmed |
| `quantization/utils/fp8_utils.py` | makes the Triton E8M0 upcast in `w8a8_triton_block_scaled_mm` unconditional instead of `is_rocm() or is_xpu()` | a different function: the same upcast in `requant_weight_ue8m0_inplace` (line ~1068) |

Both halves are still missing from our base, so neither is stale in the sense of being
unnecessary:

- `w8a8_triton_block_scaled_mm` at `69db1c26b4` still gates the upcast on
  `is_rocm() or is_xpu()`, so our hunk still changes behaviour;
- `requant_weight_ue8m0_inplace` (line 1011) has no E8M0 upcast at all, so the PR's
  current hunk would add something our tree lacks.

Next action, after the rebuild has been measured: refresh the patch to the current
revision for the added `requant_weight_ue8m0_inplace` hunk, keep our
`w8a8_triton_block_scaled_mm` hunk, and verify both apply in build order. Not done here
because the rebuild was already running against the patch as pinned.

**Tracked PR state, read 2026-09-12.** None of these is a code defect on our side; they
need the author's or a maintainer's action, which is not something this repo can do for
them.

| PR | state | note |
|---|---|---|
| #53425 | OPEN, MERGEABLE/BLOCKED | updated 2026-09-12T10:41; the diff we backport matches its file set |
| #53055 | OPEN, MERGEABLE/BLOCKED | updated 2026-09-11 |
| #47988 | OPEN, MERGEABLE/BLOCKED | see the drift note above |
| #53522 | OPEN, mergeability UNKNOWN | last touched 2026-09-08 |
| #52499 | OPEN, mergeability UNKNOWN | last touched 2026-09-06; still comment-only for us |
| #54631 | OPEN, mergeability UNKNOWN | last touched 2026-09-11; test-only file, never applied by the image |
| #41834 | OPEN, mergeability UNKNOWN | `+32347/-1477` across 221 files; needs a rebase, still comment-only for us |
| #52708 | OPEN, not ours | see the audit above; its only CI failure is the label gate |

### Follow-ups from the port (2026-09-12), not acted on

**`pr-47988.diff` has drifted from its PR.** Our pinned copy dates from 2026-08-25;
#47988 was updated 2026-09-12T07:41 (`+158/-10`, 3 files). Comparing the PR's current
source hunks with ours:

| | ours | #47988 today |
|---|---|---|
| `scaled_mm/cutlass.py` | `can_implement` SM12x `weight_shape[0] % 128 != 0` guard, plus two explanatory comment blocks | same code, comments trimmed |
| `quantization/utils/fp8_utils.py` | makes the Triton E8M0 upcast in `w8a8_triton_block_scaled_mm` unconditional instead of `is_rocm() or is_xpu()` | a different function: the same upcast in `requant_weight_ue8m0_inplace` (line ~1068) |

Both halves are still missing from our base, so neither is stale in the sense of being
unnecessary:

- `w8a8_triton_block_scaled_mm` at `69db1c26b4` still gates the upcast on
  `is_rocm() or is_xpu()`, so our hunk still changes behaviour;
- `requant_weight_ue8m0_inplace` (line 1011) has no E8M0 upcast at all, so the PR's
  current hunk would add something our tree lacks.

Next action, after the rebuild has been measured: refresh the patch to the current
revision for the added `requant_weight_ue8m0_inplace` hunk, keep our
`w8a8_triton_block_scaled_mm` hunk, and verify both apply in build order. Not done here
because the rebuild was already running against the patch as pinned.

**Tracked PR state, read 2026-09-12.** None of these is a code defect on our side; they
need the author's or a maintainer's action, which is not something this repo can do for
them.

| PR | state | note |
|---|---|---|
| #53425 | OPEN, MERGEABLE/BLOCKED | updated 2026-09-12T10:41; the diff we backport matches its file set |
| #53055 | OPEN, MERGEABLE/BLOCKED | updated 2026-09-11 |
| #47988 | OPEN, MERGEABLE/BLOCKED | see the drift note above |
| #53522 | OPEN, mergeability UNKNOWN | last touched 2026-09-08 |
| #52499 | OPEN, mergeability UNKNOWN | last touched 2026-09-06; still comment-only for us |
| #54631 | OPEN, mergeability UNKNOWN | last touched 2026-09-11; test-only file, never applied by the image |
| #41834 | OPEN, mergeability UNKNOWN | `+32347/-1477` across 221 files; needs a rebase, still comment-only for us |
| #52708 | OPEN, not ours | see the audit above; its only CI failure is the label gate |

### Which o_proj branch the 0731 model actually takes (checked 2026-09-12)

One risk of the re-anchors was worth closing: `deep_gemm_fp8_o_proj` now branches on
`use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn`, and `compute_fp8_einsum_recipe`'s
third element is reused by the new base as `quant_group_size`. If the checkpoint's
`wo_a` were packed NVFP4/MXFP4 (the `[1, 32]` branch), the SM12x override forcing
`(1, 128, 128)` would disagree with a 32-wide quant group.

`~/models/ds4-flash-0731/config.json` settles it:

```json
"quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                        "quant_method": "fp8", "scale_fmt": "ue8m0",
                        "weight_block_size": [128, 128]}
```

So `use_fp8` is True, the fp8 einsum path is the one that runs, and the block size is
128, matching the forced recipe. Both o_proj overlays are load-bearing on this model
rather than dead code, and `scale_fmt: ue8m0` is exactly why the recipe overlay exists
(the Python fallback cannot consume packed INT32 UE8M0 scales). The `[1, 32]` branch is
for MXFP4 checkpoints, which this run does not use.

### quack-kernels 0.6.5 needed a pin-script fix (2026-09-12)

The phase-1 build of `main-029-proto` failed at `stage-0 22/24` with:

```
quack rewrite expected 2, got 0 in [PosixPath('/tmp/quack-unpacked/quack_kernels-0.6.5.dist-info/METADATA')]
```

Cause, measured rather than guessed. `patch_cutlass_dsl.pin_text` rewrites only exact
`==` pins. The two METADATA files differ in shape:

| artifact | requirement in METADATA | rewrites |
|---|---|---|
| quack-kernels 0.6.4 | `nvidia-cutlass-dsl==4.6.2` plus `nvidia-cutlass-dsl[cu13]==4.6.2; extra == "cu13"` | 2 |
| quack-kernels 0.6.5 | `nvidia-cutlass-dsl>=4.7` plus `nvidia-cutlass-dsl[cu13]>=4.7; extra == "cu13"` | 0 |

4.7.0 satisfies `>=4.7`, so 0.6.5 needs no rewrite at all. The `QUACK_KERNELS_VERSION`
bump 0.6.4 -> 0.6.5 recorded under "Dependency decisions" above was applied without
this check, and the hard-coded `expected 2` guard is what broke, not the bump.

Fix: `docker/pin_quack.py` verifies the requirement set instead of counting rewrites.
`unsatisfied()` parses each `nvidia-cutlass-dsl*` requirement with
`packaging.requirements.Requirement` and reports the ones the target version does not
satisfy, so a requirement that is genuinely wrong still fails loudly whether or not the
rewriter touched it, and an unhandled pin form (`~=`, `<`) is caught too. Regression
test at `tests/test_pin_quack.py` (5 tests, green), alongside the existing
`tests/test_pin_cutlass_dsl.py`. Backup: `docker/pin_quack.py.pre-proto`; the failed
build log is `~/build-proto-phase1.fail1.log`.

Note for the next dependency bump: any `COPY` ahead of a build step invalidates it, so
touching `docker/pin_quack.py` re-runs the vLLM compile even for a metadata-only fix.

### The kernel-wheel version set is mutually constrained (2026-09-12)

The "Dependency decisions" list above says the four bumped versions were moved "with no
upstream constraint and no measurement against them". That was wrong for two of the four,
and reading the declared bounds rather than inferring them settles the whole set:

| wheel | requires |
|---|---|
| tilelang 0.1.11 | `apache-tvm-ffi>=0.1.10,~=0.1.0` |
| tilelang 0.1.12 | `apache-tvm-ffi<=0.1.11,>=0.1.10` |
| tilelang 0.1.13, 0.1.14 | `apache-tvm-ffi<0.1.13,>=0.1.11` |
| tokenspeed-mla 0.2.5 | `apache-tvm-ffi==0.1.13` |
| tokenspeed-mla 0.2.8 | `apache-tvm-ffi==0.1.13.post3` |
| quack-kernels 0.6.4 | `nvidia-cutlass-dsl==4.6.2` (and the `[cu13]` extra) |
| quack-kernels 0.6.5 | `nvidia-cutlass-dsl>=4.7` (and the `[cu13]` extra) |
| b12x (pinned commit) | five `nvidia-cutlass-dsl==4.6.2` lines |

So `tilelang >= 0.1.12` and `tokenspeed-mla` cannot both be satisfied: tokenspeed demands
exactly `0.1.13`/`0.1.13.post3`, tilelang demands `<0.1.13`. Only tilelang 0.1.11 admits
0.1.13.post3, through its loose `~=0.1.0`.

This is not enforced at install: the Dockerfile installs the three kernel wheels with
`--no-deps`, so an incompatible set lands happily and fails later. Observed, at
`stage-0 22/24` of the first proto build, with `apache-tvm-ffi==0.1.13.post3` present:

```
File ".../tilelang/layout/swizzle_mode.py", line 12, in <module>
  class SwizzleMode(Enum, type_key="tl.SwizzleMode"):
File ".../tvm_ffi/dataclasses/enum.py", line 419, in _resolve
RuntimeError: 'tl.SwizzleMode' has no enum at string index 'NONE'
```

Resolution, and the reasoning for each:

- `APACHE_TVM_FFI_VERSION=0.1.12`: the highest version any tilelang >= 0.1.12 accepts, and
  the value the v0.29.0 image already ran. Reverted from 0.1.13.post3.
- `TOKENSPEED_MLA_VERSION=0.2.8` stays. Its `==0.1.13.post3` pin is unmet with any
  tilelang >= 0.1.12, and the previous image already ran tokenspeed 0.2.5 with its
  `==0.1.13` pin equally unmet, so that declaration is not enforced in practice. 0.2.8 is
  the newer MLA kernel set and costs nothing relative to 0.2.5 on this axis.
- `TILELANG_VERSION=0.1.14` stays: its bound admits 0.1.12. If it fails to import against
  0.1.12 the fix is 0.1.13, not a tvm-ffi change.

Residual risk to check at boot rather than argue away: tokenspeed-mla 0.2.8 declares
`apache-tvm-ffi==0.1.13.post3` and now loads against 0.1.12. If MLA output misbehaves,
reverting to 0.2.5 restores the exact pairing the v0.29.0 image used.

Two process notes for the next bump round. Changing a build arg invalidates the cache from
the first step that references it, and `configs/pin.main-029.env` feeds many of them, so a
pin edit costs a full vLLM rebuild. `docs/`, `tests/`, and `*.md` are in `.dockerignore`,
so tracker and test edits do not.

### `assert_image.py` needed two updates for the new base (2026-09-12)

`assert_image.py --stack main` runs inside the image as the phase-2 gate, and two of its
assertions were keyed to v0.29.0 text. It caught both, which is the gate working.

1. `assert get_kv_quant_mode("nvfp4_ds_mla") == KVQuantMode.NVFP4` failed. The base now
   gives `nvfp4_ds_mla` its own mode and tests it ahead of the `nvfp4` prefix:

   ```python
       # Must precede the ``nvfp4`` prefix test below, which would otherwise match.
       if kv_cache_dtype == "nvfp4_ds_mla":
           return KVQuantMode.NVFP4_DS_MLA
       if kv_cache_dtype.startswith("nvfp4"):
           return KVQuantMode.NVFP4
   ```

   `NVFP4 = 5` ("packed fp4 data + fp8 block scales"), `NVFP4_DS_MLA = 10`
   ("opaque-bytes NVFP4 DS-MLA layouts"). Updated to assert `NVFP4_DS_MLA` for
   `nvfp4_ds_mla` and, separately, `NVFP4` for plain `nvfp4`, so the distinction stays
   covered.

2. `assert 'cache_dtype == "nvfp4"' in guard` failed next, and the assertion under it
   (`"startswith" not in guard`) would have failed too. Both encoded the old overlay's
   narrowing of `validate_nvfp4_kv_cache_with_mla`; upstream now writes
   `startswith("nvfp4") and not endswith("_ds_mla")`, which uses `startswith` legitimately.
   Replaced with an assertion of the actual requirement: the guard must exempt `_ds_mla`,
   so its source must contain `endswith("_ds_mla")`.

Neither is a code defect. The port had already removed the now-redundant
`config/vllm.py` sub-edit, and the gate was still demanding its output; the fixes are on
the gate side, and both were verified by re-running phase 2.

### Phase 2 result, 2026-09-12

`scripts/03-apply-main-overlays-029.sh vllm-spark-0731:main-029-proto` is green:

```
applied pr-47988.diff
applied pr-53055.diff
applied pr-53425.diff
applied pr-53522.diff
... 87 "ok" overlay edits ...
overlaid vllm-spark-0731:main-029-proto
```

Image `sha256:77841395cfcd34be7e9e5634d9e7870ba618500c4ebb684f507c51383fe229f8`,
28.85 GB. Phase 1 (from-scratch base build) is
`sha256:31aff322427fe3b5cd8a3ab3c8b19bcfd4cd1a6a8117f0d2921dc6ccd081bd71`. The v0.29.0
image keeps its own `:main-029` tag, and `IMAGE` in `configs/pin.main-029.env` is now
`${IMAGE:-vllm-spark-0731:main-029-proto}`, so a differently tagged build of the same pin
can be served without editing the pin.

Still no measurement. The next step is boot, gates, and then the protocol.

### proto base: phase 1, phase 2, boot, and validate (2026-09-12)

The whole chain is green.

| step | result |
|---|---|
| phase 1 build | `sha256:31aff322427fe3b5cd8a3ab3c8b19bcfd4cd1a6a8117f0d2921dc6ccd081bd71`, 28.8 GB |
| phase 2 overlays | four `pr-*.diff` applied, 87 `ok` edits, `assert_image.py` green, `sha256:77841395cfcd...`, 28.85 GB |
| to spark2 | `scripts/02-copy-main.sh`, same image ID on both nodes |
| boot | spark2 (rank 1) first, spark1 (rank 0) 80 s later, `scripts/05-serve.sh main-029` |
| healthy | 265 s |
| gate | `scripts/06-validate.sh main` -> `validate ok` |

The engine it brought up: `v0.1.1.dev0+g69db1c26b`, `tensor_parallel_size=2`, `nnodes=2`, NCCL
over `tcp://10.0.1.1`, `dtype=torch.bfloat16`, `quantization=deepseek_v4_fp8`,
`kv_cache_dtype=nvfp4_ds_mla`, `moe_backend=b12x`, `linear_backend=b12x`,
`attention_backend=B12X_MLA_SPARSE`, `block_size=256`, `max_num_seqs=6`,
`speculative_config={'method': 'dspark', 'num_speculative_tokens': 5}`, `load_format=instanttensor`.
Gates: greedy `' Paris. The capital of Spain is Madrid. The capital of Italy is Rome...'` with
`first_token ' Paris' logprob -0.2767 n_tie=1`, chat `content 'Paris'`.

### The new base needs a higher `--gpu-memory-utilization`

The first boot at the pin's `GPU_MEMORY_UTILIZATION=0.8` died in `_initialize_kv_caches`:

```
ValueError: To serve at least one request with the model's max seq len (65536), 9.48 GiB KV cache
is needed, which is larger than the available KV cache memory (4.42 GiB). ... the estimated
maximum model length is 11656.
```

vLLM names the mechanism itself:

```
CUDA graph memory profiling is enabled (default since v0.21.0). The current
--gpu-memory-utilization=0.8000 is equivalent to --gpu-memory-utilization=0.7611 without CUDA graph
memory profiling. To maintain the same effective KV cache size as before, increase
--gpu-memory-utilization to 0.8389. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
```

Measured on this rig, same model and weights (79.34 GiB load, unchanged):

| | graph capture | available KV at 0.8 |
|---|---|---|
| old base | 2.84 GiB | 10.49 GiB |
| this base | 4.73 GiB | 4.42 GiB |

At `GPU_MEMORY_UTILIZATION=0.8389` the new base reports **13.28 GiB** and boots. Two options, both
recorded: keep 0.8389, or set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` to reproduce the old
base's accounting. `05-serve.sh` takes `GPU_MEMORY_UTILIZATION` from the environment, so this is a
per-run choice; the pin still says 0.8.

### Open: our KV cache holds far fewer tokens per GiB than the reference

Not resolved, and it matters for long-context concurrency.

| stack | KV tokens | available GiB | tokens/GiB |
|---|---|---|---|
| this base, `nvfp4_ds_mla` | 74,058 | 13.28 | 5,576 |
| anemll k7 (`anemll_final2.log`) | 1,660,377 | 13.87 | 119,710 |
| anemll base (`anemll-base0731.log`) | 671,054 | 14.07 | 47,694 |
| our old base, indexer ablated (`ablindexer.run.log`) | 136,216 | 10.49 | 12,986 |

The two anemll rows run the same `kv_cache_dtype` string and reach 9x to 20x more tokens per GiB, so
the per-token footprint of our stack is much larger than the reference's. The two "our old base" rows
are not a valid baseline: `ablindexer` and `abreshape` are ablation runs with the indexer disabled, so
they are missing a cache the shipped stack allocates.

Consequence: at 65,536 context we fit 74,058 tokens, i.e. `Maximum concurrency ... 1.13x`, so the
long-context concurrency is capped near one request. Comparing like for like on concurrency to the
window, the old base was 136,216 tokens for a 131,072 window (1.04x) and this base is 74,058 for
65,536 (1.13x), so the new base is not worse on that axis, but neither is close to the reference.

Next diagnostic, not yet run: print the per-cache-group page sizes from both stacks. vLLM does not
log them at INFO here, so this needs a probe inside each image rather than log mining.

## 2026-09-13: the proto-v0.1.0 rebase measured, and it is a wash

Both arms run the trusted protocol on the same rig, same day, same knobs: chat mode,
`thinking=false`, seed 1234, 512-token generations, three passes, median, both nodes
restarted clean, `GPU_MEMORY_UTILIZATION=0.8389`, dspark k=7, TP=2.

| level | old base `:main-029` | spread | proto base `:main-029-proto` | spread | proto vs old |
|---|---|---|---|---|---|
| c1 | 36.9 | 1.9% | 37.8 | 5.8% | +2.4% |
| c3 | 76.5 | 3.8% | 77.6 | 5.7% | +1.4% |
| c5 | 106.9 | 10.5% | 105.0 | 6.1% | -1.8% |
| c6 | 133.7 | 0.6% | 130.8 | 3.1% | -2.2% |

All four deltas sit at or inside the spread. **The base move to `proto-v0.1.0` buys no
throughput.** It also costs nothing: correctness is identical (below).

Gates passed on every pass of both arms: `gate_france ' Paris. The capital of Spain'`,
`gate_9x8 '72, 9x9'`.

### The previously "shipped" numbers were optimistic, and should not be used as the baseline

The k=7 figures recorded earlier in this file as the shipped reference
(40.0 / 81.7 / 108.8 / 120.1 at c1 / c3 / c5 / c6) do not reproduce. The same old image,
measured today under the protocol above, gives 36.9 / 76.5 / 106.9 / 133.7. That is -8% at
c1, -6% at c3, -2% at c5, and **+11% at c6**: the shipped set was pessimistic at c6 and
optimistic at c1. Whatever produced them differed in environment or thermometer, so
today's same-conditions run is the baseline to measure against from here.

### Mechanism: same per-step output, fewer steps per second

Directly comparable meter output at c6:

| | tokens/step | accept | steps/s | agg tok/s |
|---|---|---|---|---|
| old base | 4.345 | 67.0% | 30.76 | 133.65 |
| proto base | 4.327 | 66.6% | 29.51 | 127.69 |

Draft quality is unchanged (4.345 vs 4.327 tokens/step, 67.0% vs 66.6% acceptance), so the
proto base is about 4% slower per decode step rather than worse at speculation. Nothing in
the port changed the draft path, so the likeliest owner is the newer vLLM main's target
step itself, not the overlays.

### Where this leaves the goal

The reference (anemll k7) is reported at roughly 160-190 tok/s aggregate at c6 by its own
recipe, and this stack reaches 130.8 to 133.7. The rebase was not the lever, so the gap is
elsewhere: per-step time on our side, not acceptance.

Still open and unresolved, from the boot section above: our KV cache holds 5,576 tokens per
GiB against the reference's 47,694 to 119,710 at the same `kv_cache_dtype` string. That is
a 9x to 20x footprint difference and it caps long-context concurrency, but it cannot explain
these numbers: 512-token requests at c1-c6 need only a few thousand KV tokens, so the pool
is not binding at any measured level. It is a separate problem, and worth its own
investigation rather than being folded into this one.

### Correction: our pin serves k=5, not k=7 (2026-09-13)

Found while looking for a valid reference baseline, and it invalidates part of the earlier
comparison in this file.

`configs/pin.main-029.env` sets:

```
SPEC_METHOD=dspark
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-36}"   # 6 * (5+1)
```

So the stack as pinned drafts **5** tokens. The recipe this project measures against,
`d1-k7-seeded.yaml`, sets `num_speculative_tokens: 7` and the reference's own working
launch (`anemll_k7.log`, `non-default args` dump) used
`{'method': 'dspark', 'num_speculative_tokens': 7}` with
`max_cudagraph_capture_size: 48` (= `max_num_seqs * (k+1)` = 6 * 8).

Consequences:

1. The line in the measurement section above that treats 40.0 / 81.7 / 108.8 / 120.1 as
   the "shipped k=7" numbers is comparing against a configuration that was not what the
   pin served. The two arms I measured (36.9 / 76.5 / 106.9 / 133.7 and
   37.8 / 77.6 / 105.0 / 130.8) were both at k=5. They remain a valid A/B of the two
   bases against each other, because both arms used the same k, but they are not a
   measurement of the shipped configuration.
2. `max_cudagraph_capture_size` must cover `max_num_seqs * (k + 1)`. The pin's 36 matches
   k=5. Running k=7 without raising it to 48 truncates capture, and a truncated capture
   silently costs decode throughput.
3. The reference's other differences from our pin, from the same args dump:
   `moe_backend: flashinfer_b12x` (ours `b12x`), `async_scheduling: True`,
   `max_num_batched_tokens: 12288` (same as ours), `long_prefill_token_threshold: 1024`
   (same), `generation_config: 'vllm'`, `enable_chunked_prefill: True`.

Next measurement, running now: the proto base at `NUM_SPECULATIVE_TOKENS=7` with
`MAX_CUDAGRAPH_CAPTURE_SIZE=48`, same protocol and utilisation, tagged `proto-k7b`.

Note on the reference baseline for the goal in force: the numbers previously recorded as
"anemll k7" came from the **abliterated** checkpoint
(`drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32`), which is a different
workload from the base `DeepSeek-V4-Flash-0731` this stack serves, and the user has
already ruled that model out. A valid reference arm therefore has to run the anemll image
on the base checkpoint with its own k=7 flags. Attempting it with `scripts/05-serve.sh
golden` as pinned failed at engine init with a DeepGEMM layout assertion
(`sf.size(-2) == ceil_div(mn, gran_mn)`, layout.hpp:97) plus a tilelang
`libcudart_stub.so: undefined symbol: cudaDeviceReset`, so the golden pin's flags are not
the recipe that worked; the args dump above is.

### Win 1: `--async-scheduling` (+3 to +6% at c3/c5/c6) (2026-09-13)

The reference's args dump has `async_scheduling: True`; our pin never set it. The serve
script had no way to pass it, so `SERVE_EXTRA_ARGS` was added to `scripts/05-serve.sh`
(word-split into `EXTRA_ARGS`, appended after `--trust-remote-code`) so flags can be A/B'd
without editing the script per experiment.

proto base, k=5, `GPU_MEMORY_UTILIZATION=0.8389`, 5 passes, warmup via the full sweep:

| level | proto baseline (3p) | proto + async (5p) | delta | async spread |
|---|---|---|---|---|
| c1 | 37.8 | 37.6 | -0.5% | 14.4% |
| c3 | 77.6 | 80.6 | +3.9% | 9.9% |
| c5 | 105.0 | 110.8 | +5.5% | 2.8% |
| c6 | 130.8 | 135.0 | +3.2% | 4.6% |

Adopted. Caveat recorded honestly: the baseline is 3 passes against async's 5, and at c6
the +3.2% sits just above the 3.1% baseline spread, so the c6 delta is marginal on its own;
c5's +5.5% against a 2.8% spread is the firmer part. A warm 5-pass re-measure of the
baseline would settle it, and a valid head-to-head needs the reference arm anyway.

Closed lever, for the record: the reference's `moe_backend: flashinfer_b12x` is not a
different backend, it is the anemll fork's name for the same kernels. Our base rejects it:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE. Expected one of
['b12x', 'deep_gemm', 'flashinfer_trtllm', ..., 'emulation'].
```

`b12x` is that list's first entry, so our pin already selects the equivalent path.

### Valid reference baseline, and the gap it defines (2026-09-13)

After long trying, the reference arm runs. `scripts/05-serve.sh golden` does not work: its
flags fail engine init with a DeepGEMM layout assertion
(`sf.size(-2) == ceil_div(mn, gran_mn)`, layout.hpp:97) plus a tilelang
`libcudart_stub.so: undefined symbol: cudaDeviceReset`. The recipe that works is
`tonyd2wild/sparkrun/anemll-base0731.yaml` (base `DeepSeek-V4-Flash-0731`, reference image,
k=7, capture 48, util 0.82, ctx 262144), launched with `scripts/spark-launch.sh`.

Launching it through that path needs a care point: `spark-launch.sh` runs
`pkill -9 -f '[s]parkrun'`, which matches any process whose command line contains that
string, including the caller's. Invoking it with the original
`~/tonyd2wild/sparkrun/<recipe>.yaml` argument kills the invoking shell. The recipe is
therefore staged at `~/goal/ref-base0731.yaml` and launched by `~/goal/launch-refbase.sh`.

Both arms, same rig, same day, same protocol (chat, thinking=false, seed 1234, 512 tokens,
5 passes, median, `/dev/shm` swept, both nodes restarted):

| level | anemll reference (k=7) | spread | proto base (k=5, async) | spread | gap |
|---|---|---|---|---|---|
| c1 | 64.0 | 5.9% | 37.6 | 14.4% | -41% |
| c3 | 113.1 | 8.2% | 80.6 | 9.9% | -29% |
| c5 | 148.1 | 24.7% | 110.8 | 2.8% | -25% |
| c6 | 160.1 | 4.0% | 135.0 | 4.6% | -16% |

Reference gates pass on every pass. This also confirms the "~160-190 at c6" claim: 160.1
measured. The goal in force is to close this, i.e. to exceed 160.1 at c6 and to win the
c1+c3+c5+c6 sum.

### What the two engines actually differ on

Read from each engine's own startup dump, not guessed:

| | reference | ours |
|---|---|---|
| compilation mode | `CompilationMode.VLLM_COMPILE` (inductor, 17 splitting_ops) | `CompilationMode.NONE`, `custom_ops: []` |
| cudagraph capture sizes | `[1,2,4,8,16,24,32,40,48]` | `[1,2,4,8,16,24,32]` |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` | `1` (`configs/env.spark.sh:47`) |
| `disable_custom_all_reduce` | `True` | `False` |
| `moe_backend` | `flashinfer_b12x` | `b12x` (same kernels, fork alias) |
| `linear_backend` | `auto` | `b12x` |
| draft tokens | 7 | 5 |
| `max_model_len` | 262144 | 65536 |
| `ir_op_priority.rms_norm` | `['native']` | `['vllm_c','native']` |
| vLLM version | `v0.25.2.dev0+g752a3a504.d20260714` | `v0.1.1.dev0+g69db1c26b` |

Step times derived from the meter (tok/s / tokens-per-step): at c6 the reference is
160.1 / ~4.6 = ~34.8 steps/s = ~28.7 ms per step against our 135.0 / 4.327 = 31.2 steps/s
= ~32.0 ms. At c1 the difference is much larger: the reference is ~72 ms per step against
our ~114 ms. A 1.58x per-step gap at batch 1 and 1.10x at batch 6 points at fixed per-step
overhead (launch count and graph structure), which is exactly where `VLLM_COMPILE` and a
non-breakable cudagraph should show up. Those are the next two experiments.

### Closing the gap: the levers, measured (2026-09-13)

All rows: proto base, same rig, both nodes restarted with `/dev/shm` swept, 512-token chat
generations, seed 1234, 4-5 passes, median. Reference row repeated for scale.

| config | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|
| reference anemll, k=7 capture 48 | 64.0 | 113.1 | 148.1 | 160.1 | 485.3 |
| proto, k=5, as pinned | 37.8 | 77.6 | 105.0 | 130.8 | 351.2 |
| + `--async-scheduling` | 37.6 | 80.6 | 110.8 | 135.0 | 364.0 |
| + `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | 64.2 | 115.8 | 144.3 | 160.3 | 484.6 |
| + `COMPILATION_MODE=3` (VLLM_COMPILE) | 61.8 | 113.4 | 156.4 | 159.7 | 491.3 |
| + `COMPILATION_MODE=3` and capture `[1,2,4,8,16,24,32,36]` | 62.2 | 112.9 | 151.2 | 160.2 | 486.5 |
| + `NUM_SPECULATIVE_TOKENS=7` capture 48 (with non-breakable graphs) | 64.5 | 114.7 | 144.8 | 159.9 | 483.9 |

What each result says:

- **`VLLM_USE_BREAKABLE_CUDAGRAPH=0` was the whole gap.** From 37.6/80.6/110.8/135.0 to
  64.2/115.8/144.3/160.3, i.e. +71% at c1, +44% at c3, +30% at c5, +19% at c6. `configs/env.spark.sh`
  defaulted it to 1; the reference recipe sets 0. Breakable cudagraphs split the decode step
  into several graphs, and on this 2-node rig that costs far more than it buys. This one flag
  took us from 27% behind to parity.
- **`--async-scheduling`** adds +3 to +6% at c3/c5/c6 and is what the reference runs.
- **`COMPILATION_MODE=3`** (inductor, which the reference runs and we did not) buys c5
  (+8.4%, 144.3 to 156.4) but costs c1 (-3.7%) and c3 (-2.1%), for +1.4% on the sum. Best sum
  so far, and the only config that beats the reference's sum, but it does not beat it at c6.
- **Capture size 36** was a hypothesis that failed: we capture `[1,2,4,8,16,24,32]` and a full
  6-sequence k=5 decode batch is 36 tokens, so I expected that batch to run ungraphed while the
  reference's list reaches 48. Adding 36 explicitly changed nothing (c6 159.7 to 160.2), so the
  scheduler was not falling off a graph. Closed.
- **k=7 with capture 48** is a wash under non-breakable graphs (483.9 vs 484.6). The earlier
  catastrophic k=7 numbers were a warmup artifact of breakable graphs, not the k. The pin's k=5
  is fine.

State against the goal: parity, not a win. Best sum is config with `COMPILATION_MODE=3` at
491.3 against 485.3 (+1.2%), but the criterion needs a *strict c6 win* and c6 sits at
159.7-160.3 for us and 160.1 for the reference across every config, including the reference's
own best. c6 looks like a ceiling of this workload on this rig, and c1 is where we now lose
(61.8-62.2 against 64.0) once compile is on.

Next lead, from an earlier blocked investigation recorded in this file: a per-step copy family
of 156 launches x 11,641,344 bytes at 199 us, attributed at 42.9 ms/step. At c1 our whole step
is about 67 ms, so if that copy family is real it is most of the c1 step and c1 is the level we
now lose. That attribution was never finished, and it is the next thing to profile.

### Two more results: MoE A16 forcing, and the profiler endpoint (2026-09-13)

`VLLM_B12X_MOE_FP4_FORCE_A16=1` on top of the best-sum config
(`VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `--async-scheduling`, `COMPILATION_MODE=3`), 4 passes:

| level | compile | compile + A16 | delta |
|---|---|---|---|
| c1 | 61.8 | 63.6 | +2.9% |
| c3 | 113.4 | 117.3 | +3.4% |
| c5 | 156.4 | 141.3 | -9.7% |
| c6 | 159.7 | 160.3 | +0.4% |
| sum | 491.3 | 482.5 | -1.8% |

Net negative, so the knob stays off (it is off by default; the overlay only declares it).
Worth keeping for the record that it produced our best c3 (117.3, above the reference's
113.1) and our best c1 tie, while costing c5, so it trades mid-level for high-level
throughput rather than being uniformly worse.

The torch-profiler route does not work on this engine: `PROFILE_ENABLE=1` is accepted and
the serve starts, but `POST /stop_profile` returns `{"detail":"Not Found"}` and the trace
directory (`/cache/huggingface/tprof`) is never created, so `drive-tprof.sh` captures
nothing. Op-level attribution needs a different instrument (nsys via
`drive-nsys-profile.sh`, or the in-tree `tprof_*.py` helpers against a trace captured some
other way). That is a tooling gap, not a result.

### Where the config-level search has landed

Six configurations measured against the same reference arm:

| config | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|
| reference | 64.0 | 113.1 | 148.1 | 160.1 | 485.3 |
| k=5, no-compile, async, non-breakable | 64.2 | 115.8 | 144.3 | 160.3 | 484.6 |
| + `COMPILATION_MODE=3` | 61.8 | 113.4 | 156.4 | 159.7 | **491.3** |
| + compile and capture 36 | 62.2 | 112.9 | 151.2 | 160.2 | 486.5 |
| + `COMPILATION_MODE=3` and A16 | 63.6 | 117.3 | 141.3 | 160.3 | 482.5 |
| k=7 capture 48, non-breakable | 64.5 | 114.7 | 144.8 | 159.9 | 483.9 |

c6 lands in 159.7 to 160.3 across every one of our configs and 160.1 for the reference.
Six configurations, three of them above the reference on the sum and two above at c6, but
never by more than the measured spread, and never simultaneously at c6 *and* on the sum.
That pattern is what a genuine ceiling looks like: at 6 concurrent 512-token generations
this workload pins at about 160 tok/s on both engines, so a strict c6 win will not come
from configuration. The remaining path is code: reduce per-step work in a way the
reference does not have. The lead on file is the copy family an earlier investigation
attributed at 42.9 ms/step (156 launches x 11,641,344 bytes at 199 us), which is
unfinished and is most of our c1 step roughly by itself.

### The copy-family instrument is ready to run at last (2026-09-13)

HANDOFF section 135 ends with "nothing further can be concluded about the copies until an
instrument works" and declares the `LD_PRELOAD` CUDA launch interposer unsafe, listing the
three fixes it needs: abort on a missing real symbol instead of returning -1, interpose
`cudaLaunchKernelExC` as well as `cudaLaunchKernel`, and add a mode that counts geometries
without filtering so the filter can be chosen from data.

`.scratch/dcopy_trace.c` on disk is already **v4**, which implements all three: it `abort()`s
when `dlsym(RTLD_NEXT, ...)` is NULL on either entry point, defines both entry points, and
treats `DCOPY_TRACE_BLOCK_X=0` as "any" so it can count first and filter later.

**The deployed shim was stale, which is why section 135's failure was never re-tested.**
`~/.cache/huggingface/inject/dcopy_trace.so` was dated Sep 12 11:49 at 70,784 bytes, built
from the older source; the v4 source is Sep 13 02:38. So the crashing build and the fixed
source had never been brought together.

Built v4 in the image and deployed it:

```
docker run --rm --entrypoint bash -v ~/.cache/huggingface/inject:/w \
  vllm-spark-0731:main-029-proto \
  -c "cd /w && gcc -shared -fPIC -O2 -o dcopy_trace.so.new dcopy_trace.c -ldl"
# 70,952 bytes; nm -D now exports 2 cudaLaunchKernel symbols where the stale build had 1
```

Deployed to both nodes, md5 `50b98bdec8d88bcdbfc017bf7065fc40`, stale copy kept as
`dcopy_trace.so.stale-1789238327` on spark1. Care for any repeat: `docker run` needs
`--entrypoint bash`, or the image's `vllm serve` entrypoint consumes the command and dies
with "Failed to infer device type" rather than running gcc.

Next, and it is a short chain: the container needs `LD_PRELOAD` plus `DCOPY_TRACE_OUT`,
`DCOPY_TRACE_LIMIT` and `DCOPY_TRACE_BLOCK_X=0`. `scripts/05-serve.sh` has no environment
passthrough (it adds only specific `-e` args such as `PYTORCH_CUDA_ALLOC_CONF`), so add an
`EXTRA_ENV_ARGS` knob beside `SERVE_EXTRA_ARGS`, then run one decode burst and read the
dumped host stacks. That names the owner of the 1-byte copy family, measured at 20 to 40 ms
per step against the reference's ~1 ms and worth more than any remaining flag.

### Why the interposer saw nothing: the symbols are version-tagged (2026-09-13)

Ran the rebuilt v4 shim (`LD_PRELOAD` on the head, `DCOPY_TRACE_BLOCK_X=0`,
`DCOPY_TRACE_LIMIT=120`, best config) and drove one 64-token decode burst.

Two results, one good and one decisive:

1. **v4 is safe.** The container booted and reached `health=200`, so the "shim kills the
   worker" failure in HANDOFF section 135 is fixed by the v4 source that was already on
   disk. That blocker is closed.
2. **It intercepted nothing.** The trace file was never created, and that file is opened in
   `init()`, which runs on the first intercepted launch. The constructor printed (1 match for
   `dcopy_trace` in the container log) so the library was genuinely preloaded.

Root cause, from the ELF symbols rather than inference:

```
$ nm -D .../torch/lib/libtorch_cuda.so | grep cudaLaunchKernel
                 U __cudaLaunchKernel@libcudart.so.13
                 U cudaLaunchKernel@libcudart.so.13
                 U cudaLaunchKernelExC@libcudart.so.13
$ ldd libtorch_cuda.so | grep -ci cudart
1
```

`cudart` is linked dynamically, so the reference is not resolved inside libtorch, but the
reference is **version-tagged** (`cudaLaunchKernel@libcudart.so.13`). An `LD_PRELOAD` object
that defines the plain, unversioned `cudaLaunchKernel` does not satisfy a versioned
reference, so the dynamic linker binds straight to libcudart and the shim's definition is
never entered. That is the reason four instruments in a row saw nothing below Python.

The fix is a linker-level symbol version on the definitions, which is the standard way to
interpose a versioned symbol:

```c
__asm__(".symver cudaLaunchKernel, cudaLaunchKernel@libcudart.so.13");
__asm__(".symver cudaLaunchKernelExC, cudaLaunchKernelExC@libcudart.so.13");
```

`__cudaLaunchKernel@libcudart.so.13` is libcudart's internal alias for the same entry point,
so it is worth covering too. Next slice: add those `.symver` directives to
`.scratch/dcopy_trace.c`, rebuild in the image with `--entrypoint bash`, rerun the same
one-burst capture, and read the host stacks. That names the owner of the 1-byte copy family,
which is worth 20 to 40 ms per step against the reference's ~1 ms.

Supporting detail recorded for the next session: the serve script now has both
`SERVE_EXTRA_ARGS` (vLLM flags) and `SERVE_EXTRA_ENV` (space-separated `docker run -e` pairs)
passthroughs, so the shim can be armed without editing the script.

### The interposer is version-correct now, but only PID 1 gets it (2026-09-13)

Progressed the interposer to the point where the symbols are right.

A bare `.symver` does not link: `ld: version node not found for symbol
cudaLaunchKernelExC@libcudart.so.13`. The version node has to be declared, so the build now
uses a version script, `.scratch/libcudart.vers`:

```
libcudart.so.13 { global: cudaLaunchKernel; cudaLaunchKernelExC; local: *; };
gcc -shared -fPIC -O2 -o dcopy_trace.so dcopy_trace.c -ldl -Wl,--version-script=libcudart.vers
```

The result exports exactly what libtorch asks for, which the previous build did not:

```
T cudaLaunchKernel@@libcudart.so.13
T cudaLaunchKernelExC@@libcudart.so.13
```

Deployed (md5 `0722a1d18759deff781063f31138d2c1`, both nodes), relaunched on the best
config, burst driven, server reached `health=200`. **Still zero dumps, no trace file.**

Why, with evidence: the shim's constructor prints one line per process that loads it, and
the container log contains exactly one, `dcopy_trace] loaded pid=1`. PID 1 is the API
server. The engine core and the worker are separate subprocesses, and vLLM spawns them with
a curated environment, so `LD_PRELOAD` never reaches the processes that actually enqueue the
kernels. Every prior attempt shares this defect: the library loads in a process that does no
launching, and the launching processes never load it.

So the remaining fix is an injection point that is process-wide rather than inherited from
the entry process. Two options, in order of preference:

1. `/etc/ld.so.preload` inside the container, containing the `.so` path, bind-mounted in at
   `docker run` time. Needs a small mount passthrough next to `SERVE_EXTRA_ENV` in
   `scripts/05-serve.sh`, since the script currently has no way to add a bind mount.
2. Bake the file into a profiling-only derived image with `docker commit`. Defensible for a
   profiling run, must not become the image under measurement, and must be recorded as a
   deviation if used.

Then rerun the same one-burst capture and read the host stacks. That names the owner of the
1-byte copy family, measured at 20 to 40 ms per step against the reference's ~1 ms, which is
larger than any remaining configuration lever.

Note for the record: the container had exited by the time the follow-up check ran, so the
exit is not attributed here; the evidence above is from the log of that run.

### Five instruments, one blind spot: the copies are inside the graph (2026-09-13)

Added a `SERVE_EXTRA_MOUNTS` passthrough to `scripts/05-serve.sh` beside `SERVE_EXTRA_ENV`
and injected the interposer process-wide through a bind-mounted `/etc/ld.so.preload`
containing `/root/.cache/huggingface/inject/dcopy_trace.so`. Rationale: `LD_PRELOAD`
exported on PID 1 reached only PID 1 (`loaded pid=1`), because vLLM spawns the engine core
and worker with a curated environment, so the processes that enqueue kernels never had it.

Result: server healthy, burst served, and **still exactly one `loaded pid=1` and no trace
file**. So the process-wide injection did not change the outcome, which rules out the
environment as the reason and points at the launch API itself.

The conclusion that fits every observation, including the earlier ones:

**The 1-byte copies are inside the CUDA graph.** vLLM captures the decode step as a cudagraph
and replays it with `cudaGraphLaunch`; replay does not call `cudaLaunchKernel` or
`cudaLaunchKernelExC` once per node, so an interposer on those entry points sees nothing no
matter which process it is loaded into. That is the same reason the torch profiler produced
*no op records for graph-replayed kernels*, which HANDOFF already recorded as one of the four
failures. The five instruments now line up with a single cause rather than five unrelated
defects:

| instrument | why it saw nothing |
|---|---|
| torch profiler | graph-replayed kernels have no per-op launch record |
| Python profile hook | sees views/allocations, not graph node launches |
| nsys export | backtraces not collected, and replay has no host launch to backtrace |
| launch interposer v1-v4 | unversioned symbols (`cudaLaunchKernel@libcudart.so.13`), then only PID 1 |
| launch interposer v5 | symbols correct, process-wide, and still nothing: replay bypasses the launch API |

What follows for the next slice, in order of cost:

1. **Cheapest falsification.** Add `cudaGraphLaunch` (and `cuGraphLaunch`) counting to
   `dcopy_trace.c` with the same version-script treatment. If the count is of order one per
   step while `cudaLaunchKernel` stays at zero, replay is confirmed as the mechanism in one
   run, and it is a few dozen lines.
2. **Then enumerate the graph's nodes**, which is what actually names the copies:
   `cudaGraphGetNodes` plus `cudaGraphKernelNodeGetParams` on the captured graph gives each
   node's kernel func, grid and block, so the 1-byte elementwise family is identified by
   geometry and count without needing a host stack. This can be done from a small C shim
   hooking `cudaGraphInstantiate`/`cudaGraphLaunch`, or from Python via `cuda-python` if that
   binding exposes the graph APIs in this image.
3. **Or sidestep graphs entirely**: run one capture with `--enforce-eager` (the serve script
   has `ENFORCE_EAGER`), where every kernel is launched individually and the profiler or the
   interposer can see it with a host stack. The kernel mix then differs from the shipped
   config, so it attributes the *owner* rather than the exact per-step cost, which is the
   question worth answering first anyway.

The magnitude is unchanged and still worth this: 20 to 40 ms per step of 1-byte device work
against the reference's ~1 ms, on a whole step of about 27 ms at c6 and 67 ms at c1.

### CORRECTION: a port collision invalidated five measurement sets (2026-09-13)

This must be read before the two sections above it. They contain a measurement error that
made the proto image look like it had reached parity when it had not.

**What happened.** The reference arm launched by `spark-launch.sh` runs in a container named
`sparkrun_<hash>_node_0`, not `vllm-ds4-0731`, and `scripts/05-serve.sh` only removes its own
container name. So after the reference baseline at ~01:46 the reference kept port 8000 for
over an hour. Every subsequent launch of our image started, then died at startup:

```
(APIServer pid=1) Traceback (most recent call last):
(APIServer pid=1)     sock.bind(addr)
(APIServer pid=1) OSError: [Errno 98] Address already in use
```

`docker run -d` still succeeds, so `05-serve.sh` printed `started vllm-ds4-0731 running`, and
the health poll and `drive-median.sh` both talk to `127.0.0.1:8000`, i.e. to the *reference*.
Nothing in the harness noticed.

**Invalidated, because each of these was really the reference measured again.** Compare their
numbers with the reference's own 64.0 / 113.1 / 148.1 / 160.1:

| tag | reported | really |
|---|---|---|
| proto-nobreak (the section headed "Win 2") | 64.2 / 115.8 / 144.3 / 160.3 | reference |
| proto-nb-k7 | 64.5 / 114.7 / 144.8 / 159.9 | reference |
| proto-compile | 61.8 / 113.4 / 156.4 / 159.7 | reference |
| proto-cap36 | 62.2 / 112.9 / 151.2 / 160.2 | reference |
| proto-a16 | 63.6 / 117.3 / 141.3 / 160.3 | reference |
| all three shim runs (v5, v6, preload) | no trace | our container exited before launching anything |

So **"Win 2: `VLLM_USE_BREAKABLE_CUDAGRAPH=0` was the whole gap" is withdrawn**, as are the
`COMPILATION_MODE`, capture-36 and A16 sections. They are left in place rather than deleted so
the error is visible, with this section as the correction of record.

**A second, independent reason those runs could not have been valid.** Our image rejects the
combination outright:

```
RuntimeError: DeepseekV4ForCausalLM: piecewise CUDA graphs (cudagraph_mode=FULL_AND_PIECEWISE)
unavailable, model is not torch-compiled and breakable CUDA graph is off.
Set VLLM_USE_BREAKABLE_CUDAGRAPH=1 or cudagraph_mode=NONE/FULL.
```

Non-breakable cudagraphs require either torch.compile or `cudagraph_mode` NONE/FULL. The
reference satisfies this by running `CompilationMode.VLLM_COMPILE`; our image does not, and
setting `COMPILATION_MODE=3` through the new knob did **not** make vLLM treat the model as
compiled (same error), so how the reference achieves it is still open.

**What survives, all of it measured before ~01:46 when our container could still bind:**

| config | c1 | c3 | c5 | c6 |
|---|---|---|---|---|
| reference (anemll, k=7, compile, breakable off) | 64.0 | 113.1 | 148.1 | 160.1 |
| ours: k=5, `--async-scheduling` | 37.6 | 80.6 | 110.8 | 135.0 |
| ours: k=5 as pinned, no async | 37.8 | 77.6 | 105.0 | 130.8 |
| ours: k=7, capture 48, warm | 38.9 | 80.1 | 106.9 | 120.7 |

**Honest standing: we are 16% behind at c6 and 41% at c1, not at parity.** `--async-scheduling`
is the one confirmed win (+3 to +6% at c3/c5/c6, all measured pre-collision).

**One corrected result from after the fix**, with the engine identity verified as
`v0.1.1.dev0+g69db1c26b` and health reached in 150 s:

| config | c1 | c3 | c5 | c6 |
|---|---|---|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0 CUDAGRAPH_MODE=FULL` | 26.1 | 67.4 | 91.8 | 109.5 |

Worse at every level, and noisy (spreads 18 to 98%). So full-step graphs are worse than the
shipped `FULL_AND_PIECEWISE` with breakable cudagraphs on, and that lead closes.

**Harness change required by this error, not optional.** Every measurement must now assert:
port 8000 was free before launch, our container is still `Up` when health returns 200, there
is no `address already in use` in its log, and the engine identity string is ours
(`v0.1.1.dev0+g69db1c26b`) rather than `v0.25.2.dev0+g752a3a504`. The last check alone would
have caught this at the first contaminated run.

### Harness guards so the port collision cannot recur (2026-09-13)

Two changes, both verified live rather than asserted.

**`scripts/05-serve.sh` now refuses to start when the port is already served.** `docker run -d`
succeeds even when the engine cannot bind, so previously the script printed
`started vllm-ds4-0731 running` while the container was already dead, and every client reached
whatever held the port. The script now checks before launching:

```
refusing to start vllm-ds4-0731: port 8000 is already in use:
LISTEN 0      2048         0.0.0.0:8000       0.0.0.0:*
stop the holder first, e.g. docker ps --format '{{.Names}}' and docker rm -f <name>
guard_exit=1
```

Verified by running it while `vllm-ds4-0731` legitimately held the port: it refused, exit 1,
and named the holder. This is the root fix for the contamination described in the correction
section above, because it fails at launch rather than after a measurement has been believed.

**`~/drive-median.sh` now records which engine answered.** Every result log begins with the
port holder, the container name and status, and the engine version string read from that
container's log:

```
== port holder: LISTEN 0 2048 0.0.0.0:8000 0.0.0.0:*
== container: vllm-ds4-0731 Up 9 minutes
== engine: v0.1.1.dev0+g69db1c26b
```

Verified against the live container, whose version resolves to our proto base
`v0.1.1.dev0+g69db1c26b`; the reference would print `v0.25.2.dev0+g752a3a504`. That single
line is what would have caught the whole contaminated series at its first run, so it is now an
artefact of every measurement rather than something inferred afterwards. The workload is
untouched: same chat mode, seed 1234, 512 tokens, same levels and passes. Backup of the
previous driver at `~/drive-median.sh.pre-guard`.

Next: the invalidated A/Bs can now be re-run safely, one at a time, starting with the two that
matter — `COMPILATION_MODE=3` (and the still-open question of how the reference gets the model
torch-compiled where our knob did not) and `disable_custom_all_reduce`, both against the
valid baseline `37.6 / 80.6 / 110.8 / 135.0` and the reference's
`64.0 / 113.1 / 148.1 / 160.1`.

### Why the reference can torch-compile and we cannot, located exactly (2026-09-13)

Following the gate that rejected `VLLM_USE_BREAKABLE_CUDAGRAPH=0` on our image.

The check, in `vllm/v1/worker/gpu/cudagraph_utils.py:548`:

```python
if self.cudagraph_mode.has_piecewise_cudagraphs() and not (
    self.use_breakable_cg or has_compiled_submodule(model)
):
    raise RuntimeError(...)
```

and the predicate it calls:

```python
def has_compiled_submodule(model: nn.Module) -> bool:
    """Whether any submodule is an active @support_torch_compile module."""
    return any(
        isinstance(m, TorchCompileWithNoGuardsWrapper)
        and not getattr(m, "do_not_compile", True)
        for m in model.modules()
    )
```

So piecewise graphs need either breakable cudagraphs, or a model decorated
`@support_torch_compile`. Two facts follow, both verified in the base:

1. **Our stack cannot compile at all while breakable cudagraphs are on.**
   `vllm/config/vllm.py:755`:

   ```python
   enabled = is_breakable_cudagraph_enabled()
   if enabled:
       self.compilation_config.mode = CompilationMode.NONE
   ```

   `configs/env.spark.sh:47` defaults `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, so the mode is forced
   to NONE regardless of what is passed, which is why passing `COMPILATION_MODE=3` changed
   nothing. Turning breakable off then fails the gate above.

2. **The decorator is missing on the DeepSeek V4 CUDA path.** `@support_torch_compile`
   appears in `vllm/models/deepseek_v4/cpu/model.py:452` and nowhere else under
   `deepseek_v4/`; `deepseek_v4_1/` has none at all. Nothing under `deepseek_v4/nvidia/`
   carries it. With no decorated model, no `TorchCompileWithNoGuardsWrapper` is ever
   instantiated, so `has_compiled_submodule()` is False and non-breakable piecewise graphs are
   impossible by construction.

The reference satisfies the same gate, so its fork must decorate its CUDA DeepSeek V4 model
and run breakable cudagraphs off. That is the one structural difference that lets it use
inductor plus full-and-piecewise graphs, and inductor is a plausible owner of a large part of
the 16% c6 and 41% c1 gap. This is the strongest lead remaining and it is now specific: add
the decorator, not a flag.

Risks to handle before believing any gain: our overlays inject code into the model forward
(`patch_layer_profiler`, `patch_region_profiler`, the step profiler), and a compiled forward
may conflict with those; our B12X attention backend and the b12x MOE are custom paths that
inductor must tolerate on SM12x. So the change needs the gates plus an accuracy check on top
of the protocol, not the protocol alone.

Also recorded here since it cost a cycle: the port guard added in the previous section was
initially broken on the path every normal run uses. `_holder="$(... | grep ... | head -2)"`
exits non-zero when the port is free, and under `set -e -o pipefail` that aborts the script, so
launches silently stopped happening. Only the busy path had been tested. Fixed with `|| true`
inside the substitution, and both paths now matter: busy refuses, free proceeds.

### Overlay written for the missing decorator (2026-09-13)

Following the finding in the previous section, a new overlay `patch_nvidia_support_torch_compile`
decorates the NVIDIA DeepSeek V4 model so the model can actually be torch-compiled:

```python
path = vllm / "models/deepseek_v4/nvidia/model.py"
# adds: from vllm.compilation.decorators import support_torch_compile
# and:  @support_torch_compile(dynamic_arg_dims={
#           "input_ids": 0, "positions": -1,
#           "intermediate_tensors": 0, "inputs_embeds": 0})
#       class DeepseekV4Model(nn.Module, EagleModelMixin):
```

It is registered in `apply_main` immediately before the closing message, so it runs in the
main stack like every other overlay.

Why those exact anchors, all verified in the base first: the NVIDIA `DeepseekV4Model` at
`nvidia/model.py:1309` is the same class and bases as the CPU one, and its `forward` takes
`(input_ids, positions, intermediate_tensors, inputs_embeds)` in that order, which is exactly
what the CPU decorator's `dynamic_arg_dims` describes, so the decorator transfers unchanged.
The file does not import `support_torch_compile`, hence the paired import edit.

Verification, `scripts/port_scan.py` against the pristine `~/proto-check`:

```
47  ok     patch_nvidia_support_torch_compile             2 edits
FAIL=1  ok=37  skip=9
grep: 13:from vllm.compilation.decorators import support_torch_compile
      1320:@support_torch_compile(
```

`ok` went 36 to 37 with the new overlay, and the tree holds both edits.

The one `FAIL` is expected and is the ordering rule confirming itself: `port_scan` resets the
tree and runs `apply_main` **without** the `pr-*.diff` patches, and `patch_cutlass_sm12x_guard`
deliberately matches the post-`pr-53055` text, so it misses here and would pass in the build's
patches-then-overlays order. Worth remembering when reading a bare scan: a single
`patch_cutlass_sm12x_guard` failure in a patch-less run is the known, correct behaviour.

Next: test it without a rebuild by `docker cp`-ing the updated `apply_overlays.py` into a
running container, running the overlay against `/opt/vllm/vllm` inside it, then restarting the
container so the writable layer carries it. That is the fast path to a measurement. If it wins,
fold it into the image by rebuilding; if inductor will not compile this model on SM12x with the
B12X attention backend and the profiler overlays injecting into the forward, that is the answer
and it needs recording rather than a rebuild.

### The decorator was necessary but not sufficient (2026-09-13)

Tested `patch_nvidia_support_torch_compile` without paying a rebuild: started the unpatched
image, `docker cp`-ed the updated `apply_overlays.py` in, ran
`--only torch-compile-nvidia` inside the container, committed the result as
`vllm-spark-0731:main-029-proto-ccompile`, synced that to spark2, then launched the
reference's combination (breakable cudagraphs off, `COMPILATION_MODE=3`, `VLLM_USE_AOT_COMPILE=0`).

What happened, in order:

1. The overlay applied inside the container, so the fast path works:
   `ok nvidia model torch.compile import` and `ok nvidia DeepseekV4Model torch.compile decorator`,
   with the container's own tree showing `13: from vllm.compilation.decorators import
   support_torch_compile` and `1320: @support_torch_compile(`.
2. **The gate is now passed.** The `piecewise CUDA graphs ... model is not torch-compiled`
   error is gone, which confirms the decorator is the thing that gate was missing.
3. It fails deeper, the first time the model is traced:

```
RuntimeError: Worker failed with error 'torch.* op returned non-Tensor
  Explanation: torch.* ops that return a non-Tensor cannot be traced into the Dynamo FX graph output
```

The path that reaches it, from the EngineCore traceback:

```
core.py:308 _initialize_kv_caches
  -> model_executor.determine_available_memory()
  -> executor/multiproc_executor.py collective_rpc
  -> RuntimeError: Worker failed with error 'torch.* op returned non-Tensor'
```

So the missing decorator was necessary but not sufficient: with it the model reaches Dynamo,
and Dynamo then refuses because some op's non-Tensor result is traced into the graph output.
`determine_available_memory` is the first traced forward, during memory profiling, which is
why this fires at init rather than on the first request.

Next step is to name that op from the worker-side traceback; the excerpt captured here is only
the EngineCore tail. First candidates are our own overlays that inject into the model forward
(`patch_layer_profiler`, `patch_region_profiler`, the step profiler) and any custom op in the
DSV4 path returning a tuple or None.

Supporting notes for the next session: the derived image
`vllm-spark-0731:main-029-proto-ccompile` exists only as an experiment artefact and is **not**
the image under measurement. `--only torch-compile-nvidia` is now a registered choice. And a bug
in my own test script cost a cycle: step 2 launched the head alone on a 2-node config, so it
blocked on rank 1 forever and never reached health, which looked like a broken base image; the
worker now starts first, as in every working invocation.

### Torch.compile is blocked by our own all-reduce workspace patch (2026-09-13)

With the decorator in place, Dynamo names the offending op instead of just refusing:

```
torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor
  Explanation: torch.* ops that return a non-Tensor cannot be traced into the Dynamo FX graph output

  Developer debug context: example_value type: bool; op: call_function;
    target: <function is_current_stream_capturing at 0xe4996ad774c0>

from user code:
   File ".../models/deepseek_v4/nvidia/model.py", line 1458, in forward
     hidden_states = self.embed_input_ids(input_ids)
   File ".../models/deepseek_v4/nvidia/model.py", line 1425, in embed_input_ids
     return self.embed_tokens(input_ids)
   File ".../model_executor/layers/vocab_parallel_embedding.py", line 555, in forward
     return tensor_model_parallel_all_reduce(output_parallel)
   File ".../utils/sm12x_b12x_kernels.py", line 1424, in wrapper
     return fn(*args, **kwargs)
   File ".../distributed/communication_op.py", line 55, in tensor_model_parallel_all_reduce
     tmp = _workspace(input_)
   File ".../distributed/communication_op.py", line 25, in _workspace
     if torch.cuda.is_current_stream_capturing():
```

`torch.cuda.is_current_stream_capturing()` returns a bool, and that call sits inside the traced
region (reached through the vocab-parallel embedding's all-reduce), so Dynamo rejects it. Both
`communication_op.py` and `sm12x_b12x_kernels.py` are ours:
`communication_op.py` is patched by `patch_tp_allreduce_static_workspace` (the `ar-static-ws`
overlay) and line 1424 of `sm12x_b12x_kernels.py` is our own wrapper.

**So the thing blocking torch.compile on this stack is our all-reduce workspace overlay, not
anything upstream.** That is a narrow, in-repo fix rather than a platform limitation.

This also completes the chain that kept the image uncompiled, both parts ours:

1. `VLLM_USE_BREAKABLE_CUDAGRAPH=1` in `configs/env.spark.sh` forces
   `compilation_config.mode = CompilationMode.NONE`, so nothing compiles.
2. Asking for compile with breakable off gets as far as Dynamo, which then breaks on our
   `is_current_stream_capturing()` probe inside `_workspace()`.

Next fix, and it is small: resolve the capture state outside the traced region. Options in
increasing order of cleanliness — evaluate it once per step in eager code and pass the result
in; guard the query with `torch.compiler.is_compiling()` so it is skipped while tracing; or keep
a plain Python flag that the traced path reads. Then re-test the reference's combination
(breakable off, compile on) and measure.

Worth stating plainly for the record: this is the first lead this session where the blocker is
our own code, reachable in a few lines, rather than a platform gate we cannot select. The 16%
c6 and 41% c1 gap against the reference is still unclosed, and inductor remains the best
plausible owner of it.

### Overlay added for the Dynamo break, and the test is running detached (2026-09-13)

Following the named op, `patch_tp_allreduce_dynamo_safe` treats the guard as a compile-time
constant. It replaces, in `distributed/communication_op.py`:

```python
            if torch.cuda.is_current_stream_capturing():
```

with

```python
            if (
                torch.compiler.is_compiling()
                or torch.cuda.is_current_stream_capturing()
            ):
```

`torch.compiler.is_compiling()` is a Python-level constant that Dynamo folds at trace time, so
while tracing the branch short-circuits before the CUDA query is ever reached and returns None,
which routes the traced step to the existing clone fallback. Outside compilation the expression
evaluates exactly as before, so the CUDA-graph capture path is unchanged. That is what makes it
a one-condition change rather than a restructuring of the workspace helper.

Registered in `apply_main` next to `patch_nvidia_support_torch_compile` (the two travel
together: the decorator gets us into Dynamo, this gets Dynamo past our own probe) and exposed as
`--only allreduce-dynamo-safe`. `apply_overlays.py` compiles clean with both.

Anchoring notes for the next session, both of which cost a cycle:

- The 8-space dispatch entry `        patch_nvidia_support_torch_compile(vllm)` *contains* the
  4-space call as a substring, so a bare 4-space anchor is not unique and needs the following
  line for context.
- `~/goal/test-ccompile.sh` now applies both overlays inside the container before committing.

The run is detached on spark1 as `~/goal/ccompile-run.log` (started 03:48). It applies both
overlays into a running container, commits `vllm-spark-0731:main-029-proto-ccompile`, syncs that
to spark2, launches the reference combination (breakable cudagraphs off, `COMPILATION_MODE=3`,
`VLLM_USE_AOT_COMPILE=0`), and reports either a measurement tagged `proto-ccompiled` or the next
error. Outcome not yet known and deliberately not guessed.

### The next Dynamo break: is_deep_gemm_supported inside the traced forward (2026-09-13)

The Dynamo-safe workspace guard worked. The engine cleared that break and failed one step
further, in `determine_available_memory` again (run log `~/goal/ccompile-run.log`):

```
torch._dynamo.exc.Unsupported: call to a callable object with no traceable __call__
  Explanation: Dynamo could not trace a Python `__call__` method on this object.
  Developer debug context: object=<_FuncPtr object at 0xfac1851d4950>

from user code:
   File ".../models/deepseek_v4/nvidia/model.py", line 1484, in forward
     hidden_states, residual, post_mix, res_mix = layer(
   File ".../models/deepseek_v4/nvidia/model.py", line 1237, in forward
     residual, post_mix, res_mix, x = mhc_pre_broadcast_tilelang(
   File ".../model_executor/kernels/mhc/tilelang.py", line 539, in mhc_pre_broadcast_tilelang
     use_deep_gemm = is_deep_gemm_supported()
   File ".../utils/deep_gemm.py", line 114, in is_deep_gemm_supported
     is_supported_arch = current_platform.support_deep_gemm()
   File ".../platforms/cuda.py", line 714, in support_deep_gemm
     <_FuncPtr object>
```

Dynamo cannot trace the ctypes function pointer reached through `is_deep_gemm_supported()`, and
`mhc_pre_broadcast_tilelang` calls that inside the forward. That gate at that site is what
`pr-53055.diff` introduces (the mHC pre-broadcast fallback to TileLang when DeepGEMM is off), so
this break is a consequence of a backport this repo applies, not of upstream vLLM.

Two warnings before the failure are expected and benign on this rig: `SymmMemCommunicator:
Device capability 12.1 not supported, communicator is not available` and `FlashInfer All Reduce
is disabled because it is not supported for world_size=2` (two nodes, one GPU each).

Next fix, same shape as the last one, because the value is a static platform property rather
than per-call state: compute it once at module scope and let the traced code read a plain bool.

```python
# kernels/mhc/tilelang.py, module level
_USE_DEEP_GEMM = is_deep_gemm_supported()
...
use_deep_gemm = _USE_DEEP_GEMM      # inside the forward
```

Dynamo folds a module-level bool as a constant, so the ctypes call leaves the graph. Then rerun
`~/goal/test-ccompile.sh`, which is repeatable and carries both existing overlays.

Worth stating plainly: every Dynamo break found so far has been ours or a backport's, and each
has been a small local construct rather than a platform gate. The reference runs compiled, and
the distance to that is now measured in these small fixes.

### Break 4: the eager-init fix worked, and now it is a pybind op (2026-09-13)

Run 3 applied all four overlays and verified them in the container (`_EAGER_INIT_DONE = True`
at deep_gemm.py line 1076, the decorator at model.py 1320, `is_compiling` at
communication_op.py 26, `_USE_DEEP_GEMM` at tilelang.py 9/180/353/543), committed the derived
image, synced it to spark2 and relaunched. The engine cleared break 3 and failed on:

```
RuntimeError: Worker failed with error 'Attempted to call function marked as skipped
  Developer debug context: module: vllm.third_party.deep_gemm._C,
    qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1
              .tf32_hc_prenorm_gemm,
    skip reason: cannot determine source file for vllm.third_party.deep_gemm._C
                 (likely a C extension or builtin)
```

So the importlib break is gone and the next one is a **pybind11 C++ op called inside the traced
forward**. `tf32_hc_prenorm_gemm` comes from the compiled DeepGEMM extension, which Dynamo cannot
trace, and it refuses rather than graph-breaking.

The idiomatic vLLM remedy is the pattern already imported at the top of the very file involved:
`tilelang.py` imports `direct_register_custom_op`, which exists precisely so such kernels are
presented to torch.compile as opaque custom ops instead of being traced into. Registering
`tf32_hc_prenorm_gemm` that way is the next bounded step.

**Assessment, stated plainly rather than discovered later.** The first three breaks were
one-to-two-line Python-level fixes (a workspace guard, a platform property, a lazy init) and each
cleared in a single rerun. This fourth one is different in kind: a compiled op inside the traced
region. The DSV4 plus b12x stack has more of those in traced code — the b12x kernels, the
DeepGEMM wrappers, the custom attention and MoE paths — so closing the compile gap means
converting an op surface to compile-compatible registration, one op per cycle, with a ~20 minute
measurement loop each time. That is a multi-op project, not another small fix.

It remains worth doing, and it is still not a platform gate: the reference runs compiled, so the
work is ours and the shape of it is now understood. But it should be budgeted as its own goal
rather than assumed to be the next quick win, and the same 20-minute cycles are also the cost of
attacking the other open item, the 1-byte copy family.

Recommendation for whoever picks this up, in order: (1) register `tf32_hc_prenorm_gemm` as a
custom op and rerun, which confirms the remedy on one op; (2) if that works, enumerate the other
pybind ops reachable from the traced forward rather than discovering them one break at a time;
(3) then decide between finishing the compile path and attacking the copy family.

### Break 4 persists: `torch.compiler.disable` is rejected, so the custom-op route is required (2026-09-13)

Run 5 applied overlay 6 cleanly — `ok mHC tf32 call redirect (3 sites)`, three sites where I had
predicted two — so the pybind op is no longer traced directly. The engine then failed with a new
message:

```
RuntimeError: Worker failed with error 'Skip calling `torch.compiler.disable()`d function
```

So the redirect took effect and this torch version **refuses to call a `torch.compiler.disable`d
function from inside the compiled region**. It raises rather than graph-breaking, which is the
opposite of the escape-hatch behaviour I assumed when choosing the cheap first step. That
shortcut is closed.

**The sanctioned route is vLLM's own pattern, already imported at the top of the file involved:**
`direct_register_custom_op(op_name, op_func, mutates_args, fake_impl)`. For
`tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split)` that means `mutates_args=["out", "sqrsum"]`
(it writes both), a fake impl that returns a consistent meta tensor for tracing, and callers going
through `torch.ops.vllm.tf32_hc_prenorm_gemm(...)`. The same treatment is then needed for every
other pybind op reachable from the traced forward.

Where the compile path stands, plainly:

| break | construct | fix | outcome |
|---|---|---|---|
| gate | not torch-compiled | `@support_torch_compile` on the NVIDIA model | cleared |
| 1 | bool `is_current_stream_capturing()` in the ar-static-ws guard | fold `torch.compiler.is_compiling()` | cleared |
| 2 | ctypes `_FuncPtr` via `is_deep_gemm_supported()` | module-level `_USE_DEEP_GEMM` (4 sites) | cleared |
| 3 | `importlib.import_module` via `_lazy_init()` | run it at import | cleared |
| 4 | pybind11 `tf32_hc_prenorm_gemm` | disable-wrapper + call redirect | **not cleared: torch rejects calling a disabled function from a compiled region** |

Five breaks cleared with small Python-level fixes; the sixth needs per-op custom-op registration,
and it is the first one where a plausible shortcut failed. Each attempt costs roughly twenty
minutes of cluster time.

Recommendation before spending more cycles, so the loop is "register N ops" rather than N separate
twenty-minute discoveries: enumerate in one pass every pybind op reachable from the traced
forward — `git grep` for `third_party.deep_gemm._C`, for the b12x kernel module, and for the other
`vllm._C` bindings the DSV4 forwards call — then register them together and rerun once.

Nothing about the measurement picture changes here: best valid config remains k=5 +
`--async-scheduling` at 37.6 / 80.6 / 110.8 / 135.0 against the reference's
64.0 / 113.1 / 148.1 / 160.1. The compile path is still the live lead, and still not a platform
gate, but it is now measured as a per-op project rather than the next small fix.

### The compile path's op surface, enumerated (2026-09-13)

Rather than discovering the remaining Dynamo blockers one twenty-minute run at a time, I
enumerated what sits in the traced forward. Both findings point the same way, and both are ours.

**DeepGEMM.** `vllm/utils/deep_gemm.py` reaches the extension through `_import_deep_gemm()` and
exposes a dozen-plus wrappers over `vllm.third_party.deep_gemm._C` pybind entry points:
`fp8_gemm_nt`, `fp8_einsum`, `m_grouped_fp8_gemm_nt_contiguous`, `fp8_m_grouped_gemm_nt_masked`,
`fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`, `tf32_hc_prenorm_gemm`,
`get_paged_mqa_logits_metadata`, `transform_sf_into_required_layout` and the packing helpers.
Several are called from inside the model forward, including by our own overlays: the o_proj
overlay calls `fp8_einsum`/`deepgemm_post_process_fp8_weight_block`, and the indexer overlays call
the paged MQA logits wrappers.

**b12x.** Our `patches/files/sm12x_b12x_kernels.py` imports the b12x package *inside functions*
(`from b12x.attention.dsa_indexer import ...` at line 400, `from b12x.gemm.wo_projection import
...` at 963 and 997), and those functions are called from the forward: our o_proj overlay calls
`try_b12x_wo_proj` from `deep_gemm_fp8_o_proj`, and the indexer path calls the dsa_indexer entry
points.

**What that means, plainly.** Every one of those is a Dynamo blocker of the kind we are already
fighting: a pybind call that cannot be traced, plus an in-function import that reaches importlib.
The compile path therefore needs per-op custom-op registration across the DeepGEMM and b12x
surfaces, not one more fix. And there is a design cause worth naming: our overlays were written
for an *uncompiled* forward and place kernel calls directly in it, so they are the reason the
traced region contains so many opaque calls.

**The two live leads, weighed.** The compile path is now measured as a dozen-plus-op project with
a ~20 minute verification loop per attempt, and it is unproven that compile even wins here:
the shipped uncompiled config already reaches 135.0 at c6, and graph breaks around that many
custom ops could easily give back any gain. The copy family is a single, unattributed item at
20 to 40 ms per step against the reference's ~1 ms — larger than the whole c1 step — and needs one
instrument (graph-node enumeration via `cudaGraphInstantiateWithFlags` + `cudaGraphGetNodes`)
rather than a per-op campaign.

Neither is a platform gate and the objective is not blocked; this is a statement about where the
next twenty-minute cycles are best spent. The recommendation, for the record: finish attributing
the copy family first, because it is one instrument away and its magnitude is already established,
then decide whether the compile path is worth the per-op work on top.

Best valid measurement remains k=5 + `--async-scheduling` at 37.6 / 80.6 / 110.8 / 135.0 against
the reference's 64.0 / 113.1 / 148.1 / 160.1.

### The graph-node instrument works; the shimmed run did not reach health (2026-09-13)

`dcopy_trace.c` now interposes `cudaGraphInstantiateWithFlags` (versioned through
`libcudart.vers`, four processes loaded the shim in the run: pids 1, 1114, 1117, 1129, so
`/etc/ld.so.preload` plus forked children do carry it) and dumps each captured graph's kernel
nodes via `cudaGraphGetNodes` + `cudaGraphKernelNodeGetParams`. Build is warning-free and exports
`cudaGraphInstantiateWithFlags@@libcudart.so.13` alongside the other three.

It produced real node data, which no earlier instrument managed:

```
[dcopy_trace] graph #1: 1 nodes
  node 0 func=0xe90e26da3b84 grid=(2,32,1) block=(256,1,1) smem=0
[dcopy_trace] graph #2: 8 nodes
  node 2 func=0xe90e26db6500 grid=(64,1,1) block=(256,1,1) smem=4096
  node 5 func=0xe90e26db6500 grid=(16,1,1) block=(256,1,1) smem=4096
  node 7 func=0xe90e26ec1e00 grid=(132,1,1) block=(256,1,1) smem=0
```

Two caveats, both real and both recorded rather than smoothed over:

1. **The run did not reach health.** `container gone`, `health=000` with the shim preloaded on the
   shipped config, where the same config without the shim reaches health in 150 to 265 seconds.
   So the shim itself needs investigating before its output can be trusted as representative.
2. **The dumps captured only small early graphs** (1 and 8 nodes, grids of 2x32 and 132), not the
   decode step, whose graph has hundreds of nodes. `dump_graph_nodes` stops after 3 graphs, so it
   recorded the first captures and missed the decode graph. The geometry summary in the run
   (`9 × grid=(24576,1,1)`, `7 × grid=(6144,1,1)`) comes from more node lines than the two dumps
   shown, so the file needs reading in full rather than by my `head`.

So the copy family is **still not attributed**, and the two things needed next are small and
specific: (a) find why the shimmed run fails at startup (its own log has the reason), and (b) raise
or remove the graph-dump limit, or dump every graph, so the decode graph is captured. Then join the
dumped grids and blocks against a named kernel list by geometry.

State otherwise unchanged: best valid measurement is k=5 + `--async-scheduling` at
37.6 / 80.6 / 110.8 / 135.0 against the reference's 64.0 / 113.1 / 148.1 / 160.1. The compile path
stays parked with six overlays written and registered, needing per-op custom-op registration across
the DeepGEMM and b12x surfaces.

## 2026-09-13: guarded re-baseline and engine-level diff (supersedes the copy-family lead)

Why this section exists: no measurement in the ledger before 03:15 carries the port-holder guard,
and that guard was added because a still-running reference container silently answered a whole
series. Every pre-03:15 number is withdrawn, the `k=5 37.6 / 80.6 / 110.8 / 135.0` set included.

Both arms guarded, 3 passes at 512 tokens, levels 1 3 5 6, chat, thinking=false, seed 1234,
k=7, capture 48, `--async-scheduling`, so the images are the only free variable:

| arm | c1 | c3 | c5 | c6 |
|---|---|---|---|---|
| `refg` (anemll 0.1.1, `sparkrun_..._node_0`) | 63.1 | 112.1 | 140.9 | 156.0 |
| `protog` (main-029-proto) | 38.4 | 80.8 | 105.2 | 122.3 |

Acceptance is level (ours p0 90.0 p1 76.7 against 88.1 / 73.7 at c6) and tokens per step is level
(4.585 against 4.676), so the whole gap is decode step time: 38.65 ms against 29.4 ms at c6, and
114 ms against 74 ms at c1. The gap is worst at the lowest concurrency, which is a latency, not a
bandwidth, signature.

Engine config diff, read from both containers' own logs:

| | ours | reference |
|---|---|---|
| compilation mode | `NONE` | `VLLM_COMPILE` (3), but skipped: "torch.compile is turned on, but the model ... does not support it" |
| cudagraph | breakable, `Breakable CUDA graph enabled` | plain PIECEWISE, `VLLM_USE_BREAKABLE_CUDAGRAPH=0` |
| attention | `B12X_MLA_SPARSE` (`b12x_sparse.py`, our overlay) | `FLASHINFER_MLA_SPARSE_DSV4`, autotuned `sparse_mla_sm120_decode_dsv4`, 24 cached configs |
| MoE | `B12X_MXFP4_MXFP8` (`B12X_MOE_FORCE_A8=1`) | `B12X_MXFP4` |
| linear | `b12x` | `auto`, selects `DeepGemmFp8BlockScaledMMKernel` |
| DeepGEMM E8M0 | disabled (`VLLM_USE_DEEP_GEMM_E8M0=0`) | enabled, "enabling UE8M0 for DeepGEMM" |
| all-reduce | PYNCCL | PYNCCL |
| prefix caching | on | on |

The FlashInfer autotune line is the sharpest: the reference loads 24 tuned configs for its sparse
MLA decode kernel, ours loads 0. Our image ships the same path
(`models/deepseek_v4/nvidia/flashinfer_sparse.py`) and `_select_dsv4_attn_cls` maps SM12 to
`DeepseekV4FlashInferSM120Attention` when no explicit backend is given, so the forced
`B12X_MLA_SPARSE` in `pin.main-029.env` is the only thing keeping us off it.

Two corrections to earlier claims in this file:

- The reference does not compile. Its config says `VLLM_COMPILE` but vLLM skips compilation for this
  model in both images, so "the reference runs compiled piecewise graphs" was wrong and compilation
  is not the explanation for the gap. The compile path stays parked.
- The graph-node dumps in the previous section came from a run that never reached health, and that
  run's grid histogram matched eager-launch dump lines, not graph nodes. The `24576` and `6144`
  grids are eager launches from `vllm._C_stable_libtorch` (6,291,456 and 1,572,864 elements, i.e.
  1536 and 384 tokens at hidden 4096) taken during a memory-profile pass, not the decode step. The
  copy-family lead is retired until an instrument can see inside a replayed graph.

Harness changes made here: `pin.main-029.env` now reads
`DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-B12X_MLA_SPARSE}"` so the draft backend can be
A/B'd without editing the pin. `~/goal/run-arm.sh <tag> "<env assignments>"` is the guarded arm
runner: it applies the assignments to both the worker and the head, waits for health, saves the
engine-config lines to `~/goal/<tag>-engine.log`, and then meters.

Next: A/B the knobs the diff exposes, one at a time, at the same k=7 / capture 48 / async config and
against `protog` 38.4 / 80.8 / 105.2 / 122.3:
1. stock FlashInfer sparse MLA attention (`FLASHINFER_MLA_SPARSE_DSV4`), target and draft
2. `LINEAR_BACKEND=auto` with `VLLM_USE_DEEP_GEMM_E8M0=1`
3. `B12X_MOE_FORCE_A8=0`

### Arm ledger, 2026-09-13 (all guarded, k=7, capture 48, `--async-scheduling`, 3 passes, 512 tokens, levels 1 3 5 6)

| arm | change against `protog` | c1 | c3 | c5 | c6 | verdict |
|---|---|---|---|---|---|---|
| `refg` | reference image, anemll 0.1.1 | 63.1 | 112.1 | 140.9 | 156.0 | comparator |
| `protog` | none, base | 38.4 | 80.8 | 105.2 | 122.3 | - |
| `attnfi` | `ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` = `FLASHINFER_MLA_SPARSE_DSV4` | 37.8 | 78.5 | 104.6 | 119.8 | wash, rejected |
| `deeplinear` | `LINEAR_BACKEND=auto VLLM_USE_DEEP_GEMM_E8M0=1` | - | - | - | - | crash, see below |
| `dglinear` | `LINEAR_BACKEND=auto` | 10.0 | 27.8 | - | - | broken, see below |
| `cgfull` | `CUDAGRAPH_MODE=FULL` | 38.9 | 76.0 | 87.7 | 109.2 | rejected, see below |

What the three negatives establish:

- `attnfi` at 37.8 / 78.5 / 104.6 / 119.8 is inside the spread of `protog`, so the attention
  implementation is not the gap. Our `B12X_MLA_SPARSE` overlay is not what costs us the step time,
  and the reference's 24 tuned FlashInfer configs are not what earns it its step time.
- `deeplinear` dies at startup: `DeepGEMM` assert `sf.size(-2) == ceil_div(mn, gran_mn)` at
  `csrc/utils/layout.hpp:97`, which is the ue8m0 scale layout. The pin's own comment warned that
  this path was not working, and this is the concrete failure.
- `dglinear` starts but is numerically broken: `accept_rate 0.0 %`, `tokens_per_step 1.002`, 10 tok/s
  at c1, because every draft token is rejected. So the fp8 linear layers cannot move from `b12x` to
  DeepGEMM on this pin without either a DeepGEMM bump or a layout fix, and the measured cost of
  trying is a dead arm.
- `cgfull` is worse at every level and unstable: pass-to-pass spread of 74 / 18 / 18 / 11 %, and
  generations truncate (1381 tokens at c3 where 1536 were asked for). So removing piecewise
  cudagraphing in favour of FULL is not a route.

Why step time is the whole game, stated arithmetically: at c6 `protog` does 4.585 tokens per step at
25.87 steps/s, and `refg` does 4.676 tokens per step at 33.99 steps/s. Holding our own tokens per
step and adopting the reference's step rate would give 4.585 x 33.99 = 155.8 tok/s against their
156.0. Acceptance and per-position acceptance are already level, so nothing needs to be won back on
the speculative side: only 9.25 ms of step time at c6, and about 40 ms at c1.

### `proto5` (k=5) against `protog` (k=7), 2026-09-13

| arm | k | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|---|
| `refg` | 7 | 63.1 | 112.1 | 140.9 | 156.0 | 472.1 |
| `protog` | 7 | 38.4 | 80.8 | 105.2 | 122.3 | 346.7 |
| `proto5` | 5 | 38.9 | 82.4 | 95.4 | 133.0 | 349.7 |

Gates pass in all three (`gate_france` starts " Paris", `gate_9x8` is "72, 9x9"). k=5 buys 8.7 % at
c6 and a hair at c1 and c3, and gives back 9.3 % at c5, where k=7's capture fits 5 x 8 = 40 inside
the captured size list and k=5's 5 x 6 = 30 does not. Sum is level within the spreads. So the
speculative token count is not a lever worth pulling; both are about 26 % behind the reference on
the sum and 15 % behind at c6.

### What the decode profiler showed

`VLLM_PROFILE_DECODE=1 VLLM_PROFILE_CAPTURE=1` (the profiler already in
`patches/files/sm12x_b12x_kernels.py`) prints, for the first steps of a run, per-step region and
per-layer figures. Read from the `prof2` container:

- the DSpark draft's per-step cost in steady state is 3.2 to 5.3 ms (`b12x layers: n=3 sum=11.0ms
  avg=3.67ms`), against a target step of 38.65 ms at c6. With k=7 the draft runs seven of those in
  series, so the draft is a large share of the step and is not covered by the attention arm, which
  changed the draft's backend too and measured a wash.
- one early step showed `b12x step sample: wall=78.1ms gpu=77.3ms` and `sample_tokens: gpu=181.1ms`
  as cold-start figures; the printed window is the first 12 non-dummy steps, so these are not steady
  state and are recorded here only as an observation, not as a finding.
- per-layer figures are one to three layers per print and read 3.2 to 5.5 ms, with the maximum
  always at `L0`.

The profiler cannot be pointed at the reference, which is a different build, so it attributes our
step but cannot attribute the difference. That remains the open problem.

### Step attribution at last, and the whole kernel stack differs, 2026-09-13

Raised the decode profiler's window (`VLLM_PROFILE_DECODE_STEPS=900`) so it prints in steady state
rather than on the first twelve cold steps. Reading the last samples, which are single-sequence
(`tok=8`, i.e. 1 x (7+1)) c1 steps:

| region | gpu time |
|---|---|
| `execute_model` (target forward) | 98 to 118 ms |
| `sample_tokens` (draft + sampler) | 23 to 40 ms |
| `sample` (sampler alone) | 2.5 to 3.9 ms |

So at c1 the target forward is about 85 % of the step; the sampler is noise. The reference's whole
c1 step is 74 ms, so if its split has a similar shape its target is around 63 ms against our 100 ms,
which accounts for the entire 40 ms gap. The draft is not the story and neither is sampling. That
retires the previous turn's guess that the DSpark draft was the leading suspect.

Also from that log, per-layer detail on our side:

- our WO projection runs as a batched `bmm` with a batch of 4 on 42 rows
  (`b12x wo_proj bmm ok tgd=(42, 4, 4096) z=(4, 42, 1024) out=(42, 4096)`), four small GEMMs per
  layer instead of one.
- the B12X MLA scratch is 810 MB, sized for `rows=12288`, the prefill bound
  (`B12X_MLA_SPARSE compressed MLA scratch 810028032 bytes (heads=32 rows=12288 width=704 page=64
  chunks=2)`); decode allocation only, not steady-state cost, but it is why the KV pool is tight.

The two images share almost no kernel library, which is why every A/B above came back a wash: none
of them changes the code that actually dominates.

| library | ours | reference |
|---|---|---|
| b12x | 1.2.6 (source at `3a437ab5`, the 1.3.0 release state) | 0.15.3 |
| flashinfer | 0.7.0 | 0.6.15 (+ cubin 0.6.13, jit_cache) |
| tilelang | 0.1.14 | 0.1.9 |
| humming_kernels | 0.1.13 | 0.1.10 |
| quack_kernels | 0.6.5 | 0.5.0 |
| tokenspeed_mla | 0.2.8 | 0.1.2 |
| NCCL | 2.31.2 | 2.30.7 |

b12x is the one that matters most: it supplies the MoE experts, the compressed MLA and the indexer,
and 0.15.3 and 1.2.6 are different series with different module layouts (`cute`, `distributed`,
`quant` against `_lib`, `comm`, `norm`), so the kernels behind `Using B12xExperts` are not the same
kernels. The pin's own comment already recorded the version difference; what is new here is that the
target forward is where the gap lives, so this is now a measured reason to care rather than a note.

`dspark.py` differs too but not in algorithm: ours is the later upstream file (confidence head and
sequence-parallel paths the reference lacks), 570 lines against 490.

### `stockops`, 2026-09-13: our overlays are load-bearing, and the config space is exhausted

`VLLM_USE_B12X_WO_PROJECTION=0 VLLM_USE_B12X_SPARSE_INDEXER=0` with `MAX_MODEL_LEN=32768` gives
33.1 / 59.9 / 74.7 / 81.2 against `protog` 38.4 / 80.8 / 105.2 / 122.3. Dropping our two per-layer
substitutions costs about 34 % at c6, so they are a large win over the stock paths and not a cost.

The first attempt at this arm died for a different reason worth keeping: with the stock paths the
engine needs 9.48 GiB of KV at 65536 context against 9.32 GiB available, so the overlays also save
memory, and the retry lowered `MAX_MODEL_LEN` to 32768 to fit.

B12X 0.15.3, the reference's, does not export any name our overlays import: no `dsa_indexer`, no
`logits_paged`, no `logits_contiguous`, no `prepare_paged_metadata`, no `uses_paged_schedule`, no
`plan_paged_schedule`. Its `b12x/moe/` holds `fused` and `tuning` where ours holds `_lib` and `comm`.
So the reference's kernel generation is unreachable from our overlays without a port.

Put together, every knob this repo owns is already at its best measured value:

- attention backend is a wash (`attnfi` 37.8 / 78.5 / 104.6 / 119.8)
- the fp8 linear path cannot leave b12x (`deeplinear` crashes, `dglinear` goes numerically dead)
- FULL cudagraph mode loses and is unstable (`cgfull` 38.9 / 76.0 / 87.7 / 109.2)
- the spec token count is level (`proto5` 38.9 / 82.4 / 95.4 / 133.0)
- both remaining overlay knobs are wins that have to stay on (`stockops2` 33.1 / 59.9 / 74.7 / 81.2)

The remaining 26 % on the sum, and the 40 ms of target-forward time at c1, sit in libraries the two
images do not share: b12x (1.2.6 against 0.15.3), flashinfer (0.7.0 against 0.6.15), tilelang (0.1.14
against 0.1.9), and the torch and NCCL builds (NCCL 2.31.2 against 2.30.7). Closing it is a port, or
an extraction of specific kernels from the older b12x, not a configuration change.

### Region attribution, 2026-09-13: the MoE is the dominant region and the collectives are not

`VLLM_PROFILE_DECODE=1 VLLM_PROFILE_CAPTURE=1 VLLM_PROFILE_DECODE_STEPS=12`, reading the region
marks the overlay already installs. Per recorded layer:

| region | n | gpu_sum | gpu_avg | gpu_max |
|---|---|---|---|---|
| ffn (MoE) | 3 | 5.5 ms | 1.84 ms | 1.93 ms |
| attn | 3 | 2.8 ms | 0.94 ms | 1.09 ms |
| wo | 3 | 1.8 ms | 0.60 ms | 0.69 ms |
| mla | 3 | 0.6 ms | 0.20 ms | 0.26 ms |
| allreduce | 7 | 1.4 ms | 0.20 ms | 0.67 ms |

and `b12x layers: n=3 sum=8.5ms avg=2.83ms top: L0=2.99 L1=2.79 L2=2.70`, i.e. per-layer cost is
flat across layers, not concentrated in the first dense ones.

Two conclusions follow, and they are the first region-level facts this investigation has had:

- 2.83 ms per layer over 43 layers is about 122 ms, which is the measured c1 target forward (98 to
  118 ms). So a c1 decode step is 43 sequential layers at roughly 2.8 ms each. The step is a latency
  chain, and the reference's chain is about 35 % shorter.
- the MoE accounts for 1.84 ms of that 2.83 ms, about 65 %. Attention is 0.94 ms, and the TP
  all-reduce is 0.20 ms, which is why lowering the collective cost is not worth pursuing: 43 of them
  are under 9 ms and the measured region total is 1.4 ms for the ones recorded.

Instrument caveats, recorded rather than smoothed: only three layer events and three to seven region
events are recorded per step, so these are per-recorded-unit averages rather than whole-forward sums;
and the window summary's `fwd_gpu/step=137.9ms` exceeds `wall/step=75.3ms` because async scheduling
overlaps the events, so the region averages are the trustworthy part and that total is not.

What this changes: the MoE is 65 % of the layer that dominates the step, and our MoE does not come
from our own overlays, it comes from b12x, whose version is 1.2.6 against the reference's 0.15.3.
Better still, the mxfp4 oracle exposes a whole menu of MoE implementations behind the already
env-overridable `MOE_BACKEND`, none of which has been measured:

    b12x (current), deep_gemm, flashinfer_trtllm, flashinfer_trtllm_afp8, flashinfer_cutlass,
    flashinfer_cutlass_afp8, triton, triton_unfused, humming, marlin, aiter variants

So the dominant region has eleven selectable implementations and we have measured exactly one. That
is the next A/B series, one backend per arm against `protog` 38.4 / 80.8 / 105.2 / 122.3.

### MoE backend sweep, 2026-09-13

The mxfp4 oracle's menu probed one at a time at the standard config (k=7, capture 48, async, guarded):

| `MOE_BACKEND` | outcome |
|---|---|
| `b12x` (current) | 38.4 / 80.8 / 105.2 / 122.3 |
| `flashinfer_trtllm` | dies at worker init: `Mxfp4 MoE backend 'FLASHINFER_TRTLLM_MXFP4_MXFP8' does not support the deployment configuration since kernel does not support current device cuda` |
| `flashinfer_cutlass` | never healthy, reason not captured |
| `humming` | healthy in 190 s, metered below |
| `deep_gemm`, `marlin`, `triton` | not reached; the probe stops at the first healthy backend |

`MOE_BACKEND=humming` meters 42.0 / 79.3 / 109.1 / 122.8 (spreads 3.8 / 1.5 / 7.5 / 4.6 %) against
`protog` 38.4 / 80.8 / 105.2 / 122.3: +9.4 % at c1, -1.9 % at c3, +3.7 % at c5, +0.4 % at c6, +1.9 %
on the sum. Gates pass.

That result is more informative than the numbers look. The MoE is 65 % of the per-layer *latency* at
c1, and swapping its implementation for a different vendor's moves c1 by 9 % but c6 by 0.4 %. So at
c6 the step is not MoE-latency-bound, and the c1 region split does not describe the c6 case. Sweeping
the rest of the menu (deep_gemm, marlin, triton) is therefore unlikely to find the 28 % that c6
needs, and two of the four probed backends do not run on this device at all.

`humming` is kept as the better c1 setting (its c1 spread of 3.8 % is also tighter than `b12x`'s
9.4 %), but it is a 2 % change on the sum, not the change this goal needs.

### c6 split, and the sparse-MLA autotune asymmetry checked and closed, 2026-09-13

With the profiler window in the c6 region the sample lines read: `execute_model` gpu 97 to 108 ms,
`sample_tokens` gpu 22.7 to 37.3 ms, `sample` gpu 2.5 to 3.8 ms, against a c6 step interval of
38.65 ms. So about three engine steps of GPU work are in flight at once at c6, and the level that the
completion criterion judges is throughput-bound on total GPU work rather than on per-layer latency.
That is why the c1 region split did not predict the c6 result: at c1 the MoE is 65 % of the *layer
latency*, at c6 it is a much smaller share of the *layer work*.

`VLLM_PROFILE_DECODE_STEPS` did not take effect: the window still reports `steps=12`, so the limit
is not reaching the worker even though `VLLM_PROFILE_DECODE` and `VLLM_PROFILE_CAPTURE` do. Recorded
as an instrument limitation rather than worked around.

One asymmetry looked promising and is now closed. The reference runs a FlashInfer SM120 sparse MLA
decode autotune and caches 24 configs, then hits that cache at runtime:

    [flashinfer_sparse_mla_warmup.py:124] Autotuning FlashInfer SM120 sparse MLA DSv4 decode with cache: ...
    [Autotuner]: Loaded 24 configs ...
    [Autotuner]: Config cache hit for sparse_mla_sm120_decode_dsv4 (runner=SparseMlaDecodeV3Runner)

Our log carries that autotune **zero** times, so our sparse MLA decode runs at defaults. The gate is
in `flashinfer_sparse_mla_warmup.py:225`: the autotune runs only when the attention backend's
`get_name()` is one of the DeepSeek-V4 sparse MLA backends, and `b12x_sparse.py` reports
`B12X_MLA_SPARSE`, which is not in that set. So with the default backend we skip it.

It still does not explain the gap, because the arm that passes that gate measured a wash: `attnfi`,
which sets `ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` to `FLASHINFER_MLA_SPARSE_DSV4` and so
does run the autotune, gave 37.8 / 78.5 / 104.6 / 119.8 against 38.4 / 80.8 / 105.2 / 122.3. Tuned
FlashInfer sparse MLA lands where our untuned b12x MLA already is. So attention is out on both the
implementation and the tuning axis.

### Correction to the previous section: that "c6 split" was c1 data, 2026-09-13

The section above labels the `execute_model` gpu 97 to 108 ms against `sample_tokens` gpu 22.7 to
37.3 ms split as the c6 level. It is not. Those samples carry `tok=8`, which is one sequence times
(k+1), so they are c1 steps, and they agree with the c1 numbers measured earlier rather than adding
anything at c6.

The reason the window never moved: `VLLM_PROFILE_CAPTURE=1` routes the step wrapper to
`_b12x_capture_step`, which prints its own bounded window and ignores
`VLLM_PROFILE_DECODE_STEPS`. That is also why the window summary reports `steps=12` in both the 900
and 1250 runs. So the step limit silently does nothing whenever capture mode is on.

The inference I drew from it, that c6 is throughput-bound with about three engine steps in flight,
does not hold either: a single engine step whose target forward takes 100 ms of GPU time cannot be
completing every 38.65 ms, so that arithmetic was comparing a c1 forward against a c6 interval.

What stands: the c1 split (target ~85 %, draft 22 to 40 ms, sampler 2.5 to 3.9 ms), the c1 region
table, the per-layer 2.83 ms, and the fact that humming moves c1 by 9 % while moving c6 by 0.4 %. The
c6 split is not measured. The re-measurement runs with `VLLM_PROFILE_CAPTURE` off so the step limit
applies, which gives the step-level target against draft split at c6; it gives up the region detail,
because region marks inside a replayed graph are only recorded when capture mode armed them.

### The c6 split, measured properly this time, 2026-09-13

After fixing the plumbing (`VLLM_PROFILE_DECODE_STEPS` has to go through `SERVE_EXTRA_ENV` to reach
the container as `docker run -e`; assigning it in the launching shell does not reach the worker, which
is why every earlier profiling run silently used the default 12 steps and profiled c1), the window is
1250 steps and the rows immediately before the summary are c6 steps:

| call | host wall | gpu |
|---|---|---|
| `execute_model` (target forward) | 2.7 ms | 99.1 to 104.5 ms |
| `sample` (sampler) | 0.6 ms | 2.5 ms |
| `sample_tokens` (draft + sampler) | 117 to 123 ms | 21.2 to 21.9 ms |

So at c6 the target is about 82 % of the per-forward GPU time and the draft about 18 %, which is the
same shape as c1 (roughly 100 ms against 25 ms). The draft is not the c6 bottleneck, and the claim
that the c1 split does not describe c6 was itself wrong: the shape is the same at both levels, only
the absolute scale differs. Two hypotheses are retired by this: that the draft carries c6, and that
c6 is explained by something the c1 profile cannot see.

The host and device numbers invert, which is what made this confusing to read. `execute_model`
returns to the host in 2.7 ms while its GPU work is about 100 ms, and `sample_tokens` blocks the host
for about 119 ms while its own GPU work is 21 ms. So the wall column is not per-step latency and must
not be compared against the meter's step interval.

Where this leaves the search: the target dominates at both c1 and c6, and the target's two largest
regions have both been swapped and both measured a wash, the MoE across vendors and the attention
across implementations and with its autotune active. The remaining difference is inside those kernels,
that is, in the library generation, not in which implementation is selected.

### The fp8 linear axis is closed, and it exposes a real bug, 2026-09-13

`LINEAR_BACKEND=flashinfer_cutedsl` starts and then falls back, from its own log:

    WARNING --linear-backend=flashinfer_cutedsl has no kernel for this linear layer type; using automatic selection for that layer type.
    INFO    Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod
    WARNING --linear-backend=flashinfer_cutedsl requested FlashInfer mm_bf16 backend 'cute-dsl', but it is unavailable on the current hardware or environment; using automatic selection for unquantized linear layers.

So on this image the automatic selection lands on DeepGEMM, and DeepGEMM's fp8 block-scaled MM is wrong here. Both gates fail on both
`LINEAR_BACKEND=flashinfer_cutedsl` and `LINEAR_BACKEND=auto`:

    == gate_france '<?>carecarecarecarecare'
    == gate_9x8    '<?>carecarecarecarecare'

with `accept_rate 0.0 %` and `tokens_per_step 1.002`, so this breaks the target model's own output and
not only the speculative decode.

Two conclusions:

- `LINEAR_BACKEND=b12x` is not a preference, it is the only working option on this pin, and
  `Fp8LinearMethod`'s DeepGEMM kernel, which is what the reference uses, is broken in our image. That
  is a correctness bug in the fp8 linear path, worth its own tracker entry, and it is the concrete
  form of the ue8m0 lead that the earlier `deeplinear` crash pointed at.
- It is not obviously a performance opportunity either. The broken arm's c1 did 511 steps in 53.2 s,
  which is 104 ms per step, against our healthy 114 ms per step at batch 8. So even a working DeepGEMM
  linear is unlikely to be the missing 25 %.

The harness did its job: the gate check fails closed and would have caught this without any numbers
being read.

### Closing the axes

Every axis this repo can move has now been measured, each with the guarded protocol:

| axis | measure | verdict |
|---|---|---|
| attention implementation | `ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4`, target and draft | wash |
| attention tuning | the arm that passes the sparse-MLA autotune gate | wash |
| MoE implementation | `MOE_BACKEND` probed over the oracle menu, humming healthy | +9.4 % c1, +0.4 % c6 |
| fp8 linear | `auto`, `flashinfer_cutedsl` | numerically broken |
| cudagraph mode | `CUDAGRAPH_MODE=FULL` | worse and unstable |
| spec token count | k=5 against k=7 | level on the sum |
| our own overlays | WO projection and sparse indexer off | large loss, they must stay |
| collective cost | allreduce region 0.20 ms avg | negligible, not a lever |

And the two levels agree on the shape: the target forward is 85 % of a c1 step and 82 % of a c6 step,
so speculative decode is not where the gap is, at either level. What remains is per-op performance
inside third-party kernel libraries, where our generation differs from the reference's (b12x 1.2.6
against 0.15.3, flashinfer 0.7.0 against 0.6.15, tilelang 0.1.14 against 0.1.9), and the two regions
that could be swapped individually have both been measured and are washes. Reaching the older
generation means porting our overlays onto an API that does not export the names they import, and
that port would target exactly the two regions just measured as washes.

### Option 1 first step: the reference's b12x generation is not reachable, 2026-09-13

Built `docker/Dockerfile.kernels-ref-b12x` as a derivative image, `FROM
vllm-spark-0731:main-029-proto` with the reference's `b12x-0.15.3` copied over ours. b12x is the
right library to start with because it ships as a pure-Python wheel, so it cannot break on torch or
CUDA ABI the way flashinfer would.

First serve attempt failed for a mundane reason worth recording: the derivative image existed only on
spark1, so spark2's worker could not start (`Unable to find image ... pull access denied`) and the
head then blocked forever waiting for rank 1. `scripts/02-copy-main.sh` (`docker save | ssh spark2
docker load`) is the repo's distribution path and the image is 28.9 GB.

The API question does not need the copy, and the answer is no:

| what our code imports from b12x | against 1.2.6 | against 0.15.3 |
|---|---|---|
| `attention.compressed_sparse_mla` (Caps, plan, bind, run) | ok | `ModuleNotFoundError` |
| `attention.dsa_indexer.plan_paged_schedule` | ok | `ModuleNotFoundError` |
| `gemm.wo_projection` (Caps, plan, bind, run) | ok | module present, all four names missing |

Six failures, no module usable. Checked with `check_b12x_api.py`, which is worth keeping as a gate
for any future pin move.

The functionality is not gone, it moved and was renamed. 0.15.3 exposes
`attention.mla.compressed_mla_decode_forward`, `attention.mla.sparse_mla_decode_forward`,
`MLASparseDecodeMetadata`, `clear_mla_caches`, and a `b12x.attention.indexer` package of
`IndexerPaged*` bindings. So a port is a re-expression of our three overlay call sites against a
class and metadata API, not an impossibility. It is still one more surface than the overlays: our
vLLM's own b12x integration would have to move too, namely the mxfp4 MoE oracle and the
`b12x_sparse` attention backend, both written against 1.2.6.

### Instrument note: the region table is the target's

The two docstrings in `apply_overlays.py` disagree, so this needed settling, because it decides
whether the region table is evidence at all. The gate is `_b12x_region_active`, which returns True
only while `_PROFILING_STEP[0]` is set, and that flag is set by the `execute_model` wrapper alone. The
draft runs inside `sample_tokens`, where the flag is clear. Capture mode arms the marks separately and
only for `tokens == 1`. So the layer and region events accumulate during the target forward, and the
region table describes the target. The earlier reading stands: per-layer 2.83 ms, ffn/MoE 1.84 ms at
65 %, attn 0.94, wo 0.60, mla 0.20, allreduce 0.20.

Caveat to keep: `b12x layers` printed `n=3`, not 43, so only three layer events are recorded per
print. The per-layer average remains valid, and 43 x 2.83 ms does land on the measured c1 forward
(98 to 118 ms), but the sample is smaller than it should be and the why is not established.

### Option 1 revised: side-by-side install works, 2026-09-13

Wholesale replacement of b12x is not reachable, so the cheaper form of option 1 is to make the
reference generation available under its own name and move one call site at a time. That works:

    build: docker/Dockerfile.kernels-ref-b12x-side -> vllm-spark-0731:main-029-proto-b12xref
    install: the reference b12x-0.15.3 copied to dist-packages/b12x_ref, 56 files internal
             absolute imports rewritten from b12x. to b12x_ref., 1.2.6 left untouched as b12x

check_b12x_side.py against that image:

    ok   b12x kept working (/usr/local/lib/python3.12/dist-packages/b12x/__init__.py)
    ok   b12x version (1.2.6)
    ok   b12x.attention.compressed_sparse_mla kept working
    ok   b12x.attention.dsa_indexer kept working
    ok   b12x.gemm.wo_projection kept working
    ok   b12x_ref importable (/usr/local/lib/python3.12/dist-packages/b12x_ref/__init__.py)
    ok   b12x_ref.attention.mla.compressed_mla_decode_forward
    ok   b12x_ref.attention.mla.sparse_mla_decode_forward
    ok   b12x_ref.attention.mla.MLASparseDecodeMetadata
    ok   b12x_ref.gemm.wo_projection
    FAIL b12x_ref.attention.indexer.IndexerPagedTiledLogitsKernelBinding
    ok   b12x_ref.moe.fused

So both generations coexist and the entry points a port needs are present. The one failure is an
export path, not missing functionality: that class is defined in an `attention/indexer` submodule and
is not re-exported from the package `__init__`, so the import has to name the submodule.

Next step, and it is deliberately not a rewrite: benchmark `b12x_ref.gemm.wo_projection` against our
`b12x.gemm.wo_projection` at the shapes this model uses, in this image, on the GPU. The region table
puts WO at 0.60 ms of a 2.83 ms target layer, so it is the largest single kernel our overlays own
that has a direct counterpart in the older generation. If the older one is not faster, the WO port
has no prize and the same test answers for the MLA and the indexer before any of them is rewritten.

## 2026-09-13 PR re-check, second pass

Every tracked PR read again with `gh`. No push, commit or comment was made, for the reasons in the
two sections below.

### Ours

| PR | state | action |
|---|---|---|
| vllm [#53425](https://github.com/vllm-project/vllm/pull/53425) | OPEN, MERGEABLE, `pre-run-check=FAILURE`, review required | none owed: force-pushed 2026-09-12T10:13Z by maci0 to `bb5b8cf67f`, the rebased tip the first pass prepared, so the push recorded there as pending is done and mergify cleared the conflict label |
| vllm [#53680](https://github.com/vllm-project/vllm/pull/53680), [#53522](https://github.com/vllm-project/vllm/pull/53522), [#53271](https://github.com/vllm-project/vllm/pull/53271), [#46716](https://github.com/vllm-project/vllm/pull/46716) | OPEN, blocked, review required | none available |
| llama.cpp [#28003](https://github.com/ggml-org/llama.cpp/pull/28003) | OPEN, draft | none, see below |
| vllm #52357, #53521, #53898; llama.cpp #28002 | CLOSED | none |

All five open vLLM PRs fail the same gate, and it is not a failing test. The check-run annotation
reads:

    To reduce unnecessary pre-commit runs, each PR must have the 'verified', 'ready', or
    'ready-run-all-tests' label, or the author must have at least 4 merged PRs (found 0).
    DO NOT request for the label to be added if you are an AI agent.

So the gate needs a maintainer label and the project forbids an agent asking for one. Every other
check on all five is SUCCESS or SKIPPED. No unresolved review thread exists on any of them, and no
reviewer question is unanswered.

llama.cpp #28003 stays a draft with an unfilled template. `CONTRIBUTING.md` prohibits using "AI to
write your posts for you (bug reports, feature requests, pull request descriptions, GitHub
discussions, responding to humans)", lists AI-written PR descriptions under prohibited usage that
closes the PR immediately, and makes undisclosed AI usage a possible permanent ban. The template's
required `AI usage disclosure` line is a statement only the author can make, so an agent must fill
neither. The bot's "multiple open PRs" finding is moot: #28002 was closed 2026-08-30 as a duplicate
of #18538. The local guard (`&& nsamples_dst == 1`) stays unpublished and unbuilt, which is correct
for an RDNA3 change on a GB10 box.

### Tracked upstream, not ours

| PR | state | change since the first pass |
|---|---|---|
| #53574 | merged 2026-08-31 | the overlay and `pr-53574.diff` retirement in that pass still stands as the only local follow-up |
| #47988 | OPEN, blocked | head still `e1dbe81c`; our `pr-47988.diff` remains source-hunks-only against it, and still stale |
| #53055 | OPEN, blocked | mergify reported a conflict 2026-09-13T01:21Z and the PR is MERGEABLE again, so the author rebased |
| #52941 | **CLOSED 2026-09-13** | author closed it as superseded by #50178 (MHC TileLang warmup moved to a VllmJitKernel); our `dsv4_warmup_ext.py` is still the local equivalent |
| #50645 | OPEN, DIRTY | unchanged, still superseded by #53055 |
| #52499, #52708, #41834 | OPEN | unchanged |
| DeepGEMM #419 | OPEN, clean | lucifer1004's SMEM note answered in `54d5a3e`; nothing pending. #417, #403, #337 unchanged |
| eugr #319 | closed | #349, #352, #348 still open with no reply |

Nothing in this table needs us. #47988 is the only local follow-up, and the patch refresh needs a
build to verify, so it stays open.

## 2026-09-14: #53574 retired, #47988 refreshed

Requested follow-up from the re-check above. The llama.cpp PRs (#28003, #28002) leave this file's
scope today: that work is tracked in another project.

### #53574: diff and overlay retired

`699e180df4` is an ancestor of the pin and the merged fix is in `build_c128a_topk_metadata`: on
family 120 it returns `global_decode_buffer[:num_decode_tokens]`, the full-width slice, so
`_build_c128a_metadata`'s `.view(num_decode_tokens, 1, -1)` is contiguous. The C4A path is
contiguous too (`torch.empty_like` in `models/deepseek_v4/common/ops/cache_utils.py:559`). Both
consumer `.contiguous()` calls the overlay added were no-ops.

Removed: `patches/upstream/pr-53574.diff`; `patch_flashinfer_eidx_contig` with its call in
`apply_main`, its `--only` branch and its choice slot; the `flashinfer-eidx-contig` row in
`patches/README.md` and `patches/upstream/README.md`; its entry in `scripts/verify-docs.py`; and the
knowledge rows in `knowledge/08-upstream.md` and `knowledge/03-kernels-attention.md`.

Gate, two fresh worktrees at `69db1c26b4` on spark1, each with its arm's `pr-*.diff` applied and then
`apply_overlays.py --stack main`:

| arm | `pr-*.diff` applied | `ok` | `skip` |
|---|---|---|---|
| before | 5, including `pr-53574.diff` | 94 | 20 |
| after | 4 | 93 | 20 |

The two logs differ by exactly one line, the removed `ok flashinfer eidx contiguous`, so no other
overlay depended on it. Both arms also exit rc=1 at the same pre-existing point:
`patch_mhc_tf32_uncaptured` raises "no tf32 call site with its local import found" because the pin's
`kernels/mhc/tilelang.py` no longer has that call shape, so the run stops there and overlays after it
are unexercised in both arms. Unrelated to this change, left as-is.

### #47988: refreshed against head `e1dbe81c`, one hunk kept deliberately

`gh pr diff 47988` no longer contains the Triton gate hunk. Main gained the unconditional E8M0→fp32
upcast in `e77daef89e` (#56214, 2026-09-11), five commits after our pin, so the PR's own base now
postdates it and the hunk fell out of the PR diff. The pin still needs it, so the refreshed patch
keeps it; re-fetching from `gh pr diff` without re-adding it would bring back
`KeyError: 'float8_e8m0fnu'`. The note is in `patches/upstream/README.md`. The CUTLASS hunks are
unchanged in logic; the head only dropped two comments.

Not carried: the head also rewrites `_upcast_e8m0_to_fp32` to
`scale.view(torch.float8_e8m0fnu).to(torch.float32)`. Measured in `main-029-proto`
(torch 2.14.0a0+git2b3ec34), it differs from the pin's bit-shift helper at exponent bytes 0 and 255
(value 0 → 0.0 against 2^-127; 255 → inf against NaN), and neither fix needs it, so the pin keeps its
helper.

Verified against `69db1c26b4`: `patch -p1 --forward --dry-run` and `git apply --check` both rc=0 on a
tree carrying the pin's `cutlass.py` and `fp8_utils.py`; `pytest tests/` 28 passed, 13 skipped;
`scripts/verify-docs.py` reports `[OK] overlay names`.

### PR re-check, third pass

State unchanged from the second pass, read again the same day: #53425 head `bb5b8cf67f`, MERGEABLE,
review required; #53680, #53522, #53271, #46716 unchanged and all stopped by the same label gate;
#47988 head still `e1dbe81c`; #53055 force-pushed `f5c6098936` by its author at 2026-09-13T14:13Z
and mergify cleared the conflict label, still no CI. No new review thread and no unanswered question
on any of ours, and nothing to action: the gate needs a maintainer label.

### The mhc-tf32 pair blocked the image build

This is why the last image predates the overlay port. `patch_mhc_tf32_uncaptured`'s call-site needle
is `from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm`, a blank line, then the call.
`pr-53055.diff` folds that import into a guarded `is_deep_gemm_supported, tf32_hc_prenorm_gemm` line,
so by the time the overlays run (the build applies `pr-*.diff` first) the needle is gone and the
overlay raises. `apply_main` aborts there, which cuts off every overlay after it.

The overlay inserts its wrapper before the raise, so the tree is not corrupt; the abort is what
matters. Run alone, `patch_mhc_tf32_call_redirect` then rewrites three of the four call sites to a
wrapper the plain stack never inserts, and leaves `mhc_pre_tilelang`'s single-line call on the pybind
op. Both are parked from `apply_main` and stay reachable by `--only`; the serving image never carried
either, so the rebuild reproduces the shipped configuration rather than changing it.

`scripts/port_scan.py --with-upstream-patches`, which runs the build's patches-then-overlays order and
isolates every callable so one failure does not hide the rest, reports `FAIL=0 applied=44 no-op=12
total=56` on a fresh pin. Before the park it reported `FAIL=1` at `patch_mhc_tf32_uncaptured`.

### Image rebuilt and promoted

Phase 2 from the pin's phase-1 layer (`31aff322427f`, now tagged `main-029-proto-phase1`) with the
current patches tree:

    docker build --build-arg BASE_IMAGE=vllm-spark-0731:main-029-proto-phase1 \
      -t vllm-spark-0731:main-029-proto-0914 -f docker/Dockerfile.main-overlays <ctx>

rc=0. The build applied all four `pr-*.diff`, and its own `assert_image --stack main` printed
`image OK (main): b12x importable, moe/linear b12x, fp8_ds_mla + nvfp4_ds_mla, 584B DSV4 page, mHC
TileLang guard, SM12x kernel guards, DSpark dispatch`. `main-029-proto` now points at the new build
(`3da654fb3e44`); the previous image is kept as `main-029-proto-0912` (`77841395cfcd`).

Six files differ from that previous image, and every one is ported overlay work that had never
reached an image:

| file | change | overlay |
|---|---|---|
| `nvidia/flashinfer_sparse.py` | the two consumer `.contiguous()` calls are gone | #53574 retirement |
| `kernels/linear/scaled_mm/cutlass.py` | two comments dropped | pr-47988 refresh |
| `kernels/mhc/tilelang.py` | `use_deep_gemm = is_deep_gemm_supported()` becomes the module-level `_USE_DEEP_GEMM` | `patch_mhc_deep_gemm_static` |
| `distributed/communication_op.py` | the workspace guard gains `torch.compiler.is_compiling()` | `patch_tp_allreduce_dynamo_safe` |
| `models/deepseek_v4/nvidia/model.py` | `@support_torch_compile(...)` on `DeepseekV4Model` | `patch_nvidia_support_torch_compile` |
| `utils/deep_gemm.py` | `_lazy_init()` at import | `patch_deep_gemm_eager_init` |

The middle three are Dynamo edits that are value-identical when not compiling. The last is not
compile-gated: it resolves DeepGEMM at import rather than first use. The two-node gate has **not**
been rerun on this image, so the promote is on the strength of the content diff above, not a serve.

## 2026-09-14: base moved to v0.29.1rc0

`v0.29.1rc0` is a lightweight tag on `7ee8a6dd0138` (2026-09-12), **75 commits past** the previous
pin (`69db1c26b4`) and 0 behind, so it is a clean forward move. There is no GitHub release for it.
The pin is updated in `configs/pin.main-029.env` (`VLLM_REF`, and `IMAGE` to
`vllm-spark-0731:main-029-1rc0`).

None of our PRs is merged in the range. `e77daef89e` (#56214, DeepSeek-V4.1-Flash support) is, and
that is the commit which landed the unconditional E8M0 upcast in `w8a8_triton_block_scaled_mm`.

### Two backports needed work

| patch | what broke | fix |
|---|---|---|
| `pr-53425.diff` | the `indexer.py` hunk failed: the base added a comment above `DeepseekV4IndexerBackend.get_supported_kernel_block_sizes`'s `return [256]` and moved the method to line 246. The overlay cannot cover for it either, because `patch_dsv4_sm12x_block_size` returns early once `dsv4_supported_kernel_block_sizes` is in the tree (the patch put it there), so the indexer would have stayed at `[256]` against sparse MLA's `[64]` and `select_common_block_size` could not split the manager block | regenerated from the new base; the other three files are unchanged in content |
| `pr-47988.diff` | the Triton hunk no longer applies at all: its comment context is the pre-#56214 text | trimmed to the CUTLASS hunks; `patch_triton_e8m0_sm12x` now skips on that base |

`pr-53055.diff` and `pr-53522.diff` apply unchanged. All four apply with rc=0 and no rejects on a
pristine `7ee8a6dd` worktree, and `scripts/port_scan.py --with-upstream-patches` on that base reports
`FAIL=0 applied=44 no-op=12 total=56`, the same shape as on the proto base.

### References moved with the base

`docker/Dockerfile.kernels-ref-b12x{,-side}` `FROM`, `harness/launch-refbase.sh`,
`harness/test-ccompile.sh`, and the `HANDOVER.md` base note. The b12x head-to-head in HANDOVER has
not been run on either base, so nothing measured is invalidated, but its image must be rebuilt from
the new base first.

`scripts/03-apply-main-overlays-029.sh` takes an optional second argument, the base image, so a
phase-1 tag can be kept and the overlays applied from it:

    scripts/02-build-main-029.sh vllm-spark-0731:main-029-1rc0-phase1
    scripts/03-apply-main-overlays-029.sh vllm-spark-0731:main-029-1rc0 \
      vllm-spark-0731:main-029-1rc0-phase1

Keeping the phase-1 image is what made the 2026-09-14 overlay-only rebuild possible without a vLLM
compile. Before this, `03` always used TAG as its own base and overwrote it, so the phase-1 image
survived only as a dangling layer.

### The phase-1 build is blocked on memory, not on the code

`02-build-main-029.sh` to `main-029-1rc0-phase1` has not completed. Both nodes are running the
llama.cpp workload (spark1 `llama-server`, spark2 `ggml-rpc-server`, about 78 GiB each), which leaves
roughly 41 GiB available per node. A 16-job vLLM compile does not fit in that: earlyoom logged
0.78 % and 1.47 % of 124 GiB free and sent SIGTERM to the largest processes at 10:30:45 and
11:09:29. It killed the llama-server once, and `systemd --user` once, which took the tmux server
hosting the build down with it. The build log ends with `Canceled: context canceled` in every case,
which is the client being signaled, not a compile error.

The pin, the backports and the overlay scan above are complete and do not depend on the build.
`MAX_JOBS` and `MAX_JOBS_TORCH` are now overridable in `configs/pin.main-029.env`, and the build runs
under a guard (`run-p1-guarded.sh`, kept in `.scratch/`) that stops it if `MemAvailable` falls below
a floor, so earlyoom never has to choose between the compiler and the llama.cpp workload.

The first guarded attempt at `MAX_JOBS=4` tripped that guard at 11:42:17 with 14951 MiB available
and exited 143. earlyoom logged nothing at that moment, so the floor sat above its 10 % limit and the
workload survived the abort, which is the intent. By 11:44 the llama.cpp workload was no longer
running and 120 GiB was available; no earlyoom SIGTERM and no kernel OOM entry accounts for its exit,
so who stopped it is not established. The build was relaunched at 12:41:32 at `MAX_JOBS=8` with the
floor at 20 GiB.

Rebuilding the apt layer also picked up a newer `cuda-compat-13-3` from the CUDA repo, so no layer
before it could be reused and this is a full phase-1 build, torch from source included. The trip log
is `~/.cache/0731-prcheck/p1d-guardtrip.log` on spark1.

**Resolved.** Phase 1 exited 0, and phase 2 passed its own gate:

    image OK (main): b12x importable, moe/linear b12x, fp8_ds_mla + nvfp4_ds_mla, 584B DSV4 page,
                     mHC TileLang guard, SM12x kernel guards, DSpark dispatch

| tag | id | note |
|---|---|---|
| `main-029-1rc0` | `89a2d018b564` | the new image, 28.9 GB |
| `main-029-1rc0-phase1` | `164184d8e58e` | phase 1, kept so the next overlay-only rebuild needs no vLLM compile |
| `main-029-proto` | `3da654fb3e44` | previous base, kept |
| `main-029-proto-0912` | `77841395cfcd` | the one before that, kept |

Checked inside the image: `dsv4_supported_kernel_block_sizes` is wired in both `sparse_mla.py` and
the indexer (the #53425 re-anchor), the retired consumer `.contiguous()` is gone, the parked
`mhc-tf32` wrapper is absent, and the #47988 CUTLASS N%128 fall-through is present.

One cosmetic mismatch: `vllm.__version__` in the image reads `0.29.1rc1.dev0+g7ee8a6dd0.d20260914`.
The build fetches a single commit with `--depth 1`, so the clone carries no tags and vcs-versioning
guesses; `VLLM_REF` in the pin is the `v0.29.1rc0` tag commit `7ee8a6dd0138` and that is the
authoritative provenance.

## 2026-09-14: PR re-check, fourth pass

All five of ours were **rebased and force-pushed by the account** at 03:16Z (11:16 +08) while the
image build was running. New heads, all rebased onto current main:

| PR | head | state | gate |
|---|---|---|---|
| [#53425](https://github.com/vllm-project/vllm/pull/53425) | `757eed816b` | OPEN, mergeable | `pre-run-check` only |
| [#53680](https://github.com/vllm-project/vllm/pull/53680) | `fbe9eca60d` | OPEN, mergeable | `pre-run-check` only |
| [#53522](https://github.com/vllm-project/vllm/pull/53522) | `4fdfd8dbdc` | OPEN, mergeable | `pre-run-check` only |
| [#53271](https://github.com/vllm-project/vllm/pull/53271) | `f8e1fad3da` | OPEN, mergeable | `pre-run-check` only |
| [#46716](https://github.com/vllm-project/vllm/pull/46716) | `ddc0860168` | OPEN, mergeable | `pre-run-check` only |

`pre-run-check=failure` is still the label gate, and DCO, `Meta Internal-Only Changes Check` and
`Summary` all pass; the rest is SKIPPED. No new comment and no new review thread on any of them, so
nothing is waiting on a reply. #53425's indexer re-anchor survived the rebase unchanged (the lazy
`dsv4_supported_kernel_block_sizes` import is still there, now at line 251), and its context matches
the local `pr-53425.diff` regenerated for `7ee8a6dd` in this same file.

#53055 was rebased again by its author (`d5dfca40b7`, 10:05Z) and is still MERGEABLE with the same
gate. Everything else is unchanged: #47988 `e1dbe81ccc`, #50645 and #41834 CONFLICTING, #52499,
#52708; DeepGEMM #419/#417/#403/#337, eugr #349/#352/#348 and our issues #53607/#48661 have had no
activity.

### #53680: the one review request, fixed

ivanusto's 13:20Z review on #53680 asked for the `cmake/external_projects/deepgemm.cmake` comment to
be aligned with the 09-08 retraction: it still claimed silent output corruption on stock, which was
the DSpark recipe observation we agreed to keep out. Adopted their proposed wording (a6b593d is the
last nv_dev commit with an SM12x pure-fp8 1d1d path; 8b1392b routes pure-fp8 weights to the
fp8xfp4 dispatcher; interim until DeepGEMM#419) and dropped the same over-claim from the commit
message, since that outlives the PR too. Force-pushed `fbe9eca60d` to `939824ef6f`, a comment-only
content change; DCO and the internal checks pass and `pre-run-check` is still the label gate.
Replied on the PR, including that #56255's nv_dev rebase is the natural point to re-check whether
#419 is still load-bearing.

## 2026-09-15: PR re-check, fifth pass

Nothing needs an action. Our five are unchanged since the 09-14 rebase and all sit at `blocked`,
which is mergeable and waiting on the maintainer label, with no new comment and no new review
thread: #53425 `757eed816b`, #53680 `939824ef6f`, #53522 `4fdfd8dbdc`, #53271 `f8e1fad3da`,
#46716 `ddc0860168`.

Two tracked items moved:

- **#52708 closed unmerged** (2026-09-14T16:09Z). "[Build] Add SM121 (DGX Spark / GB10) to published
  build targets" was closed by Harry-Chen "due to inability to reproduce", after wtdcode's 15:58Z
  finding that upstream publishes 12.0 and 12.0f, "both of which are guaranteed to be compatible
  with sm_121", so there is no motivation for a separate SM121 image. Retire the upstream-tracking
  item, but **not** the local patch: #52708 was carrying the `CUDA_SUPPORTED_ARCHS` fix, it closed
  unmerged, and `v0.29.1rc0`'s CMakeLists still omits 12.1 in its `CUDA >= 13.0` branch. The
  evidence that `VLLM_PRESERVE_SM12X_TARGET=1` is load-bearing in the image built on 09-14: the
  build log prints `Enabled selected SM103 and SM12x targets for CUDA 13 vLLM build`, the image's
  `/opt/vllm/CMakeLists.txt` line 126 reads `"7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1"`
  where upstream's v0.29.1rc0 has the same list without `10.3` and `12.1`, and
  `cuobjdump --list-elf vllm/_C_stable_libtorch.abi3.so` shows `sm_121a`. Drop the patch and vLLM's
  own extension runs family-12.0 code on SM121.

  What has weakened is the justification, not the mechanism: upstream's position is compatibility,
  not parity, and this repo holds no measurement of sm_121a against 12.0 or 12.0f for this
  workload. So it is an unmeasured preference. Note the scope when deciding: the patch touches only
  vLLM's `CMakeLists.txt`; torch (`TORCH_CUDA_ARCH_LIST=12.1a`), NCCL (`-gencode arch=compute_121`)
  and FlashInfer (`FLASHINFER_CUDA_ARCH_LIST=12.1a`) keep 12.1a on their own, so dropping just this
  one leaves a mixed build and the question is really native-sm_121-everywhere or nowhere. Testing
  it costs a phase-1 rebuild per arm, since the arch is baked at compile time.

  **Measured 2026-09-15 with `instruments/arch_codegen_probe.sh`**, which compiles four vLLM-shaped
  kernels (streaming copy, RMSNorm, per-group fp8 quantize, tiled GEMM) at `sm_120`, `sm_120f` and
  `sm_121a` and fingerprints the disassembly. All three emit **byte-identical instruction streams**
  (same count, same mnemonics, same encodings) and run within noise of each other; the interleaved
  repeats put the per-arch spread on the 0.1 ms RMSNorm at ~15 %, the same size as the differences
  between arches, so there is no signal. The `sm_120` binary also runs on the GB10 at all, which is
  upstream's compatibility claim confirmed directly. Supporting this, vLLM's own `csrc/` holds no
  SM121-only feature guard (the only arch test is `__CUDA_ARCH__ >= 1200 && < 1300`, a family test),
  and its CMake already prefers the family target `12.0f` on CUDA >= 13, using `12.0a;12.1a` only as
  the CUDA < 13.0 fallback. On this evidence the flag buys no codegen at all, so it should be
  described as a deliberate step off the supported path rather than a measured win, and dropping it
  is a defensible simplification of the recipe.
- #53055 is `dirty` again (mergify, 2026-09-14T22:18Z) after its latest rebase; the author needs to
  rebase a third time. Not ours.

Unchanged: #47988 `e1dbe81ccc`, #50645 and #41834 CONFLICTING, #52499 `blocked`. DeepGEMM
#419/#417/#403/#337, eugr #349/#352/#348 and our issues #53607/#48661 have had no activity.

**Decided by the owner 2026-09-14: keep the AI disclosure.** The amended commit keeps its
`Co-authored-by: Kimi Code CLI` trailer, and #53522's description keeps its `Made with [Cursor]`
line. Both are AI disclosures, vLLM's own `AGENTS.md` requires one for AI-assisted work and warns
that a breach can ban, and the platform requirement wins over the house rule against AI credit
lines in git-visible text. Do not strip these.

Only those two carry one: #53425, #53271 and #46716 are clean, and #53680's description mentions no
tool (`openai` there is the `vllm/vllm-openai` image name). Open question for #53680: its
description has no AI statement, so its only disclosure is the commit trailer, while vLLM's policy
asks for the statement in the description. A one-line addition would close that.

## 2026-09-15: patch set trimmed

Acting on the census above. Two things left the build, both verified inert before removal:

**The sm_121 target patch is gone.** `docker/patch_vllm_preserve_sm12x_target.py` is deleted, its
`COPY` and its `python3` invocation are out of `docker/Dockerfile.main`, `VLLM_PRESERVE_SM12X_TARGET`
is out of the ENV block, out of all four pins (`pin.main-029`, `pin.main`, `pin.main-dg`,
`pin.main-dg-1m`) and out of `scripts/05-serve.sh`. This rests on the measurement in the 09-15 entry
above: byte-identical SASS at `sm_120`/`sm_120f`/`sm_121a`, the `sm_120` binary running on the GB10,
and no SM121-only feature guard anywhere in vLLM's `csrc/`. vLLM's own kernels now build for
upstream's `12.0f` family target; `TORCH_CUDA_ARCH_LIST=12.1a` still governs torch, and NCCL
(`-gencode arch=compute_121`) and FlashInfer (`FLASHINFER_CUDA_ARCH_LIST`) keep their own settings.
Because that is a change to what the image contains, it still needs a rebuild plus the serve gate.

**The DeepGEMM port file is gone.** `patches/upstream/deepgemm-fp8-1d1d-port.diff` is deleted and its
`case` branch and the whole `COPY`/`RUN` step are out of `docker/Dockerfile.main`. The last phase-1
build logged `skip deepgemm-fp8-1d1d-port.diff (kernel present on a6b593d)`, all four pins carry
`DEEPGEMM_COMMIT=a6b593d`, and `sm100_fp8_gemm_1d1d.cuh` is in that commit, so the port could never
apply. A future DeepGEMM source patch needs its own step put back.

Two historical donors also went: `patches/upstream/0001-pr-52018-b12x-moe-v0.27.1.diff` and
`b12x-moe-52018-vllm-only.diff`, both never consumed by any build path and already called dead
weight by the 2026-09-13 audit. Kept deliberately, because they are the provenance of live
mechanisms rather than dead weight: `0003-nvfp4-ds-mla-v0.27.1.patch` (origin of
`patch_nvfp4_ds_mla`), `0002-pr-50645-mhc-tilelang.diff` (origin of `patch_mhc`),
`kv-offload-bounds-check.patch` (ours, #53271, referenced from the nvfp4 field notes), and the
test-only `pr47988.diff` / `pr54631.diff`.

Not dropped, against the census's fifth item: `pr-47988.diff` and `pr-53055.diff` stay. Their
CUTLASS halves are unexercised while `--linear-backend b12x` is forced, but they are the
upstream-bound content of open PRs this repo is cited as reproducing, and `pr-53055.diff` is
coupled to `patch_cutlass_sm12x_guard`, whose needle matches the post-patch text of that class.

**Rebuilt 2026-09-15: `main-029-1rc0` is now `e732cb776888`.** The ENV-line change invalidated every
later layer, so this was a full phase-1 build; run at `MAX_JOBS=2` beside the llama.cpp workload, it
completed anyway, which is the useful result for a rig that cannot be freed. Phase 1 exited 0
(`acda2b0a40a4`, tagged `main-029-1rc0-phase1`) and phase 2 passed its own
`assert_image --stack main` gate.

The trim is visible in the artifact, which is what the rebuild was for:

| check | pre-trim image (`89a2d018b564`) | new image (`e732cb776888`) |
|---|---|---|
| `CMakeLists.txt` `CUDA >= 13.0` arch list | `...;10.0;10.3;11.0;12.0;12.1` | `...;10.0;11.0;12.0` (upstream's text) |
| `cuobjdump --list-elf vllm/_C_stable_libtorch.abi3.so` | `sm_121a sm_80 sm_89 sm_90` | `sm_120 sm_80 sm_89 sm_90` |
| `/opt/spark-0731/patches/upstream/` | carried `deepgemm-fp8-1d1d-port.diff` | does not |

Kept, no image deleted: `main-029-1rc0-pretrim` (`89a2d018b564`) and `main-029-1rc0-phase1-pre`
(`164184d8e58e`, the phase-1 layer the old image was overlaid from). The vLLM version is unchanged
at `0.29.1rc1.dev0+g7ee8a6dd0`, the same `7ee8a6dd` pin. The two-node serve gate has **not** been
run on the new image.

## 2026-09-15: PR re-check, sixth pass

Four of ours are `blocked` (mergeable, label gate) and unchanged since the 09-14 rebase: #53425
`757eed816b`, #53522 `4fdfd8dbdc`, #53271 `f8e1fad3da`, #46716 `ddc0860168`. **#53680 is `dirty`**,
which is the one thing here that needs a decision.

### #53680: the fix was accepted, and upstream moved under it

ivanusto accepted the comment and commit-message correction (06:43Z: "the new comment and commit
message look right to me"). Then, at 08:47Z, mergify flagged a conflict, and at 10:21Z he explained
why. Upstream **#56876 merged at 07:57Z** and moved the DeepGEMM pin to the vLLM fork
(`vllm-project/DeepGEMM`) at `ad1f1726` (his earlier comment said `9a86ae2b`, which was the head he
tested; the only commit between them is `ad1f1726a`, SM90 paged MQA guards). #56876's merge commit
`8263ea12bd8f` touches exactly `cmake/external_projects/deepgemm.cmake`,
`tools/build_deepgemm_C.py`, `tools/install_deepgemm.sh` and `vllm/models/kimi_k3/nvidia/model.py`.

A plain re-pin of `a6b593d` on current main is therefore **not viable**, and he named the two
collisions: `deepgemm.cmake` now fetches `third-party/cutlass` and `third-party/deep_jit` and adds
`third-party/deep_jit/include` (absent at `a6b593d`, whose `third-party/` holds `cutlass` and `fmt`),
and `kimi_k3/nvidia/model.py` now passes `activation_alpha=`/`activation_beta=`, which `a6b593d`'s
`csrc/apis/mega.hpp` rejects (it asserts `activation == "swiglu"`; the `situ` path is fork-only).
Reverting those would undo part of #56876.

His kernel-level A/B on 2x GB10 (posted on #56876) compares stock `8b1392b` with that fork, pure-fp8
`fp8_gemm_nt`, block 128: **ue8m0 scales are exact in both** (rel err 0 to 7e-7 across M x N x K from
1x128x256 to 4096x13824x16384), while **float32 scales return NaN in 11 to 39 % of the output in
both**. So the fork does not resolve the pure-fp8 path #53680 and DeepGEMM#419 are about, and he
states plainly that he did not test `a6b593d` and is not claiming it fixes the NaN.

That is consistent with our own evidence, and the reconciliation is worth keeping: our image forces
`VLLM_USE_DEEP_GEMM_E8M0=0` (`docker/Dockerfile.main`), so DeepGEMM runs on **float32** scales, which
is the column that is NaN on `8b1392b` and on the fork. Our measurement is end-to-end and matches
that column (`LINEAR_BACKEND=deep_gemm` gives `' Septy Septy...'` at 4.4 tok/s on the 8b-era pin and
coherent France at ~25.8 tok/s with `a6b593d`), not the ue8m0 column, which our image does not run on
DeepGEMM because the SM12x scale-layout assertion blocks E8M0 there. Flag for verification on the
rig: `scripts/05-serve.sh` passes `VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}"`, so at
serve time it defaults to 1 and overrides the image's 0, which is not obviously intended.

His suggestion for the PR: since the fork is now vLLM's own, move the SM12x pure-fp8 fix there as a
fork PR and have #53680 bump the pin to it. One constraint he sets: `sm100_fp8_gemm_1d1d.cuh` from
`a6b593d` cannot be carried as is, because `tcgen05` is rejected for `sm_120a`/`sm_121a` at assembly
time; a port would need the shape of `sm120_fp8_fp4_gemm_1d1d.cuh`.

Safety note he passed on: `tests/kernels/quantization/test_block_fp8.py -k deep_gemm` uses float32
scales and on sm_121 triggered `NVRM: Xid 43` on each node, after which RDMA memory registration
failed until reboot. Do not run it on this pair.

### Local consequences

None today, because our vLLM pin is `7ee8a6dd`, which predates #56876. But if the vLLM pin ever
moves past it, our `DEEPGEMM_COMMIT=a6b593d` collides with the new submodule and include path and
with the Kimi K3 call, and the image build breaks. That is the thing to remember before the next
base move. The pin itself stays: the float32-scale column above is exactly why we hold `a6b593d`.

#53055 is not ours, but its latest review turn matters to the tracker: ivanusto found that the
author's commit "replaced the whole of `tilelang.py`" rather than the guarded dispatch, after
accepting the logic in comment form. Still `blocked`. #50645 is `dirty` and superseded; #47988,
#52499 `blocked`. DeepGEMM #419/#417/#403, eugr #349/#352/#348 and our issues #53607/#48661 have had
no activity.

## 2026-09-15: the SM12x fp8 NaN is root-caused in DeepGEMM

Acting on ivanusto's suggestion, the fork (`vllm-project/DeepGEMM` at `ad1f1726`) was built and run
on the GB10 pair, in the pinned image, one shape per process. It reproduces: `fp8_gemm_nt`,
block 128, float32 SF, `disable_ue8m0_cast=True` gives `max_abs_err` of `inf` or ~1e30, and the
NaN fraction is allocator-dependent (1/128 = 0.781 % at 1x128x256, matching upstream's 0.78 %; 0.81 %
to 70.8 % at 7x512x4096 for the same seed). The packed-ue8m0 path is **exact**, relative error 0
against a dequantised reference, at every shape tried. It does not track the AB-swap: forcing the
non-swapped path (`m=32`, and `accumulate`) still NaNs.

### Mechanism

`csrc/apis/layout.hpp:38` and `:42` keep a float32 SF in the SM90 float layout when
`disable_ue8m0_cast` is set. Their `or disable_ue8m0_cast` disjunct can only ever fire on arch 10 or
12, so on arch 12 that call skips the pack branch at `:46`. The SM120 kernel reads packed ue8m0
only: one e8m0 byte per `gran_k` of K, four bytes per scale word
(`deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh:121-124` sizes the SF smem and TMA
transfer in `int32_t`, `common/sm120_utils.cuh:280` returns one `uint32_t`, `mma/sm120.cuh:43`
extracts one byte). So the raw fp32 bit patterns are read as exponents: a byte of `0xFF` is `2^128`,
which is the NaN, and the observed `3.86e30` comes from a byte of `0x65`. Worse for SFB, the
descriptor is sized from `n` rather than `n / gran_mn`, so the TMA reads past the tensor. Arch 10
refuses this input (`csrc/apis/gemm.hpp:134` requires `torch::kInt`, else `DG_HOST_UNREACHABLE`);
arch 12 returns early at `gemm.hpp:113-118` and never reaches that check.

That out-of-bounds TMA read is the likely mechanism behind ivanusto's `NVRM: Xid 43`, which he saw
only on the float32-scale path.

### Fix, verified, unpushed

Branch `fix/sm120-reject-unpacked-sf`, commit `81d2d8a` off `ad1f1726`, in
`spark1:~/.cache/0731-prcheck/dg-repro/DeepGEMM`: a host-side guard as the first statement of
`sm120::fp8_fp4_gemm_nt`, before the SF transform, refusing a float SF with `disable_ue8m0_cast`
set, plus a test in `tests/test_fp8_fp4.py` asserting both directions. The other three SM120 1d1d
flows were audited and already safe (the einsum arms pass `disable_ue8m0_cast=false` literally; the
m-grouped arms require `kInt`; the k-grouped transform packs unconditionally).

Verified on the GB10: the rejecting call now compiles and launches nothing (a fresh JIT cache
directory stays empty, where before it took 6 entries), the packed-ue8m0 path is bit-identical to
the pre-fix numbers at both shapes with relative error 0 and no NaN, the new test passes, and no Xid
was logged.

### This reframes both PRs

It is a guard, not the kernel restore ivanusto's suggestion describes: the fork already has the
pure-fp8 path (`kIsFP4` / `kBIsFP4` / `kAIsFP4` all default false).

### Upstream landed the same guard before we sent it

`f9d0e2361` ("fix(sm120): correct ported GEMM and MQA execution paths", #9) merged
2026-09-16T04:36Z, about two and a half hours before we were ready to push, into the same file and
with the same protection: `DG_HOST_ASSERT(sfa.scalar_type() == torch::kInt and sfb.scalar_type() ==
torch::kInt)` after the SF transform in `sm120::fp8_fp4_gemm_nt`. That commit also reworks the
recipe and swap resolution in that function and adds five `tests/test_sm120_*.py` files including
`test_sm120_fp8_fp4.py`. So the local fix branch `fix/sm120-reject-unpacked-sf` (`81d2d8a`) is
redundant, was discarded rather than pushed, and the fork PR the suggestion called for is not
needed.

Residual nit, offered only as an optional comment rather than a PR: that assert sits after
`transform_sf_pair_into_required_layout`, so the layout kernels launch before it raises. Checking
the input dtypes first would refuse the call without GPU work.

### #53680 closed 2026-09-16T05:28Z

Closed with the reasoning posted: the fork has the pure-fp8 path, its ue8m0 column is exact (his
table and our measurements agree), the float32 column is now refused upstream instead of
mis-computed, and re-pinning `a6b593d` would collide with #56876. The comment also records the
residual scope: no float-SF path was added for arch 12, because the SM120 block-scaled MMA reads
ue8m0 exponents and there is no supported way to feed it fp32 scales; a caller that wants float SF
leaves `disable_ue8m0_cast=False` and lets the transform pack.

### The one open question left

Whether our image should keep pinning `a6b593d` at all. Its rationale was the float32-scale garbage,
which upstream now refuses, and the column we serve is ue8m0 (`05-serve.sh` defaults
`VLLM_USE_DEEP_GEMM_E8M0` to 1 over the image's `=0`), where the fork is exact. Deciding it needs a
measurement rather than more reasoning: point `DEEPGEMM_REF`/`DEEPGEMM_COMMIT` at the fork's
`ad1f1726`, rebuild, and compare gates and numbers against the `a6b593d` image on the same protocol.
That is a phase-1 rebuild plus a two-node serve. If it holds, dropping the pin also removes the
#56876 collision noted above.

## 2026-09-16: DeepGEMM is converging upstream

lucifer1004 replied on our issue #417 (08:56Z) that in **deepseek-ai/DeepGEMM#447** the SM12x
dispatch is dtype-driven end to end: `fp8_gemm_nt` enters a unified fp8/fp4 dispatcher but the
operand dtypes pick the instantiation, so fp8 x fp8 compiles the 1D1D kernel with `kIsFP4=false`,
SF granularity, swizzle modes and MMA kind all following the fp8 path. There is no fp8-as-fp4
reinterpretation, so the silent-corruption mode this issue reported is gone by construction, and the
combinations that have no SM120 implementation hit `DG_HOST_UNREACHABLE` behind rejection tests.

**#447 is open, not merged**: 100 files, +22556/-7006, and it targets `nv_dev` rather than main. It
integrates main's APIs and DeepJIT runtime and brings native SM120 kernels for BF16/FP8/FP4 and
mixed dense GEMM, M-grouped contiguous and masked, BF16/FP8 K-grouped, FP8 einsum, and DSv4.1
MXFP8/MXFP4 sparse and paged sparse MQA. This matters more than any single fix here: our image
freezes nv_dev at `a6b593d` precisely because of the fp8 regression, so #447 landing would let us
drop that freeze and the local ports that stand in for it.

Replied on #417 with what we measured on `ad1f1726` (ue8m0 exact, relative error 0; a float32 SF with
`disable_ue8m0_cast` is read as packed ue8m0, giving NaN and ~1e30 values with an SFB descriptor that
reads past the tensor, now refused by #9's int-SF assert), offered to run #447's SM12x path on the
GB10 pair, and asked whether #417 and #419 should now be closed rather than left open.

Two more items from the same pass: the fork merged `f9d0e2361` (#9, the SM120 execution paths,
including the guard above) at 04:36Z, and `40d109bfa` (#11, "Remove Mega-Gate finite-score device
trap") touches only `sm100_mega_gate.cuh` and its impl, so it does not affect GB10. Everything else
is unchanged: our four open PRs are still only waiting on the label gate, #53055 is being iterated by
its author, and #50645 and #41834 remain conflicting.

The pin question now has three arms rather than two: `a6b593d` (what we run), the vLLM fork at
`ad1f1726` (what vLLM main pins), and nv_dev with #447. #447 is the cheapest to test first, because
it needs no image rebuild: build the branch and point the repro scripts in `.scratch/dg-repro/` at
it, which is the same harness that root-caused the NaN.

## 2026-09-16: #447 tested on the GB10, and it clears our workarounds

Ran #447 at its head `70a84f8adf2d` (`refs/pull/447/head`), built in the pinned image for sm_121 and
exercised on the pair. It builds; two additions to the recipe: `git config --global --add
safe.directory` on the mount (the container is root, the tree is not) or `bdist_wheel` aborts in
`install_egg_info`, and `libdw-dev` (DeepJIT's `exception.hpp` dlopens `libdw.so.1`). `setup.py`
compiles only the host TU, so `TORCH_CUDA_ARCH_LIST` does not reach the artifact and arch is a
runtime property.

| check | result |
|---|---|
| pure-fp8 block 128, packed ue8m0, 1x128x256 and 7x512x4096 | max relative error **0** against the dequantised reference, no NaN; the JIT'd template has all three fp4 flags false |
| float32 SF with `disable_ue8m0_cast=True` | clean rejection at `csrc/apis/gemm.hpp:122`, with zero new JIT cache entries, so nothing launches |
| `tests/test_sm120_native_rejections.py` | 14 passed |
| `tests/test_sm120_k_grouped_fp4_unsupported_rejection` | 1 passed, rejected host-side before compilation |
| `tests/test_sm120_gemm_regressions.py::test_sm120_gemm_disable_ue8m0_cast` | 128 of 128 passed |

That confirms the maintainer's claim on our hardware: no fp8-as-fp4 reinterpretation, the bad input
is refused rather than mis-computed, and the rejection cases are covered by tests.

**The einsum note in this file does not reproduce on #447.** The DSv4 o_proj call shape
(`fp8_einsum("bhr,hdr->bhd")`, h=8, r=1024, d=4096) is accepted at T=10 and T=256 with packed ue8m0
scales at `recipe=(1,1,128)`, diff about 7.1e-4 with no NaN, and `(1,1,128)` is exactly what vLLM's
`compute_fp8_einsum_recipe` picks for `cap.major > 9`. The `layout.hpp:114` rejection seen at
`recipe=(1,128,128)` with pre-packed per-row scales is correct by design, not a defect. So the
`VLLM_USE_DEEP_GEMM_E8M0=0` workaround our image carries looks obsolete on that base.

One more corroboration, from the branch itself: `csrc/runtime/jit.hpp:31-32` forces `arch = "120f"`
whenever `arch_major == 12 and arch_minor != 0`. So upstream compiles SM12x kernels for the
**family** target, not `sm_121f`, which is the same conclusion our arch probe reached by measurement
when the sm_121 target patch was dropped.

Consequence: #447 is the arm that would let us drop the `a6b593d` freeze, the E8M0 workaround and the
local DeepGEMM ports together. Testing that end to end needs an image built against it, and there is
one mechanical obstacle: `docker/Dockerfile.main` fetches `DEEPGEMM_COMMIT` as a bare SHA, which does
not resolve a fork's PR head, so an end-to-end run needs either `DEEPGEMM_REPO` pointed at the
contributor's fork or a fetch of `refs/pull/447/head`.

Nothing was touched on the nodes and no Xid was logged; the artifacts are under
`spark1:~/.cache/0731-prcheck/pr447/`.

## 2026-09-17: base moved to proto-v0.2.0

`proto-v0.2.0` is an annotated tag ("vllm-proto 0.2.0", 2026-09-16T08:32Z) on `f37c550bf635`, 203
commits ahead of the previous pin and 0 behind. Like `proto-v0.1.0` it is **not a serving release**:
it marks the vllm-proto crate publication. Pin is `VLLM_REF=f37c550bf635` with
`IMAGE=vllm-spark-0731:main-029-proto2`. Two commits in the range touch us, #50178 (mHC TileLang
warmup into the JIT registry) and #56876 (the DeepGEMM pin move); none of our open PRs merged.

### The DeepGEMM pin had to go, and that is the point

`vllm/models/kimi_k3/nvidia/model.py:537` passes `activation_alpha=`, an API only the fork's
DeepGEMM has, and `cmake/external_projects/deepgemm.cmake` adds
`${deepgemm_SOURCE_DIR}/third-party/deep_jit/include`, which `a6b593d` does not have. So `a6b593d`
cannot build this base. Removed: the `DEEPGEMM_REPO`/`DEEPGEMM_COMMIT` ARGs and comment, the
`DEEPGEMM_SRC_DIR` ENV, the `/opt/DeepGEMM` clone RUN and the build-time export in
`docker/Dockerfile.main`; the two variables in all four main pins; the `--build-arg DEEPGEMM_COMMIT`
lines in both build scripts; the `sha/deepgemm` requirement in `docker/assert_main_image.py` and the
matching expectations in `scripts/verify-docs.py`. vLLM's FetchContent now selects the fork at
`ad1f1726` with the `cutlass` and `deep_jit` submodules. This is the collision ivanusto predicted,
and the arm our own #447 measurements called safe: ue8m0 exact, bad input refused.

`VLLM_USE_DEEP_GEMM_E8M0=0` is gone from the image ENV, and the pin now sets it to 1 explicitly.
The reason it needs saying: `configs/env.spark.sh:37` forces `:-0` and `05-serve.sh` sources that
after the pin, so `05-serve.sh`'s own `:-1` default never fired for `main-029`. That is the footgun
recorded on 2026-09-16, fixed at the source rather than by relaxing the shared default, so the older
stacks keep the 0 they were validated with.

### Overlays

Three re-anchors, all from the file's new text rather than guesswork: `patch_nvfp4_ds_mla`'s SWA site
onto the attribute-based form in the new `vllm/v1/attention/backends/mla/sparse_swa.py` (deliberately
not restoring the old `_dsv4_page_alignment(dtype) if packed else 512` shape, which would have moved
v41-on-SM100 from 512 to 576), `patch_kernel_warmup_ext` onto the `cudagraph_capture_sizes` line
(after #50178 deleted the mHC warmup it keyed on), and `patch_region_profiler`'s indexer signature.
Fresh worktree scan: **`FAIL=0 applied=44 no-op=12`**, against a baseline of `FAIL=3`.

A silent under-apply in our own stack was found and fixed on the way. `patch_mhc_deep_gemm_static`
had been matching 2 of 3 sites on the old base too: the third reads
`use_deep_gemm = is_deep_gemm_supported() or not use_tilelang_fallback`, upstream's merged form of
`pr-53055`, so one reachable mHC forward still called the ctypes pointer. Both shapes are patched
now and a leftover check fails loud on any other shape. The old "4 sites" was 3 in the base plus 1
added by `pr-53055.diff`.

Two application facts to keep in mind: `pr-53055`'s tilelang hunks no longer apply, because the base
carries that fix in refactored form, while its two cutlass hunks still do and
`patch_cutlass_sm12x_guard` needs them; and on a second pass over an already-overlaid tree,
`patch_o_proj_b12x` and `patch_tp_allreduce_static_workspace` fail because `patch_region_profiler`
re-indents the text they key on. The second one is pre-existing and only matters for the incremental
re-apply path.

### Verification and open items

`pytest` 28 passed / 13 skipped; `verify-docs.py` at its unchanged 12 pre-existing findings, none
naming a touched file; `py_compile` and `bash -n` clean; the patched tree inspected at all four
regions. One judgement call recorded: the warmup-ext call is kept although its mHC half may be
redundant after #50178, because `gumbel_sample` is still warmed nowhere upstream, so check the boot
log after the rebuild and drop that half if the registry covers it.

**The v0.28-era stacks are out of scope as of 2026-09-17** (owner's call). That retires two things
this file had been carrying as constraints: the note that dropping `DEEPGEMM_COMMIT` from
`pin.main`/`pin.main-dg`/`pin.main-dg-1m` would make a future rebuild of those take their base's own
cmake tag (`v0.28.0` -> `8b1392b`), which no longer matters; and `configs/env.spark.sh`'s
`VLLM_USE_DEEP_GEMM_E8M0` default of 0, which existed only for those images, while every live pin now
sets 1 explicitly. The default was left as-is rather than flipped, because it is a shared file the
older pins inherit and the pins already override it, so changing it buys nothing measurable. What is
now dead weight and could be retired wholesale, after a usage audit, since the live path shares
`docker/Dockerfile.main`, `patches/apply_overlays.py` and `scripts/05-serve.sh` with them: the three
`pin.main*` files, `docker/Dockerfile.nvfp4`, `patches/v0.27.1/`, the historical donor patches
(`0002`, `0003`, `b12x-utils-main.py`, `pr47988.diff`, `pr54631.diff`,
`kv-offload-bounds-check.patch`), `scripts/02-build-main.sh`, and the `main-b12x-028-*` images.

Stale text that still names the removed pin, reported rather than rewritten: `knowledge/06` line 76,
`knowledge/00-index.md:110`, `knowledge/08` lines 29/52/153, `knowledge/09` lines 119/140,
this file at 634/2414/3288, `PLAN-MAIN.md` lines 72-73/731/749, and `HANDOFF.md`'s `/opt/DeepGEMM`
log lines.

Next: the phase-1 rebuild plus the two-node serve gate on the new base.

## 2026-09-17: b12x_ref WO head-to-head (measured, no port) and the proto2 build

Session opened at round 1 of the beat-the-anemll goal. Three things happened.

**1. Upstream re-check.** All nine tracked vLLM PRs are still **OPEN**, none merged: #53425, #53522,
#53271, #46716 (ours), #53055 and #47988 and #52499 (tracked), #50645 and #41834 (CONFLICTING).
Most recent movement is #53055 at 2026-09-16, the author still iterating. DeepGEMM #417, #419, #403
all OPEN; #417 last moved 2026-09-16 when its #447 dispatch change was confirmed on the GB10 pair.
**No upstream write was made this session**: `HANDOVER.md`'s constraints still say do not push and do
not open or comment on any PR, and the goal contract requires the owner's agreement before any
upstream action. Recording only, deliberately.

**2. The `b12x_ref` WO head-to-head is measured and closes the port premise.**
`harness/bench_b12x_wo.py` runs in `vllm-spark-0731:main-029-proto-b12xref` on spark1, which carries
our `b12x` and the reference's 0.15.3 as `b12x_ref`. Both native WO-A/WO-B kernels were driven with
the same FP8 block-scaled weights and the same input at the DSV4-Flash shapes (hidden 4096, groups 8,
group_width 4096, rank 1024). Agreement check first: relative difference 1e-6 at 1 token and 0.0 at 8
and 64, so the two compute the same thing and the comparison is valid. Median of 50 CUDA-event
timings:

| tokens | `b12x` (ours) | `b12x_ref` | delta |
|---|---|---|---|
| 1 | 0.374 | 0.331 | ref -0.042 |
| 2 | 0.372 | 0.336 | ref -0.036 |
| 4 | 0.387 | 0.337 | ref -0.050 |
| 8 | 0.381 | 0.344 | ref -0.037 |
| 16 | 0.381 | 0.396 | ours -0.015 |
| 32 | 0.410 | 0.426 | ours -0.015 |
| 48 | 0.405 | 0.446 | ours -0.042 |
| 64 | 0.417 | 0.479 | ours -0.063 |

The older generation is faster only at or below 8 tokens (max 0.050 ms) and slower from 16 up. A c6
decode step runs about 48 rows, where ours is already ahead. **Do not port.** Best case for a port is
0.13 % of a c6 step. One caveat rides with the number: the profiler's 0.60 ms WO region is the *live*
path (fused inverse-RoPE quant, dequant, grouped `bmm`, `wo_b` linear), so this closes the
kernel-generation question and says nothing about whether the live WO path is good. Raw log:
`outputs/driver/one-off/b12x-wo-headtohead.log`, verdict in `docs/EXPERIMENTS.md` under *Kernel
probes*.

**3. The `proto-v0.2.0` port had never reached the rig.** The pin says `VLLM_REF=f37c550bf635` and
`IMAGE=vllm-spark-0731:main-029-proto2`, but spark1's repo copy was still the pre-port state
(`VLLM_REF=69db1c26b4`, `IMAGE=main-029-proto`, no `scripts/port_scan.py`, no
`docs/GOAL-BEAT-ANEMLL.md`). The Sep 17 port was made here and never synced. The working tree was
therefore the only place the new base existed, and every number in `docs/EXPERIMENTS.md` remains
proto-v0.1.0-era. The desktop copy was rsynced to spark1 (build surface plus docs; `.scratch`,
`outputs`, `.git` excluded) and the phase-1 build of `main-029-proto2` was started detached
(`~/proto2-build.log`, `MAX_JOBS=16`, `MAX_JOBS_TORCH=8`, host otherwise idle, 117 GiB available).
The build log confirms the intended commit: `vllm ... f37c550bf635`.

Next: finish the phase-1 build, copy the derivative to spark2, and re-baseline the protocol on the new
base before any A/B is believed. The b12x MLA and indexer head-to-heads remain open, and the real NVFP4
KV writer (goal item 1) is untouched.

## 2026-09-17 (later): the MLA head-to-head points the other way, and that relocates the gap

Continued the same session. The WO result said the `b12x_ref` generation holds no prize; the MLA
measurement says it holds a regression.

`harness/bench_b12x_mla.py` drives both stacks' public compressed-sparse-MLA decode entry points on
one DSV4-Flash decode contract: 48 query rows, 32 local q heads (TP=2), head_dim 512, SWA window 128,
indexed topk 512 pages, 584 B/token packed page. Getting there needed the reference's own plumbing:
inputs come from `b12x_ref.compressed_reference.pack_compressed_mla_kv_cache_reference`, the
pure-torch `compressed_sparse_mla_reference` is carried as ground truth, and its side needs a full
`B12XAttentionArena.allocate(caps)` plus `make_workspace(contract)` before
`compressed_mla_decode_forward` will run at all. Two contract details bit on the way and are worth
recording: the unified SM12x backend takes **raw slot ids only** and rejects a mapped
`indexed_page_table`, and our scratch `Caps.max_width` must cover SWA + indexed (`128 + 512 = 640`),
not the SWA window alone.

| library | run 1 | run 2 | run 3 | median | vs pure-torch reference |
|---|---|---|---|---|---|
| `b12x` (ours) | 0.2236 | 0.2238 | 0.2257 | **0.224** | 0.50 % |
| `b12x_ref` | 0.6138 | 0.6105 | 0.6319 | 0.614 | 0.50 % |

Both are 0.50 % from the pure-torch reference and agree with each other to 0.10 %, so this is the same
mathematics and the timings are comparable. **Ours is 2.7x faster.** Do not port anything from the
older generation into the attention path.

**The consequence matters more than the number.** The anemll engine cannot be earning its ~25 % lead in
attention if its attention kernel is 2.7x slower than ours. The gap must be where the profiler already
put most of the step: ffn/MoE at 1.84 ms of the 2.83 ms target layer, 65 %, flat across layers. That is
where the next attribution belongs, and it is also where goal item 2 (NVFP4 per layer family, MoE
first) already points. Caveat kept with the number: this compares each stack's public decode entry
point, and `compressed_mla_decode_forward` splits into chunks and merges while ours is a single fused
call, so part of the 2.7x is entry-point design rather than kernel throughput.

Recorded at the same time: `docs/EXPERIMENTS.md` gained a *Kernel probes* section (two probes now, WO
and MLA) generated by `scripts/experiment-ledger.py`, so offline measurements carry verdicts in the
same ledger as the protocol arms instead of living only in `outputs/driver/one-off/`.

## 2026-09-17 (later still): the MoE region gets a leading attribution, and it is structural

The MLA probe relocated the gap to ffn/MoE, which the profiler prices at 65 % of a target layer. The
first question there is whether our kernel generation even covers the regime the engine runs in.

`harness/probe_moe_tuning.py` compares both generations' decode MoE `MAX_ACTIVE_CLUSTERS` policy
coverage:

| routed rows | ours micro | ours dynamic | ref micro | ref **static** | ref dynamic |
|---|---|---|---|---|---|
| 20 | 84 | 188 | 84 | 148 | 188 |
| 48 | - | 188 | - | 149 | 188 |
| 144 | - | 188 | - | 130 | 188 |
| 240 | - | 188 | - | 141 | 188 |
| 288 | - | 188 | - | 175 | 188 |
| 640 | - | 188 | - | 188 | 188 |
| 1024 | - | 147 | - | - | 147 |

**Our generation has no `static` MoE backend at all.** `b12x.moe.fused_moe` exposes no `static`
module, while the reference's `b12x_ref.moe.fused` exposes `MoEStaticKernel`, `MoEStaticKernelSilu`,
`MoEStaticKernelRelu2` and the backend. The policy registries differ the same way: ours holds
`decode/micro`, `decode/dynamic`, `decode/dynamic_w4a8_decode`; the reference holds `decode/micro`,
`decode/static`, `decode/dynamic`. Our `micro` ladder stops at 20 routed rows and our `dynamic` ladder
is flat at 188 until 640, so everything in between is one untuned configuration.

Every level of the protocol falls in that band. Per `docs/knowledge/02-model.md` Flash has 256 routed
experts at 6 per token, and `q_rows = c * (7 + 1)`:

| level | q rows | routed rows | our policy | reference policy |
|---|---|---|---|---|
| c1 | 8 | 48 | dynamic, cap 188 | static, cap 149 |
| c3 | 24 | 144 | dynamic, cap 188 | static, cap 130 |
| c5 | 40 | 240 | dynamic, cap 188 | static, cap 141 |
| c6 | 48 | 288 | dynamic, cap 188 | static, cap 175 |

**This is recorded as structure, not as a win.** No throughput is claimed: the probe reads registries
and kernel availability, it does not time anything. The next measurement is an MoE kernel head-to-head
at these shapes, which needs NVFP4 expert weights packed for both generations, to turn this into
milliseconds. It is the highest-value offline probe left and it belongs before further protocol work.
A fix would sit under goal item 4 (third-party kernel generation) or item 3 (our own configuration),
depending on where the regime choice is actually made.

Also this round: `docs/EXPERIMENTS.md` now carries three probes (WO, MLA, MoE coverage) under
*Kernel probes*, all generated by `scripts/experiment-ledger.py`.

## 2026-09-17 (round 6): the MoE cluster-cap hypothesis came back inconclusive

The previous entry left ffn/MoE with a structural attribution: our generation has no `static` MoE
backend and runs a flat `max_active_clusters` cap of 188 across the whole mid band, where the
reference tunes a different kernel per row count. The obvious next question was whether that cap alone
costs time.

`harness/bench_moe_clusters.py` builds synthetic expert weights in the real contract — uint8 FP4
`[E, 2*inter, hidden/2]` and `[E, hidden, inter/2]`, E8M0 `[E, rows, K/32]` grids passed plain rather
than swizzled, A8 activation, unit global scales — and sweeps only the cap at the DSV4-Flash decode
shape (48 tokens x topk 6 = 288 routed rows, 256 experts, hidden 4096, intermediate 2048):

| `max_active_clusters` | median ms | vs cap 188 |
|---|---|---|
| 188 (our flat default) | 10.4787 | 1.000 |
| 175 | 10.3872 | 0.991 |
| 149 | 10.4693 | 0.999 |
| 141 | 10.3875 | 0.991 |
| 130 | 10.4787 | 1.000 |
| 96 | 10.3783 | 0.990 |

**Flat within 1 %.** The first run of this bench, without seeding, showed -5.1 % at cap 149 and looked
like a small win; it was input re-randomisation between caps in my own harness. Holding the work fixed
removed it, and the residual ordering is non-monotonic. **A cap-only change is not the lever.**

Two problems stop this counting as an attribution, and both are recorded rather than papered over:

1. **Not representative.** 10.4 ms for one MoE with 288 routed rows cannot be part of a 43-layer decode
   step measured at 38.65 ms — it is roughly an order of magnitude too slow. This single-GPU
   256-expert execution plan is not the path the engine runs; the served path shards experts and
   activations differently and gets a tuned plan.
2. **A red flag on the harness.** Output sums differ across caps by up to 35 % (516316 against 795297).
   A correct kernel should not do that, so until the cause is found this harness cannot be trusted to
   compare anything, and nothing here should be quoted as a positive result.

**Unchanged:** the absence of a `static` backend in our generation (`probe_moe_tuning.py`), which is a
registry and module-listing fact and does not depend on this bench. Next MoE step, in order: explain
the divergence, then re-shape the probe to the served configuration (TP sharding, real routing skew,
tuned plan).

`docs/EXPERIMENTS.md` now carries four probes: WO, MLA, MoE coverage, and this sweep.

## 2026-09-17 (round 6, build): proto2 is at the package stage

The phase-1 build of `main-029-proto2` on spark1 has cleared NCCL (#11 DONE 197 s), the PyTorch
submodule clone and compile (#12), and is now at #21 installing Python packages. The log still records
the intended commit, `vllm ... f37c550bf635`. No protocol measurement on the new base exists yet.

## 2026-09-17 (round 7): the MoE divergence was my harness, and the cap is a real lead

Last entry left the cluster sweep inconclusive with a red flag: output sums differing by up to 35 %
across caps. `harness/diag_moe_divergence.py` found the cause, and it was not the kernel.

The diagnostic asked three questions in one process. Repeated `run` calls at a fixed cap were perfectly
stable (identical to 0.1 in four repeats). Two *rebuilds* of the same case at the same cap disagreed
(516324 against 795311). Inputs were bit-identical (`x_equal` and `ids_equal` both true), and each
binding kept reading its own buffer correctly after a later build. So: no race, no cap effect,
build-order effect.

The cause is `fused_moe.prepare_weights`, which repacks the packed weight tensors **in place**
(`_logical_weight_to_w4a8_rp_inplace`, `_e8m0_scale_to_w4a8_sfb_inplace`). The sweep loop reused one
`PackedWeights` tuple across every cap, so every build after the first was handed already-repacked
data. Only the first build in a process was correct. **Worth knowing beyond this bench**: the
preparation call is not idempotent on its inputs, so any warmup path or repeated-build harness must
clone.

With a fresh weight copy per cap, the same sweep:

| `max_active_clusters` | median ms | min ms | vs 188 | output abs-sum |
|---|---|---|---|---|
| 188 (our flat default) | 10.8387 | 10.6737 | 1.000 | 516334.9 |
| 175 | 10.3170 | 10.1512 | 0.952 | 516318.8 |
| 149 | 10.8124 | 10.7429 | 0.998 | 516325.5 |
| 141 | 10.2148 | 10.1525 | 0.942 | 516328.1 |
| 130 | 10.7469 | 10.6717 | 0.992 | 516358.5 |
| 96 | 10.1690 | 10.1275 | 0.938 | 516327.4 |

Output sums now agree to **0.008 %** (516318.8 to 516358.5), which is FP32 atomic-accumulation order —
the kernel computes the same thing at every cap. The red flag is closed, against my own harness.

And a real effect appears: our flat default of 188 is the slowest cell, the caps the reference tunes
for this band (175, 141, 130) are 4.8-5.8 % faster, cap 96 is fastest at -6.2 %, and the ordering
reproduced across two runs. **Recorded as a lead, not a result.** It is non-monotonic, and cap 188
itself measured 10.4787 in one run and 10.8387 in the next — a 3.4 % swing on an unchanged
configuration, the same order as the effect, so by this repo's rule for keeping an arm it has not
beaten the spread. It also still cannot be projected onto the protocol: 10.4 ms for one MoE at 288
routed rows is roughly an order of magnitude above what a 43-layer step measured at 38.65 ms can
contain, so this single-GPU 256-expert execution plan is not what the engine runs.

Next: repeat the sweep to bound run-to-run noise, then reshape to the served configuration — TP
sharding, real routing skew, tuned plan — before any of it counts.

Build status unchanged in kind: `main-029-proto2` is at step #21, building the vLLM wheel with the
Rust frontend. No protocol measurement on the new base yet.

## 2026-09-17 (round 8): the MoE cap effect survives a repeat, but under contention

Round 7 left the MoE cluster sweep as a lead: a ~5-6 % separation between cap groups, but with a 3.4 %
run-to-run swing on the same configuration, which is the same order as the effect. This round repeated
the whole sweep three times per cap.

| `max_active_clusters` | rep 1 | rep 2 | spread | vs cap 188 |
|---|---|---|---|---|
| 188 (our flat default) | 11.557 | 11.761 | 1.8 % | 1.000 |
| 175 | 10.880 | 10.896 | 0.1 % | **0.941** |
| 149 | 11.582 | 11.632 | 0.4 % | 0.995 |
| 141 | 10.839 | 10.891 | 0.5 % | **0.938** |
| 130 | 11.635 | 11.591 | 0.4 % | 0.993 |
| 96 | 10.889 | 10.841 | 0.4 % | **0.939** |

Rep 0 is excluded, and the reason is a measurement trap worth keeping. The sweep ran while the phase-1
proto2 build was compiling vLLM's CUDA extensions: `ninja -j 16`, load average **16.0**, sixteen
`cicc` processes at 100 %. Rep 0 came back uniformly inflated (13.121 at cap 188 against 11.557 and
11.761 in the later reps) and cannot be compared with them. On this unified-memory part a CPU-bound
build perturbs GPU timing. Added to `HANDOVER.md`'s trap list. Re-run the sweep on an idle host before
trusting absolute values.

What the two clean reps say: the caps group the same way both times. 175, 141 and 96 land at
10.84-10.90 ms; our flat default 188, plus 149 and 130, land at 11.56-11.64 ms. That is a **6 %
separation reproducible to 0.5 % between reps**, and by the repo's keep rule it now clears the bar
(cap 175 beats 188 by 5.9-6.3 % against 188's own 1.8 % spread). Output sums agree to **0.005 %** across
all 18 cells (516314.3 to 516340.0), so the kernel computes the same thing at every cap.

The effect is **non-monotonic** in the cap value — 175 and 141 are fast, 149 and 130 are slow — so the
mechanism is not "fewer clusters is better". Two stable groups with noise inside each is the shape of a
grid or tiling threshold, which is worth understanding before anyone tunes on it.

**Still not a protocol win.** Absolute values carry the contention above, and 10.4-11.7 ms for one MoE
at 288 routed rows remains about an order of magnitude above what a 43-layer step of 38.65 ms can
contain, so this single-GPU 256-expert plan is not the path the engine runs. Next: repeat on an idle
host, then reshape to the served configuration (TP sharding, real routing skew, tuned plan).

Build: `main-029-proto2` still at step #21, vLLM CUDA extensions under `ninja -j 16`.

## 2026-09-17 (round 9): goal item 1 is closed, and the reference agrees with us

Goal item 1 asked for a real NVFP4 KV writer on the premise that our `nvfp4_ds_mla` is only an envelope
alias for the fp8 page while the reference has a genuine NVFP4 writer to port. The reference's own
source disagrees, so this was checked against the live image rather than against our notes.

`harness/probe_nvfp4_kv_width.py` runs read-only inside `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` and
locates the KV-width decisions. Both of them treat the two dtypes identically:

| reference file | dtype test | width returned |
|---|---|---|
| `vllm/v1/kv_cache_interface.py:381-386` | `fp8_ds_mla` **and** `nvfp4_ds_mla` | `storage_block_size * 584` |
| `vllm/v1/attention/backends/mla/sparse_swa.py:151-154` | `fp8_ds_mla` **and** `nvfp4_ds_mla` | 584 |
| `vllm/models/deepseek_v4/sparse_mla.py:104-107` | both | `(num_blocks, block_size, 584)` |
| `vllm/models/deepseek_v4/attention.py:619-620` | both | 584-byte DSpark envelope |

The reference's own comments, verbatim: "fp8_ds_mla/nvfp4_ds_mla are padded uint8 layouts. Keep the
upstream FP8 alignment and use the proven 584-byte DSpark NVFP4 envelope", and "DeepseekV4 uses the
padded 584-byte sparse-MLA envelope for **both** fp8_ds_mla and nvfp4_ds_mla". `sparse_mla.py` gives
the composition as "448 NoPE + 128 RoPE + 8 fp8 scale", and 448 + 128 + 8 = 584 is exactly the
arithmetic behind our own `_DSV4_TOKEN_BYTES = 584`.

**Item 1 is closed with the number that closes it: 584 B/token on both engines.** There is no narrower
writer to port on either side. The 7,650 B/token figure in the goal text is a whole-model footprint
that counts the indexer and SWA caches, not a per-layer dtype width. This confirms the 2026-08-26
correction already recorded in `docs/knowledge/04-quantization-kv.md`, now verified against the live
image rather than against field notes.

**What is left of the item is capacity, not a writer.** anemll reports a ~2.0M-token KV pool against
our ~97k at similar utilisation, and the knowledge doc attributes that to runtime and weights footprint
rather than dtype width. That is a serving-level difference the protocol measures directly, so it
belongs with the re-baseline rather than with a kernel port.

Honest limit on the closure: the widths are verified from the Python-side decision functions and their
stated byte composition. The reference's writer kernel was not byte-compared; the comments above plus
the independent FlashMLA README cross-check in `04-quantization-kv.md` are what stand in for that.

`HANDOVER.md` gained a "Goal items: what is closed and what is open" section so the next session does
not reopen item 1 or re-run the b12x generation head-to-heads.

Build: `main-029-proto2` still at step #21 under `ninja -j 16`, load average 16.0, so no GPU
measurement was trustworthy this round and none was attempted beyond read-only image inspection.

## 2026-09-17 (round 10): the comparator runs a different MoE backend, and we never tested it

Upstream re-check this round: all twelve tracked items still open, none merged. vLLM `#53425`,
`#53522`, `#53271`, `#46716`, `#53055`, `#47988`, `#52499`, `#50645`, `#41834` — `#53055` MERGEABLE
last touched 2026-09-16, `#41834` CONFLICTING; DeepGEMM `#417`, `#419`, `#403` open, `#417` last
moved 2026-09-16. No new movement since the previous check, and no upstream write made (HANDOVER
forbids it, and the contract wants the owner's agreement first).

Then the round's finding, which is the sharpest lead of the session and came from reading the
comparator's recipe rather than assuming the two arms were configured alike.

`harness/ref-base0731.yaml`, the authority on what the reference arm runs, passes
**`--moe-backend flashinfer_b12x`**. Our pin defaults `MOE_BACKEND` to **`b12x`**. These are different
implementations:

- `flashinfer_b12x` dispatches to `FlashInferB12xExperts` (`fused_moe/experts/flashinfer_b12x_moe.py`),
  whose docstring reads: "Uses `b12x_fused_moe` from FlashInfer PR #3080 which fuses token dispatch,
  two GEMMs, SwiGLU activation, and topk-weight reduction into a **single kernel call**. Input
  quantization (BF16->FP4) is performed inside the kernel so BF16 hidden states are passed directly."
  NVFP4 only.
- `b12x` dispatches to `B12X_MXFP4`, the b12x package's own MoE.

Availability was checked in both images so this is not mistaken for a missing package, which was my
first reading and was wrong:

| image | flashinfer | `b12x_fused_moe` | `has_flashinfer_b12x_moe()` |
|---|---|---|---|
| reference `0.1.1` | 0.6.15 | present | True |
| ours `main-029-proto-b12xref` | 0.7.0 | present, plus `B12xNvfp4Config/Runner`, `B12xW4A16Config/Runner` | True |

The name's absence from the *old base image's* `vllm` package is what confused the first pass; upstream
at our pinned commit `f37c550bf635` has `flashinfer_b12x_moe.py` (11717 B, confirmed through the GitHub
API), `patches/assert_stack.py` already lists `ALLOWED_MOE = ("b12x", "flashinfer_b12x")`, and
`configs/pin.golden.env` already sets it.

**It has never been measured on our arms.** The MoE sweep in `docs/EXPERIMENTS.md` covered `b12x`,
`humming`, `flashinfer_trtllm` (died at worker init: kernel does not support current device cuda) and
`flashinfer_cutlass` (never healthy). Not this one. Since ffn/MoE is where the profiler puts 65 % of a
target layer and attention is already closed as the reference's source of speed, this is the
highest-value one-variable arm left: `MOE_BACKEND=flashinfer_b12x`, no rebuild, run the protocol.

Caveat carried with it: `FlashInferB12xExperts` asserts NVFP4 expert weights, and
`docs/knowledge/04-quantization-kv.md` records NVFP4 *weight* attempts as a dead end on this model,
though `02-model.md` says the checkpoint's experts ship as fp4 and the MXFP4 oracle maps
`flashinfer_b12x` to `B12X_MXFP4`, so the MXFP4 route may reach the same kernel. It runs or it is
rejected at load; either is a result.

Evidence: `outputs/driver/one-off/moe-backend-comparator.log`. `HANDOVER.md` gained a section naming
this the first arm after the proto2 rebuild lands.

Build: `main-029-proto2` still at step #21 under `ninja -j 16`, load average 16.0. No GPU measurement
attempted this round beyond read-only image inspection.

## 2026-09-17 (round 11): why the reference had to ask for `flashinfer_b12x` by name

Round 10 found the comparator running `--moe-backend flashinfer_b12x` while our arms run `b12x`. Two
facts from the reference's oracle source turn that from a curiosity into an explanation.

**It is not a default.** `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, in the selection
ordering comment:

```
# NOTE: the kernels are selected in the following order.
# FLASHINFER_B12X is intentionally excluded from auto-selection until
# the upstream CUTLASS SM121 MMA op guard is resolved; use
# moe_backend="flashinfer_b12x" to opt in explicitly.
```

The reference arm explicitly opts in to a kernel upstream will not choose by itself.

**And the auto-selected set is the set that does not work here.** The same function narrows its
candidates to `NVFP4_BACKENDS_WITH_CLAMP = {FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, MARLIN}` whenever
`config.swiglu_limit is not None`, and DeepSeek-V4-Flash sets `swiglu_limit 10.0`. Our own MoE sweep
recorded `flashinfer_trtllm` dying at worker init ("Mxfp4 MoE backend 'FLASHINFER_TRTLLM_MXFP4_MXFP8'
does not support the deployment configuration since kernel does not support current device cuda") and
`flashinfer_cutlass` never becoming healthy. So the clamp-restricted candidate set is exactly the set
this rig rejects, and `FLASHINFER_B12X` is not in it — which is why the reference had to name it
explicitly and why our sweep, which never tried that name, came away thinking the MoE vendor menu was
exhausted and only `humming` was runnable.

**Caveat that follows, and it is a correctness caveat not a speed one:** because `FLASHINFER_B12X` is
absent from the clamp set, the explicit opt-in may bypass swiglu-clamp handling the auto-selected
backends would have applied. Both protocol gates exist to catch exactly that, so the arm is judged on
gates and acceptance, not on throughput alone.

The arm, with `MOE_BACKEND` as a host-side variable — it is read at `05-serve.sh:63` and `:334` to build
the serve argv, so unlike the `VLLM_*` profiling variables it does not need `SERVE_EXTRA_ENV`:

```
MOE_BACKEND=flashinfer_b12x NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 \
  GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS="--async-scheduling" \
  bash scripts/05-serve.sh main-029
```

Worker on spark2 first, head about 85 s later, which is what `harness/probe-moe.sh <backend>` already
does.

Build: `main-029-proto2` roughly 23 minutes in, 16 `nvcc` processes compiling vLLM's CUDA extensions,
load average 16.5. Still no image.

## 2026-09-17 (round 12): readiness for the proto2 build, and a copy-script trap

No GPU measurement this round by design: the build holds load average ~16 and round 8 measured that
same condition inflating GPU timings by 13 % on unchanged configuration, so a protocol run now would
be evidence of nothing. Time went into making the post-build sequence exact and finding the traps in
it before they cost a cycle.

**spark2 is idle and ready.** 2.1 TB free on `/`, 117 GiB available memory, zero containers, zero
`sparkrun_*` leftovers, no llama.cpp or vLLM process, load 0.07. Nothing is serving anywhere on the
rig.

**Trap found in `scripts/02-copy-main.sh`.** It sources `configs/pin.main.env`, whose `IMAGE` is the
v0.28-era `vllm-spark-0731:main-b12x`, and takes the tag as `$1`. Calling it bare therefore copies the
wrong image to spark2, and the failure mode is the confusing one this repo already documents: the
worker cannot start, the head blocks waiting for rank 1, and it looks like a broken build. The tag
must be passed explicitly: `bash scripts/02-copy-main.sh vllm-spark-0731:main-029-proto2`.

Also verified: `scripts/05-serve.sh` runs `patches/assert_stack.py` at startup, so the stack assertion
is automatic at serve time and needs no separate step; and `scripts/07-stop.sh` does not remove
`sparkrun_*` containers, which the port guard in `05-serve.sh` will otherwise catch.

The full sequence is now a runbook section in `HANDOVER.md`: verify the image records
`f37c550bf635`, copy with the explicit tag, clear leftovers on both nodes, serve worker first and head
about 85 s later, meter both arms the same day, then run the `flashinfer_b12x` arm with both gates
read.

Build: `main-029-proto2` about 24 minutes in, 16 `nvcc` processes on vLLM's CUDA extensions, load 16.4.
Still no image.

## 2026-09-17 (round 13): the arm is one line, and the runbook now says so

Preparation continued; the build still holds load ~16 so no GPU measurement was taken. The useful
finding is that the repo already has the runner, and my previous round's hand-written serve command
duplicated it.

`harness/run-arm.sh <tag> "<env assignments>"` performs the whole arm: it removes `vllm-ds4-0731` on
both nodes, deletes the `/dev/shm` `psm_*` / `nccl-*` / `sem.mp-*` leftovers that break a restart,
queues the worker on spark2 85 s ahead of the head, polls `/health` for up to 450 s, writes the engine
log to `~/goal/<tag>-engine.log`, and finishes by running `~/drive-median.sh <tag> 3 512 1 3 5 6`. The
standard flags (k=7, capture 48, util 0.8389, `--async-scheduling`) are baked in, so the second
argument is the only difference between arms.

So the first arm after the build lands is exactly:

```
bash harness/run-arm.sh fb12x "MOE_BACKEND=flashinfer_b12x"
```

and the reference arm is `bash ~/goal/launch-refbase.sh` followed by
`~/drive-median.sh refproto 3 512 1 3 5 6`, same day.

Prerequisites verified present on spark1: `~/goal/` (where the engine logs land), `harness/run-arm.sh`,
`harness/probe-moe.sh`, `~/drive-median.sh`, `~/goal/ref-base0731.yaml`, `~/goal/launch-refbase.sh`,
`scripts/spark-launch.sh`. spark2 remains idle and ready.

`HANDOVER.md`'s runbook now uses these verified one-liners instead of my hand-written serve command,
which would have worked but also would have skipped the `/dev/shm` cleanup that `run-arm.sh` does.

Build: `main-029-proto2` about 25 minutes in, still 16 active `nvcc` processes on vLLM's CUDA
extensions. No image yet; the whole protocol re-baseline is gated on it.

## 2026-09-17 (round 14): correction — the backend was never missing, and the arm needs no rebuild

**Correction to round 10.** That entry says "The name's absence from the *old base image's* `vllm`
package is what confused the first pass; upstream at our pinned commit has `flashinfer_b12x_moe.py`".
The first half is **wrong**. Our old base image has the backend too.

The mistake was the scan path. The check ran over
`/usr/local/lib/python3.12/dist-packages/vllm` and found no file naming `flashinfer_b12x`, which was
read as the backend being absent. vLLM in these images is an editable install from `/opt/vllm`:
`vllm.__file__` is `/opt/vllm/vllm/__init__.py`, and that tree has **6** files naming the backend,
including `model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py` and the nvfp4 oracle's
`FLASHINFER_B12X`. Verified on `vllm-spark-0731:main-029-proto-ccompile` (vLLM
`0.1.1.dev0+g69db1c26b`, flashinfer 0.7.0): `has_flashinfer_b12x_moe()` is True and
`map_nvfp4_backend("flashinfer_b12x")` returns `NvFp4MoeBackend.FLASHINFER_B12X`. The lesson is
recorded in the ledger probe: read the package path from `vllm.__file__`, never assume
dist-packages.

What survives from round 10 is the substance, and it is stronger: the backend was available in both
images all along, the comparator opts into it by name through `--moe-backend flashinfer_b12x`, and our
measured arms ran `MOE_BACKEND=b12x` instead. The oracle comment excluding `FLASHINFER_B12X` from
auto-selection and the `NVFP4_BACKENDS_WITH_CLAMP` set that holds the three backends this rig rejects
are unaffected — those were read from the reference image's own source.

**And this removes the rebuild from the critical path for that arm.** Comparing the two nodes'
image lists by ID shows several images byte-identical on both:

| image | ID (both nodes) |
|---|---|
| `vllm-spark-0731:main-029-proto-ccompile` | `0eea65193697` |
| `vllm-spark-0731:main-029-proto-0912` / `main-029-proto` | `77841395cfcd` |
| `vllm-spark-0731:main-029-fp32dl`, `main-029-ablate`, `main-b12x-028`, `main-b12x` | matching IDs |

So the `b12x` against `flashinfer_b12x` A/B can run on `main-029-proto-ccompile` with **no proto2
image and no 29 GB copy**, as a same-image, same-day, one-variable arm — which is exactly what the
goal's loop asks for. It is not the headline comparison against the reference (that still wants the
new base and a same-day reference run), but it answers the question the whole remaining gap hangs on
before proto2 exists.

`HANDOVER.md`'s runbook now carries this as the early path, with `IMAGE=` overriding the pin since
`pin.main-029.env` points at proto2.

Build: `main-029-proto2` about 27 minutes in, 15-16 `nvcc` processes, still compiling.

## 2026-09-17 (round 15): the proto2 build failed on a missing package, fixed; and the MoE head-to-head ran

**The build failure, and it was one line.** Step #21 (the vLLM wheel) died after 1392 s with
`cmake --build` returning exit 1. The real error, three thousand lines above the Python traceback:

```
/opt/vllm/.deps/deepgemm-src/third-party/deep_jit/include/deep_jit/utils/exception.hpp:3:10:
fatal error: elfutils/libdwfl.h: No such file or directory
```

`libdw-dev` was missing. **This repo had already recorded the requirement and never applied it**: the
2026-09-16 entry in this file, from the DeepGEMM #447 work, says the build needs "`libdw-dev`
(DeepJIT's `exception.hpp` dlopens `libdw.so.1`)". The proto2 port moved DeepGEMM to the upstream fork
at `ad1f1726` with `deep_jit` as a submodule, which is exactly the code path that needs it, and the
Dockerfile was never updated. Fixed by adding `libdw-dev` to the base apt layer in
`docker/Dockerfile.main`. `libdw-dev` pulls `libdw1`, which the `dlopen` needs at runtime as well, so
one package covers both. The failed log is preserved at spark1:`~/proto2-build-failed.log`.

BuildKit cached every earlier stage, so the restart went straight back to #21 and is re-running the
vLLM wheel now.

**And the MoE kernel head-to-head ran**, which is the measurement the whole remaining gap hangs on.
`harness/bench_moe_headtohead.py`, both implementations in one process on one GPU at the DSV4 c6
decode shape (48 tokens, topk 6 = 288 routed rows, 256 experts, hidden 4096, intermediate 2048),
median of 30 CUDA-event timings per rep, five reps:

| library | rep medians (ms) | clean median | output |
|---|---|---|---|
| `b12x` (ours, `b12x.moe.fused_moe`) | 10.108 / 10.075 / 10.058 / 10.069 / 10.063 | **10.066** | finite, mean abs 2.54 |
| `flashinfer_b12x` (`flashinfer.fused_moe.b12x_fused_moe`) | 134.500 / 127.521 / 127.613 / 127.792 / 127.991 | **127.729** | finite, mean abs 2.55 |

Reproduced across two runs (the first: 10.67 against 133.72). FlashInfer chose its `static` backend
by itself — its JIT name records `static_m48_k4096_n2048_t6_r288_...` — which is the regime its own
tuning registry uses for 20 < routed rows ≤ 640, so it is what the reference would run at this shape.

Getting there required mirroring `FlashInferB12xExperts.process_weights_after_loading` exactly:
NVFP4 with **vec-16 E4M3** scales (`k1 = k1_sf * 16`; `fp4_quantize` refuses ue8m0 at vec 16, so the
working path is E4M3 and not the checkpoint's e8m0), the scale stack reshaped to `[E*N, K/16]` and
converted once with `num_groups=E`, per-expert alphas of 1.0 with the global scale baked into the
block scales, and `fc2_input_scale` forced to 1.0. Four attempts were needed and each error was
informative: ue8m0 requires vec 32, fp32 input is rejected (`fp16/bf16/e4m3` only), the E4M3 path
requires an explicit `global_scale`, and the 6-D MMA shape from the converter is `(32,4,m_tiles,4,
k_tiles,E)` — the expert axis belongs in the group slot, not prepended.

**The result is recorded and deliberately not claimed.** A 13× separation in our favour would mean the
comparator's MoE backend is a red herring rather than its source of speed — but 127.7 ms for one MoE
layer cannot sit inside a 43-layer step measured at 38.65 ms, so the reference is not driving this
kernel the way the probe does. Prime suspect: sharding. The real engine runs TP=2 with experts split
across ranks, while the probe puts all 256 experts and all 288 rows on one GPU, where a fixed-m48
static tile pads every expert to 48 rows. Until that is eliminated the honest statement is about this
configuration only. Next: repeat with `num_local_experts` 128 and 64 and a prewarmed captured
binding.

## 2026-09-17 (round 16): FlashInfer 0.7.0's `b12x_fused_moe` is 8-12x slower than 0.6.15's

The MoE head-to-head left a contradiction: our image's `b12x_fused_moe` measured ~13x our own MoE,
which cannot be true of the engine the comparator runs, since the cost exceeds a whole step. The prime
suspect was sharding, so the probe swept local expert counts first.

**Sharding is refuted.** `harness/bench_moe_headtohead.py` now sweeps `E_local` = 256/128/64 with the
same 288 routed rows:

| local experts | rows/expert | our b12x MoE | flashinfer_b12x | ratio |
|---|---|---|---|---|
| 256 | 1.12 | 10.97 | 141.99 | 12.9x |
| 128 | 2.25 | 7.03 | 79.74 | 11.3x |
| 64 | 4.50 | 4.34 | 44.36 | 10.2x |

Both scale with expert count as weight traffic predicts, and the ratio barely moves. Not sharding.

**The version hypothesis holds.** The reference runs FlashInfer **0.6.15**, ours **0.7.0**.
`harness/bench_flashinfer_moe.py` depends only on FlashInfer, so it runs unchanged in both images and
measures the same kernel — the module is `flashinfer.fused_moe.cute_dsl.b12x_moe` in both wheels:

| local experts | 0.7.0 (ours) | 0.6.15 (reference) | ratio | `mean_abs` output |
|---|---|---|---|---|
| 256 | 150.170 / 148.978 / 150.477 | 12.291 / 12.408 / 12.430 | **12.1x** | 2.6049 vs 2.6049 |
| 128 | 71.779 / 72.158 / 71.806 | 7.672 / 7.770 / 7.672 | **9.4x** | 2.6084 vs 2.6086 |
| 64 | 38.214 / 37.795 / 37.798 | 4.464 / 4.417 / 4.424 | **8.5x** | 2.6007 vs 2.6000 |

**The outputs are identical between the wheels**, agreeing to the fourth decimal in `mean_abs` at every
expert count. Same computation, same values, only the speed differs — and ours is the slow one. The
0.6.15 signature lacks `input_global_scale`; the call filters to each build's accepted parameters and
that is the only argument dropped, so it is like for like. The standalone probe also reproduces the
head-to-head's numbers in our image (150.2 / 71.8 / 38.2 against 142.0 / 79.7 / 44.4), confirming the
earlier 13x measured this kernel rather than a mis-built weight path.

**This is the item-4 finding of the session.** The comparator opts into `flashinfer_b12x` by name, its
wheel runs that kernel an order of magnitude faster than ours, and attention is already closed as the
source of its lead. Whether it is the *cause* of the remaining gap still needs the served measurement.

**What is deliberately not claimed.** No absolute cost. At 128 local experts even the fast wheel spends
7.67 ms per layer, so 43 layers would exceed the entire 38.65 ms step — meaning this probe's geometry
is probably not the served one, expert dimensions being the prime suspect, and the profiler's own
figure is 1.84 ms per layer for ffn/MoE at c1. The version comparison survives that concern because
both wheels were handed identical inputs and returned identical outputs.

Next: correct the probe geometry against the checkpoint's real expert shape, then decide whether to pin
0.6.15's kernel or the wheel itself — checking with the owner before any upstream contact, as the
contract requires.

Build: `main-029-proto2` re-running step #21 with `libdw-dev` present, ~11 minutes in, configure phase.

## 2026-09-17 (round 17): a memory-bandwidth floor the repo's step-time figures cannot satisfy

Two probes, both cheap, and together they put a hard number under the attribution work.

**The probe geometry was right.** Both MoE probes assumed 256 routed experts, 6 per token, intermediate
2048, hidden 4096, 43 layers. The checkpoint's own `config.json` (snapshot `7872f01b1d1f`) says
`n_routed_experts 256`, `num_experts_per_tok 6`, `moe_intermediate_size 2048`, `hidden_size 4096`,
`num_hidden_layers 43`, `expert_dtype fp4`. So the caveat recorded with the FlashInfer version result
-- that the geometry might not be the served one, the expert dimensions being the prime suspect -- is
**retracted**. The version comparison stands on the served shape.

**The bandwidth, measured rather than assumed.** A 1 GiB device-to-device copy on this GB10 moves
1.074 GB in 4.955 ms: **216.7 GB/s effective** (the spec is ~273, and a copy reaches about 80 % of
it). `harness`-side: `/bench/bwprobe.py`, log `outputs/driver/one-off/gb10-bandwidth.log`.

**The traffic it implies.** At the confirmed shape, one expert is 25,165,824 params, so fp4 weights are
3.22 GB per layer and **138.5 GB across 43 layers**. At the measured bandwidth that is 14.87 ms per
layer, **639.2 ms for a full pass on one rank, 319.6 ms at TP=2**.

**The constraint, and why it matters.** The checkpoint is 155-167 GB in total and the experts are all
but all of it, so a full 43-layer forward pass must stream ~138.5 GB and cannot finish faster than
about **320 ms across two ranks**. The repo's quoted c6 step time of 38.65 ms is therefore not a
full-model pass, and neither can the profiler's 1.84 ms per layer for ffn/MoE be reconciled with one:
the two figures describe different things. Every per-layer budget in `HANDOVER.md` and
`docs/knowledge/05-performance.md` was derived from those numbers, so they need re-deriving before more
attribution is built on them.

**What it does for the kernel question.** The MoE layer is bandwidth-bound, and both good kernels sit
at the floor: at 128 local experts ours measures 7.03 ms against a 7.43 ms traffic estimate, and
FlashInfer 0.6.15 measures 7.67 ms. Our implementation is not wasting bandwidth. FlashInfer 0.7.0's
71.78 ms for the same work is about **9.7x off the floor**, which makes the version regression a
defect in that wheel rather than a property of the shape.

Build: `main-029-proto2` still in step #21, 16 `nvcc` active, ~17 minutes in. Unchanged otherwise.

## 2026-09-17 (round 18): the step-time basis was wrong by the concurrency, and it explains everything

The expert-traffic probe said a full 43-layer pass must stream 138.5 GB, which cannot be done in
38.65 ms. The meter logs in `outputs/driver/` say why the figure looked that small.

`HANDOVER.md`'s "step time is 38.65 ms against 29.4 ms at c6" divides wall time by `drafts_per_req`.
That counter is **per sequence** - 512 tokens at 4.676 tokens/step is 109.5 forward passes for one
sequence, and `drafts_per_req` 657 is 6 x 109.5 - while the forward pass serves all six sequences. So
the quoted step times are too small by exactly the concurrency:

| level | tag | wall s | tokens/step | steps per sequence | **step ms** | quoted |
|---|---|---|---|---|---|---|
| c6 | `refg` (reference) | 19.33 | 4.676 | 109.5 | **176.5** | 29.4 |
| c6 | `protog` (ours) | 25.89 | 4.585 | 111.7 | **231.8** | 38.65 |
| c1 | `refg` | 7.79 | 4.923 | 104.0 | **74.9** | 13.35 |
| c1 | `protog` | 12.49 | 5.069 | 101.0 | **123.7** | 8.09 |

Two independent checks confirm the corrected basis. The factor is exactly the concurrency, 29.4 x 6 =
176.4. And the corrected ratio 176.5 / 231.8 = 0.761 matches the measured throughput ratio 118.64 /
158.93 = 0.747, whereas the quoted pair gives 38.65 / 29.4 = 1.31, its inverse.

**This is why the attribution never closed.** Every per-layer budget in `HANDOVER.md` and in
`docs/knowledge/05-performance.md` - the target forward being 82-85 % of the step, ffn/MoE at 1.84 ms
of a 2.83 ms layer, WO at 0.60 ms, MLA at 0.20 ms - was computed from a step that is 6x larger than
assumed. A correction section was written into `HANDOVER.md` rather than silently rewriting the
diagnosis, so the old numbers and their provenance stay visible.

Against the expert-traffic floor (138.5 GB per pass at 216.7 GB/s measured for a copy, >=320 ms at
TP=2) the corrected 176.5 ms is the same order and lower, which is consistent: reads beat a copy, and
at 288 routed rows over 256 experts roughly a third of experts take no rows under Poisson(1.125)
routing and are skipped. **The step is memory-bound on expert weights, so MoE is essentially the whole
game** - which elevates the MoE leads and makes the small-kernel budget shares suspect.

Unaffected: the direct kernel comparisons. WO at ~0.4 ms, MLA at 0.224 ms against the reference's
0.614 ms, and the FlashInfer version regression are all kernel-to-kernel measurements that never used
the step-time basis.

Build: `main-029-proto2` still in step #21, 16 `nvcc`, ~18 minutes in.

## 2026-09-17 (round 19): the gap priced in expert bytes, one number instead of a list of suspects

Continuing from the basis correction. Streaming bandwidth on this part, measured three ways
(`/bench/bwprobe3.py`, log `outputs/driver/one-off/gb10-gemv-bandwidth.log`):

| pattern | working set | GB/s |
|---|---|---|
| device copy (read+write) | 1 GiB | 223.2 |
| GEMV, a weight matrix streamed once (the MoE's own pattern) | 128 MiB | 201.0 |
| GEMV | 512 MiB | 198.9 |
| GEMV | 32 MiB | 156.6 |

The 273 GB/s the spec sheet implies is not reachable; ~200-223 GB/s is what streaming gets. (An
earlier `torch.sum` attempt gave 12.1 GB/s and was discarded - that is a bad reduction kernel, not a
bandwidth measurement, and it would have produced a nonsense floor.)

**Expert traffic**, at the confirmed shape: 3.22 GB per layer, 138.5 GB over 43 layers. Against the
corrected c6 step times:

| arm | c6 step ms | ms/layer | GB streamed per rank at 223 GB/s | fraction of a full expert layer |
|---|---|---|---|---|
| reference `refg` | 176.5 | 4.10 | 0.916 | **28.4 %** |
| ours `protog` | 231.8 | 5.39 | 1.202 | 37.3 % |

**The reference's whole step is expert streaming.** Only ~28 % of a full expert layer fits in its
4.10 ms per-layer budget, and the mechanism that predicts that is expert sharding plus empty-expert
skipping: with 128 local experts per rank and half the 288 row-expert pairs falling to each,
1 - exp(-144/128) = 67.5 % of the local 128 are touched, which is 33.6 % of the full 256 - near the
28.4 % the timing implies. So the step is memory-bound on precisely that traffic and there is no room
in it for any other region to matter.

**And it prices the gap in one quantity: we stream ~31 % more expert bytes per layer for the same
work.** Our 1.202 GB against the reference's 0.916 GB is the entire 55.3 ms difference between a
231.8 ms step and a 176.5 ms one. This replaces the region-share attribution, which was computed from
a step 6x too small.

The kernel is not the suspect: at 128 local experts our MoE measured 7.03 ms against FlashInfer
0.6.15's 7.67 ms, so the two are comparable per unit of work. The candidates are concrete and testable
without touching the reference - a different expert sharding that reads more experts per rank, or a
failure to skip experts that receive no tokens. A fix would sit under goal item 3.

Caveat carried with the numbers: the touched fraction is predicted from assumed uniform routing, and
real routing skews, so 28 % and 31 % are the right order rather than exact.

Build: `main-029-proto2` still in step #21, 16 `nvcc`, ~20 minutes in. Unchanged.

## 2026-09-17 (round 20): no-skipping hypothesis refuted, and the build clears its old failure point

**The kernel does skip, and it is efficient.** The expert-bytes attribution left two candidate
mechanisms for the ~31 % gap; one was a failure to skip experts that receive no tokens. Testing it
directly - weights, shapes and 288 routed rows fixed, only the number of distinct experts the routing
targets varying - gives a straight line:

| distinct experts touched | clean median ms |
|---|---|
| 174 | 13.29 |
| 116 | 9.67 |
| 61 | 6.13 |
| 32 | 4.19 |
| 8 | 2.72 |

0.0637 ms per touched expert plus about 2.21 ms fixed. One expert is 12.6 MB of fp4 weights, which at
the measured 223 GB/s costs 0.0564 ms, so the marginal cost is **88 % of the streaming bound**. Our
MoE skips untouched experts and streams the ones it touches at near-peak bandwidth.

The first run of this probe was wrong and is not evidence: its routing pool was
`randperm(active)[:topk]`, so every case targeted only 6 distinct experts. That accident showed 6
experts at 2.1 ms against ~232 at 10.9 ms in the head-to-head probe - the same conclusion by another
route - but the curve above is the measurement. Recorded because a bug that produces a plausible
direction is worth naming.

**What it does to the attribution.** The ~31 % expert-byte gap cannot be a skipping failure and cannot
be kernel inefficiency, since both are now measured. It must be the **number of experts each rank
touches**, which is expert parallelism or routing distribution - a configuration question under goal
item 3, not a library matter under item 4. Limits: one layer in isolation, synthetic weights, near
uniform routing where the served routing skews, so the real experts-touched-per-rank still has to be
read off the engine.

**And the build cleared its old failure point.** `main-029-proto2` is at 26:05 inside step #21 with
zero `libdwfl.h` / `fatal error` / `ERROR:` lines and 16 `nvcc` processes still compiling, where the
first attempt died at 23:12 on the missing `elfutils/libdwfl.h`. The `libdw-dev` addition worked;
what remains is the rest of the CUDA extension compile and the image commit.

## 2026-09-17 (round 21): EP is closed too, so routing distribution is what is left

The expert-bytes attribution left two candidate mechanisms. Round 20 closed the skipping one. This
round closes the other.

**Both arms run EP = 1.** vLLM's own `FusedMoEParallelConfig.make` docstring, and it is identical in
both images, says: "When TP = 2, DP(PCP) = 1 and EP = False ... device 0: TP = {2, 0} DP = {1, 0}
**EP = {1, 0}** ... device 1: TP = {2, 1} DP = {1, 0} EP = {1, 0} - Comment: Tensors are sharded across
2 devices." The reference recipe passes no expert-parallel flag and neither do we:
`--enable-expert-parallel` appears nowhere in `scripts/`, `configs/` or `patches/`. So neither arm
splits the expert set across ranks; both shard inside each expert along TP, and every rank holds the
same expert set. Expert parallelism is not the ~31 %.

**So both mechanisms are closed and routing distribution is the only candidate left** - which experts
the router picks, and how many rows land per step. Neither is measured. The engine can be asked:
`enable_return_routed_experts` is **False** in this arm, and flipping it reports the routed expert ids
per request, which turns experts-touched-per-rank from an inference into a measurement. It is a config
change, not a rebuild, and it needs no contact with the reference, so it is a goal item 3 measurement
and the next thing to run.

**Configuration facts recorded along the way**, from `harness/logs/attnfi-engine.log` (an earlier arm,
so the DeepGEMM E8M0 line in it predates the 2026-09-16 footgun fix):

| item | value |
|---|---|
| `tensor_parallel_size` / `data_parallel_size` | 2 / 1 |
| MoE backend | `B12X_MXFP4_MXFP8`, then `Using B12xExperts` |
| `kv_cache_dtype` | `nvfp4_ds_mla`, "Using DeepSeek V4 padded nvfp4_ds_mla KV cache format" |
| `quantization` | `deepseek_v4_fp8` |
| capture sizes | [1, 2, 4, 8, 16, 24, 32, 40, 48] |
| `enable_return_routed_experts` | False |

**Build**: `main-029-proto2` at 27:43 inside step #21, zero error lines, 15-16 `nvcc` still active -
well past the 23:12 point where the first attempt died, so the `libdw-dev` fix is holding. Still no
image.

## 2026-09-17 (round 22): downgrading the expert-byte attribution, and prepping the routing capture

The last two rounds closed both mechanisms that could be tested for the ~31 % expert-byte gap - the
kernel skips and streams at 88 % of the bound, and both arms run EP = 1. That leaves routing
distribution, and a closer look weakens it too: **both arms run the same checkpoint and therefore the
same router**, so for the same token stream they should select nearly the same experts, and a
difference would require the numerics of the two engines to diverge enough to change expert selection.
The decomposition also assumed every layer is MoE and that the step is the target pass alone, so its
per-layer budget is an upper bound and the 28 % / 37 % fractions move once the draft model, attention
and collectives take their share.

**So the 31 % is downgraded to an undecomposed difference in the step**, and `docs/EXPERIMENTS.md`
now says so in the probe verdict. What stands: the step difference itself (176.5 ms against 231.8 ms at
c6 on the 6x-corrected basis), the kernel's efficiency, and EP = 1 on both sides. Settling it needs the
region profiler re-run on the corrected basis, which needs a serve.

**Prepared the routing capture in the meantime**, because it is self-contained and cheap:
`--enable-return-routed-experts` is real (`vllm/engine/arg_utils.py:900-903`, backed by
`ModelConfig.enable_return_routed_experts`, default False), and it adds `routed_experts` to the chat
response as a base64 numpy array - `vllm/entrypoints/openai/chat_completion/protocol.py:119-124` gives
the decode as `np.load(io.BytesIO(base64.b64decode(s)))`. `harness/capture_routing.py` sends one
protocol request and reports distinct experts touched, the load distribution, the max expert load and
the singleton count. Both images support the flag, **but the reference's configuration is read-only by
contract**, so no symmetric comparison - any routing number would be ours alone, which is part of why
this is not the measurement to chase first.

No serve was started this round on purpose: the build holds 18 GiB and a two-node serve wants ~100 GiB
per node, and earlyoom would then be choosing a victim between the build and the engine.

Build: `main-029-proto2` at 29:11 inside step #21, zero error lines, 16 `nvcc` still active.

## 2026-09-17 (round 23): the n=3 profiler puzzle, and it was timing the wrong model

Upstream re-check: all twelve tracked items still open, none merged. `#53055` and `#47988` MERGEABLE,
`#41834` CONFLICTING, the rest UNKNOWN; DeepGEMM `#417`/`#419`/`#403` open. No movement since the last
check, and no upstream write made.

Then this round's find, which closes a question this file has carried since the proto era: **"the
profiler prints `n=3` layer events, not 43, and that why is not established."**

`b12x_profile_decode_once` is applied to the DSpark **draft's** `DFlashSpeculator._run_model`
(`patches/apply_overlays.py:287-308`), and in `patches/files/sm12x_b12x_kernels.py` it does:

```python
_PROFILING_STEP[0] = True
_LAYER_EVENTS[0] = []
...
layer_ms = [a.elapsed_time(b) for a, b in _LAYER_EVENTS[0]]
print(f"b12x layers: n={n} ...")
```

`_LAYER_EVENTS[0]` is the shared list that `b12x_profile_layer` appends to on every one of the target's
43 `DeepseekV4DecoderLayer.forward` calls. The draft hook resets it and prints from it, so the printed
summary is the **draft's** layers - three MTP blocks, hence `n=3` - and the target's 43 events are
discarded before anything reads them.

Second defect in the same function: the docstring says "One-shot timing for the first real DSpark decode
step" but there is no one-shot guard, only the `is_current_stream_capturing()` early return, so on the
eager path it prints on every draft forward, up to `k+1` times per engine step.

**Consequence, and it is retroactive**: the per-layer and per-region shares in `HANDOVER.md` and
`docs/knowledge/05-performance.md` - ffn/MoE 1.84 ms of a 2.83 ms layer, WO 0.60 ms, MLA 0.20 ms, and
the derived "ffn/MoE is 65 % of a target layer" - cannot be assumed to describe the target model. The
`n=3` is the proof. That is a second independent reason those numbers failed to reconcile with the
corrected step basis, and it means the session's remaining attribution has been leaning on an
unverified figure.

**Unaffected**: every direct kernel measurement - the WO and MLA head-to-heads, the FlashInfer version
regression, the skipping curve, the bandwidth probes. None used this profiler.

**Fix, held back on purpose**: separate event lists (or no layer printing in the draft hook) plus the
missing one-shot guard. Small edits in `patches/files/sm12x_b12x_kernels.py`, but the proto2 build is
running from this exact `patches/` tree, so changing it now would desync the build or force another
one. It goes into the next rebuild, followed by a region re-measurement on the corrected basis before
any region percentage is reused.

Build: `main-029-proto2` at 31:10 inside step #21, zero errors, 16 `nvcc`.

## 2026-09-17 (round 24): the profiler fix is written, and the step-time correction cross-checks

**The fix exists now, deliberately unshipped.** `patches/files/sm12x_b12x_kernels.py`:

- `_DRAFT_PHASE` is set around the draft's `_run_model`, and `b12x_profile_layer` routes each `(e0, e1)`
  pair to `_DRAFT_LAYER_EVENTS` or `_LAYER_EVENTS` depending on which model is running, so the two no
  longer share one list.
- The draft hook no longer resets the target's list and no longer prints the target's layers; its two
  prints are relabelled `b12x draft step` / `b12x draft layers` so a log says which model it describes.
- `_DECODE_PROFILED` gives the draft hook the one-shot behaviour its docstring already promised.

It is **not** synced to the rig: the proto2 build is running from this exact `patches/` tree, and editing
it mid-build would either desync the image or force another build. The check when it lands is the profile
print itself - the target summary must read `n=43`, with the draft's three reported separately.

**Second, independent confirmation of the step-time correction.** The correction divides wall time by
`512 / tokens_per_step` rather than by `drafts_per_req`. The c1 and c3 levels cross-check it without any
concurrency factor to get wrong: at c1, `drafts_per_req` 104 and `512 / 4.923` = 104.0 agree exactly
because one sequence is one step; at c3, `drafts_per_req` 325 against `3 x (512 / 4.726)` = 325.0 agrees
the same way. So the counter really is per-request and the corrected basis is sound, which means the
repo's quoted step times were wrong by the concurrency and every per-layer budget derived from them needs
redoing.

Build: `main-029-proto2` at 31:37 inside step #21, zero errors, 16 `nvcc`.

## 2026-09-17 (round 25): the proto2 build is DONE and verified

The phase-1 build that this session opened with, failed once, fixed, and resumed is finished:

```
#27 writing image sha256:b5f2b5fdbf8e... done
#27 naming to docker.io/library/vllm-spark-0731:main-029-proto2 done
#27 DONE 24.8s
sha256:b5f2b5fdbf8e92209b13a79242f343cee0a992761652b1dec3d7df9311df3e37 [vllm-spark-0731:main-029-proto2] arm64 29160565686
built vllm-spark-0731:main-029-proto2
```

Image `vllm-spark-0731:main-029-proto2`, ID `b5f2b5fdbf8e`, 29.2 GB, zero error lines in the log. The
pinned commit is confirmed from inside the image: **`vllm 0.2.1.dev0+gf37c550bf.d20260917`**, i.e.
`f37c550bf` = `proto-v0.2.0`, which is what `configs/pin.main-029.env` says `VLLM_REF` should be. The
runbook's step 1 ("verify the image records f37c550bf635, nothing else is worth doing on a wrong
commit") passes.

Copy to spark2 launched with the tag given explicitly, per the trap recorded in round 12 -
`bash scripts/02-copy-main.sh vllm-spark-0731:main-029-proto2`, never the bare form, because the script
sources `configs/pin.main.env`, whose `IMAGE` is the v0.28-era `main-b12x`.

What this unblocks: every number in `docs/EXPERIMENTS.md` is still proto-v0.1.0-era, and the completion
criterion needs both arms measured on the same rig on the same day. With this image the protocol can be
re-baselined: `bash harness/run-arm.sh <tag> "<env>"` for our arm and `bash ~/goal/launch-refbase.sh`
for the reference, then `~/drive-median.sh` on each.

## 2026-09-17 (round 25 continued): phase 2 exposed four port bugs, all fixed

Serving the freshly built `main-029-proto2` died at startup with

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for SpeculativeConfig
attention_backend
  Value error, Unknown attention backend: 'B12X_MLA_SPARSE'.
```

and the cause is that **the image I served had no overlays in it at all**. `docker/Dockerfile.main` says in
its own header "Phase 1 only: no apply_overlays.py"; the build log runs stages 10/20 to 20/20 with no
overlay step, and inside the image `vllm/utils/sm12x_b12x_kernels.py` and
`models/deepseek_v4/nvidia/dsv4_b12x_sparse.py` are missing - the b12x files present are upstream vLLM's
own. Phase 2 is `scripts/03-apply-main-overlays-029.sh`, which I had not run. My runbook said to serve
the phase-1 image; that is wrong and is now corrected.

Running phase 2 then failed four times, each time on a real defect:

**1. `replace_once` matched a needle inside a deeper-indented line.** The overlay needle
`[16sp]compressor(compressed_kv_score, positions, rotary_emb)` matched the **tail** of the real
`[20sp]compressor(...)` line, because the 16 spaces are the last 16 of that line's 20. It wrote
`[20sp]if not b12x_skip_flag("compressor"):` with its body also at 20 spaces, and phase 2 died on
`IndentationError: expected an indented block after 'if' statement` in `attention.py`. `port_scan.py`
counted the overlay as applied, which is why the port looked clean. **Fixed systemically**:
`replace_once` now matches only at a line boundary (`re.finditer(r"(?m)^" + re.escape(old))`), so this
class of mistake becomes a loud missing-needle failure instead of silent wrong-indent corruption.

**2. The compressor needle could not pin its own indentation.** Its replacement is now anchored on the
guarding `if not skip_compressor:` line, which fixes the indent unambiguously.

**3. The same bug in the lambda variant.** The needle used 12 spaces where the base has 16; anchoring
turned it into a reported missing needle and it was re-indented to 16.

**4. `assert_image.py` had two stale expectations for proto2.** It asserted
`"is_deep_gemm_supported" in inspect.getsource(mhc.mhc_pre_broadcast_tilelang)` and
`"_tilelang_hc_prenorm_gemm" in mhc_src`. The overlay's whole purpose is to *remove* the traced
`is_deep_gemm_supported()` calls and read the static `_USE_DEEP_GEMM` instead, so asserting the string
is present cannot hold once it has done its job - and on this base `mhc_pre_broadcast_tilelang` no
longer contains the call at all, while `_tilelang_hc_prenorm_gemm` does not exist anywhere in the
module (upstream renamed the prenorm path). Replaced with the invariants the overlay actually
guarantees: `_USE_DEEP_GEMM` present, and no bare `is_deep_gemm_supported()` left in the module.

**Verification, in the phase-1 image with the fixed patches mounted:**

```
apply_rc=0
assert_rc=0
image OK (main): b12x importable, moe/linear b12x, fp8_ds_mla + nvfp4_ds_mla, 584B DSV4 page,
               mHC TileLang guard, SM12x kernel guards, DSpark dispatch
```

Also fixed on the way: **spark2's repo was stale** - never synced this session, its pin still said
`IMAGE=main-029-proto`/`VLLM_REF=69db1c26b4`, so the worker ran the old base while the head ran proto2.
Both nodes are now synced and both pins read proto2.

Phase 2 is rebuilt on top of the retagged `main-029-proto2-phase1` base. The profiler fix from round 24
is included, since phase 2 copies `patches/files/`.

## 2026-09-17: proto2 serves again, and the first protocol comparison on it

### The blocker was a stale overlay, not the new base

`vllm-spark-0731:main-029-proto2` died during `_initialize_kv_caches` with

```
Assertion error (/opt/vllm/.deps/deepgemm-src/csrc/utils/layout.hpp:113):
sf.size(-2) == ceil_div(mn, gran_mn)
```

and `Worker proc VllmWorker-0 died unexpectedly` / `Engine core initialization failed`. The stack is
`attention.py:532` -> `nvidia/flashinfer_sparse.py:585 _o_proj` -> `nvidia/ops/o_proj.py:116
deep_gemm_fp8_o_proj` -> `utils/deep_gemm.py:501 fp8_einsum`.

**Root cause: `patch_einsum_sm12x_recipe` in `patches/apply_overlays.py`, applied by `apply_main`.**
It rewrites `compute_fp8_einsum_recipe` so SM12x returns `(1, 128, 128)` with
`tma_aligned_scales=False` instead of upstream's `(1, 1, block_size)` with
`tma_aligned_scales=True`. Upstream's value is the packed UE8M0 layout the SM120 einsum kernel reads;
the override feeds it SM90-style FP32 scales, and DeepGEMM refuses the call.

This was already decided by our own measurement. The DeepGEMM #447 work in this file says the DSv4
o_proj shape (`fp8_einsum("bhr,hdr->bhd")`, h=8, r=1024, d=4096) is **accepted at T=10 and T=256 with
packed ue8m0 scales at `recipe=(1,1,128)`**, diff 7.1e-4 and no NaN, and that the `layout.hpp` rejection
at `recipe=(1,128,128)` with pre-packed per-row scales "is correct by design, not a defect". The
overlay's recorded reason -- "the Python `fp8_einsum` fallback then did `scale.to(float32)` on those
packed ints" -- no longer exists: the fallback is not applied by `apply_main` (the v0.28.0 note above
says so), and `utils/deep_gemm.py` binds `_fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)`, the
real kernel.

**Fix, verified by mount before any rebuild:** drop the override from `apply_main`, which is now
`# patch_einsum_sm12x_recipe(vllm)  # see above: breaks engine startup` with the corrected reasoning
inline, and `patch_einsum_sm12x_recipe`'s docstring now says **DO NOT apply this on proto-v0.2.0 or
later**. Verified by bind-mounting a corrected `o_proj.py` over the image's copy on both nodes: the
`layout.hpp` assertion is gone from the engine log (0 occurrences, was 1 per boot) and the engine
reaches `_check_enough_kv_cache_memory`, i.e. the whole model forward completed.

Also worth recording: the two images are **different upstream bases**. Ours is
`0.2.1.dev0+gf37c550bf.d20260917` with vLLM as an editable install at `/opt/vllm` and a JIT
DeepGEMM under `/opt/vllm/.deps/deepgemm-src`; the reference `0.1.1` is
`0.25.2.dev0+g752a3a504.d20260714` at `/usr/local/lib/python3.12/dist-packages/vllm` with **no**
`.deps/deepgemm-src`. So a ported overlay that was correct on one is not evidence about the other.

### One capacity difference, forced

With `GPU_MEMORY_UTILIZATION=0.8389` proto2 reports 9.17 GiB of KV cache memory available and refuses
`MAX_MODEL_LEN=65536`, which needs 9.48 GiB (`_check_enough_kv_cache_memory`, estimated max model
length 24204). `0.86` serves. The reference recipe runs `0.82` with `max_model_len: 262144`, so this is
our arm's own larger non-KV footprint on this base, not a workload change: at c6 with 512-token prompts
the KV pool is not the constraint, and the only knob changed is the allocation.

### Both arms, same rig, same day

`harness/run-arm.sh proto2-recipe2` (ours, `main-029` stack, k=7, capture 48, async, util 0.86) and
`harness/run-refg.sh` (reference image, its own recipe: k=7, capture 48, async, util 0.82), each
`drive-median.sh <tag> 3 512 1 3 5 6`:

| level | ours median agg tok/s | spread | reference median agg tok/s | spread |
|---|---|---|---|---|
| c1 | 34.8 | 9.2% | 66.4 | 6.0% |
| c3 | 74.9 | 1.7% | 118.4 | 5.7% |
| c5 | 100.8 | 7.3% | 139.0 | 26.0% |
| c6 | **117.0** | 0.3% | **161.0** | 9.3% |
| c1+c3+c5+c6 | **327.5** | | **484.8** | |

All six passes passed `gate_france` and `gate_9x8` in both arms. Acceptance is comparable (51.1-54.5%
ours, 53.7% reference) and so is tokens per step (4.571 ours at c6, 4.733 reference), so the deficit is
step time, not speculation quality.

**The goal's criterion is not met.** We are 27.3% below the reference at c6 and 32.5% below on the
sum, both far outside the larger spread (26.0%, the reference's own c5).

### proto2 is throughput-neutral against the proto base

Converting the reference's c6 to step time (`tokens_per_step / (agg/6)`) gives 176.6 ms, which matches
the 176.5 ms recorded earlier; ours is 234.4 ms against the earlier 231.8 ms. So the port to
`proto-v0.2.0` changed nothing measurable in either direction, and the 27-33% gap is the same gap this
file has been attributing to a ~31% excess of expert bytes streamed per layer (0.916 GB/layer reference
vs 1.202 GB/layer ours at 28.4% / 37.3%). That attribution still stands as an undecomposed difference:
both mechanisms tested against it (MoE kernel sharding and expert skipping) came back negative.

The c1 column is the sharpest new clue: our step is ~2x the reference's at batch 8 (136 ms vs 71 ms)
but only ~1.33x at batch 48, which points at a large per-step fixed cost rather than a scaling one.
The next measurement is a region profile on proto2 with the corrected profiler, which is included in
this phase 2 and whose target summary had been reporting the draft model (n=3) rather than the target
(n=43).

### The profiler fix, first check on the rebuilt image

`VLLM_PROFILE_DECODE=1` on the rebuilt `main-029-proto2`, one 64-token chat request:

```
(Worker_TP0) b12x draft step: toks=42 wall=174.0ms gpu=174.0ms overhead=0.0ms
(Worker_TP0) b12x draft layers: n=3 sum=170.3ms avg=56.76ms max=148.49ms@L0 p95=14.69ms
(Worker_TP0) b12x step execute_model tok=48: wall=164.2ms gpu=177.0ms gap=1.5ms
(Worker_TP0) b12x step sample_tokens: wall=134.7ms gpu=123.3ms gap=-114.2ms
(Worker_TP0) b12x step execute_model tok=98: wall=220.4ms gpu=284.9ms gap=11918.1ms
```

The draft is now labelled as the draft: `b12x draft layers: n=3`, on its own list, exactly what the fix
was for -- the target's 43 events were being reported as the draft's summary.

The **target** summary (`b12x layers: n=43`) did not print at all. `_b12x_print_step_profile` calls
`_b12x_print_layers()`, and that returns early when `_LAYER_EVENTS[0]` is empty, so the target list was
empty at print time even though `execute_model` had just run 43 decoder layers inside the profiled
step. So the split works but the target half is still unverified, and the region shares still cannot be
read off this profiler. Not fixed this round; the next check is to print the target list where it is
still populated instead of after `sample_tokens`.

Two independent observations from the same log, worth keeping: `sample_tokens` costs 123-213 ms of GPU
time per step, comparable to an entire `execute_model`, and the prefill steps carry host-side `gap`
values of 8.6 s and 11.9 s. Neither has been attributed yet.

## 2026-09-17 (round 26): the target breakdown finally prints, and the region table is not additive

### The pending profiler check is closed, with a condition

`b12x layers: n=43 sum=148.0ms avg=3.44ms` now prints -- but only under `--enforce-eager`. The reason
it never printed on a normal arm is structural, not a bug in the fix: with `FULL_AND_PIECEWISE` CUDA
graphs a decode step *replays a captured graph*, so the Python-level `b12x_profile_layer` wrapper never
runs and `_LAYER_EVENTS[0]` is empty when `_b12x_print_step_profile()` calls `_b12x_print_layers()`.
The module's own docstring already said this ("On a run without CUDA graphs the layer decorator collects
as well"), and `VLLM_PROFILE_CAPTURE=1` is the existing mechanism for graph-replayed work. So the
target breakdown is obtainable two ways, and neither was being used: eager mode, or the capture-armed
region marks.

The layers are **flat**: the slowest is L2 at 4.61 ms against a 3.44 ms average, and the top twelve are
all 4.13-4.61 ms. No layer family is the gap at this batch.

### The region table must not be summed

Same arm, same flags, `skip_indexer_all` armed through `VLLM_SKIP_FLAG_DIR`:

| region | control n | control gpu_sum | skip_indexer_all gpu_sum |
|---|---|---|---|
| attn | 43 | 80.3 ms | 49.0 ms |
| ffn | 43 | 65.2 ms | **86.2 ms** |
| indexer | 21 | 52.0 ms | **0.0 ms** |
| mla | 43 | 2.0 ms | 16.7 ms |
| wo_b12x | 43 | 17.4 ms | 22.4 ms |
| allreduce | 87 | 5.7 ms | 8.0 ms |
| **layers (enclosing)** | 43 | **148.0 ms** | **137.4 ms** |

Removing the indexer entirely takes its region to zero while `ffn` **rises** by 21 ms and the enclosing
layer total falls only 10.6 ms. The marks record CUDA events on a stream that is still draining earlier
queued work, and nothing synchronizes between them, so `gpu_sum` is queue-drain time rather than that
region's compute. The indexer's true net cost is therefore **~10.6 ms per step, about 7 %** of this
step, not the 52 ms (35 %) the raw table suggests. The A/B cannot be pushed further either: `mla` rises
2.0 -> 16.7 ms because a garbage index changes which keys the sparse MLA attends to.

**This invalidates every region percentage in this repo**, including the "ffn/MoE is 65 %" figure the
remaining attribution leaned on -- for a second, independent reason beyond the draft/target mix-up.
Only the enclosing per-layer total is additive, and it is flat across the 43 layers.

### Where the gap actually is, in step time

From the two same-day median logs, with ~4.6 tokens per step on both arms:

| level | ours | reference | ratio |
|---|---|---|---|
| c1 | 130 ms | 69 ms | 1.88x |
| c3 | 175 ms | 117 ms | 1.50x |
| c5 | 223 ms | 165 ms | 1.35x |
| c6 | 236 ms | 171 ms | 1.38x |

The penalty is ~61 ms per step at c1 and ~65 ms at c6: a large **fixed** per-step cost that barely
grows with batch, spread evenly over all 43 layers. That is the signature of a per-layer constant, not
of one hot kernel -- consistent with the flat layer profile and with both direct kernel head-to-heads,
which found our own MoE and MLA kernels *faster* than the alternatives, not slower.

### `vllm._deepselect_C` is a dead end (negative)

Both images log `Failed to import the DeepSelect extension (vllm._deepselect_C)` at boot, which looks
like a missing fast top-k path. The reference image is missing it too, and this repo contains no
reference to `deepselect` anywhere. It explains nothing and there is nothing to build.

### Three eliminations, and the lead that was already sitting in a saved log

**The all-reduce backend is the same.** Both engines log `Using ['PYNCCL'] all-reduce backends ... for
group 'tp:0'`, and both log `SymmMemCommunicator: Device capability 12.1 not supported`. Our arm
measures 87 all-reduce calls per step (2 per layer), which looked like a good candidate for a
batch-independent cost, but the transport is not a difference between the arms.

**Turning the CUDA graphs off is worse, not better.** `proto2-eager` (`--enforce-eager`,
`INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.8`) measured 28.4 / 68.6 / 98.2 / 112.9, sum 308.1, against
`proto2`'s 35.4 / 78.9 / 103.0 / 117.0, sum 334.3. Our `patch_tp_allreduce_eager_break` splits the
piecewise graph once per layer, so "the split is the fixed cost" was a good hypothesis -- the graphs
still save more than the breaks cost, and the deficit is *larger* at c1 with graphs off, which means part
of the fixed cost is host-side and already hidden.

**`vllm._deepselect_C` is missing in both images**, so the fast top-k extension cannot explain anything.

**The lead: our b12x MoE has no kernel for the band decode actually runs in, and the reference's does.**
`outputs/driver/one-off/moe-tuning-coverage.log` has been on disk since the MoE sweep:

| | ours (`b12x 1.2.6`) | reference (`b12x 0.15.3`) |
|---|---|---|
| decode policies | `dynamic`, `dynamic_w4a8_decode`, `micro` | `dynamic`, `micro`, **`static`** |
| cheap-policy coverage | micro ends at 20 routed rows | **static covers 20 < rows <= 640** |
| 48 routed rows (= c1) | 188 | **149** |
| 144 (= c3) | 188 | ~130-166 |
| 240 (= c5) | 188 | ~130-166 |
| 288 (= c6) | 188 | ~141-175 |

`routed_rows = q_rows * topk`, `q_rows = c * (k+1)`, topk 6, so the protocol's whole range is 48-288
routed rows -- inside the reference's `static` band and outside our `micro` band, leaving us on a flat
`dynamic` at 188. Where both libraries have a policy (`micro`, at 8-20 rows) the times are identical, so
this is a missing kernel, not a tuning difference. The two images ship different generations of the same
library: ours `b12x 1.2.6` with no `b12x.moe.tuning` module and no `static` symbol under
`b12x.moe.fused_moe`; the reference `b12x 0.15.3`.

That is goal item 4's trigger condition, met on the MoE half. The first instance was already built --
`vllm-spark-0731:main-029-proto-b12x015` puts the reference's b12x under our stack -- and it died only
for an infrastructure reason: the image existed on spark1 alone, so spark2's worker could not start and
the head blocked on rank 1. It is being copied to spark2 now, and that arm is the next measurement.

### A second boot trap, and the knob for it

`--enforce-eager` changes vLLM's memory profiling enough that InstantTensor cannot size its read
arena: `buffer_size (1059061760 B) exceeds device memory budget (757221376 B)`, twice, on both tries,
even with `buff/cache` at 1 GiB and 115 GiB free host DRAM. **Dropping the page cache is not always
sufficient**, because the budget comes from the *device* pool, which on GB10 is not the host's free
DRAM. The knob is `INSTANTTENSOR_MAX_FREE_MEM_USAGE` (or an explicit `INSTANTTENSOR_BUFFER_SIZE`);
setting it to 0.8 booted the same arm immediately. Other knobs found in the bytecode:
`INSTANTTENSOR_BACKEND`, `INSTANTTENSOR_CHUNK_SIZE`, `INSTANTTENSOR_CONCURRENCY`, `INSTANTTENSOR_IO_DEPTH`,
`INSTANTTENSOR_DEBUG`.

### The rule: current base, current library, patches on top

`proto-b12x015` and `proto-b12x015-linauto` were run and both died at worker init with
`ValueError: Failed to find a kernel that can implement the ScaledMM linear layer`, and with
`LINEAR_BACKEND=auto` too. `b12x 0.15.3` -- the generation the reference image ships -- does not provide
the ScaledMM linear that `LINEAR_BACKEND=b12x` resolves to, so the two generations are not drop-in. The
image `vllm-spark-0731:main-029-proto-b12x015` had sat on spark1 since the old sweep precisely because
nobody had copied it to spark2, and its stale copy to spark2 was the only reason that arm existed.

**Both arms are retired, and so is the direction.** The working rule for this repo is: the current base
(`proto-v0.2.0`), the current `b12x 1.2.6`, and patches on top. Recovering one kernel by regressing the
whole library is not a lever. The right question is which fast paths the *current* library already ships
and our stack is not using, and there is a large, unstripped surface for that: `b12x 1.2.6` reads 90-plus
`B12X_*` flags.

### The strongest unexplored lever: `b12x`'s own PCIe all-reduce

Our engine selects `['PYNCCL']` for `tp:0`, and the profiler counts **87 all-reduce calls per step** (two
per decoder layer). `vllm/distributed/communication_op.py` does not use b12x's communication stack at all
-- the only `b12x` string in that file is our own `b12x_profile_region_fn` import.

Meanwhile `b12x 1.2.6` ships a complete PCIe all-reduce built for exactly this hardware:

- `b12x/comm/pcie/`: `pcie_allreduce.py`, `pcie_oneshot.py`, `pcie_twoshot.py`,
  `pcie_hierarchical.py`, `pcie_dma_kernels`-backed `_oneshot_cute.py`, plus `pcie_dcp_a2a.py`
- `PCIeAllReduce.from_exchange_group(exchange_group=<ProcessGroup>, device=..., eager_buffer_bytes=...,
  max_size=..., single_channel=..., max_concurrent_channels=...)`
- `PCIeAllReduce.all_reduce(inp, *, out=None, peer_input_ptrs=None, blocks=None, stream=None,
  channel_id=None)`
- `_algorithm_for_world_size`: worlds in `DIRECT_WORLD_SIZES` get the `oneshot` runtime, described in the
  class docstring as "Worlds through TP8 use the low-latency all-peer oneshot runtime". TP=2 is the
  smallest such world.
- knobs: `B12X_PCIE_ALLREDUCE_ALGORITHM` (`auto|hierarchical|island_rs`), `B12X_PCIE_ONESHOT_THREADS`,
  `B12X_PCIE_ONESHOT_BLOCK_LIMIT`, `B12X_PCIE_ONESHOT_PUSH`, `B12X_PCIE_TP2_REMOTE_PUSH`,
  `B12X_PCIE_HIERARCHICAL_*`, `B12X_PCIE_ISLAND_RS_*`

This matters because of what our current transport costs. The all-reduce is the one thing the profiler
counts **87 times per step**, our step carries a fixed ~61 ms penalty that barely grows with batch, and
our own overlay has to **break it out of the CUDA graph** because "NCCL on GB10 cannot use CUDA-graph
pool pointers" -- replay of an in-graph PYNCCL all-reduce produced 1e33 logits. b12x's PCIe path is
written for this topology and this buffer model, and wiring it in is a patch on top of the current
library rather than a library change.

Nothing in this repo has ever referenced `PCIeAllReduce`, `pcie_allreduce` or `B12X_PCIE*` before this
entry. That is the next measurement: replace `tensor_model_parallel_all_reduce`'s PYNCCL calls with
`PCIeAllReduce` constructed from the TP process group, as an overlay, and meter it.

### Other unused fast paths in the current library, for the record

`B12X_W4A8_TINY_DECODE`, `B12X_W4A16_SMALL_M_DIRECT`, `B12X_W4A16_SMALL_M_SPLITK`, `B12X_MOE_TILE_MN`,
`B12X_MOE_WARM_MS` (MoE small-batch); `B12X_FUSED_INDEXER`, `B12X_INDEXER_DIRECT_K`,
`B12X_PAGED_INDEX_SUPERTILE_K`, `B12X_PAGED_DECODE_GRAPH_CHUNK_PAGES` (the indexer, priced at ~10.6 ms
per step); `B12X_TURBO_ATTN`, `B12X_MLA_SM120_PREFILL_*`, `B12X_PAGED_MSA` (attention);
`B12X_DYNAMIC_TILE_MN`, `B12X_DYNAMIC_SWAP_AB` (the `dynamic` MoE backend our decode runs on).

## 2026-09-17 (round 27): b12x's PCIe all-reduce is intra-host, so it cannot carry our TP

The lead from round 26 was that b12x 1.2.6 ships a low-latency all-peer collective for PCIe fabrics and
that our stack, which issues **87 all-reduce calls per step**, never uses it. It was wired in as an
overlay on `tensor_model_parallel_all_reduce`: the transport swapped, the surrounding control flow
(static workspace, graph break) untouched, with a guarded lazy construction and a fallback to PYNCCL.

Three things were learned, in order.

**1. The wrapper's predicate is broken in this build.**
`PCIeAllReduce.should_allreduce` delegates to `PCIeAllReduce._runtime`, but for the oneshot algorithm
that is a `PCIeOneshotAllReducePool`, and `should_allreduce` is defined on the *channel* class, not the
pool -- grep shows it only at `PCIeOneshotAllReduce` (class at 1821), while the pool (class at 3142)
defines `from_exchange_group`, `prepare_channels`, `for_stream`, `prepare_graph_all_reduce`,
`all_reduce`. So the documented guard raises
`AttributeError("'PCIeOneshotAllReducePool' object has no attribute 'should_allreduce'")`. The predicate
was reimplemented locally against the channel's own rules instead (device, dtype in
`(fp16, bf16, fp32)`, bytes <= `DEFAULT_MAX_SIZE` (8 MiB), bytes % 16 == 0, weak-contiguity).

**2. Eager use needs `single_channel=True`.** With the default `single_channel=False` every call raises
`RuntimeError('distributed PCIe oneshot eager use requires an explicit semantic channel_id shared by
every rank')`. `single_channel=True` makes the pool prepare `_SINGLE_CHANNEL_ID` itself and resolve
`channel_id=None` to it.

**3. It cannot work across two hosts at all.** With those two fixed, construction succeeds --
`TP all-reduce via b12x PCIe oneshot (algorithm=oneshot)` -- and then the first collective fails:

```
b12x PCIe all-reduce unavailable: RuntimeError('PCIe shared buffer CUDA IPC import open failed:
  group rank 0: RuntimeError: failed to open CUDA IPC handle for peer group rank 1;
  group rank 1: RuntimeError: failed to open CUDA IPC handle for peer group rank 0')
```

The runtime is CUDA-IPC based (`_RetryableIPCExport`, `IPC_SLAB_ALIGNMENT`, the "shared buffer CUDA IPC
import" path). CUDA IPC handles are host-local and cannot be opened across nodes. `b12x/comm` contains
exactly one module, and its own docstring says what it is for: "``pcie``: collectives for consumer PCIe
fabrics (no NVLink) -- one-shot and DMA/CE-ring all-reduce". That is an **intra-host** multi-GPU
transport, like a workstation with several consumer cards. Our `TP=2` spans spark1 and spark2, so this
lever does not apply, and b12x ships no inter-node alternative.

**Recorded as a decisive negative.** The 87 per-step all-reduces stay on PYNCCL, and the "fixed per-step
cost" search must move on-node. The patch was never baked into the image; it lived only as a bind-mount,
and it is not in `apply_overlays.py`.

### The B12X_* flag surface, surveyed

Round 26 noted that `b12x` reads 90-plus `B12X_*` flags and our arm sets none of them. Surveyed against
their defaults, they are mostly already at the useful value, which closes that avenue too:

| flag | default | note |
|---|---|---|
| `B12X_W4A8_TINY_DECODE` | **on** | "Default on; B12X_W4A8_TINY_DECODE=0 is the kill switch" |
| `B12X_FUSED_INDEXER` | **on** | |
| `B12X_INDEXER_DIRECT_K` | **on** | default `1`; `0` is a "triage kill-switch" |
| `B12X_PAGED_MSA` | **on** | the env is only a `=0` disable |
| `B12X_PAGED_INDEX_SUPERTILE_K` | 32768 | already large |
| `B12X_W4A16_SMALL_M_DIRECT` | on | W4A16, not our `MXFP4_MXFP8` MoE |
| `B12X_W4A16_SMALL_M_SPLITK` | off | W4A16 path, not ours |
| `B12X_TURBO_ATTN` | off | gated on `plan.kv_dtype == torch.float8_e4m3fn` in the generic **paged** forward; our attention is `B12X_MLA_SPARSE` |

Still genuinely unset and therefore testable, all MoE tile shape overrides that the library documents as
benchmarking knobs: `B12X_MOE_TILE_MN` (`"64x128"` syntax), `B12X_DYNAMIC_TILE_MN`, `B12X_DYNAMIC_SWAP_AB`.
Those are the next measurement, and they are single-node microbenchmarks rather than full arms.

### Correction: we are already on the newest b12x

`configs/pin.main-029.env` records `B12X_REF=3a437ab5168060e4d625f05e1625c04089f1ba37` with the finding
that this commit *is* the 1.3.0 release state -- "the newest before PyPI published b12x 1.3.0 at
17:20:50Z that day" -- and that `B12X_VERSION=1.2.6` is only what the tree's own version string says.
So the image already runs the newest b12x code, and the round-26 framing of "ours 1.2.6 against the
reference's 0.15.3" is a version-string comparison, not a newness one.

## 2026-09-17 (round 28): the gap is in the target's 43 layers, and the reference's kernel choices

### The step split, with CUDA graphs on

The step profiler wraps the model runner rather than the per-layer decorator, so unlike the `b12x
layers` breakdown it works under graph replay. One `VLLM_PROFILE_DECODE=1` boot of `proto2`, six
concurrent 512-token requests, GPU ms:

| step | `execute_model` (43 target layers) | `sample_tokens` (draft + sampling) |
|---|---|---|
| tok=1 | **105.5** | 30.6 |
| tok=2 | 109.4 | 34.1 |
| tok=8 | 139.9 | 30.8 |
| tok=9 | 164.1 | 64.5 |
| tok=48 | 185.8 | 136.4 |

Two things follow. **The 43 target layers alone cost ~105 ms at batch 1, which already exceeds the
reference's entire c1 step (69 ms)** -- so the gap is in the per-layer work of the target model, not in
DSpark, sampling or the scheduler. And `sample_tokens` has a ~30 ms floor that is nearly
batch-independent, while the draft inside it (`sample`) grows with batch. Earlier one-shot prints of
`b12x draft step: toks=42 wall=174.0ms` are first-call artifacts: `_DECODE_PROFILED` is one-shot, so that
number includes cold-start costs and must not be read as steady-state draft cost.

### What the reference engine actually selects

Read off `~/refbase-run.log`, which is the only place the reference's own choices are visible:

```
Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod        ours: B12xFp8BlockScaledMMKernel
Using 'B12X_MXFP4' Mxfp4 MoE backend                               ours: B12X_MXFP4_MXFP8
DSA indexer decode path: use_flattening=True (use_fp4_indexer_cache=False)
Warming up DeepSeek V4 sparse MLA attention for mixed tokens=16
Autotuning FlashInfer SM120 sparse MLA DSv4 decode ... flashinfer_autotune_cache/0.6.15/...
  Config cache hit for sparse_mla_sm120_decode_dsv4 (runner=SparseMlaDecodeV3Runner)
kernel_config=... moe_backend='flashinfer_b12x', linear_backend='auto'
disable_custom_all_reduce=True                                      ours: False
```

Three differences that matter, in order of how directly they are a per-layer cost:

1. **The linear kernel.** Ours runs `B12xFp8BlockScaledMMKernel`, the reference `DeepGemmFp8BlockScaledMMKernel`.
   `_POSSIBLE_FP8_BLOCK_KERNELS` orders CUDA as `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`,
   `DeepGemmFp8BlockScaledMMKernel`, `CutlassFp8BlockScaledMMKernel`, then `B12xFp8BlockScaledMMKernel`.
   `LINEAR_BACKEND=deep_gemm` is a valid value and filters to exactly the reference's kernel.
2. **The attention kernel.** Ours is `B12X_MLA_SPARSE`; the reference runs FlashInfer's SM120 sparse MLA
   DSv4 decode with a **0.6.15 autotune cache**, i.e. a tuned `SparseMlaDecodeV3Runner`.
3. **`disable_custom_all_reduce`.** The reference sets it True, our arm leaves it False.

### The linear family is not the gap (negative)

`proto2-dglin` (`LINEAR_BACKEND=deep_gemm`) booted, selected `DeepGemmFp8BlockScaledMMKernel` -- the
reference's own kernel -- and passed both gates, but measured 30.1 / 76.9 / 106.7 / 114.6, sum **328.3**,
against `proto2`'s 35.4 / 78.9 / 103.0 / 117.0, sum 334.3. It is no better on the sum and dramatically
less stable: a c6 spread of **48.6 %**, with one pass falling to 62.9, against `proto2`'s 0.3 %. Our own
b12x block-scaled MM is therefore not what is costing us, and the reference's linear kernel does not
transfer at a profit. This also retires the old `dglinear` entry ("numerically dead" on the proto base)
with a real number rather than a shrug.

### The attention family is not the gap either (two negatives)

`ATTENTION_BACKEND=DRAFT_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4` -- the attention the reference
actually runs -- measured 35.4 / 73.6 / 101.9 / 117.2, sum **328.1** against `proto2`'s 334.3. But that
run was handicapped: the reference loads a **pre-built** autotune config (`Config cache hit ... 
source=config file`, `flashinfer_autotune_cache/0.6.15/...`), while our boot logged

```
WARNING [flashinfer_sparse_mla_warmup.py:203] No FlashInfer SM120 sparse MLA DSv4 decode autotune
cache entries found. Falling back to FlashInfer's default tactic heuristic.
```

so it ran FlashInfer's heuristic rather than a tuned tactic. The boot nonetheless wrote entries under
our own key (`0.7.0/121a/cb463701b306028054d41425138d6fbede4833ca1889c34046671610916e2d0d`), which makes
the re-run a clean one-variable test of "cache populated vs not": same image, same env.

The warmed run loaded it (`autotune cache loaded on rank 0`) and measured 37.5 / 71.1 / 99.3 / 113.0,
sum **320.9** -- slightly worse on the sum than the cold run and clearly below `proto2`. The one real
gain is stability (spreads 3.2 / 4.4 / 2.3 / 2.0 % against 7.3 / 5.8 / 4.0 / 7.0 %), which is worth
remembering. **Both FlashInfer configurations lose to `B12X_MLA_SPARSE`, so the attention family is not
the gap, and the reference's tuned FlashInfer tactic does not transfer to our stack.**

### The last family difference: MoE activation precision

One difference from the reference's boot log has not been tested:

```
reference: Using 'B12X_MXFP4' Mxfp4 MoE backend
ours:      Using 'B12X_MXFP4_MXFP8' Mxfp4 MoE backend
```

Our pin sets `B12X_MOE_FORCE_A8=1`, i.e. FP8 activations against the reference's FP4. That is the last
layer-family difference, and the MoE is the region the layer profile weights most heavily. The arm is
`B12X_MOE_FORCE_A8=0` on top of `proto2` -- one variable, and it is the next measurement.

### The MoE family is a wash, and the last structural difference is inert

`B12X_MOE_FORCE_A8=0` on top of `proto2` measured 36.6 / 79.0 / 104.1 / 114.6, sum **334.3** -- the same
sum as `proto2` to the decimal, with every level inside the other arm's spread and acceptance unchanged
(50.9-55.7 % against 51-54 %). The engine still logged `Using 'B12X_MXFP4_MXFP8' Mxfp4 MoE backend`, so
the flag picks a kernel inside that backend rather than the backend itself, and it buys nothing.

**All three layer families are now eliminated against the reference's own choices**: linear
(`LINEAR_BACKEND=deep_gemm`, the reference's kernel, worse and unstable), attention (FlashInfer
`FLASHINFER_MLA_SPARSE_DSV4`, both cold and with its autotune cache populated, worse), and MoE
(FP4 activations, a wash). Our own combination -- `B12x` block-scaled MM, `B12X_MLA_SPARSE`, `b12x` MoE --
remains the best measured configuration at sum 334.3.

The remaining structural difference in the two engine configs is compilation, and it is **inert**. The
reference's config reads `'mode': <CompilationMode.VLLM_COMPILE: 3>` and `ir_enable_torch_wrap: True`,
ours reads `CompilationMode.NONE` (from `VLLM_USE_AOT_COMPILE=1`), but the reference's own log settles it:

```
WARNING [vllm.py:2270] `torch.compile` is turned on, but the model
/cache/huggingface/hub/.../7872f01b1d1fe23eabc4c98b48bffcef5a386062 does not support it.
```

vLLM skips compilation for this model on the reference too, so both arms execute uncompiled. This
confirms the standing "do not re-litigate torch.compile" note rather than overturning it, and it was
worth re-checking because the config strings alone suggest the opposite.

### What round 28 leaves

The gap is ~61 ms per step, batch-independent, spread flat across 43 layers, and it is **inside the
target model's per-layer work** -- the 43 layers alone cost ~105 ms GPU at batch 1 against a 69 ms total
c1 step for the reference. It is not the draft, not sampling, not the scheduler, not the linear family,
not the attention family, not the MoE family, not the all-reduce transport, not CUDA graphs, and not
compilation. Every one of those is now closed with a measurement rather than an argument.

## 2026-09-17 (round 29): a harness trap that silently voided arms, and the MoE format finally tested

### `05-serve.sh` forwards an explicit env list, so new host env vars vanish

`05-serve.sh` builds its `docker run` with a hand-written `-e` list (about twenty names) plus
`SERVE_EXTRA_ENV`. A variable exported on the host but absent from that list **never reaches the
container**, and nothing warns. This bit the A16 arm twice:

```
$ docker exec vllm-ds4-0731 env | grep FORCE_A16      # nothing
$ docker exec vllm-ds4-0731 python3 -c 'import vllm.envs as e; print(e.VLLM_B12X_MOE_FP4_FORCE_A16)'
False
```

while `B12X_MOE_FORCE_A8=1` *was* present, because that one is in the list. Two consequences.

**First, a correction.** `proto2-moea4` in round 28 set `B12X_MOE_FORCE_A8=0` and reported a wash. The
env var did reach the container, but `B12X_MOE_FORCE_A8` **is read by no code in vLLM or b12x**
(`grep -rn B12X_MOE_FORCE_A8 /opt/vllm /usr/local/.../b12x` finds nothing outside the serve script and the
pin). So that arm was a no-op by construction, not a measurement of the activation format, and its
"wash, and it closes the last layer family" verdict was wrong. The ledger note is corrected.

**Second, a fix.** `VLLM_B12X_MOE_FP4_FORCE_A16` was missing from the list, so the arm meant to set it
was silently running stock `proto2`. The variable is now forwarded alongside `B12X_MOE_FORCE_A8`, with
the same empty default. **Any arm that introduces a new environment knob must either add it there or use
`SERVE_EXTRA_ENV`**; the CLI-carried knobs (`MOE_BACKEND`, `LINEAR_BACKEND`, `ATTENTION_BACKEND`) are
unaffected because `05-serve.sh` passes them as vLLM arguments rather than as environment.

Also recorded: `B12X_MOE_FORCE_A8` is a dead knob in our own pin, which is goal item 3's shape -- a
configuration bottleneck in our repo -- and it should either be removed or wired.

### The MoE activation format, tested properly (negative)

The reference logs `Using 'B12X_MXFP4' Mxfp4 MoE backend`, our default logs `B12X_MXFP4_MXFP8`. In this
base the enum has exactly two b12x variants -- `B12X_BACKENDS = (B12X_MXFP4_MXFP8, B12X_MXFP4_BF16)` --
and `_get_requested_backends` says outright that "W4A8 is the high-throughput b12x path and is preferred
when the model does not request an activation format", with `VLLM_B12X_MOE_FP4_FORCE_A16` as the
supported override.

`proto2-a16` with `SERVE_EXTRA_ENV=VLLM_B12X_MOE_FP4_FORCE_A16=1` really did switch it:

```
Using 'B12X_MXFP4_BF16' Mxfp4 MoE backend.
```

and measured 31.6 / 76.8 / 106.2 / 119.4, sum **334.0** against `proto2`'s 334.3 -- the same sum, with
spreads of 22.5 / 12.8 / **40.9** / **49.5 %** and one c6 pass falling to 63.0. **Negative.** Our default
W4A8 path is not the gap, and the reference's BF16-activation format is not transferable at a profit.

### Standing after round 29

Every layer family is now closed **with the reference's own configuration actually applied**, not merely
argued: linear (`deep_gemm`, worse), attention (FlashInfer, cold and autotune-warm, worse), MoE
activation format (`B12X_MXFP4_BF16`, wash, unstable). Combined with the earlier closures -- all-reduce
transport, CUDA graphs, compilation, the draft, sampling -- the ~61 ms/step, flat across 43 layers, is in
the per-layer work of the target model and is not attributable to any single kernel family we can reach
by configuration.

## 2026-09-17 (round 30): the kernel-level composition, from a real torch profile

The region marks were useless for attribution (round 26 proved they measure queue-drain time), so this
round used vLLM's own torch profiler. Our base exposes
`--profiler-config {"profiler":"torch","torch_profiler_dir":...}` plus `POST /start_profile` and
`POST /stop_profile`, so no rebuild is needed. Run under `--enforce-eager` so nothing hides inside a
replayed CUDA graph; the composition is what matters, not the absolute time.

```
108493 kernel events, 117 distinct kernels, 6951 ms total CUDA time over the captured window

        ms       %    calls  kernel
   2088.22   30.0%     7140  at::native::elementwise_kernel<128,4,...gpu_kernel_impl_nocast<...direct...
   1490.73   21.4%     3196  ncclDevKernel_AllReduce_bf16_RING
   1419.60   20.4%     1564  kernel_cutlass...b12xmoe...siluMoEDynamicKernelSilu...
    285.01    4.1%     1521  nvjet_sm121_tst_mma_112x64x64_4_112x16x64_tmaAB_alignCD4_bz_NNNN
    252.01    3.6%     3567  cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2_tn_align8
    205.05    2.9%     3042  kernel_cutlass...b12x_libdense_gemmDenseGemmKernel...
     65.64    0.9%      714  ...tensorptrf32gmemalign16o2048...
     60.52    0.9%     2975  mhc_fused_tilelang_kernel
     57.88    0.8%     7752  per_token_group_quant_8bit_kernel<BFloat16, Float8_e4m3fn,...
     37.53    0.5%     1353  ...b12xattention_sharedmlakernelUnifiedDecodeKernel...
     26.05    0.4%     3094  mhc_pre_big_fuse_with_norm_tilelang_kernel
     18.55    0.3%     1462  _dsv4_topk_kernel
```

and the host side, by total duration: `vllm::moe_forward_shared` 2523 ms, `b12x::tp_moe_dynamic_launch`
964 ms, `b12x::blockscaled_serialized` 551 ms, `b12x::sparse_mla_sm120_decode_grid` 373 ms,
`aten::copy_` 272 ms, `aten::mm` 235 ms, `aten::fill_` 133 ms, `vllm::all_reduce` 152 ms.

**What this says.** Three things carry the step, in order: **unfused pointwise work (30 %)**, **the TP
all-reduce (21.4 %)**, and the MoE (20.4 %). Everything the repo has spent rounds tuning -- the linear
GEMM (`nvjet` 4.1 % + wmma 3.6 % + dense_gemm 2.9 %), the WO projection, the indexer (`_dsv4_topk_kernel`
0.3 %), the attention decode kernel (0.5 %) -- is in the noise by comparison. That is consistent with
every family swap having measured as a wash or worse.

**The all-reduce number is the important correction.** 3196 calls summing to 1490 ms is ~0.47 ms per
call, i.e. roughly 0.93 ms per layer for the two calls, against the ~1.45 ms per layer that the flat
step-time penalty implies. So the collective is plausibly **most** of the fixed per-layer cost after all
-- and ironically round 27's target, b12x's `PCIeAllReduce`, is CUDA-IPC based and cannot carry a 2-node
TP. The 5.7 ms the region marks reported for `allreduce` was queue-drain and low by 260x.

**The pointwise 30 % is the other half.** 7140 calls of one elementwise template, plus
`per_token_group_quant_8bit_kernel` at 7752 calls, plus `aten::copy_`/`fill_`/`zero_` on the host side.
That is the signature of an unfused pointwise chain -- norms, residual adds, quantisation and copies --
which is what our arm should have if it is running uncompiled.

And it is: `VLLM_USE_AOT_COMPILE=1` in the pin leaves `compilation_config.mode` at `CompilationMode.NONE`,
while the repo's own overlay stack contains `patch_nvidia_support_torch_compile`, which puts
`@support_torch_compile` on the NVIDIA DSV4 model precisely so it *can* be compiled. Our base reports
`mode: NONE`; the reference reports `VLLM_COMPILE` but vLLM skips it there because its model does not
declare support. **We declare support and then run uncompiled.** That is the next measurement.

### The compile path is gated by our own env default

`proto2-break0` (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`) is the informative failure of the round. The config
really did resolve to `CompilationMode.VLLM_COMPILE` and compilation began, then stopped at

```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
  module: vllm.third_party.deep_gemm._C, qualname: ...tf32_hc_prenorm_gemm
from user code:
  model.py:1543 in forward -> model.py:1284 mhc_pre_broadcast_tilelang
  -> tilelang.py:760 _hc_prenorm_gemm_outputs -> tilelang.py:61 tf32_hc_prenorm_gemm
```

which is precisely break 4 in this file's 2026-09-13 table, reproduced on proto2 unchanged. And the two
overlays that address it -- `patch_mhc_tf32_uncaptured` and `patch_mhc_tf32_call_redirect` -- are
**deliberately parked from `apply_main`**, with the reason recorded in the function's docstring: their
call-site needle predates `pr-53055.diff`, which folded the local `tf32_hc_prenorm_gemm` import into a
guarded `is_deep_gemm_supported, tf32_hc_prenorm_gemm` line, so the needle no longer matches and the
overlay's `SystemExit` would cut off every later overlay. The route those two implement is also the one
this torch rejects ("refuses to call a `torch.compiler.disable`d function from inside the compiled
region"), so re-anchoring them is not the fix. The sanctioned route is vLLM's own
`direct_register_custom_op`, which the parked patch already imports.

`patch_nvidia_support_torch_compile` *is* applied, so our model does carry `@support_torch_compile`
(model.py:1367) -- we declare compile support and then run uncompiled, which is the thing to fix.

### A dead-knob audit of our own environment surface

After round 29's discovery that `05-serve.sh` forwards an explicit `-e` list, the whole surface was
audited against what the image reads. Forwarded but read by **nothing** in vLLM or b12x:

| variable | note |
|---|---|
| `VLLM_USE_B12X_MOE` | dead; the b12x MoE path is unconditional |
| `VLLM_USE_B12X_MHC` | dead; the mHC path is unconditional |
| `B12X_MLA_SM120_UNIFIED` | dead |
| `B12X_MOE_FORCE_A8` | dead (this is what `proto2-moea4` actually was) |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | dead; the reference recipe sets it to 4096 |

`VLLM_USE_B12X_WO_PROJECTION` and `VLLM_USE_B12X_SPARSE_INDEXER` *look* unknown -- vLLM logs "Unknown
vLLM environment variable detected" for the latter -- but both are read by our own overlay in
`utils/sm12x_b12x_kernels.py`, so they are alive and the warning is benign.

**A regression I introduced and fixed.** Adding `VLLM_B12X_MOE_FP4_FORCE_A16` to the forward list in
round 29 with an empty default (`:-`) makes vLLM's parser do `bool(int(os.getenv(...)))` on an empty
string, and the engine dies with `ValueError: invalid literal for int() with base 10: ''`. Every arm that
did not set the variable explicitly would have failed to boot. It is now `:-0`.

### Break 4 is cleared, and break 5 is TileLang

Round 30 found that our arm never compiles because `configs/env.spark.sh:47` forces
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` and `config/vllm.py:786` turns that into `CompilationMode.NONE`. This
round set it to `0` and worked the compile path forward.

**Break 4 (the pybind GEMM) is cleared.** Registering `tf32_hc_prenorm_gemm` as a real custom op --
`direct_register_custom_op(op_name="tf32_hc_prenorm_gemm", mutates_args=["out", "sqrsum"], fake_impl=...)`
-- and routing the call through `torch.ops.vllm.tf32_hc_prenorm_gemm` removes the error entirely: the
`tf32_hc_prenorm_gemm` frame is gone from the traceback and compilation advances past it. Tested by
bind-mounting the patched `model_executor/kernels/mhc/tilelang.py` on both nodes, no rebuild.

Two things had to be right, and the repo had neither:

- **All three call sites, not one.** `tilelang.py` calls the GEMM from `_hc_prenorm_gemm_outputs` and
  from two functions that take `mixes` (lines 61, 176, 425 in the pristine file). Fixing one leaves two
  more breaks.
- **All three local import blocks had to be hoisted.** Each call site sits behind a local
  `from vllm.utils.deep_gemm import ...`, and a local import is an importlib call, which Dynamo skips
  even when the imported names are unused -- so it is its own graph break. Two of the three blocks are
  byte-identical. Verified non-circular: `mhc.warmup` imports `mhc.tilelang_kernels`, and
  `tilelang_kernels` does not import `mhc.warmup`.

This is `patch_mhc_tf32_customop` in `patches/apply_overlays.py`, **parked from `apply_main`** like its
two siblings, with a `--only mhc-tf32-customop` entry. It is verified against a pristine file: correct
registration, three routed call sites, three blocks hoisted, and idempotent on re-run. It supersedes
`patch_mhc_tf32_uncaptured`, whose `@torch.compiler.disable` route this torch rejects outright.

**Break 5 is TileLang, and it has a general fix.** With break 4 gone the engine dies at

```
Unsupported: Attempted to call function marked as skipped
  module: tvm.script.parser.core.diagnostics, qualname: _patched_inspect_getfile,
  skip reason: file is under skip directory (.../tilelang/3rdparty/tvm/python/tvm/)
from user code:
  model.py:1284 mhc_pre_broadcast_tilelang
  -> tilelang.py:799 _MHC_PRE_BIG_FUSE_TILELANG_KERNEL(...)
  -> jit_warmup_tilelang_helper.py:169 wrapper -> :142 launch
  -> tilelang/jit/__init__.py:527 __call__ -> :392 _infer_jit_mode
  -> self.func._is_lazy_style(...)
```

So the TileLang JIT wrapper introspects source **at call time**, and that introspection lives in a
Dynamo skip directory. The important part is the guard immediately above it
(`tilelang/jit/__init__.py:523`):

```python
if self.mode == "auto":
    self.mode = self._infer_jit_mode(*args, **kwargs)
    self.func.set_mode(self.mode)
```

**`_infer_jit_mode` is only called while `mode == "auto"`, so pinning each kernel's mode before the
traced forward runs skips the TVM introspection entirely.** That is a general fix for this whole class of
break -- `mhc_fused_tilelang_kernel` and `mhc_pre_big_fuse_with_norm_tilelang_kernel` both appear in the
round-30 kernel profile, so there is more than one such call -- as against wrapping each kernel in a
custom op one at a time. The mode to pin is whichever `initialize_jit_mode` would have chosen; the
reference's own log shows these kernels built with `out_idx=None`, which does not by itself decide it.

**Where the compile path stands now**: gate cleared, breaks 1-4 cleared, break 5 identified with a
one-line-per-kernel fix and its guard located. The remaining work is mechanical and finite: pin the
TileLang kernel modes, then re-run and repeat until the traced forward is clean.

### Break 5 needs a different approach than the obvious one

Break 5 is TileLang, and the natural fixes both fail for reasons worth recording, because each one cost a
boot to rule out.

**Pre-resolving the mode at import does not work.** The idea was sound: `tilelang/jit/__init__.py:523`
only calls `_infer_jit_mode` while `mode == "auto"`, and `_is_lazy_style`'s first check
`has_internal_prim_func(self.orig_func)` needs no arguments, so the mode could be settled once, outside
Dynamo, from `mhc/tilelang_kernels.py`'s module globals. Checked in the container:

```
module has 10 kernel objects
obj: MhcPreBigFuseTileLangKernel
kernel attr: function          <-- not the TileLang JIT wrapper
has .mode: False    value: None
has .func: False
```

`VllmTileLangJitKernel.kernel` is a plain **function**, and the object that carries `mode` is produced by
*calling* it -- `self.launch(self.kernel(*kernel_args), ...)` -- so it exists only at call time and only
inside the traced frame. There is nothing to pre-resolve. The block was removed from the overlay rather
than left in as a no-op.

**Warmup does not cover this kernel either.** `VllmTileLangJitKernel.launch` routes the warmup call
through `compile_tilelang(...)` -> `.compile()`, never `__call__`, so warmup cannot settle the mode for
the kernels it does touch. And a bound-in `launch` that resolves the mode on the first call still runs
*inside* Dynamo: the traceback with that patch in place names our own line,

```
jit_warmup_tilelang_helper.py:147 in launch
    jit_impl.mode = jit_impl._infer_jit_mode(*args, **call_kwargs)
  -> tilelang/jit/__init__.py:392 _infer_jit_mode
  -> is_lazy_style = self.func._is_lazy_style(*args, **kwargs)
```

and the Dynamo `Unsupported` is not catchable from user code, so the `try/except` around it does not
help. That patch was dropped too.

**What is left for break 5** is to keep Dynamo out of TileLang entirely -- wrap the mhc TileLang kernel
call sites (`tilelang.py:785`, and the `_MHC_FUSED_TILELANG_KERNEL` site) the same way the pybind GEMM was
wrapped, or cache `has_internal_prim_func` per function in a pre-warmed map. The first is the pattern
already proven on break 4; the second needs the first call to happen outside Dynamo, which this evidence
says it does not.

For the record, `mhc/tilelang_kernels.py` contains **no** `T.prim_func` at all, so
`has_internal_prim_func` is False for every kernel in it and the style question is not the obstacle --
the source inspection is.

### Break 5: the un-patching route is closed, and it is closed by the stdlib

Round 32 ruled out pre-resolving the TileLang mode and relying on warmup. This round attacked the
*other* side of the same break: the skip verdict comes from a function defined under tilelang's bundled
TVM, so if the offending function can be replaced with the stdlib one, Dynamo should stop refusing.

The patch chain turned out to be two deep, and it is now measured exactly:

```
before tilelang import:  inspect getfile
after  tilelang import:  tvm.script.parser.core.diagnostics _patched_inspect_getfile
tvms stash `_getfile`:   torch.package.package_importer _patched_getfile
after reload(inspect):   inspect getfile
```

`torch.package.package_importer` patches `inspect.getfile` first and keeps the real function as
`_orig_getfile` (line 735); tilelang's bundled TVM patches it again
(`.../3rdparty/tvm/.../core/diagnostics.py:185`) and stashes *torch's* version. So restoring from TVM's
stash only swaps one patch for the other, and Dynamo merely renames the skipped function.

Two forms were tried, both removed:

| attempt | result |
|---|---|
| `importlib.reload(inspect)` | **works** -- the skip error drops to 0 -- but it re-executes the module and breaks pydantic during vLLM import: `TypeError: 'BeforeValidator' object is not iterable` |
| `inspect.getfile = torch.package.package_importer._orig_getfile` | runs (prints, in all three processes) but is **useless**: the skip error returns, 4 occurrences |

Because with the chain removed, the next frame is the decisive one:

```
qualname: checkcache
skip reason: file is under skip directory (/usr/lib/python3.12/linecache.py
```

`inspect.getsource` bottoms out in `linecache`, and **`/usr/lib/python3.12/` is itself a Dynamo skip
directory** -- torch ships the stdlib in its skip list. So no amount of un-patching `inspect.getfile`
can help: Dynamo will not trace source introspection at all, it will only name the next stdlib frame
after each repair.

**This closes the question for break 5.** The only route is to keep Dynamo out of TileLang entirely --
wrap the mhc TileLang kernel call sites (`tilelang.py:785`, and the `_MHC_FUSED_TILELANG_KERNEL` site)
the way the pybind GEMM was wrapped for break 4. Break 4's fix remains cleared and verified; this
round's additions were removed rather than left in as no-ops, so `patch_mhc_tf32_customop` now contains
exactly the custom-op registration, the three routed call sites and the three hoisted import blocks.

### Break 5: untraceable from every direction, so TileLang must be kept out entirely

Round 33 showed that un-patching `inspect.getfile` only walks Dynamo to the next stdlib frame. This
round attacked the other end of the same decision -- `JITFunc._is_lazy_style`'s **fast path**:

```python
if has_internal_prim_func(self.orig_func):     # tilelang/language/eager/builder.py:1525
    return True
```

`has_internal_prim_func` scans the function's AST (`utils.get_ast` -> `inspect.getsource`). Guarding it
to return False under `torch.compiler.is_compiling()` skips that scan, and it is a *safe* substitution:
it only skips the fast path, because `_is_lazy_style` then binds the signature and calls the body, which
is the authoritative test. `builder.py:1525` is the only caller in the whole tree.

It worked exactly as intended, and that is what makes the result conclusive:

```
$ docker logs ... | grep -c "Attempted to call function marked as skipped"
0
$ docker logs ... | grep "Unsupported method call" -A3
torch._dynamo.exc.Unsupported: Unsupported method call
  Explanation: Dynamo does not know how to trace method `__new__` of class `type`
  Developer debug context: call_method UserDefinedClassVariable(<class 'tvm_ffi.core.CObject'>) __new__
    [UserDefinedClassVariable(<class 'tvm.tirx.expr.Var'>), ConstantVariable(str: 'num_tokens'), ...]
```

With the source scan out of the way, Dynamo traces into the kernel body and immediately meets **TVM IR
construction** -- `tvm.tirx.expr.Var('num_tokens', 'int32')` built through `tvm_ffi.core.CObject.__new__`
-- which it refuses.

So `_is_lazy_style` is untraceable **both** ways, and that closes the question for good:

| path through `_is_lazy_style` | what Dynamo hits |
|---|---|
| source scan (`has_internal_prim_func`) | `inspect.getsource` -> `linecache.checkcache`, and `/usr/lib/python3.12/` is a Dynamo skip directory |
| body call (`self.orig_func(*args)`) | TVM IR construction, `type.__new__` on `tvm_ffi.core.CObject` |

There is no third path. TileLang cannot be traced, and the only fix is to keep Dynamo out of it
entirely: register the mhc TileLang entry points as custom ops (or vLLM splitting ops) the way the
pybind GEMM was registered for break 4.

**The guard was removed again rather than kept.** It does clear the skip error -- and was worth running
for exactly that reason -- but it lands on an equally hard wall one frame later, so keeping it would
change which error the next reader sees without advancing anything.

Also fixed this round: round 33's removal of the `inspect.getfile` restore took the function and its
call but left the **comment block** behind, so the overlay was emitting stale commentary into the
generated file. Removed, and the overlay now emits exactly the custom-op registration, the three routed
call sites and the three hoisted import blocks. Re-verified against the pristine file.

### Break 5: `allow_in_graph` cannot substitute for a custom op, and the reason matters

The one route left for break 5 is to keep Dynamo out of TileLang. `model.py` reaches TileLang through
five entry points -- `mhc_pre_broadcast_tilelang` (1284), `mhc_pre_tilelang` (1300),
`mhc_fused_post_pre_tilelang` (1314/1343), `mhc_post_tilelang` (1553/1566) and
`hc_head_fused_kernel_tilelang` (1585) -- so the cheapest-looking fix was one
`torch._dynamo.allow_in_graph(...)` per function rather than five full custom-op registrations.

It applies cleanly (`b12x marked 5/5 mhc tilelang entry points allow_in_graph`, in all three processes)
and it *does* clear the skip error (0 occurrences), so the marking works. But it fails immediately
afterwards, and the failure is the useful part:

```
torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors:
    call_function <function mhc_pre_broadcast_tilelang ...>(
      *(FakeTensor(..., size=(s72, 4096), ...), ..., 1e-06, 1e-06, 1e-06, 2.0, 20), ...)
    got TypeError('unhashable type: non-nested SymInt')
```

`allow_in_graph` still **runs the function**, on FakeTensors, to infer its outputs -- and TileLang's
kernel cache then tries to key on a symbolic size. So the function must not merely be untraceable, it
must not be *executable on fake inputs* either.

That is exactly what `direct_register_custom_op(..., fake_impl=...)` provides and what `allow_in_graph`
cannot: with a real custom op, Dynamo calls the **fake impl** for shape inference and never runs the
kernel during tracing. The block was removed rather than left in, since it only moves the error.

**So break 5's remaining work is now fully specified**: register the five mhc entry points with
`direct_register_custom_op` plus an explicit `fake_impl` for each, and route `model.py`'s five call
sites through `torch.ops.vllm.*`. That is four separate requirements per function -- signature, fake
shapes, registration, call-site rewrite -- for five functions, and it is a porting effort inside a
third-party tracing model rather than a configuration change. Worth stating plainly: rounds 31-35 have
cleared break 4 and mapped break 5 completely, but each fix has revealed another break, and the repo
stalled at this same point once before. The fallback clause of the goal -- best measured arm plus a
written per-gap attribution -- is the alternative if this does not converge.

## 2026-09-17 (round 36-37): CUDA time attributed to our own Python frames, and a correction to round 30

Round 30 profiled under `--enforce-eager` to make every kernel a real launch, and reported "30 % of
CUDA time is unfused pointwise work, so compilation is the lever". That number drove five rounds of
compile work. **It does not hold in the arm that actually runs.** A second capture on the normal
(graph-mode) arm -- `profiler: torch` with `torch_profiler_with_stack: true`, one 128-token request --
attributes every kernel to its innermost Python frame:

| ms | share | innermost Python frame |
|---|---|---|
| 242.79 | **36.0 %** | `<built-in function linear>` |
| 203.66 | **30.2 %** | `tp_moe_dynamic_launch` (our b12x MoE) |
| 55.65 | 8.2 % | `blockscaled_serialized` (b12x block-scaled linear) |
| 54.94 | 8.1 % | `all_reduce` |
| 44.72 | 6.6 % | `reshape` |
| 27.73 | 4.1 % | `bmm` |
| 10.05 | 1.5 % | `mm` |
| 6.92 | 1.0 % | `copy_` |
| 5.40 | 0.8 % | `all_gather` |
| 3.68 | 0.5 % | `per_token_group_fp8_quant` |
| 1.78 | 0.3 % | `sparse_mla_sm120_decode_grid` |
| 0.96 | 0.1 % | `sm12x_b12x_kernels.py(255): sync_packed_indexer_k` (**ours**) |

675.3 ms of kernel time carries a Python frame in this window. The absolute total is not comparable to
round 30's 6951 ms -- stack recording inflates everything and the windows differ -- but the **rank
order is**, and it says the opposite of round 30:

- **`linear` plus `blockscaled_serialized` is ~44 %.** The linear layer family is the largest single
  consumer, and we tested only two of its backends: `b12x` (current) and `deep_gemm` (the reference's,
  which measured worse and wildly unstable). `humming` is a third, and the repo's old MoE sweep found
  it the only other backend that would even run on this rig.
- **The eager-mode pointwise block was an artifact.** In graph mode `reshape` + `copy_` + `mm` +
  `bmm` together are ~13 %, so compilation would target a far smaller slice than round 30 claimed.
  That materially weakens the case for the remaining compile work, which is a porting effort of
  unknown depth (five custom ops with fake impls, and every break fixed so far revealed another).
- **Our own overlay file is not a culprit.** `sm12x_b12x_kernels.py`'s two hot sites --
  `sync_packed_indexer_k` (a `reshape` that materialises a copy) and `try_b12x_wo_proj` (a `copy_`) --
  are 0.96 ms and ~2 ms in this window. Recorded as a negative for goal item 3.

The remaining attribution, all measured: linear family ~44 %, MoE 30 %, all-reduce 8 %, and everything
the repo has spent rounds on is below 5 %. The next arm follows from the top row:
`LINEAR_BACKEND=humming`.

### The linear family is exhausted (negative)

With the corrected profile putting `linear` + `blockscaled_serialized` at ~44 % of CUDA time,
`LINEAR_BACKEND=humming` was the one untried backend. It dies at worker init:

```
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [4, 4096]
  but got: [4, 1024].
```

The humming linear kernel cannot handle this model's O-projection shapes -- `o_lora_rank` 1024 against
hidden 4096. So the family is now closed on measurement rather than assumption:

| linear backend | outcome |
|---|---|
| `b12x` (current) | best measured; sum 334.3 |
| `deep_gemm` (the reference's own choice) | sum 328.3 and wildly unstable, c6 spread 48.6 % |
| `humming` | cannot load this model |
| `cutlass`, `marlin` | in the NVFP4 clamp set this rig rejects |

### Where that leaves the goal

The measured attribution of the serving arm is now: **linear ~44 %, MoE 30 %, all-reduce 8 %,
everything else below 8 %** -- and each of those three is closed:

- **linear** -- all four reachable backends tried; `b12x` is the best, and the reference's choice is
  worse here.
- **MoE** -- our kernel already measures 10-13x faster than FlashInfer's in the head-to-head, the
  activation format was tested (`B12X_MXFP4_BF16`, a wash with 40-50 % spreads), and the policy-ladder
  gap belongs to an older b12x generation that is not drop-in (and is not a route this repo should take).
- **all-reduce** -- both engines select PYNCCL, b12x's own PCIe transport is CUDA-IPC based and cannot
  cross hosts, and disabling the CUDA graphs (which removes our per-layer all-reduce break) is worse.

The compile path remains the only lever with a mechanism, but round 30's justification for it was an
eager-mode artifact: on the real arm the work Inductor could fuse is ~13 %, not 30 %. So it is a
porting effort of unknown depth (five custom ops with fake impls, with a new Dynamo break behind every
one fixed so far) chasing a smaller prize than the five rounds spent on it assumed.

**Recommendation, stated plainly**: take the goal's fallback clause. The best measured arm is `proto2`
at 35.4 / 78.9 / 103.0 / 117.0 (sum 334.3) against the reference's same-day 66.4 / 118.4 / 139.0 / 161.0
(sum 484.8), with gates, acceptance and tokens-per-step all sound, and the per-gap attribution above is
complete and measured. Continuing is not impossible, but it is now working inside a third-party tracing
model for a ~13 % ceiling while the three largest measured consumers are each closed by experiment.

### Correction: the stack trace cannot apportion the step either

Round 37 attributed CUDA time to Python frames and concluded "linear ~44 %, MoE 30 %". **That reading
does not survive its own numbers, and it is withdrawn.** Counting the kernels behind the top row:

```
LM-head (default_unquantized_gemm) kernels: 324
  total 242.8 ms  median 0.284 ms  max 3.037 ms
```

A 128-token generation at ~4.6 tokens per step is ~28 engine steps, so the window is roughly 28-46
steps, and the whole captured kernel total is 675.3 ms. At a c6 step of 236 ms that window is
**~7-11 s of GPU time**, of which only 675 ms appears as individual kernel events -- **about 6 %**.

The reason is structural: with `FULL_AND_PIECEWISE` CUDA graphs, a decode step **replays a captured
graph**, and the profiler records the graph launch, not the kernels inside it. So every share computed
from this trace is a share of the visible 6 %, not of the step. That also explains why `all_reduce`
came out at 54.9 ms in this window when the step-time arithmetic says ~0.9 ms per layer.

**What does survive, with its numbers:** the LM head is real, it is **unquantized**, and it can be
priced. `logits_processor._apply_head` calls `lm_head.quant_method.apply(...)`, and a quant config that
excludes `lm_head` hands out `UnquantizedEmbeddingMethod`, which lands in
`layers/utils.py:84 default_unquantized_gemm` -> `torch.nn.functional.linear` in **BF16**. The weight is
4096 x vocab, so every step streams it in full regardless of batch: **~5.3 ms per step** (242.8 ms over
the window), which is **~2.2 % of our 236 ms c6 step**. Real, measurable, and too small to be the gap.

`logits_processor._apply_head` also reveals a `head_dtype` knob -- "Project hidden states through the
lm_head, honoring head_dtype" -- so the head's dtype is configurable. Not worth pulling for 2.2 %.

**The correct instrument for step attribution is `capture_torch_profiler: true`**, which vLLM's
`ProfilerConfig` provides precisely for this: "enables a torch profiler during CUDA graph capture on
rank 0. Traces are saved to a `capture_traces` subdirectory". That records the graph's kernels instead
of the replay, and it is the next measurement. Until it is run, no per-stage share of the step should
be quoted from either trace -- the eager one distorts the regime, the graph one sees 6 % of it.

### `capture_torch_profiler` does not engage on this rig (instrument negative)

The round-38 correction identified `ProfilerConfig.capture_torch_profiler` as the way to see inside a
replayed CUDA graph, and it is the documented purpose: "enables a torch profiler during CUDA graph
capture on rank 0. Traces are saved to a `capture_traces` subdirectory under `torch_profiler_dir`."

It was enabled together with `profiler: torch` and `torch_profiler_dir`. vLLM accepted the config --
the resolved `ProfilerConfig` in the API server's log carries `capture_torch_profiler=True` -- and CUDA
graphs were captured normally (PIECEWISE 9 graphs, FULL 6). But:

- **no `capture_traces` directory was created anywhere** under the profiler dir, which itself was never
  created either;
- **neither of the two log lines that branch emits** appears: `gpu_model_runner.py:6874` logs
  "Rank %d: Torch profiler enabled for CUDA graph capture, traces will be saved to: %s" when it engages,
  and the `else` at :6882 logs "Rank %d: Torch profiler disabled for CUDA graph capture". Neither is in
  the log, so the branch was not reached at all;
- the EngineCore's printed config contains **no `profiler_config`** (0 occurrences), while the API
  server's does -- so the worker may not be carrying the setting that
  `gpu_model_runner.py:6856` tests (`self.vllm_config.profiler_config.capture_torch_profiler`).

That last line is the next check if the instrument is wanted: whether the worker's config carries
`profiler_config` at all, and if not, why. Recorded rather than chased further, because it is
instrumentation rather than a lever, and because the repo already has a method that prices stages
without the profiler.

**The method that has actually worked here is intervention, not profiling.** Round 26's skip-flag A/B
priced the indexer at ~10.6 ms per step even though the region profiler was demonstrably
non-additive; the region profiler has now failed three ways (queue-drain time, eager regime, 6 %
visibility) while the skip test was decisive the first time. `VLLM_SKIP_FLAG_DIR` already supports
`indexer_all`, `indexer_op` and `compressor`, and the same switch is trivial to extend to the two
remaining large consumers, the TP all-reduce and the MoE. **That is the next measurement**, and it is
additive by construction: each skip is priced against the enclosing layer total, which is the one number
the profiler did get right.

### The TP all-reduce costs 0.3 ms, so it is not the fixed per-step cost (decisive negative)

The method that works in this repo is intervention, and this round applied it to the collective that
three separate profiler readings had implicated (`allreduce gpu_sum` 5.7 ms from the region marks,
21.4 % of captured CUDA time in round 30, 54.9 ms in round 37). A skip guard was added to
`tensor_parallel_all_reduce` in `vllm/distributed/communication_op.py`, armed through
`VLLM_SKIP_FLAG_DIR`, and bind-mounted -- the same one-file technique used for the b12x PCIe experiment.

The skip fired, verified two ways:

```
$ docker exec vllm-ds4-0731 grep -c skip_allreduce /opt/vllm/vllm/distributed/communication_op.py
1
$ docker logs ... | grep "b12x region allreduce"
b12x region allreduce: n=87 gpu_sum=0.1ms gpu_min=0.00ms gpu_avg=0.00ms gpu_max=0.00ms wall_sum=1.9ms
```

against the control's `n=87 gpu_sum=5.7ms`. So all 87 collectives per step became free, and the
enclosing measurement barely moved:

| measurement | control | all-reduce skipped |
|---|---|---|
| `b12x layers: n=43 sum` | **148.0 ms** | **147.7 ms** |
| region `allreduce` gpu_sum | 5.7 ms | 0.1 ms |
| region `attn` gpu_sum | 80.3 ms | 80.2 ms |
| region `ffn` gpu_sum | 65.2 ms | 64.2 ms |

**The TP all-reduce contributes 0.3 ms to the 43-layer total -- about 0.2 %.** Every profiler reading
that suggested otherwise was an artifact: the region marks measure queue-drain, round 30's 21.4 % was a
share of an eager-mode capture, and round 37's 54.9 ms was a share of the 6 % of a step that is visible
outside graph replay. This closes the hypothesis pursued in round 27 (b12x's PCIe transport, which is
CUDA-IPC and cannot cross hosts anyway) and in round 30 (`all-reduce = 21 % of CUDA time`).

It also settles the method question. The profiler has now failed three distinct ways on this rig, while
a single skip A/B was decisive the first time: **price stages by removing them and measuring the
enclosing per-layer total**, which is the one number the profiler got right (it is additive, and
`attn` + `ffn` account for 144.4 of 147.7 ms). The same switch extends to the remaining large
consumers -- the MoE (region `ffn`, 64.2 ms) is the obvious next one -- and that is how the ~61 ms
per-step penalty should be attributed.

## 2026-09-17 (round 40): the target forward isolated, and the draft priced

`DISABLE_DSPARK=1` turns speculative decoding off, so exactly one token is produced per engine step and
the meter's per-stream tok/s *is* 1/step-time. That makes the target model's forward directly
measurable, with no profiler involved. The arm boots with `speculative_config=None` and passes both
gates in all three passes.

| level | no DSpark (1 tok/step) | `proto2` with DSpark (4.6 tok/step) | DSpark overhead |
|---|---|---|---|
| c1 | **105.3 ms** | 130 ms | **+24.7 ms** |
| c3 | 128.2 ms | 175 ms | +46.8 ms |
| c5 | 145.3 ms | 223 ms | +77.7 ms |
| c6 | **148.1 ms** | 234 ms | **+86.2 ms** |

Median spreads are 0.0 / 0.4 / 0.6 / 0.5 % -- the tightest arm measured on this rig, which is itself
worth noting: removing speculation removes the variance source.

Three things follow, and they are the best-supported statements in this file:

1. **The target forward is dominated by batch-independent work.** It grows only 1.4x from 8 rows
   (c1, 105.3 ms) to 48 rows (c6, 148.1 ms). That is the flat ~61 ms per-step penalty, seen directly
   for the first time: 43 layers at 2.4 ms each, barely moving with batch.
2. **The whole DSpark draft-plus-verification machinery is only +24.7 ms at c1**, rising to +86.2 ms at
   c6. So at c1 the draft is 19 % of the step, not the dominant term, and the speculative path is doing
   its job -- it buys 4.6 tokens for 130 ms against 105.3 ms for one.
3. **The gap is in the target forward.** The reference's *entire* c1 step is 69 ms, and its own draft
   sits inside that; our target forward alone at batch 8 is 105.3 ms, i.e. **1.5x the reference's whole
   step** before our draft is added. At c6 the target forward is 148.1 ms against the reference's 171 ms
   step, but the reference's step includes its draft, so its target forward is again well under ours.

This is the number the fallback attribution needed, and it was obtained by intervention -- two arms that
differ only in `DISABLE_DSPARK` -- not by any of the three profiler approaches that failed. Everything
the profilers attributed (region marks, eager shares, graph-visible shares) is superseded by it.

**What it does not yet say** is which half of the forward -- `attn` (80.2 ms region sum) or `ffn`
(64.2 ms) -- carries the excess, and the region *split* is the part shown to be unreliable. Pricing
those two by removal, the way the all-reduce was priced, is the next measurement.

### The FFN priced by removal, and the batch-independent floor

Round 40 isolated the target forward (105.3 ms at c1, 148.1 ms at c6, one token per step). This round
prices its two halves by replacing each with a pass-through in the layer forward, bind-mounting
`model.py`, and reading step time the same way.

**FFN (`self.ffn(x, input_ids)` -> `x = x`) is usable:**

| level | unmodified | FFN removed | FFN cost |
|---|---|---|---|
| c1 | 105.3 ms | **96.2 ms** | +9.1 ms |
| c3 | 128.2 ms | 97.7 ms | +30.5 ms |
| c5 | 145.3 ms | 97.8 ms | +47.5 ms |
| c6 | 148.1 ms | **98.0 ms** | **+50.1 ms** |

Spreads 0.0-0.3 %. Two consequences, and they are the sharpest statements in this file:

1. **The FFN carries essentially all of the batch dependence.** With it removed the step is *flat* at
   96-98 ms from 8 rows to 48, so the remaining 96 ms is batch-independent work -- the flat penalty
   round after round has been chasing, now measured directly.
2. **The reference's entire c1 step is 69 ms, which is below our batch-independent floor even with the
   MoE deleted.** The c1 deficit therefore is not the MoE; it is in that 96 ms of per-layer work that
   does not move with batch.

Stated confound: a degenerate residual stream can also change the attention path, so +50.1 ms is an
upper bound on the MoE, not an exact price.

**Attention (`self.attn(positions, x, None)` -> `x = x`) is not usable**, and the reason is worth keeping.
Removing it collapses the residual stream, and with it the MoE routing:

- generations ran short and varied per pass (601, 761, 75, 1230 tokens at c3 against an expected 1536);
- spreads blew out to 9.7-15.1 %;
- the implied step time was ~17 ms, i.e. a **6x speedup from deleting half the layer**, which is not
  credible;
- gates were garbage (`' a =  image:ife'`, `' 组合内~| '`).

The likely mechanism is routing collapse: degenerate hidden states send the MoE to far fewer distinct
experts, so the arm prices attention *and* a crippled MoE at the same time.

**The instrument rule this establishes**: removal is clean only when the removed stage does not feed the
*selector* of another stage. The all-reduce skip (0.3 ms), the indexer skip (~10.6 ms) and the FFN skip
all satisfy that; the attention skip does not, because the MoE's router reads the hidden states
attention produces.

### mHC's GEMM is worth ~1 %, and the attribution is complete

mHC was the one per-layer, batch-independent component never priced, and it runs a **tf32** prenorm GEMM
through DeepGEMM with a TileLang fallback sitting beside it in the same function. `_USE_DEEP_GEMM` is a
module-level bool, so pinning it False and bind-mounting `mhc/tilelang.py` is a clean one-variable A/B:

| level | DeepGEMM tf32 | TileLang fallback | delta |
|---|---|---|---|
| c1 | 105.3 ms | **104.2 ms** | -1.1 ms |
| c3 | 128.2 ms | 127.1 ms | -1.1 ms |
| c5 | 145.3 ms | 144.1 ms | -1.2 ms |
| c6 | 148.1 ms | **146.0 ms** | -2.1 ms |

**Both gates pass** with the fallback (`' Paris. The capital of Spain'`, `'72, 9x9'`), so unlike the skip
probes this is a real A/B and the comparison is meaningful. The difference is ~1 %, just outside the
control's spread, and 1 % against a 27 % gap is not a lever. So mHC -- the last unpriced per-layer
component -- is priced at ~1 ms per step.

**That completes the attribution.** With it, every component of the step has a number from an
intervention: target forward 105.3 ms at c1 / 148.1 ms at c6 (`DISABLE_DSPARK`), DSpark +24.7/+86.2 (the
difference), FFN +9.1/+50.1 (pass-through), all-reduce +0.3 (skip marker), indexer ~10.6
(`skip_indexer_all`), LM head ~5.3 (kernel counts), mHC ~1 (this A/B) -- and a residual of ~79 ms at c1
that is the attention half plus the per-layer glue, where no reachable configuration has moved the
number: three attention-backend arms all measured worse than `B12X_MLA_SPARSE`, and our MLA kernel is
2.7x faster than `b12x_ref`'s in the head-to-head.

The consolidated per-gap attribution, with the family, the library and the closing measurement for each,
is written up in `HANDOVER.md` under "Per-gap attribution". It is the goal's fallback deliverable, and
it rests entirely on intervention measurements rather than profiler shares, for the reason this file
documents at length.

### The residual is launch overhead: ~92 kernels per layer (and a correction to round 38)

Round 41 showed the step is *flat* at 96-98 ms from 8 to 48 rows once the FFN is removed. Work that does
not scale with data is not memory traffic, so the residual should be launch/overhead. Counting the
kernels in the round-37 trace tests that directly:

```
kernel events: 139353
kernel CUDA time: 5194.8 ms
execute_context_0(0)_generation_1(8):  35     <- graph replays at 8 tokens
tp_moe.flatten_routing_ids:           151
vllm::moe_forward_shared:             151
```

The window holds ~35 graph replays, so the step carries **~3981 kernel launches, i.e. ~92.6 per layer**,
and ~148 ms of kernel time. At a conservative 25 us per launch that is **~2.3 ms per layer, ~99 ms per
step** -- which is the batch-invariant residual almost exactly. **So the residual is per-layer launch
overhead, and its size is now explained rather than merely measured.**

**Correction to round 38.** That round withdrew the round-37 shares on the grounds that "only ~6 % of a
step appears as individual kernel events, because a replay hides the kernels". The first half is wrong:
the kernels *are* recorded -- 5194.8 ms of them. What covers only 675.3 ms is *attribution to a Python
frame*, because the overwhelming majority run inside Inductor's `execute_context_*` regions where no
Python frame exists. So the correct statement is "87 % of kernel **time** has no Python frame", not
"94 % of the step is invisible". The shares in round 37 were computed over the matched subset and are
therefore still not shares of the step -- that conclusion stands -- but the reason recorded for it was
wrong and is corrected here.

**What this does to the lever question.** The step is ~92 launches per layer of overhead, and a compiled
forward is precisely the thing that reduces that count. That re-justifies the compile port on a
mechanism rather than on the eager-mode 30 % figure that round 38 rightly flagged as an artifact: the
target is now quantified as *kernel count per layer*, not as an unattributed percentage. Rounds 31-35
established that reaching it needs five custom ops with fake impls for the mhc entry points; that work
is unchanged and still the next step.

### The biggest kernel cost is a direct copy, ~5 per layer, and it is not ours

Ranking the round-37 trace by launches per layer (92.59 total, over ~35 replays at 8 tokens) puts the
*time* almost entirely in a few families:

| count | per layer | ms in window | kernel |
|---|---|---|---|
| 7560 | 5.02 | **1543.03** | `elementwise_kernel<128,4,gpu_kernel_impl_nocast<direct_copy_kernel_cuda...` |
| 4177 | 2.78 | 294.65 | `b12x_libdense_gemmDenseGemmKernel...` |
| 3891 | 2.59 | 203.03 | `cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2` |
| 3636 | 2.42 | 204.06 | `ncclDevKernel_AllReduce_bf16_RING` |
| 3442 | 2.29 | 212.71 | `b12x_libdense_gemmDenseGemmKernel...` |
| 3492 | 2.32 | 17.37 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` |
| 3335 | 2.22 | 52.41 | `mhc_fused_tilelang_kernel` |

**The largest single item is `direct_copy_kernel_cuda` at 1543 ms, i.e. 44 ms per step, at 5.02 launches
per layer.** The largest single instance has `grid: [28496, 1, 1]`, which at 128 threads x 4 elements is
~15M elements, ~29 MB in bf16. Round 37's stack attribution already placed this family under
`execute_context_0 <- aten::reshape <- aten::clone <- aten::copy_`, i.e. a *clone* inside an
Inductor/AOT-compiled region.

Two things it is not, both by measurement:

- **Not the all-reduce workspace copies.** Our `patch_tp_allreduce_static_workspace` does two `copy_` per
  collective, four per layer, which would match the count -- but round 39 made the whole function a
  no-op, removing those copies, and the layer total moved **0.3 ms**.
- **Not the MoE.** Round 41 removed the FFN entirely and the step fell only 9.1 ms at c1, to 96 ms; a
  44 ms cost inside the FFN would have gone with it. So the copies live in the **attention/mHC half**.

A scan of our own overlay files finds only small `.contiguous()` calls (`scale`, `indices`, `topk_ids`,
`norm_weight`, 4096-element vectors) and the KV/`page_table` slices -- nothing that is obviously 29 MB.

**Naming it needs shapes, not counts.** `torch_profiler_record_shapes: true` records the tensor
dimensions for every op, and running it under `--enforce-eager` puts a Python frame on each launcher.
That is the next measurement, and it matters: 44 ms/step is **19 % of the c6 step and 33 % of the c1
step**, so if these copies are removable by our own code it is a larger prize than anything else left,
and if they are inside the AOT-compiled forward it is a compile question instead.

### Naming the copy: first attempt at shapes fails on flush

The decisive instrument for the 44 ms/step `direct_copy_kernel_cuda` is a trace that records *shapes*,
run under `--enforce-eager` so each launcher also carries a Python frame. Enabled as
`torch_profiler_record_shapes: true` together with `torch_profiler_with_stack: true`, one 64-token
request, `/start_profile` then `/stop_profile`.

It failed, and the failure is specific:

```
(EngineCore) ERROR [core.py:1625]  self.collective_rpc("profile", args=(is_start, profile_prefix))
(APIServer)       await engine_client(raw_request).stop_profile()
(APIServer) Exception: Call to profile method failed: cancelled
```

Only the API server's own trace landed (`spark1_1.async_llm...pt.trace.json.gz`, 1.2 MB); the worker's
was never written, because the stop RPC was cancelled before the worker finished serialising. Shapes
plus full Python stacks under eager mode is evidently too much for the worker to flush in the window.

The retry is a config change, not a new idea: **drop `with_stack`** (shapes plus the launcher names
should be enough to size the tensor being copied), keep the window short, and give `/stop_profile` a
much longer timeout than 90 s. Recorded so the next attempt does not repeat the same combination.

Rig is clean; no containers running.

## 2026-09-17 (round 45): the biggest win of the run was already documented in our own code, disabled

Round 44's shapes-recorded trace named the 44 ms/step cost exactly:

```
1344 calls   aten::copy_   [17177, 64, 132] -> [17177, 64, 132]
```

`64 x 132 = 8448`, and our own boot log prints `b12x packed indexer insert ok sidecar=(1030, 8448)` -- the
packed Lightning Indexer sidecar. Round 37 had already flagged
`sm12x_b12x_kernels.py(255): sync_packed_indexer_k`.

**`patches/files/sm12x_b12x_kernels.py` already knew.** `_indexer_direct_gather`'s docstring says it
outright:

> The default flatten+gather reshapes a strided slice of the KV cache, which materialises a full-cache
> copy on every layer of every decode step (~57 ms/step at 1 row). Reading the cache directly removes it
> and measured 10.1 -> 25.5 tok/s on the no-spec arm, but draft acceptance drops from ~60% to ~41-50%, so
> it stays off until that is understood.

The fix shipped inside the image, gated behind `VLLM_B12X_INDEXER_DIRECT_GATHER=1` (or a marker file, so
a running server can be switched within one boot). What had never been measured is the **end-to-end
effect with DSpark on**, which is the only configuration that matters here.

**Measured:**

| level | `proto2` | `proto2-dgather` | delta |
|---|---|---|---|
| c1 | 35.4 | **51.9** | +46.6 % |
| c3 | 78.9 | **101.4** | +28.5 % |
| c5 | 103.0 | **127.6** | +23.9 % |
| c6 | 117.0 | **143.4** | **+22.6 %** |
| sum | 334.3 | **424.3** | **+26.9 %** |

Gates pass in all three passes. **Acceptance is 53.5 / 51.4 / 53.7 %, unchanged** against `proto2`'s
~52 %, and tokens per step are 4.712 / 4.580 / 4.726 against 4.571. **The acceptance penalty that kept
the switch off does not reproduce on proto2.** Spreads are 14.6 / 4.3 / 6.9 / 4.7 %, so the c1 spread is
wide, but every level is far outside it.

Against the reference's same-day 66.4 / 118.4 / 139.0 / 161.0 (sum 484.8), the gap narrows from **27.3 %
behind at c6 to 10.9 %**, and from **31.0 % behind on the sum to 12.5 %**. Still not met, but this is the
largest single gain of the run and it came from our own repo -- a documented, quantified, gated
optimisation that had been left off on the strength of an acceptance regression.

**Per the loop's re-measure rule this is not standing yet**; `proto2-dgather2` is a second identical run
and must land before the arm is treated as the new best.

### The switch is promoted, and the promotion is verified from the pin

Round 45 kept `VLLM_B12X_INDEXER_DIRECT_GATHER=1` but only through `SERVE_EXTRA_ENV`, so it was not
actually kept. It is now in `configs/pin.main-029.env` (with the reasoning inline) and in
`scripts/05-serve.sh`'s forwarded `-e` list -- it is read inside the worker, so it must reach the
container -- with a `:-1` default so an unset variable cannot repeat round 29's empty-string crash.

Running the **stock** arm, no extra env:

```
$ docker exec vllm-ds4-0731 env | grep INDEXER_DIRECT_GATHER
VLLM_B12X_INDEXER_DIRECT_GATHER=1
```

and the win reproduces: 55.5 / 99.6 / 131.7 / **150.4**, sum 437.2. Three runs of the same
configuration now exist (`proto2-dgather` 424.3, `proto2-dgather2` 422.6, `proto2-dg` 437.2), which gives
an honest run-to-run range for the kept arm: **c6 139.6-150.4 (7.8 %) and sum 422.6-437.2 (3.5 %)**.

Against the reference's same-day 66.4 / 118.4 / 139.0 / 161.0 (sum 484.8) the criterion is **still not
met** -- ~10 % behind at c6 and ~11 % behind on the sum -- but that is down from 27.3 % and 31.0 % before
the switch, and the remaining deficit is now small enough to be attacked stage by stage on the new
baseline rather than in the aggregate.

The largest item left on the new arm is the MoE: round 41 priced the FFN at +9.1 ms at c1 rising to
+50.1 ms at c6, and the round-45 launch dump still shows `b12x::tp_moe_dynamic_launch` as the biggest
remaining host op (362 ms of CUDA time in the eager capture). b12x documents three MoE tile-shape
overrides it calls benchmarking knobs -- `B12X_MOE_TILE_MN`, `B12X_DYNAMIC_TILE_MN`,
`B12X_DYNAMIC_SWAP_AB` -- and **none has ever been tried on this rig**. That is the next measurement.

### The MoE tile override is a wash (negative)

First attempt at the three b12x tile-shape knobs, which it documents as benchmarking overrides and which
had never been tried on this rig: the kept arm plus `B12X_DYNAMIC_TILE_MN=64x128`.

| level | `proto2-dg` (kept) | tile 64x128 | delta |
|---|---|---|---|
| c1 | 55.5 | **58.2** | +4.9 % |
| c3 | 99.6 | 96.8 | -2.8 % |
| c5 | 131.7 | 126.4 | -4.0 % |
| c6 | **150.4** | 144.3 | -4.1 % |
| sum | **437.2** | 425.7 | -2.6 % |

Gates pass in all three passes, acceptance 51.0-55.1 % and tokens per step 4.555-4.830 are fine, so this
is a real A/B and not a correctness casualty. But **425.7 is inside the kept arm's own run-to-run range**,
which three runs now put at sums 422.6-437.2, so the override buys nothing measurable. The MoE's
remaining +50.1 ms at c6 (round 41, by removal) is therefore not addressable by tile shape, at least not
by this one. Recorded as a negative; the other two knobs (`B12X_MOE_TILE_MN`, which gates the *micro*
backend our decode band does not use, and `B12X_DYNAMIC_SWAP_AB`, a dev override) are unlikely to differ
and are left recorded rather than run.

### The win confirmed at the kernel level, and the new top item is the MoE

Re-running the round-45 instrument (shapes + CUDA-time dump, `--enforce-eager`, no stacks) on the **kept**
arm -- which now carries `VLLM_B12X_INDEXER_DIRECT_GATHER=1` from the pin -- shows the copy collapsing:

| host op | old arm (round 45) | new arm | change |
|---|---|---|---|
| `aten::copy_` | **433.03 ms** | **22.33 ms** | **19x less** |
| `elementwise_kernel<...direct_copy...>` | 416.56 ms | not in the top list | gone |
| `b12x::tp_moe_dynamic_launch` | 362.49 ms | **304.39 ms (42.55 %, now #1)** | |
| `vllm::all_reduce` | 265.62 ms | 74.22 ms | |
| `aten::mm` | 117.91 ms | 98.10 ms | |
| `b12x::blockscaled_serialized` | 103.02 ms | 84.95 ms | |
| `aten::bmm` | 68.75 ms | 49.51 ms | |

`aten::copy_` falls by **19x, from the largest single item to 3 %**, which is exactly the full-cache
indexer gather the switch removes. The window is not identical between the two captures (25 graph
replays here against 35, plus 5 prefill captures), so the other columns are indicative rather than
matched -- but the copy's collapse is unambiguous and confirms the mechanism.

**With the copy gone, the MoE is now by far the largest item at 42.55 % of kernel time.** Everything the
repo has tried against it has come back negative or a wash: our kernel is 10-13x faster than FlashInfer's
in the head-to-head, the A16 activation format is a wash, and this round's tile-shape override is inside
the arm's run-to-run range. So the largest remaining item is the one where the kernel is already the
fastest available and no configuration knob moves it.

**Where the criterion now stands**, honestly, with three runs of the kept configuration:

| level | kept arm (range over 3 runs) | reference | gap |
|---|---|---|---|
| c1 | 51.9 - 55.5 | 66.4 | 16 - 22 % behind |
| c3 | 98.2 - 101.4 | 118.4 | 14 - 17 % behind |
| c5 | 127.6 - 131.7 | 139.0 | 5 - 8 % behind |
| c6 | **139.6 - 150.4** | **161.0** | **6.6 - 13.3 % behind** |
| sum | **422.6 - 437.2** | **484.8** | **9.8 - 12.8 % behind** |

Not met -- the reference is ahead at every level -- but the gap has gone from 27.3 % / 31.0 % to single
digits at c5 and c6. The remaining deficit is largest at c1, which is the batch-independent floor: the
fixed per-layer work that rounds 41-43 attributed to launch overhead.

## 2026-09-17 (round 48): correcting round 40's DSpark attribution

Round 40 measured the target forward by turning speculative decoding off (`DISABLE_DSPARK=1`, one token
per step) and reported the difference against the normal arm as "DSpark draft + verification: +24.7 ms at
c1 rising to +86.2 ms at c6". **That attribution is wrong, and the reason is rows per step.**

With speculation off, one engine step processes `concurrency` rows -- 1, 3, 5, 6 at the four levels. With
DSpark k=7 it processes `concurrency x 8` -- 8, 24, 40, 48. The two arms therefore differ in *two* ways,
not one, and the measured difference is the draft's own work **plus the target's scaling from 6 rows to
48**:

| rows per step | configuration | step time |
|---|---|---|
| 1 | no-DSpark, c1 | 105.3 ms |
| 3 | no-DSpark, c3 | 128.2 ms |
| 5 | no-DSpark, c5 | 145.3 ms |
| 6 | no-DSpark, c6 | 148.1 ms |
| 8 | DSpark, c1 | 130.0 ms |
| 24 | DSpark, c3 | 175.0 ms |
| 40 | DSpark, c5 | 223.0 ms |
| 48 | DSpark, c6 | 234.0 ms |

The no-DSpark curve is steep at tiny batch (+8.6 ms per row from 1 to 6 rows) and then flattens; the
DSpark points sit close to where that curve would continue. So the +86.2 ms at c6 is **mostly the
target's own row scaling**, not draft overhead, and the draft itself looks close to free in step time --
which is what speculative decoding is supposed to look like, since the extra rows are what buy the 4.6x
token throughput.

**It cannot be separated within the protocol.** A clean split needs a no-DSpark point at 48 rows, i.e.
concurrency 48, and the protocol fixes the levels at 1 3 5 6. So the correct statement is the weaker one:
the two arms' step times are not comparable per row, and no draft-versus-target split should be quoted
from round 40. The per-stage numbers obtained by *removal* (FFN, all-reduce, indexer, LM head, mHC) are
unaffected, because each of those held the configuration fixed and moved one stage.

Also checked this round, since the draft was the next suspect: **the draft backbone is already graphed.**
`v1/worker/gpu/spec_decode/dflash/speculator.py` in the built image carries

```
# Graph the draft transformer only. sample_draft uses the shared
# target lm_head; capturing that GEMM + TP all-gather on 2-node GB10 sets
# lm_head.weight to inf.
if wants_full and supports_full:
    logger.info("%s CUDA graphs: backbone FULL, lm_head eager", ...)
    cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
```

which is what `patch_dspark_backbone_cudagraph` exists to reach the hard way. So the draft is not running
eager, and there is no launch-overhead lever there.

## 2026-09-17 (round 49): sweeping the overlays we wrote but do not apply

Round 45's win came from an optimisation that was already written in our own code and merely switched
off, so this round swept for others. `patches/apply_overlays.py` defines **63 `patch_*` functions; 47 are
called by `apply_main` and 16 are not.** Of those 16, exactly one docstring mentions a measurement or a
win, and it is a *revert*:

```
patch_dspark_fullstep_revert   Undo full-step capture. Graphing sequential sample cut accept and tok/s.
```

The rest are diagnostics (`patch_kv_cache_dbg`, `patch_logit_dump`), reverts
(`patch_indexer_packed_insert_revert`: "Drop the compressor sidecar. It doubled indexer-K memory and
OOM'd spark2"), the three compile-related ones this file already covers, and version-specific fixes
whose needles do not match this base (`patch_dspark_backbone_none` expects `"draft graphs drop token
accept"`, which proto2 does not contain). **So the documented-but-disabled-win pattern was unique to the
indexer gather, and that sweep is closed.**

### The one deliberate exclusion, tested (negative)

`patch_deep_gemm_sm12x_guard` is applied by `apply()` but deliberately **not** by `apply_main`. It makes
`is_deep_gemm_supported()` return False on device family 120, on the grounds that the compiled DeepGEMM
only targets SM100/SM103. Round 42 gave a reason to test it: forcing *just the mHC prenorm GEMM* off
DeepGEMM was ~1 % faster at every level (104.2 against 105.3 ms at c1, 146.0 against 148.1 at c6), so
excluding DeepGEMM globally might have been worth a similar amount.

It breaks the engine. Applied as a bind-mounted `utils/deep_gemm.py` -- verified effective, since the
`DeepGEMM E8M0 enabled on current platform` line stops appearing -- the arm never becomes healthy and
dies at worker init with **the identical assertion round 26 fixed from the other side**:

```
RuntimeError: Assertion error (/opt/vllm/.deps/deepgemm-src/csrc/utils/layout.hpp:113):
  sf.size(-2) == ceil_div(mn, gran_mn)
```

The mechanism is the same coupling: `deep_gemm_fp8_o_proj`'s recipe selection reads the DeepGEMM support
/ E8M0 state, so reporting DeepGEMM unsupported flips the scale layout to one the *still-running*
DeepGEMM FP8 einsum rejects. **So the guard's absence from `apply_main` is correct**, and this closes the
last of the uncalled overlays that looked like it might be a win. Negative, recorded.

## 2026-09-17 (round 50): the plain BF16 GEMMs are legitimate, not a fallback

With the indexer copy gone the kept arm's eager dump still showed **148 ms of plain GEMMs** -- `aten::mm`
98.1 ms over 828 calls and `aten::bmm` 49.5 ms over 276 -- which in an FP8-quantised model looks like an
unquantised fallback. It is not, and both families check out.

**The bmm is the b12x WO path itself.** `try_b12x_wo_proj`'s own docstring says it: "Fused inv-RoPE FP8 +
same dequant as SM12x einsum, then grouped bmm. ... replace torch.einsum with bmm + cached WO-A." The
recorded shapes confirm it:

```
215 calls  [4, 8, 4096] x [4, 4096, 1024] -> [4, 8, 1024]   BFloat16
 43 calls  [4, 84, 4096] x [4, 4096, 1024] -> [4, 84, 1024]
```

`4` is `o_groups` after TP=2 splits the 8 groups, `8`/`84` are token counts, `4096` is the group width
and `1024` is `o_lora_rank`. That is the documented grouped WO-A multiply, not the `else` branch of
`o_proj.py` -- which is only reached when `try_b12x_wo_proj` returns None, and that happens only when
`VLLM_USE_B12X_WO_PROJECTION=0`, the import fails, or `o_in.shape[0] > 256` (prefill, not decode).

**The `aten::mm` family is BF16 with bias against weight-shaped operands** --
`[8,4096] x [4096,256|2048|64|512|1024]`, with a `Scalar` third argument, i.e. `F.linear(x, w, bias)` --
and the counts are per *step*, not per layer (215 / 25 replays ~ 8.6), which fits small coefficient
projections rather than a main projection. In DSV4 those are the mHC coefficient generators
(`hc_attn_fn` / `hc_ffn_fn`), which are legitimately unquantised. No launcher Python frame exists for
either family, because both run inside Inductor's `execute_context_*` regions, so they cannot be named
that way -- but the bmm is positively identified from its shapes and the code, and nothing in the mm
family has the shape of a main projection.

**Recorded as a negative**: suspicion raised, mechanism checked, no unquantised fallback on the main
path. Two side notes worth keeping: `try_b12x_wo_proj` declines above 256 tokens, so **prefill uses a
different WO path than decode** (not our gap, but a real asymmetry); and the `DBG wo_proj` prints it
guards are still in the shipped file, which is why they appear in old logs.

## 2026-09-17 (round 51): upstream re-check

The contract asks for this every session, and it had not been run since round 23. Read with `gh`
(2.101.0, authenticated as `maci0`) on 2026-09-17. **No push, commit, comment or review was made** --
HANDOVER forbids upstream writes and the contract wants the owner's agreement first.

**Tracked vLLM PRs -- all nine OPEN, none merged:**

| PR | state | mergeable | title | last touched |
|---|---|---|---|---|
| #53425 (ours) | OPEN | MERGEABLE | SM12x FlashInfer sparse MLA kernel block size 64 | 2026-09-14 |
| #53522 (ours) | OPEN | MERGEABLE | Gate indexer paged MQA metadata on DeepGEMM support | 2026-09-14 |
| #53271 (ours) | OPEN | MERGEABLE | Validate computed device pointers before copy | 2026-09-14 |
| #46716 (ours) | OPEN | MERGEABLE | Fix shared-memory all-reduce deadlock across nodes | 2026-09-14 |
| #53055 | OPEN | **MERGEABLE** | Guard DeepGEMM in `mhc_pre_broadcast_tilelang` with a torch fallback | **2026-09-16** |
| #47988 | OPEN | UNKNOWN | Handle E8M0 block scales in CUTLASS and Triton FP8 linear kernels | 2026-09-12 |
| #52499 | OPEN | MERGEABLE | Fix DSV4 sparse MLA spec-decode shapes on SM120 FlashInfer path | 2026-09-06 |
| #50645 | OPEN | UNKNOWN | Guard `mhc_pre_broadcast_tilelang` on DeepGEMM support | 2026-08-23 |
| #41834 | OPEN | CONFLICTING | Add SM12x support for DeepSeek V4 Flash with essential fixes | 2026-09-09 |

**DeepGEMM -- all three OPEN:** #417 (2026-09-16, "8b1392b removes SM12x pure-fp8 1d1d kernels and
aliases `fp8_gemm_nt` to the fp4 dispatcher"), #419 (2026-08-28), #403 (2026-08-29).
**eugr:** `spark-vllm-docker#348` OPEN (2026-08-23).

**No new movement since the round-23 check, and nothing actionable:** all four of ours carry only the
`github-actions` welcome comment, so there is no review to answer.

**One item is directly load-bearing for work in this session, and should be watched.** `#53055`
("Guard DeepGEMM in `mhc_pre_broadcast_tilelang` with a torch fallback") is the **only** tracked PR that
has moved recently, and it is the upstream fix for exactly the call site that blocks the compile port:
`patch_mhc_tf32_uncaptured`'s docstring records that its needle "predates `pr-53055.diff`, which folds the
local `tf32_hc_prenorm_gemm` import into a guarded `is_deep_gemm_supported, tf32_hc_prenorm_gemm` line",
which is why both mHC overlays are parked. If `#53055` merges, the mHC DeepGEMM guard arrives upstream and
break 4's custom-op work should be re-evaluated against it rather than carried as a local overlay. `#50645`
is the older, still-UNKNOWN sibling of the same fix.

**Action taken: none beyond reading**, which is the correct outcome given the standing constraints.

### Break 5 cleared, and break 6 named

Rounds 31-35 established that break 5 needs a real custom op -- `allow_in_graph` cannot substitute because
it still runs the function on FakeTensors, where TileLang's cache raises
`unhashable type: non-nested SymInt`. This round built the first one and it works.

**What was done.** `mhc_pre_broadcast_tilelang` -- the outermost mHC entry point, which is where
`model.py:1284` enters the path -- is registered with `direct_register_custom_op` plus an explicit
`fake_impl`, and its public name is then rebound to a wrapper that calls
`torch.ops.vllm.mhc_pre_broadcast_tilelang`. Because `model.py` imports that name at module load,
**`model.py` needs no patch at all**. The fake impl mirrors the real function's own allocations, all
derivable from input shapes:

| output | shape | dtype |
|---|---|---|
| `residual_out` | `(T, hc_mult, H)` | bfloat16 |
| `post_mix` | `(T, hc_mult, 1)` | float32 |
| `comb_mix` | `(T, hc_mult, hc_mult)` | float32 |
| `layer_input` | `(T, H)` | bfloat16 |

**Result:** `Attempted to call function marked as skipped` drops to **0** and the TileLang frame is gone.
Two things had to be right, both learned the hard way:

- **The wrapper's signature must mirror the original exactly, defaults included.** The first attempt
  declared `n_splits: int` without its `= 1`, and the engine died with
  `vllm::mhc_pre_broadcast_tilelang() is missing value for argument 'n_splits'` -- `infer_schema` puts
  the Python defaults into the schema, and `model.py` relies on that one.
- **`torch.Tensor | None` is accepted by `infer_schema`** and becomes `Tensor?`, which is what makes
  `norm_weight` / `fn_broadcast` expressible at all.

Compilation then advances to **break 6**, which is a different failure mode entirely:

```
torch._dynamo.exc.Unsupported: Unsupported hasattr call
  Explanation: Dynamo does not know how to trace the function `TritonKernelVariable()`
  Hint: Avoid calling `hasattr(TritonKernelVariable, arg_names)` in your code.
from user code:
  model.py:1543 forward -> model.py:1337 x = self.attn(positions, x, None)
  -> attention.py:516 forward -> attention.py:544 _split_qkv_and_norm
```

So the next obstacle is a Triton kernel object being introspected in the attention path
(`_split_qkv_and_norm`), not TileLang. **The port is progressing again: one break cleared per round is
the rate, and four more mHC entry points remain on the same pattern** -- `mhc_pre_tilelang`,
`mhc_fused_post_pre_tilelang`, `mhc_post_tilelang`, `hc_head_fused_kernel_tilelang`.

Artifacts: the patched file is `spark1:~/fix10/tilelang.op1.py`, regenerable with
`.scratch/patch_mhc_op1.py`.

### Break 6 cleared, break 7 cleared, break 8 root-caused -- and the whole chain explained

Three results this round, one of which changes the shape of the port.

**Break 6 cleared.** `fused_q_kv_rmsnorm` (`models/common/ops/fused_qk_rmsnorm.py`, reached from
`attention.py:544 _split_qkv_and_norm`) is a `VllmTritonJitKernel` wrapper -- same class of problem as
break 5, and the same fix works:

```python
_b12x_fused_q_kv_rmsnorm_impl = fused_q_kv_rmsnorm
_b12x_drco(op_name="fused_q_kv_rmsnorm",
           op_func=_b12x_fused_q_kv_rmsnorm_op,
           fake_impl=_b12x_fused_q_kv_rmsnorm_fake)

def fused_q_kv_rmsnorm(...):            # rebound AFTER _FUSED_Q_KV_RMSNORM_KERNEL is built
    return torch.ops.vllm.fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)
```

The fake impl is trivial because the real function allocates `empty_like` on both inputs:
`return torch.empty_like(qr), torch.empty_like(kv)`. The rebind must land at the **bottom** of the
module -- `attention.py` imports the name, so the module body has to finish first.
Artifact: `spark1:~/fix10/qk.op.py` (md5 `497a5ee9bb33441369cec7b084a7a4a6`), built by
`.scratch/patch_qk_op.py`.

**Break 7 cleared, and it was not a kernel at all.** The next failure was a *logging* break:

```
RuntimeError: logging.Logger method not supported for non-export cases
  Developer debug context: method: <Logger vllm.models.deepseek_v4.nvidia.b12x_sparse (INFO)>.info_once
  b12x_sparse.py:158 _get_b12x_plan -> logger.info_once(...)
```

`b12x_sparse.py` is **our own overlay** (`patches/files/dsv4_b12x_sparse.py`; the shipped file is
byte-identical to the image's copy, which was verified by diff before editing). The `info_once` fires
only on a plan-cache miss, and the cache miss *is* the first -- traced -- invocation, so Dynamo always
sees it. Fix is a one-line guard, kept in the overlay:

```python
if not torch.compiler.is_compiling():
    logger.info_once(...)
```

**Break 8 is where the port stops being a break-chasing exercise.** With 6 and 7 cleared, Dynamo failed on
our own `sm12x_b12x_kernels.py:1086`:

```
RuntimeError: Worker failed with error 'torch.* op returned non-Tensor
  Developer debug context: example_value type: bool; op: call_function;
    target: <function is_current_stream_capturing at 0xfca9aa5374c0>
```

The call is inside a leftover `DBG wo_proj` debug print in `try_b12x_wo_proj`. But the reason it is
*fatal* -- and the reason breaks 5, 6, 7 and 8 were all fatal rather than harmless fallbacks -- is one
line we do not own:

```
/opt/vllm/vllm/compilation/wrapper.py:150
    self._compiled_callable = torch.compile(compiled_ptr, fullgraph=True, dynamic=False, ...)
```

**`fullgraph=True` is unconditional.** Under `VLLM_USE_BREAKABLE_CUDAGRAPH=1` the pin forces
`CompilationMode.NONE` (`config/vllm.py:796`), so the DSv4 model code was never required to be
Dynamo-clean. Re-enable compilation and *every* graph break becomes an `Unsupported` error instead of a
fallback. That is the entire chain, and it is open-ended: `sm12x_b12x_kernels.py` alone contains **12**
`torch.cuda.is_current_stream_capturing()` call sites (lines 197, 210, 270, 551, 622, 743, 822, 898, 978,
1194, 1407, 1545, 1590), each returning a `bool` into the graph, before counting anything in vLLM's own
DSv4 model code.

**So the laziest test of the hypothesis is the one line.** `fullgraph=False` makes all those breaks
ordinary fallbacks again while still letting Inductor fuse the rest and the cudagraph manager wrap it.
Artifact: `spark1:~/fix10/wrapper.nofullgraph.py` (md5 `ec955ed7ccf9accc7e6c1b47bf99ff5d`), built by
`.scratch/patch_nofullgraph.py`, which asserts the needle appears exactly once.

Arm `proto2-cgfg0` is `proto2-dg` plus exactly two changes -- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` and the
`wrapper.py` mount -- so a result is attributable. The four remaining mHC custom ops and the 12 capture
checks stay unbuilt until a measurement says the compiled regime is worth paying for.

**Method note, recorded because it cost nothing but could have cost two boots:** the first hypothesis this
round was that `MAX_CUDAGRAPH_CAPTURE_SIZE=36` in `configs/pin.main-029.env` was dropping the c5/c6 decode
graphs (40 and 48 rows at 8 rows per request). It is wrong: `harness/run-arm.sh` already forces
`MAX_CUDAGRAPH_CAPTURE_SIZE=48`, and the saved engine log for the best arm confirms
`cudagraph_capture_sizes: [1, 2, 4, 8, 16, 24, 32, 40, 48]`. **Read the runner, not the pin, before
believing a pin value is live.** The pin value is dead for every measured arm.

**Incidental hazard, recorded:** a bind-mount source that does not exist yet is silently created as a
root-owned *directory* by Docker rather than failing. `scp` then refuses to overwrite it
(`cannot overwrite directory ... with non-directory`) and the container mounts a directory over a `.py`
file. Check `ls -ld` on both nodes after staging a new mount.

**The `fullgraph=False` test acquired a second variable, and the second variable is not optional.**
The first run of `proto2-cgfg0` (breakable cudagraph off + `fullgraph=False`, AOT left at its pin default)
died with a new and much more specific error:

```
RuntimeError: Worker failed with error 'Graph breaks are not supported with aot compile.
  Please use torch.compile(fullgraph=True).'
```

`configs/pin.main-029.env` runs with `VLLM_USE_AOT_COMPILE=1` (`aot=1` in the serve line). AOT compile
requires a single graph by construction, so it rejects `fullgraph=False` outright. The two knobs are
therefore coupled: **you may have graph breaks, or you may have AOT, but not both.**

That leaves exactly two paths to the compiled regime, and both cost something real:

- **Path A -- breaks allowed, AOT off.** `fullgraph=False` + `VLLM_USE_AOT_COMPILE=0`. One boot, no code
  to write, but it gives up AOT (previously measured at 318.9 vs 334.3 in the `NONE` regime).
- **Path B -- AOT kept, breaks cleared.** `fullgraph=True` + AOT on means every one of the 12
  `torch.cuda.is_current_stream_capturing()` sites in `sm12x_b12x_kernels.py`, the TileLang
  `hasattr(TritonKernelVariable, 'arg_names')` sites, and whatever else is still hidden must be removed
  or wrapped before a single token is served.

**Path A was taken next**, as arm `proto2-cgfg0aot0`, because it answers the actual open question -- *is the
compiled regime worth anything here at all* -- for the price of one boot, and Path B is only worth paying
for if the answer is yes. If Path A is not faster than `proto2-dg`, the compile port closes as a recorded
negative and the four remaining mHC custom ops are not built.

### The gap is attributed: the reference compiles, we do not

This is the most load-bearing measurement in the session, and it came from reading the two engine logs
rather than from any new arm.

| | reference `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | ours `vllm-spark-0731:main-029-proto2` |
|---|---|---|
| vLLM | `0.25.2.dev0+g752a3a504` (2026-07-14) | `0.2.1.dev0+gf37c550bf` (2026-09-17) |
| path | `/usr/local/lib/python3.12/dist-packages/vllm` | `/opt/vllm/vllm` |
| `compilation_config.mode` | **`CompilationMode.VLLM_COMPILE`** | **`CompilationMode.NONE`** |
| `cudagraph_capture_sizes` | `[1, 2, 4, 8, 16, 24, 32, 40, 48]` | `[1, 2, 4, 8, 16, 24, 32, 40, 48]` |
| `is_current_stream_capturing` sites | **16** | **42** |
| `torch.compile(fullgraph=)` | `True` (`compilation/wrapper.py:150`) | `True` (`compilation/wrapper.py:150`) |

Evidence: `spark1:~/refbase-run.log` for the reference (`max_cudagraph_capture_size': 48`,
`cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48]`); `spark1:~/goal/proto2-dg-engine.log`
for ours (`cudagraph_capture_sizes: [1, 2, 4, 8, 16, 24, 32, 40, 48]`, `'mode': <CompilationMode.NONE: 0>`).
Both arms run `num_speculative_tokens: 7` with capture size 48, so **the protocol matches and the
compilation mode is the difference.**

**Why ours lands on `NONE` and theirs does not.** Our base auto-selects it. `config/vllm.py:77` defines
`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`, and **`DeepseekV4ForCausalLM` is in that set** -- alongside
`DeepseekV32ForCausalLM`, `KimiK3*`, `MiniMaxM3Sparse*`, `Qwen4Exp*`, `Glm5Next*`. When the model's
architecture is in the set and `VLLM_USE_BREAKABLE_CUDAGRAPH` is unset, `_maybe_enable_breakable_cudagraph`
sets it to `1` and then forces `CompilationMode.NONE` (`config/vllm.py:796`).

The reference build **does not have that set at all** -- its `config/vllm.py` gates on
`envs.VLLM_USE_BREAKABLE_CUDAGRAPH` directly (line 1133) with no per-architecture list. So anemll's July
build compiles DSv4 by default; our September build refuses to, on purpose, for this model family.
`configs/env.spark.sh:47` (`${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}`) agrees with the upstream default rather
than fighting it, which is why the pin is in `NONE`.

**So the remaining 9.8-12.8% is not a kernel, a library or a tuning constant. It is the compilation
regime, and it is blocked by Dynamo-cleanliness of the DSv4 b12x stack in our newer base.**
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` does lift the override -- and then every graph break is fatal, because
`fullgraph=True` is unconditional in all four `CompilationMode` values and `VllmBackend.__call__` asserts
it is called exactly once (`AssertionError: VllmBackend can only be called once` when
`fullgraph=False` is forced).

**Break chain as of this round** (each entry is one boot, and Dynamo stops at the first):

| # | site | class | status |
|---|---|---|---|
| 5 | `mhc/tilelang.py` `mhc_pre_broadcast_tilelang` | `hasattr(TritonKernelVariable, 'arg_names')` | cleared, custom op |
| 6 | `ops/fused_qk_rmsnorm.py` `fused_q_kv_rmsnorm` | `TritonKernelVariable` | cleared, custom op |
| 7 | `b12x_sparse.py:158` `logger.info_once` | `logging.Logger method not supported` | cleared, `is_compiling()` guard |
| 8 | `sm12x_b12x_kernels.py` `DBG wo_proj` prints | `bool` into graph (42 sites tree-wide) | cleared two ways: deleted the prints; folded `torch.cuda.is_current_stream_capturing` once in `vllm/__init__.py` |
| 9 | `utils/deep_gemm.py:536` via `ops/o_proj.py:99` `deepgemm_post_process_fp8_weight_block` | ctypes `_FuncPtr`, no traceable `__call__` | **open** |

Break 9 is a one-time weight repack executed lazily inside `forward`, reached during `profile_run`
(batch 12288 > 256, so `try_b12x_wo_proj` declines and the DeepGEMM fallback runs). It is not a compute
kernel that wants compiling; it wants to happen at load time. **That is the next concrete step**, and
after it the enumeration should move to a standalone `torch.compile(fullgraph=False, backend="eager")`
trace with `TORCH_LOGS=graph_breaks` so the remaining breaks are found in one shot instead of one boot
each.

**Correction recorded.** The plan to force `fullgraph=False` (`.scratch/patch_nofullgraph.py`) is dead as
a *shipping* configuration -- `VllmBackend` cannot accept multiple graphs -- but it remains the right way
to *enumerate* breaks, paired with an eager backend. The three arms it produced are negatives:
`proto2-cgfg0` died on `Graph breaks are not supported with aot compile`; `proto2-cgfg0aot0` died on
`AssertionError: VllmBackend can only be called once`; `proto2-cg` (fold + five mounts, AOT on,
`fullgraph=True`) advanced to break 9. **No arm has been metered yet, so no tok/s claim is made here.**

## Session check, 2026-09-17 (round 54)

All tracked items re-checked with `gh`. **Nothing has merged and nothing has moved since the round-51
check** except `#53055` (2026-09-16T17:14:53Z, previously noted).

| item | state | last update |
|---|---|---|
| #53425 SM12x FlashInfer sparse MLA block size 64 | OPEN | 2026-09-14 |
| #53522 Gate indexer paged MQA metadata on DeepGEMM support | OPEN | 2026-09-14 |
| #53271 Validate computed device pointers before copy | OPEN | 2026-09-14 |
| #46716 [CPU] Fix shared-memory all-reduce deadlock across nodes | OPEN | 2026-09-14 |
| #53055 Guard DeepGEMM in `mhc_pre_broadcast_tilelang` | OPEN | 2026-09-16 |
| #47988 E8M0 block scales in CUTLASS/Triton FP8 linear | OPEN | 2026-09-12 |
| #52499 DSV4 sparse MLA spec-decode shapes on SM120 | OPEN | 2026-09-06 |
| #50645 Guard `mhc_pre_broadcast_tilelang` on DeepGEMM support | OPEN | 2026-08-23 |
| #41834 SM12x DeepSeek V4 Flash support | OPEN | 2026-09-09 |
| DeepGEMM #419 SM12x pure-fp8 1d1d path | OPEN | 2026-08-28 |
| DeepGEMM #403 SM120 scale-factor layout transformation | OPEN | 2026-08-29 |
| DeepGEMM #417 SM12x pure-fp8 regression (issue) | OPEN | -- |

The four PRs are ours (`maci0`): `sm12x-dsv4-kernel-block-64`, `sm12x-indexer-paged-mqa-gate`,
`fix/kv-offload-bounds-check`, `fix-cpu-shm-allreduce-cross-node`. All four are `REVIEW_REQUIRED`; two are
`MERGEABLE` and two are `UNKNOWN`. `#53522` has a positive test-run review from `ivanusto`; `#53425` has a
substantive review from `kitch2400` about an import cycle that kills cold start when `vllm._aiter_ops` is
imported first, and it is already answered in the thread with the fix in `ed71de5`.

**The one thing that is blocked, and is blocked for a reason we must respect.** All four PRs share a
single failing check, `pre-run-check`, and it is not a code failure:

```
##[error]To reduce unnecessary pre-commit runs, each PR must have the 'verified', 'ready', or
'ready-run-all-tests' label, or the author must have at least 4 merged PRs (found 0).
DO NOT request for the label to be added if you are an AI agent.
```

So our PRs cannot reach CI until a human labels them, and the project has explicitly asked agents not to
ask for the label. **Action taken: none, and none is correct.** This is recorded so a future round does
not rediscover it and does not attempt a workaround.

**`#53055` remains the watch item.** It is the upstream fix for the `mhc_pre_broadcast_tilelang`
DeepGEMM guard that break 5 had to work around locally, and this round's finding that break 5 needs a real
custom op makes it *more* relevant, not less: if `#53055` merges with a torch fallback, the custom op may
be replaceable by the fallback plus a `direct_register_custom_op` wrapper for the fallback path.

## Break 9 root-caused, and the break class enumerated statically (round 55)

Round 54 ended on break 9 with a plan to find the remaining breaks one boot at a time. This round
replaced that plan with a static enumeration, and the enumeration says the problem is bigger and more
specific than "one break at a time".

### The reference's DSv4 stack is not our DSv4 stack

The reference is not merely an older vLLM that happens to compile. Its DSv4 implementation is a
different, simpler one:

| | reference `0.25.2` | ours `main-029` |
|---|---|---|
| `model_executor/warmup/jit_warmup_triton_helper.py` | **absent** | present, `VllmTritonJitKernel` |
| `common/ops/fused_qk_rmsnorm.py` | plain function, no kernel object | builds a `VllmTritonJitKernel` |
| `nvidia/ops/o_proj.py` | no `_sm12x_einsum_wo_a`, no `deepgemm_post_process` | memoized layout transform on the layer |
| `nvidia/b12x_sparse.py` | absent (ours) | present (our overlay) |
| `is_current_stream_capturing` sites | 16 | 42 |

So the reference compiles because its DSv4 path never grew the `VllmJitKernel` framework or the
lazy weight-repack. **Our base did grow them, and upstream then added the opt-out**
(`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` containing `DeepseekV4ForCausalLM`) instead of making the
new code Dynamo-clean. Both halves of that are upstream's, and both are visible in the two images.

### The enumeration: 7 files, one shared break site

Grepping the DSv4 forward path for `VllmTritonJitKernel` and crossing it against
`direct_register_custom_op` gives a *bounded* list instead of a boot-per-break:

| file | kernel uses | registers custom op |
|---|---|---|
| `models/common/ops/fused_qk_rmsnorm.py` | 2 | no (cleared round 54 by our overlay) |
| `common/ops/cache_utils.py` | 5 | no |
| `common/ops/fused_compress_quant_cache.py` | 2 | no |
| `common/ops/fused_indexer_q.py` | 3 | no |
| `common/ops/fused_mtp_input_rmsnorm.py` | 3 | no |
| `common/ops/save_partial_states.py` | 2 | no |
| `sparse_mla.py` | 2 | no |
| `common/ops/fused_inv_rope_fp8_quant.py` | 2 | **yes** -- the one upstream already fixed |

Every one of these routes through the same leaf:

```
jit_warmup_triton_helper.py:231
    @cached_property
    def _kernel_arg_names(self) -> tuple[str, ...]:
        arg_names = getattr(self.kernel, "arg_names", None)
```

`self.kernel` is a Triton kernel, Dynamo wraps it as `TritonKernelVariable`, and `hasattr` on that is
fatal. **So seven files, one break site, and no custom op is needed:** warm the `cached_property` once
in `_DecoratedTritonJitKernel.__init__`, guarded by `suppress` because the property legitimately raises
for kernels exposing neither `arg_names` nor `func`. The traced read then becomes an instance-`__dict__`
hit. One insertion, seven files cleared. Artifact `.scratch/patch_jitwarmup_argnames.py`, staging
`spark1:~/fix10/jitw.op.py` (md5 `842e2529cd7934f616dc5b17840f4035`).

This is the same *shape* as the round-54 fold of `torch.cuda.is_current_stream_capturing`: find the one
function Dynamo cannot const-fold, and make its value available without tracing its body. Both are worth
repeating as a pattern.

### Break 9: module arguments rule out the custom-op route, so it is a pre-pack

`o_proj.py:91-109` memoizes on the layer:

```python
cached = getattr(wo_a, "_sm12x_einsum_wo_a", None)
if cached is None or cached[2] != weight.data_ptr():
    weight, weight_scale = deepgemm_post_process_fp8_weight_block(...)
    wo_a._sm12x_einsum_wo_a = (weight, weight_scale, weight.data_ptr())
```

The cache is cold on the very first forward, and that first forward is `profile_run`. **A custom op
around `deep_gemm_fp8_o_proj` is not available**: it takes `wo_a` and `wo_b` as `nn.Module`, and a
`torch.library` schema cannot carry a module. So the only route is to make the cache warm before the
trace.

**`kernel_warmup` is too late.** `gpu_worker.py:807` calls it from `compile_or_warm_up_model`, which runs
*after* `determine_available_memory` (`gpu_worker.py:528`) has already compiled inside `profile_run`
(545 / 572). The repo's existing warmup extension is wired into `kernel_warmup`, so it cannot carry this.

The pre-pack therefore goes at the top of `determine_available_memory`, immediately after
`maybe_apply_startup_plan(self)` -- the last point guaranteed to run before the first forward, and ahead
of *both* `profile_run` call sites. It reproduces the traced call exactly, which is the reason it is safe:
same keywords, same `use_e8m0=True`, and `bmm_batch_size` read from `wo_a.bmm_batch_size`
(`attention.py:301` sets `self.wo_a.bmm_batch_size = self.n_local_groups`) rather than guessed from
`n_groups`. Layers that never got a `bmm_batch_size` are skipped, leaving the lazy path intact.
`deepgemm_post_process_fp8_weight_block` returns a *view* of `wo_a.weight` in the `is_bmm` branch, so
`weight.data_ptr()` still equals `wo_a.weight.data_ptr()` and the sentinel matches.

Artifacts: `patches/files/dsv4_warmup_ext.py` gains `deepseek_v4_wo_a_einsum_warmup`
(`~/fix10/dsv4_warmup_ext.op.py`, md5 `944dd5dafcf3c6310b24d819174b93ad`), and
`.scratch/patch_wo_a_prepack.py` inserts the call (`~/fix10/gpu_worker.op.py`, md5
`3a8fb83bc569420dbe25a69a809ce6a9`). Arm `proto2-cg2` carries eight mounts and is booting. **No tok/s
claim: nothing has been metered on any compile arm.**

### Breaks 9 and 10 cleared; 11 and 12 are one blocker, and its remedy is now certain

Arm `proto2-cg2` (nine mounts) advanced twice this round. The pre-pack is confirmed working on both
runs -- **`DSv4 wo_a DeepGEMM einsum pre-pack: 43 packed, 0 skipped, 1.75 s`** -- so all 43 layers are
packed before the first forward. Break 9 is closed.

**Break 10 -- `Data pointer comparison` -- was the predicted second half of break 9.** With the cache
warm, `o_proj.py`'s sentinel `cached[2] != weight.data_ptr()` is what Dynamo reaches, and a Python `int`
derived from a tensor is not expressible in the graph. Fixed by restructuring the guard so nothing
untraceable is evaluated while compiling, with behaviour outside compilation unchanged:

```python
if cached is None:                      _b12x_pack = True
elif torch.compiler.is_compiling():     _b12x_pack = False
else:                                   _b12x_pack = cached[2] != weight.data_ptr()
```

If the pre-pack did not run, `cached is None` still packs and the original failure returns loudly rather
than silently using a stale pack. Artifact `.scratch/patch_oproj_dptr.py` (`~/fix10/o_proj.op.py`, md5
`4f00f501c3604bf053db4c6bf64c9418`).

**Break 11 is `deep_gemm.py:501 fp8_einsum`.** `utils/deep_gemm.py` resolves its implementations lazily
and forwards with a bare `return _impl(*args, **kwargs)`; the implementations live in the third-party
`deep_gemm` package, which Dynamo skip-lists:

```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
  Hint: ... if it is traceable, use `torch.compiler.allow_in_graph`.
```

One tempting reading was that the reference must avoid this call. It does not. The reference's
`deep_gemm_fp8_o_proj` calls `fp8_einsum` **unconditionally** -- no `try_b12x_wo_proj`, no `use_fp8`
branch -- and its `fp8_einsum` wrapper is byte-identical to ours. **The call is simply not skip-listed in
that build.** Ours is. So this is another instance of the same story as the compilation mode: not a
kernel difference, a base-build difference.

**Break 12 is the answer to break 11, and it is torch stating the fix outright.** Registering the
resolved implementations with `allow_in_graph` before the first traced forward clears break 11 exactly as
the hint promises -- and immediately produces:

```
torch._dynamo.exc.ObservedRuntimeError: Dynamo failed to run FX node with fake tensors:
  call_function <built-in method fp8_einsum ...>
    got RuntimeError("Cannot access data pointer of Tensor (e.g. FakeTensor,
    FunctionalTensor). ... it is likely that we are erroneously tracing into a custom
    kernel. To fix this, please wrap the custom kernel into an opaque custom op.")
```

`allow_in_graph` makes the call a graph **leaf**, and Dynamo then executes that leaf with FakeTensors to
derive output metadata. A kernel that reads `data_ptr()` cannot survive that. **This is the repo's second
independent confirmation of the same rule** -- the first was `patch_mhc_tf32_customop`, where
`allow_in_graph` died on TileLang's `unhashable type: non-nested SymInt`.

**So the verdict on the mechanism is settled, and it settles the shape of the remaining work:**
every third-party kernel on the DSv4 forward path needs an opaque custom op with an explicit fake impl.
`allow_in_graph` is not a shortcut anywhere in this stack -- it only converts "skipped" into "fake
tensor". That is what `direct_register_custom_op` did for breaks 5 and 6, and it is what `fp8_einsum`
needs next.

**The fake impl for `fp8_einsum` is writable from the failing call itself**, which the error prints in
full:

```
fp8_einsum('bhr,hdr->bhd',
           (FakeTensor(12288, 4, 4096, fp8_e4m3fn), FakeTensor(12288, 4, 8, int32)),
           (FakeTensor(4, 1024, 4096, fp8_e4m3fn), FakeTensor(4, 1024, 8, int32)),
           FakeTensor(12288, 4, 1024, bfloat16),
           recipe=(1, 1, 128))
```

The output is the fourth argument `(12288, 4, 1024)`, so `fake_impl` is `torch.empty_like(out)`, the
tensor pairs flatten into separate tensor arguments, and `mutates_args` must be declared because the call
writes through `out`. The remaining question is only how many more entry points on this path need the
same treatment (`m_grouped_*`, `fp8_gemm_nt`, and the rest of the resolved `_impl` globals are all
candidates; the registration added this round touches them all at once, so the fake-tensor failure may
simply move to the next one).

**Method note worth keeping:** `patches/files/dsv4_warmup_ext.py` now carries
`deepseek_v4_deepgemm_allow_in_graph` with an explicit `.. warning::` recording that it is necessary but
not sufficient. It is left in place because the registration is harmless and the *second* failure it
produces is more informative than the first -- but it must not be mistaken for a fix on a later run.

## The gap is a FIXED per-step cost, not per-row work (round 56)

Three rounds of break-chasing produced no tok/s number. This round went back to the measurement side and
found something that changes what the gap *is*.

**Aggregate tok/s cannot separate the two halves of a throughput gap.** Our step could return fewer
tokens, or it could take longer; those need opposite fixes. `drive-meter.sh` already records enough to
split them, but the conversion is not obvious: its `drafts_per_req` is a **sum over the concurrent
requests**, so the engine step count is `drafts_per_req / concurrency`. That is confirmed exactly at c6
-- 640 drafts / 6 = 106.7 steps, and 512 tokens / 4.800 tokens_per_step = 106.7 -- and it is now a
committed instrument, `harness/step-time-gap.py`.

Same-day arms, three passes each, medians (`proto2-dg` 2026-09-17T20:21, `refg` 2026-09-17T12:55):

| level | ours step ms | ref step ms | delta | ours t/step | ref t/step | ours acc | ref acc |
|---|---|---|---|---|---|---|---|
| c1 | 86.5 | 73.4 | **+13.1** | 4.785 | 4.830 | 54.2 % | 54.6 % |
| c3 | 141.7 | 124.6 | **+17.1** | 4.655 | 4.907 | 52.3 % | 56.2 % |
| c5 | 176.2 | 162.3 | **+13.9** | 4.580 | 4.672 | 51.3 % | 52.7 % |
| c6 | 192.0 | 171.1 | **+20.9** | 4.800 | 4.578 | 54.4 % | 51.4 % |

**The delta is flat: median +15.5 ms, range +13.1 to +20.9 ms, across a batch range where the step time
itself more than doubles (86.5 -> 192.0 ms).** A per-row cost would grow with concurrency. This does not.
So the whole remaining gap is **one fixed cost paid every decode step, worth about 15.5 ms**, and nothing
that scales with batch is missing.

**Two things this settles.**

First, **acceptance and tokens per step are not the gap** and should stop being listed as candidates.
Acceptance is 51.3-54.4 % against 51.4-56.2 %, overlapping within pass-to-pass spread; tokens per step is
4.580-4.800 against 4.578-4.907, likewise. The contract's "our acceptance rate and tokens per step must
not be lower" holds as stated, and no draft-quality work is warranted.

Second, **the shape of the gap is exactly what a fixed per-step overhead predicts**, and it explains the
otherwise odd throughput profile: the relative gap is 16.4 % at c1 and 6.6 % at c6 precisely because a
constant 15.5 ms is 18 % of an 86.5 ms step and 8 % of a 192.0 ms step. Aggregate tok/s hid that by
turning one fixed cost into a level-dependent percentage.

**What a fixed ~15.5 ms/step can be.** The candidates are the ones that do not scale with rows: exposed
kernel-launch and Python overhead per step, and the parts of the step the breakable-cudagraph path leaves
*eager* between its captures. Prior work already named the largest known instance: the draft backbone is
graphed `FULL_DECODE_ONLY` while **`lm_head` stays eager** (HANDOVER, *The kernel composition*), and the
LM head was measured at ~5.3 ms. The reference, being compiled, has no breakable splits and no eager
lm_head.

**The next measurement is therefore not more compile work, it is a price on the cudagraphs we already
have.** `CUDAGRAPH_MODE` is consumed host-side by `scripts/05-serve.sh:212` into the compilation config
JSON, so `env CUDAGRAPH_MODE=NONE` is a clean one-variable arm against `proto2-dg`. If removing capture is
catastrophic, the capture works and the residual is fusion work that only compilation provides. If it
barely moves the step, the breakable-cudagraph path is not delivering what it claims and the fix is far
cheaper than the compile port. Arm `proto2-dg-cgnone` is booting.

## Cudagraph capture priced: worth 56 ms/step at c1, near-nothing at c6 (round 57)

Round 56 concluded the gap was "one fixed ~15.5 ms/step cost" and guessed it was launch/eager glue. **That
reading is now corrected by measurement**, and the correction matters because it changes which levels the
remaining gap can possibly live in.

Arm `proto2-dg-cgnone` is `proto2-dg` with exactly one change, `CUDAGRAPH_MODE=NONE` (`scripts/05-serve.sh:212`
consumes it host-side into the compilation config JSON). Three passes, both gates pass on every pass
(`gate_france ' Paris. The capital of Spain'`, `gate_9x8 '72, 9x9'`):

| level | proto2-dg (capture on) | proto2-dg-cgnone | spread (cgnone) |
|---|---|---|---|
| c1 | 55.5 | **32.7** | 18.3 % |
| c3 | 99.6 | **88.8** | 6.0 % |
| c5 | 131.7 | **135.9** | 1.7 % |
| c6 | 150.4 | **147.7** | 6.0 % |
| sum | 437.2 | **405.1** | |

**In step time** (`harness/step-time-gap.py proto2-dg-cgnone proto2-dg`):

| level | with capture | without | capture is worth |
|---|---|---|---|
| c1 | 86.5 ms | 143.1 ms | **-56.5 ms** |
| c3 | 141.7 ms | 153.1 ms | -11.4 ms |
| c5 | 176.2 ms | 170.3 ms | +5.8 ms |
| c6 | 192.0 ms | 184.8 ms | +7.2 ms |

**Capture removes 56.5 ms per step at c1 -- 40 % of the step -- and by c5/c6 it is inside the pass-to-pass
spread.** The c5/c6 "+5.8 / +7.2 ms" figures are **not** treated as a finding: both arms' c5/c6 spreads are
~6 %, which is ±11 ms at these step times, so a "capture costs us 7 ms at c6" claim is not supported. What
*is* supported is that capture's value collapses from 40 % of the step at batch 8 to unmeasurable at batch
48.

**This corrects round 56 rather than confirming it.** The flat +13 to +21 ms step-time delta is real, but
it cannot be a launch-overhead story at c5/c6, because at c5/c6 there is almost no launch overhead left to
remove -- capture already delivers nothing there. So the two halves of the gap have different causes:

- **c1 and c3, where we are 16 % behind:** the batch-8 and batch-24 steps are launch-dominated, and our
  breakable-cudagraph capture is *less complete* than the reference's compiled full graph. Same arena as
  the compile port.
- **c5 and c6, where we are 5-7 % behind:** capture is already spent. `cgnone` at c6 is 184.8 ms against the
  reference's 171.1 ms, so even with capture removed we are 13.6 ms/step slower. That residue has to be
  kernel execution time -- Inductor fusion, or a kernel difference -- and no amount of cudagraph tuning
  reaches it.

**A cheap consequence worth one arm.** If capture is worth 56 ms at c1 and nothing at c5/c6, then capturing
*everything up to 48* may be paying for capture sizes that do not want it -- and `cgnone`'s c6 step is
7.2 ms below `proto2-dg`'s, inside noise but on the favourable side. `MAX_CUDAGRAPH_CAPTURE_SIZE` selects
which sizes are captured (48 gives `[1, 2, 4, 8, 16, 24, 32, 40, 48]`), so a ceiling of 24 keeps the c1
(8-row) and c3 (24-row) captures and drops c5 (40) and c6 (48). Arm `proto2-dg-cg24` is booting. **The
expected size of any win is small** -- the c5/c6 capture deltas are inside spread -- so this is a cheap
probe, not a probable fix.

**Method note.** `harness/step-time-gap.py` is now committed. It exists because aggregate tok/s cannot
distinguish "fewer tokens per step" from "longer step", and the first round that used it immediately
overturned the previous round's own conclusion. Use it before theorising about any throughput gap.

### The capture ceiling is a negative: `proto2-dg-cg24`

Round 57 hypothesised that capturing up to 48 rows pays for capture sizes that do not want it, on the
strength of `cgnone`'s c6 step being 7.2 ms below `proto2-dg`'s. **Refuted.**

`proto2-dg-cg24` is `proto2-dg` with one change, `MAX_CUDAGRAPH_CAPTURE_SIZE=24`, which keeps the 8-row
(c1) and 24-row (c3) captures and drops 40 (c5) and 48 (c6). Three passes:

| level | proto2-dg | proto2-dg-cg24 | delta |
|---|---|---|---|
| c1 | 55.5 | 51.7 | -6.8 % |
| c3 | 99.6 | 96.5 | -3.1 % |
| c5 | 131.7 | 130.8 | -0.7 % |
| c6 | 150.4 | **147.5** | -1.9 % |
| sum | **437.2** | 426.5 | -2.4 % |

Worse at every level. And the decisive detail: **`cg24`'s c6 (147.5) is indistinguishable from
`cgnone`'s c6 (147.7), and both are below `proto2-dg`'s 150.4.** So dropping the 48-row capture does not
recover `cgnone`'s c6 step time; it lands on the same number, and that number is *worse* in aggregate
because the uncaptured runs also needed more steps (tokens per step 4.585 against 4.800).

That closes it two ways at once: the c5/c6 "capture costs 5.8-7.2 ms" figure was **pass-to-pass spread,
not a finding**, exactly as flagged when it was first written; and a capture ceiling is not a lever.
`MAX_CUDAGRAPH_CAPTURE_SIZE=48` stands. **`proto2-dg` remains the best measured arm at sum 437.2.**

**Two negatives from this round, both cheap and both now closed:** cudagraph capture is not paying for
itself at c5/c6 in either direction, and it cannot be profitably reduced.

## Session check, 2026-09-18 (round 57)

Re-checked with `gh`. **Nothing has merged since round 54. One item moved: DeepGEMM `#417`.**

| item | state | last update |
|---|---|---|
| #53425 SM12x FlashInfer sparse MLA block size 64 | OPEN, mergeable UNKNOWN | 2026-09-14 |
| #53522 Gate indexer paged MQA metadata on DeepGEMM support | OPEN, mergeable UNKNOWN | 2026-09-14 |
| #53271 Validate computed device pointers before copy | OPEN, mergeable UNKNOWN | 2026-09-14 |
| #46716 [CPU] Fix shared-memory all-reduce deadlock across nodes | OPEN, mergeable UNKNOWN | 2026-09-14 |
| #53055 Guard DeepGEMM in `mhc_pre_broadcast_tilelang` | OPEN | 2026-09-16 |
| #47988 E8M0 block scales in CUTLASS/Triton FP8 linear | OPEN | 2026-09-12 |
| #52499 DSV4 sparse MLA spec-decode shapes on SM120 | OPEN | 2026-09-06 |
| #50645 Guard `mhc_pre_broadcast_tilelang` on DeepGEMM support | OPEN | 2026-08-23 |
| #41834 SM12x DeepSeek V4 Flash support | OPEN | 2026-09-09 |
| DeepGEMM #419 SM12x pure-fp8 1d1d path | OPEN | 2026-08-28 |
| DeepGEMM #403 SM120 scale-factor layout transformation | OPEN | 2026-08-29 |
| DeepGEMM #417 SM12x pure-fp8 regression (issue) | OPEN | **2026-09-16** |

**`#417` is the one live thread, and this repo is already in it.** A DeepGEMM maintainer reports that
**`#447`** makes the SM12x dispatch dtype-driven end to end -- `fp8_gemm_nt` enters a unified fp8/fp4
dispatcher, the operand dtypes select the instantiation, `fp8 x fp8` compiles the 1D1D kernel with
`kIsFP4=false`, and unsupported combos become a loud `DG_HOST_UNREACHABLE` rather than silent corruption.
We have already replied on 2026-09-16 with measurements from a build of
`vllm-project/DeepGEMM@ad1f1726` on 2x GB10 (sm_121): the dtype-driven instantiation is confirmed, and
with packed ue8m0 scale factors the result is exact against a dequantised reference (relative error 0) at
1x128x256 and 7x512x4096. The reply also files a second, distinct failure mode -- a float32 SF with
`disable_ue8m0_cast=True` being read as packed ue8m0 by the SM120 kernel, because `layout.hpp:38/42` keep
a float SF in the SM90 float layout whenever that flag is set and the flag's clause never fires on that
path. **Action taken: none beyond reading; the thread is current and the ball is with DeepGEMM.**

**Why this now matters more than it did.** Round 57 established that the c5/c6 half of our gap is kernel
execution time, not launch overhead (cudagraph capture is already spent there). DeepGEMM is the
`linear_backend` family that was closed as "worse/unstable" -- **but it was closed on the regressed
dispatch that `#417` is about.** A dtype-driven dispatch that is exact and actually launches the pure-fp8
1D1D kernel is a different object from what that verdict was measured against. **This is a legitimate
item-4 reopening and the next thing to price after the compile port**, and it does not need an upstream
PR or a comment to anyone else's thread: it needs a wheel built from `ad1f1726` and one arm against
`proto2-dg` with `LINEAR_BACKEND=auto`. No contact made, and none is warranted until we have a number.

## One boot, all the breaks: the enumeration run (round 58)

Three rounds of one-break-per-boot is the wrong rate. This round made the break *list* obtainable in a
single boot, and the list is now closed enough to plan against.

**How.** `fullgraph=True` is what makes a break fatal, and `VllmBackend` refuses more than one graph, so
`fullgraph=False` alone dies on `AssertionError: VllmBackend can only be called once`. Replace *both*:
`.scratch/patch_wrapper_enum.py` compiles with `fullgraph=False` and `backend="eager"`, which has no
single-call assertion, and Dynamo then emits a break and keeps walking. With
`TORCH_LOGS=+graph_breaks` injected through `SERVE_EXTRA_ENV` (whitespace-split into `-e` by
`05-serve.sh:178`), one boot logs every break site on the whole forward path.

**One more thing had to be neutralised.** `fp8_einsum` is fatal even at `fullgraph=False`, because its
`data_ptr()` call raises *while Dynamo is tracing* -- that is an exception during tracing, not a graceful
graph break, so `fullgraph=False` cannot rescue it and it blocks everything behind it.
`.scratch/patch_deepgemm_enumstub.py` therefore returns the caller's own `out` buffer. Shapes stay right,
values are wrong on purpose, and **the stub must never appear in a measured arm** -- the run produces no
numbers. It is a diagnostic, and it is labelled as one in the file.

### The list

23 distinct sites. Filtered to the forward path, with their reasons:

| site | reason | owner |
|---|---|---|
| `sm12x_b12x_kernels.py:330, 807, 1093` | Dynamo cannot trace builtin `print` | **ours** |
| `jit_warmup_triton_helper.py:232` | `hasattr(TritonKernelVariable, 'arg_names')` | upstream |
| `b12x/attention/_shared/mla/merge.py:84, 85, 92` | `torch.* op returned non-Tensor`; unsupported method call | b12x (third-party) |
| `deep_gemm.py:1153` | `tf32_hc_prenorm_gemm` pybind call | upstream -- **PR #53055's target** |
| `multi_stream_utils.py:56` | Dynamo cannot break on `torch.fx.traceback.annotate` | upstream |
| `attention.py:852` | untraceable call | upstream |
| `import_utils.py:427` | `importlib.import_module` marked do-not-trace | upstream |
| `pynvml.py:2381`, `tvm_ffi/module.py:188,299`, `tilelang/.../diagnostics.py:110`, `tilelang/language/kernel.py:332`, `nvidia_cutlass_dsl/*`, `inspect.py:1766`, `<frozen>`, `<string>` | warmup/JIT/import noise, mostly off the decode path | mixed |

### Three things this settles

**1. Three of our own breaks are a `print`.** Not a kernel, not a library -- leftover diagnostic prints in
`utils/sm12x_b12x_kernels.py`. There are 24 in the file; the enumeration caught 330, 807 and 1093 because
those are the ones on the decode path. Rather than delete 24 blocks (some sit in exception handlers that
record why a fallback fired), the module now shadows the builtin with a no-op unless `B12X_DEBUG=1`:

```python
if os.environ.get("B12X_DEBUG", "") != "1":
    def print(*args: Any, **kwargs: Any) -> None:
        return None
```

The flag is read once at import, so the guard is a constant at trace time and Dynamo traces a trivial
function instead of an opaque builtin. `~/fix10/sm12x.op.py`, md5 `537f6a4c0368b015096bb6e32f4182e3`.

**2. The round-55 fix for `_kernel_arg_names` does not work, and the reason is worth recording.**
`_kernel_arg_names` is a `functools.cached_property`. Round 55 warmed it at construction so that the
traced read would be an instance-`__dict__` hit. **Dynamo still traces the property body**, because
`cached_property.__get__` is a `try`/`except KeyError` and Dynamo traces the `except` branch whether or not
it is taken -- and that branch is where `getattr(self.kernel, "arg_names")` lives. So the enumeration
correctly still reports line 232. **The fix is not to warm a cache; it is to stop the descriptor from
existing on the traced path** -- compute the tuple once and bind it as a plain instance attribute, so the
traced read is an ordinary `__dict__` lookup with no `try`/`except` and no attribute access on the kernel.
That is next round's first edit, and the enumeration run makes checking it one boot instead of a gamble.

**3. `deep_gemm.py:1153` is `tf32_hc_prenorm_gemm`, which is exactly what PR #53055 upstreams.** The repo's
`patch_mhc_tf32_uncaptured` / `patch_mhc_tf32_call_redirect` were parked because their needle predates
`pr-53055.diff`. The enumeration confirms the call site is live on the decode path and gives it a line
number. **If #53055 merges, this break is upstream's to fix and the parked overlays should be re-evaluated
against it rather than re-derived.**

**4. `b12x/attention/_shared/mla/merge.py:84,85,92` is third-party and needs an owner's agreement before
any contact.** It is `torch.* op returned non-Tensor` plus an unsupported method call inside b12x's MLA
merge, reached from our attention path. Recording it here is the correct step; opening a PR or commenting
on theirs is not, per the contract, until a measurement says it matters and the owner is asked.

## `_kernel_arg_names` fixed properly, and the reason the round-55 attempt could not work (round 59)

Round 58's enumeration refuted the round-55 fix for `jit_warmup_triton_helper.py:232`. This round replaces
it with one that addresses the actual mechanism.

**Why warming a `cached_property` cannot work.** `cached_property.__get__` is a `try`/`except KeyError`.
Dynamo traces the `except` branch whether or not the cache is populated, and that branch is where
`getattr(self.kernel, "arg_names")` lives. So the attribute access on the kernel happens on every traced
call no matter how warm the cache is. The descriptor itself has to leave the traced path.

**The obvious repair does not work either.** Rewriting the access cannot help, because
`TritonKernelVariable` (`torch/_dynamo/variables/functions.py:3602`) defines `tp_methods` and
`mp_subscript_impl` but **no `var_getattr`**. Any attribute read on it -- `hasattr` or `getattr` or plain
`.` -- is unsupported. There is nothing to rewrite the expression *to*.

**What works is making the value already exist as a plain instance attribute**, so the traced read is an
ordinary `__dict__` lookup. `@cached_property` is gone; the body is now an ordinary method
`_b12x_compute_kernel_arg_names`, bound in two places:

- **`VllmTritonJitKernel.__init__`** -- newly added. This is the hook that covers everything:
  `_AutomaticTritonJitKernel` declares `kernel` as a *class* attribute and defines no `__init__` of its
  own, so it inherits this one, and `_DecoratedTritonJitKernel` reaches it through `super().__init__()`.
- **`_DecoratedTritonJitKernel.__init__`**, right after `self.kernel = kernel` -- because that class sets
  the kernel as an *instance* attribute *before* calling `super().__init__()`, so the base-class bind
  would read nothing.

Both guarded with the module's existing `suppress`, so the original `TypeError` for a kernel exposing
neither `arg_names` nor `func` stays a construction-time condition rather than becoming a trace-time one.

Artifact: `.scratch/patch_jitwarmup_argnames.py` (rewritten), staging `spark1:~/fix10/jitw.op.py`, md5
`aa2a96fff43287532f0bc52f6e9bab6e`. **The verification is the enumeration run**, not a boot-and-hope:
`proto2-enum` should stop reporting line 232, and whatever it reports instead is the next break.

### `CUDAGRAPH_MODE=FULL` is a negative, and a large one

`proto2-dg-cgfull` is `proto2-dg` with one change, `CUDAGRAPH_MODE=FULL` (no piecewise splits -- one graph
for the whole decode step). It was the direct test of round 57's finding that the c1/c3 gap is capture
completeness: if the splits are what costs us there, removing them should pay.

| level | proto2-dg | cgfull | spread |
|---|---|---|---|
| c1 | 55.5 | **31.5** | 75.6 % |
| c3 | 99.6 | **74.6** | 79.9 % |
| c5 | 131.7 | **97.0** | 26.9 % |
| c6 | 150.4 | **121.9** | 26.6 % |
| sum | **437.2** | 325.0 | |

Worse at every level, and worse than `cgnone` (405.1) which removed capture entirely. The spreads are the
story: 75-80 % at c1/c3 means the arm is not stable, not merely slower -- consistent with capture
failing or re-capturing rather than with a clean no-splits graph. **`FULL_AND_PIECEWISE` stands, and this
is why the breakable path exists at all:** vLLM's own default for this architecture is not an
over-conservative choice, and forcing the whole eager model into one graph is actively bad here.

That closes the cheap config levers around capture. Both directions are now measured negatives:
less capture (`MAX_CUDAGRAPH_CAPTURE_SIZE=24`, `CUDAGRAPH_MODE=NONE`) and more capture
(`CUDAGRAPH_MODE=FULL`). **The c1/c3 gap cannot be bought by cudagraph configuration; it needs the
compiled regime, which is the compile port.**

### Enumeration v3: the `_kernel_arg_names` fix is confirmed, and the next break is ours

Re-ran `proto2-enum` with the rebuilt `jitw.op.py`. Two results.

**1. Line 232 is gone.** `jit_warmup_triton_helper.py:232` does not appear in the break list at all. The
`cached_property` was the problem and removing the descriptor was the fix -- no warming, no custom op, and
it covers every kernel class in one edit because the bind sits in `VllmTritonJitKernel.__init__`.
**The enumeration is what made this checkable**: the previous two attempts at this break were validated by
booting a nine-mount arm and reading whether it died further along, which is slow and ambiguous.

**2. The `print` class is confirmed closed, and a new break in our own code surfaced.** The
`sm12x_b12x_kernels.py` breaks are no longer the `print` ones. One remains, at a different line, with a
different reason entirely:

```
Attempted to call op with non-contiguous `out=` tensor
  Explanation: Dynamo does not support this.
from user code:
    sm12x_b12x_kernels.py:1144
      torch.bmm(a_ws, w_bmm, out=z_ws)
```

`z_ws` is `_z_ws[:, :tokens]` sliced out of a `(groups, cap_t, rank)` buffer, so it is non-contiguous
whenever `tokens < cap_t` -- which is the normal decode case. **This is our own WO-projection path**, the
`try_b12x_wo_proj` branch that serves every batch up to 256 rows, so it is squarely required work item 3.

Fixed in the overlay by computing into a fresh tensor and copying, gated so the shipping eager path is
untouched:

```python
if torch.compiler.is_compiling():
    z_ws.copy_(torch.bmm(a_ws, w_bmm))
else:
    torch.bmm(a_ws, w_bmm, out=z_ws)
```

`is_compiling()` folds to a constant at trace time, so the compiled path avoids the non-contiguous `out=`
and the eager path keeps the zero-copy form. Artifact `patches/files/sm12x_b12x_kernels.py`, md5
`4193159223f321fd15766f746fc10f65`.

### The remaining break list is now almost entirely outside our scope

That is the honest summary of the compile port as of round 59. After v3 the forward-path residue is:

| site | owner | scope |
|---|---|---|
| `b12x/attention/_shared/mla/merge.py:84,85,92` | b12x | **third-party** -- needs the owner's agreement |
| `deep_gemm.py:1153` `tf32_hc_prenorm_gemm` | vLLM | **upstream, already open as PR #53055** |
| `multi_stream_utils.py:56` `torch.fx.traceback.annotate` | vLLM | upstream |
| `attention.py:852`, `import_utils.py:427` | vLLM | upstream |
| `pynvml.py:2381`, `tvm_ffi`, `tilelang`, `nvidia_cutlass_dsl`, `inspect.py` | libraries / warmup | mostly off the decode path |

**Every remaining forward-path break is in code this repo does not own.** The two we could fix -- the
`print` class and the non-contiguous `out=` -- are fixed. The rest need either an upstream merge (`#53055`
is literally the fix for one of them) or the b12x maintainer's agreement, and the contract is explicit
that upstream contact needs the owner asked first and a measurement saying it matters.

**That is the attribution the goal's fallback asks for, and it is now complete for this workstream:** the
remaining gap is in vLLM's `CompilationMode.NONE` policy for `DeepseekV4ForCausalLM` plus the
Dynamo-uncleanliness of the DSv4 b12x stack in this base -- **layer families:** the whole DSv4 decode path
(attention o_proj, MoE, mHC, indexer); **libraries:** vLLM's `compilation/` and `model_executor/warmup/`
frameworks, DeepGEMM, b12x, TileLang; **measurements:** `refbase-run.log` (`VLLM_COMPILE`) against
`proto2-dg-engine.log` (`NONE`), and `harness/step-time-gap.py` (+13.1/+17.1/+13.9/+20.9 ms per step at
c1/c3/c5/c6, with capture worth -56.5 ms at c1 and nothing at c6).

## `--disable-custom-all-reduce`: negative, and the reference's config is now exhausted (round 60)

The reference runs with `disable_custom_all_reduce: True`; we run `False`, so we use vLLM's custom
all-reduce kernel and it falls back to NCCL. That was the last untested difference in the two engines'
startup dumps, and it is a configuration choice rather than a kernel, so it was worth one arm.

`proto2-dg-nocar` is `proto2-dg` with one change, `--disable-custom-all-reduce`:

| level | proto2-dg | nocar | delta |
|---|---|---|---|
| c1 | 55.5 | 54.9 | -1.1 % |
| c3 | 99.6 | 100.8 | +1.2 % |
| c5 | 131.7 | 129.1 | -2.0 % |
| c6 | 150.4 | **144.3** | -4.1 % |
| sum | **437.2** | 429.1 | -1.9 % |

**Negative.** c1/c3 are a wash and c5/c6 are worse, including the level the contract cares most about.
Our custom all-reduce kernel is at least as good as NCCL at the levels that matter, so the reference's
choice is not where its lead comes from. **This closes the config-difference list:** compilation mode,
`moe_backend`, `linear_backend`, `ir_op_priority.rms_norm`, `cudagraph_capture_sizes`,
`disable_custom_all_reduce`, spec tokens and `max_model_len` are now all either matched or measured.

**Harness change made to run the arm, and it is the reusable part.** `harness/run-arm.sh` forced
`SERVE_EXTRA_ARGS='--async-scheduling'` itself, and its `${EXTRA}` argument is word-split into environment
assignments, so a second `vllm serve` flag -- whose value necessarily contains a space -- could not be
passed through it at all. It now honours an optional `ARM_EXTRA_ARGS`:

```bash
ARM_EXTRA_ARGS="${ARM_EXTRA_ARGS:-}"
SERVE_ARGS="--async-scheduling${ARM_EXTRA_ARGS:+ $ARM_EXTRA_ARGS}"
```

The default is empty, so `SERVE_ARGS` is `--async-scheduling` exactly as before and **every previously
measured arm is unchanged**. Verified by expanding both branches before use. This is what makes
flag-level arms possible without editing the runner per-arm, and it should be used for future ones rather
than adding another one-off script.

## The gap, attributed (consolidated, 2026-09-18)

The contract allows the goal to end with "the best measured arm and a written attribution of each
remaining gap: which layer family, which library, which measurement". This is that attribution, assembled
from the measurements of rounds 26-60, and it is the honest state of play.

**Best measured arm: `proto2-dg`** -- `proto2` plus `VLLM_B12X_INDEXER_DIRECT_GATHER=1`, promoted into
`configs/pin.main-029.env` and `scripts/05-serve.sh`. Three runs: 424.3, 422.6, 437.2 on the c1+c3+c5+c6
sum; honest range c6 139.6-150.4, sum 422.6-437.2. Same-day reference `refg`: 66.4 / 118.4 / 139.0 / 161.0,
sum 484.8. **Gap: 9.8-12.8 % on the sum, 6.6-13.3 % at c6**, down from 27.3 % / 31.0 % before the
indexer-gather win. Gates, acceptance (50.8-54.9 % against 50.2-54.6 %) and tokens per step
(4.531-5.020 against 4.499-4.923) all hold, so **no part of the remaining gap is a draft-quality or
acceptance problem.**

**Shape of the gap** (`harness/step-time-gap.py`, medians of three passes, same day):

| level | ours step ms | ref step ms | delta |
|---|---|---|---|
| c1 | 86.5 | 73.4 | +13.1 |
| c3 | 141.7 | 124.6 | +17.1 |
| c5 | 176.2 | 162.3 | +13.9 |
| c6 | 192.0 | 171.1 | +20.9 |

**Both halves resolve to the same cause: the reference compiles and we do not.**
`spark1:~/refbase-run.log` reports `CompilationMode.VLLM_COMPILE`; `spark1:~/goal/proto2-dg-engine.log`
reports `CompilationMode.NONE`, with identical `cudagraph_capture_sizes` `[1,2,4,8,16,24,32,40,48]` and
identical spec tokens. Our base selects `NONE` deliberately for this model
(`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` contains `DeepseekV4ForCausalLM`, `config/vllm.py:77`), a
per-architecture policy the reference's older July build does not have.

- **c1/c3:** capture is worth **-56.5 ms/step** at c1 for us (`proto2-dg-cgnone`), so this half is
  launch-and-latency dominated, and the reference's compiled graph is simply more complete. Both
  directions around capture are measured negatives -- less capture (`MAX_CUDAGRAPH_CAPTURE_SIZE=24`, sum
  426.5; `CUDAGRAPH_MODE=NONE`, sum 405.1) and more (`CUDAGRAPH_MODE=FULL`, sum 325.0) -- so this cannot
  be bought by cudagraph configuration. It needs fusion.
- **c5/c6:** capture is already spent (inside spread), so the ~14-21 ms there is kernel execution time --
  Inductor fusion, or the kernel differences below.

**Which layer family:** the whole DSv4 decode path. Every family reachable by configuration is closed as a
negative: linear (`deep_gemm` worse and unstable on the SM12x dispatch `#417` is about; `humming` cannot
load; `cutlass`/`marlin` rig-rejected; `b12x` best), attention (FlashInfer cold 328.1, warm 320.9 against
334.3), MoE (A16 wash, tile-override wash; `b12x` is the anemll fork's name for the same kernels).

**Which library:** vLLM's `compilation/` and `model_executor/warmup/` frameworks for the compiled regime;
DeepGEMM, b12x and TileLang for the third-party kernels on that path.

**Which measurement:** the two engine logs above; `harness/step-time-gap.py` for the step-time split;
`proto2-dg-cgnone` / `-cg24` / `-cgfull` for the capture bounds; `outputs/driver/one-off/proto2-enum-breaks.log`
for the complete break list.

**Why it is not closed:** the compile port needs `fullgraph=True` with **zero** graph breaks
(`compilation/wrapper.py:150`, and `fullgraph=False` dies on `VllmBackend can only be called once`). One
boot now enumerates every break (`proto2-enum`). Twelve were cleared. **After round 59 every remaining
forward-path break is in code this repo does not own:**

| site | owner |
|---|---|
| `b12x/attention/_shared/mla/merge.py:84,85,92` | b12x -- third-party, needs the owner's agreement |
| `deep_gemm.py:1153` `tf32_hc_prenorm_gemm` | vLLM -- **already open upstream as PR #53055** |
| `multi_stream_utils.py:56` `torch.fx.traceback.annotate` | vLLM |
| `attention.py:852`, `import_utils.py:427` | vLLM |

That is the contract's own stop condition -- *"the remaining gap sits in a library outside the scope
above"* -- reached with a line number and a PR reference rather than by giving up. The goal stays active
because `#53055` could merge and clear one of them, and because a re-measure is cheap.

## Three structural questions closed with source evidence (round 61)

Round 60 concluded the remaining work is outside this repo. Before accepting that, three things that could
have reopened it were checked and all three are closed.

**1. No `CompilationMode` avoids `fullgraph=True`.** This was the last structural hope: if some mode
compiled without `fullgraph`, graph breaks would become harmless fallbacks and we would get partial
Inductor fusion for free. `compilation/wrapper.py:146-151` is reached from the shared compile path all
four modes use, and it is unconditional:

```python
self._compiled_callable = torch.compile(
    compiled_ptr,
    fullgraph=True,
    dynamic=False,
    backend=backend,
    options=options,
)
```

The only mode-specific branch afterwards (`wrapper.py:156`) decides the bytecode hook, not `fullgraph`.
**There is no configuration of vLLM 0.2.1 that permits a graph break.** That is now settled from the
source rather than inferred from two failed arms (`proto2-cgfg0` on AOT, `proto2-cgfg0aot0` on
`VllmBackend can only be called once`).

**2. The MoE is genuinely closed, and the two engines do not even use the same library.** The top kernel by
time is `b12x::tp_moe_dynamic_launch` at 42.55 % of kernel time, which is exactly the shape of an item-4
lead, so it was checked. It lives in
`/usr/local/lib/python3.12/dist-packages/b12x/moe/fused_moe/_impl.py` -- third-party, not our overlay.
But **that file does not exist in the reference image at all.** The two engines run different MoE
implementations:

| | MoE implementation |
|---|---|
| ours | `b12x` package, `fused_moe/_impl.py`, 12215 lines |
| reference | FlashInfer **0.6.15** `b12x_fused_moe` (`--moe-backend flashinfer_b12x`) |

and we already priced it: the two implementations were run side by side in one process at the DSV4 c6
decode shape, ours at **10.066 ms** against FlashInfer 0.7.0's **127.729 ms**, with bit-identical outputs;
separately, our FlashInfer 0.7.0 builds that kernel **8.5x to 12.1x slower** than the reference's 0.6.15.
So the reference's 0.6.15 is roughly an order of magnitude better than *our* FlashInfer, and still lands at
or above our 10.066 ms. **The MoE is not the gap, and the wheel-regression finding is the reason we are on
the `b12x` package at all.**

**3. Our own overlays have no untested opt-in flags.** Every environment flag in
`patches/files/sm12x_b12x_kernels.py` was enumerated with its default:
`VLLM_B12X_INDEXER_DIRECT_GATHER` (default `0`, pin sets `1` -- the round-45 win),
`VLLM_USE_B12X_SPARSE_INDEXER` (default `1`), `VLLM_USE_B12X_WO_PROJECTION` (default `1`), and
`VLLM_PROFILE_DECODE` / `VLLM_PROFILE_CAPTURE` / `VLLM_SKIP_FLAG_DIR`, which are diagnostics. Everything
non-diagnostic is already enabled. **There is no dormant switch left in the overlay.**

## Session check, 2026-09-18 (round 61)

Re-checked with `gh`. **Nothing has moved since round 57.** No tracked item has merged and there is no new
comment or review on any of our four PRs:

| item | state | last update | our last comment |
|---|---|---|---|
| #53425 | OPEN | 2026-09-14 | 2026-09-11 |
| #53522 | OPEN | 2026-09-14 | 2026-09-08 |
| #53271 | OPEN | 2026-09-14 | 2026-08-28 |
| #46716 | OPEN | 2026-09-14 | 2026-08-28 |
| #53055 | OPEN | 2026-09-16 | -- |
| #47988 | OPEN | 2026-09-12 | -- |
| #52499 | OPEN | 2026-09-06 | -- |
| #50645 | OPEN | 2026-08-23 | -- |
| #41834 | OPEN | 2026-09-09 | -- |
| DeepGEMM #419 / #403 | OPEN | 2026-08-28 / 2026-08-29 | -- |
| DeepGEMM #417 (issue) | OPEN | 2026-09-16 | 2026-09-16 |

**Action taken: none, and none is actionable.** All four of our PRs remain behind the `pre-run-check`
label gate recorded in round 54 -- the author needs a `verified`/`ready` label or 4 merged PRs, and the
gate's own text says *"DO NOT request for the label to be added if you are an AI agent."* There is nothing
for us to rebase or answer. `#53055` remains the one watch item, and it is the upstream fix for
`deep_gemm.py:1153`, one of the four breaks still blocking the compile port.

## Tight same-session pair: the attribution now rests on a back-to-back measurement (round 61)

Every previous comparison paired our arm with a reference run hours earlier -- the round-57 pair used our
2026-09-17T20:21 measurement against the reference's 12:55 one. Same day, same rig, but 7.5 h apart, and
"the rig drifted between them" is exactly the kind of objection that cannot be answered after the fact.
This round re-measured **both arms back to back with no idle gap**: ours finished 02:18, the reference
02:27.

**Aggregate, three passes each, medians:**

| level | ours `proto2-dg-pair` | spread | reference `refg` | spread | gap |
|---|---|---|---|---|---|
| c1 | 53.9 | 7.6 % | 65.3 | 4.0 % | -17.5 % |
| c3 | 98.6 | 3.3 % | 110.1 | 4.6 % | -10.4 % |
| c5 | 129.9 | 0.2 % | 138.6 | 16.1 % | -6.3 % |
| c6 | 144.0 | 6.7 % | 159.1 | 2.1 % | -9.5 % |
| **sum** | **426.4** | | **473.1** | | **-9.9 %** |

**Gates pass on every pass in both legs** (`gate_france` -> `' Paris. The capital of Spain'` /
`' Paris. The capital of Italy'`, `gate_9x8` -> `'72, 9x9'`). Acceptance is 51.5-52.7 % against
50.4-55.6 % and tokens per step 4.599-4.683 against 4.515-4.876 -- both overlapping, so the contract's
"acceptance rate and tokens per step must not be lower" holds.

**Step time** (`harness/step-time-gap.py proto2-dg-pair refg`):

| level | ours | reference | delta |
|---|---|---|---|
| c1 | 86.8 ms | 75.2 ms | **+11.5** |
| c3 | 142.9 ms | 123.7 ms | **+19.2** |
| c5 | 180.1 ms | 162.5 ms | **+17.6** |
| c6 | 193.5 ms | 175.4 ms | **+18.1** |

median **+17.8 ms**, range +11.5 .. +19.2.

**What this confirms, and what it corrects.**

- **The sum gap reproduces: -9.9 % against -9.8 % from the loose pair.** The headline number is not an
  artefact of time-of-day drift.
- **Both arms came in lower than their earlier runs** -- ours 426.4 against 437.2, the reference 473.1
  against 484.8 -- so 437.2 was the high end of our range and 484.8 of theirs. The honest ranges are ours
  **422.6-437.2** and the reference's **473.1-484.8**, and the gap is what is stable, not either absolute.
- **The flat per-step delta reproduces and is slightly larger: +17.8 ms median against +15.5 ms**, over a
  step time that more than doubles (86.8 -> 193.5 ms). A per-row cost would scale with concurrency. This
  does not, at any level, in two independent pairings. **That is the finding the whole attribution rests
  on, and it now has a back-to-back measurement behind it.**
- **The c6 gap is larger in the tight pair (-9.5 %) than in the loose one (-6.6 %)**, and the c6 spread was
  wider in the loose pairing (9.3 % against 2.1 %). Both readings are consistent with the same underlying
  ~18 ms/step: at c6 that is 9.5 % of a 193.5 ms step, so the loose pair's 6.6 % was the low sample.

**Artifacts:** `outputs/driver/proto2-dg-pair.median.log`, `outputs/driver/proto2-dg-pair-{1,2,3}.meter.txt`,
`outputs/driver/refg.median.log`, `outputs/driver/refg-{1,2,3}.meter.txt`. Driver:
`.scratch/arm-pair.sh` (ours, then `harness/run-refg.sh`, back to back).

## CORRECTION: "outside our scope" was wrong, and the compile port is back on (round 62)

Rounds 59-61 concluded that the compile port's remaining breaks sit "in code this repo does not own" and
treated that as the contract's stop condition. **That reasoning was wrong and is retracted here.**

The scope boundary in the contract is about **what this repo may modify and where it may run** -- this
repo's own directories, and spark1/spark2. It is not a statement about which *files inside the image* may
be overlaid. This repo has bind-mounted patched copies of image files from the start:
`utils/sm12x_b12x_kernels.py`, `models/deepseek_v4/nvidia/b12x_sparse.py`,
`model_executor/warmup/dsv4_warmup_ext.py`, `compilation/wrapper.py`, `utils/deep_gemm.py`,
`v1/worker/gpu_worker.py`, `models/deepseek_v4/nvidia/ops/o_proj.py`, `model_executor/warmup/`
`jit_warmup_triton_helper.py`, and `model_executor/kernels/mhc/tilelang.py` -- the last of which is an
*upstream vLLM file we have been overlaying since break 5 without ever calling it a scope violation.*

The only thing the contract actually forbids is **contacting upstream** -- "check with the owner before
opening an upstream PR or commenting on someone else's" -- which is about PRs and comments on other
people's threads, not about a local overlay. **So all four "residue" breaks are patchable locally**, and
the port was abandoned three rounds early on a misreading.

### What that unlocks, immediately

**The repo already contains the fix for one of the four, parked.** `patches/apply_overlays.py:5000
patch_mhc_tf32_customop` registers `tf32_hc_prenorm_gemm` as a custom op, hoists the three local import
blocks that Dynamo cannot trace (importlib is skipped), and routes the three call sites. It was parked
because it edits the same file as break 5's overlay and only one of the two was being applied. Its
docstring records it as **verified against `vllm-spark-0731:main-029-proto2`** -- the image we are running.

The two patches touch different regions of `model_executor/kernels/mhc/tilelang.py`, so they compose.
Applied in sequence to `~/fix10/tilelang.orig.py`:

```
ok mHC tf32 custom op: 1 registration, 3 call sites, 3 import blocks hoisted
wrote tilelang.merged.py: mhc_pre_broadcast_tilelang registered as a custom op
merged parse OK
mhc_pre_broadcast registration: True
tf32 registration: True
routed tf32 calls: 3
routed mhc_pre calls: 1
```

Staged as `spark1:~/fix10/tilelang.op2.py` and `spark2:~/fix10/tilelang.op2.py`, md5
`8e9dcf1989b69a572e247d56a2424ee5`. **This should clear the `deep_gemm.py:1153` break** -- the one this
repo has been calling "upstream, PR #53055" for three rounds while holding its fix in a parked function.

### The remaining piece is `fp8_einsum`, and it is bounded

`fp8_einsum` has exactly **two** call sites in the tree:

| site | on our path? |
|---|---|
| `models/deepseek_v4/nvidia/ops/o_proj.py:112` | **yes** -- the DSv4 o_proj |
| `model_executor/kernels/linear/mxfp8/deep_gemm.py:116` | no (`LINEAR_BACKEND=b12x`, so the mxfp8 linear path is unused) |

So one overlay on `o_proj.py` -- which we already mount -- covers it. The custom op needs the tensor pairs
flattened (a `torch.library` schema cannot carry a nested tuple), `mutates_args=["out"]` because the call
writes through it, and a fake impl returning `torch.empty_like(out)`. The signature is printed verbatim in
the failure from round 55:

```
fp8_einsum('bhr,hdr->bhd',
           (FakeTensor(12288, 4, 4096, fp8_e4m3fn), FakeTensor(12288, 4, 8, int32)),
           (FakeTensor(4, 1024, 4096, fp8_e4m3fn), FakeTensor(4, 1024, 8, int32)),
           FakeTensor(12288, 4, 1024, bfloat16),
           recipe=(1, 1, 128))
```

**Next: write that op, then run a real compile arm (not an enumeration) and see how far it gets.**

### The real compile arm reaches break 13, and two more breaks are confirmed cleared

Arm `proto2-cg3` is the first arm in this whole workstream that is **not** an enumeration run: no
`fullgraph=False`, no eager backend, no `fp8_einsum` stub. Nine mounts, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`,
AOT on. It got further than any previous config.

**Confirmed by what no longer appears in its log:**

- **`Attempted to call function marked as skipped` for `fp8_einsum` is gone.** The opaque custom op
  works: `o_proj.py` now calls `torch.ops.vllm.b12x_fp8_einsum` with the tensor pairs flattened,
  `mutates_args=["out"]`, and `torch.empty_like(out)` as the fake impl. This is the break the repo spent
  rounds 55-57 on and could not clear, and the answer was exactly what torch's own error message said the
  first time: *"wrap the custom kernel into an opaque custom op."*
- **`deep_gemm.py:1153` is gone**, so the merged `tilelang.op2.py` cleared the `tf32_hc_prenorm_gemm`
  break as predicted.
- **The pre-pack still works**: `DSv4 wo_a DeepGEMM einsum pre-pack: 43 packed, 0 skipped, 1.66 s`.
- **The `sm12x_b12x_kernels.py:1144` non-contiguous `out=` break is gone** (the enumeration before this
  run no longer reported it).

**So the arm advanced four breaks** and now stops at:

```
model.py:1343  residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(...)
  -> tilelang.py:1042  _MHC_POST_TILELANG_KERNEL(
  -> jit_warmup_tilelang_helper.py:169
  -> tilelang/jit/__init__.py:527 __call__ -> _infer_jit_mode
  -> tilelang/jit/__init__.py:392  is_lazy_style = self.func._is_lazy_style(*args, **kwargs)
Unsupported: Attempted to call function marked as skipped
```

**That is break 13, and it is the same class as break 5 with a different entry point.**
`.scratch/patch_mhc_op1.py` registers `mhc_pre_broadcast_tilelang` and its own docstring calls it *"first
of five"*. The four remaining entry points are `mhc_pre_tilelang`, `mhc_fused_post_pre_tilelang` (this
one), `mhc_post_tilelang` and `hc_head_fused_kernel_tilelang`. The pattern is proven, the fake impls are
shape-derivable from the inputs the same way (`residual` `(T, hc_mult, H)` bf16, `post_mix`
`(T, hc_mult, 1)` fp32, `comb_mix` `(T, hc_mult, hc_mult)` fp32, `layer_input` `(T, hc_mult... H)` bf16 --
and `dsv4_warmup_ext.py` already documents this call's return order as `residual, post_mix, comb_mix,
layer_input`), and `model.py` imports the names so no call site needs editing.

**Next: generalise the mHC patcher to all five entry points at once**, then re-run `proto2-cg3`.

Artifacts staged on both nodes: `~/fix10/tilelang.op2.py` (md5 `8e9dcf1989b69a572e247d56a2424ee5`),
`~/fix10/o_proj.op2.py` (md5 `5d7a19c399bd99d4776f364498c83309`), and the arm driver
`.scratch/arm-cg3.sh` (md5 `d25c1074b3ee427eb3c7bf9481a751bf`), which refuses to launch if any of its nine
mount sources is missing -- the Docker-creates-a-directory hazard from round 54.

### Round 63: priming `self.mode` works and moves break 13 one level deeper

Break 13 was `tilelang/jit/__init__.py:392 self.func._is_lazy_style(*args, **kwargs)`, a source scan Dynamo
cannot trace. Reading the enclosing function showed a fix that is better than four more custom ops:

```python
def _infer_jit_mode(self, *args, **kwargs):
    if self.mode in ("lazy", "eager"):
        return self.mode                      # <-- plain attribute read, Dynamo is fine
    if not isinstance(self.func, JITFunc):
        return "lazy"
    is_lazy_style = self.func._is_lazy_style(*args, **kwargs)   # <-- the break
    return "lazy" if is_lazy_style else "eager"
```

and the caller memoises it:

```python
def initialize_jit_mode(self, *args, **kwargs):
    if self.mode == "auto":
        self.mode = self._infer_jit_mode(*args, **kwargs)
```

So the cold path runs once and then `self.mode` short-circuits **on a plain attribute** -- not a
descriptor, no `try`/`except`, nothing for Dynamo to trace into. **This is the opposite of the round-55
`functools.cached_property` case**, where warming could never help because Dynamo traces the `except`
branch either way. Here priming genuinely removes the break.

**And the repo already had the priming, wired up in the wrong place.**
`deepseek_v4_mhc_layer_warmup` calls `mhc_pre_tilelang` and `mhc_fused_post_pre_tilelang` for every token
size -- but it is called from `kernel_warmup`, which runs from `compile_or_warm_up_model`, which is
**after** `determine_available_memory` has already compiled inside `profile_run`. It fires too late to
help, exactly like the wo_a pre-pack. Moving the call into the same pre-profile hook fixed the ordering:

```
DSv4 wo_a DeepGEMM einsum pre-pack: 43 packed, 0 skipped, 1.74 s.
DSv4 mHC layer warmup finished in 0.37 seconds.
```

**The break moved, which is the evidence that it worked.** `Attempted to call function marked as skipped`
became:

```
torch._dynamo.exc.Unsupported: call to a callable object with no traceable __call__
  model.py:1343  mhc_fused_post_pre_tilelang(...)
    -> tilelang.py:1042  _MHC_POST_TILELANG_KERNEL(
    -> jit_warmup_tilelang_helper.py:169  self.launch(self.kernel(*kernel_args), ...)
    -> jit_warmup_tilelang_helper.py:142  return jit_impl(*args, **call_kwargs)
```

**So the TileLang *mode inference* is cleared and the TileLang *compiled callable* is the next wall.**
`jit_impl` is the TileLang JIT's own compiled object, and Dynamo cannot call it. That is the same class as
break 9's ctypes `_FuncPtr`, and the answer is the same: an opaque custom op per entry point.

**That means the four remaining mHC entry points do need custom ops after all** -- but now for a different
and better-understood reason, and with the mode-priming out of the way. The shapes are readable straight
out of the source (`tilelang.py`), which is how break 5's fake impl was written:

| entry point | returns |
|---|---|
| `mhc_pre_tilelang` | `post_mix` `(T, hc_mult)` fp32, `comb_mix` `(T, hc_mult*hc_mult)` fp32, `layer_input` `(T, H)` bf16 |
| `mhc_fused_post_pre_tilelang` | `residual_cur` `empty_like(residual)`, `post_mix(cur)` fp32, `comb_mix(cur)` fp32, `layer_input(cur)` bf16 |
| `mhc_post_tilelang` | `out = torch.empty_like(residual)` |
| `hc_head_fused_kernel_tilelang` | `(T, H)` bf16 |

One caveat to handle carefully: `mhc_pre_tilelang` and `mhc_fused_post_pre_tilelang` each contain **two**
allocation shapes -- an early "meta device / shape inference" path using flat `num_tokens, hc_mult` and the
real path using `*outer_shape` -- so the fake impl must mirror the *real* branch, not the first `torch.empty`
in the function. That is the one place a mistake would produce a silently wrong graph rather than a crash,
so it must be read rather than inferred.

Artifact: `~/fix10/gpu_worker.op3.py` (md5 `368bb8a76c813434c551ce3c6048e14e`), arm driver
`.scratch/arm-cg4.sh`.

## The rebind discovery: upstream registers the mHC custom ops and never calls them (round 64)

Round 63 ended with a plan to write four more hand-derived fake impls. Reading the file first made that
unnecessary, and turned up what looks like an upstream bug.

**`model_executor/kernels/mhc/tilelang.py` already ends with six `direct_register_custom_op` calls** --
`mhc_pre_delayed_tilelang`, `mhc_fused_post_pre_delayed_tilelang`, `mhc_pre_tilelang`,
`mhc_post_tilelang`, `mhc_fused_post_pre_tilelang`, `hc_head_fused_kernel_tilelang` -- **each with a
hand-written fake impl already in the file** (`_mhc_pre_tilelang_fake`, `_mhc_post_tilelang_fake`,
`_mhc_fused_post_pre_tilelang_fake`, `_hc_head_fused_kernel_tilelang_fake`). So the shapes I was about to
derive by hand were already derived, correctly, by upstream.

**But nothing ever calls those ops.** `direct_register_custom_op`
(`vllm/utils/torch_utils.py:1058`) does exactly three things:

```python
my_lib.define(op_name + schema_str, tags=tags)
my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
my_lib._register_fake(op_name, fake_impl)
```

It defines and implements the op; **it does not repoint the Python name.** `grep torch.ops.vllm` over the
whole file returns nothing. So `from ... import mhc_pre_tilelang` in `model.py` hands out the *raw Python
function*, which calls TileLang, and Dynamo dies on the compiled callable.

**The fix is therefore a rebind, not a fake impl**: four two-line module-level wrappers appended after the
registrations, each body a direct call to `torch.ops.vllm.<name>`. Break 5's overlay had already
discovered this for `mhc_pre_broadcast_tilelang` -- the one entry point upstream does *not* register --
and this generalises it. `model.py` imports after the module body completes, so it picks up the rebound
names and needs no patch of its own. Artifact `.scratch/patch_tilelang_rebind.py`, staging
`~/fix10/tilelang.op3.py` (md5 `85119a1d76f6e4d6d05b4c6bed8b7bc1`), which carries break 5 + the parked
tf32 op + the four rebinds.

**Result: the entire mHC family is cleared in one patch.** Arm `proto2-cg5` no longer fails anywhere in
`tilelang.py`, and `residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(...)` at `model.py:1343`
is no longer in any traceback. Five entry points resolved by four rebinds and one registration, instead of
five fake impls.

**Recorded as an upstream-bug candidate, with no contact made.** "Registers six custom ops with fake impls
and never calls any of them" reads like an incomplete upstream change -- plausibly the call-site rewrite
was meant to live in the same commit. It is a genuine finding and a plausible PR, but the contract
requires asking the owner first, so nothing has been filed or commented. If it is filed later, the
evidence is `vllm/utils/torch_utils.py:1058` plus `grep torch.ops.vllm` on `mhc/tilelang.py`.

### Break 15 is our own diagnostic, in the traced forward

The break moved out of `tilelang.py` and into the attention path:

```
model.py:1337   x = self.attn(positions, x, None)
attention.py:518  self._prepare_and_attn_fn(
attention.py:619  q, (indexer_inputs, _) = execute_in_parallel(
multi_stream_utils.py:123    aux_results[i] = fn()
attention.py:622    lambda: indexer(
attention.py:1131     if b12x_skip_flag("indexer_all"):
sm12x_b12x_kernels.py:1396    return os.path.exists(os.path.join(root, "skip_" + name))
Unsupported: Attempted to call function marked as skipped
```

**`b12x_skip_flag` is this repo's own removal instrument** -- a file-existence probe used to skip stages
during measurement -- and it is being called on every forward inside the region Dynamo traces.
`os.path.exists` is a builtin Dynamo skip-lists, so the instrument itself is now the blocker.

This is required work item 3 with a concrete answer: the fix is to resolve the flags **once** into a frozen
set and read that from the traced path, refreshing the snapshot where it can still be refreshed -- the
pre-profile hook already added for the wo_a pre-pack and the mHC warmup. The eager path keeps
`os.path.exists` exactly as it is today, so the skip-flag instruments that produced rounds 39-48 keep
behaving identically.

## Break 15 cleared, break 16 is the same class -- and the class is now the pattern (round 65)

Break 15 (`b12x_skip_flag` -> `os.path.exists`) is cleared. `sm12x_b12x_kernels.py` now keeps the eager
path exactly as it was -- still `os.path.exists`, so a flag can be toggled while the server runs, which is
what those instruments are for -- and reads a frozen `_RUNTIME_FILES` snapshot from the traced path via
`torch.compiler.is_compiling()`. The snapshot is retaken by `b12x_refresh_runtime_flags()` from the
pre-profile hook that already carries the wo_a pre-pack and the mHC warmup. `_indexer_direct_gather` got
the same treatment even though the pin's env var short-circuits it today, because it has the same defect.

**`proto2-cg6` got past it and stopped one level deeper, at a break that is the same kind of thing:**

```
attention.py:1166  wq_b_and_q_quant
attention.py:1179  (q_quant, weights), _ = maybe_execute_in_parallel(
attention.py:1166  return fused_indexer_q_rope_quant(
fused_indexer_q.py:674    if has_cutedsl():
import_utils.py:601        return _has_module("cutlass")
Unsupported: Attempted to call function marked as skipped
```

**So breaks 15 and 16 are one class: a static, process-lifetime capability probe evaluated inside the
traced forward.** `b12x_skip_flag` asks "does this marker file exist"; `has_cutedsl` asks "is this module
importable". Both answers are fixed for the life of the process, both are reached from `model.py:1543 ->
layer -> self.attn(...)`, and Dynamo cannot trace either because the probe bottoms out in `os` or
`importlib` -- both skip-listed.

**That makes this the third appearance of one fix shape, and worth doing once rather than per probe:**

| probe | how it is folded |
|---|---|
| `torch.cuda.is_current_stream_capturing()` (round 54) | one wrapper in `vllm/__init__.py`, 42 call sites |
| `b12x_skip_flag` / runtime markers (round 65) | one snapshot in the overlay, read via `is_compiling()` |
| `_has_module` / `has_cutedsl` and friends (next) | same, in `utils/import_utils.py` |

The reason the fold works here and did not work for `functools.cached_property` (round 55) is always the
same: **Dynamo must not have to trace the body that does the work.** `is_compiling()` folds to a constant,
so the cold branch is pruned; a descriptor's `except` branch is not.

**One safety requirement for the `import_utils` version.** A capability cache that returns `False` for a
name it never warmed would silently disable a fast path -- the worst possible failure, since it would
still serve and still pass gates while quietly changing which kernels run. So the compiled path must
**raise loudly**, naming the module, if the name is absent from the cache; and the pre-profile hook must
warm it by calling every public `has_*` probe, so the loud path should never fire. Anything that does fire
is a name to add, not a default to accept.

## Break 16 cleared; break 17 is a CuTeDSL kernel, and the port is now bounded (round 66)

**Break 16 cleared.** `has_cutedsl()` no longer touches `importlib` on the traced path. `_has_module` and
`_has_module_spec` lost their `@cache` decorators -- which were never enough, because Dynamo traces the
wrapped body -- and now keep an explicit `_MODULE_RESULTS` dict that `b12x_warm_module_cache()` fills from
the pre-profile hook by calling every public probe. The compiled path is a container test plus a **loud
raise** for an unwarmed name:

```python
if module_name not in _MODULE_RESULTS:
    raise RuntimeError(f"b12x: capability probe for {module_name!r} was not warmed ...")
```

Defaulting to `False` would have quietly reported "unavailable", changing which kernels run and which fast
paths are taken, while still serving and still passing both gates. An invisible, gate-proof failure is
strictly worse than a crash, so it crashes.

**Break 17 is not a probe, and it is the useful discovery of this round.** One line further on:

```
fused_indexer_q.py:680   _INDEXER_Q_FP8_KERNEL(
jit_warmup.py:44           return self.launch(launch_spec, {})
jit_warmup_cutedsl_helper.py:75   executor = self._get_or_compile(compile_key)
jit_warmup.py:679                 self.compile(compile_key)
jit_warmup_cutedsl_helper.py:64     -> Attempted to call function marked as skipped
```

`has_cutedsl()` now answers correctly, and the answer is *yes*, so the CuTeDSL indexer kernel is taken --
and `_get_or_compile` **compiles it during tracing**, because the warmup that would have populated the
cache (`kernel_warmup`) runs after `profile_run` has already compiled. The same ordering bug as the wo_a
pre-pack and the mHC mode priming. Even warmed, the launch itself is an opaque CuTeDSL callable, so this
entry point needs a custom op regardless.

### The mHC rebind trick does not generalise, and now we know why

The mHC family cost four two-line wrappers because upstream had **already written the fake impls and
already registered the ops**, and only failed to repoint the names. The obvious hope was that the other
DSv4 op modules do the same. They do not:

```
$ grep -rln direct_register_custom_op models/deepseek_v4 model_executor/kernels/mhc
models/deepseek_v4/xpu/model.py
models/deepseek_v4/cpu/cpu_sparse.py
models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py
model_executor/kernels/mhc/triton.py
model_executor/kernels/mhc/aiter.py
model_executor/kernels/mhc/tilelang.py
```

**None of `fused_indexer_q.py`, `cache_utils.py`, `fused_compress_quant_cache.py`,
`save_partial_states.py`, `fused_mtp_input_rmsnorm.py`, `sparse_attn_compress_cutedsl.py`,
`dequant_gather_k_cutedsl.py` or `fi_moe.py` registers anything.** So the remaining entry points need
genuine custom ops with hand-derived fake impls.

### The port is now bounded

Counting the kernel singletons instantiated on this path gives **19**, of which the 5 mHC ones are cleared:

| module | singletons |
|---|---|
| `model_executor/kernels/mhc/tilelang_kernels.py` | 5 -- **cleared** (rebinds) |
| `common/ops/cache_utils.py` | 4 |
| `common/ops/fused_indexer_q.py` | 2 |
| `common/ops/fused_mtp_input_rmsnorm.py` | 2 |
| `common/ops/fused_compress_quant_cache.py` | 1 |
| `common/ops/save_partial_states.py` | 1 |
| `nvidia/ops/sparse_attn_compress_cutedsl.py` | 3 |
| `nvidia/ops/dequant_gather_k_cutedsl.py` | 1 |
| `model_executor/kernels/mhc/warmup.py` | 1 |

So roughly **13 more entry points**, each needing a custom op and a fake impl, at the rate of about one or
two per round. That is a bounded, mechanical port rather than an open-ended one -- which is new
information, and it is the reason to keep going rather than stop.

**The honest caveat, stated plainly.** Fourteen rounds in, **no arm has produced a single tok/s number**,
and there is still no measurement showing that compiling makes *our* stack faster -- only that the
reference runs compiled and that our c1/c3 deficit sits exactly where fusion would act. **Decision
criterion, recorded now so it is not renegotiated later: if the remaining ~13 entry points do not yield a
booting compiled arm within roughly five more rounds of break-clearing, the port is recorded as the
negative and the consolidated attribution already written stands as the goal's end state.**

## The arm reaches Inductor (round 67)

**This is the first arm that got through Dynamo.** The failure is now inside generated code:

```
gpu_worker.py:630        self.model_runner.profile_run()
model_runner.py:828      _dummy_run(...)
                         out = model(new_inputs)
/root/.cache/vllm/torch_compile_cache/torch_aot_compile/88912dbd.../inductor_cache/7t/c7thsgf6....py, line 474
                         assert_size_stride(arg0_1, (s72, 32, 512), (16384, 512, 1), 'input')
AssertionError: wrong number of dimensions1 for op: input
```

That `.py` path is **Inductor's own generated kernel code**, reached from a `torch_aot_compile` cache entry. Every
`Attempted to call function marked as skipped` and every `no traceable __call__` is behind us; the graph
compiled and is now *executing*. Whatever comes next is a correctness problem in the compiled graph, not a
tracing blocker -- a different and much better class of problem, and the first time the port has produced
one.

### Three things got it there

**1. `custom_ops` is settable, and the knob is new.** `config/vllm.py:1631-1638` only defaults the base mode
`when the user has not set one`:

```python
if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
    if backend == "inductor" and mode != CompilationMode.NONE:
        self.compilation_config.custom_ops.append("none")
    else:
        self.compilation_config.custom_ops.append("all")
```

So inductor+compiled deliberately defaults to `"none"`, running the *eager* implementation of every
registered `CustomOp` -- which is why the sparse-indexer path is untraceable. `scripts/05-serve.sh` now
takes a `CUSTOM_OPS` value, and the arm's log confirms it took effect:
`custom_ops': ['all', '+quant_fp8', '+quant_fp8']`. The vendored `eugr-prod*.yaml` recipes already use
`custom_ops: ["all"]` with `FULL_AND_PIECEWISE`, so it is a known-good shape on this stack.

**But it did not clear break 17.** `fused_indexer_q_rope_quant` is called *directly* by `attention.py:1166`;
vLLM's `CustomOp` mechanism only applies where a call site dispatches through the `CustomOp` class
(`sparse_attn_indexer.py:971` does). So the knob is real and worth having, and it was not the unlock it
looked like. Recorded as a near-miss rather than a win.

**2. Break 17 cleared with a custom op** on `common/ops/fused_indexer_q.py` (`.scratch/patch_fused_indexer_q_op.py`,
staging `~/fix10/fused_indexer_q.op.py`, md5 `ca7b2ac2c28395c1eb99471d1975c875`). Two lessons from getting it
right:

- **`infer_schema` requires an explicit return annotation** on both `op_func` and `fake_impl`, or
  `direct_register_custom_op` raises `No return type annotation was provided` at import -- which surfaces as
  `Model architectures [...] failed to be inspected`, not as anything mentioning custom ops.
- **The op is registered for the fp8 branch only.** `use_fp4` returns a *different structure*
  (`((packed, scale), weights)`), and one `torch.library` schema cannot express both, so the wrapper keeps
  that branch as an ordinary call. `use_fp4` is a Python bool literal at the call site, so Dynamo folds the
  branch and never traces the body on this path.
- The module-level rebind must be appended **after** the original definition; inserting it before the `def`
  raised `NameError: name 'fused_indexer_q_rope_quant' is not defined` during model inspection.

**3. The break-15/16 probe folds and the mHC rebinds all held**, and the pre-pack and mHC warmup both report
success in the same log.

### The shape assert, and a hypothesis worth testing next

`assert_size_stride(arg0_1, (s72, 32, 512), (16384, 512, 1), 'input')` says a 3-D tensor arrived with a rank
or stride the compiled graph did not expect. **The leading hypothesis is that the upstream fake impls are
untested.** The mHC win came from rebinding names to ops upstream registers and never calls -- which means
upstream's `_mhc_pre_tilelang_fake`, `_mhc_fused_post_pre_tilelang_fake` and the rest have **never been
exercised against the real functions' returns**. `mhc_pre_tilelang` allocates its outputs as
`(*outer_shape, hc_mult)` in the real body but as `num_tokens, hc_mult` in the shape-inference path, and the
fake impl was written against one of those. Routing through the op puts that discrepancy into the graph.

That is checkable without a boot: compare each upstream `_*_fake` against the real function's return
statement, exactly as was done for break 5's and break 17's fake impls. **Next round: audit the five mHC
fake impls against their real returns, and fix the ones that disagree** -- an overlay on `tilelang.py` that
replaces a wrong fake impl is a two-line change, and it is the most likely cause of this assert.

### The mHC fake impls are correct; the bisect is the next step

The leading hypothesis was that the shape assert came from an upstream fake impl that had never been
exercised. **Checked, and it is wrong** -- at least for the mHC family. `_mhc_pre_tilelang_fake` returns

```python
torch.empty(*outer_shape, hc_mult, 1),        # real: post_mix.view(*outer_shape, hc_mult, 1)
torch.empty(*outer_shape, hc_mult, hc_mult),  # real: comb_mix.view(*outer_shape, hc_mult, hc_mult)
torch.empty(*outer_shape, hidden_size),       # real: layer_input.view(*outer_shape, hidden_size)
```

which matches the real return exactly (the `.view(*outer_shape, ...)` on a `num_tokens`-shaped allocation is
the identity when `outer_shape` is one-dimensional). `_mhc_post_tilelang_fake` (`empty_like(residual)`) and
`_hc_head_fused_kernel_tilelang_fake` (`(T, H)` bf16) likewise match. So upstream's fake impls are not the
problem, and the "untested fake impl" hypothesis is recorded as **not supported** rather than left standing.

The asserting shape is `(s72, 32, 512)` with strides `(16384, 512, 1)` -- 32 heads against head_dim 512,
which is the **attention `q`**, not an mHC mix tensor. Contiguous and correct for its own layout, so the
graph fed a *different* tensor than it expected at that boundary. The prime suspects are the two overlays
whose fake impls return `empty_like` of an input: `fused_q_kv_rmsnorm` (break 6, `empty_like(qr)` /
`empty_like(kv)`, and `empty_like` preserves strides, so a non-contiguous input yields a non-contiguous
fake) and `fp8_einsum` (break 11/12). Neither has been exercised against a real compiled graph before.

**Next round is a bisect, not a guess.** Eleven mounts are in play; the method is to drop them in groups
and re-run, because the assert is deterministic and each boot is ~10 minutes. The two `empty_like`-based
fake impls get checked first, and the fix if they are at fault is to allocate contiguous outputs
(`torch.empty(shape, ...)`) rather than `empty_like`, since a compiled graph is entitled to assume the
contiguous layout the kernel actually writes.

**Round 67 cost the port nothing and gained the most important thing it has produced: it is past Dynamo.**

## `custom_ops: ["all"]` is harmful, and it was mine (round 68)

Round 67 recorded `CUSTOM_OPS='["all"]'` as a "near-miss": settable, took effect, but did not clear the
break. **That was too kind to it. It was the cause of the shape assert.**

Single-variable removal, `proto2-cg13` = `proto2-cg12` minus that one export:

| arm | `custom_ops` | `assert_size_stride` occurrences | failure |
|---|---|---|---|
| `proto2-cg12` | `['all', '+quant_fp8', '+quant_fp8']` | **2** | `wrong number of dimensions1 for op: input` |
| `proto2-cg13` | `[]` | **0** | `Index not registered in index_to_user_object_weakref` |

So the stride assert was produced by routing vLLM's registered `CustomOp` classes through their torch ops.
Those ops' fake impls carry strides or metadata that do not match what the real implementations produce,
and the compiled graph bakes in the difference -- `torch/_inductor/utils.py:3720`
`align_inputs_from_check_idxs.run` then calls `copy_misaligned_inputs` and the generated
`assert_size_stride` fires. **The knob is reverted and must stay unset**; `05-serve.sh` keeps the
`CUSTOM_OPS` hook because it is a legitimate config surface, but every compile arm leaves it empty.

This is the second time this session that a change made in one round was refuted by the next round's
measurement, and the pattern is worth naming: **"it took effect" is not evidence that it helped.**

### A real defect found and fixed on the way

`fused_q_kv_rmsnorm`'s fake impl, written in round 54, returned `torch.empty_like(qr)`. The real function
allocates:

```python
qr_out = torch.empty(qr.shape, dtype=qr.dtype, device=qr.device)
kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
```

**`empty_like` is not equivalent**: its default `memory_format=torch.preserve_format` copies the input's
strides, and `qr`/`kv` come out of `_split_qkv_and_norm` splitting a fused qkv buffer, so they can be
non-contiguous. The fake would then describe a non-contiguous output while the real kernel writes a
contiguous one. Fixed to mirror the real allocation exactly
(`.scratch/patch_qk_op.py`, `~/fix10/qk.op2.py`, md5 `8b4099163c3f4adc6917ed1bdfc84b99`).

**It did not clear the assert** -- that was the `custom_ops` knob -- but it is a genuine defect and stays
fixed. The general rule it establishes is worth carrying: **a fake impl must mirror the real function's
allocation, not approximate it with `empty_like`, because the graph specializes on the fake's strides.**

### The next blocker

```
torch/_library/custom_ops.py:503  wrapped_fn
torch/_dynamo/variables/streams.py:131   record_event
torch/_dynamo/variables/streams.py:86    _get_event_by_index
torch/_dynamo/graph_bytecode_inputs.py:44  get_external_object_by_index
AssertionError: Index not registered in index_to_user_object_weakref
```

`record_event` is a CUDA-event custom op, and it is being replayed inside the compiled region without
Dynamo having registered the event's external-object index. The reachable suspect is this repo's old
friend `multi_stream_utils.py`, whose `execute_in_parallel` / `maybe_execute_in_parallel` record events
across side streams and which appeared as a break site in the round-58 enumeration. **Next round: find
whether those helpers have a sequential fallback and force it under compilation** -- if the events are
only there to overlap work, serialising them inside the graph costs nothing structurally and removes the
custom op entirely.

## CORRECTION to round 68, and the two compile-path faults separated (round 69)

Round 68 recorded, from a single-variable bisect, that `custom_ops: ["all"]` **caused** the stride assert.
**That claim was too strong and is downgraded here.** Taking cudagraph capture out of the picture changes
the failure in a way that shows the two were not independent:

| arm | `custom_ops` | `CUDAGRAPH_MODE` | failure |
|---|---|---|---|
| `proto2-cg12` | `["all"]` | `FULL_AND_PIECEWISE` | `assert_size_stride` (2 occurrences) |
| `proto2-cg13` | unset | `FULL_AND_PIECEWISE` | `Index not registered in index_to_user_object_weakref` (0 asserts) |
| `proto2-cg15` | unset | `NONE` | **`assert_size_stride` again** (capture off) |

So the stride assert is **not** a product of the `custom_ops` knob: with capture off it appears with
`custom_ops` unset. What the round-68 bisect actually measured is that the knob **changed which failure
fired first** -- with it, the stride assert preempted; without it, the stream error did. That is a real
difference and the knob still stays unset (it is not needed and it is not free), but the causal claim was
not supported by the evidence and is retracted.

**That is now three of my own conclusions in this session revised by the next measurement.** The pattern
is consistent and worth stating once more: an A/B that changes *which* error appears has not identified a
cause until the other error is also shown to be absent.

### Two independent faults in the compiled path

**Fault A -- stream external-object index, only with capture on.**

```
model_runner.py:1943 execute_model
model.py:1956 forward -> compilation/decorators.py:681
model.py:1506 forward -> compilation/caching.py:225
                     -> compilation/cuda_graph.py:256
                     -> compilation/piecewise_backend.py:380
torch/_dynamo/variables/streams.py:77  _get_stream_by_index
AssertionError: Index not registered in index_to_user_object_weakref
```

**This is not the multi-stream helper.** Round 68's next-step plan was to force a sequential fallback in
`multi_stream_utils.py`, and that overlay was written and applied (`multi_stream_utils.op.py`, md5
`f377cc773fc2d74830214d19cc0a849e`) -- and it did not fix anything, because the failing
`_get_stream_by_index` is reached from vLLM's own **piecewise cudagraph machinery**
(`compilation/cuda_graph.py` and `piecewise_backend.py`), not from `execute_in_parallel` or
`maybe_execute_in_parallel`. The guard is kept -- it is the correct behaviour for a compiled region and it
matches the intent of the existing `BreakableCUDAGraphCapture.is_active()` guard -- but it is recorded as
**aimed at the wrong place**, not as a fix.

**Fault B -- stride mismatch, independent of capture and of `custom_ops`.**

```
assert_size_stride(arg0_1, (s72, 32, 512), (16384, 512, 1), 'input')
```

still with the same shape, which is 32 heads against head_dim 512 -- attention `q`, or the `o` that feeds
`deep_gemm_fp8_o_proj` after its `view(tokens, n_groups * heads_per_group, head_dim)`, and `4 * 8 = 32`
matches `n_groups * heads_per_group` exactly for this model. Nothing has yet shown which.

**Next round: stop guessing at the tensor and read the generated code.** vLLM writes it to
`/root/.cache/vllm/torch_compile_cache/...` *inside the container*, which is deleted with the container, so
it has never been inspectable. Binding a host directory over that path makes the failing inductor file --
including the `arg0_1` guard and the lines above it -- available after the arm dies. That is a one-line
`SERVE_EXTRA_MOUNTS` addition, and it converts this from inference into reading.

## The compiled graph is now readable, and it names the tensor (round 70)

Four rounds of inferring what the stride assert meant, and the answer was one mount away: vLLM writes the
generated code to `/root/.cache/vllm/torch_compile_cache/` **inside the container**, and the container is
deleted when the arm dies. Binding a host directory over that path keeps it:

```
$M/torch_compile_cache:/root/.cache/vllm/torch_compile_cache
```

That is now in `.scratch/arm-cg16.sh`, and it left **20 generated `.py` files** behind, four of which
contain the failing guard. The same file region reads:

```python
def call(self, args):
    arg0_1, arg1_1, ... = args
    s72 = arg1_1
    assert_size_stride(arg0_1, (s72, 32, 512), (16384, 512, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        # Topologically Sorted Source Nodes: [zero_], Original ATen: [aten.zero]
        triton_poi_fused_zero_0.run(arg0_1, 16384*s72, stream=raw_stream0)
        ...
        # Source Nodes: [fused_inv_rope_fp8_quant_kernel],
        #   Original ATen: [vllm.fused_inv_rope_fp8_quant_kernel]
        buf1 = torch.ops.vllm.fused_inv_rope_fp8_quant_kernel.default(arg0_1, arg6_1, arg7_1, 8, 128, 4, 448, 32, True, 448.0, ...)
        buf2 = buf1[0]   # (4, s72, 4096) fp8_e4m3fn
        buf3 = buf1[1]   # (4, s72, 8) int32
        ...
        # Source Nodes: [transpose, transpose_1, b12x_fp8_einsum]
        buf5 = torch.ops.vllm.b12x_fp8_einsum.default('bhr,hdr->bhd',
                   reinterpret_tensor(buf2, (s72, 4, 4096), (4096, 4096*s72, 1), 0),
                   reinterpret_tensor(buf3, (s72, 4, 8), (1, (-32)*(s72 // (-4)), (-4)*(s72 // (-4))), 0),
                   arg8_1, arg9_1, buf4, [1, 1, 128])
        ...
        torch.ops._C.per_token_group_fp8_quant.default(reinterpret_tensor(buf4, (s72, 4096), (4096, 1), 0), buf7, buf8, 128, 1e-10, -448.0, 448.0, False, False, False)
```

**This is the o_proj partition, and `arg0_1` is `o`.** `(T, 32, 512)` is `(tokens, n_groups ×
heads_per_group, head_dim)` and `4 × 8 = 32` exactly for this model, with `head_dim = 512`; the expected
stride `(16384, 512, 1)` is `(n_heads × head_dim, head_dim, 1)`, plain contiguous. Our own
`torch.ops.vllm.b12x_fp8_einsum` is in this partition, consuming `fused_inv_rope_fp8_quant_kernel`'s
outputs through `reinterpret_tensor`, and the partition ends in `per_token_group_fp8_quant`.

**Two things follow immediately, and one of them is a correction to how the assert was being read.**

- `assert_size_stride(arg0_1, (s72, 32, 512), ...)` compares against the **symbolic** `s72`, which is
  `arg1_1` -- a *separate graph argument*. So a failure here need not be a stride problem at all; it fires
  equally if `arg0_1.shape[0]` and `arg1_1` disagree, i.e. if the token count the graph was specialized
  for does not match the tensor actually passed. Every previous note in this file has described it as a
  stride mismatch "which is attention `q` or `o`". The shape is now confirmed as **`o`**, and the
  stride-versus-symbol ambiguity is recorded as unresolved rather than assumed.
- Our own `_o_ws` workspace, from `sm12x_b12x_kernels.py`, is sliced as `_o_ws[:tokens]` out of a
  `(cap_t, n_heads, head_dim)` buffer. For `n_heads = 32, head_dim = 512` that slice has strides
  `(16384, 512, 1)` -- **exactly the strides the assert expects**. So either `arg0_1` is not that slice,
  or the mismatch is the symbolic dimension and not the strides. That is now a decidable question instead
  of a guess, and it is the next thing to read.

**Method, kept:** bind the compile cache on the host for any arm that compiles. It costs one mount and it
turns "which tensor" from inference into reading, which is what the last four rounds were missing.

## The contiguous-`o` fix did not clear it, which resolves the ambiguity -- toward the symbol (round 71)

Round 71's first half read the answer out of the generated code: the failing partition's `arg0_1` is `o`,
and `attention.py` produces it as `o_padded[:, : self.n_local_heads, :]` -- a slice. That looked like the
cause, so it was fixed: `.contiguous()` on that slice, gated on `is_compiling()` so the eager path stays
byte-identical (`.scratch/patch_attention_contiguous_o.py`, `~/fix10/attention.op.py`, md5
`8f4329075ba72c428344f2b91ab926d6`).

**`proto2-cg17` still fails with the identical assert.** That is informative rather than merely
disappointing. `.contiguous()` is a no-op when the slice is already contiguous, which means
`padded_heads == n_local_heads == 32` for this backend, so `o` **already had strides `(16384, 512, 1)`** --
exactly what the guard demands. A tensor that already satisfies the assertion cannot be failing it.

**So the mismatch is the symbolic `s72`, not the strides.** Round 70 recorded that ambiguity as unresolved
and declined to assume; this is the measurement that resolves it, and it goes the other way from every note
written between rounds 67 and 70, all of which called it a stride problem. In the generated code `s72 =
arg1_1` -- a *separate graph argument* -- and `assert_size_stride` compares `arg0_1` against it, so the
failure is that the token count the partition was handed does not agree with the tensor it was handed. That
is vLLM's symbolic-shape plumbing in the piecewise path, not our kernels and not our fake impls.

### The decision criterion I set in round 66 is met

Round 66 recorded: *"if the remaining ~13 entry points do not yield a booting compiled arm within roughly
five more rounds of break-clearing, the port is recorded as the negative and the consolidated attribution
already written stands as the goal's end state."* Rounds 67-71 are those five, and **there is still no
booting compiled arm and still not one tok/s number from any compile arm.**

**The port is recorded as a negative on that criterion -- with an important qualification that must not be
lost, because it is the difference between a dead end and deferred work.** This is no longer an
architectural wall. The arm now:
- clears every Dynamo graph break (13 of them, all with a diagnosis);
- builds a complete Inductor graph -- the generated code contains 315 `mhc_fused_post_pre_tilelang`,
  231 `all_reduce`, 131 `fused_inv_rope_fp8_quant_kernel`, 113 of our `b12x_fp8_einsum`, 98
  `fused_q_kv_rmsnorm`, 88 `moe_forward_shared`, 62 `fused_indexer_q_rope_quant`, 42
  `sparse_attn_indexer`, plus the four other mHC entry points;
- executes that graph, and fails on a guard inside vLLM's own compiled bookkeeping.

**The resume path is one variable, and it is written down so it is not rediscovered.** The engine runs
`dynamic_shapes_config: {'type': DynamicShapesType.BACKED}`, and a symbolic `s72` is precisely what is
failing. Compiling with **static shapes** would replace the symbol with a concrete integer and remove the
class of failure entirely. That is a compilation-config field, so it is a one-line test, and it is the first
thing to run if this workstream is picked up again.

**What is NOT claimed:** there is still no evidence that compiling makes this stack faster. The reference
compiles and we do not, and our c1/c3 deficit sits where fusion would act -- that is the whole of the case,
and it has never been more than an inference. **The consolidated attribution in this file stands as the
goal's end state**, and `proto2-dg` stands as the best measured arm at sum 426.4-437.2 against the
reference's 473.1-484.8.

### The resume path is now a named function, not a guess

The static-shapes idea needed a mechanism, and there is **no `DynamicShapesType.STATIC`** -- the enum has
only `BACKED` and `UNBACKED` (`config/compilation.py:341-353`). But `compilation/wrapper.py:150` already
compiles with `dynamic=False`, so the symbols do not come from the compile call at all. They come from
**explicit `torch._dynamo.mark_dynamic` marking**, which overrides it:

```
vllm/compilation/decorators.py:416   def _mark_dynamic_inputs(
vllm/compilation/decorators.py:448       torch._dynamo.mark_dynamic(arg, dims)
vllm/compilation/decorators.py:475       mark_dynamic(arg, dim_shape_pairs)
vllm/compilation/decorators.py:482       mark_dynamic(tensor, dim_shape_pairs)
vllm/compilation/decorators.py:587   _mark_dynamic_inputs(...)
```

So "compile with static shapes" is a **named, single-function change**: gate `_mark_dynamic_inputs` off (or
restrict it to the dims that matter) via an overlay on `vllm/compilation/decorators.py`. With concrete
integers instead of `s72`, `assert_size_stride` compares against a constant and the whole class of failure
disappears. It costs a recompile per batch size, which is irrelevant at the four fixed protocol levels.

This is recorded rather than attempted because the round-66 decision criterion is met and the port is
already at fifteen rounds. **A future session starts here, with one bookmark, instead of at the beginning.**

## Both shape modes fail, in different places, inside the same machinery (round 72)

Round 71 resolved the failure to the symbolic `s72` and named `_mark_dynamic_inputs` as the way to remove
it. That was a one-function overlay and the obvious thing to test rather than declare negative with it
untried. `.scratch/patch_static_shapes.py` gates the function behind `VLLM_B12X_STATIC_SHAPES=1`, forwarded
through `SERVE_EXTRA_ENV` because `05-serve.sh` passes only an explicit `-e` list.

**It worked, and then it failed somewhere else.** `proto2-cg18` reports **zero `assert_size_stride`
occurrences** -- the guard that blocked rounds 67-71 is gone, which independently confirms the round-71
diagnosis that the mismatch was the symbol and not the strides. The new failure is:

```
Worker failed with error ''t0''
KeyError: 't0'
  compilation/decorators.py:682   __call__
  compilation/wrapper.py:169      aot_compile
  compilation/backends.py:1222 -> 728 run -> 755 call_module
  compilation/piecewise_backend.py:190 __init__ -> 266 compile_all_ranges
  compilation/backends.py:353     compile
  compilation/compiler_interface.py:376 compile
```

`t0` is a generated variable name, and it is being looked up as a key while the piecewise backend compiles
its ranges. **vLLM's piecewise splitter indexes the graph by the symbolic names that `_mark_dynamic_inputs`
creates.** Removing the symbols removes the keys it looks up. So the two shape modes fail in two different
places, and both are inside vLLM's piecewise-compilation machinery:

| `VLLM_B12X_STATIC_SHAPES` | failure |
|---|---|
| unset (dynamic, default) | `assert_size_stride(arg0_1, (s72, 32, 512), ...)` -- the symbolic token dim |
| `1` (static) | `KeyError: 't0'` in `piecewise_backend.py:266 compile_all_ranges` |

### This closes the question, and it vindicates upstream

The port does not fail because of anything in this repo. Both escapes are exhausted, and each dies in
vLLM's own piecewise-compilation bookkeeping. **That is consistent with the one fact this whole
investigation started from and never explained:** upstream's `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`
(`config/vllm.py:77`) deliberately contains `DeepseekV4ForCausalLM`, forcing `CompilationMode.NONE`. Round
54 recorded that policy as the reason we were not compiling and treated it as an obstacle. **It is not an
obstacle; it is upstream telling us that this path does not work for this model**, and rounds 67-72 have
now demonstrated two independent reasons why: the DSv4 stack is custom-op-heavy and the piecewise path
needs symbols it cannot keep consistent.

**The reference does not contradict this.** Its DSv4 implementation is structurally different -- no
`jit_warmup_triton_helper.py`, a plain `fused_q_kv_rmsnorm`, no lazy `wo_a` repack, no b12x overlays (round
55) -- so it compiles because it has far less to keep Dynamo-clean. It is not a configuration we are missing;
it is a different codebase.

**Final state of this workstream.** Fifteen rounds, 13 Dynamo breaks cleared, a complete Inductor graph
built and executed, and two independent failures inside vLLM's piecewise machinery. **No compile arm ever
served, no compile arm produced a tok/s number, and no measurement was ever obtained showing that compiling
makes this stack faster.** The port is recorded as the negative, with the resume path documented
(`decorators.py:416`, and now `piecewise_backend.py:266` as the second wall behind it) in case upstream
changes the piecewise path.

**`proto2-dg` remains the best measured arm: sum 426.4-437.2 against the reference's 473.1-484.8, c6 144.0
against 159.1, step-time delta a flat median +17.8 ms.** The gap is the compiled regime; the compiled
regime is closed to us by upstream's own design and by two walls we measured rather than assumed.

## Session check, 2026-09-18 (round 73), and the workstream status

Re-checked with `gh`. **Nothing has moved since round 61.** No tracked item has merged, no PR has a new
comment or review, and `#53055` -- the upstream fix for one of the compile breaks -- is still open and
still unmerged.

| item | state | last update | our last comment |
|---|---|---|---|
| #53425 | OPEN, mergeable UNKNOWN | 2026-09-14 | 2026-09-11 |
| #53522 | OPEN, mergeable UNKNOWN | 2026-09-14 | 2026-09-08 |
| #53271 | OPEN, mergeable MERGEABLE | 2026-09-14 | 2026-08-28 |
| #46716 | OPEN, mergeable UNKNOWN | 2026-09-14 | 2026-08-28 |
| #53055 | OPEN | 2026-09-16 | -- |
| #47988 | OPEN | 2026-09-12 | -- |
| #52499 | OPEN | 2026-09-06 | -- |
| #50645 | OPEN | 2026-08-23 | -- |
| #41834 | OPEN | 2026-09-09 | -- |
| DeepGEMM #419 / #403 / #417 | OPEN | 2026-08-28 / 2026-08-29 / 2026-09-16 | 2026-09-16 |

**Action taken: none, and none is actionable.** All four of our PRs remain behind the `pre-run-check` label
gate recorded in round 54, whose own text says *"DO NOT request for the label to be added if you are an AI
agent."* Nothing to rebase, nothing to answer.

### Where this workstream ends

**Delivered, per the contract's fallback:** the best measured arm, and a written attribution of each
remaining gap -- which layer family, which library, which measurement. That is
`proto2-dg` at **sum 426.4-437.2** against the reference's **473.1-484.8** (c6 144.0 against 159.1), a flat
step-time delta of **+17.8 ms median**, with the cause traced to the compiled regime through two engine
logs and two independently measured walls inside vLLM's piecewise compilation.

**Not delivered, and not deliverable within the contract's boundaries:** beating the reference.

**Why it is not deliverable, in one place:**

1. **The gap is the compiled regime.** `refbase-run.log` reports `CompilationMode.VLLM_COMPILE`;
   `proto2-dg-engine.log` reports `NONE`, with identical capture sizes and spec tokens. Both of the gap's
   halves -- c1/c3 capture completeness and c5/c6 kernel time -- are what Inductor fusion would address.
2. **The compiled regime is closed by upstream's own design, and now shown to be broken rather than merely
   unavailable.** `config/vllm.py:77` puts `DeepseekV4ForCausalLM` in
   `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` and forces `NONE`. Rounds 55-72 cleared 13 Dynamo breaks and
   built and executed a complete Inductor graph, then failed in two different places depending only on
   shape handling: `assert_size_stride` on the symbolic token dim with dynamic shapes, `KeyError: 't0'` in
   `piecewise_backend.py:266 compile_all_ranges` with static ones.
3. **Every alternative is a measured negative.** The capture knobs in both directions
   (`CUDAGRAPH_MODE=NONE` 405.1, `MAX_CUDAGRAPH_CAPTURE_SIZE=24` 426.5, `CUDAGRAPH_MODE=FULL` 325.0), the
   last config difference (`--disable-custom-all-reduce`, 429.1), and every layer family reachable by
   configuration (linear, attention, MoE) all measured and recorded.
4. **The reference is not a configuration we are missing.** Its DSv4 stack is structurally different --
   no `jit_warmup_triton_helper.py`, a plain `fused_q_kv_rmsnorm`, no lazy `wo_a` repack, no b12x overlays,
   and a different MoE library entirely (FlashInfer 0.6.15 `b12x_fused_moe`; the `b12x` package's
   `fused_moe/_impl.py` does not exist in its image).
5. **What remains is upstream.** `#53055` for one compile break, the piecewise-compilation path for the
   other two walls, and the `pre-run-check` label gate for our own four PRs -- none of which this repo may
   change, and two of which the contract forbids touching.

## 2026-09-18: base moved to v0.30.0rc1

Latest upstream serving tag is `v0.30.0rc1` (`a00a3544b93edbd66c8eda7285e4468f1202dc4b`,
2026-09-17T10:37Z, lightweight tag). Compared with `proto-v0.2.0` (`f37c550bf635`) it is
**6 commits ahead and 38 behind**: the 38 are the vllm-proto crate line, not a serving
release. The six commits on the rc are #53187, #54016, #56545, #56904, #57077, #57285.
None of our tracked PRs merged (`#53425` `#53522` `#53271` `#46716` still OPEN;
`#53055` `#47988` `#52499` `#50645` `#41834` still OPEN; `#54048` already merged 2026-08-30).

`scripts/port_scan.py --with-upstream-patches` against a clean `v0.30.0rc1` tree reports
**FAIL=0  applied=43  no-op=12  total=55**. Upstream patches: applied `pr-47988` /
`pr-53425` / `pr-53522`, skip `pr-53055` (already in the base). Same three-apply /
one-skip pattern as proto-v0.2.0.

Pin: `configs/pin.main-029.env` now has `VLLM_REF=a00a3544b93e…` and
`IMAGE=vllm-spark-0731:main-030-rc1`. Goal contract (`docs/GOAL-BEAT-ANEMLL.md`) and
`HANDOVER.md` header moved with it. No protocol measurement on this base yet; every
number in `docs/EXPERIMENTS.md` is still proto-era and must be re-baselined after the
image lands.

Action on tracked PRs: none. All four of ours remain behind the `pre-run-check` label
gate. DeepGEMM #417/#419/#403 still OPEN.


## 2026-09-18: v0.30.0rc1 image landed, protocol re-baselined

Phase 1 `vllm-spark-0731:main-030-rc1-phase1` (`c6de62f0675e`) built on spark1 with
`VLLM_REF=a00a3544b93edbd66c8eda7285e4468f1202dc4b`. `assert_main_image` reports
`sha vllm=a00a3544b93e`. Phase 2 overlays (`3c1c99d9b5d4`) applied `pr-47988` /
`pr-53425` / `pr-53522`, skipped `pr-53055`, `assert_image.py --stack main` printed
`image OK (main)`. Copied to spark2, IDs match.

Same-day protocol, both arms, both gates pass in every pass:

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |

Behind 14.7 % at c6 and 14.8 % on the sum. Acceptance 49.0-54.8 % against 49.4-55.8 %,
tokens per step 4.414-4.815 against 4.439-4.876. Standing arm on this base is
`proto2-030`. Next one-variable measurement: `MOE_BACKEND=flashinfer_b12x`.


## 2026-09-18: `flashinfer_b12x` rejected on MXFP4 (negative)

`bash harness/run-arm.sh fb12x "MOE_BACKEND=flashinfer_b12x"` on `main-030-rc1` died at
worker init, 50 s, never healthy:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE.
Expected one of ['b12x', 'deep_gemm', 'flashinfer_trtllm', ...].
```

The 0731 experts are MXFP4. `flashinfer_b12x` is the NVFP4-experts kernel. The
reference's fork aliases the name; our pin does not. Goal item 2's MoE-family
NVFP4 path is therefore not reachable on this checkpoint without a different
weight file, which is out of scope. Standing arm: `proto2-030` (sum 412.1 vs
same-day `refg030` 483.7).


## 2026-09-18: MoE cluster cap 175 is a protocol negative

`B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=175` via `SERVE_EXTRA_ENV` on `main-030-rc1`.
Confirmed in the container. First boot died 90 MiB short of the 9.48 GiB KV
floor (`gpu_memory_utilization=0.8389`); retry served.

Same-day, same image as `proto2-030`:

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `moecap175` | 55.3 | 101.0 | 128.8 | 139.2 | 424.3 | 9.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Acceptance and tokens/step hold. +3.0 % on the sum and +0.6 % at
c6 sit inside `proto2-030`'s 12.8 % spread, so by this repo's rule it is not
a keep. The offline 5.9 % at cap 175 does not survive the protocol. Standing
arm remains `proto2-030`.


## 2026-09-18: stock compile on v0.30.0rc1 dies at the first Dynamo break

`VLLM_USE_BREAKABLE_CUDAGRAPH=0` on `main-030-rc1`, no overlay mounts. The opt-out
works: engine log reports `CompilationMode.VLLM_COMPILE: 3`. Capture then dies:

```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
Dynamo does not know how to trace ... tf32_hc_prenorm_gemm
skip reason: cannot determine source file for vllm.third_party.deep_gemm._C
```

Same class as proto-era break 5. `#56904` (GPU sync under compile, in this tag)
does not make pybind custom ops Dynamo-clean. The 13-break overlay port is still
required, and rounds 55-72 already showed that port dying inside vLLM piecewise
bookkeeping (`assert_size_stride` / `KeyError: t0`). Compile is not a serving
arm on this pin. Standing: `proto2-030` sum 412.1 vs same-day `refg030` 483.7.


## 2026-09-18: MoE work source `persistent_grid` is a protocol negative

`B12X_DYNAMIC_WORK_SOURCE=persistent_grid` via `SERVE_EXTRA_ENV` on `main-030-rc1`.
Confirmed in the container. Library default is `materialized_queue`.

Same-day, same image as `proto2-030`:

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `moework` | 54.8 | 97.3 | 129.3 | 145.4 | 426.8 | 6.6 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Acceptance and tokens/step hold. +3.6 % on the sum and +5.1 % at
c6 sit inside `proto2-030`'s 12.8 % spread, so not a keep. Standing arm
remains `proto2-030`.


## 2026-09-18: GPU util 0.8663 is a protocol negative

`GPU_MEMORY_UTILIZATION=0.8663` on `main-030-rc1`. One variable: restore the
pre-profiler KV budget. Engine log confirms util 0.8663 and Available KV
13.82 GiB (was 9.39 GiB at 0.8389).

Same-day, same image as `proto2-030`:

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `util8663` | 51.9 | 102.6 | 129.2 | 145.3 | 429.0 | 6.2 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Acceptance and tokens/step hold. +4.1 % on the sum and +5.1 % at
c6 sit inside `proto2-030`'s 12.8 % spread, so not a keep. Extra KV does not
close the reference gap. Standing arm remains `proto2-030`.


## 2026-09-18: MoE work source `ready_queue` dies in Cutlass DSL

`B12X_DYNAMIC_WORK_SOURCE=ready_queue` via `SERVE_EXTRA_ENV` on `main-030-rc1`.
Dies in `profile_run` before health:

```
cutlass.base_dsl.common.DSLUserCodeError: PHASE_DYNAMIC_TO_STATIC_BOOL
b12x/moe/_shared/kernels/dynamic.py:2001
while g < num_groups:  # _publish_ready_tasks
```

Experimental overlapped publisher is not JIT-clean on this b12x. The three
work-source values are now closed: default `materialized_queue` (standing),
`persistent_grid` (protocol negative, `moework`), `ready_queue` (does not
serve). Standing arm remains `proto2-030` (sum 412.1) vs same-day `refg030`
(483.7).


## 2026-09-18: MoE tile 32x128 is a protocol negative

`B12X_DYNAMIC_TILE_MN=32x128` via `SERVE_EXTRA_ENV` on `main-030-rc1`.
Confirmed in the container. Auto planner is (16, 128) for 48-288 routed rows;
64x128 was already a wash (`proto2-moetile`).

Same-day, same image as `proto2-030`:

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `moetile32` | 52.7 | 98.2 | 129.7 | 143.6 | 424.2 | 3.4 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Acceptance and tokens/step hold. +2.9 % on the sum and +3.8 % at
c6 sit inside `proto2-030`'s 12.8 % spread, so not a keep. Tile shape is not
the remaining MoE gap. Standing arm remains `proto2-030`.


## 2026-09-18: MoE `deep_gemm` dies on KV budget

`MOE_BACKEND=deep_gemm` on `main-030-rc1`. One variable. Oracle selected
`DEEPGEMM_MXFP4` / `DeepGemmFP4Experts` (SM120 path exists). Then:

```
Available KV cache memory: 8.79 GiB
ValueError: ... 9.48 GiB KV cache is needed ... available ... 8.79 GiB
```

DeepGEMM's workspace is ~0.6 GiB hungrier than b12x at the same util
(b12x had 9.39 GiB). Raising util would be a second variable. Linear
`deep_gemm` was already a protocol negative (`proto2-dglin`). Standing
arm remains `proto2-030` (sum 412.1) vs same-day `refg030` (483.7).


## 2026-09-18: W4A8 share-input is a protocol negative

`B12X_DYNAMIC_W4A8_SHARE_INPUT=1` via `SERVE_EXTRA_ENV` on `main-030-rc1`.
Confirmed in the container. Default is on only for decode/dense candidates
(c1); this forced it on for c3-c6.

Same-day, same image as `proto2-030`:

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `moeshare` | 55.4 | 98.0 | 126.9 | 140.5 | 420.8 | 4.4 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Acceptance and tokens/step hold. +2.1 % on the sum and +1.6 % at
c6 sit inside `proto2-030`'s 12.8 % spread, so not a keep. Standing arm
remains `proto2-030`.


## 2026-09-18: standing-pin re-baseline `p030b` lands in the same cluster

Same image, same pin, no extra env. New tag so the log does not append onto
`proto2-030`.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `moeshare` / `moetile32` / `util8663` | 55.4-51.9 | 98-103 | 127-129 | 140-145 | 421-429 | 3-6 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. +2.0 % vs the first sample, inside 12.8 %. Later config arms
were sitting in the pin's own noise, not winning. Standing remains
`proto2-030`. Noise center on this pin is ~420, not 412.


## 2026-09-18: CUDA-graph memory profiler off is a protocol negative

`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` via `SERVE_EXTRA_ENV` on
`main-030-rc1`. Confirmed in the container. Available KV 13.36 GiB
(profiler-on b12x had 9.39 GiB at the same 0.8389).

Same-day, same image:

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `memprof0` | 56.5 | 97.5 | 131.0 | 140.2 | 425.2 | 9.1 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. +3.2 % vs first sample and +1.2 % vs `p030b` sit inside 12.8 %.
One pass dipped below the standing accept/tokens-per-step floor. Extra KV
does not close the reference gap. Standing arm remains `proto2-030`.


## 2026-09-18: humming MoE is the best same-image sum, not a keep

`MOE_BACKEND=humming` on `main-030-rc1`. Engine selected `HUMMING` /
indexed gemm. Gates pass. Acceptance and tokens/step hold.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `moehum` | 57.8 | 102.6 | 135.9 | 152.0 | 448.3 | 8.8 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

+8.8 % sum / +9.9 % c6 vs standing sit inside 12.8 %. +6.7 % / +7.3 % vs
`p030b` sit inside 11.0 %. Still -7.3 % / -6.2 % vs `refg030`. Standing
arm remains `proto2-030`. Humming is the current ceiling on this pin, not
a protocol win.

b12x 1.3.0 is on PyPI (pin is 1.2.6). Its MoE tree still has no `static`
backend; cluster ladders match 1.2.6. Not a drop-in for the attributed gap.


## 2026-09-18: FlashInfer CUTLASS MoE dies at SM120 JIT

`MOE_BACKEND=flashinfer_cutlass` on `main-030-rc1`. Engine selected
`FLASHINFER_CUTLASS_MXFP4_MXFP8` / `FlashInferExperts`, then ninja failed:

```
FAILED: [code=4] .../fused_moe_120/120_cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group3.generated.cuda.o
RuntimeError: Ninja build failed
```

SM120 CUTLASS MoE does not compile on this FlashInfer 0.7.0 / CUDA 13.3.1
stack. Prior `moe_cutlass` on the old pin never became healthy; this is
the reason. Standing arm remains `proto2-030` (sum 412.1) vs `refg030`
(483.7). Best same-image number remains `moehum` 448.3, still inside
keep-spread.


## 2026-09-18: Marlin MoE is the best same-image sum, not a keep

`MOE_BACKEND=marlin` on `main-030-rc1`. Engine selected `MARLIN`. Gates
pass. Acceptance and tokens/step hold.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `moehum` | 57.8 | 102.6 | 135.9 | 152.0 | 448.3 | 8.8 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

+10.5 % sum / +11.9 % c6 vs standing sit inside 12.8 %. +8.3 % / +9.3 % vs
`p030b` sit inside 11.0 %. Still -5.9 % / -4.5 % vs `refg030`. Standing
arm remains `proto2-030`. Marlin is the current ceiling on this pin, not
a protocol win.


## 2026-09-18: linear Marlin dies on O-proj shapes

`LINEAR_BACKEND=marlin` on `main-030-rc1`. Selected
`MarlinFP8ScaledMMLinearKernel`, then:

```
RuntimeError: Expected size for first two dimensions of batch2 tensor to be:
[4, 4096] but got: [4, 1024]
```

Same O-proj (`o_lora_rank` 1024 vs hidden 4096) as `proto2-linhum`. Linear
family on this pin: `b12x` works, `deep_gemm` worse/unstable, `humming` and
`marlin` cannot load. Standing arm remains `proto2-030`. Best same-image
number remains `moemar` 455.3, still inside keep-spread vs `refg030` 483.7.


## 2026-09-18: FlashInfer indexer top-k is pin noise

`--sparse-indexer-topk-backend flashinfer` on `main-030-rc1`. Confirmed in
non-default args. Auto already picks `persistent` at topk 512.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `idxfi` | 55.6 | 97.3 | 124.2 | 142.3 | 419.4 | 12.2 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. One pass dipped below the standing
accept/tokens-per-step floor. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3, still inside keep-spread.


## 2026-09-18: long-prefill 1024 is pin noise

`--long-prefill-token-threshold 1024` on `main-030-rc1`. Pin already named
it; `05-serve.sh` never forwarded it. vLLM default is 0. Confirmed in
non-default args.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `lpf1024` | 56.0 | 97.9 | 126.7 | 144.9 | 425.5 | 14.4 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Acceptance and tokens/step hold. Sum sits on `p030b`. Standing
arm remains `proto2-030`. Best same-image number remains `moemar` 455.3,
still inside keep-spread.


## 2026-09-18: b12x MXFP4 BF16 activations are not the remaining gap

`VLLM_B12X_MOE_FP4_FORCE_A16=1` on `main-030-rc1`. Engine selected
`B12X_MXFP4_BF16`. Reference aliases `flashinfer_b12x` onto `B12X_MXFP4`
(BF16 act) under b12x 0.15.3; older library is banned. This is the
analogue on 1.2.6.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `moea16` | 59.3 | 105.4 | 135.4 | 146.2 | 446.3 | 10.2 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. +8.3 % sum vs standing sits inside 12.8 %. Behind `moemar`.
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3, still inside keep-spread vs `refg030`.


## 2026-09-18: Triton FP8 linear is pin noise

`LINEAR_BACKEND=triton` on `main-030-rc1`. Engine selected
`TritonFp8BlockScaledMMKernel`. MoE stayed b12x.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `lintri` | 52.5 | 93.0 | 129.2 | 141.7 | 416.4 | 10.2 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. One pass dipped below the standing
accept/tokens-per-step floor. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3, still inside keep-spread.


## 2026-09-18: b12x dense GEMM 24-atom is pin noise

`B12X_DENSE_ATOM_24=1` on `main-030-rc1`. Default 0. Experimental atom
choice; keyed into the compile cache. Hits B12x FP8 linear via dense_gemm.
Confirmed in container env. Linear stayed `B12xFp8BlockScaledMMKernel`.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `atom24` | 52.8 | 103.0 | 125.4 | 144.8 | 426.0 | 9.7 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. One pass dipped below the standing
accept/tokens-per-step floor. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3, still inside keep-spread.


## 2026-09-18: FlashInfer MLA DSV4 is pin noise on v0.30.0rc1

`ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` = `FLASHINFER_MLA_SPARSE_DSV4`
on `main-030-rc1`. Reference auto-picks this on SM12x. Our pin forces
`B12X_MLA_SPARSE`. SM120 DSV4 specialization is present. Boot fell back to
FlashInfer's default tactic heuristic (cold autotune cache), same path as
`proto2-attnfi` on the old pin.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `attnfi030` | 56.6 | 99.7 | 129.8 | 145.3 | 431.4 | 10.6 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. +4.7 % sum vs standing sits inside 12.8 %. Behind `moemar`.
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3, still inside keep-spread vs `refg030`.


## 2026-09-18: b12x MoE fast-math off is pin noise

`B12X_FAST_MATH=0` on `main-030-rc1`. Default True; keyed into the
dynamic W4A8 kernel cache. Confirmed in container env. Linear stayed
`B12xFp8BlockScaledMMKernel`.

Warm FLASHINFER MLA autotune skipped: 0.7.0 cache JSON is metadata-only
(499 B, no tactic entries). A rerun would still hit the heuristic.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `nofast` | 56.2 | 101.1 | 124.4 | 142.7 | 424.4 | 16.4 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. One pass dipped below the standing
accept/tokens-per-step floor. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3, still inside keep-spread.


## 2026-09-18: sparse-indexer overlay-off does not fit KV

`VLLM_USE_B12X_SPARSE_INDEXER=0` on `main-030-rc1`. Overlay-off is hungrier
than the pin. Engine died: 9.48 GiB KV needed vs 9.46 GiB available.
Raising util or cutting max_model_len would be a second variable.

Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3.


## 2026-09-18: isolated WO-projection overlay is a cost

`VLLM_USE_B12X_WO_PROJECTION=0` on `main-030-rc1`. Overlay fused inv-RoPE
FP8 + bmm off; einsum fallback. Confirmed in container env. `stockops2`
mixed this with indexer-off; this isolates WO.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `nowo` | 58.8 | 105.4 | 132.3 | 151.5 | 448.0 | 5.4 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. +8.7 % sum vs standing sits inside 12.8 %. Behind `moemar`.
WO overlay is the slower path; indexer overlay stays load-bearing
(`noidx` died on KV). Standing arm remains `proto2-030`.


## 2026-09-18: dense GEMM split-K turbo-off is a quality fail

`B12X_DENSE_SPLITK_TURBO=0` on `main-030-rc1`. Default 1. Decode policy
picks 2-way split-K for m in 2..6 and k >= 4096. Confirmed in container
env. Linear stayed `B12xFp8BlockScaledMMKernel`.

| tag | c1 | c3 | c5 | c6 | sum | worst spread | gates |
|---|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % | pass |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % | pass |
| `noturbo` | 53.7 | 96.1 | 128.8 | 144.5 | 423.1 | 8.9 % | **9x8 fail** |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % | pass |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % | pass |

`gate_9x8` garbage in all three passes (`E5=8F=`, `0x9a,0`). Atomic-BF16
split-K is numerically required. Standing arm remains `proto2-030`.


## 2026-09-18: GPU util 0.82 does not fit KV on this pin

`GPU_MEMORY_UTILIZATION=0.82` on `main-030-rc1` (anemll recipe value).
Profiler maps 0.8200 -> effective 0.7944. Available KV 7.87 GiB vs 9.48
GiB needed. Cutting max_model_len or turning the profiler off would be
a second variable. Standing arm remains `proto2-030`.


## 2026-09-18: micro MoE shared-input off is pin noise

`B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS=0` on `main-030-rc1`. Default 1.
Fires on W4A8 micro at m=1 silu. Confirmed in container env.

GPU util 0.82 (anemll recipe) died: profiler maps 0.8200 -> 0.7944,
KV 7.87 GiB vs 9.48 needed. Packed-B expand is MX-FP6 only. MHC overlay
env `VLLM_USE_B12X_MHC` is unread.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `noshare` | 54.6 | 100.0 | 128.8 | 141.4 | 424.8 | 11.6 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3, still inside keep-spread.


## 2026-09-18: indexer top-k per_row is pin noise

`--sparse-indexer-topk-backend per_row` on `main-030-rc1`. Auto already
picks `persistent` at topk 512. flashinfer was `idxfi`. Confirmed in
non-default args.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `idxpr` | 52.8 | 96.2 | 128.5 | 145.9 | 423.4 | 9.4 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. One pass dipped to accept 48.9 %.
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3, still inside keep-spread.


## 2026-09-18: MoE multi-CTA off collapses decode ~6x

`B12X_DYNAMIC_ENABLE_MULTICTA=0` on `main-030-rc1`. Default 1. Off forces
`effective_mac=1`. Confirmed in container env. Linear stayed
`B12xFp8BlockScaledMMKernel`.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `nomulti` | 11.0 | 15.7 | 19.7 | 20.9 | 67.3 | 9.1 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Acceptance and tokens/step hold. Throughput collapsed ~6x.
Multi-CTA occupancy is required on this pin. Standing arm remains
`proto2-030`. Best same-image number remains `moemar` 455.3.


## 2026-09-18: deterministic MoE output does not fit KV

`B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` on `main-030-rc1`. Engine died:
9.48 GiB KV needed vs 9.24 GiB available. Raising util or cutting
max_model_len would be a second variable. Standing arm remains
`proto2-030`.


## 2026-09-18: generation_config=vllm is pin noise

`--generation-config vllm` on `main-030-rc1`. Reference recipe sets this.
Ours default `auto`. Protocol requests send temperature. Confirmed in
non-default args.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `gencvllm` | 55.9 | 101.4 | 127.7 | 141.8 | 426.8 | 6.8 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3, still inside keep-spread.


## 2026-09-18: DSpark k=5 wins c6, loses tokens/step and the sum

`NUM_SPECULATIVE_TOKENS=5` on `main-030-rc1`. Checkpoint native k=5.
Capture 48 covers 6*(5+1)=36. Confirmed `num_speculative_tokens: 5`.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `k5` | 55.5 | 96.6 | 118.1 | 163.2 | 433.4 | 10.5 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

c6 +18 % vs standing is outside 12.8 %. Sum +5.2 % is inside. Tokens/step
4.031-4.571, below standing floor 4.414. Not a keep. Standing arm remains
`proto2-030`. Best same-image number remains `moemar` 455.3.


## 2026-09-18: DSpark k=6 keeps tokens/step, still inside keep-spread

`NUM_SPECULATIVE_TOKENS=6` on `main-030-rc1`. Midpoint between checkpoint
k=5 and pin k=7. Capture 48 covers 6*(6+1)=42. Confirmed
`num_speculative_tokens: 6`.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `k5` | 55.5 | 96.6 | 118.1 | 163.2 | 433.4 | 10.5 % |
| `k6` | 58.4 | 106.7 | 132.8 | 151.6 | 449.5 | 9.6 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Tokens/step 4.437-5.120 holds (k=5 did not). Sum +9.1 % vs
standing sits inside 12.8 %. Behind `moemar`. Standing arm remains
`proto2-030`.


## 2026-09-18: async scheduling off is pin noise

`--no-async-scheduling` on `main-030-rc1`. Last-flag wins over run-arm's
`--async-scheduling`. Confirmed `async_scheduling: False`.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `noasync` | 55.7 | 97.4 | 127.0 | 143.0 | 423.1 | 11.8 % |
| `k6` | 58.4 | 106.7 | 132.8 | 151.6 | 449.5 | 9.6 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. One pass dipped below the standing
accept/tokens-per-step floor. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3.


## 2026-09-18: indexer direct gather off collapses decode

`VLLM_B12X_INDEXER_DIRECT_GATHER=0` on `main-030-rc1`. Default 1.
Proto-era +27 % win. Never isolated off on this pin. Confirmed in
container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `nogath` | 37.0 | 76.2 | 107.2 | 119.9 | 340.3 | 3.5 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Acceptance and tokens/step hold. Throughput dropped 17.4 %
on the sum. Direct gather is required. Standing arm remains
`proto2-030`. Best same-image number remains `moemar` 455.3.


## 2026-09-18: CUDA_DEVICE_MAX_CONNECTIONS=8 is pin noise

`CUDA_DEVICE_MAX_CONNECTIONS=8` on `main-030-rc1`. Pin defaults to 1.
CUDA default is 8. Anemll recipe does not set it. Confirmed in
container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `conn8` | 54.7 | 100.4 | 125.8 | 141.7 | 422.6 | 7.9 % |
| `nogath` | 37.0 | 76.2 | 107.2 | 119.9 | 340.3 | 3.5 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3. Direct gather remains
load-bearing (`nogath` -17.4 % sum).


## 2026-09-18: prefix cache off is pin noise

`--no-enable-prefix-caching` on `main-030-rc1`. Pin default on. Confirmed
`enable_prefix_caching: False`. Protocol reuses one prompt.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `nopfx` | 53.9 | 98.2 | 124.0 | 142.2 | 418.3 | 8.5 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3.


## 2026-09-18: chunked prefill off dies at engine start

`--no-enable-chunked-prefill` on `main-030-rc1`. Confirmed
`enable_chunked_prefill: False`. vLLM warns DSV4 does not officially
support disabling chunked prefill. Container exited 1. Prefix-cache
off (`nopfx`) served and sat in pin noise.

Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3.


## 2026-09-18: indexer stream scorer off is pin noise

`B12X_INDEXER_STREAM_SCORER=0` on `main-030-rc1`. Default True.
Confirmed in container env. Overlay still scores via `logits_paged`.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `nopfx` | 53.9 | 98.2 | 124.0 | 142.2 | 418.3 | 8.5 % |
| `nostream` | 52.5 | 100.0 | 133.6 | 141.6 | 427.7 | 9.7 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. One pass dipped below the standing
accept/tokens-per-step floor. Chunked-prefill off (`nochunk`) died at
engine start (DSV4 requires it). Standing arm remains `proto2-030`.
Best same-image number remains `moemar` 455.3.


## 2026-09-19: dynamic MoE down-scale is pin noise

`B12X_ENABLE_DYNAMIC_DOWN_SCALE=1` on `main-030-rc1`. Default False.
Fires on MXFP4 (`not is_w4a8`). Confirmed in container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `downsc` | 56.0 | 97.3 | 129.1 | 140.1 | 422.5 | 5.0 % |
| `nostream` | 52.5 | 100.0 | 133.6 | 141.6 | 427.7 | 9.7 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3.


## 2026-09-19: W4A8 share-input off is pin noise

`B12X_DYNAMIC_W4A8_SHARE_INPUT=0` on `main-030-rc1`. DSV4 MXFP4 experts
map to `quant_mode=w4a8_mx`. Opposite of `moeshare`. Confirmed in
container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `now4sh` | 52.5 | 100.1 | 127.4 | 140.1 | 420.1 | 8.3 % |
| `downsc` | 56.0 | 97.3 | 129.1 | 140.1 | 422.5 | 5.0 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. One pass dipped below the standing
accept/tokens-per-step floor. `downsc` is a no-op on `w4a8_mx`
(`not is_w4a8`). Standing arm remains `proto2-030`. Best same-image
number remains `moemar` 455.3.


## 2026-09-19: MoE tile 128x128 is slightly worse

`B12X_DYNAMIC_TILE_MN=128x128` on `main-030-rc1`. Auto planner is
16x128 on the protocol band. 32x128 and 64x128 were already pin noise.
Confirmed in container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `tile128` | 55.3 | 95.4 | 124.4 | 136.2 | 411.3 | 12.3 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum and c6 sit at or below standing. Auto 16x128 stays.
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3.


## 2026-09-19: W4A8 tiny-decode off is pin noise

`B12X_W4A8_TINY_DECODE=0` on `main-030-rc1`. Default 1. On SM121 DSV4F
the tiny path only owns c1 (`num_tokens>=3` excluded). Confirmed in
container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `notiny` | 52.8 | 99.2 | 133.1 | 141.0 | 426.1 | 8.9 % |
| `tile128` | 55.3 | 95.4 | 124.4 | 136.2 | 411.3 | 12.3 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3.


## 2026-09-19: MLA prefill-MG off dies at engine start

`B12X_MLA_SM120_PREFILL_MG=0` on `main-030-rc1`. Default 1. Overlay
`B12X_MLA_SPARSE` reuses the prefill MG kernel for decode when
`rows>=16`. Worker:

`ValueError: SM120 sparse MLA prefill: unsupported shape (model_type=0,
heads=32, topk=512, ..., B12X_MLA_SM120_PREFILL_MG=0). No decode-reuse
fallback.`

DSV4 requires MG. Standing arm remains `proto2-030`. Best same-image
number remains `moemar` 455.3.


## 2026-09-19: MLA num_splits=1 is pin noise

`B12X_MLA_SM120_NUM_SPLITS=1` on `main-030-rc1`. Unset uses the
wave-balanced heuristic. Overlay decode reads this env. Confirmed in
container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `split1` | 53.0 | 100.5 | 124.8 | 141.7 | 420.0 | 12.6 % |
| `notiny` | 52.8 | 99.2 | 133.1 | 141.0 | 426.1 | 8.9 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. Prefill-MG off (`nomg`) died at engine
start. Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3.


## 2026-09-19: DSV4 H16 native off is pin noise

`B12X_MLA_SM120_DSV4_H16_NATIVE=0` on `main-030-rc1`. Unset is auto; on
Spark auto turns H16 on for many-chunk / batched-row decode. Overlay
decode reads this env. Confirmed in container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `split1` | 53.0 | 100.5 | 124.8 | 141.7 | 420.0 | 12.6 % |
| `noh16` | 53.6 | 100.8 | 129.5 | 143.4 | 427.3 | 8.0 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. Standing arm remains `proto2-030`. Best
same-image number remains `moemar` 455.3.


## 2026-09-19: DSV4 H16 native force-on is pin noise

`B12X_MLA_SM120_DSV4_H16_NATIVE=1` on `main-030-rc1`. Unset is auto;
`noh16` forced 0. Force 1 so small-row c1 also uses H16. Overlay decode
reads this env. Confirmed in container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `yesh16` | 55.8 | 96.3 | 127.6 | 138.7 | 418.4 | 8.2 % |
| `noh16` | 53.6 | 100.8 | 129.5 | 143.4 | 427.3 | 8.0 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. H16 auto is not the remaining gap.
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3.


## 2026-09-19: MLA num_splits=2 is pin noise

`B12X_MLA_SM120_NUM_SPLITS=2` on `main-030-rc1`. Unset uses the
wave-balanced heuristic; `split1` pinned 1. Overlay decode reads this
env. Confirmed in container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `split1` | 53.0 | 100.5 | 124.8 | 141.7 | 420.0 | 12.6 % |
| `split2` | 54.7 | 93.1 | 130.5 | 147.5 | 425.8 | 9.1 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. c6 +6.7 % vs standing is inside 12.8 %.
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3.


## 2026-09-19: MLA num_splits=4 is pin noise

`B12X_MLA_SM120_NUM_SPLITS=4` on `main-030-rc1`. Unset uses the
wave-balanced heuristic; `split1` pinned 1 and `split2` pinned 2.
Overlay decode reads this env. Confirmed in container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `split1` | 53.0 | 100.5 | 124.8 | 141.7 | 420.0 | 12.6 % |
| `split2` | 54.7 | 93.1 | 130.5 | 147.5 | 425.8 | 9.1 % |
| `split4` | 51.9 | 100.9 | 126.6 | 142.8 | 422.2 | 11.0 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. The num_splits ladder is not the
remaining gap. Standing arm remains `proto2-030`. Best same-image
number remains `moemar` 455.3.


## 2026-09-19: MLA num_splits=8 is pin noise

`B12X_MLA_SM120_NUM_SPLITS=8` on `main-030-rc1`. Unset uses the
wave-balanced heuristic; `split1`/`split2`/`split4` already sat in pin
noise. Overlay decode reads this env. Confirmed in container env.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `split1` | 53.0 | 100.5 | 124.8 | 141.7 | 420.0 | 12.6 % |
| `split2` | 54.7 | 93.1 | 130.5 | 147.5 | 425.8 | 9.1 % |
| `split4` | 51.9 | 100.9 | 126.6 | 142.8 | 422.2 | 11.0 % |
| `split8` | 53.0 | 97.2 | 125.9 | 142.4 | 418.5 | 8.5 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Gates pass. Sum sits on `p030b`. The num_splits ladder is closed.
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3.


## 2026-09-19: config leftover closed; remaining gap is static MoE

One-variable leftover on `main-030-rc1` that fires on overlay decode is
closed. Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).

Closed this span (all same image, gates pass unless noted):

- MLA `num_splits` ladder 1/2/4/8: pin noise (`split1`/`split2`/`split4`/`split8`).
- H16 native off and force-on: pin noise (`noh16`/`yesh16`).
- Prefill-MG off: dies (`nomg`). DSV4 requires MG.
- Tiny-decode off: pin noise (`notiny`). Only owns c1 on SM121 DSV4F.
- Tile 128x128: slightly worse (`tile128`). Auto 16x128 stays.
- W4A8 share-input off: pin noise (`now4sh`).
- Down-scale on: no-op on `w4a8_mx` (`downsc`).

Dead leftover (not run): `B12X_DYNAMIC_W4A8_MATERIALIZED` is a no-op on
this pin. SM121 DSV4F sets `direct_limit=0`, so `m1_candidate` is false.
Dense candidate needs tile in {32,64,128}x128; auto is 16x128. Forcing
the env still ANDs that candidate. `_apply_mla_prefill_strategy` has no
callers.

| tag | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % |

Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3 (Marlin MoE), still inside keep-spread vs standing and
still 5.9 % / 4.5 % behind `refg030`.

Remaining gap, attributed:

1. KV: real NVFP4 writer closed. Both engines 584 B/token envelope.
2. NVFP4 per family: MoE-NVFP4 unreachable on this MXFP4 checkpoint
   (`quant_mode=w4a8_mx`). Linear `b12x` FP8 is the only working option.
3. Overlay/config bottlenecks: leftover env on this pin sits in ~420
   noise except load-bearing defaults (MULTICTA, split-K turbo, sparse
   indexer, direct gather, prefill-MG, chunked prefill).
4. Third-party kernels: WO overlay is a cost vs einsum. FlashInfer MLA
   is not the gap. **MoE `static` is still missing in b12x 1.2.6 and
   1.3.0.** Reference fork aliases `flashinfer_b12x` onto a per-row-count
   `static` ladder. Do not swap older b12x. Do not bump 1.3.0 without
   isolating cutlass-dsl (two variables).


## 2026-09-19: b12x 1.2.8 and 1.3.0 still have no static MoE

PyPI latest is 1.3.0. Wheel file lists for 1.2.6, 1.2.8, and 1.3.0 have
no `static` MoE module. Decode cluster policy is still only `micro` /
`dynamic` / `dynamic_w4a8_decode`. Dynamic ladder still flat at 188
until 640 routed rows.

Both 1.2.8 and 1.3.0 pin `nvidia-cutlass-dsl==4.6.2`. The
`main-030-rc1` image has cutlass-dsl 4.7.0. A bump is two variables
(b12x + cutlass-dsl). Do not bump.

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3. Remaining gap is the missing `static` MoE kernel in
current b12x, not overlay leftover.


## 2026-09-19: leftover closed; fallback is moemar plus static-MoE attribution

Goal fallback: best measured arm plus written attribution. Latest
upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).

Best same-image arm: `moemar` (`MOE_BACKEND=marlin`) 455.3 / c6 154.8.
Standing keep-rule arm remains `proto2-030` 412.1 / 138.3. `moemar` is
inside keep-spread vs standing (12.8 %) and vs `p030b` (11.0 %). vs
`refg030` 483.7 / 162.1 still -5.9 % / -4.5 %.

Attribution:

- Layer family: MoE experts (ffn), not attention, not linear.
- Library: b12x 1.2.6 (and 1.2.8 / 1.3.0) has no `static` MoE kernel.
  Reference fork runs a per-row-count `static` ladder under the
  `flashinfer_b12x` name. Our MXFP4 oracle maps that name to NVFP4 and
  rejects it. Marlin is the best reachable MXFP4 expert path on this
  pin and still does not clear the keep-gate vs reference.
- Measurement: protocol medians on `main-030-rc1`, same day as
  `refg030`. Overlay leftover env sits in ~420 noise except load-bearing
  defaults. Do not bump b12x without isolating cutlass-dsl 4.7.0 vs
  4.6.2. Do not swap older b12x.


## 2026-09-19: MXFP4 oracle leftover closed; fallback holds

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).

MXFP4 `moe_backend` names that can serve on this pin were measured:
`b12x` (standing), `marlin` (`moemar` 455.3, best same-image),
`humming` (`moehum` 448.3). Dead or unsupported: `flashinfer_b12x`
(NVFP4-experts name), `deep_gemm` (KV short), `flashinfer_cutlass`
(ninja SM120), `flashinfer_trtllm` (device), `triton` (noise on
linear; MoE unfused has MTP bug). `emulation` is correctness, not
speed. `BATCHED_MARLIN` is auto-substituted from `marlin`, not a
separate string. AITER is ROCm.

Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3. vs `refg030` still -5.9 % / -4.5 %. Remaining gap is
missing `static` MoE in b12x 1.2.6/1.2.8/1.3.0.


## 2026-09-19: MXFP4 Triton MoE excludes SM120

`MOE_BACKEND=triton` maps to `OAITritonMxfp4ExpertsMonolithic` /
`OAITritonExperts`. Device gate
`_triton_kernel_moe_supports_current_device` keeps CUDA only in
`(9, 0) <= cap < (11, 0)` (Hopper SM90 / Blackwell SM100). SM120/SM121
is excluded by design; comment says the broader CUDA range was not
validated. Do not launch. `triton_unfused` shares the same gate.

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3.


## 2026-09-19: FlashInfer MoEStaticKernel is in the image, unused

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).

`flashinfer.fused_moe.cute_dsl.blackwell_sm12x` in `main-030-rc1`
already ships `MoEStaticKernel` plus `select_sm120_moe_backend`:
routed pairs <= 640 (MXFP4) take `static`. Protocol c6 is 48 tokens
x topk 6 = 288 pairs, inside that band. `b12x_moe.run` accepts
`quant_mode="mxfp4"`.

Overlay never calls it. `patches/files/fused_moe_b12x.py` goes
through `get_b12x_fused_moe()` -> `b12x.moe.fused_moe` (1.2.6),
which has no `static` module and only `dynamic` above 20 routed
rows. `MOE_BACKEND=flashinfer_b12x` is `FlashInferB12xExperts`,
NVFP4-only (`kNvfp4Static`); MXFP4 rejects the name (`fb12x`).

Wiring MXFP4 source weights into `launch_sm120_moe` is an overlay
code change, not an env leftover. Do not treat it as a
`SERVE_EXTRA_ENV` arm. Weight layout / e8m0 scales / in-kernel
BF16 quant vs overlay `w4a8_mx` still unproven. Next one-variable
is that overlay, not another config knob.

`MOE_BACKEND=triton` excludes SM120 (`cap < (11,0)`). Do not
launch.

Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3. vs `refg030` still -5.9 % / -4.5 %.


## 2026-09-19: FlashInfer static MXFP4 overlay dies at worker IMA

One-variable overlay: bind-mount `patches/files/fused_moe_b12x.py` so
`w4a8_mx` apply uses image FlashInfer `B12xMoEWrapper(quant_mode="mxfp4")`.
Dummy zeros on protocol shapes compiled
`static_m8_k6144_n1024_t6_r48`. Live checkpoint died at engine start
with `Triton Error [CUDA]: an illegal memory access was encountered`
during KV-cache sizing / warmup. `MOE_BACKEND` stayed `b12x`.

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3. vs `refg030` still -5.9 % / -4.5 %. Remaining gap is
still the missing `static` MoE on the live MXFP4 weight path.


## 2026-09-19: FlashInfer static overlay hangs on live MXFP4

`fistat` IMA'd during warmup. `fistat2` stashed contiguous source
weights and skipped warmup. Prepare logged
`b12x MoE MXFP4 using FlashInfer MoEStaticKernel`. Weights loaded.
Then GPU 96 % for 20+ min, no CuTe compile line, shm_broadcast 60 s
timeouts, health never 200.

Dummy zeros compiled `static_m8_k6144_n1024_t6_r48`. Live checkpoint
does not. Overlay wiring of MXFP4 into image `MoEStaticKernel` is not
a keep path without a working weight-view / graph-capture fix.

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3. vs `refg030` still -5.9 % / -4.5 %. Remaining gap is
still the missing `static` MoE on the live MXFP4 path.


## 2026-09-19: FlashInfer static overlay cannot profile at 12288

`fistat` IMA'd on live warmup. `fistat2` hung allocating 43 wrappers
at `max_num_tokens=12288`. `fistat3` used one shared wrapper at
capture 48. Prepare logged. Then
`ValueError: num_tokens (12288) exceeds max_num_tokens (48)` during
KV-cache profile.

Dummy zeros compiled `static_m8`. Live path needs a workspace that
covers profile 12288 without 43x hang, and a weight-view that does
not IMA. That is still overlay code, not a keep. Do not keep
iterating wrapper sizes this round.

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3. vs `refg030` still -5.9 % / -4.5 %. Remaining gap is
still the missing `static` MoE on the live MXFP4 path.


## 2026-09-19: one shared 12288 FI wrapper still hangs

`fistat` IMA on live warmup. `fistat2` 43x 12288 hang. `fistat3`
shared 48 dies on profile 12288. `fistat4` one shared wrapper at
12288: prepare logged, weights loaded, GPU 96 %, no CuTe compile,
same hang as `fistat2`.

Dummy zeros compiled `static_m8_k6144_n1024_t6_r48`. Live MXFP4
into image `MoEStaticKernel` does not serve. Stop iterating wrapper
sizes. Remaining gap is still the missing `static` MoE on the live
MXFP4 path, not overlay leftover env.

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3. vs `refg030` still -5.9 % / -4.5 %.


## 2026-09-19: reverted experimental FlashInfer static overlay

`patches/files/fused_moe_b12x.py` restored to stock (no
`_FI_SHARED_WRAPPER`). Bind-mounts of the experimental file are
still recorded as `fistat`/`fistat2`/`fistat3`/`fistat4` failures.
Rebuilds of `main-030-rc1` stay on b12x 1.2.6 dynamic MoE.

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).
Standing arm remains `proto2-030`. Best same-image number remains
`moemar` 455.3. vs `refg030` still -5.9 % / -4.5 %. Remaining gap:
missing `static` MoE on the live MXFP4 path. Dummy zeros compiled
`static_m8`; live weights did not serve.


## 2026-09-19: fallback stands — moemar plus static-MoE attribution

Latest upstream serving tag remains `v0.30.0rc1` (`a00a3544b93e`).
Overlay leftover env closed. Experimental FlashInfer-static overlay
reverted (`fistat`–`fistat4` failed). Stock `fused_moe_b12x.py`.

Best same-image arm: `moemar` 455.3 / c6 154.8. Standing keep-rule
arm: `proto2-030` 412.1 / 138.3. vs `refg030` 483.7 / 162.1 still
-5.9 % / -4.5 %. Keep-rule not met.

Attribution: MoE experts, not attention/linear. Library: b12x 1.2.6
(and 1.2.8 / 1.3.0) has no `static` kernel. Image FlashInfer has
`MoEStaticKernel`; dummy zeros compiled `static_m8`; live MXFP4
weights did not serve. Measurement: protocol medians on
`main-030-rc1`, same day as `refg030`.


## 2026-09-19: base moved to v0.30.0rc2

Latest upstream serving tag is `v0.30.0rc2` (`fa6ff060667f2cd96f142af021ba9e35165beb78`).
One commit ahead of `v0.30.0rc1` (`a00a3544b93e`): `#57570` NIXL skip
receive reports for notification-only requests.

`scripts/port_scan.py --with-upstream-patches` against a clean
`v0.30.0rc2` tree reports **FAIL=0  applied=43  no-op=12  total=55**.
Upstream patches: applied `pr-47988` / `pr-53425` / `pr-53522`, skip
`pr-53055` (already in the base). Same shape as rc1.

Pin: `configs/pin.main-029.env` now has `VLLM_REF=fa6ff060667f…` and
`IMAGE=vllm-spark-0731:main-030-rc2`. Goal contract
(`docs/GOAL-BEAT-ANEMLL.md`) and `HANDOVER.md` / `docs/LINEAGE.md`
moved with it. Recipe: `configs/examples/port-v0.30.0rc2.sh`.

Dirty-tree scan after a first apply reported FAIL=2 (`patch_o_proj_b12x`,
`patch_tp_allreduce_static_workspace`) because rc2 already contains a
partial form of those needles. A reset + one clean scan is FAIL=0.
Do not treat the dirty FAIL=2 as a real overlay miss.

No protocol measurement on this base yet; rc1 numbers stay the
previous-pin record until phase 1 + overlays land.


## 2026-09-19: other pins to latest tagged versions

Alongside `v0.30.0rc2`, floating git HEADs and stale PyPI pins moved
to the latest *tagged* versions. Overlay patches stay on top.

| pin | was | now | note |
|---|---|---|---|
| `FLASHINFER_REF` | `main` | `v0.7.0rc3` | latest tag |
| `INSTANTTENSOR_REF` | `main` | `v0.2.0` | tag == previous main HEAD |
| `FASTSAFETENSORS_REF` | `main` | `0.4.0` | latest tag |
| `LMCACHE_REF` | `dev` | `v0.5.5` | unused at serve |
| `DEEPEP_REF` | `main` | `v1.2.1` | latest tag |
| `HUMMING_KERNELS_VERSION` | 0.1.13 | 0.1.15 | latest PyPI |
| `TILELANG_VERSION` | 0.1.14 | 0.1.14 | already latest |
| `QUACK_KERNELS_VERSION` | 0.6.5 | 0.6.5 | already latest |
| `B12X_VERSION` | 1.2.6 | 1.2.6 | 1.3.0 pins cutlass-dsl 4.6.2 vs image 4.7.0 |
| `CUTLASS_DSL_VERSION` | 4.7.0 | 4.7.0 | 4.7.1 exists; leave until b12x bump is isolated |
| `TRITON_VERSION` | 3.7.1 | 3.7.1 | torch 2.14 ships 3.7.1 |
| `APACHE_TVM_FFI_VERSION` | 0.1.12 | 0.1.12 | tilelang 0.1.14 requires <0.1.13 |
| `TOKENSPEED_MLA_VERSION` | 0.2.8 | 0.2.8 | 0.2.9 requires tvm-ffi 0.1.13.post3 |
| `NVIDIA_CUDA_NVDISASM_VERSION` | 13.3.73 | 13.3.73 | keep with CUDA 13.3.1 image |
| `TORCH_REF` | release/2.14 | release/2.14 | branch pin, not a tag |

In-flight `main-030-rc2-phase1` used the old floating HEADs. Kill and
rebuild so the image matches these tags.


## 2026-09-19: v0.30.0rc2 image landed, protocol re-baselined

Phase 1 `vllm-spark-0731:main-030-rc2-phase1` (`641ef9529e73`) on spark1
with `VLLM_REF=fa6ff060667f`. `vllm.__version__` =
`0.30.0rc3.dev0+gfa6ff0606`. Phase 2 overlays (`5837f2704e77`)
`assert_image.py --stack main` printed `image OK`. Copied to spark2
(same sha).

Protocol `proto2-030-rc2`: 51.7 / 98.0 / 126.6 / **141.2**, sum
**417.5**, worst spread 14.0 %. Gates pass. Accept 49.2-56.7 %,
tokens/step 4.427-4.963. Sits on rc1 `p030b` (420.3). vs `refg030`
still -13.7 % / -12.9 %.

Tagged deps in this image: FlashInfer `v0.7.0rc3`, InstantTensor
`v0.2.0`, fastsafetensors `0.4.0`, LMCache `v0.5.5`, DeepEP `v1.2.1`,
humming-kernels 0.1.15. b12x 1.2.6 / cutlass-dsl 4.7.0 / tilelang
0.1.14 / tvm-ffi 0.1.12 unchanged.


## 2026-09-19: standing goal rewritten

`docs/GOAL-BEAT-ANEMLL.md` is now the standing contract: latest tagged
vLLM + tagged deps + overlay set, beat anemll, upstream via PRs, check
our four PRs every round, port on every new tag. Current pin remains
`v0.30.0rc2` (`fa6ff060667f`). Ours still OPEN behind `pre-run-check`:
#53425 #53522 #53271 #46716. Action: none until rebase / review /
merge.


## 2026-09-19: goal file rewritten as standing rules

`docs/GOAL-BEAT-ANEMLL.md` no longer holds pin SHAs or protocol numbers.
Those live in `configs/pin.main-029.env` and `docs/EXPERIMENTS.md`. The
goal file is the loop: tags first, PRs second, one-variable third,
keep-rule, same-day `refg`. Closed arms listed so they are not re-run.


## 2026-09-19: same-day anemll refg-rc2

Latest vLLM tag still `v0.30.0rc2`. Dep tags current except known holds
(b12x 1.3.0 / cutlass-dsl 4.7.1 / triton 3.8 / tvm-ffi 0.1.14 / tokenspeed
0.2.9). Ours still OPEN behind `pre-run-check`: #53425 #53522 #53271
#46716. Action: none.

Same-day anemll `refg-rc2`: 58.6 / 113.4 / 138.8 / **157.1**, sum
**467.9**, worst spread 21.9 %. Gates pass. vs `refg030` (483.7 / 162.1)
this is -3.3 % / -3.1 % (rig). Standing control `proto2-030-rc2`
(417.5 / 141.2) is still **-10.8 % / -10.1 %** vs this `refg-rc2`.

Recipe: `configs/examples/refg-rc2.sh`. Log:
`outputs/driver/refg-rc2.median.log`.


## 2026-09-19: moemar-rc2 dies at KV floor

`MOE_BACKEND=marlin` on `main-030-rc2` needs 9.48 GiB KV, has 9.27 GiB
at util 0.8389 / max_model_len 65536. rc1 `moemar` served. Do not raise
util as a second variable. Standing arm remains `proto2-030-rc2`.
Same-day anemll `refg-rc2` is 467.9 / 157.1.


## 2026-09-19: nowo-rc2 WO overlay still a cost

`VLLM_USE_B12X_WO_PROJECTION=0` on `main-030-rc2`: 57.7 / 102.6 / 132.9 /
**149.0**, sum **442.2**, spread 5.7 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2, 14.0 %) this is +5.9 % / +5.5 %, inside
keep-spread. vs same-day `refg-rc2` (467.9 / 157.1) still -5.5 % / -5.2 %.
WO overlay remains a cost. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-19: moehum-rc2 not a keep

`MOE_BACKEND=humming` on `main-030-rc2` (humming-kernels 0.1.15): 58.1 /
102.2 / 132.3 / **152.9**, sum **445.5**, spread 4.7 %. Gates pass. vs
standing `proto2-030-rc2` (417.5 / 141.2, 14.0 %) this is +6.7 % / +8.3 %,
inside keep-spread. One pass accept 49.1 % dips the 49.2 % floor. vs
same-day `refg-rc2` (467.9 / 157.1) still -4.8 % / -2.7 %. Best same-pin
number so far, not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
#46716 mergeable_state was `unknown` this round (API flake); still OPEN.


## 2026-09-19: k6-rc2 not a keep

`NUM_SPECULATIVE_TOKENS=6` on `main-030-rc2`: 55.9 / 107.6 / 132.2 /
**149.9**, sum **445.6**, spread 10.9 %. Gates pass. Accept 58.1-63.5 %,
tokens/step 4.476-4.800. vs standing `proto2-030-rc2` (417.5 / 141.2,
14.0 %) this is +6.7 % / +6.2 %, inside keep-spread. vs same-day
`refg-rc2` (467.9 / 157.1) still -4.8 % / -4.6 %. Tied with `moehum-rc2`
(445.5) for best same-pin sum. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-19: moea16-rc2 not a keep

`VLLM_B12X_MOE_FP4_FORCE_A16=1` on `main-030-rc2` (`B12X_MXFP4_BF16`):
56.7 / 105.7 / 138.5 / **152.8**, sum **453.7**, spread 7.1 %. Gates
pass. vs standing `proto2-030-rc2` (417.5 / 141.2, 14.0 %) this is
+8.7 % / +8.2 %, inside keep-spread. vs same-day `refg-rc2` (467.9 /
157.1) still -3.0 % / -2.7 %. Best same-pin sum so far. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-19: k5-rc2 c6 win, tokens/step fail

`NUM_SPECULATIVE_TOKENS=5` on `main-030-rc2`: 56.7 / 106.5 / 118.8 /
**162.0**, sum **444.0**, spread 6.2 %. Gates pass. Accept 66.2-72.0 %.
Tokens/step min 4.303 is below standing floor 4.427. c6 162.0 beats
same-day `refg-rc2` 157.1, but the sum is still -5.1 % vs 467.9 and
+6.3 % vs standing 14.0 % keep-spread. Same pattern as rc1 `k5`. Not
a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-19: attnfi-rc2 pin noise

`FLASHINFER_MLA_SPARSE_DSV4` on `main-030-rc2` (FlashInfer `v0.7.0rc3`):
53.0 / 99.3 / 129.3 / **142.8**, sum **424.4**, spread 5.7 %. Gates
pass. vs standing `proto2-030-rc2` (417.5 / 141.2, 14.0 %) this is
+1.7 % / +1.1 %, pin noise. One pass accept 49.0 % / tps 4.414 dips
the floor. vs same-day `refg-rc2` (467.9 / 157.1) still -9.3 % / -9.1 %.
FlashInfer MLA DSV4 is not the remaining gap. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-19: p030-rc2b confirms ~420 noise center

Same-pin re-baseline of `proto2-030-rc2` (no extra env): 52.8 / 99.6 /
128.0 / **144.2**, sum **424.6**, spread 13.6 %. Gates pass. One pass
accept 45.6 % / tps 4.197 dips the floor. vs original 417.5 / 141.2
this is +1.7 % / +2.1 %, pin noise. The 442-453 cluster is a real
one-variable lift vs the pin, still inside 14.0 % keep-spread. Treat
~420 as the noise center. Best same-pin sum remains `moea16-rc2` 453.7
vs same-day `refg-rc2` 467.9 (-3.0 % / -2.7 %).

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-19: moefic-rc2 still ninja-fails

`MOE_BACKEND=flashinfer_cutlass` on `main-030-rc2` (FlashInfer
`v0.7.0rc3`): engine selected `FLASHINFER_CUTLASS_MXFP4_MXFP8`, then
ninja `FAILED: [code=4]` on `fused_moe_120` SM120 grouped GEMM
(`M128_BS_group5`, `sm_121a`). Same as rc1 `moefic`. CUTLASS MoE does
not compile on this stack. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7 vs `refg-rc2` 467.9.


## 2026-09-19: fifunc IMA on live MXFP4

New hypothesis vs wrapper-size retries: FlashInfer `b12x_fused_moe`
functional API (auto static at c6, dynamic at profile 12288,
module-level workspace). Prepare logged. Then
`CUDA_ERROR_ILLEGAL_ADDRESS` at first launch. Same live-weight IMA as
`fistat`. Overlay reverted to stock. Dummy zeros compiled `static_m8`;
live MXFP4 into image FlashInfer still does not serve.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7 vs `refg-rc2` 467.9.


## 2026-09-19: util8663-rc2 extra KV is pin noise

`GPU_MEMORY_UTILIZATION=0.8663` on `main-030-rc2`: 52.4 / 100.0 / 128.8 /
**143.1**, sum **424.3**, spread 6.9 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.6 % / +1.3 %. vs same-day control
`p030-rc2b` (424.6 / 144.2) a wash. Extra KV does not close the
reference gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.3 % /
-8.9 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: tile128-rc2 pin noise, tile ladder closed

`B12X_DYNAMIC_TILE_MN=128x128` on `main-030-rc2`: 53.8 / 98.8 / 122.6 /
**142.5**, sum **417.7**, spread 6.3 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) a wash. vs `p030-rc2b` (424.6) slightly
worse. Auto (16,128) / 32x128 / 64x128 / 128x128 are all pin noise.
Tile shape is not the remaining MoE gap. vs same-day `refg-rc2` (467.9 /
157.1) still -10.7 % / -9.3 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: moecap175-rc2 pin noise

`B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=175` on `main-030-rc2`: 55.3 / 100.1 /
124.5 / **142.5**, sum **422.4**, spread 8.3 %. Gates pass. One pass
accept 48.2 % / tps 4.376 dips the floor. vs standing `proto2-030-rc2`
(417.5 / 141.2) +1.2 % / +0.9 %. vs `p030-rc2b` (424.6) a wash. Cluster
cap 175 is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1)
still -9.7 % / -9.3 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: gencvllm-rc2 pin noise

`--generation-config vllm` on `main-030-rc2`: 55.9 / 99.1 / 130.4 /
**142.4**, sum **427.8**, spread 7.6 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.5 % / +0.8 %. vs `p030-rc2b` (424.6)
a wash. Neutral vLLM sampling defaults are not the remaining gap. vs
same-day `refg-rc2` (467.9 / 157.1) still -8.6 % / -9.4 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: conn8-rc2 pin noise

`CUDA_DEVICE_MAX_CONNECTIONS=8` on `main-030-rc2`: 55.0 / 98.7 / 130.6 /
**140.3**, sum **424.6**, spread 3.3 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.7 % / -0.6 %. vs `p030-rc2b` (424.6)
a wash. CUDA max connections 8 is not the remaining gap. vs same-day
`refg-rc2` (467.9 / 157.1) still -9.3 % / -10.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: moeshare-rc2 pin noise

`B12X_DYNAMIC_W4A8_SHARE_INPUT=1` on `main-030-rc2`: 51.3 / 98.0 / 129.3 /
**144.2**, sum **422.8**, spread 13.5 %. Gates pass. One pass accept
46.5 % / tps 4.231 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +1.3 % / +2.1 %. vs `p030-rc2b` (424.6 / 144.2) a wash. Forced
share-input on c3-c6 is not the remaining gap. vs same-day `refg-rc2`
(467.9 / 157.1) still -9.6 % / -8.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: moework-rc2 pin noise

`B12X_DYNAMIC_WORK_SOURCE=persistent_grid` on `main-030-rc2`: 53.9 /
102.9 / 132.2 / **142.2**, sum **431.2**, spread 10.2 %. Gates pass. One
pass accept 48.9 % / tps 4.414 dips the floor. vs standing
`proto2-030-rc2` (417.5 / 141.2) +3.3 % / +0.7 %. vs `p030-rc2b` (424.6)
inside keep-spread. Work source is not the remaining gap. vs same-day
`refg-rc2` (467.9 / 157.1) still -7.8 % / -9.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: atom24-rc2 pin noise

`B12X_DENSE_ATOM_24=1` on `main-030-rc2`: 54.0 / 97.6 / 130.8 / **141.6**,
sum **424.0**, spread 15.4 %. Gates pass. One pass accept 44.2 % / tps
4.096 dips the floor. vs standing `proto2-030-rc2` (417.5 / 141.2)
+1.6 % / +0.3 %. vs `p030-rc2b` (424.6) a wash. 24-atom dense GEMM is
not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still
-9.4 % / -9.9 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: nofast-rc2 pin noise

`B12X_FAST_MATH=0` on `main-030-rc2`: 51.4 / 99.5 / 130.2 / **142.3**,
sum **423.4**, spread 5.1 %. Gates pass. One pass accept 48.1 % / tps
4.339 dips the floor. vs standing `proto2-030-rc2` (417.5 / 141.2)
+1.4 % / +0.8 %. vs `p030-rc2b` (424.6) a wash. Fast-math off is not
the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still
-9.5 % / -9.4 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: lpf1024-rc2 pin noise

`--long-prefill-token-threshold 1024` on `main-030-rc2`: 53.9 / 99.2 /
129.0 / **143.4**, sum **425.5**, spread 7.3 %. Gates pass. One pass
accept 49.0 % / tps 4.414 dips the floor. vs standing `proto2-030-rc2`
(417.5 / 141.2) +1.9 % / +1.6 %. vs `p030-rc2b` (424.6) a wash.
Forwarding the documented flag does not close the reference gap. vs
same-day `refg-rc2` (467.9 / 157.1) still -9.1 % / -8.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: idxpr-rc2 pin noise

`--sparse-indexer-topk-backend per_row` on `main-030-rc2`: 53.5 / 98.5 /
128.8 / **143.4**, sum **424.2**, spread 5.3 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.6 % / +1.6 %. vs `p030-rc2b` (424.6)
a wash. per_row topk is not the remaining gap. vs same-day `refg-rc2`
(467.9 / 157.1) still -9.3 % / -8.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: lintri-rc2 slightly worse

`LINEAR_BACKEND=triton` on `main-030-rc2`: 52.4 / 96.7 / 125.4 /
**141.1**, sum **415.6**, spread 5.0 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) -0.5 % / -0.1 %. vs `p030-rc2b` (424.6)
slightly worse. Triton linear is not the remaining gap. vs same-day
`refg-rc2` (467.9 / 157.1) still -11.2 % / -10.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: noshare-rc2 pin noise

`B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS=0` on `main-030-rc2`: 51.7 / 99.5 /
127.4 / **145.0**, sum **423.6**, spread 7.2 %. Gates pass. One pass
accept 49.0 % / tps 4.414 dips the floor. vs standing `proto2-030-rc2`
(417.5 / 141.2) +1.5 % / +2.7 %. vs `p030-rc2b` (424.6) a wash. Micro
share-input off is not the remaining gap. vs same-day `refg-rc2` (467.9 /
157.1) still -9.5 % / -7.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: notiny-rc2 pin noise

`B12X_W4A8_TINY_DECODE=0` on `main-030-rc2`: 55.2 / 98.9 / 127.9 /
**144.2**, sum **426.2**, spread 7.2 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.1 % / +2.1 %. vs `p030-rc2b` (424.6)
a wash. Tiny-decode off is not the remaining gap. vs same-day
`refg-rc2` (467.9 / 157.1) still -8.9 % / -8.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: nostream-rc2 pin noise

`B12X_INDEXER_STREAM_SCORER=0` on `main-030-rc2`: 55.5 / 99.4 / 129.1 /
**143.7**, sum **427.7**, spread 7.4 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.4 % / +1.8 %. vs `p030-rc2b` (424.6)
a wash. Stream scorer off is not the remaining gap. vs same-day
`refg-rc2` (467.9 / 157.1) still -8.6 % / -8.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: nopfx-rc2 pin noise

`--no-enable-prefix-caching` on `main-030-rc2`: 54.6 / 100.0 / 128.3 /
**140.2**, sum **423.1**, spread 6.6 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.3 % / -0.7 %. vs `p030-rc2b` (424.6)
a wash. Prefix cache off is not the remaining gap. vs same-day
`refg-rc2` (467.9 / 157.1) still -9.6 % / -10.8 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: memprof0-rc2 pin noise

`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` on `main-030-rc2`: 50.2 /
98.4 / 130.6 / **140.4**, sum **419.6**, spread 10.0 %. Gates pass. One
pass accept 45.4 % / tps 4.163 dips the floor. vs standing
`proto2-030-rc2` (417.5 / 141.2) +0.5 % / -0.6 %. vs `p030-rc2b` (424.6)
slightly worse. Extra KV without the profiler is not the remaining gap.
vs same-day `refg-rc2` (467.9 / 157.1) still -10.3 % / -10.6 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: now4sh-rc2 pin noise

`B12X_DYNAMIC_W4A8_SHARE_INPUT=0` on `main-030-rc2`: 52.1 / 99.1 / 130.9 /
**147.3**, sum **429.4**, spread 6.2 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.9 % / +4.3 %. vs `p030-rc2b` (424.6)
inside keep-spread. W4A8 share-input off is not the remaining gap. vs
same-day `refg-rc2` (467.9 / 157.1) still -8.2 % / -6.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: idxfi-rc2 pin noise

`--sparse-indexer-topk-backend flashinfer` on `main-030-rc2`: 50.7 /
97.6 / 127.8 / **146.2**, sum **422.3**, spread 11.0 %. Gates pass. One
pass accept 45.3 % / tps 4.163 dips the floor. First health window
expired at 450s (CUDA-graph capture); protocol ran against the live
container. vs standing `proto2-030-rc2` (417.5 / 141.2) +1.1 % / +3.5 %.
vs `p030-rc2b` (424.6) a wash. FlashInfer indexer top-k is not the
remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.7 % /
-6.9 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-19: noasync-rc2 pin noise

`--no-async-scheduling` on `main-030-rc2`: 52.3 / 97.6 / 126.0 /
**142.6**, sum **418.5**, spread 12.6 %. Gates pass. One pass accept
46.0 % / tps 4.231 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +0.2 % / +1.0 %. vs `p030-rc2b` (424.6) a wash. Async scheduling
off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1)
still -10.6 % / -9.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: dsl471-rc2 pin noise

`nvidia-cutlass-dsl[cu13]==4.7.1` on `main-030-rc2`: 55.9 / 98.5 / 128.3 /
**140.6**, sum **423.3**, spread 6.0 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.4 % / -0.4 %. vs `p030-rc2b` (424.6)
a wash. Do not pin 4.7.1. vs same-day `refg-rc2` (467.9 / 157.1) still
-9.5 % / -10.5 %. Not a keep.

Latest vLLM tag still `v0.30.0rc2`. Ours still OPEN behind
`pre-run-check`. Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: fusedq-rc2 KV floor

`B12X_DENSE_FUSED_QUANT=1` on `main-030-rc2` dies at KV floor: 9.47 GiB
available vs 9.48 GiB needed at util 0.8389 / max_model_len 65536. Do not
raise util. Fused-quant workspace is hungrier than standing. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: moetile32-rc2 pin noise

`B12X_DYNAMIC_TILE_MN=32x128` on `main-030-rc2`: 53.7 / 96.4 / 127.5 /
**144.4**, sum **422.0**, spread 8.2 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.1 % / +2.3 %. vs `p030-rc2b` (424.6)
a wash. Tile 32x128 is not the remaining gap. vs same-day `refg-rc2`
(467.9 / 157.1) still -9.8 % / -8.1 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: nogqa6-rc2 pin noise

`B12X_PAGED_GQA6_COMPACT_SYNC=0` on `main-030-rc2`: 54.6 / 94.2 / 132.4 /
**142.2**, sum **423.4**, spread 9.2 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.4 % / +0.7 %. vs `p030-rc2b` (424.6)
a wash. Compact-sync off is not the remaining gap. vs same-day
`refg-rc2` (467.9 / 157.1) still -9.5 % / -9.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: noprki-rc2 pin noise

`B12X_DENSE_PER_ROW_IN_KERNEL=0` on `main-030-rc2`: 55.5 / 99.3 / 126.5 /
**139.4**, sum **420.7**, spread 7.9 %. Gates pass. One pass accept 49.0 %
/ tps 4.414 dips the floor. vs standing `proto2-030-rc2` (417.5 / 141.2)
+0.8 % / -1.3 %. vs `p030-rc2b` (424.6) a wash. Host per-row GS is not
the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -10.1 % /
-11.3 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: noidxk-rc2 pin noise

`B12X_INDEXER_DIRECT_K=0` on `main-030-rc2`: 54.0 / 100.0 / 129.0 /
**141.7**, sum **424.7**, spread 7.7 %. Gates pass. One pass tps 4.420
dips the floor. vs standing `proto2-030-rc2` (417.5 / 141.2) +1.7 % /
+0.4 %. vs `p030-rc2b` (424.6) a wash. Direct-K off is not the remaining
gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.2 % / -9.8 %. Not a
keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: nogemv-rc2 pin noise

`B12X_DISABLE_BF16_GEMV=1` on `main-030-rc2`: 54.1 / 97.1 / 127.2 /
**144.6**, sum **423.0**, spread 7.6 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.3 % / +2.4 %. vs `p030-rc2b` (424.6)
a wash. GEMV off is not the remaining gap. vs same-day `refg-rc2`
(467.9 / 157.1) still -9.6 % / -8.0 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: refg-rc2b same-day anemll

`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` on 2026-09-20: 63.0 / 114.6 /
146.1 / **162.2**, sum **485.9**, spread 20.7 %. Gates pass. vs
yesterday `refg-rc2` (467.9 / 157.1) +3.8 % / +3.2 % — rig noise up.
Standing `proto2-030-rc2` (417.5 / 141.2) is now -14.1 % / -12.9 % vs
this same-day anemll. Best same-pin `moea16-rc2` (453.7 / 152.8) is
-6.6 % / -5.8 %. Not a candidate.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-20: noreuse-rc2 pin noise

`B12X_MICRO_REUSE_COMPILED=0` on `main-030-rc2`: 54.7 / 100.5 / 127.0 /
**143.8**, sum **426.0**, spread 5.1 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.0 % / +1.8 %. vs `p030-rc2b` (424.6)
a wash. Micro reuse off is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -12.3 % / -11.3 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: moetilem-rc2 KV floor

`B12X_MOE_TILE_MN=128x128` on `main-030-rc2` dies at KV floor: 9.44 GiB
available vs 9.48 GiB needed at util 0.8389 / max_model_len 65536. Do not
raise util. Micro 128x128 workspace is hungrier than standing. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7 vs same-day `refg-rc2b` 485.9.


## 2026-09-20: moetilem32-rc2 pin noise

`B12X_MOE_TILE_MN=32x128` on `main-030-rc2`: 51.9 / 98.4 / 130.4 /
**146.8**, sum **427.5**, spread 17.0 %. Gates pass. One pass accept
49.1 % / tps 4.414 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +2.4 % / +4.0 %. vs `p030-rc2b` (424.6) a wash. Micro 32x128 is
not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-12.0 % / -9.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: moetilem16-rc2 pin noise

`B12X_MOE_TILE_MN=16x128` on `main-030-rc2`: 55.3 / 97.5 / 128.7 /
**142.7**, sum **424.2**, spread 6.1 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.6 % / +1.1 %. vs `p030-rc2b` (424.6)
a wash. Micro 16x128 is not the remaining gap. vs same-day `refg-rc2b`
(485.9 / 162.2) still -12.7 % / -12.0 %. Not a keep. Micro tile ladder
closed (16/32 wash, 128 KV-floor).

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: nopagemax-rc2 pin noise

`B12X_MSA_DECODE_PAGEMAX=0` on `main-030-rc2`: 54.0 / 97.7 / 128.7 /
**141.8**, sum **422.2**, spread 11.5 %. Gates pass. One pass accept
48.8 % / tps 4.414 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +1.1 % / +0.4 %. vs `p030-rc2b` (424.6) a wash. Page-max off is
not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-13.1 % / -12.6 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: bs128-rc2 misaligned

`BLOCK_SIZE=128` on `main-030-rc2` dies at worker init:
`ValueError: Misaligned Tensor data on argument #6` (`cos_sin_cache` /
`k_cache`, expected 16-byte alignment). Manager 128 is not a legal
`nvfp4_ds_mla` layout on this pin. Keep 256. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN (labels: bug/DSv4; no
`pre-run-check` in this dump). Best same-pin `moea16-rc2` 453.7 vs
same-day `refg-rc2b` 485.9.


## 2026-09-20: micromax-rc2 pin noise

Native b12x has no static kernel (micro vs dynamic only; FlashInfer
static IMA'd on live MXFP4). One overlay: `_MICRO_MAX_TOKENS` 8→48 and
cutover 64→320 so protocol c6 stays micro. Confirmed overlay. 54.8 /
99.9 / 133.9 / **142.4**, sum **431.0**, spread 10.6 %. Gates pass. One
pass accept 48.8 % / tps 4.401 dips the floor. vs standing
`proto2-030-rc2` (417.5 / 141.2) +3.2 % / +0.8 %. vs `p030-rc2b` (424.6)
a wash. Extending micro past m=8 is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -11.3 % / -12.2 %. Overlay not kept.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: maxlen32k-rc2 pin noise

`MAX_MODEL_LEN=32768` on `main-030-rc2`: 54.3 / 97.7 / 128.6 / **144.3**,
sum **424.9**, spread 6.8 %. Gates pass. vs standing `proto2-030-rc2`
(417.5 / 141.2) +1.8 % / +2.2 %. vs `p030-rc2b` (424.6) a wash. Smaller
page table is not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -12.6 % / -11.0 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: profeager-rc2 attribution + wochunk8-rc2

Eager c1 profile (`B12X_PROFILE_*`, dump after 12 `execute_model`): last
live `tok=8` step gpu **150.6 ms**. FFN **80.3 ms (53 %)**, WO **57.4 ms
(38 %)**, all-reduce 0.93 ms avg × 87, indexer 1.4, MLA 0.9. Capture-mode
profile dies `CUDA invalid argument`. Overlay `print` is a Dynamo no-op
unless `B12X_DEBUG=1`. EngineCore drops unregistered `VLLM_PROFILE_*`.

`B12X_WO_QUANT_CHUNKS_PER_PROGRAM=8`: 54.5 / 96.6 / 129.9 / **143.1**,
sum **424.1**, spread 13.5 %. Gates pass. vs `p030-rc2b` (424.6) a wash.
Halving WO chunks is not the remaining gap. vs same-day `refg-rc2b`
(485.9 / 162.2) still -12.7 % / -11.8 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: wochunk32-rc2 pin noise

`B12X_WO_QUANT_CHUNKS_PER_PROGRAM=32` on `main-030-rc2`: 53.1 / 99.4 /
129.0 / **143.2**, sum **424.7**, spread 10.2 %. Gates pass. One pass
accept 48.6 % / tps 4.376 dips the floor. vs standing `proto2-030-rc2`
(417.5 / 141.2) +1.7 % / +1.4 %. vs `p030-rc2b` (424.6) a wash. WO
chunks 8 and 32 are both pin noise. vs same-day `refg-rc2b` (485.9 /
162.2) still -12.6 % / -11.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: wochunk4-rc2 pin noise

`B12X_WO_QUANT_CHUNKS_PER_PROGRAM=4` on `main-030-rc2`: 55.3 / 101.5 /
130.3 / **145.8**, sum **432.9**, spread 5.9 %. Gates pass. One pass
accept 48.9 % / tps 4.406 dips the floor. vs standing `proto2-030-rc2`
(417.5 / 141.2) +3.7 % / +3.3 %. vs `p030-rc2b` (424.6) a wash. WO
chunks 4/8/32 are all pin noise. vs same-day `refg-rc2b` (485.9 / 162.2)
still -10.9 % / -10.1 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: wochunk2-rc2 pin noise

`B12X_WO_QUANT_CHUNKS_PER_PROGRAM=2` on `main-030-rc2`: 56.8 / 94.8 /
130.2 / **147.7**, sum **429.5**, spread 13.9 %. Gates pass. One pass
accept 47.5 % / tps 4.303 dips the floor. vs standing `proto2-030-rc2`
(417.5 / 141.2) +2.9 % / +4.6 %. vs `p030-rc2b` (424.6) a wash. WO
chunks 2/4/8/32 are all pin noise. vs same-day `refg-rc2b` (485.9 /
162.2) still -11.6 % / -8.9 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: wochunk1-rc2 negative

`B12X_WO_QUANT_CHUNKS_PER_PROGRAM=1` on `main-030-rc2`: 53.7 / 99.9 /
127.7 / **136.7**, sum **418.0**, spread 6.1 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +0.1 % / -3.2 %. vs `p030-rc2b` (424.6)
worse at c6. WO chunks 1 is a cost. Ladder closed (1 cost, 2/4/8/32
wash). vs same-day `refg-rc2b` (485.9 / 162.2) still -14.0 % / -15.7 %.
Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: nocar-rc2 pin noise

`--disable-custom-all-reduce` on `main-030-rc2`: 54.1 / 99.6 / 129.7 /
**140.6**, sum **424.0**, spread 12.0 %. Gates pass. One pass accept
47.4 % / tps 4.303 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +1.6 % / -0.4 %. vs `p030-rc2b` (424.6) a wash, slightly worse at
c6. Custom AR off is not the remaining gap. vs same-day `refg-rc2b`
(485.9 / 162.2) still -12.7 % / -13.3 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: loadsaf-rc2 pin noise

`--load-format safetensors` on `main-030-rc2`: 55.3 / 96.9 / 127.0 /
**144.9**, sum **424.1**, spread 7.9 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.6 % / +2.6 %. vs `p030-rc2b` (424.6)
a wash. Loader format is not the remaining gap. vs same-day `refg-rc2b`
(485.9 / 162.2) still -12.7 % / -10.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: fiswizzle-rc2 IMA

FlashInfer MXFP4 with `swizzle_block_scale` then
`convert_sf_to_mma_layout(..., sf_vec_size=32)` on `main-030-rc2`.
Confirmed overlay log `swizzle-then-mma k32`. Dies at `profile_run`
with `CUDA_ERROR_ILLEGAL_ADDRESS` (700). Same live-weight IMA as
`fistat`/`fifunc`. Name-only `mxfp4` and swizzle-before-convert are
both closed. Stock overlay stays stock. vs same-day `refg-rc2b`
(485.9 / 162.2) still -6.6 % / -5.8 % at best same-pin `moea16-rc2`
453.7 / 152.8.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: moecut32-rc2 pin noise

`B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=32` on `main-030-rc2`: 55.4 / 99.6 /
129.9 / **141.0**, sum **425.9**, spread 5.4 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.0 % / -0.1 %. vs `p030-rc2b` (424.6)
a wash. Sending c1 (48 pairs) to dynamic is not the remaining gap. vs
same-day `refg-rc2b` (485.9 / 162.2) still -12.3 % / -13.1 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: cuteopt2-rc2 KV floor

`B12X_DIRECT_CUTE_OPTIONS=--opt-level=2` on `main-030-rc2`. Dies at KV
floor: 9.46 GiB available vs 9.48 GiB needed. Micro-direct OptLevel 2
is hungrier than standing. Do not raise util. vs same-day `refg-rc2b`
(485.9 / 162.2) still -6.6 % / -5.8 % at best same-pin `moea16-rc2`
453.7 / 152.8.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: nccpll-rc2 negative

`NCCL_PROTO=LL` on `main-030-rc2`: 51.1 / 86.8 / 101.6 / **112.8**, sum
**352.3**, spread 8.8 %. Gates pass. One pass accept 46.8 % / tps 4.267
dips the floor. vs standing `proto2-030-rc2` (417.5 / 141.2) -15.6 % /
-20.1 %. vs `p030-rc2b` (424.6) a large cost. NCCL LL is not the
remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -27.5 % /
-30.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclch1-rc2 pin noise

`NCCL_MAX_NCHANNELS=1` on `main-030-rc2`: 53.5 / 100.1 / 127.4 /
**142.9**, sum **423.9**, spread 7.9 %. Gates pass. One pass accept
48.8 % / tps 4.391 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +1.5 % / +1.2 %. vs `p030-rc2b` (424.6) a wash. One NCCL channel
is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-12.8 % / -11.9 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: nccllls-rc2 pin noise

`NCCL_PROTO=LL,Simple` on `main-030-rc2`: 54.2 / 89.0 / 131.3 /
**146.4**, sum **420.9**, spread 6.4 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +0.8 % / +3.7 %. vs `p030-rc2b` (424.6)
a wash. Optional LL is not exclusive LL (`nccpll-rc2` 352.3) and is not
the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-13.4 % / -9.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclring-rc2 negative

`NCCL_ALGO=Ring` on `main-030-rc2`: 55.1 / 100.2 / 128.0 / **138.2**,
sum **421.5**, spread 12.9 %. Gates pass. vs standing `proto2-030-rc2`
(417.5 / 141.2) +1.0 % / -2.1 %. vs `p030-rc2b` (424.6) worse at c6.
Forced Ring is not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -13.3 % / -14.8 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclnt64-rc2 pin noise

`NCCL_NTHREADS=64` on `main-030-rc2`: 54.8 / 97.3 / 125.7 / **147.8**,
sum **425.6**, spread 6.6 %. Gates pass. vs standing `proto2-030-rc2`
(417.5 / 141.2) +1.9 % / +4.7 %. vs `p030-rc2b` (424.6) a wash. NCCL
thread count is not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -12.4 % / -8.9 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclcu0-rc2 pin noise

`NCCL_CUMEM_ENABLE=0` on `main-030-rc2`: 53.6 / 96.2 / 126.1 /
**145.0**, sum **420.9**, spread 5.6 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +0.8 % / +2.7 %. vs `p030-rc2b` (424.6)
a wash. NCCL cuMem off is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -13.4 % / -10.6 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclbuf1m-rc2 pin noise

`NCCL_BUFFSIZE=1048576` on `main-030-rc2`: 52.8 / 101.3 / 134.9 /
**143.7**, sum **432.7**, spread 9.3 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +3.6 % / +1.8 %. vs `p030-rc2b` (424.6)
a wash. NCCL buffer 1 MiB is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -10.9 % / -11.4 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclgrp-rc2 pin noise

`NCCL_LAUNCH_MODE=GROUP` on `main-030-rc2`: 54.4 / 95.3 / 126.9 /
**141.6**, sum **418.2**, spread 8.8 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +0.2 % / +0.3 %. vs `p030-rc2b` (424.6)
a wash. GROUP launch is not the remaining gap. vs same-day `refg-rc2b`
(485.9 / 162.2) still -13.9 % / -12.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: fiw31-rc2 CUDA error

FlashInfer MXFP4 with in-place w31 [gate;up] to [up;gate] flip, then
`swizzle_block_scale` and `convert_sf_to_mma_layout(..., sf_vec_size=32)`
on `main-030-rc2`. Confirmed overlay log
`w31-flip-then-swizzle-mma k32`. Dies at `profile_run` with
`CUBLAS_STATUS_INTERNAL_ERROR` / `CUDA_ERROR_ILLEGAL_ADDRESS`. Same
live-weight failure class as `fistat`/`fifunc`/`fiswizzle-rc2`. w31
flip plus swizzle is not the remaining gap. Stock overlay stays stock.
vs same-day `refg-rc2b` (485.9 / 162.2) still -6.6 % / -5.8 % at best
same-pin `moea16-rc2` 453.7 / 152.8.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclp2p0-rc2 pin noise

`NCCL_P2P_DISABLE=1` on `main-030-rc2`: 55.8 / 98.4 / 137.7 /
**148.7**, sum **440.6**, spread 13.8 %. 9x8 gate passes. Pass 3
france gate is garbled. One pass accept 48.2 % / tps 4.376 dips the
floor. vs standing `proto2-030-rc2` (417.5 / 141.2) +5.5 % / +5.3 %,
inside keep-spread. vs `p030-rc2b` (424.6) a wash. P2P off is not the
remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -9.3 % /
-8.3 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclqps4-rc2 pin noise

`NCCL_IB_QPS_PER_CONNECTION=4` on `main-030-rc2`: 55.6 / 101.2 / 135.5 /
**146.4**, sum **438.7**, spread 8.1 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +5.1 % / +3.7 %. vs `p030-rc2b` (424.6)
a wash. Four IB QPs is not the remaining gap. vs same-day `refg-rc2b`
(485.9 / 162.2) still -9.7 % / -9.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: cgsizes-rc2 pin noise

`CUDAGRAPH_CAPTURE_SIZES=[8,24,40,48]` on `main-030-rc2`: 53.8 / 99.2 /
129.5 / **145.6**, sum **428.1**, spread 9.3 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.5 % / +3.1 %. vs `p030-rc2b` (424.6)
a wash. Explicit capture sizes are not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -11.9 % / -10.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: cgcopy-rc2 pin noise

`CUDAGRAPH_COPY_INPUTS=true` on `main-030-rc2`: 55.0 / 102.6 / 128.7 /
**142.3**, sum **428.6**, spread 7.8 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.7 % / +0.8 %. vs `p030-rc2b` (424.6)
a wash. Copying graph inputs is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -11.8 % / -12.3 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: moewarm-rc2 pin noise

`B12X_MOE_WARM_MS=8,24,40,48` on `main-030-rc2`: 53.8 / 98.9 / 131.1 /
**148.9**, sum **432.7**, spread 11.3 %. Gates pass. One pass accept
47.5 % / tps 4.303 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +3.6 % / +5.5 %. vs `p030-rc2b` (424.6) a wash. Explicit MoE
warm sizes are not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -10.9 % / -8.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclxnic-rc2 pin noise

`NCCL_CROSSNIC=1` on `main-030-rc2`: 56.9 / 101.9 / 127.0 / **145.3**,
sum **431.1**, spread 7.9 %. Gates pass. vs standing `proto2-030-rc2`
(417.5 / 141.2) +3.3 % / +2.9 %. vs `p030-rc2b` (424.6) a wash. CrossNIC
is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-11.3 % / -10.4 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclhca-rc2 pin noise

`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` on `main-030-rc2`: 53.2 / 101.1 /
130.5 / **141.7**, sum **426.5**, spread 11.3 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +2.2 % / +0.4 %. vs `p030-rc2b` (424.6)
a wash. Naming both HCAs is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -12.2 % / -12.6 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclpxn0-rc2 pin noise

`NCCL_PXN_DISABLE=1` on `main-030-rc2`: 54.2 / 101.2 / 127.7 /
**144.8**, sum **427.9**, spread 7.3 %. 9x8 gate passes. Pass 3 france
gate is garbled. One pass accept 48.1 % / tps 4.339 dips the floor. vs
standing `proto2-030-rc2` (417.5 / 141.2) +2.5 % / +2.5 %. vs
`p030-rc2b` (424.6) a wash. PXN off is not the remaining gap. vs
same-day `refg-rc2b` (485.9 / 162.2) still -11.9 % / -10.7 %. Not a
keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclchk0-rc2 pin noise

`NCCL_CHECKS_DISABLE=1` on `main-030-rc2`: 56.1 / 92.3 / 126.5 /
**145.1**, sum **420.0**, spread 18.3 %. Gates pass. One pass accept
46.9 % / tps 4.267 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +0.6 % / +2.8 %. vs `p030-rc2b` (424.6) a wash. NCCL checks off
is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-13.6 % / -10.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclshm0-rc2 pin noise

`NCCL_SHM_DISABLE=1` on `main-030-rc2`: 52.9 / 95.7 / 125.4 /
**142.5**, sum **416.5**, spread 11.4 %. Gates pass. One pass accept
48.1 % / tps 4.339 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) -0.2 % / +0.9 %. vs `p030-rc2b` (424.6) a wash. SHM off is not
the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-14.3 % / -12.1 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: nohma-rc2 KV layout

`--disable-hybrid-kv-cache-manager` on `main-030-rc2`. Confirmed
`disable_hybrid_kv_cache_manager': True`. Dies at worker init:
`ValueError: The resolved KV cache layout (BLHNC) does not store
blocks as dense, unpadded pages (block stride 83889984 != page
149504)`. Standing auto-HMA is load-bearing for `nvfp4_ds_mla` block
256 on SM120. Explicit disable is not the remaining gap.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: kvlbnhc-rc2 KV layout

`VLLM_KV_CACHE_LAYOUT=LBNHC` on `main-030-rc2`. Dies at EngineCore
init: `ValueError: VLLM_KV_CACHE_LAYOUT=LBNHC does not satisfy every
supported set; valid layouts: ['BLHNC', 'BLNHC']`. LBNHC is not legal
for this `nvfp4_ds_mla` + DSV4 indexer pair. Standing auto BLHNC stays.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: kvblnhc-rc2 pin noise

`VLLM_KV_CACHE_LAYOUT=BLNHC` on `main-030-rc2`: 56.8 / 97.3 / 132.1 /
**147.0**, sum **433.2**, spread 5.1 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +3.8 % / +4.1 %. vs `p030-rc2b` (424.6)
a wash. BLNHC is not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -10.8 % / -9.4 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: cgwarm3-rc2 pin noise

`CUDAGRAPH_NUM_OF_WARMUPS=3` on `main-030-rc2`: 53.6 / 99.6 / 129.3 /
**146.8**, sum **429.3**, spread 9.3 %. Gates pass. One pass accept
46.4 % / tps 4.197 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +2.8 % / +4.0 %. vs `p030-rc2b` (424.6) a wash. Extra graph
warmups are not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -11.6 % / -9.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: igp-rc2 pin noise

`USE_INDUCTOR_GRAPH_PARTITION=true` on `main-030-rc2`: 53.1 / 95.8 /
130.7 / **143.2**, sum **422.8**, spread 4.9 %. 9x8 gate passes. Pass 1
france gate is garbled. vs standing `proto2-030-rc2` (417.5 / 141.2)
+1.3 % / +1.4 %. vs `p030-rc2b` (424.6) a wash. Inductor graph
partition is not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -13.0 % / -11.7 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclsock4-rc2 negative

`NCCL_SOCKET_NTHREADS=4` on `main-030-rc2`: 53.4 / 96.4 / 128.2 /
**139.4**, sum **417.4**, spread 7.1 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) -0.0 % / -1.3 %. Worse at c6. vs
`p030-rc2b` (424.6) a wash-to-negative. Socket helper threads are not
the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-14.1 % / -14.1 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: moetile64-rc2 pin noise

`B12X_DYNAMIC_TILE_MN=64x128` on `main-030-rc2`: 55.4 / 102.0 / 131.0 /
**142.2**, sum **430.6**, spread 12.6 %. Gates pass. One pass accept
47.3 % / tps 4.303 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +3.1 % / +0.7 %. vs `p030-rc2b` (424.6) a wash. Forced M64 is
not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-11.4 % / -12.3 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclnsock4-rc2 pin noise

`NCCL_NSOCKS_PERTHREAD=4` on `main-030-rc2`: 53.0 / 99.0 / 128.1 /
**142.0**, sum **422.1**, spread 12.8 %. Gates pass. One pass accept
49.0 % / tps 4.414 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +1.1 % / +0.6 %. vs `p030-rc2b` (424.6) a wash. Extra sockets
per thread are not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -13.1 % / -12.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclto22-rc2 pin noise

`NCCL_IB_TIMEOUT=22` on `main-030-rc2`: 55.1 / 97.6 / 131.2 /
**145.6**, sum **429.5**, spread 11.6 %. 9x8 gate passes. Pass 1 france
gate is garbled. One pass accept 47.7 % / tps 4.339 dips the floor. vs
standing `proto2-030-rc2` (417.5 / 141.2) +2.9 % / +3.1 %. vs
`p030-rc2b` (424.6) a wash. IB timeout 22 is not the remaining gap. vs
same-day `refg-rc2b` (485.9 / 162.2) still -11.6 % / -10.2 %. Not a
keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ensp-rc2 SP+DSpark

Forced `pass_config.enable_sp=true` with `sp_min_token_num=1` on
`main-030-rc2`. Confirmed in compilation_config. Dies at VllmConfig:
`Model Runner V1 does not support: dspark speculative decoding`.
Forced SP is incompatible with DSpark on this pin.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: fnormq-rc2 hang

`PASS_CONFIG={"fuse_norm_quant":true}` on `main-030-rc2`. Confirmed
`fuse_norm_quant': True` and `Enabled custom fusions: norm_quant,
act_quant`. Hangs at EngineCore init, never healthy. Forced
norm+quant fusion is not the remaining gap.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclmin2-rc2 negative

`NCCL_MIN_NCHANNELS=2` on `main-030-rc2`: 51.5 / 96.6 / 127.0 /
**140.3**, sum **415.4**, spread 8.4 %. Gates pass. One pass accept
47.5 % / tps 4.303 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) -0.5 % / -0.6 %. Worse at c6. vs `p030-rc2b` (424.6) a
wash-to-negative. MIN channels 2 is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -14.5 % / -13.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclpeer2-rc2 negative

`NCCL_NCHANNELS_PER_NET_PEER=2` on `main-030-rc2`: 53.3 / 96.6 / 129.8 /
**140.3**, sum **420.0**, spread 4.9 %. Gates pass. One pass accept
48.8 % / tps 4.401 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +0.6 % / -0.6 %. Worse at c6. vs `p030-rc2b` (424.6) a
wash-to-negative. Per-peer channels 2 is not the remaining gap. vs
same-day `refg-rc2b` (485.9 / 162.2) still -13.6 % / -13.5 %. Not a
keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: nccltc106-rc2 pin noise

`NCCL_IB_TC=106` on `main-030-rc2`: 52.0 / 102.8 / 128.0 / **141.2**,
sum **424.0**, spread 6.9 %. Gates pass. One pass accept 48.2 % / tps
4.339 dips the floor. vs standing `proto2-030-rc2` (417.5 / 141.2)
+1.6 % / +0.0 %. vs `p030-rc2b` (424.6) a wash. IB TC 106 is not the
remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.7 % /
-12.9 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclsl0-rc2 pin noise

`NCCL_IB_SL=0` on `main-030-rc2`: 54.8 / 97.7 / 130.8 / **142.6**,
sum **425.9**, spread 10.8 %. Gates pass. One pass accept 47.9 % / tps
4.339 dips the floor. vs standing `proto2-030-rc2` (417.5 / 141.2)
+2.0 % / +1.0 %. vs `p030-rc2b` (424.6) a wash. IB SL 0 is not the
remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.3 % /
-12.1 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: factq-rc2 hang

`PASS_CONFIG={"fuse_act_quant":true}` on `main-030-rc2`. Confirmed
`fuse_act_quant': True` and `Enabled custom fusions: norm_quant,
act_quant`. Hangs at EngineCore init, never healthy. Same class as
`fnormq-rc2`: either fusion flag enables both and hangs. Forced
act+quant fusion is not the remaining gap.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclretry1-rc2 pin noise

`NCCL_IB_RETRY_CNT=1` on `main-030-rc2`: 53.1 / 101.1 / 125.1 /
**142.5**, sum **421.8**, spread 9.8 %. Gates pass. One pass accept
48.8 % / tps 4.376 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +1.0 % / +0.9 %. vs `p030-rc2b` (424.6) a wash. IB retry 1 is
not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-13.2 % / -12.1 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: ncclar1-rc2 pin noise

`NCCL_IB_AR_ALGORITHM=1` on `main-030-rc2`: 56.4 / 99.1 / 131.5 /
**141.8**, sum **428.8**, spread 10.1 %. Gates pass. One pass accept
49.1 % / tps 4.420 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +2.7 % / +0.4 %. vs `p030-rc2b` (424.6) a wash. IB adaptive
routing is not the remaining gap. vs same-day `refg-rc2b` (485.9 /
162.2) still -11.8 % / -12.6 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: now4mat-rc2 negative

`B12X_DYNAMIC_W4A8_MATERIALIZED=0` on `main-030-rc2`: 53.8 / 99.1 /
127.6 / **139.3**, sum **419.8**, spread 15.7 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +0.6 % / -1.3 %. Worse at c6. vs
`p030-rc2b` (424.6) a wash-to-negative. Materialized off is not the
remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.6 % /
-14.1 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-20: w4scr128-rc2 pin noise

`B12X_W4A8_CONVERT_SCRATCH_MB=128` on `main-030-rc2`: 53.0 / 98.3 /
129.2 / **143.5**, sum **424.0**, spread 7.4 %. Gates pass. vs standing
`proto2-030-rc2` (417.5 / 141.2) +1.6 % / +1.6 %. vs `p030-rc2b` (424.6)
a wash. Convert scratch 128 is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -12.7 % / -11.5 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-21: ncclcnet0-rc2 pin noise

`NCCL_COLLNET_ENABLE=0` on `main-030-rc2`: 52.9 / 99.6 / 126.1 /
**143.0**, sum **421.6**, spread 7.8 %. Gates pass. One pass accept
47.3 % / tps 4.267 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +1.0 % / +1.3 %. vs `p030-rc2b` (424.6) a wash. CollNet off is
not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-13.2 % / -11.8 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-21: ncclplug0-rc2 pin noise

`NCCL_NET_PLUGIN=none` on `main-030-rc2`: 53.7 / 101.3 / 132.0 /
**142.4**, sum **429.4**, spread 10.0 %. Gates pass. One pass accept
45.6 % / tps 4.197 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +2.9 % / +0.8 %. vs `p030-rc2b` (424.6) a wash. NET plugin none
is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still
-11.6 % / -12.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-21: ncclnetib-rc2 negative

`NCCL_NET=IB` on `main-030-rc2`: 54.4 / 96.1 / 127.1 / **139.1**,
sum **416.7**, spread 12.3 %. Gates pass. vs standing `proto2-030-rc2`
(417.5 / 141.2) -0.2 % / -1.5 %. Worse at c6. vs `p030-rc2b` (424.6) a
wash-to-negative. Forced NET=IB is not the remaining gap. vs same-day
`refg-rc2b` (485.9 / 162.2) still -14.2 % / -14.2 %. Not a keep.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-21: fattnq-rc2 hang

`PASS_CONFIG={"fuse_attn_quant":true}` on `main-030-rc2`. Confirmed
`fuse_attn_quant': True` and `Enabled custom fusions: norm_quant,
act_quant, attn_quant, rope_kvcache_cat_mla`. Hangs at EngineCore
init, never healthy. Same class as `fnormq-rc2`/`factq-rc2`: any
forced fusion flag hangs this pin. Forced attn+quant fusion is not
the remaining gap.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-21: nnoop-rc2 hang

`PASS_CONFIG={"eliminate_noops":false}` on `main-030-rc2`. Confirmed
`eliminate_noops': False` and `Enabled custom fusions: norm_quant,
act_quant`. Hangs at EngineCore init, never healthy. Same class as
`fnormq`/`factq`/`fattnq`: any non-empty `pass_config` enables custom
fusions and hangs. Standing empty `pass_config` is load-bearing.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-21: cgpiece-rc2 positive not a keep

`CUDAGRAPH_MODE=PIECEWISE` on `main-030-rc2`: 54.6 / 106.4 / 136.1 /
**150.4**, sum **447.5**, spread 17.8 %. Gates pass. One pass accept
47.6 % / tps 4.339 dips the floor. vs standing `proto2-030-rc2` (417.5 /
141.2) +7.2 % / +6.5 %, inside keep-spread 14.0 %. vs `p030-rc2b`
(424.6) +5.4 % / +4.3 %. Second-best same-pin sum after `moea16-rc2`
453.7. vs same-day `refg-rc2b` (485.9 / 162.2) still -7.9 % / -7.3 %.
Not a keep. FULL_AND_PIECEWISE stays the standing default.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-21: noaot-rc2 pin noise

`VLLM_USE_AOT_COMPILE=0` on `main-030-rc2`: 54.3 / 95.8 / 126.3 /
**141.1**, sum **417.5**, spread 9.4 %. Gates pass; accept 49.3-56.9 %
and tps 4.439-4.971 hold the floor. vs standing `proto2-030-rc2`
(417.5 / 141.2) +0.0 % / -0.1 %. Resolved config still
`CompilationMode.NONE`, so the arm only measured losing the AOT persist
cache. vs `p030-rc2b` (424.6) a wash. vs same-day `refg-rc2b`
(485.9 / 162.2) still -14.1 % / -13.0 %. Not a keep. AOT stays on.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.
Best same-pin sum remains `moea16-rc2` 453.7.


## 2026-09-21: profile pass (c6 phase accounting, NCCL transport)

Goal unmet, so the standing pin was profiled end to end. Three instruments
were tried; only the overlay's own region marks produced usable device time.

**What the instruments cost.** `nsys --attach-pid` cannot enable CUPTI in a
process not launched under it, so it wrote no report (`nsysc6-rc2`). vLLM's
in-process torch profiler (`PROFILE_ENABLE=1`) traced the API process only:
188k `python_function` events, zero CUDA activity, so no kernel table
(`torchc6`). The overlay's `b12x_profile_region` marks are therefore the only
device-time source, and they only collect on a run without CUDA graphs.

**c1 region split** (`profeager-rc2`, last clean step `execute_model tok=8`,
gpu 150.6 ms, layers sum 150.2 ms over 43):

| region | n | gpu_sum | avg | wall_sum |
|---|---|---|---|---|
| ffn | 43 | 80.3 ms | 1.87 | 57.7 |
| attn (contains wo) | 43 | 67.8 | 1.58 | 67.9 |
| wo | 43 | 57.4 | 1.34 | 14.8 |
| wo_b12x | 43 | 57.3 | 1.33 | 14.2 |
| allreduce | 87 | 81.3 | 0.93 | 5.4 |
| indexer | 21 | 1.4 | 0.06 | 7.5 |
| mla | 43 | 0.9 | 0.02 | 22.9 |

Two named hot spots: **ffn 53 %** and **wo 38 %** (o_proj, our own overlay).

**c6 step time.** 3072 tokens over 6 streams in 22.86 s with 675 draft slots
(112.5 engine steps) is a **203 ms step**. The target forward's own device
time at c6 could **not** be measured, so the earlier version of this entry
claimed ~81 ms/step (40 %) was "host-bound". **That claim is retracted.** It
came from comparing a profiler-window target time (~100 ms) against a
production step time (203 ms), which mixes two regimes, and the profiler run
behind it was also misconfigured (see below). The honest statement is: a
203 ms step at c6 is established, the split inside it is not.

Three defects found and fixed in the instrumentation, so nobody repeats them:

- The ad-hoc profile launchers called `scripts/05-serve.sh` without
  `NUM_SPECULATIVE_TOKENS`/`MAX_CUDAGRAPH_CAPTURE_SIZE`, so they ran the **pin**
  defaults (**k=5, capture 36**) while every measured arm runs k=7 / capture
  48 via `harness/run-arm.sh`. At capture 36 the 48-token target forward is
  not even captured. Launchers must pass both explicitly.
- The profiler env cannot be passed as `VLLM_PROFILE_*`: it reaches the
  container but the GPU worker that executes `execute_model` does not carry it
  (the process that does is a different one), so the `b12x_profile_target_step`
  decorator returns the bare function and nothing prints. The `B12X_PROFILE_*`
  alias plus a bind-mounted patched `sm12x_b12x_kernels.py` binds the mount and
  the module correctly (`grep -c B12X_PROFILE` = 5 in-container) and the
  decorator is applied to `vllm/v1/worker/gpu/model_runner.py`, but the worker
  still comes up without the alias, so both the step lines and the region
  marks (which need `_PROFILING_STEP` set by that same wrapper) stay dark.
- Net effect: the region table we do have (`profeager-rc2`) was produced when
  the route happened to arm, and it is an eager c1 table. There is currently
  no working route to a graph-regime c6 region split.

**NCCL transport (`ncclinfo-rc2`).** Healthy and already correct: RoCE on
both HCAs (`rocep1s0f1`, `roceP2p1s0f1`) alternating per channel, 64 coll
channels, 0 NVLS, `CC Off`, no CollNet, `RMA_IB_PROXY`. The 0.93 ms average
all-reduce is contention, not bandwidth: `gpu_min` is 0.08 ms. Twenty NCCL
env arms were measured blind and were all pin noise or worse; none of them
was addressing a real misconfiguration.

**Fix landed.** The o_proj overlay is our own code, is named by the profile at
38 % of the step, and `nowo-rc2` proved it is a net loss (sum 442.2 vs 417.5,
better at every level). `configs/pin.main-029.env` now defaults
`VLLM_USE_B12X_WO_PROJECTION` to 0, so the einsum path is the served default.
Re-measured as `wooff-rc2`.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: wooff re-measure + A16 does not replicate

`wooff-rc2` (WO overlay off as the pin default, the fix landed from the
profile): 56.8 / 96.5 / 129.4 / **144.2**, sum **426.9**, spread **7.5 %**,
gates pass, acceptance 49.6-57.0 %, tokens per step 4.452-4.971. Against
standing `proto2-030-rc2` (417.5 / 141.2) +2.3 % / +2.1 %; against
`p030-rc2b` (424.6 / 144.2) a wash. It is 3.5 % below the earlier `nowo-rc2`
(442.2), so part of that single-arm gain was noise. Both arms favour the
overlay off, so the default stays off. Not a keep.

`wooff-a16-rc2` (adds `VLLM_B12X_MOE_FP4_FORCE_A16=1`, the MoE lever that
measured best alone at 453.7): 54.9 / 102.1 / 130.8 / **139.6**, sum
**427.4**, spread 5.9 %. A wash against `wooff-rc2` (+0.1 %) and 5.8 % / 8.6 %
**worse** than the same knob measured by itself. The `moea16-rc2` result did
not replicate.

Conclusion for this pin: the surviving one-variable knobs are all inside the
rig's own ~14 % median swing, which is why no arm clears the keep-rule. The
profile names two real hot spots (MoE 53 %, o_proj 38 % at c1), and those
need a structural change, not another env knob. The c6 split inside the
203 ms step is still unmeasured.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: k=6 adopted, 5-pass re-baseline

The c1 median swing is ~8 % even at five passes, which is larger than any
single-knob effect found so far, so the keep-rule cannot be cleared by a 3-pass
arm. Both sides of the comparison were therefore re-measured at five passes.

| arm | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `wooff5-rc2` (k=7 default) | 54.2 | 99.5 | 131.1 | 144.6 | **429.4** | 8.5 % |
| `wooff-k6-5-rc2` (k=6) | 56.3 | 103.7 | 132.9 | 152.3 | **445.2** | 7.6 % |

Both pass both gates in all five passes. k=6 raises acceptance from
49.8-56.9 % to **57.5-67.5 %** and holds the tokens-per-step floor (min 4.437
against 4.427). Effect: **+3.7 %** on the sum, **+5.3 %** at c6. Against the
contract standing arm `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) it is
+6.6 % / +7.9 %. It does not clear the keep bar (max(8.5 %, 7.6 %) = 8.5 %),
so it is a served-default improvement, not a keep.

Adopted anyway because it is positive in three independent measurements
(`k6-rc2` +6.7 %, `wooff-k6-rc2` +5.0 %, `wooff-k6-5-rc2` +3.7 %), it improves
acceptance rather than trading against it, and its mechanism is the one the
profile supports (one fewer draft pass per engine step). `NUM_SPECULATIVE_TOKENS`
now defaults to 6 in both `configs/pin.main-029.env` and `harness/run-arm.sh`,
which also gains `PASSES` and `MAX_CUDAGRAPH_CAPTURE_SIZE` overrides.

Best same-pin vs same-day anemll `refg-rc2b` (485.9 / 162.2) is now
**-8.4 % / -6.1 %**, up from -12.0 % / -13.9 %.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: humming MoE adopted — gap to anemll down to ~1 %

With WO overlay off and k=6 landed, the remaining single variables were tested
at five passes, because the c1 median swing (~8-12 %) is larger than any
effect found so far.

| arm | c1 | c3 | c5 | c6 | sum | worst spread | verdict |
|---|---|---|---|---|---|---|---|
| `wooff-k6-5-rc2` (standing) | 56.3 | 103.7 | 132.9 | 152.3 | 445.2 | 7.6 % | baseline |
| `cgpiece-k6-rc2` | 55.1 | 96.1 | 129.9 | 139.5 | 420.6 | 8.9 % | **-5.5 %** loss |
| `hum-k6-rc2b` | 58.5 | 111.6 | 146.8 | **160.6** | **477.5** | 11.8 % | **+7.3 %** win |

`CUDAGRAPH_MODE=PIECEWISE` is a clear loss at five passes: `cgpiece-rc2`'s
earlier +7.2 % came with a 17.8 % spread and did not replicate.
`FULL_AND_PIECEWISE` stands, consistent with `cgfull` and `proto2-dg-cgfull`.

`MOE_BACKEND=humming` passes both gates in all five passes, holds acceptance
(53.1-64.3 %), and is the best same-pin result on record. Adopted as the
served default in `configs/pin.main-029.env`.

**Channel trap, recorded so it is not repeated.** `MOE_BACKEND` is read by
`scripts/05-serve.sh` on the launcher to build `--moe-backend`, so passing it
as container env (`SERVE_EXTRA_ENV=MOE_BACKEND=humming`) is too late: the flag
still said `b12x`. The invalid arm `hum-k6-rc2` measured the unchanged default
instead (448.9, a third sample of the k=6 / WO-off config). Launcher-side
variables must go through the assignment prefix
(`run-arm.sh <tag> "MOE_BACKEND=humming"`); `SERVE_EXTRA_ENV` is for variables
read inside the container, which is why the NCCL and `VLLM_B12X_*` arms were
correct and this one was not.

**Where the gap stands.** Against the same-day reference `refg-rc2b`
(485.9 / 162.2) the best arm is now **-1.7 % / -1.0 %**, from -12.0 % / -13.9 %
at the start of round 103. Against the contract standing arm
`proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) it is **+14.4 % / +13.7 %**:
the sum clears the 14.0 % bar, c6 misses it by 0.3 points. Not yet a keep, and
not yet a win over anemll.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: k=5 degenerate, A16 closed under humming

Two single-variable arms on the humming default (WO off, k=6), five passes.

| arm | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `hum-k6-rc2b` (standing) | 58.5 | 111.6 | 146.8 | 160.6 | **477.5** | 11.8 % |
| `hum-k5-rc2` (k=5) | 29.7 | 76.5 | 75.4 | 93.5 | **275.1** | 83.3 % |
| `hum-a16-rc2` (A16) | 57.8 | 102.9 | 134.9 | 149.5 | **445.1** | 7.6 % |

**k=5 is below a cliff**, not a gentle continuation of the ladder: -42.4 % on the
sum, spreads up to 83.3 %, tokens per step 4.071-6.250, and the final `/metrics`
snapshot refused. All five passes degrade, so it is not one flaky pass. k=6
stays the default.

**A16 is closed.** Forcing W4A16 on top of humming costs **-6.8 % / -6.9 %**
with no quality dip at all (acceptance 58.2-65.5 %, tokens per step 4.452-4.923,
both gates 5/5). Under b12x the same env gave +8.7 % once and then +0.1 % on the
WO-off default; under humming it is a clear loss. The lever that looked like the
closest analogue of anemll's `flashinfer_b12x` does not reproduce anywhere.

The serving default is unchanged: humming MoE, WO overlay off, k=6, best
same-pin **477.5 / 160.6** against same-day `refg-rc2b` **485.9 / 162.2**, i.e.
**-1.7 % / -1.0 %**.

Latest tag still `v0.30.0rc2`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: pinned v0.30.0 (9ed533eb4adf)

A newer upstream serving tag landed while round 107 was running, so the pin
moves to it under the standing rule (always the latest upstream tagged serving
version, then rebuild).

`v0.30.0` is a lightweight tag on commit `9ed533eb4adfe48aef7e569a08daeccd2a773fed`
(2026-09-21T03:14:00Z). It is **one commit ahead of `v0.30.0rc2`**, touching two
files: `#57554 "[Build] Fix DeepGEMM CUDA 12.9 release builds"`. So it is the
DeepGEMM build fix and nothing else — no runtime-visible change is expected,
but the pin still moves because it is the latest tag.

Our four tracked PRs are all still open and unmerged as of this round:
`#53425` (SM12x FlashInfer sparse MLA block size 64), `#53522` (gate indexer
paged MQA metadata on DeepGEMM support), `#53271` (KV offload device-pointer
validation), `#46716` (CPU shared-memory all-reduce deadlock). No new upstream
PR was opened.

**Overlay scan: FAIL=0 applied=43 no-op=12 total=55**, identical to the rc2
scan, run against a clean checkout of the new ref with
`scripts/port_scan.py --vllm-dir <tree>/vllm --with-upstream-patches`. The port
recipe is `configs/examples/port-v0.30.0.sh`
(`check|scan|phase1|overlay|copy|arm`), and the pin now reads
`IMAGE=vllm-spark-0731:main-030-0`, `VLLM_REF=9ed533eb4adf…`.

Note for the scan: `--vllm-dir` wants the **package** directory (the one holding
`config/kernel.py`), i.e. `<repo>/vllm`, not the repository root; passing the
root fails with `not a vllm package`.

Build in progress at the time of writing; re-baseline and then one-variable wins
follow once the image exists.

The three landed service defaults from rounds 103-105 carry over unchanged:
WO overlay off, `NUM_SPECULATIVE_TOKENS=6`, `MOE_BACKEND=humming`. Best
same-pin result so far is **477.5 / 160.6** against same-day `refg-rc2b`
**485.9 / 162.2** (-1.7 % / -1.0 %).


## 2026-09-21: v0.30.0 is a wash, and the rig drifts 6.4 % between sessions

The v0.30.0 port finished: pin, scan FAIL=0 (applied=43 no-op=12), phase-1
build, overlays, copy, re-baseline. The image is on both nodes
(`vllm-spark-0731:main-030-0`, sha `f2f9e43a7088`) and serves
`v0.30.1.dev0+g9ed533eb4`.

**The tag bump is a wash**, and this time it was measured properly. A pin bump
is not a single-variable change on its own, because the rebuild also
re-resolves `TORCH_REF`; here torch turned out to be byte-identical
(`torch 2.14.0a0+git2b3ec34`, `triton 3.7.1`, `cuda 13.3` in both images), so
the only difference is the two-file DeepGEMM build fix.

| arm (5 passes) | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `refg-now` (anemll, same session) | 63.7 | 117.0 | 142.0 | 161.2 | **483.9** | 26.6 % |
| `rc2back-rc2` (rc2 image) | 56.9 | 104.0 | 134.5 | 153.3 | **448.7** | 11.3 % |
| `proto2-030-0` (v0.30.0 image) | 57.3 | 104.8 | 132.2 | 150.2 | **444.5** | 12.6 % |

rc2 vs v0.30.0 back-to-back: **+0.9 % / +2.0 %**, inside noise.

**The drift finding matters more than the result.** The *identical* image and
config (`hum-k6-rc2b`) measured **477.5 / 160.6** at 14:14 and **448.7 / 153.3**
at 16:23 — a **6.4 %** session-to-session shift with nothing changed. That is
the same magnitude as every effect chased in rounds 103-106, so:

- the keep-rule's "larger median spread" (7-14 %) is dominated by
  *between-session* drift, not within-session noise;
- every arm-vs-standing comparison in this ledger that spans two sessions is
  confounded, including the k=6, PIECEWISE and A16 verdicts;
- only back-to-back, same-session pairs are admissible, which is why the
  reference was re-measured here rather than compared against the 2026-09-20
  `refg-rc2b` number.

**Where we actually stand, same session.** Against `refg-now` the v0.30.0
default is **-8.1 % / -6.8 %**; per level, c1 -10.0 %, c3 -10.4 %, c5 -6.9 %,
c6 -6.8 %. The small-batch levels are the worst, so the remaining gap is not
one kernel.

Our four tracked PRs remain open and unmerged. No new upstream PR was opened.


## 2026-09-21: parity with the reference; the gap was session state

A control arm settled what the earlier drift finding implied. Measured
back-to-back, five passes, same session:

| arm | c1 | c3 | c5 | c6 | sum | worst spread |
|---|---|---|---|---|---|---|
| `refg-now` (anemll) | 63.7 | 117.0 | 142.0 | 161.2 | **483.9** | 26.6 % |
| `proto2-030-0` (16:13) | 57.3 | 104.8 | 132.2 | 150.2 | 444.5 | 12.6 % |
| `proto2-030-0b` (16:43, same image+config) | 64.1 | 117.8 | 145.1 | 158.9 | **485.9** | 10.8 % |
| `capsz-030-0` (16:50, capture sizes) | 62.3 | 117.6 | 145.6 | 159.5 | **485.0** | 11.6 % |

`proto2-030-0` and `proto2-030-0b` are the **same image and the same config**,
thirty minutes apart, and differ by **+9.3 %** on the sum. So the ~8 % "gap to
anemll" reported at the start of this round was session state, not a
performance difference: against the same-session reference the control is
**+0.4 % / -1.4 %**, i.e. parity.

The capture-size arm is a **wash** (-0.2 % / +0.4 % against its adjacent
control), so the padding that adopting k=6 introduced (7->8, 21->24, 35->40,
42->48) is not a real cost. The standing capture list stays.

**What this means for the ledger.** The rig swings about +-9 % between
sessions. Every arm-vs-standing verdict here that spans two sessions is
therefore confounded, and the keep-rule's 7-14 % "spread" is mostly that
drift. The valid comparisons are the adjacent ones: `hum-k6-rc2` (b12x, 448.9)
vs `hum-k6-rc2b` (humming, 477.5) at 14:04-14:14 is a genuine +6.4 % for
humming; the rest of rounds 103-106 need re-testing as interleaved A/B pairs
before they can be believed.

We are at parity with the reference within this rig's noise, not ahead and not
behind. Beating it by more than the larger spread (the reference's own 26.6 %)
needs a change larger than anything found so far.

Our four tracked PRs remain open and unmerged. No new upstream PR was opened.


## 2026-09-21: k=6 vs k=7 validated, drift-cancelled (+6.0 % / +9.0 %)

The landed k=6 default was tested the way round 107 proved it has to be: an
interleaved, drift-cancelling A/B. Four three-pass arms in the order
k7,k6,k6,k7 so a monotonic session drift cancels in the average of the two
differences.

| arm | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|
| `abk7-a` | 52.5 | 102.6 | 127.2 | 142.9 | 425.2 |
| `abk6-a` | 53.6 | 104.8 | 132.7 | 157.7 | 448.8 |
| `abk6-b` | 55.8 | 105.3 | 133.1 | 155.2 | 449.4 |
| `abk7-b` | 51.5 | 97.0 | 129.3 | 144.1 | 421.9 |

k6 minus k7: pair1 **+5.6 %**, pair2 **+6.5 %**, mean **+6.0 %** on the sum and
**+9.0 %** at c6. Acceptance is higher for k6 (52.2-64.3 %) than k7
(48.5-57.2 %). All gates pass in all passes. The k=6 default is correct, and
the original `k6-rc2` +6.7 % now has a mechanism and a clean confirmation.

**Session context:** this window was a "slow" one (the k=6 arms sit ~449 while
the identical config measured 485.9 at 16:43), so absolute numbers still
cannot be compared across windows. What survives is the *within-pair* delta.

Since the reference (`refg-now`, k=7) is at ~484, the same +6-9 % advantage our
k=6 config holds over k=7 is what keeps us at parity with it rather than
behind. The validated sequence of defaults is: humming MoE, WO overlay off,
k=6.

Latest tag still `v0.30.0`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: all three landed defaults validated by interleaved A/B

The third landed default was tested the drift-cancelling way and is a wash.

| pair | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|
| `abwo-a` (WO off) | 57.1 | 107.8 | 134.6 | 150.9 | 450.4 |
| `abwo-b` (WO on) | 56.5 | 103.6 | 133.9 | 150.8 | 444.8 |
| `abwo-c` (WO on) | 55.1 | 107.9 | 138.8 | 151.3 | 453.1 |
| `abwo-d` (WO off) | 56.8 | 101.8 | 134.8 | 154.2 | 447.6 |

WO on minus off: **0.0 %** on the sum, **-1.0 %** at c6. The b12x WO overlay
and the einsum path are equivalent; keeping the overlay off removes our own
emulation layer at no cost.

**All landed defaults now validated by adjacent same-session pairs:**

- k=6 vs k=7: **+6.0 % / +9.0 %** (interleaved, round 109)
- humming vs b12x MoE: **+6.4 %** (adjacent pair, round 105)
- WO overlay off vs on: **0.0 % / -1.0 %** (interleaved, this round)

So the serving default is at its measured optimum among the tested levers.
Within this rig's +-9 % session swing we sit at parity with the same-session
reference, not ahead and not behind. A win over the reference that exceeds its
own 26.6 % spread needs a change larger than anything one-variable has
produced.

Latest tag still `v0.30.0`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: decisive interleaved ours-vs-reference — parity (-0.3 % / -1.0 %)

The only valid way to compare against the reference is interleaved in one
session, so four arms ran in the order ours, ref, ref, ours (three passes
each).

| arm | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|
| `abOTH-a` (ours) | 60.3 | 107.5 | 138.3 | 151.3 | 457.4 |
| `abREF-b` (ref) | 61.8 | 114.9 | 145.0 | 159.0 | 480.7 |
| `abREF-c` (ref) | 63.1 | 111.9 | 139.6 | 155.7 | 470.3 |
| `abOTH-d` (ours) | 67.6 | 116.0 | 146.6 | 160.3 | **490.5** |

Means: ours **474.0**, reference **475.5** → **-0.3 %** on the sum and
**-1.0 %** at c6. Per level: c1 **+1.4 %** (ours ahead), c3 -1.4 %, c5 +0.1 %,
c6 -1.0 %. Our own interleaved swing is 457-491 (+-3.6 %) and the reference's
is 470-481 (+-1.2 %), so every remaining cross-engine delta sits inside the
measurement noise of either engine.

**Standing conclusion:** with the validated defaults (humming, k=6, WO overlay
off) we are at statistical parity with the reference, neither ahead nor
behind. A win that clears the reference's own 26.6 % median spread would need
a change larger than anything one-variable has produced on this rig, and the
+-9 % session drift would make it unverifiable anyway. The fallback holds:
best validated configuration plus this written attribution.

Latest tag still `v0.30.0`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: attention diligence + reference max_model_len not reproducible

The owner approved a structural attention attempt (attention shares ~48 % of the
step in the eager profile). Diligence before writing any overlay:

- The sparse MLA forward (`b12x_sparse.py`) has no host-side sync or
  `.wait()`; the eager-region wall time was kernel-launch latency, not a real
  serving cost (under CUDA graphs it collapses). At parity we already serve
  attention as fast as the reference.
- The per-step "hot" ops in our overlay are no-ops: `attn_sink` is a 32-wide
  float32 parameter, so `sink.float().detach().contiguous()` allocates nothing
  and launches nothing; `output.copy_` only fires when a run returns a fresh
  buffer (not in the graph path).
- The b12x decode-attention kernels are already on their fast settings:
  `B12X_PAGED_KV_TMA`, `B12X_PAGED_DECODE_FP8_PV_M16N16_B8`, and
  `B12X_PAGED_MSA_UNION_PREFILL` all default to enabled in the image.

Conclusion: there is no exploitable inefficiency in the attention path a
blind in-repo overlay could fix. A rewrite without a measured target would be
speculation and risks regressing parity.

The one reproducible serving-config axis where we differ from the reference,
`MAX_MODEL_LEN` (65536 vs 262144), was then tested as a single variable,
interleaved (65536, 262144, 262144, 65536). Both 262144 arms never became
healthy ("container gone") — the reference's KV configuration is **not
reproducible on this rig** at our util/setup, so that axis is immovable. The
two 65536 controls (`abml-a` 56.7/109.7/138.2/151.6 sum 456.2, `abml-d`
55.7/104.7/133.6/151.7 sum 449.7) sit on our default.

Every reproducible config axis now either matches the reference or is inside
the +-9 % rig noise; the one immovable axis is the reference's max_model_len.
Parity is the measured standing.

Latest tag still `v0.30.0`. Ours still OPEN behind `pre-run-check`.


## 2026-09-21: multi-step scheduling unavailable; all axes closed

The last untested mechanical lever, `--num-scheduler-steps` (multi-step
scheduling), does not exist in this vLLM: `SchedulerConfig` has no
`num_scheduler_steps` field and `VLLM_NUM_SCHEDULER_STEPS` is not a vLLM env.
So the scheduler-host-latency hypothesis cannot be tested as a flag.

With this, every reproducible axis is closed:

- validated by interleaved same-session A/B: k=6 (+6.0 % / +9.0 % over k=7),
  humming MoE (+6.4 % over b12x), WO overlay off (0 %, a wash);
- closed by measurement: capture sizes, NCCL x20, MoE tiles/micro/dynamic/
  cutover/materialized/share/tiny-decode, indexer family, A16 (loss),
  PIECEWISE (loss), fusion/pass_config (hang), KV layouts, SP, hybrid manager,
  AOT, InstantTensor, linear backends;
- structural attention: no exploitable inefficiency found (diligence, round
  111; all b12x decode kernels already on);
- reference `max_model_len` 262144: not reproducible on this rig (round 111);
- multi-step scheduling: not implemented (this round).

The blocking condition is unchanged and demonstrated repeatedly: the rig's
session drift is about +-9 %, larger than every achievable one-variable effect
(max +6 %), and the win bar (beat the reference by more than the larger median
spread, 26.6 % for the reference's 5-pass run) is unreachable by any in-scope
lever. The decisive interleaved comparison measured parity: -0.3 % sum /
-1.0 % c6. Goal marked blocked with this attribution.

Latest tag still `v0.30.0`. Ours still OPEN behind `pre-run-check`.


## 2026-09-22: KV-capacity sweep on the clean v0.30.0 build

`MAX_MODEL_LEN` probed upward from the pin's 65536 toward the reference's
262144, one variable per arm (protocol 3x512, levels 1 3 5 6), on
`vllm-spark-0731:main-030-0` with the validated default (humming, k=6, WO off).

| max_model_len | c1 | c3 | c5 | c6 | sum | spread | gates |
|---|---|---|---|---|---|---|---|
| 65536 (pin) | 57.8 | 107.9 | 135.2 | 153.5 | 454.4 | 8.7 % | pass |
| 98304 | 59.8 | 109.7 | 136.2 | 157.6 | **463.3** | 9.9 % | pass |
| 131072 | 59.1 | 103.5 | 133.3 | 150.9 | 446.8 | 6.1 % | pass |
| 196608 | 56.6 | 107.1 | 133.9 | 148.9 | 446.5 | 9.9 % | pass |
| 229376 | 55.0 | 106.9 | 139.5 | 152.8 | 454.2 | 10.0 % | pass |
| 245760 | 57.1 | 107.2 | 137.4 | 147.6 | 449.3 | 5.6 % | pass |
| 262144 | die | die | die | die | — | — | — |

**Finding: the serving frontier tops out at `MAX_MODEL_LEN=245760` (KV cache
11.32 GiB, from the worker report); `262144` dies at the KV/memory floor, same
as round 111, so the reference's max_model_len (262144) is not reproducible
on this rig at our util/layout.** Across 65536-245760 the sums sit in a
446-463 band whose spread overlaps the +-9 % session drift, so KV depth is
performance-neutral up to the frontier; the best single-arm sum (98304,
463.3) is not distinguishable from the pin default within noise. No reason to
change the pin default; the capacity gain is 4x (65536 -> 245760) at no
measured throughput cost. The reference's KV pool is unreachable.

Latest tag still `v0.30.0`. Ours still OPEN behind `pre-run-check`.

