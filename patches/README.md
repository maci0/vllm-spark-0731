# Patches

Applied at image build by `apply_overlays.py`. `--stack rc2` is the
v0.27.1 + rc2 overlay image. `--stack main` is `vllm-spark-0731:main-b12x`;
`--stack main` on vLLM v0.29.0 is `vllm-spark-0731:main-029`.

## vLLM v0.29.0 port (`vllm-spark-0731:main-029`)

v0.29.0 absorbed several overlays, which now skip themselves instead of
editing the tree:

| Overlay | v0.29.0 |
|---|---|
| `patch_moe_backend` | no-op: `MoEBackend` in `config/kernel.py` already lists `"b12x"`, docstring included |
| `patch_utils_b12x` | no-op: `B12xWarmupUnit` and `get_b12x_fused_moe` are upstream |
| `patch_mxfp4_oracle` | no-op: `B12X_MXFP4_*`, `B12X_BACKENDS`, `map_mxfp4_backend`, `_get_requested_backends` are upstream |
| `patch_mxfp4_process_weights` | no-op: `Mxfp4MoEMethod` builds the kernel without `layer` and preps the experts in the caller |
| `patch_dspark_skip_cudagraph` | no-op: upstream graphs the draft backbone only and keeps the shared lm_head eager |
| `patch_dsv4_sm12x_block_size` | no-op once `pr-53425.diff` applies (see below) |
| `patch_flashinfer_dsv4_dispatch`, `patch_flashinfer_dsv4_cu_dispatch` | retired for this flashinfer: topk is a runtime kernel argument |
| `patch_indexer_b12x_schedule` | re-anchored: v0.29 inlines the DeepGEMM eligibility test (upstream PR #53522) |
| `patch_einsum_sm12x_recipe` | **re-enabled**: `compute_fp8_einsum_recipe()` now returns `(1,1,128)` for every major >= 10, which family 120 cannot satisfy once our DeepGEMM guard keeps `is_deep_gemm_e8m0_used()` false |
| `patch_o_proj_einsum_e8m0` (new, `--only o-proj-einsum-e8m0`) | gives the SM12x o_proj einsum DeepGEMM's 3-D `is_bmm` wo_a pair, mirroring `deepgemm_post_process_fp8_weight_block` |

`patches/upstream/pr-53522.diff` used the pre-v0.29 spec attribute
`storage_block_size`; v0.29.0 renamed it `num_states`, so the first decode
build raised `AttributeError` during cudagraph capture. The diff now uses
`num_states` (the other occurrence was a context line that `patch` had
fuzz-matched, leaving the file's own correct attribute).

`patch_sm12x_kv_insert`'s `nvidia/dspark.py` insertion read
`getattr(self, "kv_cache_dtype", None)`; v0.29.0 moved that body into the
module-level `_insert_context_kv(attn, ...)`, so it is now
`getattr(attn, "kv_cache_dtype", None)` (`self` survives in the
attention.py insertion, which is still a method).

`patches/upstream/pr-53425.diff` dropped its
`from vllm.platforms import current_platform` hunk. v0.29.0 already has
that import, and `patch --forward` then classifies the hunk as
already-applied and skips the **whole file**, so
`dsv4_supported_kernel_block_sizes()` stayed out while the indexer hunk
still landed, leaving a name the indexer imports and `sparse_mla.py` never
defined.

`docker/assert_main_image.py` probes
`(H, 192) in flashinfer.mla._sparse_mla_sm120._DECODE_DSV4_DISPATCH` for
all five head counts instead of scanning source text for `"(8, 192)"`.
FlashInfer main made `topk` a runtime kernel argument and moved the
dispatch to `_sparse_mla_sm120_plan.py` as a `_DecodeDispatchEnvelope`
predicate, so the text marker is gone and the C++ `DSV4_DISPATCH(H, topk)`
table no longer exists.

`configs/pin.main-029.env` pins `B12X_REF` to the b12x commit behind
version 1.2.6, the version vLLM v0.29.0's `b12x` extra declares. b12x
master is now 1.3.0 and its `dsa_indexer` package no longer exposes
`PagedDecodeMetadata`, `prepare_paged_metadata`, `uses_paged_schedule`, or
`plan_paged_schedule`, which `files/sm12x_b12x_kernels.py` imports.

Build and serve: `scripts/02-build-main-029.sh`,
`scripts/03-apply-main-overlays-029.sh`, `scripts/05-serve.sh main-029`.
Drop the page cache on both nodes before serving: InstantTensor sizes its
I/O buffer from free device memory and fails with `buffer_size ... exceeds
device memory budget` while ~57 GiB of page cache is held.

The -029 image now boots, serves, and answers coherently (France greedy ->
" Paris. The capital of Spain is Madrid..."). The DSV4 prefill o_proj needed
the 3-D `is_bmm` wo_a pair:

`wo_a.weight` is 2-D (`[groups*rank, group_width]`) because the b12x FP8
linear kernel does not do the `is_bmm` 3-D view that the DeepGEMM FP8 linear
kernel does in `deepgemm_post_process_fp8_weight_block`, and
`files/sm12x_b12x_kernels.py::try_b12x_wo_proj` (which needs the 2-D
checkpoint layout) only serves decode batches of at most 256 rows, so every
prefill-sized batch reaches the einsum and `fp8_bmm` asserts `t.dim() == N`
(`get_shape<3>`, `csrc/utils/layout.hpp:39`).

`patch_o_proj_einsum_e8m0` builds that pair once per module with
`use_e8m0=True` and passes it to `fp8_einsum` under the `(1, 128, 128)`
recipe. On-device, the only combination that both runs and computes is the
UE8M0-rounded scale with that recipe; fp32 non-power-of-two scales run but
return NaN, and recipe `(1,1,128)` trips
`layout.hpp:97 sf.size(-2) == ceil_div(mn, gran_mn)`. Full matrix and the
upstream code mirrored: HANDOFF.md "2026-09-10 (8)".

Benchmark at k=7 (capture 48, 6 seqs, batched 12288, `max_model_len=8192`
because the official FP8 checkpoint leaves only ~5.8 GiB for KV at util
0.8): c1 54.8 / c3 115.7 / c5 94.0 / c6 106.1 against the anemll k=7 bar of
71.5 / 157.6 / 195.1 / 233.6. It loses at every level, so the box is left
on anemll k=7.

## Build-time overlays (baked into image)

| File | Why |
|---|---|
| `files/fused_moe_b12x.py` | B12xExperts MoE (rc2 only; main already has this module) |
| `files/b12x_moe.py` | b12x weight helpers (rc2 only) |
| `files/dsv4_b12x_sparse.py` | Main: `B12X_MLA_SPARSE` on the 584 B DSV4 page (`b12x.attention.compressed_mla`) |
| `apply_overlays.py` | SM12x guards, nvfp4_ds_mla 584 B, MQA ReLU, AR eager-break, b12x-sparse register/select |
| `pin_cutlass_dsl.py` | Main-image only: rewrite b12x (5) and quack-kernels (2) `==4.6.2` cutlass pins to 4.7.0 before `--no-deps` |
| `assert_image.py` | build-time: SM12x guards, DSV4 shape, KV dtypes; `--stack main` also checks `B12X_MLA_SPARSE` |
| `assert_0731.py` | checkpoint pin: dspark_block_size=5, DeepseekV4ForCausalLM |
| `assert_stack.py` | runtime: DSpark k=5 required; B12X_MLA_SPARSE + nvfp4_ds_mla is the 584 B DSV4 mix |

Incremental on a running main image: `--only b12x-sparse --vllm-dir /opt/vllm/vllm`.
Pass `--vllm-dir` so apply does not `import vllm`. A duplicated
`B12X_MLA_SPARSE` enum makes that import raise `TypeError`.

Decode overlays and fine-grained patches on main (re-apply with `--vllm-dir`):

| `--only` | Why |
|---|---|
| `b12x-sparse` | Register and wire `B12X_MLA_SPARSE` backend on the 584 B DSV4 page (`b12x.attention.compressed_mla`). |
| `o-proj-b12x` | SM12x WO via fused inv-RoPE + dequant + `torch.bmm`. Leave MXFP8 `wo_proj.run()` off (France loops). |
| `indexer-store-page64` | Store indexer K as four 64-token packed pages per 256-token manager block. Gather of that layout is numerically wrong. |
| `indexer-b12x-schedule` | SM120 `plan_paged_schedule` into `scheduler_metadata_buffer` only when `q_rows==1`. Helper consumes that 1-row schedule. Multi-row (DSpark 6/8) stays on the unscheduled 1023-page scorer; 48-row 8-way is already unscheduled. Skip the page64 expand workspace when manager tables are already 1024-wide. |
| `indexer-mqa` | Guard indexer DeepGEMM metadata builder against SM12x / 2-state pages. |
| `mqa-packed-gather` | Route MQA logits through `packed_gather_mqa_logits` to fix packed K-then-scale gather offsets. |
| `mqa-paged-kernel` | Enable b12x paged MQA logits decode kernel. |
| `triton-e8m0-sm12x` | Upcast E8M0 scale to fp32 on SM12x in Triton block scaled MM (PR #47988). |
| `einsum-sm12x` | Apply SM90 `(1,128,128)` einsum recipe on SM12x (PR #53521). |
| `sm12x-kv-insert` | Use XPU/Triton fused qnorm-RoPE KV insert on SM12x with eager scratch pool. |
| `instanttensor-hybrid` | InstantTensor loader with hybrid lazy draft. |
| `dsv4-block64` | Allow block size 64 for DSV4 MLA on SM12x (PR #53425). |
| `dspark-backbone-cg` | Graph DSpark transformer backbone. |
| `dspark-backbone-none` | Graph DSpark transformer only. `_sample_sequential` stays eager (shared `lm_head`). |
| `ar-piecewise-ws` | In-graph TP all-reduce (no per-layer PIECEWISE eager-break). |
| `mhc-tf32-uncaptured` | **Parked from `apply_main` 2026-09-14.** Inserts the `torch.compiler.disable` wrapper that keeps the tf32 prenorm GEMM and its local pybind import out of the compiled region. Its call-site needle predates `pr-53055.diff`, which folds that import into a guarded `is_deep_gemm_supported, tf32_hc_prenorm_gemm` line, so on the current pin it raises and cuts off every later overlay. Re-anchor before re-enabling. |
| `mhc-tf32-redirect` | **Parked from `apply_main` 2026-09-14** with the overlay above. Points the mHC forwards at that wrapper; on the post-`pr-53055` tree it rewrites three of the four call sites and leaves `mhc_pre_tilelang`'s single-line call on the pybind op. |

`patches/files/sm12x_b12x_kernels.py` is copied onto the image as
`vllm/utils/sm12x_b12x_kernels.py`.

`dsv4_b12x_sparse.py` contracts:

- Cache view is `[pages, page_bytes]` uint8. Page size comes from the tensor
  (SM12x kernel page 64 → 37376 B), not C4 `block_size/4=16`. Indices are
  raw slot ids (`page * page_size + offset`), same as FlashInfer.
- Decode scratch: 128 rows × 16 chunks. Prefill scratch: `max_num_batched_tokens`
  × 2 chunks. A single 8192×16 plan reserved 4.3 GiB and starved 65k KV.
- Prefill MG (and Spark decode at 16+ tokens with ≤10 chunks) rejects DSpark's
  padded SWA width 192. SWA-only pads to 512. Dual-cache prefill clips SWA to
  128 (the real window). Decode batches under 16 tokens keep width 192.

## Diagnostic & Warmup Scripts

Standalone validation and analysis tools in `patches/`:

| Script | Purpose |
|---|---|
| `diag_hc_head.py` | Cosine TileLang `hc_head` / `mHC` post / small-FMA fused path vs torch reference. France prefill is 5 tokens. |
| `diag_lm_head.py` | Isolated embed + `lm_head` scoring of the France prompt without full transformer execution. |
| `diag_mqa_b12x.py` | Prefill MQA verification comparing B12x contiguous MQA against Python ReLU reference. |
| `diag_router.py` | Cosine verification of DeepSeek-V4 router logits on SM12x vs torch sqrtsoftplus+hash. |
| `diag_sm12x_accuracy.py` | Accuracy probes comparing CuteDSL indexer-Q vs Triton vs GPT-J/UE8M0 reference. |
| `warmup_sm12x_kv_insert.py` | Pre-compiles SM12x Triton KV-insert kernels while GPU is empty, preventing SIGKILL at peak memory on spark2. |

## Reference files (not applied directly)

| Dir | Contents |
|---|---|
| `hotfixes/` | Full-file versions used during v0.27.1 bring-up (kept for reference) |
| `upstream/` | Active vLLM main open PR backports (`pr-*.diff`, the only glob the build consumes), plus historical donor diffs kept for provenance (`0002*`, `0003*`, `*-vllm-only.diff`, `kv-offload-bounds-check.patch`) |
| `v0.27.1/` | Historical verified patch set for the pristine `v0.27.1` image (`combined-v0.27.1.patch`, `eugr-nvfp4.patch`) — see `docker/Dockerfile.nvfp4` |
| `v0.28/` | Legacy git-diff patch set for the rc2 overlay track (`0001`–`0008`); superseded by `apply_overlays.py --stack rc2`, kept for history |

## Do not use (historical donor patches — not for standalone application)

- `upstream/0003-nvfp4-ds-mla-v0.27.1.patch` — the 191-line v0.27.1 NVFP4
  envelope. Only the rc2 fallback applies it (via `patch_nvfp4_ds_mla`);
  do **not** apply it to the matched-main image (main handles the dtype via
  the `b12x-sparse` overlay).
- `v0.27.1/eugr-nvfp4.patch` — 89-line eugr-image-only patch (432 vs 584);
  not for this repo's images.
- `v0.27.1/combined-v0.27.1.patch` — applies to a pristine v0.27.1 tree only;
  superseded by the overlays on both live stacks (`--stack rc2` / `--stack main`).
- GLM 432/368 writer
- Stage-C `head_bytes = 584` probe without a writer
