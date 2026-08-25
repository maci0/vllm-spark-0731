# Patches

Applied at image build by `apply_overlays.py`. `--stack rc2` is the
v0.27.1 + rc2 overlay image. `--stack main` is `vllm-spark-0731:main-b12x`.

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
| `flashinfer-eidx-contig` | Enforce `.contiguous()` on FlashInfer extra_sparse_indices (PR #53574 defense-in-depth). |
| `triton-e8m0-sm12x` | Upcast E8M0 scale to fp32 on SM12x in Triton block scaled MM (PR #47988). |
| `einsum-sm12x` | Apply SM90 `(1,128,128)` einsum recipe on SM12x (PR #53521). |
| `sm12x-kv-insert` | Use XPU/Triton fused qnorm-RoPE KV insert on SM12x with eager scratch pool. |
| `instanttensor-hybrid` | InstantTensor loader with hybrid lazy draft. |
| `dsv4-block64` | Allow block size 64 for DSV4 MLA on SM12x (PR #53425). |
| `dspark-backbone-cg` | Graph DSpark transformer backbone. |
| `dspark-backbone-none` | Graph DSpark transformer only. `_sample_sequential` stays eager (shared `lm_head`). |
| `ar-piecewise-ws` | In-graph TP all-reduce (no per-layer PIECEWISE eager-break). |

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
| `upstream/` | Active vLLM main open PR backports (`pr-*.diff`) and DeepGEMM diffs (`deepgemm-*.diff`) |

## Do not use

- `maci0/vllm-spark-nvfp4` `nvfp4-ds-mla-v0.27.1.patch` (191-line envelope)
- `eugr-nvfp4.patch` (89-line; 432 vs 584)
- GLM 432/368 writer
- Stage-C `head_bytes = 584` probe without a writer
