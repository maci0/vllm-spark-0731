# Reference stack: v0.28.0 (`main-b12x-028-rdma`) — the new golden

> **Status 2026-08-29: our v0.28.0 stack is the throughput reference.**
> `vllm-spark-0731:main-b12x-028-rdma` (v0.28.0 + b12x + RoCE + o_proj bmm
> + mHC/gumbel warmup) now **exceeds** the anemll image on aggregate
> throughput: **306.8 tok/s @ c32** (c24 260.7, c16 183, c8 117, SM util
> 95%) vs the anemll image's max **216.8 tok/s @ c6**. This chapter
> documents the full picture: the current stack (this repo), the legacy
> anemll reference it surpassed, and the two dimensions where the anemll
> image still leads (real NVFP4 KV capacity, single-stream latency).
>
> - Current stack numbers: [05-performance.md](05-performance.md),
>   [HANDOFF.md](../../HANDOFF.md).
> - Legacy anemll operational field notes:
>   [docs/field-notes/dgx-spark/GOLDEN.md](../../field-notes/dgx-spark/GOLDEN.md).
> - DeepGEMM internals: [09-golden-deepgemm.md](09-golden-deepgemm.md).

## 1. The current reference stack (this repo)

`vllm-spark-0731:main-b12x-028-rdma` = `main-b12x-028-p1` (v0.28.0 release
`2cf0a6915`, CUDA 13.3.1 / torch 2.14 `12.1a`, full source build) +
rdma-core v54 libmlx5 overlay (NCCL RoCE) + `patches/files` donors +
warmup ext. Reproducible: `docker/Dockerfile.ov-rdma` +
`scripts/ov-rebuild-rdma.sh`. Config: `configs/pin.main-dg.env`
(`B12X_MLA_SPARSE`, `nvfp4_ds_mla` 584 B envelope, DSpark k=5, util 0.8,
`max_num_seqs` 32).

Measured (2026-08-28/29, golden methodology, completions temp 0.0 / chat
temp 0.7): c1 steady-state **40.2-43.5**, c8 **117.2**, c16 **183.0**,
c24 **260.7**, c32 **306.8** tok/s agg; France logprob -0.254 (matches the
einsum reference exactly); o_proj decode bmm active with 0 fallbacks.

## 2. The legacy anemll reference (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`)

The prebuilt anemll/eugr image this stack was benchmarked against. It was
the **only** image tested (of the eugr / stage-c / upstream lineages) that
delivers **real NVFP4 KV compression** (7,650 B/token) and historically the
decode-throughput leader up to ~c6. It runs **stock, unpatched**.

Key identity facts:
- vLLM **0.25.2** (Jul, pre-0.28), torch **2.11**, Ubuntu **22.04**.
- DeepGEMM fork self-labeled `2.5.0` (the `a6b593d`-era kernel set with
  SM12x support), **all 1,762 linear kernels compiled at image build**.
- NCCL **2.28.9** + libmlx5 **1.22** — RoCE out of the box (our v0.28.0
  stack needed the rdma-core v54 libmlx5 overlay).
- Real NVFP4 KV writer (17 files) without the over-broad
  `cache_dtype.startswith("nvfp4") and use_mla` guard.

## 3. The exact legacy setup (shipped recipe)

Recipe: `configs/examples/anemll-nvfp4-golden.yaml` (sparkrun),
deployed via `~/spark-launch.sh anemll-nvfp4-golden.yaml ~/anemll.log`.

| Setting | Value | Why |
|---|---|---|
| container | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | only image with real NVFP4 KV |
| model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` | **abliterated**, DSpark draft head bundled |
| `kv_cache_dtype` | `nvfp4_ds_mla` | 7,650 B/token |
| `gpu_memory_utilization` | 0.82 | 0.835 allocates then dies in FlashInfer autotune |
| `max_num_seqs` | 6 | 5 clients + headroom |
| `num_speculative_tokens` | 5 (DSpark) | k=5 floor |
| `max_cudagraph_capture_size` | 36 | exactly `max_num_seqs × (k+1)` |
| `max_model_len` | 1,048,576 | full 1M context |
| `block_size` | 256 | |
| `max_num_batched_tokens` | 8192 | |
| `long_prefill_token_threshold` | 1024 | |
| tool calling / reasoning / tokenizer | `deepseek_v4` parsers + auto-tool-choice | feature flags matter as much as perf |
| patches | **none** | stock image |

Endpoint: `http://192.168.0.211:8000/v1` · model aliases `deepseek-v4-flash`,
`dsv4` (the served-name **replaces** the HF path).

## 4. Measured numbers (one harness, this cluster)

Methodology: warm, 128 tok/req, single shared coding prompt, temp 0.7,
aggregate tok/s (`scripts/bench-concurrency.py --chat`). The 2026-08-24
golden-harness France run: **c1 65.2, c6 216.8 tok/s**, KV **2,047,170**
tokens @ **7,650 B/token** (DSpark accept ~66.7%).

Three-lineage comparison (2026-08-22, same harness):

| | **anemll (legacy)** | eugr + PIECEWISE | stage-c |
|---|---:|---:|---:|
| KV pool | 1,971,682 - 2,002,497 | 1,659,937 - 1,768,024 | 1,438,916 |
| bytes/token | **7,650** | 11,317 | ~11,900 |
| max concurrency @ 1M | **1.91x** | 1.58x | 1.37x |
| c1 | 51.4 | 54.3 | **56.1** |
| c3 | **112.7** | 90.3 | 93.5 |
| c5 | 126.2 | **127.4** | 116.0 |
| c6 | **157.9** | ~109 | 141.1 |
| vLLM | 0.25.2 | 0.27.x | 0.21.1rc1 |

Caveats:
- **Workload-dependent ~1.7×**: shared coding prompt 141 vs unique prose
  prompts + forced `ignore_eos` 81 tok/s at c5 (prefix-cache sharing +
  DSpark acceptance both collapse).
- **KV pool varies ~1.5% between identical boots** (vLLM derives it from
  free memory at profile time) — read it as a range, not a regression.

## 5. Why the legacy was faster, and where our stack surpassed it

The old gap was **whole-stack**, not one kernel. Decomposition (from
[05-performance.md](05-performance.md), [09-golden-deepgemm.md](09-golden-deepgemm.md)):

1. **Real NVFP4 KV** (7,650 B/token vs our `nvfp4_ds_mla` 584-byte
   envelope **alias** of fp8): +32% KV capacity — the legacy serves 2.0M
   tokens vs our ~98k at util 0.8. **We still trail here (capacity).**
2. **DeepGEMM fork with SM12x kernels pre-compiled at build** — our stack
   ships the v0.27.1 DeepGEMM main `.so` (SM90/SM100) and falls back to
   b12x/PyTorch/TileLang on SM12x; the legacy compiled its 1,762-kernel
   warmup into the image.
3. **Older vLLM core (0.25.2)** with lower per-step overhead.
4. **RoCE out of the box** (NCCL 2.28.9 + libmlx5 1.22) — we fixed ours
   with the rdma-core v54 overlay.
5. **DSpark acceptance ~66.7%** vs our measured ~0.9 tokens/step at c1 —
   **we still trail here (single-stream latency).** Our greedy A/B did not
   close it; it is a whole-path difference, not the sampler flag alone.

**Where our stack now surpasses it:** aggregate throughput (306.8 @ c32 vs
216.8 @ c6 — the legacy's `max_num_seqs=6` caps it), SM util 95%, and the
correctness of the o_proj decode path (France logprob -0.254, einsum-exact).

## 6. How to reproduce the legacy (if needed)

**The real path (matches the legacy numbers):**
```bash
# sparkrun recipe + abliterated checkpoint (downloaded on spark1)
bash ~/spark-launch.sh anemll-nvfp4-golden.yaml ~/anemll.log
curl -s localhost:8000/health   # 200
```
Model: `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32`.
Boot ~8-10 min. Operational notes (memory release, `/health` caveat,
`spark-launch.sh` teardown): field note §6.

**The serve-script path (`scripts/05-serve.sh golden`, pin.golden.env) does
NOT reproduce the numbers (2026-08-26):** it feeds the **vanilla**
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint from the node's
`/models/ds4-flash-0731`, and (a) vLLM 0.25.2 dies on empty
`VLLM_USE_B12X_MOE` (`int('')`) unless the B12X_* vars are pinned
(pin.golden.env pins them to 0), and (b) with the vanilla checkpoint
`profile_run` hits the fp8_einsum `layout.hpp:97` scale assert (the same
ue8m0-recipe family fixed on main by vllm-project/vllm#53521).

## 7. Current status / what the legacy settles

- Real NVFP4 on 0731 = the anemll image (adopted 2026-08-24). Porting the
  NVFP4 writer into our v0.28.0 stack is **not** needed for the aggregate
  speed goal; the b12x `nvfp4_ds_mla` 584-byte alias stays for our image.
  It WOULD be needed to close the KV-capacity gap (98k vs 2.05M tokens).
- The legacy's DeepGEMM pin (`a6b593d`) is the SM12x-fp8-known-good; our
  DeepGEMM #419 PR restores the removed pure-fp8 1d1d kernels (see
  [09-golden-deepgemm.md](09-golden-deepgemm.md)).
- Rollback from the legacy: `bash ~/spark-launch.sh eugr-prod.yaml`
  (v0.27.x, -13% capacity, -45% c6).
- SSD offload does NOT work for this model on any lineage (hybrid
  multi-group cache vs single-flat-layout offload assumption) — see the
  field note §7.
- Remaining gaps vs the legacy, tracked in
  [05-performance.md](05-performance.md) and HANDOFF: c1 single-stream
  latency (40-43 vs 51-65) and KV capacity (real NVFP4 writer).

## 8. Real NVFP4 into our v0.28.0 image — experiment verdict (2026-08-29)

Asked "how can we get NVFP4 into the v0.28.0 image", tested end-to-end:

- **v0.28.0 already ships an `nvfp4_ds_mla` path**: the
  `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` writer op (compiled in
  `_C_stable_libtorch.abi3.so`, registered when vllm imports) + the FlashInfer
  DSV4 sparse MLA reader (`flashinfer_sparse.py` accepts `nvfp4_ds_mla`).
  Switching to `--attention-backend FLASHINFER_MLA_SPARSE_DSV4` +
  `nvfp4_ds_mla` boots and serves coherently on SM12x — after removing the
  SM12x Triton-fp8 KV-insert diversion for the nvfp4 case (the
  `patch_sm12x_kv_insert` overlay bypasses the CUDA op on SM12x; the exclusion
  lets `nvfp4_ds_mla` fall through, verified `kv_cache_dtype='nvfp4_ds_mla'` +
  0 Triton diversions + no CUDA errors).
- **BUT the density is unchanged**: KV cache 100-115k tokens at ~214 KB/token —
  identical to the fp8 path. v0.28.0's `nvfp4_ds_mla` is the 584-aligned padded
  uint8 layout (584 vs fp8's 576 = +1.4%), i.e. an **fp8-density alias**.
- **The golden's 7,650 B/token (28.7× denser) is a different cache layout in
  its vLLM 0.25.2 fork** — not the v0.28.0 `nvfp4_ds_mla` format. Porting it
  means porting the golden's dense FP4-packed MLA cache (writer op + cache
  layout) AND a reader (decode kernels that dequantize it) — the writer is a
  compiled op (torch 2.11 ABI, not binary-liftable); the source may exist in
  vLLM git history (the op name matches v0.28.0's). Multi-day kernel/porting
  work, not a config change.
- **Verdict/revert**: FlashInfer+`nvfp4_ds_mla` gives no capacity or speed
  gain over `B12X_MLA_SPARSE` (which is the 306.8-tok/s @ c32 config), so the
  stack stays on `B12X_MLA_SPARSE` + `nvfp4_ds_mla` (the fp8 alias). Real NVFP4
  capacity only matters for 1M-context serving; our 64k `max_model_len` holds
  1.5× concurrency at the current 99k-token pool.

## 9. Port plan: FP4-packed MLA cache into v0.28.0 (scoped 2026-08-29)

Goal: close the 28× KV-density gap (our 214 KB/token vs the golden's 7,650
B/token). Investigation findings + the plan:

**Confirmed:**
- The golden's density formula (its `MLAAttentionSpec.real_page_size_bytes`,
  `vllm/v1/kv_cache_interface.py`): for DSV4 with fp8/nvfp4_ds_mla the page =
  `storage_block_size × 584`, where `storage_block_size = block_size //
  compress_ratio`. The compress-ratio-aware storage blocks (C4: 146 B/token,
  C128: 4.6 B/token, SWA separate) are what make the golden ~28× denser.
- v0.28.0 has the SAME `MLAAttentionSpec` machinery
  (`storage_block_size = block_size // compress_ratio`,
  `state_content_bytes=584` for fp8_ds_mla, `vllm/v1/kv_cache_interface.py`)
  — the compressed layout machinery EXISTS upstream.
- **BUT our measured cache is 214 KB/token, ~28× above the formula** — the
  running DSV4 (b12x AND flashinfer backends both measured ~100-115k tokens)
  does not allocate via the upstream packed-MLA layout path (a debug print
  injected in `_get_kv_cache_config_packed` never fired; the exec-patch
  raced the engine import, so this needs the debug baked into the image
  build to confirm precisely which allocation path runs).
- The FP4 writer op (`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`) is
  compiled + registered in our image and RUNS on SM12x (verified when the
  SM12x fp8-diversion was bypassed); its v0.28.0 layout is the 584/576
  padded uint8, not the dense FP4.

**Remaining unknowns (bake debug into the image build, then one boot):**
1. Which cache-allocation path the live DSV4 actually uses (dump the per-layer
   `page_size_bytes`/`storage_block_size` at allocation).
2. Whether v0.28.0's MLA cache layer honors `compress_ratio` for the b12x
   backend or whether b12x bypasses the upstream layout entirely.

**Plan (in order):**
1. Bake the KV-spec debug dump into the overlay build (`--only` patch on
   `kv_cache_utils`), boot, capture the per-layer page sizes — pins down
   whether the compressed layout is reachable with a config/spec change.
2. If the b12x cache bypasses the upstream layout: switch the DSV4 cache to
   the upstream MLA cache path (the spec already carries compress_ratio +
   state_content_bytes; likely needs the FLASHINFER_MLA_SPARSE_DSV4 backend
   + the upstream cache layer to be the allocation path), measure capacity.
3. Port the golden's dense-FP4 writer behavior (the op's FP4 packing; the
   v0.28.0 csrc op source is available at
   `csrc/libtorch_stable/torch_bindings.cpp` + `ops.h`) and the reader
   (flashinfer/b12x decode dequant for the dense layout) — the C++/kernel
   work. The golden's op is compiled (torch 2.11 ABI, not liftable); the
   layout intent is documented in its `attention.py` comment
   ("padded 584-byte DSpark NVFP4 envelope") and the
   `MLAAttentionSpec.real_page_size_bytes` formula above.
4. Validate: KV capacity jump (target ~2M tokens), France coherence,
   throughput (c1/c32) — keep `B12X_MLA_SPARSE` unless the upstream layout
   path is required for the density.

**Blockers/risks:** the golden's dense layout reader is the binding unknown
(kernel-level); the writer op's FP4 mode in v0.28.0 is unverified on SM12x;
boot/import races make exec-patching unreliable — all experiment patches must
be baked into the overlay image.
