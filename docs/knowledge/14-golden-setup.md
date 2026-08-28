# Golden setup: anemll dspark-vllm-gx10 (the reference target)

> Why this chapter exists: the golden image is the **performance and
> correctness reference** for this project — every number in
> [05-performance.md](05-performance.md) and every "golden" mention in
> HANDOFF/UPSTREAM refers to it. This chapter is the single place that
> documents what it is, the exact setup, why it is ~2× faster than the
> v0.28.0 stack, how to reproduce it, and what it measured.
>
> Deep-dive operational field notes: [docs/field-notes/dgx-spark/GOLDEN.md](../../field-notes/dgx-spark/GOLDEN.md).
> DeepGEMM internals: [09-golden-deepgemm.md](09-golden-deepgemm.md).
> Gap analysis vs our stack: [05-performance.md](05-performance.md).

## 1. What it is

`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` — a **prebuilt** container from the
anemll/eugr lineage, built for 2x DGX Spark (GB10). It is the **only** image
tested (of the eugr / stage-c / upstream lineages) that delivers **real
NVFP4 KV compression** (7,650 B/token, 32% below fp8) and the only one whose
decode throughput exceeds ours at every concurrency up to ~c6. It runs
**stock, unpatched** — "patches: none".

Key identity facts:
- vLLM **0.25.2** (Jul, pre-0.28), torch **2.11**, Ubuntu **22.04**.
- DeepGEMM fork self-labeled `2.5.0` (the `a6b593d`-era kernel set with
  SM12x support), **all 1,762 linear kernels compiled at image build**
  (warmup runs in <1 s at runtime).
- NCCL **2.28.9** + libmlx5 **1.22** — an OLDER NCCL that works over RoCE
  out of the box (predates the mlx5dv-symbol requirement; our v0.28.0
  stack needed the rdma-core v54 libmlx5 overlay — see
  [05-performance.md](05-performance.md) §2b).
- Real NVFP4 KV writer (17 files) without the over-broad
  `cache_dtype.startswith("nvfp4") and use_mla` guard that makes the dtype
  unreachable on eugr/upstream builds.

## 2. The exact setup (shipped recipe)

Recipe: `configs/examples/anemll-nvfp4-golden.yaml` (sparkrun),
deployed via `~/spark-launch.sh anemll-nvfp4-golden.yaml ~/anemll.log`.

| Setting | Value | Why |
|---|---|---|
| container | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | only image with real NVFP4 KV |
| model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` | **abliterated**, DSpark draft head bundled |
| `kv_cache_dtype` | `nvfp4_ds_mla` | 7,650 B/token |
| `gpu_memory_utilization` | 0.82 | 0.835 allocates then dies in FlashInfer autotune (§4 of the field note) |
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

## 3. Measured numbers (one harness, this cluster)

Methodology: warm, 128 tok/req, single shared coding prompt, temp 0.7,
aggregate tok/s (`scripts/bench-concurrency.py --chat`). The 2026-08-24
golden-harness France run: **c1 65.2, c6 216.8 tok/s**, KV **2,047,170**
tokens @ **7,650 B/token** (DSpark accept ~66.7%).

Three-lineage comparison (2026-08-22, same harness):

| | **anemll (golden)** | eugr + PIECEWISE | stage-c |
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

## 4. Why it is faster than our v0.28.0 stack

The gap is **whole-stack**, not one kernel. Decomposition (from
[05-performance.md](05-performance.md), [09-golden-deepgemm.md](09-golden-deepgemm.md)):

1. **Real NVFP4 KV** (7,650 B/token vs our `nvfp4_ds_mla` 584-byte envelope
   **alias** of fp8): +32% KV capacity → the golden serves 2.0M tokens vs
   our ~98k at util 0.8. Capacity, not just speed.
2. **DeepGEMM fork with SM12x kernels pre-compiled at build** — our stack
   ships the v0.27.1 DeepGEMM main `.so` (SM90/SM100) and falls back to
   b12x/PyTorch/TileLang on SM12x; the golden compiled its 1,762-kernel
   warmup into the image (<1 s at runtime vs our per-boot JIT).
3. **Older vLLM core (0.25.2)** with lower per-step overhead and an
   attention/DSpark path tuned for this exact model.
4. **RoCE out of the box** (NCCL 2.28.9 + libmlx5 1.22) — no libmlx5
   overlay needed (we fixed ours with rdma-core v54).
5. **DSpark acceptance ~66.7%** vs our measured ~0.9 tokens/step at c1 —
   the golden's draft/verify path accepts more (our greedy A/B did not
   close this; it is a whole-path difference, not the sampler flag alone).

Where we now EXCEED it: our v0.28.0 stack aggregates **306.8 tok/s at c32**
(golden caps at c6 ≈ 216.8) and our single-stream **c1 40-43 tok/s** is
closer than the c6 gap suggests. The golden remains the reference for
NVFP4 capacity + per-stream latency + acceptance.

## 5. How to reproduce

**The real path (matches the shipped numbers):**
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
ue8m0-recipe family fixed on main by vllm-project/vllm#53521). To get the
golden numbers, use the abliterated model + `spark-launch.sh`.

## 6. Current status / what the golden settles

- Real NVFP4 on 0731 = the golden image (adopted 2026-08-24). Porting the
  NVFP4 writer into our v0.28.0 stack is **not** needed for the speed goal;
  the b12x `nvfp4_ds_mla` 584-byte alias stays for our image.
- The golden's DeepGEMM pin (`a6b593d`) is the SM12x-fp8-known-good; our
  DeepGEMM #419 PR restores the removed pure-fp8 1d1d kernels (see
  [09-golden-deepgemm.md](09-golden-deepgemm.md)).
- Rollback from golden: `bash ~/spark-launch.sh eugr-prod.yaml` (v0.27.x,
  -13% capacity, -45% c6).
- SSD offload does NOT work for this model on any lineage (hybrid
  multi-group cache vs single-flat-layout offload assumption) — see the
  field note §7.
