[← Index](00-index.md) · [Glossary](glossary.md)

# Performance

> **Scope:** Measured throughput, root causes, levers, benchmark methodology, and how the golden image compares.

All numbers measured on the live 2× GB10 cluster unless noted. Harness: greedy decode, 128 output tokens; aggregate = `N×128 / wall_time`.

---

## Current State (2026-08-26, re-bench after cluster restore)

| Metric | Value | Note |
|--------|-------|------|
| Single-stream (c1) | **26.3 tok/s** | France prompt, temp 0 (steady run; 21.5 on cold run) |
| Aggregate c8 | 73.4 tok/s | |
| Aggregate c16 | 115.3 tok/s | |
| Aggregate c32 | **174.7 tok/s** | best aggregated on matched-main (was ~172) |
| France quality | **green** | `' Paris…'` coherent, n_tie=1 |
| KV pool | 97,737 tokens | nvfp4_ds_mla 584 B/layer/token (35,624 B/token whole-model) |

**Throughput ceiling evidence (300+ unattainable on this hardware with
current stacks):** the golden image (fastest known lineage, real NVFP4
writer, `max_num_seqs=6`) measured c1 65.2 / c6 216.8 on 2026-08-24 — its
design-point cap is ~220 aggregated. Matched-main tops at c32 **174.7**.
No stack on this 2×GB10 reaches 300 aggregated with the
`scripts/bench-concurrency.py` harness. Re-boot of the golden via
`05-serve.sh golden` fails (2026-08-26): (a) the serve-script path feeds the
**vanilla** checkpoint while the golden image's tested recipe is the
**abliterated** model (`drowzeys/keys-…-Abliterated-32-32`, deployed via
`spark-launch.sh anemll-nvfp4.yaml`, see `docs/field-notes/dgx-spark/GOLDEN.md`);
(b) the golden's vLLM 0.25.2 dies on empty `VLLM_USE_B12X_MOE` (`int('')` —
serve script passes it empty unless pinned; `pin.golden.env` now pins the
B12X_* vars to 0); (c) with the vanilla checkpoint it hits the fp8_einsum
`layout.hpp:97` scale assert at `profile_run` — the same ue8m0 recipe family
fixed for vLLM main in #53521.

---

## Golden image (anemll) measured on this cluster — 2026-08-24

Deployed `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (stock, zero patches) with the
abliterated checkpoint, `nvfp4_ds_mla` (real NVFP4 writer), util 0.82,
`max_num_seqs=6`, DSpark k=5, capture 32 (recipe asked 36; vLLM truncates).
KV pool **2,047,170 tokens** (GOLDEN.md range 1.97–2.0M; 7,650 B/token).

France greedy (temp 0, 32 tok): `' Paris. The capital of Spain is Madrid…'`
— same string as main-b12x. The abliterated checkpoint does not change it.

| Harness | Level | Golden (anemll) | main-b12x | Ratio |
|---------|-------|-----------------|-----------|-------|
| France temp 0, 128 tok | c1 | **65.2** | ~25.8 | **2.5×** |
| France temp 0, 128 tok | c6 | **216.8** | ~95 (c8) | — |
| France temp 0, 128 tok | c16 / c32 | 183.9 / 186.2 | ~116 / ~172 | 1.08× @ c32 |
| BST coding, temp 0.7, 128 tok | c1 | 54.6–58.1 (GOLDEN 51.4) | — | — |
| BST coding, temp 0.7, 128 tok | c3 | 108.5 (GOLDEN 112.7) | — | — |
| BST coding, temp 0.7, 128 tok | c5 | 124.2 (GOLDEN 126.2) | — | — |
| BST coding, temp 0.7, 128 tok | c6 | 155.0 (GOLDEN 157.9) | — | — |

Readings:

- The golden reproduces the GOLDEN.md numbers on this hardware — the deploy
  path (sparkrun + recipe + abliterated checkpoint) is validated.
- The 2.5× c1 gap and the c6 aggregate lead are **not** the attention backend
  (our A/B was parity): it is the whole stack — vLLM v0.25.2 core + their
  kernels + real NVFP4 KV (different cache geometry/memory traffic).
- The golden plateaus past c6 by design (`max_num_seqs=6`; requests queue,
  c32 per-stream 5.8 tok/s). It is tuned for 5 clients, not batch throughput.
- KV capacity: 2.05M tokens @ 7,650 B/token vs our 97.7K @ 584 B fp8 alias —
  the real NVFP4 writer is the only route to both more tokens AND less
  memory traffic per token.

**Adopted speed conclusion:** serve with the golden image for max speed +
real NVFP4; the matched-main image stays the upstream/PR track.

---

## Root Causes (Ranked by Impact)

### 1. `max_num_seqs` Capped Aggregate Throughput **(FIXED)**

The shipped config had `MAX_NUM_SEQS=8`, capping aggregate at ~88 tok/s.

| max_num_seqs | Peak Aggregate | At |
|--------------|----------------|-----|
| 6 | 159 tok/s | c6 |
| 16 | 293 tok/s | c16 |
| 32 | 421 tok/s | c32 |
| 48 | hangs | boot ceiling |

**Fix:** raise to 32 (done, verified ~172 tok/s @ c32, France green). The main-b12x reaches ~172, not 421, because of cause #2.

---

### 2. The ~2× gap vs anemll/eugr is whole-stack, NOT the attention backend

The three lineages in `GOLDEN.md` (one harness, coding prompt, temp 0.7):

| Lineage | c1 | c5 | c6 | KV B/token |
|---------|-----|-----|-----|------------|
| anemll (FlashInfer) | 51.4 | 126.2 | 157.9 | 7650 |
| eugr (b12x) | 54.3 | 127.4 | ~109 | 11317 |
| stage-c (tonyd2wild) | 56.1 | 116.0 | 141.1 | ~11900 |
| **main-b12x (this)** | **~25.8** | — | **~95 (c8)** | **584 B DSV4 page (fp8 alias)** |

Our own A/B settled the attention-backend question:
`FLASHINFER_MLA_SPARSE_DSV4` vs `B12X_MLA_SPARSE` on this image = **parity**
(c32 179 vs 172 tok/s; c1 22.6 vs ~25.8 — FlashInfer slightly worse
single-stream). So the attention backend is NOT the 2×. The gap is the
**whole stack**: anemll runs vLLM v0.25.2 (Jul) + their kernels + a real
NVFP4 writer; eugr runs 0.27.x + their b12x/FlashInfer builds; stage-c runs
0.21.1rc1. Matched-main `e25c586b9` + b12x 1.2.6 + our overlays is a
different engine — newer vLLM core, but without their kernel maturity and
without real NVFP4 KV (which changes cache geometry and memory traffic).

#### Real remaining sub-gaps on this image (all smaller than the stack gap)

1. **Packed-at-store paged indexer 1-way gap** (~14%): the sparse indexer K
   is stored K-then-scale packed. The b12x paged kernel reads it correctly;
   the interleaved FlashInfer gather was numerically wrong (FIXED:
   `packed_gather_mqa_logits`, 0.0-diff unit test). 1-way ≥30.6 with the
   paged indexer on is still an open perf gap.

2. **WO-projection (o_proj)**: torch.bmm after fused inv-RoPE dequant
   (MXFP8 `wo_proj.run()` France-loops). ~0.25 ms/layer.

3. **MQA logits prefill fallback**: PyTorch dequant+ReLU instead of DeepGEMM
   (SM12x unsupported on DeepGEMM main).

4. **MoE ~2.2 ms/layer, attention ~1.0 ms/layer** at 192 tok/step — GPU 96%
   util at 26 W says the kernels are latency-bound, not memory-bound.

---

### 2b. NCCL was running on TCP, not RoCE — the 200G fabric was idle **(FIXED 2026-08-28)**

The single biggest sub-gap found in the v0.28.0 stack, and it applies to
every image lineage (any container with the same rdma-core).

**Symptom**: 44% SM util at c5, ~67 ms/step at c1, aggregate degrading past
~8 concurrent — classic sync-overhead signature, not compute-bound.

**Diagnosis chain**:
- `nvidia-smi dmon`: SM util ~44% during a c5 burst (not saturated).
- NCCL all_reduce microbench: 40 KB = **0.371 ms**, 128 MB = **3.9 GB/s** —
  only ~16% of the fabric's capability.
- `NCCL_DEBUG=INFO`: `NET/IB: No device found` + `GPU Direct RDMA
  Disabled for HCA ...` → **NCCL fell back to TCP sockets** over the CPU
  NIC; the 200 Gbps RoCE link (active, `ethtool` 200000Mb/s) was idle.
- Root cause: the container's `libmlx5.so.1` (rdma-core 50.0 /
  libmlx5 1.24.50.0, Ubuntu 24.04) lacks the `mlx5dv_reg_dmabuf_mr` /
  `mlx5dv_get_data_direct_sysfs_path` symbols (MLX5_1.25+, added in
  rdma-core v54). **NCCL 2.30.7 requires those symbols for IB device
  detection** → sees zero IB devices → TCP fallback. `ibv_devinfo` still
  enumerates the devices (libibverbs itself works); only NCCL's probe fails.
- Why the golden isn't affected: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
  ships Ubuntu 22.04 + **NCCL 2.28.9** + libmlx5 1.22 — an OLDER NCCL that
  predates the mlx5dv-symbol requirement, so basic RoCE works for it.

**Fix**: build rdma-core **v54.0** from source (`linux-rdma/rdma-core`
tag v54.0, CMake, `cmake --install --prefix /usr`) and replace
`/usr/lib/aarch64-linux-gnu/libmlx5.so.1` (symlink → 1.25.54.0, `ldconfig`).
Deployed as overlay image `main-b12x-028-rdma`.

**Measured (before → after, 2-node, same harness)**:
| Size | TCP (old libmlx5) | RoCE (v54) | Δ |
|------|------|------|------|
| 40 KB all_reduce | 0.371 ms | **0.034 ms** | 10.9× |
| 320 KB | 1.068 ms | **0.213 ms** | 5× |
| 128 MB | 3.9 GB/s | **39.3 GB/s** | 10× |

Decode does ~86 all_reduces/step (2/layer × 43); the sync cost dropped from
~32 ms to ~3 ms per step.

**End-to-end measured (v0.28.0 stack, golden methodology, shared coding
prompt, natural EOS, temp 0.7) — before (TCP) -> after (RoCE)**:
| concurrency | before | after |
|------|------|------|
| c1 | 17.2 | **29.8** |
| c3 | 32.4 | **51.6** |
| c5 | 60.4 | **83.9** |
| c6 | 49.6 | **85.1** |
| c8 | — | **108.0** |
| worst-case bench c8 | 63.0 | **94.8** |

Gap vs golden (c5 141, c6 157.9) narrowed ~2.3x -> ~1.7x. The remaining
saturation/stall at higher concurrency is JIT-compiles during inference
(mHC TileLang kernels + DSpark gumbel sampler compile on first use for new
shapes - warmup gaps) and per-layer kernel overhead (SM util still ~47%).

**NCCL env caveat**: with the v54 libmlx5, forcing `NCCL_IB_GID_INDEX=3` /
`NCCL_NET_GDR_LEVEL=PHB` / `NCCL_IB_HCA=mlx5` re-breaks the path (0.37ms);
leave them unset (defaults auto-select rocep1s0f1 + roceP2p1s0f1 at
0.031-0.045ms). `configs/env.spark.sh` updated accordingly.

**Implication for all images**: any vLLM container on these nodes built on
Ubuntu 24.04 with rdma-core 50.0 + NCCL ≥2.29 runs its TP traffic over TCP.
Verify with `NCCL_DEBUG=INFO` (look for `NET/IB`) or the all_reduce
microbench; fix with the v54 libmlx5 overlay.

### 3. CUDA Graph Capture Overhead

| Mode | Capture Sizes | Status |
|------|---------------|--------|
| PIECEWISE | 11/11 | ✅ working |
| FULL | 7/7 | ✅ working |
| TP AR | in-graph | ✅ working |
| DSpark backbone | FULL | sample eager (not graphed) |

**Do not** add capture size 6. **Do not** graph DSpark `_sample_sequential` (shared `lm_head`). **Do not** feed 1-row scheduled scorer into 8-row decode.

---

## Levers (Ranked by Impact)

| Lever | Current | Range Tested | Impact |
|-------|---------|--------------|--------|
| `max_num_seqs` | 32 | 6→48 | **Huge** (88→172→421 theoretical) |
| `GPU_MEMORY_UTILIZATION` | 0.8 | 0.8→0.85 | KV capacity, not speed |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | 192 | 6→192 | Minor |
| `BLOCK_SIZE` | 256 | 128/256 | Page table depth |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | FULL/PIECEWISE | Stability vs overhead |

---

## Benchmark Commands

```bash
# Quality gate (must pass)
VALIDATE_STACK=main ./scripts/06-validate.sh

# 1-way and 8-way decode throughput benchmark (run on spark1)
python3 scripts/bench.py

# Concurrency sweep across levels 1, 6, 16, 32 (completions harness)
python3 scripts/bench-concurrency.py --levels 1 6 16 32

# Chat harness mirroring golden BST coding prompt
python3 scripts/bench-concurrency.py --chat --max-tokens 128 --levels 1 3 5 6

# Single-stream latency
curl -X POST http://10.0.1.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","prompt":"The capital of France is","max_tokens":128,"temperature":0}' \
  -w "\nTotal: %{time_total}s\n"
```

---

## Benchmark Methodology (external lab corpus)

The Korean lab corpus's benchmarking chapter (`book/05-*`) — a discipline
checklist for measuring Spark servers, distilled with provenance tags:
**[MEASURED]** = run on real hardware; **[CLAIMED]** = recipe/forum-reported;
**[QUOTED]** = official vendor/model-card docs.


- **Metric taxonomy**: `prefill` (input-prompt speed) and `decode` (output-token
  speed) are NOT interchangeable — end-to-end feel changes when context/output
  lengths change. `TTFT`, `single-stream`, `aggregate throughput` and
  `latency` are separate measurements; never rank by a single tok/s.
- **Verification ladder**: `L0 loaded → L1 serves → L2 generates → L3 decode/prefill → L4 concurrency/long-context → L5 tool loop → L6 soak`.
  Failing one stage does not fail all stages; passing L0 does not imply L5.
  Do not produce higher-level numbers while a lower level failed — a loaded
  model is not a license to run agent benchmarks.
- **Level detail**: L2 single-stream = 3 warmups, ≥5 trials with the same
  prompt/output budget, first-request vs steady-state shown separately; L3
  concurrency = c1/c2/c4/c8 as separate rows (never rewrite "c8 aggregate
  180 tok/s" as a single-user rate); L4 long context = 8K→32K→128K→256K→recipe
  ceiling with needle position + exact recall + memory peak; L5 tool/agent =
  separate from raw generation; L6 soak = repeated short requests + periodic
  long prompts + c1/c4 mix + memory/temperature/health polling.
- **Report schema** (required fields): `date, hardware, model_revision, quant, runtime_commit, context, kv_dtype, speculative, concurrency, prompt_tokens, output_tokens, prefill_tok_s, decode_tok_s, latency_p50, latency_p95, quality_gate, status`. `status` accepts only actually-passed
  values: `loaded` / `serves` / `benchmarked` / `tool-tested` / `agent-tested`.
  Prefill and decode never merge into one number; official numbers and local
  measurements stay in separate rows; clock cap, power mode and temperature
  are never hidden. Save a results bundle per model:
  `<date>-<model>-<hardware>-<profile>.json` + raw logs + env snapshot.
- **Failures are results too**: model loads but repeats characters, OOM at the
  context ceiling, parser 400, KV pool exhaustion at c4, socket fallback
  instead of RDMA, NCCL hang, shutdown after temperature rise — record them
  with the reproduction conditions; if the root cause is unconfirmed, mark
  `issue`/`unresolved`, never write a temporary workaround as the official fix.
- **Clock-cap A/B protocol**: measure stock at sweep start AND end (thermal/
  power drift); apply the same cap on all nodes; keep c1 and c4, decode and
  cold prefill as separate rows; store GPU-rail (`nvidia-smi`/`nvtop`) and
  wall-AC values in separate fields; do not generalize 2200 MHz as optimal for
  all models/temps/concurrencies.
- **Quality gate**: speed and quality live in separate columns of the same
  table; a speed gain that fails the quality gate is NOT adopted as a
  production profile (and a small speed gain may still improve agent tail
  latency — don't switch it off on one tok/s number alone). Temperature 0 is
  not fully deterministic — record seed and repetition count.

## Single-Spark DeepSeek-V4: Measured vs Claimed

**Profile this section refers to**: 1× GB10/SM121, TP=1, DeepSeek V4 Flash 0731
via the MiaAI-Lab one-Spark recipe — EXL3 **3.0 bpw** + REAP-K216 (216 of 256
experts), SparkInfer, DSpark K5/K64 draft, native NVFP4 KV, `MAX_MODEL_LEN=384000`,
`MAX_NUM_SEQS=1`, `GPU_MEMORY_UTILIZATION=0.94`.


**[MEASURED]** on this repo's single Spark (2026-08-21 recipe run + 2026-08-22
C1 re-run, image digest `sha256:2e077489…`, commit `d1dc9e7`):

| Item | Value | Note |
|------|-------|------|
| Non-streaming decode (3 runs, wall clock) | 29.86 / 32.43 / 31.74 → **31.34 tok/s mean** | email-validation prompt, `thinking=false`, 256 tokens — end-to-end wall clock, NOT raw decode-kernel speed |
| C1 code decode (gate 35) | min 37.553 / median **41.358** / mean 40.915 tok/s | semantic, JSON-schema and 5-language code gates PASS |
| Cold prefill | **985.377 tok/s** | 251,968-token real request — **FAILED the 1,000 tok/s gate** |
| Token-count gate | FAIL | 251,968 actual vs 252,047 target — 79-token `/tokenize` vs chat-usage drift (`thinking=false` applied differently) |
| Tool call | `lookup_weather` → `{"city":"서울"}` | one mock multi-turn loop passed — parser plumbing only, not real tool execution |
| KV pool (one boot) | 469,175 tokens | boot-dependent, not a fixed spec |

**[CLAIMED]** (recipe/community, conditions stated — treat as reported values,
not a guarantee): structured decode **44–47 tok/s** (MiaAI README, 384K
setting); KV pool ~439,622 tokens (one cold boot); 370,104-token needle exact
recall (thinking off, temp 0 — a capacity stress test, not general long-doc
quality certification); prefill ~1,024 tok/s early, ~350–614 tok/s past 300K,
effective ~625 tok/s (one 370K request ≈ 10 minutes end-to-end). Other
reproductions report **23–37 tok/s** (emiluzelac); Y-Computer IQ3M 16.93 tok/s
target vs 28.29 tok/s sidecar speculative; tpurtell EXL3 K2 c1 24.7 / c4
aggregate 31.7 — evidence that 47 tok/s is a top, not the expected value.
**The public 44–47 tok/s was NOT reproduced here; the current record for this
machine/setup is ≈31 tok/s.**

**How to read the numbers**: `44–47` is one combination (structured decode +
`start.sh` + 384K + c1 fresh boot, thinking-off only in the needle stress
test). Never merge prefill, decode, long-context stress, or c4/c8 into one
always-on number; the single-Spark EXL3 artifact is NOT the official full-FP8/
full-expert checkpoint and must never be recorded under the same model profile.

## Multi-Spark Scaling Numbers (claimed, cross-harness)

All figures below are **[CLAIMED]** community/recipe measurements on
non-identical harnesses — evidence of ranges and conditions, NOT a ranking,
and NOT directly comparable to this repo's own 2×GB10 tables (different
checkpoint/quant/engine/context/concurrency). DeepSeek V4 Flash 0731, TP=2,
unless noted.


| Source / profile | Numbers |
|---|---|
| MiaAI 2-Spark (1M setting, FP8) | c1 by prompt length: 2K 68.8 / 8K 73.9 / 32K 64.0 / 128K 65.2 tok/s; 256-token c1 75.4 / agg 69.1; 256-token c6 36.9 per-request / agg **191.2**; ~900K acceptance ≈875 prefill tok/s |
| Weschera (40K fixture, DSpark K7) | spec off 27.103 → DSpark K7 83.808 tok/s; draft acceptance 84.57% (SparkBench TrueScore 87.8 was the 40K·K7 profile, NOT the 1M default) |
| tonyd2wild (1M profile, NVFP4 KV) | c1 61.0; aggregate c2 91.7 / c4 151.1 / c6 197.3; 100K prefill 2,639 tok/s |
| m9e production-shaped (RoCEv2, 1M, 16 slots) | sampled/max coding C1 median 52.03; C6 aggregate 132.18; C16 aggregate 211.38; cold 131K prefill 1,928.93 tok/s |
| Nacyot (2× ASUS GX10, original FP8) | single decode ~56.1 @ 256-token context → ~45.9 @ 512K; 12 concurrent × 256 aggregate 206.9; measured CX-7 dual-rail 196 Gb/s |
| Reddit ASUS vs NVIDIA (same 0731 TP=2 stack) | 515K cold retrieval 3/3 exact recall both; 515K prefill ASUS 1,450.81 vs NVIDIA 1,113.93 tok/s; 6-worker stream ASUS 105.21 vs NVIDIA 95.50 tok/s — host/firmware/thermal state changes results |
| 4× Spark TP=4 | DeepSeek single 49.4, c8 aggregate 180 tok/s (vLLM fork, NCCL 2.30.4, 200G RoCE, FP8 KV, 384K) |

**Synthesized realistic ranges** (as of 2026-08-22): 1-node single response
~20–47 tok/s by recipe/workload (44–47 is a possible top, not the expected
value); 1-node aggregate with short requests ~50–60 tok/s total (per-request
can drop to ~5); 2-node single response 60–95 tok/s on 40K–8K short fixtures,
lower at ≥32K and long reasoning; 2-node aggregate 100–340 tok/s at higher
concurrency — that is the sum across responses, not per-user speed. Adding
nodes does not raise single-stream performance; only aggregate can grow, and
aggregate growth can worsen per-user TTFT — report both together.

---


## External reference data (linked recipes/papers, 2026-08-26)

- **DSpark acceptance is content-dependent, 2.1× range [RECIPE — NTillmann]**: same healthy tier, same minute — **80.6 tok/s @ 94.5% acceptance vs 37.6 tok/s @ 34.4%**. Throughput claims without the prompt set are meaningless.
- **Per-position acceptance on real agent content [RECIPE — AlexLJC]**: 86.0 / 69.9 / 55.4 / 43.0 / 33.2% vs 72.4 / 47.5 / 28.9 / 16.0 / 9.4% synthetic — k=5 suits agent content, k=2 suits free-form synthetic; greedy draft sampling rejected on tool-calling.
- **Decode latency model [RECIPE — AlexLJC, anemll stack]**: `T_forward(N) = 82.5 + 22.94·N ms` (fixed weight-read + per-seq attention/MTP), aggregate asymptote ~170 tok/s; TTFT floor 0.29 s idle; fresh prefill 1,739 tok/s; 12 seats = 120.67 tok/s aggregate, 13.90/stream.
- **Clock cap is free for decode [RECIPE — agjs]**: 2200 MHz → −1.0% decode, −36% GPU power (see [01-hardware](01-hardware.md)).
- **Single-Spark EXL3 measured [RECIPE — MiaAI one-Spark]**: decode **44–47 tok/s** (structured, 384K ctx), effective prefill ~625–630 tok/s end-to-end at 320k–370k prompts, KV pool 439,622 (native 432 B NVFP4 records, util 0.94).
- **2-node DDP precedent [PAPER — arXiv:2608.07226]**: NanoChat pretraining over 200 Gb/s QSFP56 direct fiber + Tailscale: 69.4 s/step, ~1,890 tokens/s (131k-token global batch) — a reference for distributed runs on this pair.
- **Official NVIDIA inference figures [OFFICIAL — developer blog]**: at ISL|OSL 2048|128, BS=1 — Qwen3 14B NVFP4/TRT-LLM 5,929 prefill / 22.7 gen; GPT-OSS-20B MXFP4/llama.cpp 3,670 / 82.7; GPT-OSS-120B MXFP4/llama.cpp 1,725 / 55.4; Llama 3.1 8B NVFP4/TRT-LLM 10,257 / 38.7; Qwen2.5-VL-7B NVFP4 65,832 / 41.7; Qwen3 235B NVFP4 dual-Spark 23,477 / 11.7 tok/s. Agent 128K|1K: Nemotron 3 Super 120B NVFP4/TRT-LLM 2,855 prefill / 18 gen / 99 s e2e; Qwen3.5 35B-A3B FP8/vLLM 3,080 / 35.75 / 73 s; Qwen3 Coder Next 80B FP8/vLLM 2,390 / 28.95 / 89 s (1→4 concurrent: prefill 3,261 → 9,616 tok/s). Official fine-tune (seq 2048, batch 8): full FT Llama 3.2 3B 82,739; LoRA 8B 53,658; QLoRA 70B 5,079 tok/s — none fit a 32 GB consumer GPU.
- **TP scaling & spec-dec [OFFICIAL]**: TPOT ~2× @TP2, ~4× @TP4 (Llama 3.3 70B NVFP4, 32K|1K — TTFT 33,415/21,384/15,552 ms; TPOT 269/133/72 ms); Qwen 235B dual-Spark NVFP4 + spec dec up to **2.6× vs FP8** (NVFP4 −40% memory); llama.cpp MoE updates ≈ +35% on Spark.
- **DSpark headroom vs MTP [PAPER — arXiv:2607.05147]**: DSpark (confidence-scheduled, semi-autoregressive draft, adaptive verification length) gives **+60–85% per-user generation speed vs MTP-1** at matched throughput in V4 production — the design headroom behind the k=5 flag in every V4 recipe here.
- **Single-GB10 0731 measured [RECIPE]**: lrozewicz vLLM-Moet (2-bit planes + FP4 delta tier) — **~21 tok/s code decode (DSpark k=2, 81% acceptance)**, ~7.7–8 spec-off, **753 tok/s prefill**, 256K ctx at ~117/121.6 GiB; model-card k=7 collapses to 5.1 tok/s. 0xSero SparkInfer EXL3: C1 code median 38.12 / mean 39.49 tok/s, cold 252K prefill 1,055 tok/s. tpurtell K2: C1 median 52.47 (acceptance 63.81%), 252K prefill 1,058.8; cross-hardware 1×Spark 24.7 (c1) / 31.7 (c4) vs 1×RTX PRO 6000 133.6 / 189.8 tok/s. emiluzelac (ds4 engine): ~1,000 tok/s prefill reproduced (993.6 @ 127,532-token needle, TTFT 128.4 s), 23–37 single-stream, 59.7 aggregate @ 12 concurrent (~5 tok/s each).
- **ds4 engine sweep [RECIPE — antirez]**: prefill 825.76 @ 2048 → 822.98 @ 64K tok/s; generation 18.05 → 13.84 tok/s; DSpark via separate ~5.6 GiB 0731 support GGUF (confidence gate 0.7); `--power N` throttles GPU for heat/noise.
- **Community bench repos [COMMUNITY]**: sparkbench.dev leaderboard (PBM 4k-decode, real hardware): Qwen3.6-35B-A3B NVFP4+MTP 86.3; Qwen3-30B-A3B 74.2; Ornith 1.5 35B MTP 70.8; DS4F-0731 (EXL3 13B/180B) 23.4; Qwen3.8-27B DSpark NVFP4 22.0. warpcore (ALCF GB10 cards): gpt-oss-120b MXFP4 ~709; Lightning-30B ~719; Qwen3.5-122B-A10B-int4 26.9 single / 228 agg. dgx_spark_benchy: Qwen3.6-35B-A3B-NVFP4 LocalScore 92.0; Qwen3.6-27B-NVFP4 Hermes 82.1.
- **Engine-side comparisons [COMMUNITY]**: 67ailab Qwen3.8-27B llama.cpp — `spec-type draft-mtp` (not the `--mtp` download flag): 10.9 → **28.9 tok/s (2.7×)**, acceptance 81%; older llama.cpp dies with `missing tensor 'blk.64.ssm_conv1d.weight'` (MTP block miscounted as layer 65). MiaAI SGLang 3-engine on NVFP4 Qwen3.8-27B: code 51.5 (DSpark) / 34.5 (MTP) / 50.9 (DFlash2) — SSE event-counting under-reads ~4× (count `completion_tokens`). 0xBakeer Qwen3.8-27B: DSpark k=14 edit-heavy 58.5 / k=7 46.8; NVFP4-vs-FP8 edge collapses with concurrency (+27% c1 → +10% c8 → +0.2% c16). PTT: SGLang+NVFP4+DFlash2 50 tok/s @ 40 W vs llama.cpp Vulkan 10 tok/s @ 80 W; CUDA sm_121 build is 2–3× Vulkan.
- **Ollama-path usability [COMMUNITY — Kleybrink dgx-spark-bench]**: 30B MoE 69 tok/s vs 27B dense 11 tok/s; thinking transforms quality (Nemotron Cascade 2 30B 44% → 100%); cloud leaderboard inverts locally (Qwen3.5:27b AA-42 → rank 38/39).

## Related Docs

- [02-model.md](02-model.md) — DSpark acceptance impact on throughput
- [03-kernels-attention.md](03-kernels-attention.md) — Kernel differences causing the gap
- [04-quantization-kv.md](04-quantization-kv.md) — KV capacity vs util
- [06-deployment.md](06-deployment.md) — Config values for max_num_seqs, util
- [07-gotchas.md](07-gotchas.md) — "Do not raise spark2 to util 0.85"
- [08-upstream.md](08-upstream.md) — b12x kernel optimization PRs needed

### Raw evidence (field notes)

- [`../field-notes/dgx-spark/TUNING.md`](../field-notes/dgx-spark/TUNING.md) — the util→KV-pool lever, 0.85 startup cliff, k>=5 constraint
- [`../field-notes/dgx-spark/GOLDEN.md`](../field-notes/dgx-spark/GOLDEN.md) — three-lineage head-to-head on one harness

---

**[← Prev](04-quantization-kv.md) · [Glossary](glossary.md) · [Next](06-deployment.md) →**
