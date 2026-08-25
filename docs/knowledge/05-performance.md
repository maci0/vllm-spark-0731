[← Back to Knowledge Index](00-index.md)

# Performance

All numbers measured on the live 2× GB10 cluster unless noted. Harness: greedy decode, 128 output tokens; aggregate = `N×128 / wall_time`.

---

## Current State (2026-08-24, after `max_num_seqs` 8→32)

| Metric | Value | Note |
|--------|-------|------|
| Single-stream (c1) | ~25.8 tok/s | France prompt, temp 0 |
| Aggregate c8 | ~95 tok/s | was ~88 before the fix |
| Aggregate c16 | ~116 tok/s | |
| Aggregate c32 | ~172 tok/s | was capped at 8 seqs before |
| France quality | **green** | `' Paris'` logprob -0.24, n_tie=1 |
| KV pool | 97,737 tokens | nvfp4_ds_mla 584 B/layer/token (35,624 B/token whole-model) |

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

## Related Docs

- [03-kernels-attention.md](03-kernels-attention.md) — Kernel differences causing the gap
- [04-quantization-kv.md](04-quantization-kv.md) — KV capacity vs util
- [06-deployment.md](06-deployment.md) — Config values for max_num_seqs, util
- [07-gotchas.md](07-gotchas.md) — "Do not raise spark2 to util 0.85"
- [08-upstream.md](08-upstream.md) — b12x kernel optimization PRs needed