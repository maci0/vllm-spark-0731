[← Index](00-index.md) · [Glossary](glossary.md)

# Quantization & KV Cache

> **Scope:** Weight quantization and KV cache dtypes — FP8/MXFP4/NVFP4, the 584 B DSV4 envelope, KV capacity, offload.

## Weight Quantization

| Component | Quantization | Kernel | Notes |
|-----------|--------------|--------|-------|
| **Linear** | FP8 E4M3 block-scaled | `B12xFp8BlockScaledMM` (`--linear-backend b12x`) | UE8M0 scales upcast to fp32 on SM12x (native 128×128 UE8M0 unsupported; LINEAR_BACKEND=deep_gemm fails on SM12x due to gemm.hpp:851 aliasing, see 01-hardware.md and 09-golden-deepgemm.md) |
| **MoE Routed Experts** | MXFP4 | `B12X_MXFP4_MXFP8` (`--moe-backend b12x`) | `w13_layout="w31"` (swapped w1/w3) |
| **MoE Shared Experts** | FP8 | b12x linear | |
| **NVFP4 Weights** | — | — | **Dead end**: all vendor attempts fail on this model; footprint doesn't shrink (156–168 GB). NVFP4 win is **KV**, not weights. |

### b12x Weight Prep (from `patches/files/b12x_moe.py`)

```python
# Gated weights padded to 64-row multiple
# w13_scale swizzled via swizzle_blockscale()
# w2_scale padded to 16-col multiple
# Layout: w31 (not w13) for MXFP4
```

---

## KV Cache Dtypes

| dtype | Bytes / Layer / Token | Whole-Model (61 Layers) | Notes |
|-------|-----------------------|-------------------------|-------|
| `fp8_ds_mla` (Stock vLLM main) | ~185 B | 11,317 B (~11.3 KB) | Stock DSV4 envelope (vLLM main) |
| anemll `nvfp4_ds_mla` | ~125 B | 7,650 B (~7.65 KB) | The "real" NVFP4 KV (32% cheaper than fp8) |
| `nvfp4_ds_mla` (this repo) | **584 B** | **35,624 B (~35.6 KB)** | DSV4 envelope alias (61 layers × 584 B) — same page as `fp8_ds_mla`, **not a real NVFP4 writer** |
| GLM NVFP4 | 432 / 368 B | — | `scale_format=2` — different writer, **do not mix** |

> ⚠️ **Layer-count basis**: the `Whole-Model (61 Layers)` column uses a
> 61-layer basis from early field notes, but the official `config.json` says
> `num_hidden_layers: 43` (+1 MTP = 44). Per-layer figures (584 B / ~125 B)
> are unaffected; whole-model totals need re-derivation
> (43×584 ≈ 25,112 B; 44×584 ≈ 25,696 B with MTP). See
> [02-model](02-model.md) → layer-pattern decode.

### Critical Distinction

The **584 B/layer DSV4 envelope** used here (~35.6 KB/token across 61 layers) is **not a memory win** over `fp8_ds_mla` — it **IS** the fp8 page geometry repurposed ($448\text{ B NoPE} + 128\text{ B RoPE} + 8\text{ B scale} = 584\text{ B/layer}$). The only authentic NVFP4 KV saving measured on this hardware is **anemll's 7,650 B/token total** (see `docs/NVFP4_DS_MLA_LINEAGE.md` in TRT-LLM tree).

> **Footprint Derivation Note (11,317 B vs 576 B Base Latent)**:
> In stock vLLM main's `fp8_ds_mla`, the theoretical compressed base latent vector is $576\text{ B}$ (512 FP8 latent dimensions + scale metadata). When amortized across the full 61-layer model under compressed indexer sharing, the whole-model footprint reaches **11,317 B/token** ($11,317 / 61 \approx 185.5\text{ B/layer/token}$). In contrast, this repository's discrete 584 B/layer page allocation maintains independent per-layer indexer and latent slots, resulting in $61 \times 584 = 35,624\text{ B/token}$ total aggregate footprint.

### Page Geometry Details

```python
# From vllm/models/deepseek_v4/attention.py
_DSV4_TOKEN_BYTES = 584  # nvfp4_ds_mla / fp8_ds_mla on DSV4
```

The `validate_nvfp4_kv_cache_with_mla` in vLLM 0.28 uses `startswith("nvfp4")` and would reject the DSV4 dtype; the overlay narrows that guard to exact `"nvfp4"`.

### Page Layout (from `patches/files/sm12x_b12x_kernels.py`)

```
Packed K-then-scale per 64-token page:
[page][8192 bytes K][256 bytes scale]
= 128 B K + 4 B scale per token, SEPARATED (not interleaved)
```

This layout skips the unused page64 workspace → **KV 94,516 → 97,737 tokens**.

---

## KV Capacity

| Util | KV Pool Tokens | Notes |
|------|----------------|-------|
| 0.80 | ~97,737 | Live pin (packed-at-store + skipped page64 workspace) |
| 0.82 | ~2.0M | anemll (real NVFP4 writer, different dtype) |
| 0.85 | — | ~11-min startup cliff (KV-pool quantization), earlyoom on spark2 |

**Capacity is util→KV-pool, not decode speed.** 2.5M tokens is unreachable on 2× GB10 (weights ~81 GiB + workspaces).

**2026-08-26 correction:** the golden's `nvfp4_ds_mla` uses the **same
584-byte envelope as matched-main** (golden code: "584B per token (448 NoPE +
128 RoPE + 8 fp8 scale)", "the proven 584-byte DSpark NVFP4 envelope";
matched-main overlay: `_DSV4_TOKEN_BYTES = 584`). There is **no different
NVFP4 writer to port** — the step-2 "real NVFP4 writer" item is a non-issue
at the layout level. The golden's ~2.0M vs our ~97k pool at similar util is
a runtime/weights-footprint difference (older vLLM core, leaner activation
workspace), not a kernel or dtype-layout gap. The `7,650 B/token` figure in
the golden field notes refers to the whole-model footprint including the
indexer/SWA caches, not a different per-layer dtype width.

---

## KV Offload (Experimental)

### LMCache GDS (`configs/lmcache.gds.yaml`)

```yaml
local_cpu: false
max_local_cpu_size: 0
chunk_size: 256
gds_path: /mnt/nvme/lmcache-0731
gds_buffer_size: 2048  # MiB, UMA
use_gds: true
gds_backend: cufile
```

### vLLM Native OffloadingConnector (`configs/kv-offload.native.json`)

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 268435456,
    "blocks_per_chunk": 1,
    "eviction_policy": "lru",
    "secondary_tiers": [{
      "type": "fs",
      "root_dir": "/mnt/nvme/lmcache-0731",
      "n_read_threads": 8,
      "n_write_threads": 4
    }]
  }
}
```


### External reference: Lookahead Sparse Attention (arXiv:2606.09079)

A research KV-compression route for this model family: a **Neural Memory Indexer** keeps only query-critical KV chunks — **KV footprint 13.5%** of full-context (LongBench-v2/LongMemEval/RULER, +0.6% accuracy), decode compute **0.30×** at 1M context, GPU KV 3.73 → 0.37 GB; 2.8× throughput / 2.7× concurrency on 8× H20 PD-disaggregated. A design target for any KV-offload scheme on this pair, not yet shipped in vLLM.

### External references: DSA lineage, official V4 KV layout, real-NVFP4 records (2026-08-26)

Sources: DeepSeek-V3.2 paper (arXiv:2512.02556), DeepSeek-V4 report (arXiv:2606.19348), DSpark card, fetched community KV recipes ([REFERENCES.md](../../REFERENCES.md) → papers / GitHub recipes).

- **DSA is the V4 attention baseline [PAPER — arXiv:2512.02556]**: DeepSeek Sparse Attention (V3.2) = lightning indexer (score Σ w·ReLU(q·k), H^I indexer heads) + top-k token selection, O(L²) → O(L·k), sparse stage trained at **2048 selected KV/query** (15,000 steps × 480×128K-token seqs). V4's CSA = DSA + sequence compression — the 27%/10% (Pro) and 10%/7% (Flash) FLOPs/KV-vs-V3.2 claims at 1M ctx are measured against this baseline.
- **Official V4 KV-cache layout [PAPER — arXiv:2606.19348]**: mixed storage — BF16 on RoPE dims, FP8 elsewhere (~half of BF16 size); indexer attention computed in FP4 (`use_fp4_indexer_cache`); cache = classical CSA/HCA cache in blocks of lcm(4,128) = **128 original tokens** + per-request state cache (SWA recent-128 window + uncompressed compression-tail buffer). On-disk KV for shared-prefix reuse: SWA is ≈8× the compressed volume → Full Caching / Periodic Checkpointing / recompute options.
- **Real 432-B vs padded 584-B NVFP4 record [RECIPE — 0xSero / tpurtell]**: the true NVFP4 sparse-MLA record is **432 B/token**; 0xSero's serving used a **584-byte padded FP8 record** under `nvfp4_ds_mla` (boots + passes an isolated oracle, but corrupts full-model text — disabled); tpurtell's calibrated K2 checkpoint runs the **native 432-B record** (1,183,301-token KV pool @ util 0.85). Envelope: 432 B native vs 584 B padded — the real writer is both smaller AND correct.
- **NVFP4 KV sizes elsewhere [RECIPE — kacper-daftcode/vLLM-Moet]**: NVFP4 KV = **352 B/token vs 656 B for fp8_ds_mla (+38% pool)** on GLM-5.2 — consistent with "the real NVFP4 writer is the only measured memory win" (table above).
- **LSA follow-up: FlashMemory-Deepseek-V4 [COMMUNITY — libertywing]**: working CPU-offload recall on DS-V4-Flash (sglang fork): CSA KV→CPU mirror, page-granular recall, fused remap; 1M ctx 30 conc / ~1,266 tok/s vs 11 / ~455 baseline (KMAX 96–384 pages); compressed-K chunk = **132 bytes** (128 fp8_e4m3 + fp32 scale); recall length-generalizes to 2× training ctx (trained ≤512K). Project formally suspended — ships complete; use as design reference only.

### Current Status

**SSD KV offload faults on this model under every dtype** — hybrid multi-group cache vs flat transfer path mismatch. Not yet functional.

---

## Related Docs

- [01-hardware.md](01-hardware.md) — UMA memory constraints
- [02-model.md](02-model.md) — MLA page geometry
- [03-kernels-attention.md](03-kernels-attention.md) — Packed-at-store indexer layout
- [05-performance.md](05-performance.md) — KV capacity vs throughput
- [06-deployment.md](06-deployment.md) — How to enable/disable offload
- [08-upstream.md](08-upstream.md) — Real NVFP4 writer PR needed

### Raw evidence (field notes)

- [`../field-notes/nvfp4/README.md`](../field-notes/nvfp4/README.md) — the NVFP4 MLA KV patch: 584 B envelope, alignment sites, dead ends
- [`../field-notes/nvfp4/KV_OFFLOAD_MLA.md`](../field-notes/nvfp4/KV_OFFLOAD_MLA.md) — why disk offload faults under every KV dtype
- [`../field-notes/dgx-spark/KV_CEILING.md`](../field-notes/dgx-spark/KV_CEILING.md) — why ~2.5M KV is unreachable; the util→pool ladder
- [`../field-notes/dgx-spark/PROD_C5_SSD.md`](../field-notes/dgx-spark/PROD_C5_SSD.md) — c5 SSD recipe (two claims disproven by later tests)

---

**[← Prev](03-kernels-attention.md) · [Glossary](glossary.md) · [Next](05-performance.md) →**
