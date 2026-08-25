[← Back to Knowledge Index](00-index.md)

# Quantization & KV Cache

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