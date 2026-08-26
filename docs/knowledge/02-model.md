[← Index](00-index.md) · [Glossary](glossary.md)

# Model: DeepSeek-V4-Flash-0731

## Checkpoint Details

| Property | Value |
|----------|-------|
| **HF ID** | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| **Revision** | `main` |
| **Architecture** | `DeepseekV4ForCausalLM` |
| **Model Type** | `deepseek_v4` |
| **Size** | 155.43 GiB safetensors (~77.7 GiB/rank at TP=2) |
| **Weight Precision** | FP8 (linear), MXFP4 (MoE expert weights), FP8 (shared experts) |
| **Tokenizer/Reasoning** | `deepseek_v4` |

---

## MLA (Multi-Head Latent Attention)

### Core Mechanism

K/V are compressed to a **latent representation** (512-dim for DSV4 vs 576 for GLM). The **sparse-attention indexer** (lightning indexer) scores which KV blocks to attend:

```
score[h,m,n] = (q[m,h] · k[n]) · scale[n]
logits[m,n] = Σ_h w[m,h] · relu(score[h,m,n])
```

### Critical: ReLU Not Weighted-Q

The formula uses **ReLU(score)** — dropping ReLU selects the wrong kernel. This is the key differentiator from standard MLA.

### Page Geometry & Memory Allocation

| KV Cache Dtype | Bytes / Layer / Token | Layout & Memory Allocation Details |
|----------------|-----------------------|------------------------------------|
| `fp8_ds_mla` | 576 B | Standard unpadded vLLM MLA latent cache (512 latent + 64 RoPE) |
| `nvfp4_ds_mla` | **584 B** | DSV4 envelope: 448 B NoPE (FP8) + 128 B RoPE (64-dim BF16) + 8 B scale / token = 584 B |
| Full Model Footprint | **35,624 B** | 61 layers × 584 B = 35,624 B total per sequence token across all layers |
| GLM `nvfp4` | 432/368 B | `scale_format=2` — **incompatible layout, do not mix** |

*Note on Memory Math:* The 584-byte allocation is the per-layer, per-token memory footprint within each 64-token KV page block. Multiplying by 61 transformer layers yields the full-model sequence footprint of 35,624 bytes per token. At `GPU_MEMORY_UTILIZATION=0.80` on 2× DGX Spark (128 GiB UMA), this yields an active KV token pool of 97,737 tokens.

---

## MoE: 256 Routed Experts

- **Routed experts**: 256, weights in **MXFP4** (FP4 with block scaling)
- **Shared experts**: FP8
- **Activation**: SiLU (SwiGLU-style gating)
- **TP=2** splits experts across nodes (128 experts/node)

---

## DSpark Speculative Decoding

### Locked Configuration (from checkpoint config)

```json
{
  "dspark_block_size": 5,
  "num_nextn_predict_layers": 1,
  "compress_ratios": [0, 0, 4, 128, ...]
}
```

### Rules

1. **k=5 is locked** — `dspark_block_size=5`, `num_nextn_predict_layers=1`. k must be a multiple of `n_predict=5`.
2. **Draft = MTP head** on the same checkpoint (not a separate model).
3. **TOPK for DSV4 dispatch** = `ceil(133/64)*64 = 192` (FlashInfer #4380), not stock {128, 512, 1024, 2048}.
4. **1-way decode = 1+5=6 tokens/step**.

### Acceptance Rates (Content-Driven)

| Domain | Acceptance | Mean Accepted Tokens |
|--------|------------|---------------------|
| Code | ~65% | 4.26 |
| Math | ~28% | ~2.4 |
| General | ~45% | ~3.1 |

**Single-stream decode is acceptance-bound**, not config-tunable.

---

## Model Config Validation (from `patches/assert_0731.py`)

```python
# Must have these exact values
"architectures": ["DeepseekV4ForCausalLM"]
"model_type": "deepseek_v4"
"dspark_block_size": 5
"num_nextn_predict_layers": 1
"compress_ratios": [0, 0, 4, 128, ...]  # prefix must match
```

---

## Related Docs

- [01-hardware.md](01-hardware.md) — Why SM12x needs b12x kernels for this model
- [03-kernels-attention.md](03-kernels-attention.md) — Attention backend implementations
- [04-quantization-kv.md](04-quantization-kv.md) — KV cache dtype details
- [05-performance.md](05-performance.md) — DSpark acceptance impact on throughput
- [07-gotchas.md](07-gotchas.md) — "Do not serve without DSpark k=5"

### Raw evidence (field notes)

- [`../field-notes/dgx-spark/MODEL_VARIANTS.md`](../field-notes/dgx-spark/MODEL_VARIANTS.md) — which HF checkpoints fit this setup
- [`../field-notes/dgx-spark/CLIENT_INTEGRATION.md`](../field-notes/dgx-spark/CLIENT_INTEGRATION.md) — OpenAI-compat harness quirks (reasoning field)

---

**[← Prev](01-hardware.md) · [Glossary](glossary.md) · [Next](03-kernels-attention.md) →**
