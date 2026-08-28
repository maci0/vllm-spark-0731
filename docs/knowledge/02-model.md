[← Index](00-index.md) · [Glossary](glossary.md)

# Model: DeepSeek-V4-Flash-0731

> **Scope:** DeepSeek-V4-Flash-0731 architecture — MLA, MoE 256 experts, DSpark k=5, page geometry.

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


## Official model card & architecture facts (linked sources, 2026-08-26)

From the DeepSeek-V4-Flash-0731 HF card and the DeepSeek-V4 technical report (arXiv:2606.19348) ([REFERENCES.md](../../REFERENCES.md)):

- **Model sizes [PAPER]**: DeepSeek-V4-Pro = **1.6T params / 49B active**; **V4-Flash = 284B / 13B active**; both 1M-token context. V4-Flash-0731 is a later checkpoint of the Flash series. (Our 155.43 GiB safetensors / ~77.7 GiB per rank at TP=2 matches the FP8/FP8+MXFP4 weight layout.)
- **Architecture [PAPER]**: hybrid attention = **Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA)**; **Manifold-Constrained Hyper-Connections (mHC)** replace residual connections; Muon optimizer. At 1M context: V4-Pro needs **27% of single-token FLOPs and 10% of KV cache vs V3.2** — the efficiency baseline any serving claim must be measured against.
- **Card facts [OFFICIAL]**: **no Jinja chat template** — ships an `encoding/` folder (`encoding_dsv4.py`); OpenAI-format messages must be encoded/parsed with that API. `reasoning_effort` = low / high / max. License **MIT**. vLLM DSpark reference (4×GB300 node): `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' --kv-cache-dtype fp8 --block-size 256 --moe-backend deep_gemm_mega_moe`; SGLang: `--speculative-algorithm DSPARK` (no separate draft path).
- Agent evals (card, effort=max): Terminal-Bench 2.1 **82.7**; NL2Repo 54.2; Cybergym 76.7; DeepSWE 54.4; Agents' Last Exam 25.2.
- Recommended sampling: temp=1.0, top_p=0.95 (agentic), max output 384K tokens at high/max effort.

### Exact architecture configs (fetched HF config.json, 2026-08-26)

Sources: raw `config.json` files fetched from huggingface.co (deepseek-ai, Qwen, nvidia, apetersson) + the DeepSeek-V4-Flash-DSpark card ([REFERENCES.md](../../REFERENCES.md) → model cards / HF).

- **DeepSeek-V4-Flash-0731 (official config) [OFFICIAL]**: hidden **4096**, **43 layers**, **64 attn heads**, **KV heads 1 (MLA-style)**, head_dim **512**, qk_rope_head_dim 64, q_lora_rank / o_lora_rank 1024, o_groups 8. MoE: **256 routed / 6 per-tok / 1 shared**, moe_intermediate **2048**, `topk_method noaux_tc`, `scoring_func sqrtsoftplus`, routed_scaling 1.5, swiglu_limit 10, **expert_dtype fp4**. Routing: hash layers 3 (`hc_mult 4`, Sinkhorn 20 iters) + index head 64 heads × dim 128, index_topk 512. Attention: sliding_window **128**; `compress_ratios` **46 entries** — layers 0–1 = 0 (SWA), layers 2–41 alternate **4/128** (CSA4/HCA128), **layer 42 = 4 (terminal CSA)**; the 3 trailing entries = 0,0,0 — **resolved: the three MTP blocks (`mtp.0/1/2`, all sliding-window-only); `len(compress_ratios) == num_hidden_layers + 3`** (see [13-qwenseek.md](13-qwenseek.md)); compress_rope_theta 160000. RoPE theta 10000 + YaRN factor 16 (orig 65536) → **max_position_embeddings 1,048,576**; vocab **129,280** (bos 0 / eos 1). Quant: FP8 e4m3 dynamic, weight_block_size [128,128], scale_fmt ue8m0 (attn/dense); experts FP4. Checkpoint: **48 shards** (`model-00001..00048`), safetensors index `total_size` **166,878,536,440 B = 155.43 GiB** (matches the field-measured figure; the ~304 GB on-disk file sum matches the **304.2B total params incl. MTP** — the published 284B excludes MTP; see [13-qwenseek.md](13-qwenseek.md)), ships `encoding/` (no Jinja template) + `inference/` (custom inference stack — not HF transformers). *(The 584-B KV math elsewhere in this corpus counts 61 layers/token — vLLM DSV4 layer bookkeeping — vs 43 config layers; see [04-quantization-kv](04-quantization-kv.md).)*
- **0731 DSpark geometry [OFFICIAL]**: `dspark_block_size 5`, `dspark_target_layer_ids [40,41,42]`, `dspark_markov_rank 256`, `dspark_noise_token_id 128799`, `num_nextn_predict_layers 1`. The shipped `inference/config.json` additionally sets **n_mtp_layers 3** (HF config says 1) + dtype fp8 — confirmed on the repo (2026-08-26); the two configs disagree, verify which the runtime reads. `inference/config.json` mirrors all other fields (window_size 128, original_seq_len 65536, rope_factor 16, index/hc fields). Shipped `generation_config.json`: `do_sample true, temperature 1.0, top_p 1.0` (card recommends top_p 0.95 for agentic).
- **Qwen3.8-27B (config) [OFFICIAL]**: `Qwen3_5ForConditionalGeneration` omni (text+vision). Text: hidden 5120, **64 layers**, 24 heads, **4 KV heads**, head_dim 256, intermediate 17408, vocab 248320, max_pos 262144. Hybrid attention `full_attention_interval 4` → **16 full-attn + 48 linear-attn layers**; linear: conv kernel 4, 16 key heads × 128, 48 value heads × 128, `mamba_ssm_dtype float32`. RoPE theta 10M, partial_rotary 0.25, mrope [11,11,10]; MTP 1 layer. Vision: 27-layer ViT, hidden 1152, patch 16, temporal patch 2, out 5120. Checkpoint 55.59 GB bf16 (18 shards) ≈ 26.8B params.
- **Qwen3.8-27B-FP8 (config) [OFFICIAL]**: identical arch; FP8 e4m3 dynamic [128,128]; `modules_to_not_convert` = vision tower, lm_head, embed, norms, mamba/linear-attn params (A_log, conv1d, dt_bias, in_proj_*), MTP layers → stay BF16. **Repo-completeness flag**: 16.38 GB with only ~43 of 64 text-layer shards and no vision shards — verify before use.
- **MiniMax M2.7-NVFP4 (config) [OFFICIAL]**: `MiniMaxM2ForCausalLM`, hidden 3072, **62 layers**, 48 heads, 8 KV heads, head_dim 128, intermediate 1536, rotary_dim 64 (partial 0.5), vocab 200064, max_pos 196608, rope_theta 5M. MoE: **256 local experts / 8 per-tok / no shared** (shared_intermediate_size 0), sigmoid scoring, routing bias. MTP: use_mtp true, 1 layer, num_mtp_modules 3. Quant: NVFP4 weights+activations, **group_size 16, static**; **KV cache 8-bit float static**; lm_head / self_attn* / MoE gates → BF16. Checkpoint 139.92 GB / 15 shards ≈ 155B params — **too dense for pure NVFP4, verify actual tensor bit-widths**.
- **MiniMax-M3-DSpark (draft-head config) [OFFICIAL]**: `Qwen3DSparkModel` draft for the 60-layer M3: 6 layers, hidden 6144, 32 heads, 8 KV heads, head_dim 128, intermediate 12288, sliding_window 1024, max_pos 1,048,576, torch_dtype float32 (checkpoint 10.72 GB ≈ 3B params). `dflash_config`: block_size **8**, target_layer_ids [1,12,23,35,46,57], num_target_layers 60, markov_rank 256, markov_head_type vanilla, use_confidence_head true, swa 1024, mask_token_id 200063, projector_type dspark — **block 8 / 60 targets vs DeepSeek's block 5 / layers 40–42** (useful calibration for the DSpark patches here).
- **Nemotron-3 Super 120B-A12B-NVFP4 (config) [OFFICIAL]**: `NemotronHForCausalLM` hybrid: **88 layers**, hidden 4096, 32 heads, 2 KV heads, head_dim 128, intermediate 2688; mamba chunk 128, conv 4, expand 2, ssm_state 128; hybrid pattern → **8 full-attention layers**. MoE: **512 routed / 22 per-tok / 1 shared** (shared inter 5376), latent MoE `moe_latent_size 1024`, n_groups 8 / topk_group 1, routed_scaling 5.0, mlp relu2. MTP 1 layer; max_pos 262144, vocab 131072. Quant MIXED_PRECISION: FP8 (139 targets incl. shared experts + latent proj) + **NVFP4 4-bit g16** (40,961 targets: routed experts); KV cache 8-bit float static. Checkpoint 80.37 GB / 17 shards.
- **Nemotron-3.5 Lightning 30B-A3B-NVFP4 (config) [OFFICIAL]**: **52 layers**, hidden 2688, 32 heads, 2 KV heads, head_dim 128, intermediate 1856; 7 attention layers among mamba/moe; MoE **128 routed / 6 per-tok / 1 shared** (`moe_shared_expert_overlap true`, no latent), routed_scaling 2.5; MTP [attention, moe]; max_pos **1,048,576**, vocab 131072. Quant: FP8 (46 targets: mamba + attention) + **NVFP4 4-bit g16 weights-only** (5,935 expert targets — activations stay BF16); KV 8-bit float static. Checkpoint 13.26 GB / 37 shards.
- **Nemotron-3 Nano-Omni 30B-A3B-Reasoning-NVFP4 (config) [OFFICIAL]**: omni (LLM+vision+audio): LLM = NemotronH 52 layers, hidden 2688, **128 routed / 6 topk / 1 shared**, inter 1856, vocab 131072, max_pos 262144; vision **C-RADIOv2-H** (radio_v2.5-h, vit_huge_patch16_224, 768×768 pref, max 2048); audio parakeet ASR 24 layers, hidden 1024, 8 heads, 16 kHz, 128 mel; max_sequence_length 131072. Quant: FP8 (98 targets) + NVFP4 g16 (5,888 expert targets); KV 8-bit float static. Checkpoint 22.43 GB / 3 shards.
- **Abliterated FP8 variant [COMMUNITY — apetersson]**: config byte-identical to official; checkpoint 76.22 GB / 22 shards vs official **48 shards / 155.43 GiB (index total)** — drop-in for the official config; both repos' shard naming is stale (48-shard naming, ~half actual shards) — download-time gotcha.

### Layer-pattern decode & config dialects (from the Shallowseek brief, 2026-08-26)

- **`compress_ratios` semantics**: `0` = sliding-window-only (SWA), `4` = CSA, `128` = HCA. Over 43 layers: **2 SWA (0–1), 21 CSA, 20 HCA**, strictly alternating from layer 2 through 41, terminating on CSA at 42. The list has **46 entries** for 43+1 layers — the 3 trailing zeros exceed one MTP layer; derive the length empirically, never from a formula. **Verified against the shipped `model.safetensors.index.json`**: per-layer weight counts are 1565 (SWA) / 1576 (CSA — adds indexer) / 1569 (HCA), and layers 0–1 are 1565, evens 1576, odds 1569 — matching the decode exactly.
- **Faithful depth shrink** keeps the shape: `[0,0,4,128,4,128,4]`. Precedent: NVIDIA NeMo AutoModel bringup used `[0,0,4,128]` + `num_hash_layers 2`, validated by per-tensor dump bisection (0.998 final-logits cosine, every block ≥ 0.987) against DeepSeek's official reference.
- **Two config dialects** for the same architecture: native (DeepSeek/vLLM/SGLang — `compress_ratios` flat int list, `qk_rope_head_dim 64`, tensor names `layers.N.attn.wq_a`) vs HF transformers (`layer_types: ["sliding_attention", …]`, `compress_rates` dict, `partial_rotary_factor 64/512`, `model.layers.N.self_attn.*`). Training uses HF; serving uses native; conversion required. Map: `0 → sliding_attention`, `4 → compressed_sparse_attention`, `128 → heavily_compressed_attention`.
- **Dual RoPE**: theta 10000 for SWA layers; theta 160000 + YaRN for compressed layers (main Q/KV and the Compressor); interleaved pairs.
- **mHC formulas**: `pre = sigmoid + eps`, `post = 2·sigmoid`, `comb = softmax + eps` → col-norm-first Sinkhorn (`iters-1` alternating row/col passes) → doubly-stochastic per block.
- A tiny 1:1 debugging stand-in (Shallowseek), the hard constraints (`head_dim` must stay 512, etc.), and the differential-test workflow: [12-debug-standin.md](12-debug-standin.md).

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
