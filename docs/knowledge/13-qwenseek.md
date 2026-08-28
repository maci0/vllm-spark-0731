[← Index](00-index.md) · [Glossary](glossary.md)

# Qwenseek: Re-Homing the Qwen MoE Trunk onto V4 Attention

> **Scope:** Convert Qwen3.5-35B-A3B into the DeepSeek-V4-Flash architecture —
> transfer the MoE trunk verbatim, train the attention stack from scratch against
> the teacher. Companion to [12-debug-standin.md](12-debug-standin.md)
> (Shallowseek: tiny V4 debug fixture). Qwenseek is the production-scale transplant:
> can a real MoE trunk be re-homed onto a foreign attention family?

Source: the *Qwenseek Agent Brief* (user-measured, 2026-08-26; `.scratch/` scripts
read safetensors headers over HTTP range requests — no weights downloaded).
`[MEASURED]` = read from a primary artifact with the command recorded; `[VERIFY]` =
must be checked before reliance.

---

## Measured ground truth

**Qwen3.5-35B-A3B-Base `[MEASURED]`** (config + weight index): hidden 2048, **40
layers**, 16 heads / head_dim 256 / 2 KV heads, `full_attention_interval 4` → 30
GatedDeltaNet + 10 gated-full-attention layers, all MoE (256 experts / top-8 /
shared 512, moe_inter 512, vocab 248320, 262144 ctx, MTP 1 block). Base stores
routed experts **fused** `[E, in, out]` (transposed vs `nn.Linear`); fine-tunes may
store them **unfused** — the transfer must detect and assert, never guess.

**DSV4-Flash-0731 measured tensor shapes `[MEASURED]`** (hidden 4096 / 64 heads /
head_dim 512 / q_lora=o_lora 1024 / o_groups 8): `wq_a [1024,4096]` (F8), `wq_b
[32768,1024]`, `wkv [512,4096]` (single KV head, read as K and V), `wo_a
[8192,4096]`, `wo_b [4096,8192]`, plus **`attn.q_norm [1024]`, `attn.kv_norm
[512]`** (on the LoRA bottleneck / head_dim — omitted by the Shallowseek brief).
Compressor: CSA `wkv/wgate [1024,4096]` + `ape [4,1024]`; HCA `wkv [512,4096]` +
`ape [128,512]`, `norm [512]` both. Indexer (CSA only): `wq_b [8192,1024]`,
`weights_proj [64,4096]`, compressor `[256,4096]`. FFN: `gate.weight [256,4096]`,
`gate.bias [256]` (non-hash layers only), **`gate.tid2eid [129280,6]` I64 (hash
layers only)**, experts `E.w1/w2/w3` **FP4-packed I8** `[2048,2048]/[4096,1024]`,
**shared expert FP8** `[2048,4096]` (not FP4). mHC: `hc_attn_fn [24,16384]`
(3·m + m² with m=hc_mult), `base [24]`, `scale [3]`; `hc_head` `[4,16384]/[1]`.
Hash layers are **0, 1, 2**; compressor on all non-zero layers; indexer only on
ratio-4 layers; **all 43 layers are MoE — no dense first layer**.

## Two findings that correct the corpus

1. **There are three MTP blocks, not one `[MEASURED]`**: `mtp.0` (`main_proj
   [4096,12288]`, `main_norm`, full block, hc), `mtp.1` (full block, hc), `mtp.2`
   (full block, `norm`, `hc_head`, `confidence_head.proj [1,4352]`,
   `markov_head.markov_w1/w2 [129280,256]`). All three carry no compressor/indexer
   → all **sliding-window-only**. This **resolves the corpus's open [UNVERIFIED]**
   on `compress_ratios` (46 entries): the trailing `[0,0,0]` are the three MTP
   blocks. **Rule: `len(compress_ratios) == num_hidden_layers + 3`** — assert it,
   never recompute from `num_nextn_predict_layers` (= prediction *depth*, not block
   count). The `e_proj/h_proj/enorm/hnorm` names in the Shallowseek brief are DSV3
   MTP, not V4.
2. **The published 284B excludes MTP**: measured parameter model gives **304.2B
   total** (matches the HF on-disk file sum ~304 GB) and **284.34B** with the three
   MTP blocks removed. Shallowseek's "284.16B total" was wrong by the MTP stack.

## Transfer ledger (why this is tractable)

| Component | Params | Verdict |
|---|---|---|
| Routed experts | 31.4 B | **Yes, exactly** (transpose + permute; fused vs unfused branch) |
| Shared expert | 0.12 B | Yes, minus its gate (fold `mean(sigmoid(x@g))` into w2) |
| Router `gate.weight` | 20 M | Yes, exactly (top-k selection provably identical — softmax and sqrt(softplus) are both monotone) |
| `embed_tokens`/`lm_head` | 1.02 B | Yes — **only if the Qwen tokenizer is kept** |
| Norms | small | Yes |
| MTP FFN + norms | 2.5 B | Yes (one Qwen block copied into three) |
| Attention + compressors + indexer + mHC + DSpark heads | ~1.6 B | **No — random init** (the work) |
| Vision tower + GatedDeltaNet | 0.6 B + 30 layers | **Dropped** (no target module) |

~34 B of 37.7 B transfers; **1.41 B trained from scratch (3.7 %)** — the earlier
"7–13 B" appendix estimate was wrong. (37.70 B total / 34.97 B ex-MTP / 2.26 B
active; name "35B-A3B" is accurate on DeepSeek's convention.)

## Config decisions

- **Keep Qwen's tokenizer (248320)**, not DSV4's 129280: transfers embed+lm_head
  exactly and keeps teacher hidden states meaningful. Cost: the `deepseek_v4`
  parser flags don't apply; `dspark_noise_token_id` must be remapped to a reserved
  Qwen id `[VERIFY] which`.
- **Depth: 39 layers** — `[0,0] + [4,128]*18 + [4]` (the pattern is always odd;
  40 is even). Drop Qwen's **last** layer (nearest `lm_head`; `[VERIFY]` with a
  layer-drop ablation). `compress_ratios` = 42 entries.
- **`o_groups = 4, n_heads = 16`** (at hidden 2048 / head_dim 512): halves the
  from-scratch attention params vs the faithful 8/32 (1.41 B vs 2.39 B), and lands
  closer to Flash-0731's attention/FFN balance (62 % vs 40 % of active — Qwenseek
  is attention-heavy by construction).
- **Hard equality (new validator rule)**: `n_heads * head_dim == o_groups *
  hidden_size` — derived from `wo_a = [o_groups*o_lora, hidden]`. The Shallowseek
  brief's `% o_groups == 0` is too weak and would admit unloadable configs.
- **Free correctness wins**: fold `1/1.5` into expert `w2` (routed_scaling_factor
  stays 1.5 with identical math); warm-start the hash layers' `tid2eid` from the
  teacher's measured per-token expert picks instead of a hash (it's loaded data,
  not a required hash function — `[VERIFY]` vLLM/SGLang treat it as opaque).

## Training stages

- **A — transfer & freeze**: everything transferred is frozen; trainable = 1.41 B
  (attention/compressors/indexer/mHC/gate.bias/MTP attn). **Gate**: attention
  outputs zeroed → logits must match a same-ablated teacher (catches every
  transpose/split/scale bug before any GPU-hour).
- **B — layer-local attention distillation** (MOHAWK-style, SSM→attention
  direction): per-layer MSE against teacher's attention output on the teacher's
  input hidden state; mHC's near-identical residual streams broadcast the single
  teacher state as a warm start. **Gate**: per-layer cosine; the *worst* layer
  decides. NeMo's 0.998 parity target doesn't apply (that was identical
  architectures) — beat the stage-A ablation baseline.
- **C — end-to-end**: KL + CE; router weight frozen (preserve expert
  specialisation; `[VERIFY]` with an ablation), teacher logits precomputed
  sparsely (top-64 + tail LSE) to drop the 70 GB teacher from the loop.
- **D — heal**: unfreeze everything at low LR, short and watched.
- **E — optional QAT** on FP4 experts if post-quant drift exceeds tolerance.

## Traps

- **§9.1 Context cap is the main limitation**: the HF CSA reference OOMs past 4k
  (dense mask, O(S²·index_topk)); Qwenseek ships as a 35B model with a
  few-thousand-token usable context — state that above the fold in the card. The
  way out: rewrite CSA/HCA masks as `flex_attention` `BlockMask`s (true cost
  O(S·index_topk), few hundred lines, unlocks 32k). **Decision point after stage
  B's gate, not before** — don't build a training kernel for a transplant that
  hasn't converged.
- **Attention-heavy active compute** (62 % of active) — don't compare throughput
  against Qwen3.5 and blame vLLM.
- **Greedy loops** (V4 family) — evaluate at temp 1.0 / top_p 0.95.
- **sm_121 kernel selection** still open — log selected kernels at startup.
- FP4 packing is 2 values per int8 with `[out, in//32]` scales; the **shared
  expert stays FP8** — easy to get wrong by treating "expert" uniformly.

## Publishing

Card must state, above the fold: not affiliated with / endorsed by Qwen, Alibaba,
or DeepSeek; derivative of Qwen3.5-35B-A3B-Base with the attention stack replaced;
usable context = training context, not `max_position_embeddings`; which serving
flags were tested on which hardware; Qwen's licence governs the derivative
`[VERIFY]`. If stage B's worst-layer cosine is poor, publish the measured negative
result — it's worth more than a mediocre model.

## Key links

- [Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) — source (config + weight index read, no download)
- MOHAWK: [arXiv:2408.10189](https://arxiv.org/abs/2408.10189) · LoLCATs: [arXiv:2410.10254](https://arxiv.org/abs/2410.10254)
- [12-debug-standin.md](12-debug-standin.md) — Shallowseek (build first: de-risks format/validator/conversion at 0.85B)

---

## Related Docs

- [02-model.md](02-model.md) — DSV4 architecture ground truth; the MTP-3 finding corrects its `compress_ratios` note
- [12-debug-standin.md](12-debug-standin.md) — Shallowseek, the tiny-V4 prerequisite and format reference

---

**[← Prev](12-debug-standin.md) · [Glossary](glossary.md) · [Next](glossary.md) →**
