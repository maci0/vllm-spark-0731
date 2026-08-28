[← Index](00-index.md) · [Glossary](glossary.md)

# Debugging Stand-In: Shallowseek & Tiny V4 Models

> **Scope:** A small, architecturally 1:1 stand-in for DeepSeek-V4-Flash-0731 so
> vLLM's V4 inference path can be debugged in seconds instead of loading a 284B
> checkpoint — plus the exact architecture ground truth and a differential-test
> methodology.

Source: the *Shallowseek Agent Brief* (user-provided, 2026-08-26), itself built
from primary sources (the HF config, the V4 technical report, NVIDIA NeMo
bringup notes). All numbers below are either read from a primary source or
measured; anything unestablished is marked **[UNVERIFIED]**.

---

## Mission & success criteria

A tiny model, architecturally 1:1 with DeepSeek-V4-Flash-0731, that runs through
vLLM's real V4 code path so inference bugs reproduce in seconds. Secondary
contribution: no trained tiny V4 in native checkpoint format exists — publishing
one is the deliverable.

Success = loads in vLLM in native FP4+FP8 layout, dispatches every V4-specific
path (SWA / CSA / HCA layers, Lightning Indexer, mHC, hash-MoE bootstrap, MTP,
DSpark, the `deepseek_v4` tokenizer/tool-call/reasoning parsers), outputs
coherent enough that "vLLM is wrong" is visible, and ships reference logits.

Non-goals: language quality, benchmarks, long context (blocked — see traps),
and any claim of being a DeepSeek release (mandatory disclaimer, see Publishing).

## Layer pattern, decoded

`compress_ratios` (46 entries for 43 hidden + 1 MTP layer — derive the length
empirically, never from a formula): **0 = sliding-window-only (SWA), 4 = CSA,
128 = HCA**. Real shape over 43 layers: `[0, 0] + [4, 128]*20 + [4]` → 2 SWA,
21 CSA, 20 HCA, strictly alternating from layer 2 through 41, terminating on
CSA at layer 42. The 3 trailing zeros exceed one MTP layer's worth —
**Resolved (measured, [13-qwenseek.md](13-qwenseek.md))**: the extra entries are
the **three MTP blocks** (`mtp.0/1/2`, all sliding-window-only — no compressor,
no indexer). Rule: `len(compress_ratios) == num_hidden_layers + 3`. `num_nextn_predict_layers: 1` is a prediction *depth*, not a block count.

**Faithful depth shrink** keeps that exact shape: `[0, 0, 4, 128, 4, 128, 4]`
(2 SWA, alternating pairs, terminal CSA). Precedent: NVIDIA's NeMo AutoModel
bringup used a 4-layer parity harness `[0,0,4,128]`, `num_hash_layers=2`,
validated against DeepSeek's official reference by per-tensor dump bisection
(0.998 final-logits cosine, top-1 token match, every block ≥ 0.987).

## Two config dialects (critical)

| Aspect | Native (DeepSeek / vLLM / SGLang) | HF transformers |
|---|---|---|
| Layer schedule | `compress_ratios: [0,0,4,128,…]` flat int list | `layer_types: ["sliding_attention", …]` string list |
| Compression rates | same int list | `compress_rates: {"compressed_sparse_attention": 4, …}` |
| MoE schedule | `num_hash_layers: 3` | `mlp_layer_types: ["hash_moe",…]` |
| RoPE fraction | `qk_rope_head_dim: 64` | `partial_rotary_factor: 64/512` |
| Tensor names | `layers.N.attn.wq_a` | `model.layers.N.self_attn.*` |

Mapping: `0 → sliding_attention`, `4 → compressed_sparse_attention`,
`128 → heavily_compressed_attention`. Training uses the HF dialect, serving
uses native; conversion is required.

## Architecture invariants (copy exactly, never tune)

`head_dim 512` · `qk_rope_head_dim 64` · `num_key_value_heads 1` ·
`sliding_window 128` · `index_head_dim 128` · `index_n_heads 64` ·
`index_topk 512` · `hc_mult 4` · `hc_sinkhorn_iters 20` · `hc_eps 1e-6` ·
`num_hash_layers 3` · `num_nextn_predict_layers 1` · `num_experts_per_tok 6` ·
`n_shared_experts 1` · `scoring_func sqrtsoftplus` · `topk_method noaux_tc` ·
`norm_topk_prob true` · `routed_scaling_factor 1.5` · `swiglu_limit 10.0` ·
`attention_bias false` · `expert_dtype fp4` · `vocab_size 129280` ·
`rope_theta 10000` · `compress_rope_theta 160000` · YaRN (factor 16,
beta_fast 32, beta_slow 1, orig 65536) · fp8 e4m3/ue8m0 scales,
`weight_block_size [128,128]` · `dspark_block_size 5`, noise token 128799,
`dspark_markov_rank 256`.

`dspark_target_layer_ids [40,41,42]` = the **last three** of 43 layers — keep
that meaning when shrinking depth, don't keep the literal numbers.

Scalable axes: `hidden_size, num_hidden_layers, num_attention_heads,
n_routed_experts, moe_intermediate_size, q_lora_rank, o_lora_rank, o_groups`.

## SS-7 (recommended tiny config)

`hidden_size 1024` · `num_hidden_layers 7` · `num_attention_heads 16` ·
`n_routed_experts 32` · `moe_intermediate_size 512` · `q/o_lora_rank 1024` ·
`o_groups 8` · `compress_ratios [0,0,4,128,4,128,4]` — **0.85B params**,
verified to instantiate and run forward+loss in transformers 5.15.1.
Variants: F-full (depth-only, 47.15B @ 7 layers), SS-min (512/8/16/256,
0.30B).

## Native checkpoint tensor layout (vLLM / SGLang)

- `embed.weight`, `head.weight`, `norm.weight` (not `model.embed_tokens`/`lm_head`)
- `layers.N.attn.wq_a/wq_b` (q_lora split), `wkv` (single KV head, read as K and V), `wo_a/wo_b` (grouped output split), `attn_sink` (fp32, per-head)
- `layers.N.attn.compressor.*` only when `compress_ratios[N] != 0`; `indexer.*` only when `== 4`
- `layers.N.ffn.gate.tid2eid` (int64, hash layers only, shape `[vocab, num_experts_per_tok]`), `gate.bias` (fp32, non-hash only)
- `layers.N.ffn.experts.E.{w1,w2,w3}` per-expert individual tensors (not a fused 3D stack); `shared_experts.*`
- `layers.N.hc_{attn,ffn}_{fn,base,scale}` (mHC, fp32); `hc_head_{fn,base,scale}`
- `mtp.0` (`main_proj [hidden, 3*hidden]`, `main_norm`, full block), `mtp.1` (full block), `mtp.2` (full block, `norm`, `hc_head`, `confidence_head.proj`, `markov_head.markov_w1/w2`) — `e_proj`/`h_proj`/`enorm`/`hnorm` are DSV3 MTP names, not V4
- Quantized linears: `.weight` + separate `.scale`. FP8: `float8_e4m3fn` weights, `float8_e8m0fnu` scales at `[out//128, in//128]`. FP4 (routed experts): packed 2-per-byte into int8, scales at `[out, in//32]`.

**mHC formulas**: `pre = sigmoid + eps`; `post = 2·sigmoid` (no +eps); `comb = softmax(dim=-1) + eps`, then col-norm-first Sinkhorn with `iters-1` alternating row/col passes → doubly-stochastic per block.

**Dual RoPE**: theta 10000 for `compress_ratio == 0` layers; theta 160000 + YaRN for `> 0` layers (both main attention Q/KV and the Compressor). RoPE is **interleaved** pairs (view_as_complex style).

## Hard constraints (non-negotiable)

- `head_dim == 512` — SGLang's V4 KV layout fixes non-RoPE at 448 elements; with `qk_rope_head_dim 64` that forces exactly 512. Lowering breaks loading.
- `hidden_size >= 256` and `% 128 == 0` (SGLang Hopper MXFP4 Marlin pads to 256); `num_key_value_heads == 1`; `num_attention_heads * head_dim == o_groups * hidden_size` (hard equality, from `wo_a [o_groups*o_lora, hidden]` — the `% o_groups == 0` form is too weak); every quantized linear tiles into `[128,128]`; FP4 experts need `in_dim % 32 == 0`.
- `len(compress_ratios)` derived empirically and asserted.

## Prior art (tiny V4 models)

- **`yujiepan/deepseek-v4-tiny-random`** — the primary reference: derived from V4-Pro, 7 layers + 1 MTP, 0.2B, 277.3 MB, native FP4+FP8, full generation script in the card, working vLLM + SGLang commands. Author caveat: not fully tested. Tested 2026-07-22 against safetensors 0.8.0 / sglang 0.5.15.post1 / torch 2.11.0+cu129 / transformers 5.12.1 — re-derive constraints rather than assume. Its SGLang env toggles (`SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=0`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`, `SGLANG_OPT_USE_TILELANG_INDEXER=1`) reveal the separately toggleable mHC pre/post + indexer kernel paths — the vLLM equivalents are the bisection surface.
- **`silence09/DeepSeek-V4-Pro-Tiny`** — 6-layer random init, HF-oriented, second data point.
- Why random-init isn't enough: differential testing still works, but plausible-but-wrong output can't be caught without a notion of "right" — random weights through FP4 give degenerate routing/top-k that mask exactly the bugs under test. A **trained** tiny V4 in native format is the contribution.

## Workflow (phases)

0. **Reproduce with the existing model first**: serve `yujiepan/deepseek-v4-tiny-random` (vLLM: `--trust-remote-code --kv-cache-dtype fp8 --block-size 256 --tensor-parallel-size 2 --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 --speculative-config '{"method":"mtp","num_speculative_tokens":2}'`). Record vLLM version, whether real kernels select on sm_121, which flags are load-bearing. **Gate: if the bug reproduces, STOP — don't build a model to debug what an existing one reproduces.**
1. **Differential harness** (worth more than the model): per-position logits from HF (`DeepseekV4ForCausalLM`, bf16, eager — the reference) vs vLLM (native checkpoint). Report per-layer divergence via hidden-state hooks — the layer where divergence first exceeds tolerance tells you almost everything. Use transformers' `visualize_attention_masks.py` to audit masks + indexer top-k.
2. **Train SS-7**: bf16, seq 2048 (capped — see traps), plain cross-entropy, modest corpus. Train with `index_topk` lowered (128), ship config at 512 (indexer weight shapes don't depend on it). Gate: locally-coherent text, non-uniform routing, indexer entropy well below uniform.
3. **Convert → quantize → verify**: bf16 HF → native layout (§6 of the brief → chapter 02), FP4 experts + FP8 elsewhere. Round-trip logits must match within quantization tolerance — a silent conversion bug is indistinguishable from a vLLM bug.
4. **Publish**: native checkpoint, config, tokenizer, chat template (pin whatever you use — the release shipped no Jinja template), conversion/generation scripts, harness, reference logits. Mandatory above-the-fold disclaimer (see below).

## Known traps (measured)

- **The HF reference cannot train at long context**: CSA right-pads the mask by `S·k` columns; mask memory is O(S²·index_topk) — measured ~4× per doubling (256→2.79 s, 2048→10.13 s, 4096→OOM at 1.07 GB mask). Cap training at seq 2048.
- **`index_topk=512` conflicts with trainability**: at seq 2048 with compression 4 the pool is exactly 512 → CSA degenerates to dense. Train low, ship at 512.
- **Compressor warm-up distorts early-position loss** (queries before the first window close are genuinely SWA-only; the −1 sentinel is correct behavior) — exclude early positions from loss curves.
- **`head_dim=512` makes small configs top-heavy** (16×512 = 8192 before the grouped fold-back) — faithful, don't "fix" by lowering head_dim (breaks loading).
- **transformers docstring bug**: shows a non-existent `mistralai/DeepseekV4-8x7B-v0.1` — ignore.
- **[UNVERIFIED] sm_121 kernel selection**: whether vLLM's V4 path selects real kernels on GB10 or silently falls back — log the selected kernel at startup.
- **Greedy loops**: official sampling is temp 1.0 / top_p 0.95 (1.0 outside agentic); greedy has a documented looping tendency on this family — don't evaluate greedy.

## Corrections to earlier drafts (don't regress)

`routed_scaling_factor` is **1.5** (2.5 is Pro-derived) · `attention_bias` is **false** · `index_n_heads` **64** / `index_topk` **512** for Flash · keep `num_hash_layers 3` · **head_dim must be 512** · **build the MTP layer** (prime bug surface) · convert to native layout rather than shipping HF · mHC blocks don't transfer weights (alpha=0.01 init keeps them near a plain residual stream).

**Measured corrections (from [13-qwenseek.md](13-qwenseek.md), 2026-08-26)**:
`compress_ratios` length 46 = 43 layers + **3 MTP blocks** (not 2 extras; resolved
above) · **284B excludes MTP** — total is 304.2B (284.34B body) · MTP layout is
`main_proj [hidden, 3*hidden]` + `main_norm` + `norm` + confidence/markov heads on
`mtp.2`, not `e_proj/h_proj` · tensor table omits `attn.q_norm`, `attn.kv_norm`,
`ffn.gate.weight` · the width constraint is the equality `n_heads * head_dim ==
o_groups * hidden_size`.

## Publishing (disclaimer, mandatory)

Model card must state parameter count, trained-but-weak, which serving flags were tested on which hardware, and credit `yujiepan/deepseek-v4-tiny-random` as the structural basis. Above the fold, first paragraph: *not affiliated with / endorsed by / released by DeepSeek; a stand-in for testing inference stacks, not a capable model; must not be used as a proxy for V4 quality or benchmark behavior* — the failure mode to prevent is someone wiring it into CI as a stand-in for real V4.

## Key links

- Ground-truth config: [huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json) (verbatim in [02-model](02-model.md))
- V4 technical report: [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) · mHC paper: [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
- Prior art: [yujiepan/deepseek-v4-tiny-random](https://huggingface.co/yujiepan/deepseek-v4-tiny-random) · [silence09/DeepSeek-V4-Pro-Tiny](https://huggingface.co/silence09/DeepSeek-V4-Pro-Tiny)
- NeMo AutoModel bringup (4-layer parity harness): [docs.nvidia.com/nemo/automodel — deepseek-v4-flash](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/deepseek-v4-flash)
- vLLM recipe: [recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) · DeepSpec/DSpark: [github.com/deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)

---

## Related Docs

- [02-model.md](02-model.md) — the ground-truth `config.json`, layer decode, dialect mapping, invariants
- [03-kernels-attention.md](03-kernels-attention.md) — kernel paths the stand-in exercises
- [06-deployment.md](06-deployment.md) — serving recipes the harness targets

---

**[← Prev](11-cost-decision.md) · [Glossary](glossary.md) · [Next](13-qwenseek.md) →**
