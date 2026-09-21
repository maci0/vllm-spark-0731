# NVFP4 + DSpark on eugr B12X image — the SwiGLU-clamp adapter patch

## The problem

On the eugr `spark-vllm-b12x` image, an **NVFP4** DeepSeek-V4-Flash checkpoint (e.g.
`neko-legends/...-Abliterated-NVFP4`) cannot do DSpark speculative decoding, both paths blocked:

- `--moe-backend flashinfer_b12x` → **rejected at the oracle**:
  `Model sets swiglu_limit=10.0, but moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp.`
- `--moe-backend flashinfer_cutlass` → **won't torch.compile** the NVFP4 model →
  `AttributeError: 'GPUModelRunner' object has no attribute 'block_tables'` at spec-verify warmup.

So NVFP4 is stuck eager/no-spec. (FP8 checkpoints compile with b12x MoE natively → spec works — that's
why `apetersson` FP8 is the proven serve.) This is *not* a synthetic-vs-real-calibration issue: both
`sakamakismile` (synthetic scales) and `neko-legends` (real 2-node GB10 calibration) fail identically.

## Root cause (verified in the image)

The NVFP4 model sets a SwiGLU activation clamp (`swiglu_limit=10.0`, alpha 1.0, beta 0.0). vLLM's
NVFP4 MoE oracle (`fused_moe/oracle/nvfp4.py`) only allows backends that apply that clamp
(`NVFP4_BACKENDS_WITH_CLAMP` = TRTLLM, CUTLASS, MARLIN). `flashinfer_b12x` is excluded — **not because
its kernel can't clamp, but because the eugr adapter never passes the clamp params.**

FlashInfer's `B12xMoEWrapper.__init__` **does** accept `swiglu_alpha/beta/limit` (confirmed:
`flashinfer/fused_moe/core.py`). The eugr adapter `fused_moe/experts/flashinfer_b12x_moe.py`
constructs the wrapper **without** them → they default to None → no clamp → oracle rejects it. This is
exactly the gap neko's card describes ("the vLLM adapter did not pass the alpha, beta, and limit
arguments through").

## The patch (2 files, ~15 lines)

1. **`fused_moe/oracle/nvfp4.py`** — add `FLASHINFER_B12X` to `NVFP4_BACKENDS_WITH_CLAMP`.
2. **`fused_moe/experts/flashinfer_b12x_moe.py`** — in `__init__`, read the clamp from `quant_config`
   (`gemm1_alpha` / `gemm1_beta` / `gemm1_clamp_limit`), default alpha=1.0 beta=0.0; in
   `_ensure_wrapper`, pass `swiglu_alpha/beta/limit` to `B12xMoEWrapper(...)`.
   **Scalars, not tensors** — the FlashInfer wrapper does `float(swiglu_alpha)` internally (a
   per-expert tensor raises `only one element tensors can be converted to Python scalars`).

Patch script: `scratchpad/patch/apply_patch.py` (with the scalar fix applied to the injected code).

## Deploy (no rebuild — overlay into the prebuilt image)

Bind-mount is fragile (a `dst.py:ro` spec got mangled to `dst.pyo`); use **`docker cp` + purge the
stale `.pyc`** instead:
```bash
# per node, after `docker run ... ray start ...`:
docker cp patch/nvfp4.py            ds4:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
docker cp patch/flashinfer_b12x_moe.py ds4:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
docker exec ds4 bash -c 'rm -f .../oracle/__pycache__/nvfp4*.pyc .../experts/__pycache__/flashinfer_b12x_moe*.pyc'
# verify: docker exec ds4 grep -c _clamp_scalar <b12x path>   # -> non-zero
```
Then serve with `--moe-backend flashinfer_b12x` (+ the neko tuned profile: cudagraph 48, seqs 8,
batched 16384, util 0.87, dspark 5). Needs `HF_TOKEN` in the container (neko repo is gated).

## Remaining risk

neko also cached **one shared B12xMoEWrapper per rank** (all 43 MoE layers share identical geometry,
run serially on the CUDA stream). eugr's adapter builds a max-size wrapper **per layer** (~0.76 GiB
each ≈ 33-46 GiB extra) which can OOM at startup. This patch does **not** yet add the shared-wrapper
cache — if startup OOMs, that's the next fix (module-level wrapper cache keyed by geometry).

## Result — partial success, 3 walls peeled, blocked on a 4th

The patch (all three parts: oracle allow + scalar swiglu clamp + shared-wrapper cache) works and
peels the stack in order:

1. ✅ `flashinfer_b12x` accepted for swiglu (oracle patch).
2. ✅ swiglu clamp reaches the kernel (scalar, not tensor — FlashInfer does `float()` on it).
3. ✅ **NVFP4 now `torch.compile`s** — the `does not support torch.compile` warning is GONE and the
   log shows `torch.compile took 2.62 s`. This overturns the earlier hypothesis: NVFP4's spec block
   was **never** the compile step or synthetic-vs-real scales.
4. ✅ shared wrapper clears the memory-profiling forward (one wrapper vs 43 × ~0.76 GiB = no
   46 GiB blowup / startup timeout).
5. ❌ **Blocked at the DSpark proposer's dummy run**:
   `AttributeError: 'GPUModelRunner' object has no attribute 'block_tables'` in
   `runner._dummy_run(...) -> prepare_dummy_attn -> self.block_tables.get_dummy_block_tables`.

Wall #5 is the real remaining blocker for NVFP4 + spec, and it is **not** an MoE/adapter issue —
it's the dspark **proposer's** model-runner not having `block_tables` initialized (the FP8 path
gets it; NVFP4 does not). `prepare_dummy_attn`/`get_dummy_block_tables` are B12X-fork/model runner
code, so fixing this needs deeper spec-decode runner surgery — almost certainly one of neko's other
(undocumented) patches, not the swiglu adapter.

**Verdict:** the adapter patch is correct and valuable (it makes NVFP4 compile with `flashinfer_b12x`
on the eugr image — genuinely new), but **full NVFP4 + DSpark needs more than the adapter.** Stopped
here: `apetersson` FP8 already delivers abliterated + spec, and NVFP4 wouldn't beat it single-stream
anyway (bandwidth-bound). Keep this patch for when the proposer `block_tables` init is also fixed
(watch neko's repo / a newer eugr build / #41834 merging upstream).
