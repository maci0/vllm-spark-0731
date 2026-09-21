
## 2026-09-13: guarded re-baseline and engine-level diff (supersedes the copy-family lead)

Why this section exists: no measurement in the ledger before 03:15 carries the port-holder guard,
and that guard was added because a still-running reference container silently answered a whole
series. Every pre-03:15 number is withdrawn, the `k=5 37.6 / 80.6 / 110.8 / 135.0` set included.

Both arms guarded, 3 passes at 512 tokens, levels 1 3 5 6, chat, thinking=false, seed 1234,
k=7, capture 48, `--async-scheduling`, so the images are the only free variable:

| arm | c1 | c3 | c5 | c6 |
|---|---|---|---|---|
| `refg` (anemll 0.1.1, `sparkrun_..._node_0`) | 63.1 | 112.1 | 140.9 | 156.0 |
| `protog` (main-029-proto) | 38.4 | 80.8 | 105.2 | 122.3 |

Acceptance is level (ours p0 90.0 p1 76.7 against 88.1 / 73.7 at c6) and tokens per step is level
(4.585 against 4.676), so the whole gap is decode step time: 38.65 ms against 29.4 ms at c6, and
114 ms against 74 ms at c1. The gap is worst at the lowest concurrency, which is a latency, not a
bandwidth, signature.

Engine config diff, read from both containers' own logs:

| | ours | reference |
|---|---|---|
| compilation mode | `NONE` | `VLLM_COMPILE` (3), but skipped: "torch.compile is turned on, but the model ... does not support it" |
| cudagraph | breakable, `Breakable CUDA graph enabled` | plain PIECEWISE, `VLLM_USE_BREAKABLE_CUDAGRAPH=0` |
| attention | `B12X_MLA_SPARSE` (`b12x_sparse.py`, our overlay) | `FLASHINFER_MLA_SPARSE_DSV4`, autotuned `sparse_mla_sm120_decode_dsv4`, 24 cached configs |
| MoE | `B12X_MXFP4_MXFP8` (`B12X_MOE_FORCE_A8=1`) | `B12X_MXFP4` |
| linear | `b12x` | `auto`, selects `DeepGemmFp8BlockScaledMMKernel` |
| DeepGEMM E8M0 | disabled (`VLLM_USE_DEEP_GEMM_E8M0=0`) | enabled, "enabling UE8M0 for DeepGEMM" |
| all-reduce | PYNCCL | PYNCCL |
| prefix caching | on | on |

The FlashInfer autotune line is the sharpest: the reference loads 24 tuned configs for its sparse
MLA decode kernel, ours loads 0. Our image ships the same path
(`models/deepseek_v4/nvidia/flashinfer_sparse.py`) and `_select_dsv4_attn_cls` maps SM12 to
`DeepseekV4FlashInferSM120Attention` when no explicit backend is given, so the forced
`B12X_MLA_SPARSE` in `pin.main-029.env` is the only thing keeping us off it.

Two corrections to earlier claims in this file:

- The reference does not compile. Its config says `VLLM_COMPILE` but vLLM skips compilation for this
  model in both images, so "the reference runs compiled piecewise graphs" was wrong and compilation
  is not the explanation for the gap. The compile path stays parked.
- The graph-node dumps in the previous section came from a run that never reached health, and that
  run's grid histogram matched eager-launch dump lines, not graph nodes. The `24576` and `6144`
  grids are eager launches from `vllm._C_stable_libtorch` (6,291,456 and 1,572,864 elements, i.e.
  1536 and 384 tokens at hidden 4096) taken during a memory-profile pass, not the decode step. The
  copy-family lead is retired until an instrument can see inside a replayed graph.

Harness changes made here: `pin.main-029.env` now reads
`DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-B12X_MLA_SPARSE}"` so the draft backend can be
A/B'd without editing the pin. `~/goal/run-arm.sh <tag> "<env assignments>"` is the guarded arm
runner: it applies the assignments to both the worker and the head, waits for health, saves the
engine-config lines to `~/goal/<tag>-engine.log`, and then meters.

Next: A/B the knobs the diff exposes, one at a time, at the same k=7 / capture 48 / async config and
against `protog` 38.4 / 80.8 / 105.2 / 122.3:
1. stock FlashInfer sparse MLA attention (`FLASHINFER_MLA_SPARSE_DSV4`), target and draft
2. `LINEAR_BACKEND=auto` with `VLLM_USE_DEEP_GEMM_E8M0=1`
3. `B12X_MOE_FORCE_A8=0`
