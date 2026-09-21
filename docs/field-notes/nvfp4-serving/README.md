# `nvfp4-serving/` (raw archive, added 2026-09-13)

Raw, verbatim DeepSeek-V4-serving write-ups, folded in on 2026-09-13 from
`~/Desktop/Projects/nvfp4/serving/` on the workstation.

**Why this was not already here.** The 2026-08-25 absorption named three predecessor *repositories*
(`vllm-spark-nvfp4`, `dgx-spark-deepseek-v4-flash-0731`, `vllm-spark-main-b12x`) and archived them
under [`docs/field-notes/`](README.md). This set came from a sibling directory of the `nvfp4`
project, which is not one of those three and has no `.git` of its own, so it was left out. It is
DeepSeek-V4-Flash-0731 on 2x GB10 material, so it belongs with the rest.

| File | Content |
|---|---|
| `DEEPSEEK_V4_QUANT_FRAMEWORK_SWEEP.md` | Quant x framework x image sweep on 2x GB10: official / abliterated / NVFP4 / FP8-repack checkpoints against vLLM and SGLang across eugr B12X, eugr, stock vLLM, nightly, NGC and SGLang |
| `DEEPSEEK_V4_2NODE_SERVE.md` | Two-node serve runbook for the NVFP4 abliterated checkpoint at full 1M context: expert parallel plus DeepEP, MLA, sparse attention, 200Gb RoCE |
| `DEEPSEEK_V4_SPECULATIVE_B12X.md` | Getting DSpark speculative decoding working on GB10, which stock vLLM cannot do here, via the eugr B12X image and an FP8 checkpoint. Companion to the 2-node runbook |
| `DEEPSEEK_V4_NVFP4_B12X_PATCH.md` | The SwiGLU-clamp adapter patch needed for an NVFP4 checkpoint on the eugr `spark-vllm-b12x` image |
| `DEEPSEEK_V4_SPARK_ARENA_SUBMIT.md` | How submission to spark-arena.com actually works: via the `sparkrun` CLI running the standard sweep, not a manual form |
| `deepseek-v4-flash-0731-2xspark.yaml` | The `sparkrun` recipe from that period: `runtime: vllm-ray`, `builder: eugr`, `container: vllm-node-b12x` |

## Caveats, stated rather than smoothed

- These are **predecessor documents, not current instructions**. The recipes and image names in
  them predate the proto-v0.1.0 base and the `main-029` line; the live protocol is in
  [`HANDOVER.md`](../../../HANDOVER.md) and the current tracker is
  [`docs/UPSTREAM.md`](../../UPSTREAM.md).
- They are kept **unedited** for provenance, like the rest of this directory. Where they disagree
  with the current knowledge chapters, the chapters win.
- `DEEPSEEK_V4_QUANT_FRAMEWORK_SWEEP.md` shares subject matter with what was distilled into
  [`dgx-spark/TEST_LOG.md`](../dgx-spark/TEST_LOG.md) and `docs/knowledge/`, so expect overlap. Both
  copies are kept; neither was merged or rewritten.
