# Patches

Applied at image build by `apply_overlays.py` onto `vllm/vllm-openai:v0.28.0rc2`.

## Applied

| File | Why |
|---|---|
| `files/fused_moe_b12x.py` | vLLM #52018 `vllm/model_executor/layers/fused_moe/b12x.py` |
| `files/b12x_moe.py` | vLLM #52018 weight helpers |
| `apply_overlays.py` | MoEBackend `b12x`, mxfp4 oracle, B12xWarmupUnit, #50645 mHC, nvfp4_ds_mla 584 B |
| `assert_image.py` | build-time: both KV dtypes, 584 not 432, indexer untouched, DSV4 shape |
| `assert_0731.py` | checkpoint pin, including `dspark_block_size=5` |
| `assert_stack.py` | DSpark k=5 required; refuse B12X_MLA_SPARSE + nvfp4 |

## Provenance diffs (not applied blindly)

`upstream/` and `v0.28/` keep the 0.27.1 / PR diffs for review. rc2 linear
already has `b12x` (#52016). rc2 warmup is a different file than #52018's
hunk, so MoE warmup units are not rewritten.

## Do not drop in

- `maci0/vllm-spark-nvfp4` `nvfp4-ds-mla-v0.27.1.patch` (191-line envelope)
- `eugr-nvfp4.patch` (89-line; 432 vs 584)
- GLM 432/368 writer
- Stage-C `head_bytes = 584` probe without a writer
