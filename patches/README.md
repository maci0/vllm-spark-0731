# Patches

Applied at image build by `apply_overlays.py` onto vLLM v0.28.0rc2
(installed from source on the `vllm/vllm-openai:v0.27.1` arm64 base).

## Build-time overlays (baked into image)

| File | Why |
|---|---|
| `files/fused_moe_b12x.py` | B12xExperts MoE implementation (`vllm/.../fused_moe/b12x.py`) |
| `files/b12x_moe.py` | b12x weight helpers (`vllm/.../quantization/utils/b12x_moe.py`) |
| `apply_overlays.py` | MoEBackend b12x, mxfp4 oracle + process_weights fix, fp8_einsum SM12x fallback, B12xWarmupUnit, mHC TileLang guard, nvfp4_ds_mla 584 B, Mxfp4MoEMethod caller fix |
| `assert_image.py` | build-time: SM12x guards present, DSV4 shape, KV dtypes |
| `assert_0731.py` | checkpoint pin: dspark_block_size=5, DeepseekV4ForCausalLM |
| `assert_stack.py` | runtime: DSpark k=5 required, refuse B12X_MLA_SPARSE + nvfp4 |

## Reference files (not applied directly)

| Dir | Contents |
|---|---|
| `hotfixes/` | Full-file versions used during v0.27.1 bring-up (kept for reference) |
| `upstream/` | v0.27.1 PR diffs |

## Do not use

- `maci0/vllm-spark-nvfp4` `nvfp4-ds-mla-v0.27.1.patch` (191-line envelope)
- `eugr-nvfp4.patch` (89-line; 432 vs 584)
- GLM 432/368 writer
- Stage-C `head_bytes = 584` probe without a writer
