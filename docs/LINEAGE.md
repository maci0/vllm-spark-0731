# Lineage

Base: `vllm/vllm-openai:v0.28.0rc2` (`74a6576`, 2026-08-21 06:47 UTC).

#52018 (b12x MXFP4 MoE) merged to main 2026-08-21 15:04 UTC, after rc2.
rc2 `MoEBackend` has `flashinfer_b12x` only. Linear `b12x` (#52016) is already
in rc2.

#50645 (mHC `mhc_pre_broadcast_tilelang` DeepGEMM guard) still open. Sibling
mHC kernels already fall back to TileLang; the broadcast variant does not.
SM12x DeepGEMM has no `tf32_hc_prenorm_gemm`.

DSpark is in rc2 (`method=dspark`). 0731 locks k=5 (`dspark_block_size=5`,
`num_nextn_predict_layers=1`). This recipe never serves without that spec.

`nvfp4_ds_mla` is a community control-path name. 0.28
`validate_nvfp4_kv_cache_with_mla` uses `startswith("nvfp4")` and would reject
the DSV4 dtype; the overlay narrows that guard to exact `"nvfp4"`. Page
geometry is 584 B for DSV4 main/SWA, indexer left at upstream 512/576.

Measured real NVFP4 KV on 2x Spark remains anemll `dspark-vllm-gx10:0.1.1`
(7,650 B/token). See parent research note
`docs/NVFP4_DS_MLA_LINEAGE.md` in the TRT-LLM tree, and
[maci0/dgx-spark-deepseek-v4-flash-0731](https://github.com/maci0/dgx-spark-deepseek-v4-flash-0731).
