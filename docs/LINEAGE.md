# Lineage

Base: vLLM **v0.28.0rc2 Python** (`74a6576`, 2026-08-21 06:47 UTC) on
`vllm/vllm-openai:v0.27.1` arm64 runtime.

#52016 (b12x **linear**) merged 2026-08-14, **in rc2**. `LinearBackend` has
`"b12x"`. `b12x_warmup.py` warms linear GEMMs.

#52018 (b12x **MXFP4 MoE**, `MoEBackend="b12x"`, `B12xExperts`) merged
2026-08-21 15:04 UTC, **after** rc2. rc2 `MoEBackend` has `flashinfer_b12x`
only. Cherry-picked via `patches/apply_overlays.py`.

#50645 / #53055 (`mhc_pre_broadcast_tilelang` DeepGEMM guard): still open.
Sibling mHC kernels already fall back to TileLang in rc2; the broadcast
kernel is still unguarded.

Tracker: [UPSTREAM.md](UPSTREAM.md). Matched-main build plan (CUDA 13.3.1
devel, source PyTorch 2.14): [PLAN-MAIN.md](PLAN-MAIN.md).

DeepGEMM pins (not interchangeable):

- this image `.so`: v0.27.1 cmake `e21c821` (DeepGEMM main, no SM12x MQA)
- v0.28.0rc2 / vLLM main cmake: `8b1392b` (nv_dev HEAD)
- eugr Dockerfile: `a6b593d` (nv_dev frozen; MXFP4 grouped-scale regression
  at `f8e8fb5` / PR #384)

DSpark is in rc2 (`method=dspark`). 0731 locks k=5 (`dspark_block_size=5`,
`num_nextn_predict_layers=1`). This recipe never serves without that spec.

`nvfp4_ds_mla` is a community control-path name. 0.28
`validate_nvfp4_kv_cache_with_mla` uses `startswith("nvfp4")` and would reject
the DSV4 dtype; the overlay narrows that guard to exact `"nvfp4"`. Page
geometry is 584 B for DSV4 main/SWA, indexer left at upstream 512/576.

Stock vLLM main has no `B12X_MLA_SPARSE` enum. Matched-main registers it
via `patches/files/dsv4_b12x_sparse.py` onto the same 584 B page (b12x
`COMPRESSED_MLA_BYTES_PER_TOKEN=584`). GLM NVFP4 is 432/368 and
`scale_format=2`; do not mix that writer with this dtype.

Measured real NVFP4 KV on 2x Spark remains anemll `dspark-vllm-gx10:0.1.1`
(7,650 B/token). See parent research note
`NVFP4_DS_MLA_LINEAGE.md` in the TRT-LLM tree (external repo). The predecessor
repos
(`vllm-spark-main-b12x`, `vllm-spark-nvfp4`,
`dgx-spark-deepseek-v4-flash-0731`) were absorbed here on 2026-08-25; raw
archives: [docs/field-notes/](field-notes/README.md).
