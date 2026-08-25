#!/usr/bin/env python3
"""Build-time behavioural asserts for overlayed vLLM images.

rc2: overlay on v0.28.0rc2 + v0.27.1 .so (blanket DeepGEMM SM12x kill).
main: PLAN-MAIN keep/add overlays (no blanket DeepGEMM kill).
"""

from __future__ import annotations

import argparse
import inspect
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=["rc2", "main"], default="rc2")
    args = parser.parse_args(argv)

    import b12x  # noqa: F401

    from vllm.config.cache import CacheDType
    from vllm.config.kernel import LinearBackend, MoEBackend
    from vllm.utils.b12x import B12xWarmupUnit, get_b12x_fused_moe
    from vllm.models.deepseek_v4 import attention as a
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode
    from vllm.v1.attention.backends.mla import sparse_swa
    from vllm.models.deepseek_v4 import sparse_mla
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mx
    from vllm.model_executor.kernels.mhc import tilelang as mhc
    from vllm.config.vllm import VllmConfig
    from vllm.utils.deep_gemm import is_deep_gemm_supported, fp8_einsum

    opts = getattr(CacheDType, "__args__", ())
    assert "fp8_ds_mla" in opts, opts
    assert "nvfp4_ds_mla" in opts, f"nvfp4_ds_mla not an accepted --kv-cache-dtype: {opts}"

    assert "b12x" in getattr(MoEBackend, "__args__", ()), "MoEBackend missing b12x"
    assert "b12x" in getattr(LinearBackend, "__args__", ()), "LinearBackend missing b12x"

    assert a._dsv4_page_alignment("nvfp4_ds_mla") == 584
    assert a._dsv4_page_alignment("fp8_ds_mla") == 576
    assert a._dsv4_page_alignment("auto") == 512

    idx_src = inspect.getsource(a.DeepseekV4IndexerCache.get_kv_cache_spec)
    assert "_dsv4_page_alignment" not in idx_src
    assert "nvfp4_ds_mla" not in idx_src

    sw_src = inspect.getsource(sparse_swa.DeepseekV4SWACache.get_kv_cache_spec)
    assert "584" in sw_src and "_dsv4_page_alignment" in sw_src

    swa_shape = getattr(sparse_swa.DeepseekSparseSWABackend, "get_kv_cache_shape", None)
    if swa_shape is not None:
        assert swa_shape(1, 64, 1, 512, cache_dtype_str="nvfp4_ds_mla") == (1, 64, 584)
        assert swa_shape(1, 64, 1, 512, cache_dtype_str="fp8_ds_mla") == (1, 64, 584)
    else:
        sw_src = inspect.getsource(sparse_swa.DeepseekV4SWACache.get_kv_cache_spec)
        assert "_dsv4_page_alignment" in sw_src
    backend_cls = getattr(sparse_mla, "DeepseekV4SparseMLABackend",
                          getattr(sparse_mla, "DeepseekV4FlashMLABackend", None))
    assert backend_cls is not None, "sparse_mla backend class not found"
    mla_shape = getattr(backend_cls, "get_kv_cache_shape", None)
    if mla_shape is not None:
        assert mla_shape(1, 256, 1, 512, cache_dtype_str="nvfp4_ds_mla") == (1, 256, 584)
    else:
        attn_spec_src = inspect.getsource(a.DeepseekV4Attention.get_kv_cache_spec)
        assert "_dsv4_page_alignment" in attn_spec_src

    assert get_kv_quant_mode("nvfp4_ds_mla") == KVQuantMode.NVFP4
    assert get_kv_quant_mode("fp8_ds_mla") != KVQuantMode.NONE

    src = inspect.getsource(a._resolve_dsv4_kv_cache_dtype)
    assert "nvfp4_ds_mla" in src

    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLASparseBackend,
    )
    if args.stack == "main":
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.models.deepseek_v4.nvidia.b12x_sparse import (
            DeepseekV4B12xMLASparseBackend,
            DeepseekV4B12xSM120Attention,
            as_page_bytes,
        )
        assert AttentionBackendEnum.B12X_MLA_SPARSE.value.endswith(
            "DeepseekV4B12xMLASparseBackend"
        )
        assert DeepseekV4B12xMLASparseBackend.get_name() == "B12X_MLA_SPARSE"
        assert "nvfp4_ds_mla" in DeepseekV4B12xMLASparseBackend.supported_kv_cache_dtypes
        fake = __import__("torch").zeros((2, 64, 584), dtype=__import__("torch").uint8)
        pages, page_size = as_page_bytes(fake)
        assert page_size == 64 and tuple(pages.shape) == (2, 64 * 584)
        from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls
        sel_src = inspect.getsource(_select_dsv4_attn_cls)
        assert "B12X_MLA_SPARSE" in sel_src
        assert "DeepseekV4B12xSM120Attention" in sel_src
        assert DeepseekV4B12xSM120Attention is not None
    assert "nvfp4_ds_mla" in DeepseekV4FlashInferMLASparseBackend.supported_kv_cache_dtypes
    dsv4_comb = inspect.getsource(
        DeepseekV4FlashInferMLASparseBackend.supports_combination
    )
    assert "nvfp4_ds_mla" in dsv4_comb, "DSV4 SM12x still rejects nvfp4_ds_mla"
    dsv4_bs_src = inspect.getsource(
        DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes
    )
    assert "is_device_capability_family(120)" in dsv4_bs_src
    from vllm.platforms import current_platform
    if current_platform.is_device_capability_family(120):
        assert (
            DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
            == [64]
        )
        from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend
        from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLABackend
        assert DeepseekV4IndexerBackend.get_supported_kernel_block_sizes() == [64]
        assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [64]

    assert hasattr(mx.Mxfp4MoeBackend, "B12X_MXFP4_MXFP8")
    assert "b12x" in inspect.getsource(mx.map_mxfp4_backend)

    mhc_src = inspect.getsource(mhc.mhc_pre_broadcast_tilelang)
    assert "is_deep_gemm_supported" in mhc_src
    assert "_tilelang_hc_prenorm_gemm" in mhc_src

    guard = inspect.getsource(VllmConfig.validate_nvfp4_kv_cache_with_mla)
    assert 'cache_dtype == "nvfp4"' in guard
    assert "startswith" not in guard.split("def validate_nvfp4_kv_cache_with_mla", 1)[1][
        :400
    ]

    assert B12xWarmupUnit is not None
    assert get_b12x_fused_moe is not None

    from vllm.model_executor.layers.fused_moe.b12x import B12xExperts  # noqa: F401

    # SM12x guards
    dg_src = inspect.getsource(is_deep_gemm_supported)
    if args.stack == "rc2":
        assert "is_device_capability_family(120)" in dg_src, (
            "is_deep_gemm_supported missing SM12x exclusion"
        )
    else:
        assert "is_device_capability_family(120)" not in dg_src, (
            "main stack must not blanket-kill DeepGEMM on family 120"
        )

    einsum_src = inspect.getsource(fp8_einsum)
    assert "is_device_capability_family(120)" in einsum_src, "fp8_einsum missing SM12x fallback"

    from vllm.utils import deep_gemm as dg
    assert hasattr(dg, "_sm12x_fp8_scale_fp32"), "fp8_einsum missing UE8M0 scale upcast"

    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        compute_fp8_einsum_recipe,
        deep_gemm_fp8_o_proj,
    )
    recipe_src = inspect.getsource(compute_fp8_einsum_recipe)
    assert "cap.major == 12" in recipe_src, "o_proj recipe still uses SM100 packed scales on SM12x"
    o_src = inspect.getsource(deep_gemm_fp8_o_proj)
    assert "try_b12x_wo_proj" in o_src, "o_proj missing b12x WO projection try"

    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits
    mqa_src = inspect.getsource(fp8_fp4_mqa_logits)
    assert "is_device_capability_family(120)" in mqa_src, "fp8_fp4_mqa_logits missing SM12x guard"
    paged_src = inspect.getsource(fp8_fp4_paged_mqa_logits)
    assert "is_device_capability_family(120)" in paged_src, "fp8_fp4_paged_mqa_logits missing SM12x guard"
    paged_fn = inspect.getsource(dg._sm12x_fp8_paged_mqa_logits)
    assert ".item()" not in paged_fn, "paged MQA fallback still host-syncs via .item() (breaks cudagraph)"
    assert "gather_len" in paged_fn, "paged MQA fallback missing static gather_len"
    assert ".relu(" in paged_fn or "relu()" in paged_fn, "paged MQA fallback missing ReLU"
    assert "contiguous_logits" in paged_fn, "paged MQA missing B12x contiguous scorer"
    assert "try_paged_mqa_logits" in paged_fn, "paged MQA missing b12x paged kernel try"
    from vllm.utils import sm12x_b12x_kernels as b12x_k
    assert hasattr(b12x_k, "try_paged_mqa_logits")
    assert hasattr(b12x_k, "try_b12x_wo_proj")
    mqa_fn = inspect.getsource(dg._sm12x_fp8_mqa_logits)
    assert "relu" in mqa_fn, "prefill MQA fallback missing ReLU"
    assert "contiguous_logits" in mqa_fn, "prefill MQA missing B12x contiguous scorer"
    assert hasattr(dg, "_sm12x_b12x_mqa_pack"), "missing B12x MQA helper"

    from vllm.models.deepseek_v4.attention import DeepseekV4Attention
    insert_src = inspect.getsource(DeepseekV4Attention._fused_qnorm_rope_kv_insert)
    assert "xpu_qnorm_rope_kv_fp8_insert" in insert_src, "SM12x still uses CUDA fused KV insert"
    assert "is_device_capability_family(120)" in insert_src, "SM12x KV insert missing family 120 guard"
    assert "getattr(self, \"eager_scratch_pool\", None)" in insert_src, (
        "SM12x KV insert still requires eager_scratch_pool on SM120 attention"
    )

    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFp8BlockScaledMMKernel
    cutlass_src = inspect.getsource(CutlassFp8BlockScaledMMKernel.is_supported)
    assert "is_device_capability_family(120)" in cutlass_src, "CUTLASS FP8 missing SM12x exclusion"

    from vllm.v1.attention.backends.mla import indexer
    idx_build_src = inspect.getsource(indexer.DeepseekV32IndexerMetadataBuilder.build)
    assert (
        "is_deep_gemm_supported" in idx_build_src
        or "_should_build_paged_mqa_logits_metadata" in idx_build_src
    ), "indexer build() missing is_deep_gemm_supported guard"
    assert (
        "is_device_capability_family(120)" in idx_build_src
        or "_should_build_paged_mqa_logits_metadata" in idx_build_src
    ), (
        "indexer still calls DeepGEMM paged MQA metadata on family 120"
    )

    if args.stack == "rc2":
        mkk_src = inspect.getsource(mx.make_mxfp4_moe_kernel)
        assert "process_weights_after_loading" in mkk_src, (
            "make_mxfp4_moe_kernel missing process_weights call"
        )
        assert "layer" in inspect.signature(mx.make_mxfp4_moe_kernel).parameters, (
            "make_mxfp4_moe_kernel missing layer param"
        )
    else:
        from vllm.model_executor.layers.fused_moe.b12x import B12xExperts as _B12xExperts
        assert hasattr(_B12xExperts, "process_weights_after_loading"), (
            "main B12xExperts missing process_weights_after_loading"
        )

    # FlashInfer DSV4 dispatch: all (H, 192) must be registered for DSpark k=5
    from flashinfer.mla._sparse_mla_sm120 import _DECODE_DSV4_DISPATCH
    for h in (8, 16, 32, 64, 128):
        assert (h, 192) in _DECODE_DSV4_DISPATCH, (
            f"FlashInfer _DECODE_DSV4_DISPATCH missing ({h}, 192): {sorted(_DECODE_DSV4_DISPATCH)}"
        )

    # FlashInfer DSV4 C++ source: TOPK=192 dispatch entries for JIT compilation
    from pathlib import Path as P
    cu_path = P("/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu")
    if cu_path.is_file():
        cu_src = cu_path.read_text()
        assert "DSV4_DISPATCH(32, 192)" in cu_src, "C++ DSV4 dispatch missing TOPK=192"

    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    ar_src = inspect.getsource(tensor_model_parallel_all_reduce)
    assert "static workspace" in ar_src, "TP all-reduce missing default-allocator workspace"

    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    cap_src = inspect.getsource(DFlashSpeculator.capture)
    assert "_draft_hidden" in cap_src, "DSpark capture() missing backbone buffer"
    assert "lm_head eager" in cap_src, "DSpark capture() still skips graphs entirely"
    bb_src = inspect.getsource(DFlashSpeculator._draft_backbone)
    assert "_run_model" in bb_src, "DSpark backbone missing _run_model"
    assert "hs.shape[1]" in bb_src, "DSpark backbone still copies into target hidden_size"

    if args.stack == "main":
        from vllm.model_executor.model_loader import get_model
        gm_src = inspect.getsource(get_model)
        assert "_instanttensor_draft_load_config" in gm_src, (
            "main stack missing InstantTensor hybrid draft loader"
        )
        from vllm.model_executor.kernels.linear.scaled_mm.b12x import (
            _run_b12x_fp8_block_scaled_mm,
        )
        b12x_mm_src = inspect.getsource(_run_b12x_fp8_block_scaled_mm)
        assert "block_fp8=True" in b12x_mm_src, (
            "main stack missing git-b12x mm_block_fp8 compatibility"
        )
        from vllm.v1.worker.utils import allocate_kv_cache
        alloc_src = inspect.getsource(allocate_kv_cache)
        assert "tokens_per_state" in alloc_src, (
            "main stack missing DSV4 compressed-page kernel-split skip"
        )

    print(
        f"image OK ({args.stack}): b12x importable, moe/linear b12x, "
        "fp8_ds_mla + nvfp4_ds_mla, 584B DSV4 page, "
        "mHC TileLang guard, SM12x kernel guards, DSpark dispatch"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
