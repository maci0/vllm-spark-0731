#!/usr/bin/env python3
"""Build-time behavioural asserts for overlayed vLLM images.

rc2: overlay on v0.28.0rc2 + v0.27.1 .so (blanket DeepGEMM SM12x kill).
main: PLAN-MAIN keep/add overlays (no blanket DeepGEMM kill).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
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

    # The base gives nvfp4_ds_mla its own mode (10, opaque-bytes DS-MLA layouts) and
    # tests it before the "nvfp4" prefix, which returns the plain NVFP4 mode (5).
    assert get_kv_quant_mode("nvfp4_ds_mla") == KVQuantMode.NVFP4_DS_MLA
    assert get_kv_quant_mode("nvfp4") == KVQuantMode.NVFP4
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
    from vllm.models.deepseek_v4.sparse_mla import (
        dsv4_supported_kernel_block_sizes,
    )
    # pr-53425 on v0.28.0: the FlashInfer subclass override was removed and the
    # SM12x 64-token page logic lives in the sparse_mla helper the base class
    # method delegates to. Assert on the helper source.
    dsv4_bs_src = inspect.getsource(dsv4_supported_kernel_block_sizes)
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

    # The overlay's invariant is the opposite of what used to be asserted here: it
    # *removes* the traced `is_deep_gemm_supported()` calls and reads the static
    # `_USE_DEEP_GEMM` instead, so asserting the string is present in a forward
    # cannot hold once the overlay has done its job. On proto-v0.2.0
    # `mhc_pre_broadcast_tilelang` no longer contains the call at all, so assert
    # the real property: the static flag exists and no bare call survives anywhere
    # in the module. The overlay raises on leftovers too; this is the belt to that
    # brace, and it is what the old assertion was reaching for.
    mhc_src = inspect.getsource(mhc)
    assert "_USE_DEEP_GEMM" in mhc_src
    # `_tilelang_hc_prenorm_gemm` was asserted here before proto-v0.2.0 and no longer
    # exists in this module at all (upstream renamed the prenorm path), so the name
    # check is dropped rather than retargeted at another private symbol. What the
    # overlay actually guarantees, and what is checked here, is that the static flag
    # is present and that no traced call to the ctypes pointer survives.
    leftover = re.findall(r"(?<!_)\bis_deep_gemm_supported\(\)", mhc_src)
    assert not leftover, f"{len(leftover)} traced is_deep_gemm_supported() call(s) remain"

    guard = inspect.getsource(VllmConfig.validate_nvfp4_kv_cache_with_mla)
    # The guard must not reject nvfp4_ds_mla. Upstream now exempts it with a
    # `endswith("_ds_mla")` test ahead of the nvfp4 prefix match; the older overlay
    # narrowed the test to `cache_dtype == "nvfp4"` and is no longer needed.
    assert 'endswith("_ds_mla")' in guard, "MLA guard would reject nvfp4_ds_mla"

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

    from vllm.utils import deep_gemm as dg
    # 2026-08-27: the einsum "fallback" was a misdiagnosis - DeepGEMM's einsum
    # kernel is correct on SM12x once the recipe/scale pair is coherent
    # (measured 0.000000 mean_rel on the v0.28 path; 0.027 max_rel against a
    # torch reference on v0.29). The stock passthrough is required; the
    # fallback/upcast helper must NOT be present.
    einsum_src = inspect.getsource(fp8_einsum)
    assert "is_device_capability_family(120)" not in einsum_src, (
        "fp8_einsum still carries the obsolete SM12x dequant fallback"
    )
    assert not hasattr(dg, "_sm12x_fp8_scale_fp32"), (
        "fp8_einsum still carries the obsolete UE8M0 scale upcast"
    )

    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        compute_fp8_einsum_recipe,
        deep_gemm_fp8_o_proj,
    )
    # proto-v0.2.0: the SM12x (1,128,128)-with-fp32-scales override is gone.
    # apply_main no longer calls patch_einsum_sm12x_recipe, and upstream's own
    # recipe is what the SM120 DeepGEMM einsum kernel needs: (1,1,block_size)
    # with tma_aligned_scales=True emits packed UE8M0, where the override fed
    # the kernel SM90-style fp32 scales and it rejected the call at
    # csrc/utils/layout.hpp:113 during _initialize_kv_caches, so the engine
    # never reached health. The override's precondition is also gone -- the
    # Python fp8_einsum fallback/fp8_bmm path is not applied any more (asserted
    # above), and utils/deep_gemm.py binds DeepGEMM's real fp8_einsum.
    recipe_src = inspect.getsource(compute_fp8_einsum_recipe)
    assert "cap.major == 12" not in recipe_src, (
        "o_proj recipe still carries the SM12x (1,128,128) override, which "
        "DeepGEMM rejects at layout.hpp:113 and which stops the engine starting"
    )
    assert "einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)" in recipe_src, (
        "o_proj recipe is not upstream's (1,1,block_size) for major >= 10"
    )
    assert "tma_aligned_scales = cap.major >= 10" in recipe_src, (
        "o_proj recipe no longer requests TMA-aligned scales on major >= 10"
    )
    o_src = inspect.getsource(deep_gemm_fp8_o_proj)
    assert "try_b12x_wo_proj" in o_src, "o_proj missing b12x WO projection try"
    assert "deepgemm_post_process_fp8_weight_block" in o_src and "is_bmm=True" in o_src, (
        "o_proj einsum missing the 3-D is_bmm wo_a weight (fp8_bmm needs 3-D)"
    )

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

    from vllm.model_executor.warmup import dsv4_warmup_ext
    assert hasattr(dsv4_warmup_ext, "deepseek_v4_mhc_layer_warmup")
    assert hasattr(dsv4_warmup_ext, "dspark_gumbel_warmup")
    from vllm.model_executor.warmup import kernel_warmup as _kw
    assert "deepseek_v4_mhc_layer_warmup" in inspect.getsource(
        _kw.kernel_warmup
    ), "kernel_warmup missing mHC broadcast warmup call"
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

    # FlashInfer DSV4 dispatch: every (H, 192) must be served for DSpark k=5.
    # flashinfer main takes topk as a runtime kernel argument, so membership
    # answers over the whole envelope instead of an enumerated pair set.
    from flashinfer.mla._sparse_mla_sm120 import _DECODE_DSV4_DISPATCH
    for h in (8, 16, 32, 64, 128):
        assert (h, 192) in _DECODE_DSV4_DISPATCH, (
            f"FlashInfer _DECODE_DSV4_DISPATCH missing ({h}, 192): "
            f"{_DECODE_DSV4_DISPATCH!r}"
        )

    # FlashInfer DSV4 C++ source: only a pre-runtime-topk flashinfer carries a
    # TOPK dispatch table to extend; current main instantiates on num_heads.
    from pathlib import Path as P
    cu_path = P("/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu")
    if cu_path.is_file():
        cu_src = cu_path.read_text()
        if "DSV4_DISPATCH(8, 128)" in cu_src:
            assert "DSV4_DISPATCH(32, 192)" in cu_src, "C++ DSV4 dispatch missing TOPK=192"

    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    # The SM12x profiling overlay wraps this callable (functools.wraps), so
    # follow __wrapped__ to keep checking the patched body.
    ar_src = inspect.getsource(inspect.unwrap(tensor_model_parallel_all_reduce))
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
        # v0.28.0 split the combined b12x.py into b12x_block.py / b12x_tensor.py
        # (#52016); v0.29.0 merged it back into b12x.py. Try both names.
        b12x_mm = None
        for name in ("b12x_block", "b12x"):
            try:
                b12x_mm = importlib.import_module(
                    f"vllm.model_executor.kernels.linear.scaled_mm.{name}"
                )
                break
            except ImportError:
                continue
        assert b12x_mm is not None, "main stack missing scaled_mm b12x kernel"
        b12x_mm_src = inspect.getsource(b12x_mm._run_b12x_fp8_block_scaled_mm)
        assert "block_fp8=True" in b12x_mm_src, (
            "main stack missing git-b12x mm_block_fp8 compatibility"
        )
        try:
            from vllm.v1.worker.utils import allocate_kv_cache
            alloc_src = inspect.getsource(allocate_kv_cache)
            assert "tokens_per_state" in alloc_src, (
                "main stack missing DSV4 compressed-page kernel-split skip"
            )
        except ImportError:
            # v0.28.0 KV-cache refactor (#51612/#51704) removed the BLHNC
            # kernel-split code this assert guarded - the overlay is a no-op.
            print("skip allocate_kv_cache assert: function absent in v0.28.0")


    print(
        f"image OK ({args.stack}): b12x importable, moe/linear b12x, "
        "fp8_ds_mla + nvfp4_ds_mla, 584B DSV4 page, "
        "mHC TileLang guard, SM12x kernel guards, DSpark dispatch"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
