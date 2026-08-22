#!/usr/bin/env python3
"""Build-time behavioural asserts for the 0.28 + b12x + dual-KV image."""

from __future__ import annotations

import inspect
import sys


def main() -> int:
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

    assert sparse_swa.DeepseekSparseSWABackend.get_kv_cache_shape(
        1, 64, 1, 512, cache_dtype_str="nvfp4_ds_mla"
    ) == (1, 64, 584)
    assert sparse_swa.DeepseekSparseSWABackend.get_kv_cache_shape(
        1, 64, 1, 512, cache_dtype_str="fp8_ds_mla"
    ) == (1, 64, 584)
    backend_cls = getattr(sparse_mla, "DeepseekV4SparseMLABackend",
                          getattr(sparse_mla, "DeepseekV4FlashMLABackend", None))
    assert backend_cls is not None, "sparse_mla backend class not found"
    assert backend_cls.get_kv_cache_shape(
        1, 256, 1, 512, cache_dtype_str="nvfp4_ds_mla"
    ) == (1, 256, 584)

    assert get_kv_quant_mode("nvfp4_ds_mla") == KVQuantMode.NVFP4
    assert get_kv_quant_mode("fp8_ds_mla") != KVQuantMode.NONE

    src = inspect.getsource(a._resolve_dsv4_kv_cache_dtype)
    assert "nvfp4_ds_mla" in src

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
    assert "is_device_capability_family(120)" in dg_src, "is_deep_gemm_supported missing SM12x exclusion"

    einsum_src = inspect.getsource(fp8_einsum)
    assert "is_device_capability_family(120)" in einsum_src, "fp8_einsum missing SM12x fallback"

    mkk_src = inspect.getsource(mx.make_mxfp4_moe_kernel)
    assert "process_weights_after_loading" in mkk_src, "make_mxfp4_moe_kernel missing process_weights call"
    assert "layer" in inspect.signature(mx.make_mxfp4_moe_kernel).parameters, "make_mxfp4_moe_kernel missing layer param"

    # FlashInfer DSV4 dispatch: (32, 192) must be registered for DSpark k=5
    from flashinfer.mla._sparse_mla_sm120 import _DECODE_DSV4_DISPATCH
    assert (32, 192) in _DECODE_DSV4_DISPATCH, (
        f"FlashInfer _DECODE_DSV4_DISPATCH missing (32, 192): {sorted(_DECODE_DSV4_DISPATCH)}"
    )

    print(
        "image OK: b12x importable, moe/linear b12x, "
        "fp8_ds_mla + nvfp4_ds_mla, 584B DSV4 page, "
        "mHC TileLang guard, SM12x kernel guards, DSpark dispatch"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
