"""Warm the mHC first-layer broadcast path + DSpark gumbel sampler kernels.

Upstream ``deepseek_v4_mhc_warmup`` drives ``layer.hc_pre()`` with a 3D
``[T, hc_mult, H]`` residual (the per-layer, non-broadcast path). The first
``DeepseekV4DecoderLayer`` instead calls ``mhc_pre_broadcast_tilelang()``
with a 2D ``[T, H]`` residual plus ``fn_broadcast`` -- never warmed. On the
first served request after boot the TileLang kernel
``mhc_pre_big_fuse_broadcast_with_norm_tilelang`` JITs (~30-120 s) and the
DeepGEMM ``tf32_hc_prenorm_gemm`` compiles once per token count.

The DSpark draft path samples eagerly through ``gumbel_sample()`` (triton);
no upstream warmup covers it, so the first request pays the triton compile.

Both warmups are no-ops for non-DSv4 models / absent modules. Called from
``kernel_warmup`` right after the upstream mHC warmup.
"""

import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_WARMUP_TOKEN_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 192)


def _is_deepseek_v4(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    return model_type == "deepseek_v4"


def _token_sizes(*, cudagraph_capture_sizes: list[int]) -> list[int]:
    sizes = set(_WARMUP_TOKEN_SIZES)
    sizes.update(int(s) for s in cudagraph_capture_sizes if int(s) > 0)
    return sorted(sizes)


def _find_first_broadcast_mhc_layer(
    model: torch.nn.Module,
) -> torch.nn.Module | None:
    """First decoder layer whose mHC pre runs the 2D broadcast path."""
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if not all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_attn_fn_broadcast",
                "attn_norm",
            )
        ):
            continue
        if getattr(module, "hc_attn_fn_broadcast", None) is not None:
            return module
    return None


def deepseek_v4_mhc_broadcast_warmup(
    model: torch.nn.Module,
    *,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    """Pre-compile the first-layer mHC broadcast kernels for serving shapes.

    Mirrors the exact call the model makes for layer 0
    (``mhc_pre_broadcast_tilelang`` with ``fn_broadcast``), which runs
    DeepGEMM ``tf32_hc_prenorm_gemm`` (compiles per M) and the TileLang
    ``mhc_pre_big_fuse_broadcast_with_norm_tilelang``.
    """
    if not _is_deepseek_v4(model):
        return
    layer = _find_first_broadcast_mhc_layer(model)
    if layer is None:
        logger.info_once(
            "Skipping mHC broadcast warmup: no first-layer broadcast mHC found."
        )
        return

    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_pre_broadcast_tilelang,
    )

    sizes = _token_sizes(cudagraph_capture_sizes=cudagraph_capture_sizes or [])
    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    hidden_size = int(layer.hidden_size)
    max_t = max(sizes)
    residual = torch.zeros(max_t, hidden_size, dtype=torch.bfloat16, device=device)
    norm_weight = layer.attn_norm.weight.data
    norm_eps = layer.attn_norm.variance_epsilon

    started = time.perf_counter()
    logger.info(
        "Warming up DSv4 mHC broadcast TileLang + DeepGEMM kernels for token sizes: %s",
        sizes,
    )
    with torch.inference_mode():
        for size in sizes:
            mhc_pre_broadcast_tilelang(
                residual[:size],
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
                fn_broadcast=layer.hc_attn_fn_broadcast,
            )
        torch.accelerator.synchronize()
    logger.info(
        "DSv4 mHC broadcast warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )


def dspark_gumbel_warmup(
    model: torch.nn.Module,
    *,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    """Pre-compile the DSpark draft gumbel triton kernels (all constexpr combos)."""
    if not _is_deepseek_v4(model):
        return
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    vocab_size = int(model.config.vocab_size)
    if vocab_size <= 0:
        return
    device = next(model.parameters()).device
    if device.type != "cuda":
        return

    logits = torch.zeros(1, vocab_size, dtype=torch.bfloat16, device=device)
    idx_map = torch.zeros(1, dtype=torch.int64, device=device)
    temperature = torch.ones(1, dtype=torch.float32, device=device)
    seed = torch.zeros(1, dtype=torch.int64, device=device)
    pos = torch.zeros(1, dtype=torch.int64, device=device)

    started = time.perf_counter()
    logger.info("Warming up DSpark gumbel sampler kernels (vocab %d).", vocab_size)
    with torch.inference_mode():
        for use_fp64 in (False, True):
            for apply_temperature in (False, True):
                for per_token_col in (False, True):
                    cache = None
                    cache_col = None
                    if per_token_col:
                        cache = torch.zeros(
                            1, 1, vocab_size, dtype=torch.bfloat16, device=device
                        )
                        cache_col = torch.zeros(1, dtype=torch.int64, device=device)
                    gumbel_sample(
                        logits,
                        idx_map,
                        temperature,
                        seed,
                        pos,
                        apply_temperature,
                        cache,
                        cache_col,
                        use_fp64,
                    )
        torch.accelerator.synchronize()
    logger.info(
        "DSpark gumbel warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
