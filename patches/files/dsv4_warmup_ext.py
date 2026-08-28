"""Warm the DSv4 mHC TileLang kernels + DSpark gumbel sampler kernels.

The v0.28.0 nvidia DeepseekV4DecoderLayer calls ``mhc_pre_tilelang`` /
``mhc_fused_post_pre_tilelang`` directly (it has no ``hc_pre``/``hc_post``
methods), so upstream ``deepseek_v4_mhc_warmup`` (which gates on those
methods) is a silent no-op here — every boot the first served request pays
a TileLang JIT (~30-120 s) plus the hc_head compile. This warmup drives the
same functions the layer calls, for every serving token size.

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


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    """First decoder layer carrying the direct-call mHC params."""
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if not all(
            hasattr(module, attr)
            for attr in (
                "hidden_size",
                "hc_mult",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
                "attn_norm",
                "ffn_norm",
                "rms_norm_eps",
                "hc_eps",
                "hc_post_alpha",
                "hc_sinkhorn_iters",
            )
        ):
            continue
        return module
    return None


def deepseek_v4_mhc_layer_warmup(
    model: torch.nn.Module,
    *,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    """Pre-compile the mHC pre / fused-post-pre / hc_head TileLang kernels.

    Mirrors the calls the v0.28.0 nvidia layer makes every decode step:
    ``mhc_pre_tilelang`` (layer 0, hc_attn_fn) then, per layer,
    ``mhc_fused_post_pre_tilelang`` (hc_attn_fn + hc_ffn_fn pairs), plus the
    model-level ``hc_head_op``.
    """
    if not _is_deepseek_v4(model):
        return
    layer = _find_first_mhc_layer(model)
    if layer is None:
        logger.info_once("Skipping mHC layer warmup: no DSv4 mHC layer found.")
        return

    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_tilelang,
        mhc_pre_tilelang,
    )

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return
    sizes = _token_sizes(cudagraph_capture_sizes=cudagraph_capture_sizes or [])
    hidden = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    max_t = max(sizes)
    attn_norm_w = layer.attn_norm.weight.data
    attn_norm_e = layer.attn_norm.variance_epsilon
    ffn_norm_w = layer.ffn_norm.weight.data
    ffn_norm_e = layer.ffn_norm.variance_epsilon

    started = time.perf_counter()
    logger.info(
        "Warming up DSv4 mHC layer TileLang kernels for token sizes: %s",
        sizes,
    )
    with torch.inference_mode():
        for size in sizes:
            x = torch.zeros(
                size, hc_mult, hidden, dtype=torch.bfloat16, device=device
            )
            post_mix, comb_mix, layer_input = mhc_pre_tilelang(
                x,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.rms_norm_eps,
                layer.hc_eps,
                layer.hc_eps,
                layer.hc_post_alpha,
                layer.hc_sinkhorn_iters,
                norm_weight=attn_norm_w,
                norm_eps=attn_norm_e,
            )
            residual = layer_input
            for fn, scale, base, nw, ne in (
                (
                    layer.hc_attn_fn,
                    layer.hc_attn_scale,
                    layer.hc_attn_base,
                    attn_norm_w,
                    attn_norm_e,
                ),
                (
                    layer.hc_ffn_fn,
                    layer.hc_ffn_scale,
                    layer.hc_ffn_base,
                    ffn_norm_w,
                    ffn_norm_e,
                ),
            ):
                residual, post_mix, comb_mix, layer_input = (
                    mhc_fused_post_pre_tilelang(
                        layer_input,
                        residual,
                        post_mix,
                        comb_mix,
                        fn,
                        scale,
                        base,
                        layer.rms_norm_eps,
                        layer.hc_eps,
                        layer.hc_eps,
                        layer.hc_post_alpha,
                        layer.hc_sinkhorn_iters,
                        n_splits=1,
                        tile_n=1,
                        norm_weight=nw,
                        norm_eps=ne,
                    )
                )

        # hc_head (model level) — same dispatch as upstream's warmup.
        hc_head_op = getattr(model, "hc_head_op", None)
        if hc_head_op is not None:
            hc_head_fn = getattr(model, "hc_head_fn", None)
            hc_head_scale = getattr(model, "hc_head_scale", None)
            hc_head_base = getattr(model, "hc_head_base", None)
            if all(t is not None for t in (hc_head_fn, hc_head_scale, hc_head_base)):
                hh = torch.zeros(
                    max_t, hc_mult, hidden, dtype=torch.bfloat16, device=device
                )
                for size in sizes:
                    hc_head_op(
                        hh[:size],
                        hc_head_fn,
                        hc_head_scale,
                        hc_head_base,
                        model.rms_norm_eps,
                        model.hc_eps,
                    )
        torch.accelerator.synchronize()
    logger.info(
        "DSv4 mHC layer warmup finished in %.2f seconds.",
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
