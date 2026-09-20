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
    try:
        with torch.inference_mode():
            for size in sizes:
                x3d = torch.zeros(
                    size, hc_mult, hidden, dtype=torch.bfloat16, device=device
                )
                post_mix, comb_mix, layer_input = mhc_pre_tilelang(
                    x3d,
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
                # layer_input is [T, hidden] (2D); residual stays the
                # multi-stream [T, hc_mult, hidden] tensor — matches the
                # model's fused call.
                residual = x3d
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
                if all(
                    t is not None for t in (hc_head_fn, hc_head_scale, hc_head_base)
                ):
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
    except Exception as exc:  # warmup must never take the worker down
        logger.warning(
            "DSv4 mHC layer warmup failed (serving without it): %s: %s",
            type(exc).__name__,
            exc,
        )
        return
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
    try:
        with torch.inference_mode():
            for use_fp64 in (False, True):
                for apply_temperature in (False, True):
                    for is_drafting in (False, True):
                        for per_token_col in (False, True):
                            cache = None
                            cache_col = None
                            if per_token_col:
                                cache = torch.zeros(
                                    1, 1, vocab_size, dtype=torch.bfloat16, device=device
                                )
                                cache_col = torch.zeros(
                                    1, dtype=torch.int64, device=device
                                )
                            # Keywords on purpose: gumbel_sample gained
                            # is_drafting after this warmup was written, and
                            # binding positionally put every later argument one
                            # slot out, so the whole warmup raised and the
                            # kernels JIT-compiled during inference instead.
                            gumbel_sample(
                                logits=logits,
                                expanded_idx_mapping=idx_map,
                                temperature=temperature,
                                seed=seed,
                                pos=pos,
                                apply_temperature=apply_temperature,
                                is_drafting=is_drafting,
                                logits_cache=cache,
                                logits_cache_col=cache_col,
                                use_fp64=use_fp64,
                            )
            torch.accelerator.synchronize()
    except Exception as exc:  # warmup must never take the worker down
        logger.warning(
            "DSpark gumbel warmup failed (serving without it): %s: %s",
            type(exc).__name__,
            exc,
        )
        return
    logger.info(
        "DSpark gumbel warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )


def deepseek_v4_wo_a_einsum_warmup(model: torch.nn.Module) -> None:
    """Pre-pack every ``wo_a`` for the DeepGEMM fp8 einsum, before compilation.

    ``models/deepseek_v4/nvidia/ops/o_proj.py`` memoizes the layout transform on
    the layer::

        cached = getattr(wo_a, "_sm12x_einsum_wo_a", None)
        if cached is None or cached[2] != weight.data_ptr():
            weight, weight_scale = deepgemm_post_process_fp8_weight_block(...)
            wo_a._sm12x_einsum_wo_a = (weight, weight_scale, weight.data_ptr())

    On the very first forward the cache is cold, so the transform runs *inside*
    the region Dynamo is tracing. It terminates in a ctypes call
    (``utils/deep_gemm.py``), which Dynamo rejects::

        torch._dynamo.exc.Unsupported: call to a callable object with no traceable __call__
          Developer debug context: object=<_FuncPtr object at 0x...>

    and because ``torch.compile(fullgraph=True)`` is unconditional in
    ``compilation/wrapper.py``, that ends engine init. The same first forward is
    ``profile_run``, so the pack has to happen strictly before it -- the
    ``kernel_warmup`` extension below runs too late, after ``profile_run``.

    Reproduces the traced call exactly, which is why it is safe: the same
    keyword arguments, the same ``use_e8m0=True``, and ``bmm_batch_size`` from
    ``attention.py`` (``self.wo_a.bmm_batch_size = self.n_local_groups``) rather
    than a guess at ``n_groups``. Layers whose ``bmm_batch_size`` was never set
    are skipped, leaving the original lazy path untouched.

    ``deepgemm_post_process_fp8_weight_block`` returns a *view* of ``wo_a.weight``
    in the ``is_bmm`` branch, so ``weight.data_ptr()`` equals
    ``wo_a.weight.data_ptr()`` and the cache sentinel still matches.
    """
    if not _is_deepseek_v4(model):
        return

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    packed = 0
    skipped = 0
    started = time.perf_counter()
    try:
        for module in model.modules():
            wo_a = getattr(module, "wo_a", None)
            if wo_a is None or not hasattr(wo_a, "weight"):
                continue
            block = getattr(wo_a, "weight_block_size", None)
            wq = wo_a.weight
            if not block or wq.dtype != torch.float8_e4m3fn:
                continue
            ws = getattr(wo_a, "weight_scale", None)
            if ws is None:
                ws = getattr(wo_a, "weight_scale_inv", None)
            if ws is None:
                continue
            g = int(getattr(wo_a, "bmm_batch_size", 0) or 0)
            if g <= 0:
                skipped += 1
                continue
            cached = getattr(wo_a, "_sm12x_einsum_wo_a", None)
            if cached is not None and cached[2] == wq.data_ptr():
                continue
            with torch.inference_mode():
                weight, weight_scale = deepgemm_post_process_fp8_weight_block(
                    wq=wq,
                    ws=ws,
                    quant_block_shape=tuple(block),
                    use_e8m0=True,
                    is_bmm=True,
                    bmm_batch_size=g,
                )
            wo_a._sm12x_einsum_wo_a = (weight, weight_scale, weight.data_ptr())
            packed += 1
    except Exception as exc:  # warmup must never take the worker down
        logger.warning(
            "DSv4 wo_a DeepGEMM einsum pre-pack failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        return
    if packed or skipped:
        logger.info(
            "DSv4 wo_a DeepGEMM einsum pre-pack: %d packed, %d skipped, %.2f s.",
            packed,
            skipped,
            time.perf_counter() - started,
        )


def deepseek_v4_deepgemm_allow_in_graph() -> None:
    """Make DeepGEMM's lazily-resolved entry points visible to Dynamo.

    ``vllm/utils/deep_gemm.py`` resolves its implementations lazily and then
    forwards with a bare ``return _impl(*args, **kwargs)`` (``fp8_einsum`` is
    ``deep_gemm.py:501``). The implementations live in the third-party
    ``deep_gemm`` package, which Dynamo skip-lists, so the forward is fatal under
    ``torch.compile(fullgraph=True)``:

        torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
          Hint: ... if it is traceable, use `torch.compiler.allow_in_graph`.

    Do **not** be tempted by the reference image here: its ``utils/deep_gemm.py``
    has a byte-identical ``fp8_einsum`` wrapper and no registration either, and its
    ``deep_gemm_fp8_o_proj`` calls ``fp8_einsum`` unconditionally -- so the call is
    simply not skip-listed in that build. Ours is.

    Registration must land strictly before the first traced forward. It cannot go
    in ``_lazy_init`` (Dynamo would trace the registration itself) and it cannot go
    in ``kernel_warmup``, which runs after ``profile_run`` has already compiled.
    So it is called from the same pre-profile hook as the wo_a pre-pack.

    .. warning::
       **Measured on 2026-09-17: this is necessary but NOT sufficient, and on its
       own it moves the failure rather than clearing it.** It does clear
       ``Attempted to call function marked as skipped``, and the next failure is
       torch telling us the same thing more precisely::

           torch._dynamo.exc.ObservedRuntimeError: Dynamo failed to run FX node with
           fake tensors: call_function <built-in method fp8_einsum ...>
             got RuntimeError("Cannot access data pointer of Tensor (e.g. FakeTensor,
             FunctionalTensor). ... it is likely that we are erroneously tracing into
             a custom kernel. To fix this, please wrap the custom kernel into an
             opaque custom op.")

       ``allow_in_graph`` makes the call a graph *leaf*, and Dynamo then executes
       that leaf with FakeTensors to derive metadata -- which a kernel that reads
       ``data_ptr()`` cannot survive. This is the same wall the repo already hit
       with TileLang's ``allow_in_graph`` and the unhashable SymInt. **The working
       remedy is an opaque custom op with an explicit fake impl**, which is what
       ``fp8_einsum`` still needs. Registered entry points are harmless to leave in
       place; they simply do not finish the job.
    """
    try:
        import torch._dynamo as dynamo

        from vllm.utils import deep_gemm as dg

        dg._lazy_init()
        allowed = 0
        skipped = 0
        for name in [n for n in vars(dg) if n.endswith("_impl")]:
            fn = getattr(dg, name, None)
            if fn is None:
                continue
            if getattr(fn, "_b12x_allowed", False):
                continue
            try:
                dynamo.allow_in_graph(fn)
                fn._b12x_allowed = True
                allowed += 1
            except Exception:
                skipped += 1
        logger.info(
            "DeepGEMM Dynamo registration: %d entry points allowed, %d refused.",
            allowed,
            skipped,
        )
    except Exception as exc:  # never take the worker down
        logger.warning(
            "DeepGEMM Dynamo registration failed: %s: %s", type(exc).__name__, exc
        )
