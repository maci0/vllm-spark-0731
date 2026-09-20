# SPDX-License-Identifier: Apache-2.0
"""SM12x b12x helpers: paged NSA indexer logits and WO-projection o_proj.

Copied onto the image as vllm/utils/sm12x_b12x_kernels.py. Decode indexer
pages in vLLM are manager 256 x 132-byte tokens (128 fp8 + 4 scale). The
b12x paged kernel wants page_size 64 and packed [pages, 8448] (K then
scales). o_proj on SM12x is otherwise PyTorch einsum dequant.
"""

from __future__ import annotations

import contextlib
import functools
import os
from typing import Any

import torch

# --- Dynamo cannot trace builtin `print` -------------------------------------
# There are 24 diagnostic `print(...)` calls left in this module, and inside a
# `torch.compile(fullgraph=True)` capture a single one is fatal:
#
#   torch._dynamo.exc.Unsupported: Encountered graph break when attempting to
#   trace CALL: Dynamo does not know how to trace builtin operator `print` with
#   argument types ['str'] (has_kwargs True)
#
# Found by the round-58 enumeration run at lines 330, 807 and 1093, all on the
# decode path. Deleting 24 blocks would be a larger and riskier diff than it
# looks -- some sit in exception handlers that log why a fallback fired -- so
# shadow the builtin with a module-level no-op instead. Dynamo then traces a
# trivial function returning None rather than an opaque builtin. Set
# `B12X_DEBUG=1` to get the prints back; the read happens once at import, so the
# guard is a constant at trace time.
if os.environ.get("B12X_DEBUG", "") != "1":

    def print(*args: Any, **kwargs: Any) -> None:  # noqa: A001
        """No-op stand-in for the builtin; see the note above."""
        return None


_INDEX_HEAD_DIM = 128
_PAGE_SIZE = 64
_SCALE_BYTES = 4
_PACKED_PAGE_BYTES = _PAGE_SIZE * (_INDEX_HEAD_DIM + _SCALE_BYTES)  # 8448
_PACKED_K_BYTES = _PAGE_SIZE * _INDEX_HEAD_DIM  # 8192
_TOKEN_BYTES = _INDEX_HEAD_DIM + _SCALE_BYTES  # 132

_o_proj_fail_logged = False
_paged_fail_logged = False
_paged_ok_logged = False
_paged_sched_logged = False
_paged_shapes_logged: set[tuple[int, int, bool]] = set()
_QUANT_GROUP = 128
_tgd_ws: torch.Tensor | None = None
_fp32_ws: torch.Tensor | None = None
_scale_ws: torch.Tensor | None = None
_wo_plan: Any = None
_wo_scratch: torch.Tensor | None = None
_wo_max_tokens = 0
_wo_run_ok_logged = False
_o_ws: torch.Tensor | None = None
_cs_ws: torch.Tensor | None = None
_even_src: torch.Tensor | None = None
_odd_src: torch.Tensor | None = None
_mul_a: torch.Tensor | None = None
_mul_b: torch.Tensor | None = None
_acc_a: torch.Tensor | None = None
_acc_b: torch.Tensor | None = None
_a_ws: torch.Tensor | None = None
_z_ws: torch.Tensor | None = None
_flat_ws: torch.Tensor | None = None
_wo_bmm_ok_logged = False
_packed_sidecars: dict[int, torch.Tensor] = {}
_insert_ok_logged = False
_SIDECAR_MIN_OVERFLOW = 1024
_arange128: torch.Tensor | None = None
_arange4: torch.Tensor | None = None
_paged_sched_cache: dict[tuple[int, str], torch.Tensor] = {}
# b12x uses the scheduled paged scorer when max_pages >= 1024 and q_rows <= 8
# (verified on the -029 image: uses_paged_schedule(rows, 4096) is True for
# rows 1-8 and False from 16 up). The old 1-row cap left even the 8-row
# DSpark c1 batch on the unscheduled 1023-page scorer, trimmed by
# trim_page_table_skip_schedule. Multi-row (2-8) was measured slower than the
# unscheduled scorer on the v0.28 pin; that is what the scheduled path has to
# beat again here. Do not plan inside CUDA-graph capture (frozen warmup
# seqlens). If the vLLM buffer is missing or q_rows > 8, trim one page so
# decode stays unscheduled; the indexer writes an empty schedule for those
# shapes, which is what keeps a stale plan from being consumed later.
_B12X_SCHEDULE_MIN_PAGES = 1024
_B12X_SCHEDULE_MAX_Q_ROWS = 8


def expand_block_table_to_page64(
    block_tables: torch.Tensor,
    block_size: int,
    page_size: int = _PAGE_SIZE,
) -> torch.Tensor:
    """Remap manager block ids to 64-token kernel page ids."""
    if block_size == page_size:
        return block_tables
    if block_size % page_size != 0:
        raise ValueError(
            f"block_size {block_size} is not a multiple of page_size {page_size}"
        )
    ratio = block_size // page_size
    sub = torch.arange(ratio, device=block_tables.device, dtype=block_tables.dtype)
    expanded = block_tables.unsqueeze(-1) * ratio + sub
    return expanded.reshape(block_tables.shape[0], block_tables.shape[1] * ratio)


def trim_page_table_skip_schedule(page_table: torch.Tensor) -> torch.Tensor:
    """Drop the last page so b12x does not pick the scheduled 1-way scorer."""
    if int(page_table.shape[1]) < _B12X_SCHEDULE_MIN_PAGES:
        return page_table
    return page_table[:, : _B12X_SCHEDULE_MIN_PAGES - 1].contiguous()


def _consume_vllm_paged_schedule(
    *,
    need_sched: bool,
    q_rows: int,
    schedule_ok: bool,
) -> bool:
    """Use the vLLM-filled schedule for every shape b12x schedules."""
    return (
        need_sched
        and schedule_ok
        and 1 <= int(q_rows) <= _B12X_SCHEDULE_MAX_Q_ROWS
    )


def _usable_b12x_schedule(
    meta: torch.Tensor | None, device: torch.device
) -> bool:
    """True when vLLM already filled a b12x (num_sms+1, 2) int32 schedule."""
    if meta is None:
        return False
    if meta.device != device:
        return False
    if meta.dtype != torch.int32 or meta.ndim != 2 or int(meta.shape[-1]) != 2:
        return False
    if int(meta.shape[0]) < 2:
        return False
    return True


def _indexer_k_pages_view(kv_cache: torch.Tensor) -> torch.Tensor | None:
    """[kernel_pages, 64, 132] uint8 view over interleaved indexer K cache."""
    if kv_cache.dim() == 4:
        raw = kv_cache[:, :, 0, :]
    elif kv_cache.dim() == 3:
        raw = kv_cache
    else:
        return None
    if raw.shape[-1] < _TOKEN_BYTES:
        return None
    raw = raw[..., :_TOKEN_BYTES]
    pages, block_size, _width = raw.shape
    if block_size % _PAGE_SIZE != 0:
        return None
    kernel_pages = pages * (block_size // _PAGE_SIZE)
    x = raw.reshape(kernel_pages, _PAGE_SIZE, _TOKEN_BYTES)
    if x.dtype != torch.uint8:
        x = x.view(torch.uint8)
    return x


def pack_indexer_k_pages(kv_cache: torch.Tensor) -> torch.Tensor | None:
    """Pack [P, S, 1, 132] interleaved tokens into [P*S/64, 8448] K-then-scale."""
    x = _indexer_k_pages_view(kv_cache)
    if x is None:
        return None
    kernel_pages = int(x.shape[0])
    packed = _packed_workspace(x.device, kernel_pages)
    if packed is None:
        return None
    packed[:, : _PAGE_SIZE * _INDEX_HEAD_DIM] = x[:, :, :_INDEX_HEAD_DIM].reshape(
        kernel_pages, -1
    )
    packed[:, _PAGE_SIZE * _INDEX_HEAD_DIM :] = x[:, :, _INDEX_HEAD_DIM:].reshape(
        kernel_pages, -1
    )
    return packed


def pack_indexer_k_pages_from_ids(
    kv_cache: torch.Tensor, page_ids: torch.Tensor
) -> torch.Tensor | None:
    """Pack only kernel pages listed in ``page_ids`` (row-major)."""
    x = _indexer_k_pages_view(kv_cache)
    if x is None:
        return None
    ids = page_ids.reshape(-1).to(dtype=torch.int32)
    ids = ids.clamp(0, x.shape[0] - 1)
    return _pack_k_page_rows(x, ids)


def _pack_k_page_rows(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor | None:
    n = int(ids.numel())
    if n == 0:
        return None
    packed = _packed_workspace(x.device, n)
    if packed is None:
        return None
    rows = x[ids]
    packed[:, : _PAGE_SIZE * _INDEX_HEAD_DIM] = rows[:, :, :_INDEX_HEAD_DIM].reshape(
        n, -1
    )
    packed[:, _PAGE_SIZE * _INDEX_HEAD_DIM :] = rows[:, :, _INDEX_HEAD_DIM:].reshape(
        n, -1
    )
    return packed


def _packed_workspace(device: torch.device, pages: int) -> torch.Tensor | None:
    ws = getattr(_packed_workspace, "_ws", None)
    if ws is not None and ws.shape[0] >= pages and ws.device == device:
        return ws[:pages]
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return None
    ws = torch.empty(pages, _PACKED_PAGE_BYTES, dtype=torch.uint8, device=device)
    _packed_workspace._ws = ws
    return ws


def _local_page_table(page_table: torch.Tensor) -> torch.Tensor | None:
    """Row-major 0..N-1 table matching compact packed pages."""
    m, n = int(page_table.shape[0]), int(page_table.shape[1])
    need = m * n
    ws = getattr(_local_page_table, "_ws", None)
    if ws is None or ws.numel() < need or ws.device != page_table.device:
        if page_table.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None
        cap = max(need, 4096)
        ws = torch.arange(cap, dtype=torch.int32, device=page_table.device)
        _local_page_table._ws = ws
    return ws[:need].view(m, n)


def _kernel_page_count(kv_cache: torch.Tensor) -> int | None:
    view = _indexer_k_pages_view(kv_cache)
    if view is None:
        return None
    return int(view.shape[0])


def lookup_packed_indexer_k(kv_cache: torch.Tensor) -> torch.Tensor | None:
    """Packed [kernel_pages, 8448] sidecar maintained at insert, or None."""
    n_pages = _kernel_page_count(kv_cache)
    if n_pages is None:
        return None
    sc = _packed_sidecars.get(kv_cache.data_ptr())
    if sc is None or sc.device != kv_cache.device or sc.shape[0] < n_pages:
        return None
    return sc[:n_pages]


# --- runtime flag probes, off the traced graph -------------------------------
# `b12x_skip_flag` and the marker check in `_indexer_direct_gather` both probe the
# filesystem, and `os.path.exists` is a builtin Dynamo skip-lists. Reached from
# inside the traced forward that is fatal:
#
#   attention.py:1131  if b12x_skip_flag("indexer_all"):
#   sm12x_b12x_kernels.py:1396  return os.path.exists(os.path.join(root, "skip_" + name))
#   Unsupported: Attempted to call function marked as skipped
#
# `/cache/runtime` is a host bind mount and the whole point of the markers is that
# they can be created or removed while the server runs, so the answer cannot simply
# be frozen. Instead the traced path reads `_RUNTIME_FILES`, a snapshot that
# `b12x_refresh_runtime_flags()` retakes, while the eager path keeps `os.path.exists`
# exactly as it was -- so every skip-flag instrument from rounds 39-48 behaves
# identically in the shipping `CompilationMode.NONE` regime.
_RUNTIME_FILES: frozenset[str] = frozenset()


def b12x_refresh_runtime_flags() -> frozenset[str]:
    """Retake the runtime-flag snapshot. Safe to call at any time."""
    global _RUNTIME_FILES
    root = os.environ.get("VLLM_SKIP_FLAG_DIR", "/cache/runtime")
    try:
        _RUNTIME_FILES = frozenset(os.listdir(root))
    except OSError:
        _RUNTIME_FILES = frozenset()
    return _RUNTIME_FILES


def _indexer_direct_gather() -> bool:
    """Opt-in fast path for the packed-sidecar gather.

    The default flatten+gather reshapes a strided slice of the KV cache, which
    materialises a full-cache copy on every layer of every decode step (~57
    ms/step at 1 row). Reading the cache directly removes it and measured
    10.1 -> 25.5 tok/s on the no-spec arm, but draft acceptance drops from ~60%
    to ~41-50%, so it stays off until that is understood.

    VLLM_B12X_INDEXER_DIRECT_GATHER=1 enables it. A marker file at
    $VLLM_SKIP_FLAG_DIR/indexer-direct-gather also enables it, so a running
    server can be switched within one boot for an A/B under identical state.
    """
    if os.environ.get("VLLM_B12X_INDEXER_DIRECT_GATHER", "0") == "1":
        return True
    if torch.compiler.is_compiling():
        # os.path.exists is a Dynamo skip-listed builtin; see the note above.
        return "indexer-direct-gather" in _RUNTIME_FILES
    root = os.environ.get("VLLM_SKIP_FLAG_DIR", "/cache/runtime")
    return os.path.exists(os.path.join(root, "indexer-direct-gather"))


def sync_packed_indexer_k(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor | None:
    """Scatter newly inserted interleaved tokens into the packed sidecar.

    slot_mapping is block_id * block_size + offset, the same ids passed to
    ops.indexer_k_quant_and_cache. Decode then reads the sidecar without
    packing the whole cache.
    """
    global _insert_ok_logged, _arange128, _arange4
    n_pages = _kernel_page_count(kv_cache)
    if n_pages is None or slot_mapping is None or slot_mapping.numel() == 0:
        return None
    capturing = bool(
        kv_cache.is_cuda and torch.cuda.is_current_stream_capturing()
    )
    T = int(slot_mapping.numel())
    overflow = max(T, _SIDECAR_MIN_OVERFLOW)
    need = n_pages + overflow
    key = kv_cache.data_ptr()
    sc = _packed_sidecars.get(key)
    if sc is None or sc.device != kv_cache.device or sc.shape[0] < need:
        if capturing:
            return None
        sc = torch.zeros(
            need, _PACKED_PAGE_BYTES, dtype=torch.uint8, device=kv_cache.device
        )
        _packed_sidecars[key] = sc
    if _arange128 is None or _arange128.device != kv_cache.device:
        if capturing:
            return None
        _arange128 = torch.arange(128, dtype=torch.int64, device=kv_cache.device)
        _arange4 = torch.arange(4, dtype=torch.int64, device=kv_cache.device)
    slots = slot_mapping.reshape(-1).to(dtype=torch.int64)
    valid = slots >= 0
    block_size = int(kv_cache.shape[1])
    ratio = block_size // _PAGE_SIZE
    clamped = slots.clamp(min=0)
    block_id = torch.div(clamped, block_size, rounding_mode="floor")
    off = clamped - block_id * block_size
    page_off = torch.div(off, _PAGE_SIZE, rounding_mode="floor")
    kpage = block_id * ratio + page_off
    within = off - page_off * _PAGE_SIZE
    dummy = n_pages + torch.arange(T, device=kv_cache.device, dtype=torch.int64)
    dummy = dummy.clamp(max=sc.shape[0] - 1)
    kpage = torch.where(valid, kpage.clamp(max=n_pages - 1), dummy)
    within = torch.where(valid, within, torch.zeros_like(within))
    if _indexer_direct_gather():
        # Gather only the T inserted tokens. Flattening the sliced cache first
        # materialises a full-cache copy here on every layer of every step.
        blocks = int(kv_cache.shape[0])
        blk = block_id.clamp(max=blocks - 1)
        if kv_cache.dim() == 4:
            tok = kv_cache[blk, off, 0, :_TOKEN_BYTES]
        else:
            tok = kv_cache[blk, off, :_TOKEN_BYTES]
    else:
        if kv_cache.dim() == 4:
            raw = kv_cache[:, :, 0, :_TOKEN_BYTES]
        else:
            raw = kv_cache[..., :_TOKEN_BYTES]
        flat_tokens = raw.reshape(-1, _TOKEN_BYTES)
        tok = flat_tokens[clamped.clamp(max=flat_tokens.shape[0] - 1)]
    k_idx = kpage.unsqueeze(1) * _PACKED_PAGE_BYTES + within.unsqueeze(1) * _INDEX_HEAD_DIM + _arange128
    s_idx = (
        kpage.unsqueeze(1) * _PACKED_PAGE_BYTES
        + _PAGE_SIZE * _INDEX_HEAD_DIM
        + within.unsqueeze(1) * _SCALE_BYTES
        + _arange4
    )
    flat = sc.view(-1)
    flat.index_copy_(0, k_idx.reshape(-1), tok[:, :_INDEX_HEAD_DIM].reshape(-1))
    flat.index_copy_(0, s_idx.reshape(-1), tok[:, _INDEX_HEAD_DIM:].reshape(-1))
    if not _insert_ok_logged:
        print(
            f"b12x packed indexer insert ok sidecar={tuple(sc.shape)} "
            f"n_pages={n_pages} T={T}",
            flush=True,
        )
        _insert_ok_logged = True
    return sc[:n_pages]


def view_as_packed_indexer_k(kv_cache: torch.Tensor) -> torch.Tensor | None:
    """View manager blocks as b12x 64-token packed pages (K then scale).

    Same storage as interleaved [blocks, 256, 132]. Valid only after the
    compressor store kernel writes that packed layout in place.
    """
    if kv_cache.dim() == 4:
        blocks, block_size, _, width = kv_cache.shape
    elif kv_cache.dim() == 3:
        blocks, block_size, width = kv_cache.shape
    else:
        return None
    if block_size % _PAGE_SIZE != 0 or width < _TOKEN_BYTES:
        return None
    n_pages = blocks * (block_size // _PAGE_SIZE)
    need = n_pages * _PACKED_PAGE_BYTES
    raw = kv_cache.view(torch.uint8).reshape(-1)
    if raw.numel() < need:
        return None
    return raw[:need].view(n_pages, _PACKED_PAGE_BYTES)


def try_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    schedule_metadata: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Score DSA indexer logits with the b12x paged kernel. None = caller fallback.

    Uses the packed sidecar written at insert. Does not pack the whole
    index-K cache on decode (that path was slower than gather).
    """
    global _paged_fail_logged, _paged_ok_logged, _paged_sched_logged
    if os.environ.get("VLLM_USE_B12X_SPARSE_INDEXER", "1") == "0":
        return None
    q_values, q_scale = q
    if q_scale is not None:
        return None
    if kv_cache.dim() < 3:
        return None
    block_size = int(kv_cache.shape[1])
    if block_size % _PAGE_SIZE != 0:
        return None
    B = int(block_tables.shape[0])
    if q_values.dim() == 4:
        _bq, next_n, heads, dim = q_values.shape
        q_fp8 = q_values.reshape(_bq * next_n, heads, dim)
    else:
        q_fp8 = q_values
        heads, dim = q_fp8.shape[-2], q_fp8.shape[-1]
        next_n = q_fp8.shape[0] // B if B > 0 else 1
    if q_fp8.dtype != torch.float8_e4m3fn or dim != _INDEX_HEAD_DIM:
        return None
    m_rows = int(q_fp8.shape[0])
    if B == 0 or m_rows == 0:
        return None
    try:
        from b12x.attention.dsa_indexer import (
            PagedDecodeMetadata,
            logits_paged,
            prepare_paged_metadata,
            uses_paged_schedule,
        )
    except Exception as exc:
        if not _paged_fail_logged:
            print(f"b12x paged indexer import: {type(exc).__name__}: {exc}", flush=True)
            _paged_fail_logged = True
        return None

    try:
        if (
            block_size != _PAGE_SIZE
            and int(block_tables.shape[1]) * _PAGE_SIZE >= int(max_model_len)
        ):
            page_table = block_tables
        else:
            page_table = expand_block_table_to_page64(block_tables, block_size)
    except ValueError:
        return None
    if page_table.shape[0] == B and m_rows == B * next_n and next_n > 1:
        page_table = page_table.repeat_interleave(next_n, dim=0)
    if page_table.shape[0] != m_rows:
        return None
    packed = lookup_packed_indexer_k(kv_cache)
    if packed is None:
        packed = view_as_packed_indexer_k(kv_cache)
    local_table = None
    if packed is None:
        return None
    if context_lens.dim() == 2 and context_lens.shape[1] == next_n:
        seqlens = context_lens.to(dtype=torch.int32).reshape(m_rows)
    else:
        ctx = context_lens[:, -1] if context_lens.dim() == 2 else context_lens
        seqlens = (
            ctx.to(dtype=torch.int32).unsqueeze(1).expand(B, next_n).reshape(m_rows)
        )
    if local_table is not None:
        page_table_i32 = local_table
    else:
        page_table_i32 = page_table.to(torch.int32).contiguous()
    seqlens_i32 = seqlens.contiguous()
    w = weights[:m_rows].to(torch.float32)
    if w.shape != (m_rows, heads):
        w = w.reshape(m_rows, heads)
    q_c = q_fp8.contiguous()
    w_c = w.contiguous()
    try:
        need_sched = bool(
            uses_paged_schedule(
                q_rows=int(page_table_i32.shape[0]),
                max_pages=int(page_table_i32.shape[1]),
            )
        )
        sched = None
        used_vllm_sched = False
        q_rows = int(page_table_i32.shape[0])
        if _consume_vllm_paged_schedule(
            need_sched=need_sched,
            q_rows=q_rows,
            schedule_ok=_usable_b12x_schedule(
                schedule_metadata, page_table_i32.device
            ),
        ):
            sched = schedule_metadata
            if not sched.is_contiguous():
                sched = sched.contiguous()
            used_vllm_sched = True
        elif need_sched:
            page_table_i32 = trim_page_table_skip_schedule(page_table_i32)
            seqlens_i32 = torch.clamp(
                seqlens_i32,
                max=int(page_table_i32.shape[1]) * _PAGE_SIZE,
            )
            need_sched = False
        prepared = prepare_paged_metadata(
            real_page_table=page_table_i32,
            cache_seqlens_int32=seqlens_i32,
            page_size=_PAGE_SIZE,
            validate_raw_lengths=False,
            schedule_metadata=sched,
            build_schedule=False,
        )
        decode_meta = PagedDecodeMetadata(
            real_page_table=prepared.real_page_table,
            cache_seqlens_int32=prepared.cache_seqlens_int32,
            paged_mqa_schedule_metadata=getattr(
                prepared, "schedule_metadata", None
            ),
        )
        scored = logits_paged(
            q_fp8=q_c,
            weights=w_c,
            index_k_cache=packed,
            metadata=decode_meta,
            page_size=_PAGE_SIZE,
        )
    except Exception as exc:
        if not _paged_fail_logged:
            print(f"b12x paged indexer fallback: {type(exc).__name__}: {exc}", flush=True)
            _paged_fail_logged = True
        return None
    shape_key = (
        int(page_table_i32.shape[0]),
        int(page_table_i32.shape[1]),
        bool(used_vllm_sched),
    )
    if shape_key not in _paged_shapes_logged and len(_paged_shapes_logged) < 16:
        _paged_shapes_logged.add(shape_key)
        print(
            f"b12x paged indexer ok packed={tuple(packed.shape)} "
            f"table={tuple(page_table_i32.shape)} sched={used_vllm_sched} "
            f"q_rows={q_rows} view=packed-at-store",
            flush=True,
        )
        _paged_ok_logged = True
        if used_vllm_sched:
            _paged_sched_logged = True
    n_out = min(int(scored.shape[1]), max_model_len)
    if (
        int(scored.shape[0]) == m_rows
        and n_out == max_model_len
        and scored.dtype == torch.float32
    ):
        return scored
    logits = q_c.new_full((m_rows, max_model_len), float("-inf"), dtype=torch.float32)
    if n_out > 0:
        logits[:, :n_out] = scored[:, :n_out]
    return logits


def _dequant_grouped_fp8(
    o_fp8: torch.Tensor,
    o_scale: torch.Tensor,
) -> torch.Tensor | None:
    """Dequant fused_inv_rope_fp8_quant output to bf16 [T, G, D]."""
    global _tgd_ws, _fp32_ws, _scale_ws
    if o_fp8.dim() != 3:
        return None
    tokens, groups, width = o_fp8.shape
    nblocks = width // _QUANT_GROUP
    if width % _QUANT_GROUP != 0:
        return None
    scale = o_scale
    if scale.dim() != 3 or scale.shape[0] != tokens or scale.shape[1] != groups:
        return None
    if scale.shape[-1] < nblocks:
        return None
    scale = scale[..., :nblocks]
    capturing = torch.cuda.is_current_stream_capturing()

    def _ok(ws: torch.Tensor | None, dtype: torch.dtype) -> bool:
        return (
            ws is not None
            and ws.device == o_fp8.device
            and ws.dtype == dtype
            and ws.shape[0] >= tokens
            and ws.shape[1] == groups
            and ws.shape[2] == width
        )

    if not (_ok(_tgd_ws, torch.bfloat16) and _ok(_fp32_ws, torch.float32) and _ok(_scale_ws, torch.float32)):
        if capturing:
            return None
        cap_t = max(tokens, 64)
        if _tgd_ws is None:
            cap_t = max(cap_t, 256)
        _tgd_ws = torch.empty(
            (cap_t, groups, width), dtype=torch.bfloat16, device=o_fp8.device
        )
        _fp32_ws = torch.empty(
            (cap_t, groups, width), dtype=torch.float32, device=o_fp8.device
        )
        _scale_ws = torch.empty(
            (cap_t, groups, width), dtype=torch.float32, device=o_fp8.device
        )
    tgd = _tgd_ws[:tokens]
    fp32 = _fp32_ws[:tokens]
    sc_ws = _scale_ws[:tokens]
    sc_view = sc_ws.view(tokens, groups, nblocks, _QUANT_GROUP)
    sc_src = scale if scale.dtype == torch.float32 else scale.to(torch.float32)
    if capturing and sc_src.data_ptr() != scale.data_ptr() and scale.dtype != torch.float32:
        return None
    sc_view.copy_(sc_src.unsqueeze(-1).expand(tokens, groups, nblocks, _QUANT_GROUP))
    fp32.copy_(o_fp8)
    fp32.mul_(sc_ws)
    tgd.copy_(fp32)
    return tgd


def _expand_block_scales(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    s = scale.to(torch.float32)
    if s.dim() == 1:
        s = s.view(-1, 1)
    if s.shape[-1] != cols:
        s = s.repeat_interleave(cols // s.shape[-1], dim=-1)
    if s.dim() >= 2 and s.shape[-2] != rows:
        s = s.repeat_interleave(rows // s.shape[-2], dim=-2)
    return s[..., :rows, :cols]


def b12x_profile_decode_once(fn):
    """One-shot timing for the first real DSpark decode step.

    Enabled by ``VLLM_PROFILE_DECODE=1``. CUDA-event timing only:
    torch.profiler's kineto/CUPTI activity segfaults against vLLM's
    always-on SyncActivityProfilerHandler. Prints step wall/GPU split and
    arms the per-layer timing (``b12x_profile_layer``) for this step.
    """

    import os

    if os.environ.get("VLLM_PROFILE_DECODE") != "1":
        return fn

    def wrapper(self, *args, **kwargs):
        import time

        import torch

        if torch.cuda.is_current_stream_capturing() or _DECODE_PROFILED[0]:
            return fn(self, *args, **kwargs)
        _PROFILING_STEP[0] = True
        # This hook wraps the DSpark DRAFT (`DFlashSpeculator._run_model`). It must
        # not clobber `_LAYER_EVENTS[0]`, which the target's 43 decoder layers write
        # into and `_b12x_print_layers()` reports: the draft forward runs after the
        # target's `execute_model` in the same engine step, so resetting the shared
        # list here made the reported layer count the draft's three MTP blocks.
        _DRAFT_PHASE[0] = True
        _DRAFT_LAYER_EVENTS[0] = []
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = fn(self, *args, **kwargs)
            e1.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            wall_ms = 1000.0 * (t1 - t0)
            gpu_ms = e0.elapsed_time(e1)
            ntoks = int(args[0]) if args else "?"
            layer_ms = [a.elapsed_time(b) for a, b in _DRAFT_LAYER_EVENTS[0]]
            print(
                f"b12x draft step: toks={ntoks} wall={wall_ms:.1f}ms "
                f"gpu={gpu_ms:.1f}ms overhead={wall_ms - gpu_ms:.1f}ms",
                flush=True,
            )
            if layer_ms:
                n = len(layer_ms)
                total = sum(layer_ms)
                print(
                    f"b12x draft layers: n={n} sum={total:.1f}ms avg={total / n:.2f}ms "
                    f"max={max(layer_ms):.2f}ms@L{layer_ms.index(max(layer_ms))} "
                    f"p95={sorted(layer_ms)[int(n * 0.95) - 1]:.2f}ms",
                    flush=True,
                )
            _DECODE_PROFILED[0] = True
        finally:
            _PROFILING_STEP[0] = False
            _DRAFT_PHASE[0] = False
        return out

    return wrapper


#: Set by b12x_profile_decode_once around the profiled step so the per-layer
#: timing below only prints for that one step.
_PROFILING_STEP = [False]

#: True only while the DSpark draft's `_run_model` is executing. The draft and the
#: target both run `DeepseekV4DecoderLayer.forward`, so the layer hook needs to know
#: which model it is timing or the two sets of events interleave in one list.
_DRAFT_PHASE = [False]

#: The draft's own (e0, e1) pairs. Kept separate from `_LAYER_EVENTS` so the target's
#: 43 layers survive into `_b12x_print_layers()`; the draft's reset used to discard
#: them and its three MTP layers were reported as "n=3" in their place.
_DRAFT_LAYER_EVENTS = [[]]

#: b12x_profile_decode_once runs per draft forward (up to k+1 times per engine step)
#: on the eager path; the docstring promises one-shot, so stop after the first real
#: print rather than printing the same step repeatedly.
_DECODE_PROFILED = [False]


def b12x_profile_layer(fn):
    """Per-layer CUDA-event timing, active only inside the profiled step.

    Events are recorded per layer on the current stream WITHOUT per-layer
    synchronize (that distorted timing via aux-stream waits); the step
    decorator's final synchronize settles everything.
    """

    import os

    if os.environ.get("VLLM_PROFILE_DECODE") != "1":
        return fn

    def wrapper(self, *args, **kwargs):
        import torch

        if not _b12x_region_active(_b12x_tokens_of(args, kwargs)):
            return fn(self, *args, **kwargs)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        try:
            e0.record()
        except Exception:
            _CAPTURE_PROFILE["on"] = False
            return fn(self, *args, **kwargs)
        out = fn(self, *args, **kwargs)
        e1.record()
        # The draft and the target are both DeepseekV4DecoderLayer, so route the pair
        # to whichever phase is running instead of letting them share one list.
        (_DRAFT_LAYER_EVENTS if _DRAFT_PHASE[0] else _LAYER_EVENTS)[0].append((e0, e1))
        return out

    return wrapper


#: (e0, e1) pairs per layer, consumed by the step decorator after sync.
_LAYER_EVENTS = [[]]


def _cached_wo_a_bmm_weight(
    wo_a: Any,
    n_groups: int,
    o_lora_rank: int,
    group_width: int,
) -> torch.Tensor | None:
    """Dequant WO-A to bf16 [G, D, R] once (same expand as SM12x fp8_einsum).

    vLLM's deepgemm is_bmm post-processing (deepgemm_post_process_fp8_weight_block)
    stores the local shard as 3D [G, R, D] with 3D scale [G, R/128, D/128];
    the raw checkpoint is 2D [G*R, D] with 2D scale. Handle both.
    """
    cached = getattr(wo_a, "_b12x_w_bmm", None)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None
    w = wo_a.weight
    scale = wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    if w.dim() == 3:
        g, r, d = w.shape
        if not (g == n_groups and r == o_lora_rank and d == group_width):
            global _w_bmm_none_logged
            if not _w_bmm_none_logged:
                _w_bmm_none_logged = True
                print(
                    "DBG wo_proj w_bmm NONE (3D): "
                    f"w={tuple(w.shape)} (wa={tuple(w.shape)} sa={tuple(scale.shape)} {scale.dtype}) "
                    f"want [g={n_groups}, r={o_lora_rank}, d={group_width}]",
                    flush=True,
                )
            return None
        if (
            scale.dim() == 3
            and scale.dtype == torch.int32
            and scale.shape[2] * 4 == group_width // 128
        ):
            # DeepGEMM MN-major TMA-aligned packed UE8M0 scale
            # [G, R, D/512] int32: each int32 packs 4 ue8m0 exponents, byte j
            # = k-block 4i+j (little-endian); rows are already per-gran-block
            # broadcast. Unpack -> [G, R, 32] fp32 -> repeat to [G, R, D].
            s32 = scale.contiguous()  # data order == contiguous (copy_ applied)
            e = s32.view(torch.uint8).to(torch.int32)  # [G, R, D/512, 4]
            fp = (e << 23).view(torch.float32)  # [G, R, D/512, 4], k-block = 4i+j
            s = fp.reshape(g, r, -1).repeat_interleave(128, dim=-1)  # [G, R, D]
        elif (
            scale.dim() == 3
            and scale.dtype == torch.uint8
            and scale.shape[2] == group_width // 128
        ):
            # Raw UE8M0 exponents [G, R, D/128] (no packing).
            e = scale.to(torch.int32)
            fp = (e << 23).view(torch.float32)  # [G, R, D/128]
            s = fp.repeat_interleave(128, dim=-1)  # [G, R, D]
        else:
            # [G, R/128, D/128] -> [G, R, D]
            s = _expand_block_scales(scale, o_lora_rank, group_width)
        w_dq = w.to(torch.bfloat16) * s.to(dtype=torch.bfloat16, device=w.device)
        # [G, R, D] -> [G, D, R] for bmm(a[G,T,D], w[G,D,R])
        w_bmm = w_dq.transpose(1, 2).contiguous()
    else:
        rows, cols = int(w.shape[0]), int(w.shape[1])
        s = _expand_block_scales(scale, rows, cols)
        w_dq = w.to(torch.bfloat16) * s.to(dtype=torch.bfloat16, device=w.device)
        if w_dq.shape[0] == n_groups * o_lora_rank and w_dq.shape[1] == group_width:
            # [G*R, D] -> [G, R, D] -> [G, D, R] for bmm(a[G,T,R], w[G,R,D])
            w_bmm = w_dq.view(n_groups, o_lora_rank, group_width).transpose(1, 2).contiguous()
        else:
            global _w_bmm_none_logged2
            if not _w_bmm_none_logged2:
                _w_bmm_none_logged2 = True
                print(
                    "DBG wo_proj w_bmm NONE (2D): "
                    f"w_dq={tuple(w_dq.shape)} (wa={tuple(w.shape)} sa={tuple(scale.shape)}) "
                    f"want rows=G*R={n_groups * o_lora_rank} cols=gw={group_width}",
                    flush=True,
                )
            return None
    wo_a._b12x_w_bmm = w_bmm
    print(
        f"b12x wo_proj w_bmm ok {tuple(w_bmm.shape)} from wa={tuple(w.shape)} sa={tuple(scale.shape)}",
        flush=True,
    )
    return w_bmm


def _ensure_bmm_ws(
    device: torch.device,
    tokens: int,
    groups: int,
    group_width: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    global _a_ws, _z_ws, _flat_ws
    capturing = torch.cuda.is_current_stream_capturing()

    def _ok(ws: torch.Tensor | None, shape: tuple[int, ...]) -> bool:
        return (
            ws is not None
            and ws.device == device
            and ws.dtype == torch.bfloat16
            and ws.shape[0] >= shape[0]
            and ws.shape[1:] == shape[1:]
        )

    need_a = (groups, tokens, group_width)
    need_z = (groups, tokens, rank)
    need_f = (tokens, groups * rank)

    def _ok3(ws: torch.Tensor | None, n0: int, n1: int, n2: int) -> bool:
        return (
            ws is not None
            and ws.device == device
            and ws.dtype == torch.bfloat16
            and ws.dim() == 3
            and ws.shape[0] == n0  # groups must match exactly (shape[1] is a token cap)
            and ws.shape[1] >= n1
            and ws.shape[2] == n2
        )

    def _ok2(ws: torch.Tensor | None, n0: int, n1: int) -> bool:
        return (
            ws is not None
            and ws.device == device
            and ws.dtype == torch.bfloat16
            and ws.dim() == 2
            and ws.shape[0] >= n0
            and ws.shape[1] == n1
        )

    if not (
        _ok3(_a_ws, groups, tokens, group_width)
        and _ok3(_z_ws, groups, tokens, rank)
        and _ok2(_flat_ws, tokens, groups * rank)
    ):
        if capturing:
            return None
        cap_t = max(tokens, 64)
        if _a_ws is None:
            cap_t = max(cap_t, 256)
        _a_ws = torch.empty((groups, cap_t, group_width), dtype=torch.bfloat16, device=device)
        _z_ws = torch.empty((groups, cap_t, rank), dtype=torch.bfloat16, device=device)
        _flat_ws = torch.empty((cap_t, groups * rank), dtype=torch.bfloat16, device=device)
    return _a_ws[:, :tokens], _z_ws[:, :tokens], _flat_ws[:tokens]


def _inv_rope_tgd(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor | None:
    """Match vLLM fused inv-RoPE, stay in bf16, return [T, G, group_width].

    Rope lives in the last ``rope_dim`` of each head. Pair xor-1, even:
    x*cos + partner*sin; odd: x*cos - partner*sin.
    """
    global _o_ws, _cs_ws, _even_src, _odd_src, _mul_a, _mul_b, _acc_a, _acc_b
    if o.dim() != 3:
        return None
    tokens, n_heads, head_dim = o.shape
    if n_heads != n_groups * heads_per_group:
        return None
    if head_dim != nope_dim + rope_dim or rope_dim % 2 != 0:
        return None
    half = rope_dim // 2
    capturing = torch.cuda.is_current_stream_capturing()

    def _ok3(ws: torch.Tensor | None, shape: tuple[int, ...], dtype: torch.dtype) -> bool:
        return (
            ws is not None
            and ws.device == o.device
            and ws.dtype == dtype
            and ws.shape[0] >= shape[0]
            and ws.shape[1:] == shape[1:]
        )

    need_o = (tokens, n_heads, head_dim)
    need_pair = (tokens, n_heads, half)
    need_cs = (tokens, rope_dim)
    if not (
        _ok3(_o_ws, need_o, o.dtype)
        and _ok3(_cs_ws, need_cs, torch.float32)
        and _ok3(_even_src, need_pair, o.dtype)
        and _ok3(_odd_src, need_pair, o.dtype)
        and _ok3(_mul_a, need_pair, o.dtype)
        and _ok3(_mul_b, need_pair, o.dtype)
        and _ok3(_acc_a, need_pair, o.dtype)
        and _ok3(_acc_b, need_pair, o.dtype)
    ):
        if capturing:
            return None
        cap_t = max(tokens, 64)
        if _o_ws is None:
            cap_t = max(cap_t, 8192)
        _o_ws = torch.empty((cap_t, n_heads, head_dim), dtype=o.dtype, device=o.device)
        _cs_ws = torch.empty((cap_t, rope_dim), dtype=torch.float32, device=o.device)
        _even_src = torch.empty((cap_t, n_heads, half), dtype=o.dtype, device=o.device)
        _odd_src = torch.empty((cap_t, n_heads, half), dtype=o.dtype, device=o.device)
        _mul_a = torch.empty((cap_t, n_heads, half), dtype=o.dtype, device=o.device)
        _mul_b = torch.empty((cap_t, n_heads, half), dtype=o.dtype, device=o.device)
        _acc_a = torch.empty((cap_t, n_heads, half), dtype=o.dtype, device=o.device)
        _acc_b = torch.empty((cap_t, n_heads, half), dtype=o.dtype, device=o.device)
    o_ws = _o_ws[:tokens]
    o_ws.copy_(o)
    pos = positions.reshape(-1)[:tokens]
    if pos.dtype != torch.int64:
        if capturing:
            return None
        pos = pos.to(torch.int64)
    idx = pos.view(tokens, 1).expand(tokens, rope_dim)
    torch.gather(cos_sin_cache, 0, idx, out=_cs_ws[:tokens])
    rope = o_ws[:, :, nope_dim:]
    even = rope[:, :, 0::2]
    odd = rope[:, :, 1::2]
    even_s = _even_src[:tokens]
    odd_s = _odd_src[:tokens]
    even_s.copy_(even)
    odd_s.copy_(odd)
    cos = _mul_a[:tokens]
    sin = _mul_b[:tokens]
    cos.copy_(_cs_ws[:tokens, :half].view(tokens, 1, half).expand(tokens, n_heads, half))
    sin.copy_(_cs_ws[:tokens, half:].view(tokens, 1, half).expand(tokens, n_heads, half))
    acc_e = _acc_a[:tokens]
    acc_o = _acc_b[:tokens]
    # even' = even*cos + odd*sin ; odd' = odd*cos - even*sin (vLLM fused kernel).
    torch.mul(even_s, cos, out=acc_e)
    torch.addcmul(acc_e, odd_s, sin, out=acc_e)
    torch.mul(odd_s, cos, out=acc_o)
    torch.addcmul(acc_o, even_s, sin, value=-1, out=acc_o)
    even.copy_(acc_e)
    odd.copy_(acc_o)
    return o_ws.view(tokens, n_groups, heads_per_group * head_dim)


def _ensure_wo_scratch(
    device: torch.device,
    tokens: int,
    *,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
) -> tuple[Any, torch.Tensor] | tuple[None, None]:
    """Persistent caller-owned WO scratch. Allocates only when not capturing."""
    global _wo_plan, _wo_scratch, _wo_max_tokens
    capturing = torch.cuda.is_current_stream_capturing()
    if (
        _wo_plan is not None
        and _wo_scratch is not None
        and _wo_scratch.device == device
        and _wo_max_tokens >= tokens
    ):
        return _wo_plan, _wo_scratch
    if capturing:
        return None, None
    from b12x.gemm.wo_projection import Caps, plan

    need = max(int(tokens), int(_wo_max_tokens), 64)
    if _wo_max_tokens == 0:
        need = max(need, 8192)
    caps = Caps(
        device=device,
        max_tokens=need,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )
    _wo_plan = plan(caps)
    spec = _wo_plan.scratch_specs()[0]
    _wo_scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    _wo_max_tokens = need
    print(
        f"b12x wo_proj scratch ok tokens={need} nbytes={_wo_scratch.numel()}",
        flush=True,
    )
    return _wo_plan, _wo_scratch


_DECODE_SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64)


def _warmup_wo_decode_sizes(
    plan: Any,
    scratch: torch.Tensor,
    tgd: torch.Tensor,
    packed: Any,
) -> None:
    """Compile b12x WO kernels for CUDA-graph decode sizes before capture."""
    from b12x.gemm.wo_projection import bind, run

    max_m = int(tgd.shape[0])
    for m in _DECODE_SIZES:
        if m > max_m:
            break
        binding = bind(
            plan,
            scratch=scratch,
            source_tgd=tgd[:m],
            weights=packed,
            expected_m=m,
        )
        run(binding=binding)
    print("b12x wo_proj decode warmup ok", flush=True)


def try_b12x_wo_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: Any,
    wo_b: Any,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor | None:
    """Fused inv-RoPE FP8 + same dequant as SM12x einsum, then grouped bmm.

    b12x MXFP8 WO does not match block-FP8 einsum (chat loops). Keep the
    fused quant + dequant math, replace torch.einsum with bmm + cached WO-A.
    """
    if os.environ.get("VLLM_USE_B12X_WO_PROJECTION", "1") == "0":
        return None
    try:
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
            fused_inv_rope_fp8_quant,
        )
    except Exception:
        return None
    o_in = o
    head_dim = nope_dim + rope_dim
    n_heads = n_groups * heads_per_group
    group_width = heads_per_group * head_dim
    if o_in.dim() == 2 and o_in.shape[-1] == n_heads * head_dim:
        o_in = o_in.view(o_in.shape[0], n_heads, head_dim)
    elif o_in.dim() == 4:
        o_in = o_in.reshape(o_in.shape[0], n_heads, head_dim)
    # Kept as a plain shape guard: returning None hands the layer back to the
    # einsum path. The DBG prints that used to sit here were removed because
    # torch.cuda.is_current_stream_capturing() returns a Python bool and that is a
    # fatal `torch.* op returned non-Tensor` graph break under fullgraph compile.
    if o_in.dim() != 3 or o_in.shape[0] > 256:
        return None
    try:
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o_in,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            tma_aligned_scales=False,
        )
        fk = getattr(try_b12x_wo_proj, "_fk", 0)
        if fk < 8:
            try_b12x_wo_proj._fk = fk + 1
            print(
                f"DBG wo_proj FUSED#{fk + 1}: o_fp8={tuple(o_fp8.shape)} {o_fp8.dtype} "
                f"o_scale={tuple(o_scale.shape)} {o_scale.dtype}",
                flush=True,
            )
        tgd = _dequant_grouped_fp8(o_fp8, o_scale)
        if tgd is None:
            return None
        w_bmm = _cached_wo_a_bmm_weight(
            wo_a, n_groups, o_lora_rank, group_width
        )
        if w_bmm is None:
            return None
        ok = getattr(try_b12x_wo_proj, "_ok", 0)
        if ok < 8:
            try_b12x_wo_proj._ok = ok + 1
            print(
                f"DBG wo_proj OK#{ok + 1}: o={tuple(o.shape)} g={n_groups} hpg={heads_per_group} "
                f"gw={group_width} rank={o_lora_rank} tgd={tuple(tgd.shape)} w_bmm={tuple(w_bmm.shape)}",
                flush=True,
            )
        tokens = int(tgd.shape[0])
        ws = _ensure_bmm_ws(
            tgd.device, tokens, n_groups, group_width, o_lora_rank
        )
        if ws is None:
            return None
        a_ws, z_ws, flat_ws = ws
        a_ws.copy_(tgd.permute(1, 0, 2))
        # `z_ws` is `_z_ws[:, :tokens]` out of a `(groups, cap_t, rank)` buffer, so
        # it is non-contiguous whenever tokens < cap_t, and Dynamo refuses an
        # `out=` into a non-contiguous tensor:
        #
        #   Attempted to call op with non-contiguous `out=` tensor
        #     torch.bmm(a_ws, w_bmm, out=z_ws)
        #
        # Compute into a fresh tensor and copy. Gated on is_compiling(), which
        # Dynamo folds to a constant, so the eager path keeps the zero-copy `out=`
        # form and only the compiled path pays the extra (groups, tokens, rank)
        # copy.
        if torch.compiler.is_compiling():
            z_ws.copy_(torch.bmm(a_ws, w_bmm))
        else:
            torch.bmm(a_ws, w_bmm, out=z_ws)
        d = o_lora_rank
        for g in range(n_groups):
            flat_ws[:, g * d : (g + 1) * d].copy_(z_ws[g])
        out = wo_b(flat_ws)
        global _wo_bmm_ok_logged
        if not _wo_bmm_ok_logged:
            print(
                f"b12x wo_proj bmm ok tgd={tuple(tgd.shape)} z={tuple(z_ws.shape)} out={tuple(out.shape)}",
                flush=True,
            )
            _wo_bmm_ok_logged = True
        return out
    except Exception as exc:
        fl = getattr(try_b12x_wo_proj, "_fl", 0)
        if fl < 3:
            try_b12x_wo_proj._fl = fl + 1
            import traceback as _tb

            _stack = _tb.format_exception(type(exc), exc, exc.__traceback__)
            print(
                f"b12x wo_proj fallback#{fl + 1} (o={tuple(o_in.shape)}): "
                f"{type(exc).__name__}: {exc}\n"
                + "".join(_stack[-8:]),
                flush=True,
            )
        return None


def _pack_wo_weights(
    wo_a: Any,
    wo_b: Any,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    pack_weights: Any,
) -> Any | None:
    wa = wo_a.weight
    wb = wo_b.weight
    sa = wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    sb = wo_b.weight_scale if hasattr(wo_b, "weight_scale") else wo_b.weight_scale_inv
    group_width = heads_per_group * (nope_dim + rope_dim)
    hidden = int(wb.shape[0])
    # Checkpoint WO-A is [groups*rank, group_width]. pack_weights views that
    # as [groups, rank, width] itself. Do not permute here.
    if torch.cuda.is_current_stream_capturing() and getattr(wo_a, "_b12x_wo_packed", None) is None:
        return None
    try:
        return pack_weights(
            wa,
            sa,
            wb,
            sb,
            groups=n_groups,
            group_width=group_width,
            rank=o_lora_rank,
            hidden=hidden,
        )
    except Exception as exc:
        global _o_proj_fail_logged
        if not _o_proj_fail_logged:
            print(
                "b12x wo_proj pack failed "
                f"wa={tuple(wa.shape)} sa={tuple(sa.shape)} "
                f"wb={tuple(wb.shape)} sb={tuple(sb.shape)} "
                f"groups={n_groups} width={group_width} rank={o_lora_rank} "
                f"hidden={hidden}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _o_proj_fail_logged = True
        return None


def packed_gather_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    schedule_metadata: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """MQA indexer logits with a layout-correct gather over packed-at-store K.

    The packed sidecar is [page, 8448] uint8: the first 8192 bytes hold 64
    tokens of 128-byte fp8 K, the last 256 bytes hold 64 fp32 dequant scales
    (K-then-scale). The interleaved FlashInfer gather strides 132 bytes per
    token and reads this layout wrong (HANDOFF item 18: DSpark accept drops
    to 38-70% vs ~73%). This reads the correct offsets and computes

        logits[m, n] = sum_h w[m, h] * relu((q[m, h] . k[n]) * scale[n])

    with a chunked gather to bound memory. Returns None when the packed
    sidecar is unavailable (caller falls back to the paged kernel).
    """
    q_values, q_scale = q
    if q_scale is not None:
        return None
    if os.environ.get("VLLM_USE_B12X_SPARSE_INDEXER", "1") == "0":
        return None
    packed = lookup_packed_indexer_k(kv_cache)
    if packed is None:
        packed = view_as_packed_indexer_k(kv_cache)
    if packed is None or packed.dim() != 2 or int(packed.shape[1]) < _PACKED_PAGE_BYTES:
        return None
    if kv_cache.dim() < 3:
        return None
    block_size = int(kv_cache.shape[1])
    if block_size % _PAGE_SIZE != 0:
        return None
    B = int(block_tables.shape[0])
    if B == 0:
        return None
    if q_values.dim() == 4:
        _bq, next_n, heads, dim = q_values.shape
        q_fp8 = q_values.reshape(_bq * next_n, heads, dim)
    else:
        q_fp8 = q_values
        heads, dim = q_fp8.shape[-2], q_fp8.shape[-1]
        next_n = q_fp8.shape[0] // B if B > 0 else 1
    if dim != _INDEX_HEAD_DIM:
        return None
    m_rows = int(q_fp8.shape[0])
    if m_rows == 0:
        return None
    try:
        if (
            block_size != _PAGE_SIZE
            and int(block_tables.shape[1]) * _PAGE_SIZE >= int(max_model_len)
        ):
            page_table = block_tables
        else:
            page_table = expand_block_table_to_page64(block_tables, block_size)
    except ValueError:
        return None
    if page_table.shape[0] == B and m_rows == B * next_n and next_n > 1:
        page_table = page_table.repeat_interleave(next_n, dim=0)
    if page_table.shape[0] != m_rows:
        return None
    if context_lens.dim() == 2 and context_lens.shape[1] == next_n:
        seqlens = context_lens.to(dtype=torch.int32).reshape(m_rows)
    else:
        ctx = context_lens[:, -1] if context_lens.dim() == 2 else context_lens
        seqlens = (
            ctx.to(dtype=torch.int32).unsqueeze(1).expand(B, next_n).reshape(m_rows)
        )
    w = weights[:m_rows].to(torch.float32)
    if w.shape != (m_rows, heads):
        w = w.reshape(m_rows, heads)

    n_pages = int(packed.shape[0])
    k_deq = (
        packed[:, : _PACKED_K_BYTES].view(torch.float8_e4m3fn).to(torch.float32)
    )
    k_deq = k_deq.view(n_pages, _PAGE_SIZE, _INDEX_HEAD_DIM)
    s_f32 = packed[:, _PACKED_K_BYTES : _PACKED_PAGE_BYTES].view(torch.float32)
    k_deq = k_deq * s_f32.unsqueeze(-1)
    q_f32 = q_fp8.to(torch.float32)
    table = page_table.to(torch.int64).clamp(0, n_pages - 1)

    logits = q_f32.new_full((m_rows, max_model_len), float("-inf"))
    neg_inf = torch.tensor(
        float("-inf"), dtype=torch.float32, device=logits.device
    )
    chunk_pages = 64  # 64 pages x 64 tokens = 4096 tokens per gather chunk
    p = int(table.shape[1])
    for start in range(0, p, chunk_pages):
        end = min(start + chunk_pages, p)
        kc = k_deq[table[:, start:end]].reshape(m_rows, -1, _INDEX_HEAD_DIM)
        score = torch.einsum("mhd,mnd->mhn", q_f32, kc)
        part = torch.einsum("mh,mhn->mn", w, torch.relu(score))
        n = int(part.shape[1])
        pos = torch.arange(n, device=part.device) + start * _PAGE_SIZE
        valid = pos.unsqueeze(0) < seqlens.to(device=part.device).unsqueeze(1)
        part = torch.where(valid, part, neg_inf)
        out_start = start * _PAGE_SIZE
        out_end = min(out_start + n, max_model_len)
        if out_end > out_start:
            logits[:, out_start:out_end] = part[:, : out_end - out_start]
    return logits


# Per-step target decode profiler (VLLM_PROFILE_DECODE=1).
#
# b12x_profile_decode_once/b12x_profile_layer cover the DSpark draft
# (_run_model) only, which the no-spec arm never calls. These wrap the model
# runner instead: forward (execute_model), sampling (sample_tokens/sample) and
# the host gap in between. Nothing is synchronized per step (that would
# serialize host and device); the window's events resolve behind one sync when
# it closes. On a run without CUDA graphs the layer decorator collects as well
# and the per-layer breakdown is printed with the step.

_STEP_PROFILE: dict[str, Any] = {
    "limit": int(os.environ.get("VLLM_PROFILE_DECODE_STEPS", "12")),
    "steps": 0,
    "samples": [],
    "last_end": 0.0,
    "printed": False,
}

#: device time per named region of the profiled step, filled by the
#: b12x_profile_region marks (decoder layer, indexer, WO, all-reduce...).
_REGION_EVENTS: dict[str, list[tuple[Any, Any, float]]] = {}

#: VLLM_PROFILE_CAPTURE=1: arm the region marks while the CUDA graph is
#: captured, so the event records become graph nodes and the next replay
#: re-records them. That is the only way to get gap-free device time for a
#: graph-replayed step. Only 1-row work is recorded, so the piecewise prefill
#: captures cannot mix in.
_CAPTURE_PROFILE: dict[str, Any] = {
    "on": os.environ.get("VLLM_PROFILE_CAPTURE") == "1",
    "printed": False,
}


def _b12x_tokens_of(args: tuple, kwargs: dict) -> int | None:
    """First tensor argument's leading dim: the batch token count."""
    for value in list(args) + list(kwargs.values()):
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            try:
                return int(shape[0])
            except Exception:
                return None
    return None


def b12x_skip_flag(name: str) -> bool:
    """Diagnostic switch: /cache/runtime/skip_<name> exists.

    /cache/runtime is a host bind mount, so a flag file can be created or
    removed while the server runs. Correctness is lost while one is set:
    these only measure what a stage costs.

    Under `torch.compile` the probe reads the `_RUNTIME_FILES` snapshot instead,
    because `os.path.exists` is a Dynamo skip-listed builtin and this call sits
    inside the traced forward; see the note above `_RUNTIME_FILES`. The eager path
    is unchanged, so flags can still be toggled mid-run in the shipping regime.
    """
    if torch.compiler.is_compiling():
        return ("skip_" + name) in _RUNTIME_FILES
    root = os.environ.get("VLLM_SKIP_FLAG_DIR", "/cache/runtime")
    return os.path.exists(os.path.join(root, "skip_" + name))


def _b12x_tokens_value(value: Any) -> int | None:
    """Token count from an int or a tensor's leading dim."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        try:
            return int(shape[0])
        except Exception:
            return None
    return None


def _b12x_region_active(tokens: int | None) -> bool:
    if _PROFILING_STEP[0]:
        return True
    if not _CAPTURE_PROFILE["on"] or tokens != 1:
        return False
    return bool(torch.cuda.is_current_stream_capturing())


@contextlib.contextmanager
def b12x_profile_region(name: str, tokens: int | None = None):
    """CUDA-event + wall time for one named region of the profiled forward.

    ``tokens`` is the batch token count, given as an int or as a tensor whose
    leading dim is the token axis (the layer marks pass ``x``).
    """
    tokens = _b12x_tokens_value(tokens)
    if not _b12x_region_active(tokens):
        yield
        return
    import time

    import torch

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    try:
        e0.record()
    except Exception:
        _CAPTURE_PROFILE["on"] = False
        yield
        return
    try:
        yield
    finally:
        e1.record()
        t1 = time.perf_counter()
        _REGION_EVENTS.setdefault(name, []).append((e0, e1, (t1 - t0) * 1e3))


def b12x_profile_region_fn(name: str):
    """Wrap a callable in b12x_profile_region, deriving the token count."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with b12x_profile_region(name, _b12x_tokens_of(args, kwargs)):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def _b12x_print_step_profile() -> None:
    if not _STEP_PROFILE["samples"]:
        return
    torch.cuda.synchronize()
    rows = [
        (label, info, wall_ms, e0.elapsed_time(e1), gap_ms)
        for label, info, e0, e1, wall_ms, gap_ms in _STEP_PROFILE["samples"]
    ]
    _STEP_PROFILE["samples"] = []
    for label, info, wall_ms, gpu_ms, gap_ms in rows:
        print(
            f"b12x step {label}{info}: wall={wall_ms:.1f}ms gpu={gpu_ms:.1f}ms "
            f"gap={gap_ms:.1f}ms",
            flush=True,
        )
    steps = [r for r in rows if r[0] == "sample_tokens"]
    if steps:
        fwd = [r for r in rows if r[0] == "execute_model"]
        samp = [r for r in rows if r[0] == "sample"]
        step_wall = sum(r[2] for r in steps)
        gap_before_fwd = sum(r[4] for r in fwd if r[4] > 0)
        print(
            f"b12x profile window: steps={len(steps)} "
            f"wall/step={step_wall / len(steps):.1f}ms "
            f"fwd_gpu/step={sum(r[3] for r in fwd) / len(fwd):.1f}ms "
            f"samp_gpu/step={(sum(r[3] for r in samp) / len(samp)) if samp else -1:.1f}ms "
            f"gap_before_fwd/step={gap_before_fwd / len(fwd):.1f}ms",
            flush=True,
        )
    _b12x_print_detail()
    _b12x_print_layers()


def _b12x_print_layers() -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        return
    layer_ms = [a.elapsed_time(b) for a, b in _LAYER_EVENTS[0]]
    if not layer_ms:
        return
    order = sorted(range(len(layer_ms)), key=lambda i: layer_ms[i], reverse=True)
    total = sum(layer_ms)
    top = " ".join(f"L{i}={layer_ms[i]:.2f}" for i in order[:12])
    print(
        f"b12x layers: n={len(layer_ms)} sum={total:.1f}ms "
        f"avg={total / len(layer_ms):.2f}ms top: {top}",
        flush=True,
    )


def _b12x_print_detail() -> None:
    """Resolve and print the region events collected for the last forward."""
    try:
        torch.cuda.synchronize()
    except Exception:
        return
    for name in sorted(_REGION_EVENTS):
        events = _REGION_EVENTS[name]
        try:
            ms = [a.elapsed_time(b) for a, b, _ in events]
        except Exception as exc:
            print(f"b12x region {name}: unresolvable ({exc})", flush=True)
            continue
        wall = [w for _, _, w in events]
        print(
            f"b12x region {name}: n={len(ms)} gpu_sum={sum(ms):.1f}ms "
            f"gpu_min={min(ms):.2f}ms gpu_avg={sum(ms) / len(ms):.2f}ms "
            f"gpu_max={max(ms):.2f}ms wall_sum={sum(wall):.1f}ms",
            flush=True,
        )
    _REGION_EVENTS.clear()


def _b12x_capture_step(self, fn, args: tuple, kwargs: dict):
    """Capture mode: print the first real 1-row step's captured region times.

    The region/layer events were recorded while the CUDA graph was captured,
    so they are graph nodes and the replay re-records them. Nothing else is
    timed: the point is the gap-free device split inside the graph.
    """
    import time

    import torch

    if (
        _CAPTURE_PROFILE["printed"]
        or fn.__name__ != "execute_model"
        or kwargs.get("dummy_run", False)
        or torch.cuda.is_current_stream_capturing()
    ):
        return fn(self, *args, **kwargs)
    scheduler_output = args[0] if args else kwargs.get("scheduler_output")
    if (
        getattr(scheduler_output, "total_num_scheduled_tokens", None) != 1
        or not _REGION_EVENTS
    ):
        return fn(self, *args, **kwargs)
    _CAPTURE_PROFILE["printed"] = True
    t0 = time.perf_counter()
    out = fn(self, *args, **kwargs)
    wall_ms = (time.perf_counter() - t0) * 1e3
    print(
        f"b12x capture profile: 1-row replay wall={wall_ms:.1f}ms",
        flush=True,
    )
    _b12x_print_detail()
    _b12x_print_layers()
    return out


def b12x_profile_target_step(fn):
    """Wall + CUDA-event timing of one target decode step (VLLM_PROFILE_DECODE=1).

    Prints the first VLLM_PROFILE_DECODE_STEPS real (non-dummy, non-capture)
    calls as "b12x step <name>: wall/gpu/gap"; gap is the host time since the
    previous profiled call returned.
    """

    import os

    if os.environ.get("VLLM_PROFILE_DECODE") != "1":
        return fn

    def wrapper(self, *args, **kwargs):
        import time

        import torch

        if _CAPTURE_PROFILE["on"]:
            return _b12x_capture_step(self, fn, args, kwargs)
        if (
            _STEP_PROFILE["printed"]
            or kwargs.get("dummy_run", False)
            or torch.cuda.is_current_stream_capturing()
        ):
            return fn(self, *args, **kwargs)
        label = fn.__name__
        info = ""
        if label == "execute_model":
            so = args[0] if args else kwargs.get("scheduler_output")
            info = f" tok={getattr(so, 'total_num_scheduled_tokens', '?')}"
            _PROFILING_STEP[0] = True
            _LAYER_EVENTS[0] = []
            _REGION_EVENTS.clear()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        out = fn(self, *args, **kwargs)
        e1.record()
        t1 = time.perf_counter()
        if label == "execute_model":
            _PROFILING_STEP[0] = False
        last_end = _STEP_PROFILE["last_end"]
        gap_ms = (t0 - last_end) * 1e3 if last_end else -1.0
        _STEP_PROFILE["last_end"] = t1
        _STEP_PROFILE["samples"].append((label, info, e0, e1, (t1 - t0) * 1e3, gap_ms))
        if label == "sample_tokens":
            _STEP_PROFILE["steps"] += 1
            if _STEP_PROFILE["steps"] >= _STEP_PROFILE["limit"]:
                _b12x_print_step_profile()
                _STEP_PROFILE["printed"] = True
        return out

    return wrapper
