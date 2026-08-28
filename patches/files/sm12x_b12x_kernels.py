# SPDX-License-Identifier: Apache-2.0
"""SM12x b12x helpers: paged NSA indexer logits and WO-projection o_proj.

Copied onto the image as vllm/utils/sm12x_b12x_kernels.py. Decode indexer
pages in vLLM are manager 256 x 132-byte tokens (128 fp8 + 4 scale). The
b12x paged kernel wants page_size 64 and packed [pages, 8448] (K then
scales). o_proj on SM12x is otherwise PyTorch einsum dequant.
"""

from __future__ import annotations

import os
from typing import Any

import torch

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
# b12x uses the scheduled paged scorer when max_pages >= 1024 and q_rows <= 8.
# The 1-row kernel is the only scheduled path that is a win on this pin.
# Multi-row (q_rows 2-8) measured slower than the unscheduled 1023-page
# scorer for DSpark 1-way (6 tokens, often padded to 8). 8-way capture is
# 48 rows, so uses_paged_schedule is already false. Do not plan inside
# CUDA-graph capture (frozen warmup seqlens). If the vLLM buffer is missing
# or q_rows != 1, trim one page so decode stays unscheduled. Last 64 tokens
# of a 65536 context are then not indexed.
_B12X_SCHEDULE_MIN_PAGES = 1024
_B12X_SCHEDULE_MAX_Q_ROWS = 1


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
    """Use the vLLM-filled schedule only for the b12x 1-row scorer."""
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
    if kv_cache.dim() == 4:
        raw = kv_cache[:, :, 0, :_TOKEN_BYTES]
    else:
        raw = kv_cache[..., :_TOKEN_BYTES]
    flat_tokens = raw.reshape(-1, _TOKEN_BYTES)
    tok_idx = clamped.clamp(max=flat_tokens.shape[0] - 1)
    tok = flat_tokens[tok_idx]
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
                    f"w={tuple(w.shape)} (wa={tuple(w.shape)} sa={tuple(scale.shape)}) "
                    f"want [g={n_groups}, r={o_lora_rank}, d={group_width}]",
                    flush=True,
                )
            return None
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
    dbg_n = getattr(try_b12x_wo_proj, "_n", 0) + 1
    try_b12x_wo_proj._n = dbg_n
    if dbg_n <= 8:
        wa = getattr(wo_a, "weight", None)
        wb = getattr(wo_b, "weight", None)
        sa = getattr(wo_a, "weight_scale", None)
        if sa is None:
            sa = getattr(wo_a, "weight_scale_inv", None)
        print(
            "DBG wo_proj ENTRY#%d: o=%s dim=%d g=%d hpg=%d nope=%d rope=%d gw=%d "
            "rank=%d wa=%s sa=%s wb=%s capt=%s"
            % (
                dbg_n, tuple(o_in.shape), o_in.dim(), n_groups, heads_per_group,
                nope_dim, rope_dim, group_width, o_lora_rank,
                tuple(wa.shape) if wa is not None else None,
                tuple(sa.shape) if sa is not None else None,
                tuple(wb.shape) if wb is not None else None,
                torch.cuda.is_current_stream_capturing(),
            ),
            flush=True,
        )
    if o_in.dim() == 2 and o_in.shape[-1] == n_heads * head_dim:
        o_in = o_in.view(o_in.shape[0], n_heads, head_dim)
    elif o_in.dim() == 4:
        o_in = o_in.reshape(o_in.shape[0], n_heads, head_dim)
    if o_in.dim() != 3 or o_in.shape[0] > 256:
        er = getattr(try_b12x_wo_proj, "_er", 0)
        if er < 8:
            try_b12x_wo_proj._er = er + 1
            print(
                f"DBG wo_proj EARLY-RETURN#{er + 1}: dim={o_in.dim()} shape={tuple(o_in.shape)}",
                flush=True,
            )
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
