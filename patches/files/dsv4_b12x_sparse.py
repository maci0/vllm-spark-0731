# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 sparse MLA through b12x compressed MLA (SM12x).

Stock vLLM main has no B12X_MLA_SPARSE enum. This module registers that
name onto the DSV4 584-byte packed page (fp8_ds_mla / nvfp4_ds_mla
envelope). b12x compressed MLA already documents 584 B/token; it is not
the GLM 432/368 NVFP4 writer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.compressor_utils import get_dspark_swa_index_width

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)

_DSV4_TOKEN_BYTES = 584
_DECODE_MAX_CHUNKS = 16
_PREFILL_MAX_CHUNKS = 2
_DECODE_MAX_ROWS = 128
# b12x SM120 prefill MG (also used for Spark decode at rows>=16):
# SWA-only topk in {128, 512, 1024, 2048}; dual-cache main topk must be 128.
# 0731 SWA window is 128; DSpark pads the index tensor to 192.
_PREFILL_SWA_TOPK = 128
_PREFILL_SWA_TOPK_FP8 = 512
_b12x_plans: dict[tuple[Any, ...], tuple[Any, torch.Tensor]] = {}


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def as_page_bytes(cache: torch.Tensor) -> tuple[torch.Tensor, int]:
    """View a DSV4 packed cache as b12x [pages, page_bytes], plus page_size."""
    if cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        cache = cache.view(torch.uint8)
    if cache.dtype != torch.uint8:
        raise TypeError(f"DSV4 packed cache must be uint8, got {cache.dtype}")
    if cache.dim() == 4:
        cache = cache.squeeze(-2)
    if cache.dim() == 3:
        pages, page_size, width = cache.shape
        if int(width) != _DSV4_TOKEN_BYTES:
            raise ValueError(
                f"DSV4 page token width must be {_DSV4_TOKEN_BYTES}, got {width}"
            )
        return cache.reshape(pages, page_size * width), int(page_size)
    if cache.dim() == 2:
        page_bytes = int(cache.shape[1])
        if page_bytes % _DSV4_TOKEN_BYTES != 0:
            raise ValueError(
                f"DSV4 page byte width {page_bytes} is not a multiple of "
                f"{_DSV4_TOKEN_BYTES}"
            )
        return cache, page_bytes // _DSV4_TOKEN_BYTES
    raise ValueError(f"unexpected DSV4 cache rank {cache.dim()}: {tuple(cache.shape)}")


def _index_2d(indices: torch.Tensor | None) -> torch.Tensor | None:
    if indices is None:
        return None
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    return indices.to(torch.int32).contiguous()


def _lens_i32(lens: torch.Tensor, rows: int) -> torch.Tensor:
    out = lens.to(torch.int32).reshape(-1).contiguous()
    if int(out.numel()) != rows:
        raise ValueError(f"topk lengths have {out.numel()} rows, expected {rows}")
    return out


def _fit_index_width(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad or slice a [rows, W] index matrix to a kernel-legal width."""
    rows, cur = indices.shape
    lengths = lengths.clamp(max=width)
    if cur == width:
        return indices, lengths
    out = indices.new_full((rows, width), -1)
    n = min(cur, width)
    if n:
        out[:, :n] = indices[:, :n]
    return out, lengths


def _max_sparse_width(vllm_config: VllmConfig, window_size: int) -> int:
    spec_k = 0
    spec = vllm_config.speculative_config
    if spec is not None:
        spec_k = int(getattr(spec, "num_speculative_tokens", 0) or 0)
    swa = get_dspark_swa_index_width(window_size, spec_k)
    max_len = int(vllm_config.model_config.max_model_len)
    c128 = _cdiv(_cdiv(max_len, 128), 128) * 128
    index_topk = int(getattr(vllm_config.model_config.hf_config, "index_topk", 512))
    return int(swa) + max(int(index_topk), int(c128))


def _get_b12x_plan(
    *,
    device: torch.device,
    num_q_heads: int,
    max_q_rows: int,
    max_width: int,
    page_size: int,
    max_chunks: int,
) -> tuple[Any, torch.Tensor]:
    if device.type != "cuda" or device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (
        device.index,
        int(num_q_heads),
        int(max_q_rows),
        int(max_width),
        int(page_size),
        int(max_chunks),
    )
    cached = _b12x_plans.get(key)
    if cached is not None:
        return cached
    from b12x.attention.compressed_sparse_mla import Caps, plan

    caps = Caps(
        device=device,
        num_q_heads=int(num_q_heads),
        max_q_rows=int(max_q_rows),
        max_width=int(max_width),
        page_size=int(page_size),
        max_chunks_per_row=int(max_chunks),
        decode_row_capacity=int(max_q_rows),
    )
    planned = plan(caps)
    spec = planned.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    _b12x_plans[key] = (planned, scratch)
    logger.info_once(
        "B12X_MLA_SPARSE compressed MLA scratch %s bytes "
        "(heads=%d rows=%d width=%d page=%d chunks=%d)",
        int(scratch.numel()),
        int(num_q_heads),
        int(max_q_rows),
        int(max_width),
        int(page_size),
        int(max_chunks),
    )
    return planned, scratch


class DeepseekV4B12xMLASparseBackend(DeepseekV4FlashInferMLASparseBackend):
    """Same 584-byte DSV4 page as FlashInfer DSV4; kernels come from b12x."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA_SPARSE"

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if device_capability.major != 12:
            return "B12X_MLA_SPARSE requires SM12x"
        if kv_cache_dtype not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
            "nvfp4_ds_mla",
        ):
            return "kv_cache_dtype not supported"
        return None


class DeepseekV4B12xSM120Attention(DeepseekV4FlashInferSM120Attention):
    """FlashInfer DSV4 control path, b12x compressed MLA kernels."""

    backend_cls = DeepseekV4B12xMLASparseBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = True

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        DeepseekV4Attention.__init__(self, vllm_config, *args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
        self._b12x_max_rows = int(vllm_config.scheduler_config.max_num_batched_tokens)
        self._b12x_max_width = _max_sparse_width(vllm_config, self.window_size)
        self._b12x_page_size = 64
        capture = getattr(
            vllm_config.compilation_config, "max_cudagraph_capture_size", None
        )
        self._b12x_decode_rows = max(_DECODE_MAX_ROWS, int(capture or 0), 8)

    def _reserve_empty_forward_workspace(self) -> None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
        common = dict(
            device=device,
            num_q_heads=int(self.padded_heads),
            max_width=self._b12x_max_width,
            page_size=self._b12x_page_size,
        )
        _get_b12x_plan(
            max_q_rows=self._b12x_decode_rows,
            max_chunks=_DECODE_MAX_CHUNKS,
            **common,
        )
        _get_b12x_plan(
            max_q_rows=self._b12x_max_rows,
            max_chunks=_PREFILL_MAX_CHUNKS,
            **common,
        )

    def _run_b12x(
        self,
        *,
        q: torch.Tensor,
        output: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        extra_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_lens: torch.Tensor | None,
        prefill: bool,
    ) -> None:
        from b12x.attention.compressed_sparse_mla import bind, run

        q = self._prepare_query(q, output)
        rows = int(q.shape[0])
        # Indices are raw slot ids (page * page_size + offset). FlashInfer and
        # b12x both address the cache with the tensor's page size, not the
        # C4/C128 block_size used only to build those slot ids.
        swa_pages, swa_ps = as_page_bytes(swa_cache)
        swa_page_size = swa_ps
        swa_idx = _index_2d(swa_indices)
        assert swa_idx is not None
        swa_len = _lens_i32(swa_lens, rows)
        indexed_pages = None
        indexed_idx = None
        indexed_len = None
        indexed_page_size = None
        if extra_cache is not None:
            indexed_pages, extra_ps = as_page_bytes(extra_cache)
            indexed_page_size = extra_ps
            indexed_idx = _index_2d(extra_indices)
            if indexed_idx is None or extra_lens is None:
                raise RuntimeError(
                    "compressed sparse MLA requires extra indices and lengths"
                )
            indexed_len = _lens_i32(extra_lens, rows)

        # Decode with rows>=16 on SM121 reuses the prefill MG kernel
        # (chunks <= 10). SWA-only 192 is illegal there; pad to 512.
        # Dual-cache prefill requires main topk == 128. Dual-cache decode
        # with 192+512 is 11 chunks, so it stays on the decode kernel.
        indexed_width = int(indexed_idx.shape[1]) if indexed_idx is not None else 0
        swa_chunks = (int(swa_idx.shape[1]) + 63) // 64
        indexed_chunks = (indexed_width + 63) // 64
        hits_prefill_kernel = prefill or (
            rows >= 16
            and int(q.shape[1]) == 32
            and int(swa_page_size) == 64
            and (swa_chunks + indexed_chunks) <= 10
        )
        if hits_prefill_kernel:
            if indexed_idx is not None:
                swa_idx, swa_len = _fit_index_width(
                    swa_idx, swa_len, _PREFILL_SWA_TOPK
                )
            else:
                swa_idx, swa_len = _fit_index_width(
                    swa_idx, swa_len, _PREFILL_SWA_TOPK_FP8
                )

        need_width = max(
            self._b12x_max_width,
            int(swa_idx.shape[1])
            + (int(indexed_idx.shape[1]) if indexed_idx is not None else 0),
        )
        if prefill or rows > self._b12x_decode_rows:
            planned, scratch = _get_b12x_plan(
                device=q.device,
                num_q_heads=int(q.shape[1]),
                max_q_rows=max(self._b12x_max_rows, rows),
                max_width=need_width,
                page_size=int(swa_page_size),
                max_chunks=_PREFILL_MAX_CHUNKS,
            )
        else:
            planned, scratch = _get_b12x_plan(
                device=q.device,
                num_q_heads=int(q.shape[1]),
                max_q_rows=max(self._b12x_decode_rows, rows),
                max_width=need_width,
                page_size=int(swa_page_size),
                max_chunks=_DECODE_MAX_CHUNKS,
            )
        binding = bind(
            planned,
            scratch=scratch,
            q=q,
            swa_indices=swa_idx,
            swa_lengths=swa_len,
            indexed_indices=indexed_idx,
            indexed_lengths=indexed_len,
        )
        binding.scratch.mode = "extend" if prefill else "decode"
        sink = self.attn_sink
        if sink.dtype != torch.float32:
            sink = sink.float()
        sink = sink.detach().contiguous()
        out = output if output.is_contiguous() else output.contiguous()
        result = run(
            swa_k_cache=swa_pages,
            binding=binding,
            sm_scale=self.scale,
            swa_page_size=int(swa_page_size),
            indexed_k_cache=indexed_pages,
            indexed_page_size=indexed_page_size,
            attn_sink=sink,
            expected_num_q_heads=int(q.shape[1]),
            out=out,
        )
        if result is not None and result.data_ptr() != output.data_ptr():
            output.copy_(result)
        elif out.data_ptr() != output.data_ptr():
            output.copy_(out)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        extra_cache = kv_cache if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        self._run_b12x(
            q=q,
            output=output,
            swa_cache=self.swa_cache_layer.kv_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            extra_cache=extra_cache,
            extra_indices=extra_sparse_indices,
            extra_lens=extra_sparse_lengths,
            prefill=False,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio <= 1
        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif self.compress_ratio == 4:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "C4A prefill requires top-k indices from the indexer."
                )
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            if attn_metadata is None:
                raise RuntimeError("C128A prefill metadata is missing.")
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if local_topk_indices is not None:
            if attn_metadata is None:
                raise RuntimeError("C4A prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError("C4A prefill request mapping is missing.")
            if swa_metadata.is_valid_token is None:
                raise RuntimeError("C4A prefill validity metadata is missing.")
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // self.compress_ratio
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        extra_kv = None if swa_only else compressed_k_cache
        if extra_kv is None and not swa_only:
            raise RuntimeError(
                "Compressed sparse MLA layers require their compressed KV cache."
            )

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            extra_idx_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_len_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )
            if extra_kv is not None and extra_idx_chunk is None:
                raise RuntimeError(
                    "Compressed sparse MLA prefill requires compressed sparse indices."
                )
            self._run_b12x(
                q=q[query_start:query_end],
                output=output[query_start:query_end],
                swa_cache=swa_k_cache,
                swa_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                swa_lens=swa_metadata.prefill_swa_lens[query_start:query_end],
                extra_cache=extra_kv,
                extra_indices=extra_idx_chunk,
                extra_lens=extra_len_chunk,
                prefill=True,
            )
