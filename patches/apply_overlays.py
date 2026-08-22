#!/usr/bin/env python3
"""Apply the v0.28.0rc2 cherry-picks onto an installed vLLM tree.

Copies new modules from patches/files/, then does unique-string edits.
Fails if a needle is missing or not unique, unless the replacement is
already present (idempotent).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"


def _vllm_dir() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if old == new:
        raise SystemExit(f"{label}: old == new")
    if old not in text:
        if new in text:
            print(f"skip {label}: already applied")
            return
        raise SystemExit(f"{label}: missing needle in {path}")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: needle not unique ({count}) in {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"ok {label}")


def insert_after(path: Path, anchor: str, addition: str, label: str) -> None:
    if addition.strip() and addition in path.read_text():
        print(f"skip {label}: already applied")
        return
    replace_once(path, anchor, anchor + addition, label)


def copy_new_modules(vllm: Path) -> None:
    dest_moe = vllm / "model_executor/layers/fused_moe/b12x.py"
    src_moe = FILES / "fused_moe_b12x.py"
    dest_prep = vllm / "model_executor/layers/quantization/utils/b12x_moe.py"
    src_prep = FILES / "b12x_moe.py"
    if not src_moe.is_file() or not src_prep.is_file():
        raise SystemExit(f"missing donor files under {FILES}")
    dest_moe.parent.mkdir(parents=True, exist_ok=True)
    dest_prep.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_moe, dest_moe)
    shutil.copyfile(src_prep, dest_prep)
    print(f"ok copied {dest_moe.relative_to(vllm.parent)}")
    print(f"ok copied {dest_prep.relative_to(vllm.parent)}")


def patch_moe_backend(vllm: Path) -> None:
    path = vllm / "config/kernel.py"
    replace_once(
        path,
        '    "flashinfer_b12x",\n    "marlin",\n    "humming",\n    "triton_unfused",',
        '    "flashinfer_b12x",\n    "b12x",\n    "marlin",\n    "humming",\n    "triton_unfused",',
        "MoEBackend +b12x",
    )
    replace_once(
        path,
        '    - "flashinfer_b12x": Use FlashInfer CuteDSL fused MoE for SM12x\n'
        "      (RTX Pro 6000 / DGX Spark)\n"
        '    - "marlin": Use Marlin kernels (weight-only quantization)',
        '    - "flashinfer_b12x": Use FlashInfer CuteDSL fused MoE for SM12x\n'
        "      (RTX Pro 6000 / DGX Spark)\n"
        '    - "b12x": Use b12x FP4 MoE kernels on SM12x\n'
        '    - "marlin": Use Marlin kernels (weight-only quantization)',
        "MoEBackend docs +b12x",
    )


def patch_envs(vllm: Path) -> None:
    path = vllm / "envs.py"
    replace_once(
        path,
        "    VLLM_HUMMING_MOE_GEMM_TYPE: Literal[\"indexed\", \"grouped\", \"auto\"] | None = None\n"
        "    VLLM_DEEPEPLL_NVFP4_DISPATCH: bool = False",
        "    VLLM_HUMMING_MOE_GEMM_TYPE: Literal[\"indexed\", \"grouped\", \"auto\"] | None = None\n"
        "    VLLM_B12X_MOE_FP4_FORCE_A16: bool = False\n"
        "    VLLM_DEEPEPLL_NVFP4_DISPATCH: bool = False",
        "envs VLLM_B12X_MOE_FP4_FORCE_A16 decl",
    )
    replace_once(
        path,
        '    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": lambda: bool(\n'
        '        int(os.getenv("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1"))\n'
        "    ),\n"
        "    # Allow use of FlashInfer MxInt4 MoE kernels for fused moe ops.",
        '    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": lambda: bool(\n'
        '        int(os.getenv("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1"))\n'
        "    ),\n"
        "    # Force b12x FP4 MoE to use BF16 activations.\n"
        '    "VLLM_B12X_MOE_FP4_FORCE_A16": lambda: bool(\n'
        '        int(os.getenv("VLLM_B12X_MOE_FP4_FORCE_A16", "0"))\n'
        "    ),\n"
        "    # Allow use of FlashInfer MxInt4 MoE kernels for fused moe ops.",
        "envs VLLM_B12X_MOE_FP4_FORCE_A16 getter",
    )


def patch_utils_b12x(vllm: Path) -> None:
    path = vllm / "utils/b12x.py"
    replace_once(
        path,
        "from collections.abc import Iterable\n"
        "from dataclasses import fields, is_dataclass\n",
        "from collections.abc import Callable, Hashable, Iterable\n"
        "from dataclasses import dataclass, fields, is_dataclass\n",
        "utils.b12x imports",
    )
    replace_once(
        path,
        "import torch\n\n\n@functools.cache\ndef has_b12x() -> bool:\n",
        "import torch\n\n\n@dataclass(frozen=True)\n"
        "class B12xWarmupUnit:\n"
        "    name: str\n"
        "    key: Hashable\n"
        "    compile: Callable[[], None]\n\n\n"
        "@functools.cache\ndef has_b12x() -> bool:\n",
        "B12xWarmupUnit",
    )
    replace_once(
        path,
        "def get_b12x_tensor_fp8_linear() -> ModuleType | None:\n"
        '    return _get_submodule("b12x.gemm.tensor_fp8_linear")\n\n\n'
        "def b12x_warmup_token_counts(\n",
        "def get_b12x_tensor_fp8_linear() -> ModuleType | None:\n"
        '    return _get_submodule("b12x.gemm.tensor_fp8_linear")\n\n\n'
        "def get_b12x_fused_moe() -> ModuleType | None:\n"
        '    return _get_submodule("b12x.moe.fused_moe")\n\n\n'
        "def b12x_warmup_token_counts(\n",
        "get_b12x_fused_moe",
    )


def patch_mxfp4_oracle(vllm: Path) -> None:
    path = vllm / "model_executor/layers/fused_moe/oracle/mxfp4.py"
    replace_once(
        path,
        "import torch\n\nimport vllm.model_executor.layers.fused_moe.modular_kernel as mk\n"
        "from vllm.config import get_current_vllm_config\n",
        "import torch\n\nimport vllm.model_executor.layers.fused_moe.modular_kernel as mk\n"
        "from vllm import envs\n"
        "from vllm.config import get_current_vllm_config\n",
        "mxfp4 import envs",
    )
    replace_once(
        path,
        'class Mxfp4MoeBackend(Enum):\n    NONE = "None"\n    # DeepGEMM FP8xFP4 backend (SM100+)\n',
        'class Mxfp4MoeBackend(Enum):\n    NONE = "None"\n'
        "    # b12x backends\n"
        '    B12X_MXFP4_MXFP8 = "B12X_MXFP4_MXFP8"\n'
        '    B12X_MXFP4_BF16 = "B12X_MXFP4_BF16"\n'
        "    # DeepGEMM FP8xFP4 backend (SM100+)\n",
        "Mxfp4MoeBackend b12x enum",
    )
    replace_once(
        path,
        "TRITON_BACKENDS = (\n"
        "    Mxfp4MoeBackend.TRITON,\n"
        "    Mxfp4MoeBackend.TRITON_UNFUSED,\n"
        ")\n\n\ndef backend_to_kernel_cls(\n",
        "TRITON_BACKENDS = (\n"
        "    Mxfp4MoeBackend.TRITON,\n"
        "    Mxfp4MoeBackend.TRITON_UNFUSED,\n"
        ")\n\n"
        "B12X_BACKENDS = (\n"
        "    Mxfp4MoeBackend.B12X_MXFP4_MXFP8,\n"
        "    Mxfp4MoeBackend.B12X_MXFP4_BF16,\n"
        ")\n\n\ndef backend_to_kernel_cls(\n",
        "B12X_BACKENDS tuple",
    )
    replace_once(
        path,
        "    backend: Mxfp4MoeBackend,\n"
        ") -> list[type[mk.FusedMoEExperts]]:\n"
        "    if backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n",
        "    backend: Mxfp4MoeBackend,\n"
        ") -> list[type[mk.FusedMoEExperts]]:\n"
        "    if backend in B12X_BACKENDS:\n"
        "        from vllm.model_executor.layers.fused_moe.b12x import B12xExperts\n"
        "\n"
        "        return [B12xExperts]\n"
        "\n"
        "    elif backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n",
        "backend_to_kernel_cls b12x",
    )
    replace_once(
        path,
        "    mapping: dict[str, list[Mxfp4MoeBackend]] = {\n"
        '        "deep_gemm": [Mxfp4MoeBackend.DEEPGEMM_MXFP4],\n',
        "    mapping: dict[str, list[Mxfp4MoeBackend]] = {\n"
        '        "b12x": list(B12X_BACKENDS),\n'
        '        "deep_gemm": [Mxfp4MoeBackend.DEEPGEMM_MXFP4],\n',
        "map_mxfp4_backend b12x",
    )
    replace_once(
        path,
        '    """Map backend to its activation key (FP8, MXFP8, or None for BF16)."""\n'
        "    if backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n"
        "        return kFp8Dynamic128Sym\n",
        '    """Map backend to its activation key (FP8, MXFP8, or None for BF16)."""\n'
        "    if backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n"
        "        return kFp8Dynamic128Sym\n"
        "    if backend == Mxfp4MoeBackend.B12X_MXFP4_MXFP8:\n"
        "        return kMxfp8Dynamic\n",
        "_backend_activation_key b12x",
    )
    replace_once(
        path,
        "    return bf16 if bf16 else backends\n\n\ndef select_mxfp4_moe_backend(\n",
        "    return bf16 if bf16 else backends\n\n\n"
        "def _get_requested_backends(\n"
        "    runner_backend: MoEBackend,\n"
        "    requested_activation_key: QuantKey | None,\n"
        ") -> list[Mxfp4MoeBackend]:\n"
        "    backends = map_mxfp4_backend(runner_backend)\n"
        '    if runner_backend == "b12x":\n'
        "        if envs.VLLM_B12X_MOE_FP4_FORCE_A16:\n"
        "            return [Mxfp4MoeBackend.B12X_MXFP4_BF16]\n"
        "        if requested_activation_key is None:\n"
        "            return backends\n"
        "    return _filter_by_activation(backends, requested_activation_key)\n\n\n"
        "def select_mxfp4_moe_backend(\n",
        "_get_requested_backends",
    )
    replace_once(
        path,
        "    runner_backend = config.moe_backend\n"
        '    if runner_backend != "auto":\n'
        "        requested_backends = map_mxfp4_backend(runner_backend)\n"
        "        if activation_format == mk.FusedMoEActivationFormat.BatchedExperts:\n"
        "            requested_backends = [\n"
        "                Mxfp4MoeBackend.BATCHED_MARLIN if b == Mxfp4MoeBackend.MARLIN else b\n"
        "                for b in requested_backends\n"
        "            ]\n"
        "        candidates = _filter_by_activation(requested_backends, requested_activation_key)\n"
        "        if not candidates:\n",
        "    runner_backend = config.moe_backend\n"
        '    if runner_backend != "auto":\n'
        "        requested_backends = _get_requested_backends(\n"
        "            runner_backend, requested_activation_key\n"
        "        )\n"
        "        if activation_format == mk.FusedMoEActivationFormat.BatchedExperts:\n"
        "            requested_backends = [\n"
        "                Mxfp4MoeBackend.BATCHED_MARLIN if b == Mxfp4MoeBackend.MARLIN else b\n"
        "                for b in requested_backends\n"
        "            ]\n"
        "        candidates = requested_backends\n"
        "        if not candidates:\n",
        "select_mxfp4_moe_backend uses _get_requested_backends",
    )
    # DeepSeek-V4 selector honors --moe-backend b12x.
    replace_once(
        path,
        '    if runner_backend != "auto":\n'
        "        requested_backends = map_mxfp4_backend(runner_backend)\n",
        '    if runner_backend != "auto":\n'
        "        requested_backends = _get_requested_backends(runner_backend, None)\n",
        "select_deepseek_v4_mxfp4_moe_backend b12x",
    )
    replace_once(
        path,
        '    """Round up hidden_size and intermediate_size based on backend requirements."""\n'
        "    if backend == Mxfp4MoeBackend.EMULATION:\n",
        '    """Round up hidden_size and intermediate_size based on backend requirements."""\n'
        "    if backend in B12X_BACKENDS:\n"
        "        return hidden_size, intermediate_size\n"
        "    if backend == Mxfp4MoeBackend.EMULATION:\n",
        "mxfp4_round_up skip b12x",
    )
    replace_once(
        path,
        "        is_gfx1250 = on_gfx1250()\n\n"
        "    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n",
        "        is_gfx1250 = on_gfx1250()\n\n"
        "    if mxfp4_backend in B12X_BACKENDS:\n"
        "        return (\n"
        "            w13_weight.data,\n"
        "            w2_weight.data,\n"
        "            w13_weight_scale.data,\n"
        "            w2_weight_scale.data,\n"
        "            w13_bias,\n"
        "            w2_bias,\n"
        "        )\n\n"
        "    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n",
        "convert_weight b12x identity",
    )
    replace_once(
        path,
        '    """Create a FusedMoEQuantConfig for the given MXFP4 backend."""\n'
        "    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n",
        '    """Create a FusedMoEQuantConfig for the given MXFP4 backend."""\n'
        "    if mxfp4_backend == Mxfp4MoeBackend.B12X_MXFP4_MXFP8:\n"
        "        return mxfp4_mxfp8_moe_quant_config(\n"
        "            w1_bias=w1_bias,\n"
        "            w2_bias=w2_bias,\n"
        "            w1_scale=w1_scale,\n"
        "            w2_scale=w2_scale,\n"
        "            gemm1_alpha=gemm1_alpha,\n"
        "            gemm1_beta=gemm1_beta,\n"
        "            gemm1_clamp_limit=swiglu_limit,\n"
        "        )\n"
        "    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n",
        "make_mxfp4_moe_quant_config B12X_MXFP4_MXFP8",
    )
    replace_once(
        path,
        "    elif mxfp4_backend in (\n"
        "        Mxfp4MoeBackend.MARLIN,\n"
        "        Mxfp4MoeBackend.BATCHED_MARLIN,\n"
        "        Mxfp4MoeBackend.TRITON,\n",
        "    elif mxfp4_backend in (\n"
        "        Mxfp4MoeBackend.B12X_MXFP4_BF16,\n"
        "        Mxfp4MoeBackend.MARLIN,\n"
        "        Mxfp4MoeBackend.BATCHED_MARLIN,\n"
        "        Mxfp4MoeBackend.TRITON,\n",
        "make_mxfp4_moe_quant_config B12X_MXFP4_BF16",
    )


def patch_mhc(vllm: Path) -> None:
    path = vllm / "model_executor/kernels/mhc/tilelang.py"
    replace_once(
        path,
        "    residual_flat = residual\n"
        "    num_tokens = residual.shape[0]\n\n"
        "    n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))\n",
        "    residual_flat = residual\n"
        "    num_tokens = residual.shape[0]\n\n"
        "    from vllm.utils.deep_gemm import is_deep_gemm_supported\n\n"
        "    use_deep_gemm = is_deep_gemm_supported()\n"
        "    if use_deep_gemm:\n"
        "        n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))\n"
        "    else:\n"
        "        n_splits = 1\n",
        "mhc_pre_broadcast n_splits guard",
    )
    replace_once(
        path,
        "    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm\n\n"
        "    tf32_hc_prenorm_gemm(\n"
        "        residual_flat,\n"
        "        fn_broadcast,\n"
        "        gemm_out_mul,\n"
        "        gemm_out_sqrsum,\n"
        "        n_splits,\n"
        "    )\n"
        "    mhc_pre_big_fuse_broadcast_with_norm_tilelang(\n",
        "    if use_deep_gemm:\n"
        "        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm\n\n"
        "        tf32_hc_prenorm_gemm(\n"
        "            residual_flat,\n"
        "            fn_broadcast,\n"
        "            gemm_out_mul,\n"
        "            gemm_out_sqrsum,\n"
        "            n_splits,\n"
        "        )\n"
        "    else:\n"
        "        _tilelang_hc_prenorm_gemm(\n"
        "            residual_flat,\n"
        "            fn_broadcast,\n"
        "            gemm_out_mul,\n"
        "            gemm_out_sqrsum,\n"
        "            hidden_size,\n"
        "            1,\n"
        "        )\n"
        "    mhc_pre_big_fuse_broadcast_with_norm_tilelang(\n",
        "mhc_pre_broadcast TileLang fallback",
    )


def patch_nvfp4_ds_mla(vllm: Path) -> None:
    cache = vllm / "config/cache.py"
    replace_once(
        cache,
        '    "fp8_ds_mla",\n    "turboquant_k8v4",\n',
        '    "fp8_ds_mla",\n    "nvfp4_ds_mla",\n    "turboquant_k8v4",\n',
        "CacheDType nvfp4_ds_mla",
    )

    torch_utils = vllm / "utils/torch_utils.py"
    replace_once(
        torch_utils,
        '    "fp8_ds_mla": torch.uint8,\n    "turboquant_k8v4": torch.uint8,\n',
        '    "fp8_ds_mla": torch.uint8,\n'
        '    "nvfp4_ds_mla": torch.uint8,\n'
        '    "turboquant_k8v4": torch.uint8,\n',
        "torch_utils nvfp4_ds_mla",
    )

    vllm_cfg = vllm / "config/vllm.py"
    replace_once(
        vllm_cfg,
        "            self.cache_config.cache_dtype.startswith(\"nvfp4\")\n"
        "            and self.model_config.use_mla\n",
        "            self.cache_config.cache_dtype == \"nvfp4\"\n"
        "            and self.model_config.use_mla\n",
        "MLA guard exact nvfp4 only",
    )

    attn = vllm / "models/deepseek_v4/attention.py"
    replace_once(
        attn,
        "def _resolve_dsv4_kv_cache_dtype(\n",
        "def _dsv4_page_alignment(kv_cache_dtype: str) -> int:\n"
        '    """Per-token page slot for a DeepSeek-V4 KV layout.\n\n'
        "    NVFP4 uses a 584-byte DSV4 envelope (same billed size as fp8_ds_mla\n"
        "    content). fp8_ds_mla page-step is 576. Plain rows are 512.\n"
        "    Indexer / compressor caches must not call this helper.\n"
        '    """\n'
        '    if kv_cache_dtype == "nvfp4_ds_mla":\n'
        "        return 584\n"
        '    if kv_cache_dtype == "fp8_ds_mla":\n'
        "        return 576\n"
        "    return 512\n\n\n"
        "def _dsv4_packed_layout(kv_cache_dtype: str) -> bool:\n"
        '    return kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")\n\n\n'
        "def _resolve_dsv4_kv_cache_dtype(\n",
        "attention _dsv4_page_alignment",
    )
    replace_once(
        attn,
        "    if use_fp8_ds_mla_layout:\n"
        "        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.\n"
        "        assert kv_cache_dtype.startswith(\"fp8\"), (\n",
        "    if use_fp8_ds_mla_layout:\n"
        "        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.\n"
        '        if kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla"):\n'
        "            if cache_config is not None:\n"
        '                cache_config.cache_dtype = "nvfp4_ds_mla"\n'
        "            logger.info_once(\n"
        '                "Using DeepSeek V4 padded nvfp4_ds_mla KV cache format."\n'
        "            )\n"
        '            return "nvfp4_ds_mla", torch.uint8\n'
        "        assert kv_cache_dtype.startswith(\"fp8\"), (\n",
        "attention resolve nvfp4_ds_mla before fp8 assert",
    )
    replace_once(
        attn,
        "        uses_fp8_ds_mla_layout = self.kv_cache_dtype == \"fp8_ds_mla\"\n"
        "        return MLAAttentionSpec(\n"
        "            block_size=vllm_config.cache_config.block_size,\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,\n"
        "            compress_ratio=self.compress_ratio,\n"
        "            cache_dtype_str=self.kv_cache_dtype,\n"
        "            alignment=576 if uses_fp8_ds_mla_layout else 512,\n"
        "            model_version=\"deepseek_v4\",\n"
        "            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n"
        "            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token;\n"
        "            # head_size stays semantic (512).\n"
        "            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,\n"
        "        )\n",
        "        packed = _dsv4_packed_layout(self.kv_cache_dtype)\n"
        "        return MLAAttentionSpec(\n"
        "            block_size=vllm_config.cache_config.block_size,\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=torch.uint8 if packed else self.kv_cache_torch_dtype,\n"
        "            compress_ratio=self.compress_ratio,\n"
        "            cache_dtype_str=self.kv_cache_dtype,\n"
        "            alignment=_dsv4_page_alignment(self.kv_cache_dtype),\n"
        "            model_version=\"deepseek_v4\",\n"
        "            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n"
        "            # DeepseekV4: 448B NoPE + 128B RoPE + 8B scale = 584B per token;\n"
        "            # head_size stays semantic (512). Indexer cache is not this path.\n"
        "            state_content_bytes=584 if packed else None,\n"
        "        )\n",
        "DeepseekV4Attention main KV 584/576/512 ladder",
    )

    swa = vllm / "v1/attention/backends/mla/sparse_swa.py"
    replace_once(
        swa,
        "        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == \"fp8_ds_mla\"\n"
        "        return SlidingWindowMLASpec(\n"
        "            block_size=self.block_size,\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=self.dtype,\n"
        "            sliding_window=self.window_size,\n"
        "            cache_dtype_str=self.cache_config.cache_dtype,\n"
        "            # DeepseekV4 fp8_ds_mla: 584B per token (448B NoPE + 128B RoPE + 8B scales)\n"
        "            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,\n"
        "            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).\n"
        "            alignment=576 if uses_fp8_ds_mla_layout else 512,\n",
        "        packed = self.cache_config.cache_dtype in (\"fp8_ds_mla\", \"nvfp4_ds_mla\")\n"
        "        from vllm.models.deepseek_v4.attention import _dsv4_page_alignment\n"
        "        return SlidingWindowMLASpec(\n"
        "            block_size=self.block_size,\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=self.dtype,\n"
        "            sliding_window=self.window_size,\n"
        "            cache_dtype_str=self.cache_config.cache_dtype,\n"
        "            state_content_bytes=584 if packed else None,\n"
        "            alignment=_dsv4_page_alignment(self.cache_config.cache_dtype),\n",
        "SWA alignment 584/576/512 ladder",
    )
    replace_once(
        swa,
        '        if cache_dtype_str == "fp8_ds_mla":\n'
        "            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).\n"
        "            # head_size passed in is the semantic head_dim (512).\n"
        "            return (num_blocks, block_size, 584)\n",
        '        if cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):\n'
        "            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 scale).\n"
        "            # head_size passed in is the semantic head_dim (512).\n"
        "            return (num_blocks, block_size, 584)\n",
        "SWA get_kv_cache_shape 584",
    )

    sparse = vllm / "models/deepseek_v4/sparse_mla.py"
    replace_once(
        sparse,
        '        "fp8_ds_mla",\n        "fp8",  # alias for fp8_ds_mla\n',
        '        "fp8_ds_mla",\n'
        '        "nvfp4_ds_mla",\n'
        '        "fp8",  # alias for fp8_ds_mla\n',
        "sparse_mla supported nvfp4_ds_mla",
    )
    replace_once(
        sparse,
        '        if cache_dtype_str == "fp8_ds_mla":\n'
        "            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).\n"
        "            # head_size passed in is the semantic head_dim (512).\n"
        "            return (num_blocks, block_size, 584)\n",
        '        if cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):\n'
        "            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 scale).\n"
        "            # head_size passed in is the semantic head_dim (512).\n"
        "            return (num_blocks, block_size, 584)\n",
        "sparse_mla get_kv_cache_shape 584",
    )


def apply(vllm: Path) -> None:
    copy_new_modules(vllm)
    patch_moe_backend(vllm)
    patch_envs(vllm)
    patch_utils_b12x(vllm)
    patch_mxfp4_oracle(vllm)
    patch_mhc(vllm)
    patch_nvfp4_ds_mla(vllm)
    print(f"overlays applied under {vllm}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-dir",
        type=Path,
        default=None,
        help="Installed vllm package dir. Default: import vllm.",
    )
    args = parser.parse_args()
    vllm = args.vllm_dir.resolve() if args.vllm_dir else _vllm_dir()
    if not (vllm / "config/kernel.py").is_file():
        raise SystemExit(f"not a vllm package: {vllm}")
    apply(vllm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
