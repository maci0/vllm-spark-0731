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


def patch_deep_gemm_sm12x_guard(vllm: Path) -> None:
    """Exclude SM12x from is_deep_gemm_supported().

    rc2's support_deep_gemm() returns True for family 120, but the v0.27.1
    compiled DeepGEMM .so (hyperconnection, grouped_gemm, etc.) only targets
    SM100/SM103. Every caller (mHC tilelang, fp8_einsum, e8m0) that trusts
    is_deep_gemm_supported() would crash on SM12x without this guard.
    """
    path = vllm / "utils/deep_gemm.py"
    replace_once(
        path,
        "def is_deep_gemm_supported() -> bool:\n"
        '    """Return `True` if DeepGEMM is supported on the current platform.\n'
        "    Currently, only Hopper and Blackwell GPUs are supported.\n"
        '    """\n'
        "    is_supported_arch = current_platform.support_deep_gemm()\n"
        "    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch\n",
        "def is_deep_gemm_supported() -> bool:\n"
        '    """Return `True` if DeepGEMM is supported on the current platform.\n'
        "    Currently, only Hopper and Blackwell GPUs are supported.\n"
        '    """\n'
        "    is_supported_arch = current_platform.support_deep_gemm()\n"
        "    if is_supported_arch and current_platform.is_device_capability_family(120):\n"
        "        return False\n"
        "    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch\n",
        "is_deep_gemm_supported SM12x exclusion",
    )


def patch_fp8_einsum_fallback(vllm: Path) -> None:
    path = vllm / "utils/deep_gemm.py"
    replace_once(
        path,
        "def fp8_einsum(*args, **kwargs):\n"
        "    _lazy_init()\n"
        "    if _fp8_einsum_impl is None:\n"
        "        return _missing(*args, **kwargs)\n"
        "    return _fp8_einsum_impl(*args, **kwargs)\n",
        "def fp8_einsum(subscripts, a_and_scale, b_and_scale, out, recipe=(1, 128, 128)):\n"
        "    if current_platform.is_device_capability_family(120):\n"
        "        a_fp8, a_scale = a_and_scale\n"
        "        w_fp8, w_scale = b_and_scale\n"
        "        a_scale_f32 = a_scale.to(torch.float32)\n"
        "        if a_scale_f32.shape[-1] != a_fp8.shape[-1]:\n"
        "            a_scale_f32 = a_scale_f32.repeat_interleave(a_fp8.shape[-1] // a_scale_f32.shape[-1], dim=-1)\n"
        "        if a_scale_f32.dim() >= 2 and a_fp8.dim() >= 2 and a_scale_f32.shape[-2] != a_fp8.shape[-2]:\n"
        "            a_scale_f32 = a_scale_f32.repeat_interleave(a_fp8.shape[-2] // a_scale_f32.shape[-2], dim=-2)\n"
        "        a_dq = a_fp8.to(out.dtype) * a_scale_f32.to(out.dtype)\n"
        "        w_scale_f32 = w_scale.to(torch.float32)\n"
        "        if w_scale_f32.shape[-1] != w_fp8.shape[-1]:\n"
        "            w_scale_f32 = w_scale_f32.repeat_interleave(w_fp8.shape[-1] // w_scale_f32.shape[-1], dim=-1)\n"
        "        if w_scale_f32.dim() >= 2 and w_fp8.dim() >= 2 and w_scale_f32.shape[-2] != w_fp8.shape[-2]:\n"
        "            w_scale_f32 = w_scale_f32.repeat_interleave(w_fp8.shape[-2] // w_scale_f32.shape[-2], dim=-2)\n"
        "        w_dq = w_fp8.to(out.dtype) * w_scale_f32.to(out.dtype)\n"
        "        b, h, r = a_dq.shape\n"
        "        if w_dq.dim() == 2 and out.dim() == 3 and w_dq.shape[0] == h * out.shape[2]:\n"
        "            w_dq = w_dq.view(h, out.shape[2], w_dq.shape[1])\n"
        "        elif w_dq.dim() == 2 and w_dq.shape[1] == h * r:\n"
        "            w_dq = w_dq.view(w_dq.shape[0], h, r)\n"
        "        res = torch.einsum(subscripts, a_dq, w_dq)\n"
        "        out.copy_(res)\n"
        "        return out\n"
        "    _lazy_init()\n"
        "    if _fp8_einsum_impl is None:\n"
        "        return _missing(subscripts, a_and_scale, b_and_scale, out, recipe)\n"
        "    return _fp8_einsum_impl(subscripts, a_and_scale, b_and_scale, out, recipe)\n",
        "fp8_einsum SM12x dequant fallback",
    )


def patch_mxfp4_process_weights(vllm: Path) -> None:
    path = vllm / "model_executor/layers/fused_moe/oracle/mxfp4.py"
    replace_once(
        path,
        "    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,\n"
        ") -> mk.FusedMoEKernel:\n",
        "    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,\n"
        "    layer: \"torch.nn.Module | None\" = None,\n"
        ") -> mk.FusedMoEKernel:\n",
        "make_mxfp4_moe_kernel +layer param",
    )
    replace_once(
        path,
        "    else:\n"
        "        experts = experts_cls(\n"
        "            moe_config=moe_config,\n"
        "            quant_config=moe_quant_config,\n"
        "        )\n"
        "\n"
        "    kernel = mk.FusedMoEKernel(\n",
        "    else:\n"
        "        experts = experts_cls(\n"
        "            moe_config=moe_config,\n"
        "            quant_config=moe_quant_config,\n"
        "        )\n"
        "\n"
        "    if layer is not None and hasattr(experts, \"process_weights_after_loading\"):\n"
        "        experts.process_weights_after_loading(layer)\n"
        "\n"
        "    kernel = mk.FusedMoEKernel(\n",
        "make_mxfp4_moe_kernel process_weights_after_loading",
    )
    caller = vllm / "model_executor/layers/quantization/mxfp4.py"
    replace_once(
        caller,
        "                routing_tables=layer._expert_routing_tables(),\n"
        "            )\n"
        "\n"
        "    def process_weights_after_loading(self, layer):\n",
        "                routing_tables=layer._expert_routing_tables(),\n"
        "                layer=layer,\n"
        "            )\n"
        "\n"
        "    def process_weights_after_loading(self, layer):\n",
        "Mxfp4MoEMethod pass layer to kernel factory",
    )


def patch_cutlass_sm12x_guard(vllm: Path) -> None:
    """Exclude SM12x from CutlassFp8BlockScaledMMKernel.is_supported().

    The v0.27.1 compiled CUTLASS .so doesn't target SM12x. With DeepGEMM
    already excluded, auto-selection falls through to CUTLASS which crashes
    with "dispatch_scaled_mm". This guard lets B12xFp8BlockScaledMMKernel
    (next in priority) get selected instead.
    """
    path = vllm / "model_executor/kernels/linear/scaled_mm/cutlass.py"
    replace_once(
        path,
        "    @classmethod\n"
        "    def is_supported(cls, compute_capability=None):\n"
        "        if not CUTLASS_BLOCK_FP8_SUPPORTED:\n"
        "            return (\n"
        "                False,\n"
        '                "The device compute capability of"\n'
        '                f"{compute_capability} is not supported.",\n'
        "            )\n"
        "        return True, None\n",
        "    @classmethod\n"
        "    def is_supported(cls, compute_capability=None):\n"
        "        if not CUTLASS_BLOCK_FP8_SUPPORTED:\n"
        "            return (\n"
        "                False,\n"
        '                "The device compute capability of"\n'
        '                f"{compute_capability} is not supported.",\n'
        "            )\n"
        "        from vllm.platforms import current_platform\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            return False, \"CUTLASS FP8 not supported on SM12x\"\n"
        "        return True, None\n",
        "CUTLASS FP8 SM12x exclusion",
    )


def patch_indexer_deepgemm_guard(vllm: Path) -> None:
    """Guard indexer paged MQA logits metadata against SM12x.

    The indexer calls get_paged_mqa_logits_metadata (DeepGEMM C++) gated
    only on has_deep_gemm() (importable), not is_deep_gemm_supported()
    (architecture). On SM12x, the compiled .so crashes. The schedule_metadata
    is not consumed by FlashInfer sparse MLA, so skipping it is safe.
    """
    path = vllm / "v1/attention/backends/mla/indexer.py"
    replace_once(
        path,
        "            if current_platform.is_cuda() and has_deep_gemm():\n"
        "                metadata = get_paged_mqa_logits_metadata(\n",
        "            from vllm.utils.deep_gemm import is_deep_gemm_supported\n"
        "            if current_platform.is_cuda() and is_deep_gemm_supported():\n"
        "                metadata = get_paged_mqa_logits_metadata(\n",
        "indexer paged MQA logits SM12x guard",
    )


def patch_mqa_logits_sm12x_fallback(vllm: Path) -> None:
    """Add SM12x dequant fallbacks for fp8_fp4_mqa_logits (prefill) and
    fp8_fp4_paged_mqa_logits (decode).

    DeepGEMM's C++ kernel asserts arch_major in {9, 10}. On SM12x the
    indexer's forward path calls both for scoring. We provide pure-PyTorch
    dequant + matmul fallbacks. Uses weighted-Q to avoid materializing
    a [M, H, N] intermediate.
    """
    path = vllm / "utils/deep_gemm.py"
    replace_once(
        path,
        "def fp8_fp4_mqa_logits(\n"
        "    q: tuple[torch.Tensor, torch.Tensor | None],\n"
        "    kv: tuple[torch.Tensor, torch.Tensor],\n"
        "    weights: torch.Tensor,\n"
        "    cu_seqlen_ks: torch.Tensor,\n"
        "    cu_seqlen_ke: torch.Tensor,\n"
        "    clean_logits: bool,\n"
        ") -> torch.Tensor:\n",
        "def _sm12x_dequant_q_weighted(\n"
        "    q_values: torch.Tensor,\n"
        "    q_scale: torch.Tensor | None,\n"
        "    weights: torch.Tensor,\n"
        ") -> torch.Tensor:\n"
        "    q_f = q_values.to(torch.float32)\n"
        "    if q_scale is not None:\n"
        "        q_sf = q_scale.to(torch.float32)\n"
        "        if q_sf.shape != q_f.shape:\n"
        "            q_sf = q_sf.expand_as(q_f)\n"
        "        q_f = q_f * q_sf\n"
        "    return torch.einsum('...hd,...h->...d', q_f, weights.float())\n\n\n"
        "def _sm12x_dequant_k(\n"
        "    k_packed: torch.Tensor,\n"
        "    k_scales: torch.Tensor,\n"
        ") -> torch.Tensor:\n"
        "    k_f = k_packed.to(torch.float32)\n"
        "    k_sf = k_scales.to(torch.float32)\n"
        "    if k_sf.dim() == 1:\n"
        "        return k_f * k_sf.unsqueeze(-1)\n"
        "    return k_f * k_sf\n\n\n"
        "def _sm12x_fp8_mqa_logits(\n"
        "    q: tuple[torch.Tensor, torch.Tensor | None],\n"
        "    kv: tuple[torch.Tensor, torch.Tensor],\n"
        "    weights: torch.Tensor,\n"
        "    cu_seqlen_ks: torch.Tensor,\n"
        "    cu_seqlen_ke: torch.Tensor,\n"
        "    clean_logits: bool,\n"
        ") -> torch.Tensor:\n"
        "    q_values, q_scale = q\n"
        "    k_packed, k_scales = kv\n"
        "    q_w = _sm12x_dequant_q_weighted(q_values, q_scale, weights)\n"
        "    k_dq = _sm12x_dequant_k(k_packed, k_scales)\n"
        "    logits = q_w @ k_dq.T\n"
        "    if clean_logits:\n"
        "        N = logits.shape[1]\n"
        "        pos = torch.arange(N, device=logits.device).unsqueeze(0)\n"
        "        mask = (pos >= cu_seqlen_ks.unsqueeze(1)) & (pos < cu_seqlen_ke.unsqueeze(1))\n"
        "        logits = logits.masked_fill(~mask, float('-inf'))\n"
        "    return logits\n\n\n"
        "def _sm12x_fp8_paged_mqa_logits(\n"
        "    q: tuple[torch.Tensor, torch.Tensor | None],\n"
        "    kv_cache: torch.Tensor,\n"
        "    weights: torch.Tensor,\n"
        "    context_lens: torch.Tensor,\n"
        "    block_tables: torch.Tensor,\n"
        "    schedule_metadata: torch.Tensor,\n"
        "    max_model_len: int,\n"
        "    clean_logits: bool,\n"
        "    indices: torch.Tensor | None = None,\n"
        ") -> torch.Tensor:\n"
        "    q_values, q_scale = q\n"
        "    B = block_tables.shape[0]\n"
        "    block_size = kv_cache.shape[1]\n"
        "    head_size = kv_cache.shape[3]\n"
        "    if context_lens.dim() == 2:\n"
        "        ctx = context_lens[:, -1]\n"
        "    else:\n"
        "        ctx = context_lens\n"
        "    max_ctx = int(ctx.max().item()) if ctx.numel() > 0 else 0\n"
        "    if q_values.dim() == 4:\n"
        "        Bq, next_n, H, D = q_values.shape\n"
        "        q_flat = q_values.reshape(Bq * next_n, H, D)\n"
        "    else:\n"
        "        q_flat = q_values\n"
        "        H, D = q_flat.shape[-2], q_flat.shape[-1]\n"
        "        next_n = q_flat.shape[0] // B if B > 0 else 1\n"
        "    M = q_flat.shape[0]\n"
        "    q_w = _sm12x_dequant_q_weighted(\n"
        "        q_flat, q_scale.reshape(M, H, -1) if q_scale is not None else None,\n"
        "        weights[:M],\n"
        "    )\n"
        "    if max_ctx == 0:\n"
        "        return q_w.new_zeros((M, max_model_len), dtype=torch.float32)\n"
        "    pos = torch.arange(max_ctx, device=kv_cache.device)\n"
        "    lb = pos // block_size\n"
        "    off = pos % block_size\n"
        "    max_blocks = block_tables.shape[1]\n"
        "    lb_c = lb.clamp(max=max_blocks - 1)\n"
        "    phys = block_tables[:, lb_c]\n"
        "    k_raw = kv_cache[phys.reshape(-1), off.repeat(B), 0]\n"
        "    k_raw = k_raw.reshape(B, max_ctx, head_size)\n"
        "    k_fp8 = k_raw[:, :, :D].contiguous().view(torch.float8_e4m3fn)\n"
        "    k_sf = k_raw[:, :, D:D+4].contiguous().view(torch.float32).squeeze(-1)\n"
        "    k_dq = k_fp8.to(torch.float32) * k_sf.unsqueeze(-1)\n"
        "    q_w_4d = q_w.reshape(B, next_n, D)\n"
        "    logits_short = torch.bmm(q_w_4d, k_dq.transpose(1, 2))\n"
        "    logits_short = logits_short.reshape(M, max_ctx)\n"
        "    if max_ctx >= max_model_len:\n"
        "        logits = logits_short[:, :max_model_len]\n"
        "    else:\n"
        "        logits = q_w.new_full((M, max_model_len), float('-inf') if clean_logits else 0.0, dtype=torch.float32)\n"
        "        logits[:, :max_ctx] = logits_short\n"
        "    ctx_exp = ctx.unsqueeze(1).expand(B, next_n).reshape(M, 1)\n"
        "    pos_row = torch.arange(min(max_ctx, max_model_len), device=logits.device).unsqueeze(0)\n"
        "    invalid = pos_row >= ctx_exp\n"
        "    logits[:, :pos_row.shape[1]].masked_fill_(invalid, float('-inf') if clean_logits else 0.0)\n"
        "    return logits\n\n\n"
        "def fp8_fp4_mqa_logits(\n"
        "    q: tuple[torch.Tensor, torch.Tensor | None],\n"
        "    kv: tuple[torch.Tensor, torch.Tensor],\n"
        "    weights: torch.Tensor,\n"
        "    cu_seqlen_ks: torch.Tensor,\n"
        "    cu_seqlen_ke: torch.Tensor,\n"
        "    clean_logits: bool,\n"
        ") -> torch.Tensor:\n",
        "fp8_fp4_mqa_logits SM12x dequant fallback decl",
    )
    replace_once(
        path,
        "    _lazy_init()\n"
        "    if _fp8_fp4_mqa_logits_impl is None:\n"
        "        return _missing()\n"
        "    return _fp8_fp4_mqa_logits_impl(\n"
        "        q,\n"
        "        kv,\n"
        "        weights,\n"
        "        cu_seqlen_ks,\n"
        "        cu_seqlen_ke,\n"
        "        clean_logits=clean_logits,\n"
        "    )\n",
        "    if current_platform.is_device_capability_family(120):\n"
        "        return _sm12x_fp8_mqa_logits(\n"
        "            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits\n"
        "        )\n"
        "    _lazy_init()\n"
        "    if _fp8_fp4_mqa_logits_impl is None:\n"
        "        return _missing()\n"
        "    return _fp8_fp4_mqa_logits_impl(\n"
        "        q,\n"
        "        kv,\n"
        "        weights,\n"
        "        cu_seqlen_ks,\n"
        "        cu_seqlen_ke,\n"
        "        clean_logits=clean_logits,\n"
        "    )\n",
        "fp8_fp4_mqa_logits SM12x dispatch guard",
    )
    replace_once(
        path,
        "    _lazy_init()\n"
        "    if _fp8_fp4_paged_mqa_logits_impl is None:\n"
        "        return _missing()\n"
        "    kwargs = {} if indices is None else {\"indices\": indices}\n"
        "    return _fp8_fp4_paged_mqa_logits_impl(\n"
        "        q,\n"
        "        kv_cache,\n"
        "        weights,\n"
        "        context_lens,\n"
        "        block_tables,\n"
        "        schedule_metadata,\n"
        "        max_model_len,\n"
        "        clean_logits=clean_logits,\n"
        "        **kwargs,\n"
        "    )\n",
        "    if current_platform.is_device_capability_family(120):\n"
        "        return _sm12x_fp8_paged_mqa_logits(\n"
        "            q, kv_cache, weights, context_lens, block_tables,\n"
        "            schedule_metadata, max_model_len, clean_logits,\n"
        "            indices=indices,\n"
        "        )\n"
        "    _lazy_init()\n"
        "    if _fp8_fp4_paged_mqa_logits_impl is None:\n"
        "        return _missing()\n"
        "    kwargs = {} if indices is None else {\"indices\": indices}\n"
        "    return _fp8_fp4_paged_mqa_logits_impl(\n"
        "        q,\n"
        "        kv_cache,\n"
        "        weights,\n"
        "        context_lens,\n"
        "        block_tables,\n"
        "        schedule_metadata,\n"
        "        max_model_len,\n"
        "        clean_logits=clean_logits,\n"
        "        **kwargs,\n"
        "    )\n",
        "fp8_fp4_paged_mqa_logits SM12x dispatch guard",
    )


def patch_flashinfer_dsv4_dispatch(site: Path) -> None:
    """Add (H, 192) entries to FlashInfer's _DECODE_DSV4_DISPATCH for DSpark k=5.

    DSpark k=5 with window_size=128 requires top_k=ceil(133/64)*64=192.
    All head counts get 192 to support any TP configuration.
    """
    path = site / "flashinfer/mla/_sparse_mla_sm120.py"
    if not path.is_file():
        print(f"skip flashinfer dsv4 dispatch: {path} not found")
        return
    replace_once(
        path,
        "_DECODE_DSV4_DISPATCH = frozenset(\n"
        "    {\n"
        "        (8, 128),\n"
        "        (8, 512),\n"
        "        (8, 1024),\n"
        "        (16, 128),\n"
        "        (16, 512),\n"
        "        (16, 1024),\n"
        "        (32, 128),\n"
        "        (32, 512),\n"
        "        (32, 1024),\n"
        "        (64, 128),\n"
        "        (64, 512),\n"
        "        (64, 1024),\n"
        "        (128, 128),\n"
        "        (128, 512),\n"
        "        (128, 1024),\n"
        "    }\n"
        ")\n",
        "_DECODE_DSV4_DISPATCH = frozenset(\n"
        "    {\n"
        "        (8, 128),\n"
        "        (8, 192),\n"
        "        (8, 512),\n"
        "        (8, 1024),\n"
        "        (16, 128),\n"
        "        (16, 192),\n"
        "        (16, 512),\n"
        "        (16, 1024),\n"
        "        (32, 128),\n"
        "        (32, 192),\n"
        "        (32, 512),\n"
        "        (32, 1024),\n"
        "        (64, 128),\n"
        "        (64, 192),\n"
        "        (64, 512),\n"
        "        (64, 1024),\n"
        "        (128, 128),\n"
        "        (128, 192),\n"
        "        (128, 512),\n"
        "        (128, 1024),\n"
        "    }\n"
        ")\n",
        "flashinfer _DECODE_DSV4_DISPATCH +TOPK=192 for DSpark k=5",
    )


def patch_flashinfer_dsv4_cu_dispatch(site: Path) -> None:
    """Add TOPK=192 dispatch entries to the FlashInfer DSV4 decode C++ kernel.

    The JIT-compiled C++ source only dispatches TOPK in {128, 512, 1024}.
    DSpark k=5 needs top_k=192. The template is generic over TOPK, so adding
    new dispatch entries lets the JIT compiler instantiate the kernel for 192.
    Also removes the pre-compiled .so so FlashInfer recompiles from source.
    """
    path = site / "flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu"
    if not path.is_file():
        print(f"skip flashinfer dsv4 cu dispatch: {path} not found")
        return
    replace_once(
        path,
        "  DSV4_DISPATCH(8, 128)\n"
        "  DSV4_DISPATCH(8, 512)\n"
        "  DSV4_DISPATCH(8, 1024)\n"
        "  DSV4_DISPATCH(16, 128)\n"
        "  DSV4_DISPATCH(16, 512)\n"
        "  DSV4_DISPATCH(16, 1024)\n"
        "  DSV4_DISPATCH(32, 128)\n"
        "  DSV4_DISPATCH(32, 512)\n"
        "  DSV4_DISPATCH(32, 1024)\n"
        "  DSV4_DISPATCH(64, 128)\n"
        "  DSV4_DISPATCH(64, 512)\n"
        "  DSV4_DISPATCH(64, 1024)\n"
        "  DSV4_DISPATCH(128, 128)\n"
        "  DSV4_DISPATCH(128, 512)\n"
        "  DSV4_DISPATCH(128, 1024)\n",
        "  DSV4_DISPATCH(8, 128)\n"
        "  DSV4_DISPATCH(8, 192)\n"
        "  DSV4_DISPATCH(8, 512)\n"
        "  DSV4_DISPATCH(8, 1024)\n"
        "  DSV4_DISPATCH(16, 128)\n"
        "  DSV4_DISPATCH(16, 192)\n"
        "  DSV4_DISPATCH(16, 512)\n"
        "  DSV4_DISPATCH(16, 1024)\n"
        "  DSV4_DISPATCH(32, 128)\n"
        "  DSV4_DISPATCH(32, 192)\n"
        "  DSV4_DISPATCH(32, 512)\n"
        "  DSV4_DISPATCH(32, 1024)\n"
        "  DSV4_DISPATCH(64, 128)\n"
        "  DSV4_DISPATCH(64, 192)\n"
        "  DSV4_DISPATCH(64, 512)\n"
        "  DSV4_DISPATCH(64, 1024)\n"
        "  DSV4_DISPATCH(128, 128)\n"
        "  DSV4_DISPATCH(128, 192)\n"
        "  DSV4_DISPATCH(128, 512)\n"
        "  DSV4_DISPATCH(128, 1024)\n",
        "flashinfer DSV4 decode C++ dispatch +TOPK=192",
    )
    cached_so = site / "flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so"
    if cached_so.is_file():
        cached_so.unlink()
        print("ok removed pre-compiled sparse_mla_sm120.so (forces JIT recompile)")


def apply(vllm: Path) -> None:
    copy_new_modules(vllm)
    patch_moe_backend(vllm)
    patch_envs(vllm)
    patch_utils_b12x(vllm)
    patch_mxfp4_oracle(vllm)
    patch_mhc(vllm)
    patch_nvfp4_ds_mla(vllm)
    patch_deep_gemm_sm12x_guard(vllm)
    patch_cutlass_sm12x_guard(vllm)
    patch_indexer_deepgemm_guard(vllm)
    patch_fp8_einsum_fallback(vllm)
    patch_mqa_logits_sm12x_fallback(vllm)
    patch_mxfp4_process_weights(vllm)
    patch_flashinfer_dsv4_dispatch(vllm.parent)
    patch_flashinfer_dsv4_cu_dispatch(vllm.parent)
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
