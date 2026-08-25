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
import textwrap
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


def replace_one_of(path: Path, variants: list[tuple[str, str, str]]) -> None:
    """Try (old, new, label) variants until one needle matches.

    Keeps rc2 overlay needles working while main trees drift.
    """
    text = path.read_text()
    for old, new, label in variants:
        if old in text:
            replace_once(path, old, new, label)
            return
        if new in text:
            print(f"skip {label}: already applied")
            return
    labels = ", ".join(v[2] for v in variants)
    raise SystemExit(f"{path}: none of the needles matched ({labels})")


def replace_optional(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if old not in text:
        if new in text:
            print(f"skip {label}: already applied")
            return
        print(f"skip {label}: needle not in {path.name}")
        return
    replace_once(path, old, new, label)


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


def copy_dsv4_b12x_sparse(vllm: Path) -> None:
    src = FILES / "dsv4_b12x_sparse.py"
    dest = vllm / "models/deepseek_v4/nvidia/b12x_sparse.py"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"ok copied {dest.relative_to(vllm.parent)}")


def copy_sm12x_b12x_kernels(vllm: Path) -> None:
    src = FILES / "sm12x_b12x_kernels.py"
    dest = vllm / "utils/sm12x_b12x_kernels.py"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"ok copied {dest.relative_to(vllm.parent)}")


def patch_dsv4_b12x_sparse_backend(vllm: Path) -> None:
    """Register B12X_MLA_SPARSE and select the DSV4 b12x attention class."""
    copy_dsv4_b12x_sparse(vllm)
    registry = vllm / "v1/attention/backends/registry.py"
    model = vllm / "models/deepseek_v4/nvidia/model.py"
    enum_block = (
        "    B12X_MLA_SPARSE = (\n"
        '        "vllm.models.deepseek_v4.nvidia.b12x_sparse."\n'
        '        "DeepseekV4B12xMLASparseBackend"\n'
        "    )\n"
    )
    select_block = (
        "    if backend == AttentionBackendEnum.B12X_MLA_SPARSE:\n"
        "        return DeepseekV4B12xSM120Attention\n"
    )
    import_block = (
        "from vllm.models.deepseek_v4.nvidia.b12x_sparse import (\n"
        "    DeepseekV4B12xSM120Attention,\n"
        ")\n"
    )
    # Idempotent: copy always refreshes the module. Collapse duplicate inserts.
    reg_text = registry.read_text()
    while reg_text.count(enum_block) > 1:
        reg_text = reg_text.replace(enum_block, "", 1)
        registry.write_text(reg_text)
        print("ok removed duplicate B12X_MLA_SPARSE enum")
        reg_text = registry.read_text()
    model_text = model.read_text()
    while model_text.count(select_block) > 1:
        model_text = model_text.replace(select_block, "", 1)
        print("ok removed duplicate B12X_MLA_SPARSE selector")
        model.write_text(model_text)
        model_text = model.read_text()
    while model_text.count(import_block) > 1:
        model_text = model_text.replace(import_block, "", 1)
        print("ok removed duplicate B12x SM120 import")
        model.write_text(model_text)
        model_text = model.read_text()
    if (
        enum_block in registry.read_text()
        and select_block in model.read_text()
        and import_block in model.read_text()
    ):
        print("skip B12X_MLA_SPARSE registry/selector: already applied")
        return
    replace_once(
        registry,
        "    FLASHINFER_MLA_SPARSE_DSV4 = (\n"
        '        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse."\n'
        '        "DeepseekV4FlashInferMLASparseBackend"\n'
        "    )\n"
        "    ROCM_FLASHMLA_SPARSE_DSV4 = (\n",
        "    FLASHINFER_MLA_SPARSE_DSV4 = (\n"
        '        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse."\n'
        '        "DeepseekV4FlashInferMLASparseBackend"\n'
        "    )\n"
        + enum_block
        + "    ROCM_FLASHMLA_SPARSE_DSV4 = (\n",
        "AttentionBackendEnum B12X_MLA_SPARSE",
    )
    model = vllm / "models/deepseek_v4/nvidia/model.py"
    replace_once(
        model,
        "from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (\n"
        "    DeepseekV4FlashInferMLAAttention,\n"
        "    DeepseekV4FlashInferSM120Attention,\n"
        ")\n",
        "from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (\n"
        "    DeepseekV4FlashInferMLAAttention,\n"
        "    DeepseekV4FlashInferSM120Attention,\n"
        ")\n"
        "from vllm.models.deepseek_v4.nvidia.b12x_sparse import (\n"
        "    DeepseekV4B12xSM120Attention,\n"
        ")\n",
        "import DeepseekV4B12xSM120Attention",
    )
    replace_once(
        model,
        "    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:\n"
        "        if device_capability is not None and device_capability.major == 12:\n"
        "            return DeepseekV4FlashInferSM120Attention\n"
        "        return DeepseekV4FlashInferMLAAttention\n",
        "    if backend == AttentionBackendEnum.B12X_MLA_SPARSE:\n"
        "        return DeepseekV4B12xSM120Attention\n"
        "    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:\n"
        "        if device_capability is not None and device_capability.major == 12:\n"
        "            return DeepseekV4FlashInferSM120Attention\n"
        "        return DeepseekV4FlashInferMLAAttention\n",
        "select B12X_MLA_SPARSE attention class",
    )


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
    text = path.read_text()
    if "_tilelang_hc_prenorm_gemm" in text or "use_deep_gemm" in text:
        print("skip mhc patch (already present)")
        return
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
    attn_spec_new_common = (
        "        packed = _dsv4_packed_layout(self.kv_cache_dtype)\n"
        "        return MLAAttentionSpec(\n"
        "            block_size=vllm_config.cache_config.block_size,\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=torch.uint8 if packed else self.kv_cache_torch_dtype,\n"
    )
    attn_spec_tail = (
        "            cache_dtype_str=self.kv_cache_dtype,\n"
        "            alignment=_dsv4_page_alignment(self.kv_cache_dtype),\n"
        "            model_version=\"deepseek_v4\",\n"
        "            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n"
        "            # DeepseekV4: 448B NoPE + 128B RoPE + 8B scale = 584B per token;\n"
        "            # head_size stays semantic (512). Indexer cache is not this path.\n"
        "            state_content_bytes=584 if packed else None,\n"
        "        )\n"
    )
    replace_one_of(
        attn,
        [
            (
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
                attn_spec_new_common
                + "            compress_ratio=self.compress_ratio,\n"
                + attn_spec_tail,
                "DeepseekV4Attention main KV 584/576/512 ladder",
            ),
            (
                "        uses_fp8_ds_mla_layout = self.kv_cache_dtype == \"fp8_ds_mla\"\n"
                "        return MLAAttentionSpec(\n"
                "            block_size=vllm_config.cache_config.block_size,\n"
                "            num_kv_heads=1,\n"
                "            head_size=self.head_dim,\n"
                "            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,\n"
                "            tokens_per_state=self.compress_ratio,\n"
                "            cache_dtype_str=self.kv_cache_dtype,\n"
                "            alignment=576 if uses_fp8_ds_mla_layout else 512,\n"
                "            model_version=\"deepseek_v4\",\n"
                "            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n"
                "            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token;\n"
                "            # head_size stays semantic (512).\n"
                "            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,\n"
                "        )\n",
                attn_spec_new_common
                + "            tokens_per_state=self.compress_ratio,\n"
                + attn_spec_tail,
                "DeepseekV4Attention main KV 584/576/512 ladder (tokens_per_state)",
            ),
        ],
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
    replace_optional(
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
    replace_optional(
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


def patch_dsv4_nvfp4_attn(vllm: Path) -> None:
    """Let FLASHINFER_MLA_SPARSE_DSV4 accept nvfp4_ds_mla on SM12x.

    rc2 supports_combination only lists fp8 / fp8_e4m3 / fp8_ds_mla, so
    ``05-serve.sh nvfp4`` never selects the DSV4 backend. SM120 DSV4
    kernels already consume the packed uint8 584-byte page (same layout
    as fp8_ds_mla). This image does not vendor a separate NVFP4 CUDA
    writer; nvfp4_ds_mla is the envelope name.
    """
    path = vllm / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    replace_once(
        path,
        '        "fp8_e4m3",\n        "fp8_ds_mla",\n    ]\n',
        '        "fp8_e4m3",\n        "fp8_ds_mla",\n        "nvfp4_ds_mla",\n    ]\n',
        "DSV4 supported_kv_cache_dtypes +nvfp4_ds_mla",
    )
    replace_once(
        path,
        '            if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla"):\n'
        '                return "kv_cache_dtype not supported"\n',
        '            if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla", "nvfp4_ds_mla"):\n'
        '                return "kv_cache_dtype not supported"\n',
        "DSV4 SM12x supports_combination +nvfp4_ds_mla",
    )


def patch_dsv4_sm12x_block_size(vllm: Path) -> None:
    """SM120 FlashInfer DSV4 decode is compiled for 64-token footer pages.

    Keep `--block-size 256` so C128 storage_block_size = 256/128 = 2.
    SWA pages are already hardcoded to 64. Returning [64] here makes
    select_common_block_size split each 256-token manager block into
    four kernel pages. Listing [256] skips the specialized decode kernel
    (page 256 is not dispatchable; isolated page-64 cosine is 0.99966).
    """
    path = vllm / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    sparse = vllm / "models/deepseek_v4/sparse_mla.py"
    if "dsv4_supported_kernel_block_sizes" in sparse.read_text() or "dsv4_supported_kernel_block_sizes" in path.read_text():
        print("skip dsv4 sm12x block size (already present)")
        return
    replace_once(
        path,
        "    @staticmethod\n"
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [256]\n",
        "    @staticmethod\n"
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        from vllm.platforms import current_platform\n\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            return [64]\n"
        "        return [256]\n",
        "DSV4 SM12x kernel block size 64",
    )
    # Same KV group as FLASHINFER_MLA_SPARSE_DSV4. If this stays [256],
    # select_common_block_size cannot split manager 256 down to kernel 64.
    indexer = vllm / "v1/attention/backends/mla/indexer.py"
    indexer_block64 = (
        "    @staticmethod\n"
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        from vllm.platforms import current_platform\n"
        "\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            return [64]\n"
        "        return [256]\n"
    )
    replace_one_of(
        indexer,
        [
            (
                "class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):\n"
                "    @staticmethod\n"
                "    def get_name() -> str:\n"
                '        return "DEEPSEEK_V4_INDEXER"\n'
                "\n"
                "    @staticmethod\n"
                "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
                "        return [256]\n",
                "class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):\n"
                "    @staticmethod\n"
                "    def get_name() -> str:\n"
                '        return "DEEPSEEK_V4_INDEXER"\n'
                "\n" + indexer_block64,
                "DSV4 indexer SM12x kernel block size 64",
            ),
            (
                "    @classmethod\n"
                "    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:\n"
                "        # DeepSeek-V4 packs the indexer pages beside the MLA latent pages inside\n"
                "        # each block, so the layer dim must sit inside the block dim.\n"
                "        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)\n"
                "\n"
                "    @staticmethod\n"
                "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
                "        return [256]\n",
                "    @classmethod\n"
                "    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:\n"
                "        # DeepSeek-V4 packs the indexer pages beside the MLA latent pages inside\n"
                "        # each block, so the layer dim must sit inside the block dim.\n"
                "        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)\n"
                "\n" + indexer_block64,
                "DSV4 indexer SM12x kernel block size 64 (layouts)",
            ),
        ],
    )
    sparse = vllm / "models/deepseek_v4/sparse_mla.py"
    replace_once(
        sparse,
        "    @staticmethod\n"
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [256]\n",
        "    @staticmethod\n"
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        from vllm.platforms import current_platform\n"
        "\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            return [64]\n"
        "        return [256]\n",
        "sparse MLA SM12x kernel block size 64",
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
        "def _sm12x_fp8_scale_fp32(scale: torch.Tensor) -> torch.Tensor:\n"
        "    e8 = getattr(torch, \"float8_e8m0fnu\", None)\n"
        "    if scale.dtype == torch.uint8 or (e8 is not None and scale.dtype == e8):\n"
        "        b = scale.view(torch.uint8).to(torch.int32)\n"
        "        s = torch.ldexp(\n"
        "            torch.ones((), dtype=torch.float32, device=scale.device), b - 127\n"
        "        )\n"
        "        return torch.where(b == 0, torch.zeros_like(s), s)\n"
        "    return scale.to(torch.float32)\n"
        "\n"
        "\n"
        "def fp8_einsum(subscripts, a_and_scale, b_and_scale, out, recipe=(1, 128, 128)):\n"
        "    if current_platform.is_device_capability_family(120):\n"
        "        a_fp8, a_scale = a_and_scale\n"
        "        w_fp8, w_scale = b_and_scale\n"
        "        a_scale_f32 = _sm12x_fp8_scale_fp32(a_scale)\n"
        "        if a_scale_f32.shape[-1] != a_fp8.shape[-1]:\n"
        "            a_scale_f32 = a_scale_f32.repeat_interleave(a_fp8.shape[-1] // a_scale_f32.shape[-1], dim=-1)\n"
        "        if a_scale_f32.dim() >= 2 and a_fp8.dim() >= 2 and a_scale_f32.shape[-2] != a_fp8.shape[-2]:\n"
        "            a_scale_f32 = a_scale_f32.repeat_interleave(a_fp8.shape[-2] // a_scale_f32.shape[-2], dim=-2)\n"
        "        a_dq = a_fp8.to(out.dtype) * a_scale_f32.to(out.dtype)\n"
        "        w_scale_f32 = _sm12x_fp8_scale_fp32(w_scale)\n"
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


def patch_einsum_sm12x_recipe(vllm: Path) -> None:
    """SM12x o_proj must emit SM90 FP32 128x128 scales, not SM100 packed INT32.

    compute_fp8_einsum_recipe treats major>=10 as Blackwell TMA (recipe
    (1,1,128), tma_aligned_scales=True). GB10 is family 120 / major 12, so
    fused_inv_rope_fp8_quant packed UE8M0 into int32. The Python fp8_einsum
    fallback then did scale.to(float32) on those packed ints and o_proj
    became noise. Force the SM90 FP32 layout on SM12x.
    """
    path = vllm / "models/deepseek_v4/nvidia/ops/o_proj.py"
    text = path.read_text()
    if ("cap.major == 12" in text and "tma_aligned_scales = False" in text) or "return (1, 128, 128), False" in text:
        print("skip einsum sm12x recipe (already present)")
        return
    replace_once(
        path,
        "    cap = current_platform.get_device_capability()\n"
        '    assert cap is not None, "DeepseekV4 attention requires a CUDA device"\n'
        "    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)\n"
        "    tma_aligned_scales = cap.major >= 10\n"
        "    return einsum_recipe, tma_aligned_scales\n",
        "    cap = current_platform.get_device_capability()\n"
        '    assert cap is not None, "DeepseekV4 attention requires a CUDA device"\n'
        "    # SM12x has no TMA. Packed INT32 UE8M0 is for SM100 DeepGEMM.\n"
        "    # The Python fp8_einsum fallback needs SM90 FP32 128x128 scales.\n"
        "    if cap.major == 12:\n"
        "        return (1, 128, 128), False\n"
        "    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)\n"
        "    tma_aligned_scales = cap.major >= 10\n"
        "    return einsum_recipe, tma_aligned_scales\n",
        "compute_fp8_einsum_recipe SM12x FP32 scales",
    )


def patch_einsum_sm12x_scale_upcast(vllm: Path) -> None:
    """Upgrade an already-applied SM12x fp8_einsum fallback to UE8M0 upcast.

    No-op on a fresh tree: patch_fp8_einsum_fallback already emits the helper.
    """
    path = vllm / "utils/deep_gemm.py"
    text = path.read_text()
    if "_sm12x_fp8_scale_fp32" in text:
        print("skip fp8_einsum UE8M0 upcast (already present)")
        return
    replace_once(
        path,
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
        "        w_scale_f32 = w_scale.to(torch.float32)\n",
        "def _sm12x_fp8_scale_fp32(scale: torch.Tensor) -> torch.Tensor:\n"
        "    e8 = getattr(torch, \"float8_e8m0fnu\", None)\n"
        "    if scale.dtype == torch.uint8 or (e8 is not None and scale.dtype == e8):\n"
        "        b = scale.view(torch.uint8).to(torch.int32)\n"
        "        s = torch.ldexp(\n"
        "            torch.ones((), dtype=torch.float32, device=scale.device), b - 127\n"
        "        )\n"
        "        return torch.where(b == 0, torch.zeros_like(s), s)\n"
        "    return scale.to(torch.float32)\n"
        "\n"
        "\n"
        "def fp8_einsum(subscripts, a_and_scale, b_and_scale, out, recipe=(1, 128, 128)):\n"
        "    if current_platform.is_device_capability_family(120):\n"
        "        a_fp8, a_scale = a_and_scale\n"
        "        w_fp8, w_scale = b_and_scale\n"
        "        a_scale_f32 = _sm12x_fp8_scale_fp32(a_scale)\n"
        "        if a_scale_f32.shape[-1] != a_fp8.shape[-1]:\n"
        "            a_scale_f32 = a_scale_f32.repeat_interleave(a_fp8.shape[-1] // a_scale_f32.shape[-1], dim=-1)\n"
        "        if a_scale_f32.dim() >= 2 and a_fp8.dim() >= 2 and a_scale_f32.shape[-2] != a_fp8.shape[-2]:\n"
        "            a_scale_f32 = a_scale_f32.repeat_interleave(a_fp8.shape[-2] // a_scale_f32.shape[-2], dim=-2)\n"
        "        a_dq = a_fp8.to(out.dtype) * a_scale_f32.to(out.dtype)\n"
        "        w_scale_f32 = _sm12x_fp8_scale_fp32(w_scale)\n",
        "fp8_einsum SM12x UE8M0 scale upcast",
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
    text = path.read_text()
    if "cutlass_scaled_mm_supports_fp8" in text or "CUTLASS FP8 not supported on SM12x" in text:
        print("skip cutlass SM12x guard (already present)")
        return
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
    (architecture). On main, nv_dev DeepGEMM is supported on family 120
    but still requires block_kv 32 or 64. 0731 compress-128 pages have
    2 states. SM12x FlashInfer sparse MLA does not consume the schedule
    metadata, so skipping it is safe.
    """
    path = vllm / "v1/attention/backends/mla/indexer.py"
    text = path.read_text()
    if "_should_build_paged_mqa_logits_metadata" in text:
        print("skip indexer paged MQA guard: pr-53522 already applied")
        return
    replace_one_of(
        path,
        [
            (
                "            if current_platform.is_cuda() and has_deep_gemm():\n"
                "                metadata = get_paged_mqa_logits_metadata(\n",
                "            from vllm.utils.deep_gemm import is_deep_gemm_supported\n"
                "            if (\n"
                "                current_platform.is_cuda()\n"
                "                and is_deep_gemm_supported()\n"
                "                and not current_platform.is_device_capability_family(120)\n"
                "                and self.kv_cache_spec.num_states in (32, 64)\n"
                "            ):\n"
                "                metadata = get_paged_mqa_logits_metadata(\n",
                "indexer paged MQA logits SM12x guard",
            ),
            (
                "            from vllm.utils.deep_gemm import is_deep_gemm_supported\n"
                "            if current_platform.is_cuda() and is_deep_gemm_supported():\n"
                "                metadata = get_paged_mqa_logits_metadata(\n",
                "            from vllm.utils.deep_gemm import is_deep_gemm_supported\n"
                "            if (\n"
                "                current_platform.is_cuda()\n"
                "                and is_deep_gemm_supported()\n"
                "                and not current_platform.is_device_capability_family(120)\n"
                "                and self.kv_cache_spec.num_states in (32, 64)\n"
                "            ):\n"
                "                metadata = get_paged_mqa_logits_metadata(\n",
                "indexer paged MQA skip SM12x / non-32-64 pages",
            ),
        ],
    )


# Host-sync (.item()) version shipped in the first SM12x MQA overlay.
# Kept so already-built images can be migrated without a full rebuild.
_SM12X_PAGED_MQA_OLD = textwrap.dedent(
    """\
    def _sm12x_fp8_paged_mqa_logits(
        q: tuple[torch.Tensor, torch.Tensor | None],
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        schedule_metadata: torch.Tensor,
        max_model_len: int,
        clean_logits: bool,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values, q_scale = q
        B = block_tables.shape[0]
        block_size = kv_cache.shape[1]
        head_size = kv_cache.shape[3]
        if context_lens.dim() == 2:
            ctx = context_lens[:, -1]
        else:
            ctx = context_lens
        max_ctx = int(ctx.max().item()) if ctx.numel() > 0 else 0
        if q_values.dim() == 4:
            Bq, next_n, H, D = q_values.shape
            q_flat = q_values.reshape(Bq * next_n, H, D)
        else:
            q_flat = q_values
            H, D = q_flat.shape[-2], q_flat.shape[-1]
            next_n = q_flat.shape[0] // B if B > 0 else 1
        M = q_flat.shape[0]
        q_w = _sm12x_dequant_q_weighted(
            q_flat, q_scale.reshape(M, H, -1) if q_scale is not None else None,
            weights[:M],
        )
        if max_ctx == 0:
            return q_w.new_zeros((M, max_model_len), dtype=torch.float32)
        pos = torch.arange(max_ctx, device=kv_cache.device)
        lb = pos // block_size
        off = pos % block_size
        max_blocks = block_tables.shape[1]
        lb_c = lb.clamp(max=max_blocks - 1)
        phys = block_tables[:, lb_c]
        k_raw = kv_cache[phys.reshape(-1), off.repeat(B), 0]
        k_raw = k_raw.reshape(B, max_ctx, head_size)
        k_fp8 = k_raw[:, :, :D].contiguous().view(torch.float8_e4m3fn)
        k_sf = k_raw[:, :, D:D+4].contiguous().view(torch.float32).squeeze(-1)
        k_dq = k_fp8.to(torch.float32) * k_sf.unsqueeze(-1)
        q_w_4d = q_w.reshape(B, next_n, D)
        logits_short = torch.bmm(q_w_4d, k_dq.transpose(1, 2))
        logits_short = logits_short.reshape(M, max_ctx)
        if max_ctx >= max_model_len:
            logits = logits_short[:, :max_model_len]
        else:
            logits = q_w.new_full((M, max_model_len), float('-inf') if clean_logits else 0.0, dtype=torch.float32)
            logits[:, :max_ctx] = logits_short
        ctx_exp = ctx.unsqueeze(1).expand(B, next_n).reshape(M, 1)
        pos_row = torch.arange(min(max_ctx, max_model_len), device=logits.device).unsqueeze(0)
        invalid = pos_row >= ctx_exp
        logits[:, :pos_row.shape[1]].masked_fill_(invalid, float('-inf') if clean_logits else 0.0)
        return logits
    """
)

# CUDA-graph-safe: gather length comes from Python tensor shapes, never GPU
# .item(). Invalid positions (pos >= context_lens) are masked to -inf.
_SM12X_PAGED_MQA_NEW = textwrap.dedent(
    """\
    def _sm12x_fp8_paged_mqa_logits(
        q: tuple[torch.Tensor, torch.Tensor | None],
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        schedule_metadata: torch.Tensor,
        max_model_len: int,
        clean_logits: bool,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values, q_scale = q
        B = block_tables.shape[0]
        block_size = kv_cache.shape[1]
        head_size = kv_cache.shape[3]
        max_blocks = block_tables.shape[1]
        num_blocks = kv_cache.shape[0]
        if q_values.dim() == 4:
            Bq, next_n, H, D = q_values.shape
            q_flat = q_values.reshape(Bq * next_n, H, D)
        else:
            q_flat = q_values
            H, D = q_flat.shape[-2], q_flat.shape[-1]
            next_n = q_flat.shape[0] // B if B > 0 else 1
        M = q_flat.shape[0]
        q_w = _sm12x_dequant_q_weighted(
            q_flat, q_scale.reshape(M, H, -1) if q_scale is not None else None,
            weights[:M],
        )
        logits = q_w.new_full((M, max_model_len), float('-inf'), dtype=torch.float32)
        gather_len = min(max_blocks * block_size, max_model_len)
        if B == 0 or gather_len == 0 or num_blocks == 0:
            return logits
        if context_lens.dim() == 2 and context_lens.shape[1] == next_n:
            ctx_exp = context_lens.to(dtype=torch.int32).reshape(M, 1)
        else:
            ctx = context_lens[:, -1] if context_lens.dim() == 2 else context_lens
            ctx_exp = (
                ctx.to(dtype=torch.int32)
                .unsqueeze(1)
                .expand(B, next_n)
                .reshape(M, 1)
            )
        q_w_4d = q_w.reshape(B, next_n, D)
        chunk = 2048
        phys_hi = max(num_blocks - 1, 0)
        blk_hi = max(max_blocks - 1, 0)
        for start in range(0, gather_len, chunk):
            n = min(chunk, gather_len - start)
            pos = torch.arange(start, start + n, device=kv_cache.device)
            lb = (pos // block_size).clamp(max=blk_hi)
            off = pos % block_size
            phys = block_tables[:, lb].clamp(0, phys_hi)
            k_raw = kv_cache[phys.reshape(-1), off.repeat(B), 0]
            k_raw = k_raw.reshape(B, n, head_size)
            k_fp8 = k_raw[:, :, :D].contiguous().view(torch.float8_e4m3fn)
            k_sf = k_raw[:, :, D:D+4].contiguous().view(torch.float32).squeeze(-1)
            k_dq = k_fp8.to(torch.float32) * k_sf.unsqueeze(-1)
            chunk_logits = torch.bmm(q_w_4d, k_dq.transpose(1, 2)).reshape(M, n)
            invalid = pos.unsqueeze(0) >= ctx_exp
            chunk_logits = chunk_logits.masked_fill(invalid, float('-inf'))
            logits[:, start:start + n] = chunk_logits
        return logits
    """
)

# Matches vllm.v1.attention.ops.rocm_aiter_mla_sparse.fp8_mqa_logits_torch:
# logits[m,n] = sum_h w[m,h] * relu((q[m,h]·k[n]) * scale[n])
# The weighted-Q contraction is not equivalent once ReLU is in the mix.
_SM12X_MQA_PREFILL_OLD = textwrap.dedent(
    """\
    def _sm12x_fp8_mqa_logits(
        q: tuple[torch.Tensor, torch.Tensor | None],
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        clean_logits: bool,
    ) -> torch.Tensor:
        q_values, q_scale = q
        k_packed, k_scales = kv
        q_w = _sm12x_dequant_q_weighted(q_values, q_scale, weights)
        k_dq = _sm12x_dequant_k(k_packed, k_scales)
        logits = q_w @ k_dq.T
        if clean_logits:
            N = logits.shape[1]
            pos = torch.arange(N, device=logits.device).unsqueeze(0)
            mask = (pos >= cu_seqlen_ks.unsqueeze(1)) & (pos < cu_seqlen_ke.unsqueeze(1))
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits
    """
)

_SM12X_MQA_PREFILL_NEW = textwrap.dedent(
    """\
    def _sm12x_fp8_mqa_logits(
        q: tuple[torch.Tensor, torch.Tensor | None],
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        clean_logits: bool,
    ) -> torch.Tensor:
        q_values, q_scale = q
        k_packed, k_scales = kv
        q_f = q_values.to(torch.float32)
        if q_scale is not None:
            q_sf = q_scale.to(torch.float32)
            if q_sf.shape != q_f.shape:
                q_sf = q_sf.expand_as(q_f)
            q_f = q_f * q_sf
        if q_f.dim() == 4:
            q_f = q_f.reshape(-1, q_f.shape[2], q_f.shape[3])
        k_f = k_packed.to(torch.float32)
        k_sf = k_scales.to(torch.float32).reshape(-1)
        M, H, D = q_f.shape
        N = k_f.shape[0]
        w = weights[:M].to(torch.float32)
        logits = q_f.new_empty((M, N), dtype=torch.float32)
        chunk = 256
        for start in range(0, N, chunk):
            n = min(chunk, N - start)
            scores = torch.einsum('mhd,nd->mhn', q_f, k_f[start:start + n])
            scores = (scores * k_sf[start:start + n]).relu() * w.unsqueeze(-1)
            logits[:, start:start + n] = scores.sum(dim=1)
        pos = torch.arange(N, device=logits.device).unsqueeze(0)
        mask = (pos >= cu_seqlen_ks.unsqueeze(1)) & (pos < cu_seqlen_ke.unsqueeze(1))
        return logits.masked_fill(~mask, float('-inf'))
    """
)

_SM12X_PAGED_MQA_RELU = textwrap.dedent(
    """\
    def _sm12x_fp8_paged_mqa_logits(
        q: tuple[torch.Tensor, torch.Tensor | None],
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        schedule_metadata: torch.Tensor,
        max_model_len: int,
        clean_logits: bool,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values, q_scale = q
        B = block_tables.shape[0]
        block_size = kv_cache.shape[1]
        head_size = kv_cache.shape[3]
        max_blocks = block_tables.shape[1]
        num_blocks = kv_cache.shape[0]
        if q_values.dim() == 4:
            Bq, next_n, H, D = q_values.shape
            q_flat = q_values.reshape(Bq * next_n, H, D)
        else:
            q_flat = q_values
            H, D = q_flat.shape[-2], q_flat.shape[-1]
            next_n = q_flat.shape[0] // B if B > 0 else 1
        M = q_flat.shape[0]
        q_f = q_flat.to(torch.float32)
        if q_scale is not None:
            q_sf = q_scale.to(torch.float32).reshape(M, H, -1)
            if q_sf.shape[-1] != D:
                q_sf = q_sf.expand(M, H, D)
            q_f = q_f * q_sf
        w = weights[:M].to(torch.float32).reshape(B, next_n, H)
        logits = q_f.new_full((M, max_model_len), float('-inf'))
        gather_len = min(max_blocks * block_size, max_model_len)
        if B == 0 or gather_len == 0 or num_blocks == 0:
            return logits
        if context_lens.dim() == 2 and context_lens.shape[1] == next_n:
            ctx_exp = context_lens.to(dtype=torch.int32).reshape(M, 1)
        else:
            ctx = context_lens[:, -1] if context_lens.dim() == 2 else context_lens
            ctx_exp = (
                ctx.to(dtype=torch.int32)
                .unsqueeze(1)
                .expand(B, next_n)
                .reshape(M, 1)
            )
        q_4d = q_f.reshape(B, next_n, H, D)
        chunk = 256
        phys_hi = max(num_blocks - 1, 0)
        blk_hi = max(max_blocks - 1, 0)
        for start in range(0, gather_len, chunk):
            n = min(chunk, gather_len - start)
            pos = torch.arange(start, start + n, device=kv_cache.device)
            lb = (pos // block_size).clamp(max=blk_hi)
            off = pos % block_size
            phys = block_tables[:, lb].clamp(0, phys_hi)
            k_raw = kv_cache[phys.reshape(-1), off.repeat(B), 0]
            k_raw = k_raw.reshape(B, n, head_size)
            k_fp8 = k_raw[:, :, :D].contiguous().view(torch.float8_e4m3fn)
            k_sf = k_raw[:, :, D:D+4].contiguous().view(torch.float32).squeeze(-1)
            k_dq = k_fp8.to(torch.float32) * k_sf.unsqueeze(-1)
            scores = torch.einsum('bnhd,bkd->bnhk', q_4d, k_dq)
            scores = scores.relu() * w.unsqueeze(-1)
            chunk_logits = scores.sum(dim=2).reshape(M, n)
            invalid = pos.unsqueeze(0) >= ctx_exp
            chunk_logits = chunk_logits.masked_fill(invalid, float('-inf'))
            logits[:, start:start + n] = chunk_logits
        return logits
    """
)

# B12x 1.2.6 NSA indexer: contiguous ReLU-sum scorer. Paged CUDA kernel is
# hardcoded to page_size=64 with a packed 2-D uint8 layout; FlashInfer DSV4
# keeps 256-token pages, so decode gathers then scores contiguous.
_SM12X_B12X_MQA_HELPER = textwrap.dedent(
    """\
    def _sm12x_b12x_mqa_pack():
        cached = getattr(_sm12x_b12x_mqa_pack, "_cached", None)
        if cached is not None:
            return cached
        try:
            from b12x.attention.nsa_indexer._impl import (
                IndexerContiguousMetadata,
                contiguous_logits,
                supports_contiguous_logits_kernel,
            )
            cached = (
                IndexerContiguousMetadata,
                contiguous_logits,
                supports_contiguous_logits_kernel,
            )
        except Exception:
            cached = False
        _sm12x_b12x_mqa_pack._cached = cached
        return cached
    """
)

_SM12X_MQA_PREFILL_B12X = textwrap.dedent(
    """\
    def _sm12x_fp8_mqa_logits(
        q: tuple[torch.Tensor, torch.Tensor | None],
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        clean_logits: bool,
    ) -> torch.Tensor:
        q_values, q_scale = q
        k_packed, k_scales = kv
        q_fp8 = q_values
        if q_fp8.dim() == 4:
            q_fp8 = q_fp8.reshape(-1, q_fp8.shape[2], q_fp8.shape[3])
        pack = _sm12x_b12x_mqa_pack()
        if (
            pack
            and q_scale is None
            and q_fp8.dtype == torch.float8_e4m3fn
            and q_fp8.dim() == 3
            and q_fp8.shape[-1] == 128
            and k_packed.dtype == torch.float8_e4m3fn
        ):
            meta_cls, contiguous_logits, supports = pack
            M, H, _D = q_fp8.shape
            w = weights[:M].to(torch.float32)
            if w.shape != (M, H):
                w = w.reshape(M, H)
            k_fp8 = k_packed.reshape(k_packed.shape[0], -1).contiguous()
            k_sf = k_scales.to(torch.float32).reshape(-1).contiguous()
            ks = cu_seqlen_ks.to(torch.int32).contiguous()
            ke = cu_seqlen_ke.to(torch.int32).contiguous()
            q_c = q_fp8.contiguous()
            w_c = w.contiguous()
            if supports(
                q_fp8=q_c,
                weights=w_c,
                k_quant=k_fp8,
                k_scale=k_sf,
                k_start=ks,
                k_end=ke,
            ):
                return contiguous_logits(
                    q_fp8=q_c,
                    weights=w_c,
                    kv_fp8=(k_fp8, k_sf),
                    metadata=meta_cls(k_start=ks, k_end=ke),
                    score_mode=0,
                )
        q_f = q_values.to(torch.float32)
        if q_scale is not None:
            q_sf = q_scale.to(torch.float32)
            if q_sf.shape != q_f.shape:
                q_sf = q_sf.expand_as(q_f)
            q_f = q_f * q_sf
        if q_f.dim() == 4:
            q_f = q_f.reshape(-1, q_f.shape[2], q_f.shape[3])
        k_f = k_packed.to(torch.float32)
        k_sf = k_scales.to(torch.float32).reshape(-1)
        M, H, D = q_f.shape
        N = k_f.shape[0]
        w = weights[:M].to(torch.float32)
        logits = q_f.new_empty((M, N), dtype=torch.float32)
        chunk = 256
        for start in range(0, N, chunk):
            n = min(chunk, N - start)
            scores = torch.einsum('mhd,nd->mhn', q_f, k_f[start:start + n])
            scores = (scores * k_sf[start:start + n]).relu() * w.unsqueeze(-1)
            logits[:, start:start + n] = scores.sum(dim=1)
        pos = torch.arange(N, device=logits.device).unsqueeze(0)
        mask = (pos >= cu_seqlen_ks.unsqueeze(1)) & (pos < cu_seqlen_ke.unsqueeze(1))
        return logits.masked_fill(~mask, float('-inf'))
    """
)

_SM12X_PAGED_MQA_B12X = textwrap.dedent(
    """\
    def _sm12x_fp8_paged_mqa_logits(
        q: tuple[torch.Tensor, torch.Tensor | None],
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        schedule_metadata: torch.Tensor,
        max_model_len: int,
        clean_logits: bool,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values, q_scale = q
        B = block_tables.shape[0]
        block_size = kv_cache.shape[1]
        head_size = kv_cache.shape[3]
        max_blocks = block_tables.shape[1]
        num_blocks = kv_cache.shape[0]
        if q_values.dim() == 4:
            Bq, next_n, H, D = q_values.shape
            q_flat = q_values.reshape(Bq * next_n, H, D)
        else:
            q_flat = q_values
            H, D = q_flat.shape[-2], q_flat.shape[-1]
            next_n = q_flat.shape[0] // B if B > 0 else 1
        M = q_flat.shape[0]
        if context_lens.dim() == 2 and context_lens.shape[1] == next_n:
            ctx_exp = context_lens.to(dtype=torch.int32).reshape(M, 1)
        else:
            ctx = context_lens[:, -1] if context_lens.dim() == 2 else context_lens
            ctx_exp = (
                ctx.to(dtype=torch.int32)
                .unsqueeze(1)
                .expand(B, next_n)
                .reshape(M, 1)
            )
        gather_len = min(max_blocks * block_size, max_model_len)
        logits = q_flat.new_full((M, max_model_len), float('-inf'), dtype=torch.float32)
        if B == 0 or gather_len == 0 or num_blocks == 0:
            return logits
        pack = _sm12x_b12x_mqa_pack()
        if (
            pack
            and q_scale is None
            and q_flat.dtype == torch.float8_e4m3fn
            and D == 128
        ):
            meta_cls, contiguous_logits, supports = pack
            w = weights[:M].to(torch.float32)
            if w.shape != (M, H):
                w = w.reshape(M, H)
            phys_hi = max(num_blocks - 1, 0)
            blk_hi = max(max_blocks - 1, 0)
            used = True
            for b in range(B):
                pos = torch.arange(gather_len, device=kv_cache.device)
                lb = (pos // block_size).clamp(max=blk_hi)
                off = pos % block_size
                phys = block_tables[b, lb].clamp(0, phys_hi)
                k_raw = kv_cache[phys, off, 0]
                k_fp8 = k_raw[:, :D].contiguous().view(torch.float8_e4m3fn)
                k_sf = k_raw[:, D:D + 4].contiguous().view(torch.float32).reshape(-1)
                q_b = q_flat[b * next_n:(b + 1) * next_n].contiguous()
                w_b = w[b * next_n:(b + 1) * next_n].contiguous()
                ke = ctx_exp[b * next_n:(b + 1) * next_n, 0].to(torch.int32).clamp(
                    max=gather_len
                )
                ks = torch.zeros(q_b.shape[0], dtype=torch.int32, device=kv_cache.device)
                if not supports(
                    q_fp8=q_b,
                    weights=w_b,
                    k_quant=k_fp8,
                    k_scale=k_sf,
                    k_start=ks,
                    k_end=ke,
                ):
                    used = False
                    break
                scored = contiguous_logits(
                    q_fp8=q_b,
                    weights=w_b,
                    kv_fp8=(k_fp8, k_sf),
                    metadata=meta_cls(k_start=ks, k_end=ke),
                    score_mode=0,
                )
                n_out = scored.shape[1]
                if n_out > max_model_len:
                    n_out = max_model_len
                logits[b * next_n:(b + 1) * next_n, :n_out] = scored[:, :n_out]
            if used:
                return logits
        q_f = q_flat.to(torch.float32)
        if q_scale is not None:
            q_sf = q_scale.to(torch.float32).reshape(M, H, -1)
            if q_sf.shape[-1] != D:
                q_sf = q_sf.expand(M, H, D)
            q_f = q_f * q_sf
        w = weights[:M].to(torch.float32).reshape(B, next_n, H)
        q_4d = q_f.reshape(B, next_n, H, D)
        chunk = 256
        phys_hi = max(num_blocks - 1, 0)
        blk_hi = max(max_blocks - 1, 0)
        for start in range(0, gather_len, chunk):
            n = min(chunk, gather_len - start)
            pos = torch.arange(start, start + n, device=kv_cache.device)
            lb = (pos // block_size).clamp(max=blk_hi)
            off = pos % block_size
            phys = block_tables[:, lb].clamp(0, phys_hi)
            k_raw = kv_cache[phys.reshape(-1), off.repeat(B), 0]
            k_raw = k_raw.reshape(B, n, head_size)
            k_fp8 = k_raw[:, :, :D].contiguous().view(torch.float8_e4m3fn)
            k_sf = k_raw[:, :, D:D+4].contiguous().view(torch.float32).squeeze(-1)
            k_dq = k_fp8.to(torch.float32) * k_sf.unsqueeze(-1)
            scores = torch.einsum('bnhd,bkd->bnhk', q_4d, k_dq)
            scores = scores.relu() * w.unsqueeze(-1)
            chunk_logits = scores.sum(dim=2).reshape(M, n)
            invalid = pos.unsqueeze(0) >= ctx_exp
            chunk_logits = chunk_logits.masked_fill(invalid, float('-inf'))
            logits[:, start:start + n] = chunk_logits
        return logits
    """
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
        + _SM12X_PAGED_MQA_NEW
        + "\n\n"
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


def patch_mqa_paged_cudagraph_safe(vllm: Path) -> None:
    """Replace the host-sync paged MQA fallback with a CUDA-graph-safe one.

    FULL cudagraph capture forbids ctx.max().item(). Gather length is taken
    from block_tables / kv_cache shapes (Python ints) and chunked.
    """
    path = vllm / "utils/deep_gemm.py"
    replace_once(
        path,
        _SM12X_PAGED_MQA_OLD,
        _SM12X_PAGED_MQA_NEW,
        "fp8_fp4_paged_mqa_logits CUDA-graph-safe gather",
    )


def patch_mqa_relu_formula(vllm: Path) -> None:
    """DSA indexer scores are sum_h w_h * relu(q_h · k), not weighted-Q.

    Matches fp8_mqa_logits_torch in vllm.v1.attention.ops.rocm_aiter_mla_sparse.
    """
    path = vllm / "utils/deep_gemm.py"
    replace_once(
        path,
        _SM12X_MQA_PREFILL_OLD,
        _SM12X_MQA_PREFILL_NEW,
        "fp8_fp4_mqa_logits ReLU formula",
    )
    replace_once(
        path,
        _SM12X_PAGED_MQA_NEW,
        _SM12X_PAGED_MQA_RELU,
        "fp8_fp4_paged_mqa_logits ReLU formula",
    )


def patch_mqa_b12x(vllm: Path) -> None:
    """Score DSA indexer logits with B12x contiguous ReLU-sum on SM12x.

    Prefill is a drop-in for gathered FP8 K. Decode gathers from 256-token
    pages then calls the same contiguous kernel (the B12x paged kernel is
    page_size=64 only). Python ReLU remains the fallback.
    """
    path = vllm / "utils/deep_gemm.py"
    replace_once(
        path,
        _SM12X_MQA_PREFILL_NEW,
        _SM12X_B12X_MQA_HELPER + "\n\n" + _SM12X_MQA_PREFILL_B12X,
        "fp8_fp4_mqa_logits B12x contiguous",
    )
    replace_once(
        path,
        _SM12X_PAGED_MQA_RELU,
        _SM12X_PAGED_MQA_B12X,
        "fp8_fp4_paged_mqa_logits B12x gather+contiguous",
    )


def patch_sm12x_kv_insert(vllm: Path) -> None:
    """On SM12x, write SWA KV with the Triton footer-scale insert.

    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert is the v0.27.1 CUDA
    op. FlashInfer DSV4 reads block-footer UE8M0 (576B data then 8B scales).
    The XPU Triton path (quantize_and_insert_k_cache) writes that layout
    and runs on CUDA.
    """
    attn = vllm / "models/deepseek_v4/attention.py"
    sm12x_insert = (
        "        # kv is unchanged; attention reads kv solely via swa_kv_cache.\n"
        "        if cache_dtype == torch.uint8:\n"
        "            from vllm.platforms import current_platform\n"
        "\n"
        "            if current_platform.is_device_capability_family(120):\n"
        "                from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (\n"
        "                    xpu_qnorm_rope_kv_fp8_insert,\n"
        "                )\n"
        "\n"
        "                q_work = q\n"
        "                if self.eager_scratch_pool is not None:\n"
        "                    q_work = self.eager_scratch_pool.q_out(q.shape[0])\n"
        "                    q_work.copy_(q)\n"
        "                logger.info_once(\n"
        "                    \"SM12x: Triton qnorm-rope-KV insert \"\n"
        "                    \"(CUDA fused insert skipped).\"\n"
        "                )\n"
        "                xpu_qnorm_rope_kv_fp8_insert(\n"
        "                    q_work,\n"
        "                    kv,\n"
        "                    swa_kv_cache,\n"
        "                    swa_metadata.slot_mapping,\n"
        "                    positions,\n"
        "                    cos_sin_cache,\n"
        "                    self.eps,\n"
        "                    swa_metadata.block_size,\n"
        "                )\n"
        "                return q_work\n"
    )
    replace_one_of(
        attn,
        [
            (
                "        # kv is unchanged; attention reads kv solely via swa_kv_cache.\n"
                "        if cache_dtype == torch.uint8:\n"
                "            # fp8_ds_mla UE8M0 paged path. Horizontally fused:\n"
                "            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling\n"
                "            #            the padding head slots.\n"
                "            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.\n"
                "            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)\n"
                "            if self.eager_scratch_pool is not None:\n"
                "                q_out = self.eager_scratch_pool.q_out(q.shape[0])\n"
                "                torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(\n"
                "                    q,\n"
                "                    kv,\n"
                "                    q_out,\n"
                "                    swa_kv_cache_2d,\n"
                "                    swa_metadata.slot_mapping,\n"
                "                    positions,\n"
                "                    cos_sin_cache,\n"
                "                    self.padded_heads,\n"
                "                    self.eps,\n"
                "                    swa_metadata.block_size,\n"
                "                )\n"
                "                return q_out\n"
                "            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(\n"
                "                q,\n"
                "                kv,\n"
                "                swa_kv_cache_2d,\n"
                "                swa_metadata.slot_mapping,\n"
                "                positions,\n"
                "                cos_sin_cache,\n"
                "                self.padded_heads,\n"
                "                self.eps,\n"
                "                swa_metadata.block_size,\n"
                "            )\n",
                sm12x_insert
                + "            # fp8_ds_mla UE8M0 paged path. Horizontally fused:\n"
                "            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling\n"
                "            #            the padding head slots.\n"
                "            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.\n"
                "            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)\n"
                "            if self.eager_scratch_pool is not None:\n"
                "                q_out = self.eager_scratch_pool.q_out(q.shape[0])\n"
                "                torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(\n"
                "                    q,\n"
                "                    kv,\n"
                "                    q_out,\n"
                "                    swa_kv_cache_2d,\n"
                "                    swa_metadata.slot_mapping,\n"
                "                    positions,\n"
                "                    cos_sin_cache,\n"
                "                    self.padded_heads,\n"
                "                    self.eps,\n"
                "                    swa_metadata.block_size,\n"
                "                )\n"
                "                return q_out\n"
                "            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(\n"
                "                q,\n"
                "                kv,\n"
                "                swa_kv_cache_2d,\n"
                "                swa_metadata.slot_mapping,\n"
                "                positions,\n"
                "                cos_sin_cache,\n"
                "                self.padded_heads,\n"
                "                self.eps,\n"
                "                swa_metadata.block_size,\n"
                "            )\n",
                "SM12x Triton SWA KV insert",
            ),
            (
                "        # kv is unchanged; attention reads kv solely via swa_kv_cache.\n"
                "        if cache_dtype == torch.uint8:\n"
                "            # fp8_ds_mla UE8M0 paged path. Horizontally fused:\n"
                "            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling\n"
                "            #            the padding head slots; the kernel allocates and returns\n"
                "            #            the padded q tensor.\n"
                "            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.\n"
                "            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)\n"
                "            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(\n"
                "                q,\n"
                "                kv,\n"
                "                swa_kv_cache_2d,\n"
                "                swa_metadata.slot_mapping,\n"
                "                positions,\n"
                "                cos_sin_cache,\n"
                "                self.padded_heads,\n"
                "                self.eps,\n"
                "                swa_metadata.block_size,\n"
                "            )\n",
                sm12x_insert
                + "            # fp8_ds_mla UE8M0 paged path. Horizontally fused:\n"
                "            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling\n"
                "            #            the padding head slots; the kernel allocates and returns\n"
                "            #            the padded q tensor.\n"
                "            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.\n"
                "            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)\n"
                "            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(\n"
                "                q,\n"
                "                kv,\n"
                "                swa_kv_cache_2d,\n"
                "                swa_metadata.slot_mapping,\n"
                "                positions,\n"
                "                cos_sin_cache,\n"
                "                self.padded_heads,\n"
                "                self.eps,\n"
                "                swa_metadata.block_size,\n"
                "            )\n",
                "SM12x Triton SWA KV insert (main)",
            ),
        ],
    )

    dspark = vllm / "models/deepseek_v4/nvidia/dspark.py"
    replace_once(
        dspark,
        "    if cache_dtype == torch.uint8:\n"
        "        # fp8_ds_mla UE8M0 paged layout\n"
        "        swa_2d = swa_cache.view(swa_cache.shape[0], -1)\n"
        "        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(\n"
        "            dummy_q,\n"
        "            kv,\n"
        "            swa_2d,\n"
        "            slot_mapping,\n"
        "            positions,\n"
        "            cos_sin_cache,\n"
        "            attn.padded_heads,\n"
        "            attn.eps,\n"
        "            block_size,\n"
        "        )\n",
        "    if cache_dtype == torch.uint8:\n"
        "        from vllm.platforms import current_platform\n"
        "\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (\n"
        "                xpu_qnorm_rope_kv_fp8_insert,\n"
        "            )\n"
        "\n"
        "            xpu_qnorm_rope_kv_fp8_insert(\n"
        "                dummy_q,\n"
        "                kv,\n"
        "                swa_cache,\n"
        "                slot_mapping,\n"
        "                positions,\n"
        "                cos_sin_cache,\n"
        "                attn.eps,\n"
        "                block_size,\n"
        "            )\n"
        "            return\n"
        "        # fp8_ds_mla UE8M0 paged layout\n"
        "        swa_2d = swa_cache.view(swa_cache.shape[0], -1)\n"
        "        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(\n"
        "            dummy_q,\n"
        "            kv,\n"
        "            swa_2d,\n"
        "            slot_mapping,\n"
        "            positions,\n"
        "            cos_sin_cache,\n"
        "            attn.padded_heads,\n"
        "            attn.eps,\n"
        "            block_size,\n"
        "        )\n",
        "SM12x Triton DSpark SWA KV insert",
    )
    attn_text = attn.read_text()
    scratch_old = "self.eager_scratch_pool"
    scratch_new = 'getattr(self, "eager_scratch_pool", None)'
    if scratch_new in attn_text:
        print("skip SM12x eager_scratch getattr: already applied")
    elif scratch_old not in attn_text:
        raise SystemExit("SM12x eager_scratch getattr: missing needle")
    else:
        n = attn_text.count(scratch_old)
        attn.write_text(attn_text.replace(scratch_old, scratch_new))
        print(f"ok SM12x eager_scratch getattr ({n})")


_LOGIT_DUMP_LIVE = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        n = getattr(self, \"_logit_dumps\", 0)\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        if n < 6 and logits is not None:\n"
    "            self._logit_dumps = n + 1\n"
    "            h = hidden_states.detach().float()\n"
    "            lf = logits.detach().float()\n"
    "            last = lf.reshape(-1, lf.shape[-1])[-1]\n"
    "            mx = last.max()\n"
    "            n_tie = int((last >= mx - 1e-3).sum().item())\n"
    "            topv, topi = torch.topk(last, 8)\n"
    "            w = self.lm_head.weight.detach().float()\n"
    "            print(\n"
    "                f\"LOGIT_DUMP#{n} hidden={tuple(hidden_states.shape)} \"\n"
    "                f\"h_finite={torch.isfinite(h).all().item()} \"\n"
    "                f\"h_nan={torch.isnan(h).any().item()} \"\n"
    "                f\"h_mean={h.mean().item():.5g} h_std={h.std().item():.5g} \"\n"
    "                f\"h_absmax={h.abs().max().item():.5g} \"\n"
    "                f\"h_rms={h.pow(2).mean().sqrt().item():.5g} \"\n"
    "                f\"logits={tuple(logits.shape)} \"\n"
    "                f\"l_finite={int(torch.isfinite(lf).sum().item())}/{lf.numel()} \"\n"
    "                f\"l_nan={torch.isnan(lf).any().item()} \"\n"
    "                f\"l_mean={lf.mean().item():.5g} l_std={lf.std().item():.5g} \"\n"
    "                f\"l_max={mx.item():.5g} n_tie={n_tie} \"\n"
    "                f\"top_ids={topi.tolist()} top_v={[round(x, 4) for x in topv.tolist()]} \"\n"
    "                f\"lm_head={tuple(self.lm_head.weight.shape)} \"\n"
    "                f\"w_rms={w.pow(2).mean().sqrt().item():.5g}\",\n"
    "                flush=True,\n"
    "            )\n"
    "        return logits\n"
)

_LOGIT_DUMP_SAFE = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        n = getattr(self, \"_logit_dumps\", 0)\n"
    "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
    "        capturing = torch.cuda.is_current_stream_capturing()\n"
    "        if n < 16 and logits is not None and not capturing:\n"
    "            self._logit_dumps = n + 1\n"
    "            last = logits.detach().reshape(-1, logits.shape[-1])[-1]\n"
    "            mx = last.max()\n"
    "            n_tie = int((last >= mx - 1e-3).sum().item())\n"
    "            topv, topi = torch.topk(last.float(), 8)\n"
    "            w = self.lm_head.weight\n"
    "            w_finite = bool(torch.isfinite(w).all().item())\n"
    "            w_rms = float(w[:32].detach().float().pow(2).mean().sqrt())\n"
    "            print(\n"
    "                f\"LOGIT_DUMP#{n} hidden={tuple(hidden_states.shape)} \"\n"
    "                f\"h_finite={bool(torch.isfinite(hidden_states).all().item())} \"\n"
    "                f\"logits={tuple(logits.shape)} \"\n"
    "                f\"l_max={float(mx):.5g} n_tie={n_tie} \"\n"
    "                f\"top_ids={topi.tolist()} \"\n"
    "                f\"top_v={[round(x, 4) for x in topv.tolist()]} \"\n"
    "                f\"w_finite={w_finite} w_sample_rms={w_rms:.5g}\",\n"
    "                flush=True,\n"
    "            )\n"
    "        return logits\n"
)


def patch_logit_dump(vllm: Path) -> None:
    """Print hidden/logit stats for the first few compute_logits calls."""
    path = vllm / "models/deepseek_v4/nvidia/model.py"
    text = path.read_text()
    if _LOGIT_DUMP_SAFE in text:
        print("skip compute_logits hidden/logit dump: already applied")
        return
    if _LOGIT_DUMP_LIVE in text:
        replace_once(
            path,
            _LOGIT_DUMP_LIVE,
            _LOGIT_DUMP_SAFE,
            "compute_logits dump skip-capture + no full fp32 copy",
        )
        return
    replace_once(
        path,
        "    def compute_logits(\n"
        "        self,\n"
        "        hidden_states: torch.Tensor,\n"
        "    ) -> torch.Tensor | None:\n"
        "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
        "        return logits\n",
        _LOGIT_DUMP_SAFE,
        "compute_logits hidden/logit dump",
    )


def patch_dspark_skip_cudagraph(vllm: Path) -> None:
    """Keep the DSpark draft step eager.

    DSpark FULL graphs wrap `_generate_draft`, which runs
    `compute_draft_logits` on the *shared* target `lm_head` plus a TP
    all-gather. On 2-node GB10 that capture leaves `lm_head.weight` at inf
    and greedy France collapses to a 96-way tie (-ln(96)). Target-model
    FULL graphs stay enabled.
    """
    path = vllm / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    replace_once(
        path,
        "        # PIECEWISE cudagraphs are not supported for dflash.\n"
        "        if wants_full and supports_full:\n"
        "            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY\n"
        "        else:\n"
        "            cudagraph_mode = CUDAGraphMode.NONE\n",
        "        # PIECEWISE cudagraphs are not supported for dflash.\n"
        "        # 2-node GB10: capturing the draft step (shared lm_head GEMM\n"
        "        # + TP all-gather) corrupts lm_head.weight (w_rms -> inf).\n"
        "        logger.info(\n"
        "            \"%s CUDA graphs disabled: draft capture corrupts shared lm_head\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        cudagraph_mode = CUDAGraphMode.NONE\n",
        "DSpark/DFlash skip CUDA graphs (shared lm_head)",
    )
    replace_once(
        path,
        "    def capture(self) -> None:\n"
        "        logger.info(\"Capturing model for %s speculator...\", self._speculator_name)\n",
        "    def capture(self) -> None:\n"
        "        logger.info(\n"
        "            \"Skipping %s CUDA graph capture (shared lm_head)\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        return\n"
        "        logger.info(\"Capturing model for %s speculator...\", self._speculator_name)\n",
        "DSpark/DFlash capture() no-op",
    )


def patch_instanttensor_hybrid_draft(vllm: Path) -> None:
    """Keep InstantTensor for the target; lazy safetensors for same-path draft.

    DSpark loads a second model from the 0731 checkpoint. A second InstantTensor
    pass restreams 155 GiB. eugr mods/instanttensor-hybrid-draft-loader.
    INSTANTTENSOR_DRAFT_LOADER=auto|safetensors|instanttensor.
    """
    path = vllm / "model_executor/model_loader/__init__.py"
    replace_once(
        path,
        "def get_model(\n"
        "    *,\n"
        "    vllm_config: VllmConfig,\n"
        "    model_config: ModelConfig | None = None,\n"
        "    prefix: str = \"\",\n"
        "    load_config: LoadConfig | None = None,\n"
        ") -> nn.Module:\n"
        "    loader = get_model_loader(load_config or vllm_config.load_config)\n"
        "    if model_config is None:\n"
        "        model_config = vllm_config.model_config\n"
        "    return loader.load_model(\n"
        "        vllm_config=vllm_config, model_config=model_config, prefix=prefix\n"
        "    )\n",
        "# spark-0731: instanttensor-hybrid-draft-loader\n"
        "def _instanttensor_draft_load_config(\n"
        "    vllm_config: VllmConfig,\n"
        "    model_config: ModelConfig,\n"
        "    load_config: LoadConfig | None,\n"
        ") -> LoadConfig:\n"
        "    import os\n"
        "\n"
        "    from vllm.config import replace\n"
        "\n"
        "    mode = os.environ.get(\"INSTANTTENSOR_DRAFT_LOADER\", \"auto\").strip().lower()\n"
        "    allowed_modes = (\"auto\", \"safetensors\", \"instanttensor\")\n"
        "    if mode not in allowed_modes:\n"
        "        raise ValueError(\n"
        "            \"INSTANTTENSOR_DRAFT_LOADER must be one of \"\n"
        "            f\"{', '.join(allowed_modes)}; got {mode!r}\"\n"
        "        )\n"
        "    effective = load_config or vllm_config.load_config\n"
        "    load_format = getattr(effective.load_format, \"value\", effective.load_format)\n"
        "    if mode == \"instanttensor\" or str(load_format).lower() != \"instanttensor\":\n"
        "        return effective\n"
        "    speculative_config = getattr(vllm_config, \"speculative_config\", None)\n"
        "    draft_model_config = getattr(speculative_config, \"draft_model_config\", None)\n"
        "    if draft_model_config is None or model_config is not draft_model_config:\n"
        "        return effective\n"
        "    if mode == \"auto\":\n"
        "        target_model_config = (\n"
        "            getattr(speculative_config, \"target_model_config\", None)\n"
        "            or vllm_config.model_config\n"
        "        )\n"
        "        draft_source = (\n"
        "            getattr(draft_model_config, \"model\", None),\n"
        "            getattr(draft_model_config, \"revision\", None),\n"
        "        )\n"
        "        target_source = (\n"
        "            getattr(target_model_config, \"model\", None),\n"
        "            getattr(target_model_config, \"revision\", None),\n"
        "        )\n"
        "        if draft_source != target_source:\n"
        "            return effective\n"
        "    logger.info_once(\n"
        "        \"Hybrid draft loading: using lazy safetensors for speculative draft \"\n"
        "        \"weights while preserving InstantTensor for the target model \"\n"
        "        \"(INSTANTTENSOR_DRAFT_LOADER=%s).\",\n"
        "        mode,\n"
        "    )\n"
        "    return replace(\n"
        "        effective,\n"
        "        load_format=\"safetensors\",\n"
        "        safetensors_load_strategy=\"lazy\",\n"
        "    )\n"
        "\n"
        "\n"
        "def get_model(\n"
        "    *,\n"
        "    vllm_config: VllmConfig,\n"
        "    model_config: ModelConfig | None = None,\n"
        "    prefix: str = \"\",\n"
        "    load_config: LoadConfig | None = None,\n"
        ") -> nn.Module:\n"
        "    if model_config is None:\n"
        "        model_config = vllm_config.model_config\n"
        "    resolved_load_config = _instanttensor_draft_load_config(\n"
        "        vllm_config, model_config, load_config\n"
        "    )\n"
        "    loader = get_model_loader(resolved_load_config)\n"
        "    return loader.load_model(\n"
        "        vllm_config=vllm_config, model_config=model_config, prefix=prefix\n"
        "    )\n",
        "InstantTensor hybrid draft loader",
    )


def patch_lm_head_restore_after_graphs(vllm: Path) -> None:
    """CPU-clone lm_head before CUDA graph capture and copy it back after.

    One-shot writes into the parameter during capture then persist. Replay
    of F.linear still reads the same storage, so restoring after capture
    recovers greedy quality if the graph itself does not keep writing inf.
    """
    path = vllm / "v1/worker/gpu/model_runner.py"
    replace_once(
        path,
        "        assert self.cudagraph_manager is not None\n"
        "        capture_encoder = (\n",
        "        assert self.cudagraph_manager is not None\n"
        "        _lm = getattr(self.model, \"lm_head\", None)\n"
        "        _lm_w = getattr(_lm, \"weight\", None) if _lm is not None else None\n"
        "        _lm_backup = _lm_w.detach().cpu().clone() if _lm_w is not None else None\n"
        "        capture_encoder = (\n",
        "capture_model CPU-clone lm_head before graphs",
    )
    replace_once(
        path,
        "        logger.info(\n"
        "            \"Graph capturing finished in %.0f secs, took %.2f GiB\",\n"
        "            elapsed_time,\n"
        "            cuda_graph_size / (1 << 30),\n"
        "        )\n"
        "        return cuda_graph_size\n",
        "        logger.info(\n"
        "            \"Graph capturing finished in %.0f secs, took %.2f GiB\",\n"
        "            elapsed_time,\n"
        "            cuda_graph_size / (1 << 30),\n"
        "        )\n"
        "        if _lm_backup is not None and _lm_w is not None:\n"
        "            _lm_w.copy_(_lm_backup.to(device=_lm_w.device, dtype=_lm_w.dtype))\n"
        "            _ok = bool(torch.isfinite(_lm_w).all().item())\n"
        "            print(\n"
        "                f\"LM_HEAD_RESTORE finite={_ok} shape={tuple(_lm_w.shape)}\",\n"
        "                flush=True,\n"
        "            )\n"
        "            del _lm_backup\n"
        "        return cuda_graph_size\n",
        "capture_model restore lm_head after graphs",
    )


def patch_kv_zeroer_skip(vllm: Path) -> None:
    """When num_blocks is not a multiple of manager/kernel ratio, zero as-is.

    get_kv_cache_block_dim returns the num_blocks axis (dim 0). SM12x kernel
    pages are 64 and C128 stores 2 tokens, so num_blocks % 4 != 0 is common.
    Skipping left prefix-cache pages dirty. ratio=1 zeros each allocated
    page without virtual splitting.
    """
    path = vllm / "v1/worker/utils.py"
    text = path.read_text()
    ratio1 = (
        "                if kv.shape[block_dim] % ratio != 0:\n"
        "                    print(\n"
        "                        f\"KVBlockZeroer ratio=1 {layer_name} \"\n"
        "                        f\"shape={tuple(kv.shape)} block_dim={block_dim} \"\n"
        "                        f\"was_ratio={ratio} spec_bs={spec.block_size} \"\n"
        "                        f\"kernel_bs={kernel_bs}\",\n"
        "                        flush=True,\n"
        "                    )\n"
        "                    ratio = 1\n"
    )
    if "KVBlockZeroer skip" in text:
        replace_once(
            path,
            "                if kv.shape[block_dim] % ratio != 0:\n"
            "                    print(\n"
            "                        f\"KVBlockZeroer skip {layer_name} \"\n"
            "                        f\"shape={tuple(kv.shape)} block_dim={block_dim} \"\n"
            "                        f\"ratio={ratio} spec_bs={spec.block_size} \"\n"
            "                        f\"kernel_bs={kernel_bs}\",\n"
            "                        flush=True,\n"
            "                    )\n"
            "                    continue\n",
            ratio1,
            "KVBlockZeroer unaligned ratio=1 (from skip)",
        )
    replace_one_of(
        path,
        [
            (
                "                el = kv.element_size()\n"
                "                block_stride_bytes = kv.stride(block_dim) * el\n"
                "                assert block_stride_bytes % 4 == 0\n"
                "                assert kv.shape[block_dim] % ratio == 0\n",
                "                el = kv.element_size()\n"
                "                block_stride_bytes = kv.stride(block_dim) * el\n"
                "                assert block_stride_bytes % 4 == 0\n" + ratio1,
                "KVBlockZeroer unaligned ratio=1",
            ),
            (
                "                el = kv.element_size()\n"
                "                block_stride_bytes = kv.stride(0) * el\n"
                "                assert block_stride_bytes % 4 == 0\n"
                "                assert kv.shape[0] % ratio == 0\n",
                "                el = kv.element_size()\n"
                "                block_stride_bytes = kv.stride(0) * el\n"
                "                assert block_stride_bytes % 4 == 0\n"
                "                if kv.shape[0] % ratio != 0:\n"
                "                    print(\n"
                "                        f\"KVBlockZeroer ratio=1 {layer_name} \"\n"
                "                        f\"shape={tuple(kv.shape)} block_dim=0 \"\n"
                "                        f\"was_ratio={ratio} spec_bs={spec.block_size} \"\n"
                "                        f\"kernel_bs={kernel_bs}\",\n"
                "                        flush=True,\n"
                "                    )\n"
                "                    ratio = 1\n",
                "KVBlockZeroer unaligned ratio=1 (main shape[0])",
            ),
        ],
    )
    replace_one_of(
        path,
        [
            (
                "                        if (idx := seen_ptrs.get(addr)) is not None:\n"
                "                            assert (\n"
                "                                seg_block_strides[idx]\n"
                "                                == logical_block_stride_bytes // 4\n"
                "                            )\n"
                "                            seg_page_sizes[idx] = max(\n"
                "                                seg_page_sizes[idx], kernel_page_bytes // 4\n"
                "                            )\n",
                "                        if (idx := seen_ptrs.get(addr)) is not None:\n"
                "                            # Mixed DSV4 pages (fp8 576 vs nvfp4 584, indexer vs MLA)\n"
                "                            # overlay the same block bytes with different strides.\n"
                "                            seg_block_strides[idx] = max(\n"
                "                                seg_block_strides[idx],\n"
                "                                logical_block_stride_bytes // 4,\n"
                "                            )\n"
                "                            seg_page_sizes[idx] = max(\n"
                "                                seg_page_sizes[idx], kernel_page_bytes // 4\n"
                "                            )\n",
                "KVBlockZeroer overlay mixed-page strides",
            ),
        ],
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


def patch_tp_allreduce_eager_break(vllm: Path) -> None:
    """Run TP all-reduce outside breakable CUDA graphs, in-place.

    PIECEWISE graphs on 2-node GB10 captured host-staged PYNCCL all-reduce
    (RowParallelLinear down_proj). Replay produced 1e33 dummy logits and
    greedy France looped Actors/ligands with Paris at -9999. Break the
    collective out of the graph. Copy into the caller buffer so later
    segments keep a stable address (all-reduce is out-of-place).

    Import of breakable_cudagraph is lazy: a top-level import from
    communication_op.py circular-imports vllm.config.

    NCCL on GB10 cannot use CUDA-graph pool pointers. Replay of a weak-ref
    into the previous GEMM's graph-pool output died with
    `ncclAllReduce` / illegal memory access. Clone to the default allocator,
    reduce, copy back. Strong-ref the caller buffer (weak_ref dangled).
    """
    path = vllm / "distributed/communication_op.py"
    text = path.read_text()
    lazy = (
        "def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:\n"
        "    \"\"\"All-reduce the input tensor across model parallel group.\"\"\"\n"
        "    def _ar(buf: torch.Tensor) -> torch.Tensor:\n"
        "        tmp = buf.detach().clone()\n"
        "        out = get_tp_group().all_reduce(tmp)\n"
        "        buf.copy_(out)\n"
        "        return buf\n"
        "\n"
        "    # Lazy: top-level import of breakable_cudagraph circular-imports\n"
        "    # vllm.config (communication_op is imported from distributed/__init__).\n"
        "    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture\n"
        "    from vllm.config import CUDAGraphMode\n"
        "    from vllm.forward_context import (\n"
        "        get_forward_context,\n"
        "        is_forward_context_available,\n"
        "    )\n"
        "\n"
        "    capture = BreakableCUDAGraphCapture.current()\n"
        "    if capture is None or not capture._capturing:\n"
        "        out = get_tp_group().all_reduce(input_)\n"
        "        if out is not input_:\n"
        "            input_.copy_(out)\n"
        "            return input_\n"
        "        return out\n"
        "    if is_forward_context_available():\n"
        "        mode = get_forward_context().cudagraph_runtime_mode\n"
        "        if mode == CUDAGraphMode.FULL:\n"
        "            return _ar(input_)\n"
        "    if not getattr(tensor_model_parallel_all_reduce, \"_logged_break\", False):\n"
        "        print(\n"
        "            \"TP all-reduce eager-break during capture \"\n"
        "            \"(clone off graph pool)\",\n"
        "            flush=True,\n"
        "        )\n"
        "        tensor_model_parallel_all_reduce._logged_break = True\n"
        "    return capture.add_eager(lambda buf=input_: _ar(buf))\n"
    )
    if "clone off graph pool" in text:
        print("skip TP all-reduce eager-break + in-place: already applied")
        return
    if "return capture.add_eager(lambda: _ar(weak_in))" in text:
        replace_once(
            path,
            "def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:\n"
            "    \"\"\"All-reduce the input tensor across model parallel group.\"\"\"\n"
            "    def _ar(buf: torch.Tensor) -> torch.Tensor:\n"
            "        out = get_tp_group().all_reduce(buf)\n"
            "        if out is not buf:\n"
            "            buf.copy_(out)\n"
            "            return buf\n"
            "        return out\n"
            "\n"
            "    # Lazy: top-level import of breakable_cudagraph circular-imports\n"
            "    # vllm.config (communication_op is imported from distributed/__init__).\n"
            "    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture\n"
            "    from vllm.config import CUDAGraphMode\n"
            "    from vllm.forward_context import (\n"
            "        get_forward_context,\n"
            "        is_forward_context_available,\n"
            "    )\n"
            "    from vllm.utils.torch_utils import weak_ref_tensor\n"
            "\n"
            "    capture = BreakableCUDAGraphCapture.current()\n"
            "    if capture is None or not capture._capturing:\n"
            "        return _ar(input_)\n"
            "    if is_forward_context_available():\n"
            "        mode = get_forward_context().cudagraph_runtime_mode\n"
            "        if mode == CUDAGraphMode.FULL:\n"
            "            return _ar(input_)\n"
            "    if not getattr(tensor_model_parallel_all_reduce, \"_logged_break\", False):\n"
            "        print(\"TP all-reduce eager-break during capture\", flush=True)\n"
            "        tensor_model_parallel_all_reduce._logged_break = True\n"
            "    weak_in = weak_ref_tensor(input_)\n"
            "    return capture.add_eager(lambda: _ar(weak_in))\n",
            lazy,
            "TP all-reduce clone off graph pool",
        )
        return
    if "@eager_break_during_capture\n" in text and "tensor_model_parallel_all_reduce" in text:
        replace_once(
            path,
            "from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n"
            "from .parallel_state import get_tp_group\n"
            "\n"
            "\n"
            "@eager_break_during_capture\n"
            "def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:\n"
            "    \"\"\"All-reduce the input tensor across model parallel group.\"\"\"\n"
            "    out = get_tp_group().all_reduce(input_)\n"
            "    if out is not input_:\n"
            "        input_.copy_(out)\n"
            "        return input_\n"
            "    return out\n",
            "from .parallel_state import get_tp_group\n"
            "\n"
            "\n" + lazy,
            "TP all-reduce eager-break lazy import",
        )
        return
    replace_once(
        path,
        "from .parallel_state import get_tp_group\n"
        "\n"
        "\n"
        "def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:\n"
        "    \"\"\"All-reduce the input tensor across model parallel group.\"\"\"\n"
        "    return get_tp_group().all_reduce(input_)\n",
        "from .parallel_state import get_tp_group\n"
        "\n"
        "\n" + lazy,
        "TP all-reduce eager-break + in-place",
    )


def patch_kv_kernel_split_padded_blhnc(vllm: Path) -> None:
    """Do not token-split compressed DSV4 pages (rc2 storage_block_size).

    0731 compress_ratios are 4 and 128. Manager --block-size 256 already
    stores 64 slots (256/4) or 2 slots (256/128). FlashInfer SM120 DSV4
    page_block_size is 64 slots, so compress-4 matches with no split.

    vLLM main's create_kv_cache_views treats kernel 64 as a 256->64 token
    split (16 slots, BLHNC OOB). rc2 skipped that split whenever
    storage_block_size != block_size. Keep manager 256. France is the gate.
    """
    interface = vllm / "v1/kv_cache_interface.py"
    replace_one_of(
        interface,
        [
            (
                "        dense_page_size = prod(compute_layer_kv_cache_shape_bytes(spec, 1)[1:])\n"
                "        if block_stride != dense_page_size and block_stride % ratio != 0:\n",
                "        dense_page_size = prod(compute_layer_kv_cache_shape_bytes(spec, 1)[1:])\n"
                "        if block_stride != dense_page_size:\n",
                "revert naive BLHNC block_stride split",
            ),
        ],
    )
    replace_one_of(
        interface,
        [
            (
                "    padded_page_size = getattr(spec, \"page_size_padded\", None)\n"
                "    if (\n"
                "        padded_page_size is not None\n"
                "        and kernel_block_size is not None\n"
                "        and kernel_block_size != spec.block_size\n"
                "    ):\n"
                "        # SM12x DSV4: 64-token FlashInfer pages inside 256-token manager blocks.\n"
                "        padded_page_size = None\n"
                "    if padded_page_size is not None:\n"
                "        assert kernel_block_size is None or kernel_block_size == spec.block_size, (\n"
                "            \"Padded KV pages do not support kernel block splitting.\"\n"
                "        )\n",
                "    padded_page_size = getattr(spec, \"page_size_padded\", None)\n"
                "    if padded_page_size is not None:\n"
                "        assert kernel_block_size is None or kernel_block_size == spec.block_size, (\n"
                "            \"Padded KV pages do not support kernel block splitting.\"\n"
                "        )\n",
                "revert padded kernel-split bypass",
            ),
        ],
    )

    alloc = vllm / "v1/worker/utils.py"
    replace_once(
        alloc,
        "        kernel_block_size = None\n"
        "        if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):\n"
        "            kernel_block_size = kernel_block_sizes[group_id]\n"
        "\n"
        "        views = create_kv_cache_views(\n",
        "        kernel_block_size = None\n"
        "        if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):\n"
        "            kernel_block_size = kernel_block_sizes[group_id]\n"
        "        # Compressed DSV4 (0731 compress 4/128): a manager block already\n"
        "        # holds block_size/tokens_per_state slots. compress 4 + 256 is 64\n"
        "        # slots, which is the SM120 FlashInfer page. Token-splitting 256->64\n"
        "        # yields 16 slots, skips the specialized kernel, and OOB on BLHNC.\n"
        "        tps = getattr(spec, \"tokens_per_state\", 1)\n"
        "        if tps is not None and tps > 1:\n"
        "            kernel_block_size = spec.block_size\n"
        "\n"
        "        views = create_kv_cache_views(\n",
        "DSV4 compressed pages skip token kernel-split",
    )


def patch_b12x_mm_block_fp8_compat(vllm: Path) -> None:
    """Git b12x 1.2.6 dropped mm_block_fp8; vLLM main still calls it.

    New oneshot is gemm.blockscaled.mm(..., block_fp8=True) with 3D values
    (M/N, K, 1) and FP32 128x128 scales. Keep the old name when present.
    """
    path = vllm / "model_executor/kernels/linear/scaled_mm/b12x.py"
    replace_once(
        path,
        "    return blockscaled.mm_block_fp8(\n"
        "        a,\n"
        "        a_scale,\n"
        "        weight,\n"
        "        weight_scale,\n"
        "        out_dtype=out_dtype,\n"
        "    )\n",
        "    if hasattr(blockscaled, \"mm_block_fp8\"):\n"
        "        return blockscaled.mm_block_fp8(\n"
        "            a,\n"
        "            a_scale,\n"
        "            weight,\n"
        "            weight_scale,\n"
        "            out_dtype=out_dtype,\n"
        "        )\n"
        "    a_3d = a if a.ndim == 3 else a.unsqueeze(-1)\n"
        "    w_3d = weight if weight.ndim == 3 else weight.unsqueeze(-1)\n"
        "    c_dtype = {\n"
        "        torch.bfloat16: \"bfloat16\",\n"
        "        torch.float16: \"float16\",\n"
        "        torch.float32: \"float32\",\n"
        "    }[out_dtype]\n"
        "    return blockscaled.mm(\n"
        "        (a_3d, a_scale.contiguous()),\n"
        "        (w_3d, weight_scale.contiguous()),\n"
        "        ab_dtype=\"float8_e4m3fn\",\n"
        "        sf_dtype=\"float32\",\n"
        "        c_dtype=c_dtype,\n"
        "        sf_vec_size=128,\n"
        "        block_fp8=True,\n"
        "        expected_m=max(1, int(a.shape[0])),\n"
        "    )\n",
        "B12x mm_block_fp8 -> mm(block_fp8=True)",
    )


def apply(vllm: Path) -> None:
    copy_new_modules(vllm)
    patch_moe_backend(vllm)
    patch_envs(vllm)
    patch_utils_b12x(vllm)
    patch_mxfp4_oracle(vllm)
    patch_mhc(vllm)
    patch_nvfp4_ds_mla(vllm)
    patch_dsv4_nvfp4_attn(vllm)
    patch_dsv4_sm12x_block_size(vllm)
    patch_deep_gemm_sm12x_guard(vllm)
    patch_cutlass_sm12x_guard(vllm)
    patch_indexer_deepgemm_guard(vllm)
    patch_fp8_einsum_fallback(vllm)
    patch_einsum_sm12x_recipe(vllm)
    patch_einsum_sm12x_scale_upcast(vllm)
    patch_mqa_logits_sm12x_fallback(vllm)
    patch_mqa_paged_cudagraph_safe(vllm)
    patch_mqa_relu_formula(vllm)
    patch_mqa_b12x(vllm)
    patch_sm12x_kv_insert(vllm)
    patch_mxfp4_process_weights(vllm)
    patch_flashinfer_dsv4_dispatch(vllm.parent)
    patch_flashinfer_dsv4_cu_dispatch(vllm.parent)
    patch_tp_allreduce_eager_break(vllm)
    print(f"overlays applied under {vllm}")


def _flashinfer_has_topk192(site: Path) -> bool:
    for path in (
        site / "flashinfer/mla/_sparse_mla_sm120.py",
        Path("/opt/flashinfer/flashinfer/mla/_sparse_mla_sm120.py"),
    ):
        if path.is_file() and "(8, 192)" in path.read_text(errors="ignore"):
            return True
    return False


_PAGED_MQA_GATHER_NEEDLE = (
    "    logits = q_flat.new_full((M, max_model_len), float('-inf'), dtype=torch.float32)\n"
    "    if B == 0 or gather_len == 0 or num_blocks == 0:\n"
    "        return logits\n"
    "    pack = _sm12x_b12x_mqa_pack()\n"
)

_PAGED_MQA_PAGED_TRY_OLD = (
    "    logits = q_flat.new_full((M, max_model_len), float('-inf'), dtype=torch.float32)\n"
    "    if B == 0 or gather_len == 0 or num_blocks == 0:\n"
    "        return logits\n"
    "    from vllm.utils.sm12x_b12x_kernels import try_paged_mqa_logits\n"
    "    paged = try_paged_mqa_logits(\n"
    "        q, kv_cache, weights, context_lens, block_tables, max_model_len,\n"
    "    )\n"
    "    if paged is not None:\n"
    "        return paged\n"
    "    pack = _sm12x_b12x_mqa_pack()\n"
)

_PAGED_MQA_PAGED_TRY = (
    "    if B == 0 or gather_len == 0 or num_blocks == 0:\n"
    "        return q_flat.new_full((M, max_model_len), float('-inf'), dtype=torch.float32)\n"
    "    from vllm.utils.sm12x_b12x_kernels import try_paged_mqa_logits\n"
    "    paged = try_paged_mqa_logits(\n"
    "        q, kv_cache, weights, context_lens, block_tables, max_model_len,\n"
    "        schedule_metadata=schedule_metadata,\n"
    "    )\n"
    "    if paged is not None:\n"
    "        return paged\n"
    "    logits = q_flat.new_full((M, max_model_len), float('-inf'), dtype=torch.float32)\n"
    "    pack = _sm12x_b12x_mqa_pack()\n"
)


def patch_mqa_paged_kernel(vllm: Path) -> None:
    """Prefer the b12x page_size=64 paged indexer over gather+contiguous."""
    copy_sm12x_b12x_kernels(vllm)
    path = vllm / "utils/deep_gemm.py"
    text = path.read_text()
    if _PAGED_MQA_PAGED_TRY in text and _PAGED_MQA_GATHER_NEEDLE not in text:
        print("skip fp8_fp4_paged_mqa_logits b12x paged kernel: already applied")
        return
    if _PAGED_MQA_PAGED_TRY_OLD in text and _PAGED_MQA_GATHER_NEEDLE not in text:
        print("skip fp8_fp4_paged_mqa_logits b12x paged kernel: paged try present")
        return
    if _PAGED_MQA_GATHER_NEEDLE not in text:
        raise SystemExit(
            f"fp8_fp4_paged_mqa_logits b12x paged kernel: missing needle in {path}"
        )
    n = text.count(_PAGED_MQA_GATHER_NEEDLE)
    path.write_text(text.replace(_PAGED_MQA_GATHER_NEEDLE, _PAGED_MQA_PAGED_TRY))
    print(f"ok fp8_fp4_paged_mqa_logits b12x paged kernel x{n}")


_PAGED_MQA_PACKED_GATHER_TRY = (
    "    if B == 0 or gather_len == 0 or num_blocks == 0:\n"
    "        return q_flat.new_full((M, max_model_len), float('-inf'), dtype=torch.float32)\n"
    "    from vllm.utils.sm12x_b12x_kernels import (\n"
    "        packed_gather_mqa_logits,\n"
    "        try_paged_mqa_logits,\n"
    "    )\n"
    "    paged = try_paged_mqa_logits(\n"
    "        q, kv_cache, weights, context_lens, block_tables, max_model_len,\n"
    "        schedule_metadata=schedule_metadata,\n"
    "    )\n"
    "    if paged is not None:\n"
    "        return paged\n"
    "    gathered = packed_gather_mqa_logits(\n"
    "        q, kv_cache, weights, context_lens, block_tables, max_model_len,\n"
    "    )\n"
    "    if gathered is not None:\n"
    "        return gathered\n"
    "    logits = q_flat.new_full((M, max_model_len), float('-inf'), dtype=torch.float32)\n"
    "    pack = _sm12x_b12x_mqa_pack()\n"
)


def patch_mqa_packed_gather(vllm: Path) -> None:
    """Add the layout-correct packed gather as a fallback after the paged kernel.

    HANDOFF item 18: gathering packed-at-store indexer K with the interleaved
    FlashInfer gather is numerically wrong (DSpark accept 38-70% vs ~73%).
    packed_gather_mqa_logits reads the packed K-then-scale offsets and is the
    correct fallback when try_paged_mqa_logits returns None.
    """
    copy_sm12x_b12x_kernels(vllm)
    path = vllm / "utils/deep_gemm.py"
    text = path.read_text()
    if _PAGED_MQA_PACKED_GATHER_TRY in text:
        print("skip fp8_fp4_paged_mqa_logits packed gather: already applied")
        return
    if _PAGED_MQA_PAGED_TRY not in text:
        raise SystemExit(
            f"fp8_fp4_paged_mqa_logits packed gather: paged try missing in {path}"
        )
    n = text.count(_PAGED_MQA_PAGED_TRY)
    path.write_text(
        text.replace(_PAGED_MQA_PAGED_TRY, _PAGED_MQA_PACKED_GATHER_TRY)
    )
    print(f"ok fp8_fp4_paged_mqa_logits packed gather x{n}")


def patch_flashinfer_eidx_contig(vllm: Path) -> None:
    """Make extra_sparse_indices contiguous for the FlashInfer SM120 kernel.

    The SM120 sparse MLA C++ checks ``eidx.IsContiguous()``. vLLM's C4A decode
    path passes ``global_indices.view(...)`` (non-contiguous) and the C128A
    path may pass a non-contiguous metadata tensor, so FlashInfer attention
    dies with "eidx must be contiguous" during warmup.

    Root cause (upstream #53574, backported as patches/upstream/pr-53574.diff):
    ``_build_c128a_metadata`` publishes a width-narrowed slice of the
    persistent ``global_decode_buffer``; the view keeps the buffer's row
    stride, so DSpark verification batches (num_decodes*(1+K) > 64 tokens)
    hit the paged-pre fill orchestrator's eidx check and crash. The C4A
    branch is already contiguous (``empty_like`` of the contiguous
    ``topk_indices_buffer`` row slice); the ``.contiguous()`` there is a
    harmless no-op kept for defense-in-depth.
    """
    path = vllm / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    text = path.read_text()
    old_c4a = (
        "extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)"
    )
    new_c4a = old_c4a + ".contiguous()"
    old_c128a = (
        "extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices"
    )
    new_c128a = old_c128a + ".contiguous()"
    if new_c4a in text and new_c128a in text:
        print("skip flashinfer eidx contiguous: already applied")
        return
    if old_c4a not in text and old_c128a not in text:
        raise SystemExit(
            f"flashinfer eidx contiguous: needles missing in {path}"
        )
    text = text.replace(old_c4a, new_c4a)
    text = text.replace(old_c128a, new_c128a)
    path.write_text(text)
    print("ok flashinfer eidx contiguous")


def patch_triton_e8m0_sm12x(vllm: Path) -> None:
    """Upcast E8M0 scales to fp32 for the Triton block-scaled MM on SM12x.

    ``w8a8_triton_block_scaled_mm`` gates the E8M0->fp32 upcast on rocm/xpu
    only; on CUDA family 120 the e8m0 scales reach Triton and crash with
    ``KeyError: 'float8_e8m0fnu'``. b12x linear already upcasts; make Triton
    do the same so ``LINEAR_BACKEND=triton`` boots (measured slower than b12x
    on this pair, but the crash fix is upstreamable).
    """
    path = vllm / "model_executor/layers/quantization/utils/fp8_utils.py"
    text = path.read_text()
    old = "    if current_platform.is_rocm() or current_platform.is_xpu():"
    new = (
        "    if (\n"
        "        current_platform.is_rocm()\n"
        "        or current_platform.is_xpu()\n"
        "        or current_platform.is_device_capability_family(120)\n"
        "    ):"
    )
    if new in text:
        print("skip triton e8m0 sm12x: already applied")
        return
    _upstream_47988_form = (
        "    if As.dtype == torch.float8_e8m0fnu:\n"
        "        As = _upcast_e8m0_to_fp32(As).contiguous()\n"
        "    if Bs.dtype == torch.float8_e8m0fnu:\n"
        "        Bs = _upcast_e8m0_to_fp32(Bs).contiguous()"
    )
    if _upstream_47988_form in text:
        print("skip triton e8m0 sm12x: upstream #47988 unconditional upcast present")
        return
    if old not in text:
        raise SystemExit(f"triton e8m0 sm12x: needle missing in {path}")
    path.write_text(text.replace(old, new, 1))
    print("ok triton e8m0 sm12x")


_INDEXER_B12X_SCHEDULE_OLD = (
    "            schedule_metadata = self.scheduler_metadata_buffer\n"
    "            from vllm.utils.deep_gemm import is_deep_gemm_supported\n"
    "            if (\n"
    "                current_platform.is_cuda()\n"
    "                and is_deep_gemm_supported()\n"
    "                and not current_platform.is_device_capability_family(120)\n"
    "                and self.kv_cache_spec.num_states in (32, 64)\n"
    "            ):\n"
    "                metadata = get_paged_mqa_logits_metadata(\n"
    "                    seq_lens,\n"
    "                    self.kv_cache_spec.num_states,\n"
    "                    self.num_sms,\n"
    "                    indices=decode_indices,\n"
    "                )\n"
    "                schedule_metadata = self.scheduler_metadata_buffer[: metadata.shape[0]]\n"
    "                schedule_metadata[:] = metadata\n"
    "\n"
    "            decode_metadata = DeepSeekV32IndexerDecodeMetadata(\n"
)

_INDEXER_B12X_SCHEDULE_SM120_ALWAYS = (
    "            elif (\n"
    "                current_platform.is_cuda()\n"
    "                and current_platform.is_device_capability_family(120)\n"
    "            ):\n"
    "                try:\n"
    "                    from b12x.attention.nsa_indexer import plan_paged_schedule\n"
    "\n"
    "                    seq = seq_lens\n"
    "                    if seq.dtype != torch.int32:\n"
    "                        seq = seq.to(torch.int32)\n"
    "                    if not seq.is_contiguous():\n"
    "                        seq = seq.contiguous()\n"
    "                    plan_paged_schedule(\n"
    "                        seq,\n"
    "                        64,\n"
    "                        num_sms=self.num_sms,\n"
    "                        out=self.scheduler_metadata_buffer,\n"
    "                    )\n"
    "                    schedule_metadata = self.scheduler_metadata_buffer\n"
    "                except Exception as exc:\n"
    "                    print(\n"
    "                        f\"b12x paged schedule plan skipped: \"\n"
    "                        f\"{type(exc).__name__}: {exc}\",\n"
    "                        flush=True,\n"
    "                    )\n"
    "                    schedule_metadata = self.scheduler_metadata_buffer[:0]\n"
)

_INDEXER_B12X_SCHEDULE_SM120_Q1 = (
    "            elif (\n"
    "                current_platform.is_cuda()\n"
    "                and current_platform.is_device_capability_family(120)\n"
    "            ):\n"
    "                q_rows = (\n"
    "                    int(seq_lens.numel())\n"
    "                    if seq_lens.dim() == 1\n"
    "                    else int(seq_lens.shape[0])\n"
    "                )\n"
    "                if q_rows == 1:\n"
    "                    try:\n"
    "                        from b12x.attention.nsa_indexer import plan_paged_schedule\n"
    "\n"
    "                        seq = seq_lens\n"
    "                        if seq.dtype != torch.int32:\n"
    "                            seq = seq.to(torch.int32)\n"
    "                        if not seq.is_contiguous():\n"
    "                            seq = seq.contiguous()\n"
    "                        plan_paged_schedule(\n"
    "                            seq,\n"
    "                            64,\n"
    "                            num_sms=self.num_sms,\n"
    "                            out=self.scheduler_metadata_buffer,\n"
    "                        )\n"
    "                        schedule_metadata = self.scheduler_metadata_buffer\n"
    "                    except Exception as exc:\n"
    "                        print(\n"
    "                            f\"b12x paged schedule plan skipped: \"\n"
    "                            f\"{type(exc).__name__}: {exc}\",\n"
    "                            flush=True,\n"
    "                        )\n"
    "                        schedule_metadata = self.scheduler_metadata_buffer[:0]\n"
    "                else:\n"
    "                    schedule_metadata = self.scheduler_metadata_buffer[:0]\n"
)

_INDEXER_B12X_SCHEDULE_NEW = (
    "            schedule_metadata = self.scheduler_metadata_buffer\n"
    "            from vllm.utils.deep_gemm import is_deep_gemm_supported\n"
    "            if (\n"
    "                current_platform.is_cuda()\n"
    "                and is_deep_gemm_supported()\n"
    "                and not current_platform.is_device_capability_family(120)\n"
    "                and self.kv_cache_spec.num_states in (32, 64)\n"
    "            ):\n"
    "                metadata = get_paged_mqa_logits_metadata(\n"
    "                    seq_lens,\n"
    "                    self.kv_cache_spec.num_states,\n"
    "                    self.num_sms,\n"
    "                    indices=decode_indices,\n"
    "                )\n"
    "                schedule_metadata = self.scheduler_metadata_buffer[: metadata.shape[0]]\n"
    "                schedule_metadata[:] = metadata\n"
    + _INDEXER_B12X_SCHEDULE_SM120_Q1
    + "\n"
    "            decode_metadata = DeepSeekV32IndexerDecodeMetadata(\n"
)


_INDEXER_PAGE64_BUFFER_OLD = (
    "        self.scheduler_metadata_buffer = torch.empty(\n"
    "            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device\n"
    "        )\n"
    "\n"
    "        # KV compression. Default to 1 for no compression.\n"
)

_INDEXER_PAGE64_BUFFER_NEW = (
    "        self.scheduler_metadata_buffer = torch.empty(\n"
    "            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device\n"
    "        )\n"
    "        self.page64_sub = torch.arange(4, dtype=torch.int32, device=self.device)\n"
    "        self.page64_block_table_buffer = torch.empty(\n"
    "            (scheduler_config.max_num_batched_tokens, block_table_width * 4),\n"
    "            dtype=torch.int32,\n"
    "            device=self.device,\n"
    "        )\n"
    "\n"
    "        # KV compression. Default to 1 for no compression.\n"
)

_INDEXER_PAGE64_BUFFER_SKIP = (
    "        self.scheduler_metadata_buffer = torch.empty(\n"
    "            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device\n"
    "        )\n"
    "        self.page64_sub = torch.arange(4, dtype=torch.int32, device=self.device)\n"
    "        if int(block_table_width) * 64 < int(\n"
    "            self.vllm_config.model_config.max_model_len\n"
    "        ):\n"
    "            self.page64_block_table_buffer = torch.empty(\n"
    "                (scheduler_config.max_num_batched_tokens, block_table_width * 4),\n"
    "                dtype=torch.int32,\n"
    "                device=self.device,\n"
    "            )\n"
    "\n"
    "        # KV compression. Default to 1 for no compression.\n"
)

_INDEXER_PAGE64_EXPAND_OLD = (
    "                    schedule_metadata = self.scheduler_metadata_buffer[:0]\n"
    "\n"
    "            decode_metadata = DeepSeekV32IndexerDecodeMetadata(\n"
)

_INDEXER_PAGE64_EXPAND_NEW = (
    "                    schedule_metadata = self.scheduler_metadata_buffer[:0]\n"
    "\n"
    "            if (\n"
    "                current_platform.is_cuda()\n"
    "                and current_platform.is_device_capability_family(120)\n"
    "                and block_table.numel() > 0\n"
    "                and int(block_table.shape[1]) * 64\n"
    "                < int(self.vllm_config.model_config.max_model_len)\n"
    "            ):\n"
    "                t_rows = int(block_table.shape[0])\n"
    "                t_cols = int(block_table.shape[1])\n"
    "                out = self.page64_block_table_buffer[:t_rows, : t_cols * 4]\n"
    "                out.copy_(\n"
    "                    (\n"
    "                        block_table.to(torch.int32).unsqueeze(-1) * 4\n"
    "                        + self.page64_sub\n"
    "                    ).reshape(t_rows, t_cols * 4)\n"
    "                )\n"
    "                block_table = out\n"
    "\n"
    "            decode_metadata = DeepSeekV32IndexerDecodeMetadata(\n"
)


def patch_indexer_b12x_schedule(vllm: Path) -> None:
    """Fill the CUDA-graph-stable indexer schedule from b12x on SM12x.

    Hopper already writes DeepGEMM metadata into scheduler_metadata_buffer
    every decode step, outside the graph. SM12x skipped that, so the paged
    kernel either planned inside capture (frozen seqlens) or trimmed to
    1023 pages. Plan into the same buffer with page_size 64, and expand
    manager block ids to page64 ids once per step instead of per layer.
    """
    copy_sm12x_b12x_kernels(vllm)
    indexer = vllm / "v1/attention/backends/mla/indexer.py"
    replace_one_of(
        indexer,
        [
            (
                _INDEXER_B12X_SCHEDULE_OLD,
                _INDEXER_B12X_SCHEDULE_NEW,
                "indexer SM12x b12x paged schedule",
            ),
            (
                _INDEXER_B12X_SCHEDULE_SM120_ALWAYS,
                _INDEXER_B12X_SCHEDULE_SM120_Q1,
                "indexer SM12x b12x paged schedule q_rows==1",
            ),
        ],
    )
    replace_one_of(
        indexer,
        [
            (
                _INDEXER_PAGE64_BUFFER_OLD,
                _INDEXER_PAGE64_BUFFER_SKIP,
                "indexer page64 block table buffer",
            ),
            (
                _INDEXER_PAGE64_BUFFER_NEW,
                _INDEXER_PAGE64_BUFFER_SKIP,
                "indexer page64 skip already-64 workspace",
            ),
        ],
    )
    replace_one_of(
        indexer,
        [
            (
                _INDEXER_PAGE64_EXPAND_OLD,
                _INDEXER_PAGE64_EXPAND_NEW,
                "indexer page64 block table expand",
            ),
            (
                "                and int(block_table.shape[1])\n"
                "                == int(self.expanded_block_table_buffer.shape[1])\n",
                "                and int(block_table.shape[1]) * 64\n"
                "                < int(self.vllm_config.model_config.max_model_len)\n",
                "indexer page64 expand skip already-64 tables",
            ),
        ],
    )
    path = vllm / "utils/deep_gemm.py"
    text = path.read_text()
    if _PAGED_MQA_PAGED_TRY in text:
        print("skip fp8_fp4_paged_mqa_logits pass vLLM schedule: already applied")
        return
    if _PAGED_MQA_PAGED_TRY_OLD not in text:
        raise SystemExit(
            f"fp8_fp4_paged_mqa_logits pass vLLM schedule: missing needle in {path}"
        )
    n = text.count(_PAGED_MQA_PAGED_TRY_OLD)
    path.write_text(text.replace(_PAGED_MQA_PAGED_TRY_OLD, _PAGED_MQA_PAGED_TRY))
    print(f"ok fp8_fp4_paged_mqa_logits pass vLLM schedule x{n}")


def patch_indexer_packed_insert(vllm: Path) -> None:
    """Keep a packed index-K sidecar at insert so decode can skip full pack."""
    copy_sm12x_b12x_kernels(vllm)
    path = vllm / "model_executor/layers/sparse_attn_indexer.py"
    if "sync_packed_indexer_k(kv_cache, slot_mapping_for_cache)" in path.read_text():
        print("skip indexer packed insert sidecar: already applied")
    else:
        replace_once(
            path,
            "        ops.indexer_k_quant_and_cache(\n"
            "            k,\n"
            "            kv_cache,\n"
            "            slot_mapping_for_cache,\n"
            "            quant_block_size,\n"
            "            scale_fmt,\n"
            "        )\n",
            "        ops.indexer_k_quant_and_cache(\n"
            "            k,\n"
            "            kv_cache,\n"
            "            slot_mapping_for_cache,\n"
            "            quant_block_size,\n"
            "            scale_fmt,\n"
            "        )\n"
            "        from vllm.utils.sm12x_b12x_kernels import sync_packed_indexer_k\n"
            "\n"
            "        sync_packed_indexer_k(kv_cache, slot_mapping_for_cache)\n",
            "indexer packed insert sidecar",
        )
    compressor = vllm / "models/deepseek_v4/compressor.py"
    replace_once(
        compressor,
        "            token_stride=self._token_stride,\n"
        "            scale_dim=self._scale_dim,\n"
        "            **extra_kwargs,\n"
        "        )\n",
        "            token_stride=self._token_stride,\n"
        "            scale_dim=self._scale_dim,\n"
        "            **extra_kwargs,\n"
        "        )\n"
        "        if self.head_dim == 128:\n"
        "            from vllm.utils.sm12x_b12x_kernels import sync_packed_indexer_k\n"
        "\n"
        "            sync_packed_indexer_k(kv_cache, slot_mapping)\n",
        "compressor packed indexer sidecar",
    )


def patch_indexer_packed_insert_revert(vllm: Path) -> None:
    """Drop the compressor sidecar. It doubled indexer-K memory and OOM'd spark2."""
    compressor = vllm / "models/deepseek_v4/compressor.py"
    replace_once(
        compressor,
        "            token_stride=self._token_stride,\n"
        "            scale_dim=self._scale_dim,\n"
        "            **extra_kwargs,\n"
        "        )\n"
        "        if self.head_dim == 128:\n"
        "            from vllm.utils.sm12x_b12x_kernels import sync_packed_indexer_k\n"
        "\n"
        "            sync_packed_indexer_k(kv_cache, slot_mapping)\n",
        "            token_stride=self._token_stride,\n"
        "            scale_dim=self._scale_dim,\n"
        "            **extra_kwargs,\n"
        "        )\n",
        "compressor packed indexer sidecar revert",
    )


def patch_indexer_store_page64(vllm: Path) -> None:
    """Write indexer K as 64-token packed pages so decode can view, not copy."""
    copy_sm12x_b12x_kernels(vllm)
    path = vllm / "models/deepseek_v4/common/ops/fused_compress_quant_cache.py"
    replace_once(
        path,
        "    Cache block layout:\n"
        "      [0, bs*128):       FP8 data (128 bytes/token)\n"
        "      [bs*128, +bs*4):   float32 scales (4 bytes/token)\n"
        "\n"
        "    For head_dim=128 we have exactly one quant block, so we skip the\n"
        "    [N_QUANT_BLOCKS, QUANT_BLOCK] reshape entirely and use a flat\n"
        "    ``tl.max`` reduction.\n"
        "    \"\"\"\n",
        "    Cache block layout (b12x page_size=64 packed):\n"
        "      four pages per 256-token manager block, each\n"
        "      [0, 64*128):     FP8 data, [64*128, +64*4): float32 scales.\n"
        "\n"
        "    For head_dim=128 we have exactly one quant block, so we skip the\n"
        "    [N_QUANT_BLOCKS, QUANT_BLOCK] reshape entirely and use a flat\n"
        "    ``tl.max`` reduction.\n"
        "    \"\"\"\n",
        "indexer store docstring page64",
    )
    replace_once(
        path,
        "    kv_pos_in_block = kv_slot_idx % kv_cache_block_size\n"
        "\n"
        "    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE\n"
        "    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE\n"
        "    scale_ptr = (\n"
        "        cache_block_ptr\n"
        "        + kv_cache_block_size * TOKEN_STRIDE\n"
        "        + kv_pos_in_block * SCALE_DIM\n"
        "    )\n"
        "\n"
        "    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM\n"
        "    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2\n",
        "    kv_pos_in_block = kv_slot_idx % kv_cache_block_size\n"
        "\n"
        "    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE\n"
        "    PAGE: tl.constexpr = 64\n"
        "    page_in_block = kv_pos_in_block // PAGE\n"
        "    within = kv_pos_in_block % PAGE\n"
        "    packed_page_bytes: tl.constexpr = PAGE * TOKEN_STRIDE + PAGE * SCALE_DIM\n"
        "    page_base = cache_block_ptr + page_in_block.to(tl.int64) * packed_page_bytes\n"
        "    fp8_ptr = page_base + within * TOKEN_STRIDE\n"
        "    scale_ptr = page_base + PAGE * TOKEN_STRIDE + within * SCALE_DIM\n"
        "\n"
        "    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM\n"
        "    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2\n",
        "indexer store pointers page64",
    )


def patch_o_proj_b12x(vllm: Path) -> None:
    """SM12x o_proj: b12x fused inv-RoPE WO GEMM instead of einsum dequant."""
    copy_sm12x_b12x_kernels(vllm)
    path = vllm / "models/deepseek_v4/nvidia/ops/o_proj.py"
    replace_once(
        path,
        "    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /\n"
        "    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.\n"
        "    \"\"\"\n"
        "    o_fp8, o_scale = fused_inv_rope_fp8_quant(\n",
        "    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /\n"
        "    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.\n"
        "    \"\"\"\n"
        "    from vllm.utils.sm12x_b12x_kernels import try_b12x_wo_proj\n"
        "\n"
        "    b12x_out = try_b12x_wo_proj(\n"
        "        o,\n"
        "        positions,\n"
        "        cos_sin_cache,\n"
        "        wo_a,\n"
        "        wo_b,\n"
        "        n_groups=n_groups,\n"
        "        heads_per_group=heads_per_group,\n"
        "        nope_dim=nope_dim,\n"
        "        rope_dim=rope_dim,\n"
        "        o_lora_rank=o_lora_rank,\n"
        "    )\n"
        "    if b12x_out is not None:\n"
        "        return b12x_out\n"
        "    o_fp8, o_scale = fused_inv_rope_fp8_quant(\n",
        "deep_gemm_fp8_o_proj b12x WO projection",
    )


def patch_dspark_backbone_cudagraph(vllm: Path) -> None:
    """Graph the DSpark transformer; keep shared lm_head sample_draft eager."""
    path = vllm / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    replace_once(
        path,
        "        # PIECEWISE cudagraphs are not supported for dflash.\n"
        "        # 2-node GB10: capturing the draft step (shared lm_head GEMM\n"
        "        # + TP all-gather) corrupts lm_head.weight (w_rms -> inf).\n"
        "        logger.info(\n"
        "            \"%s CUDA graphs disabled: draft capture corrupts shared lm_head\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        cudagraph_mode = CUDAGraphMode.NONE\n",
        "        # PIECEWISE cudagraphs are not supported for dflash.\n"
        "        # Graph the draft transformer only. sample_draft uses the shared\n"
        "        # target lm_head; capturing that GEMM + TP all-gather on 2-node\n"
        "        # GB10 sets lm_head.weight to inf.\n"
        "        if wants_full and supports_full:\n"
        "            logger.info(\n"
        "                \"%s CUDA graphs: backbone FULL, lm_head eager\",\n"
        "                self._speculator_name,\n"
        "            )\n"
        "            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY\n"
        "        else:\n"
        "            cudagraph_mode = CUDAGraphMode.NONE\n",
        "DSpark backbone CUDA graphs (lm_head eager)",
    )
    replace_once(
        path,
        "    def capture(self) -> None:\n"
        "        logger.info(\n"
        "            \"Skipping %s CUDA graph capture (shared lm_head)\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        return\n"
        "        logger.info(\"Capturing model for %s speculator...\", self._speculator_name)\n",
        "    def _draft_backbone(\n"
        "        self,\n"
        "        num_reqs: int,\n"
        "        num_tokens_padded: int,\n"
        "        attn_metadata: dict[str, Any] | None,\n"
        "        slot_mappings: dict[str, torch.Tensor] | None,\n"
        "        num_tokens_across_dp: torch.Tensor | None,\n"
        "        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n"
        "    ) -> None:\n"
        "        hs = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        if (\n"
        "            getattr(self, \"_draft_hidden\", None) is None\n"
        "            or self._draft_hidden.shape[1] != hs.shape[1]\n"
        "            or self._draft_hidden.shape[0] < hs.shape[0]\n"
        "        ):\n"
        "            if torch.cuda.is_current_stream_capturing():\n"
        "                raise RuntimeError(\n"
        "                    \"DSpark _draft_hidden missing during CUDA graph capture\"\n"
        "                )\n"
        "            self._draft_hidden = torch.empty(\n"
        "                max(self.max_num_tokens, hs.shape[0]),\n"
        "                hs.shape[1],\n"
        "                dtype=hs.dtype,\n"
        "                device=hs.device,\n"
        "            )\n"
        "        self._draft_hidden[: hs.shape[0]].copy_(hs)\n"
        "\n"
        "    def capture(self) -> None:\n"
        "        logger.info(\n"
        "            \"Capturing %s CUDA graphs (backbone only, lm_head eager)\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        self._draft_hidden = None\n",
        "DSpark capture() backbone into _draft_hidden",
    )
    replace_once(
        path,
        "        last_hidden_states = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        num_sample = num_reqs * self.num_speculative_steps\n",
        "        if cudagraph_runtime_mode == CUDAGraphMode.FULL:\n"
        "            last_hidden_states = self._draft_hidden[:num_tokens_padded]\n"
        "        else:\n"
        "            last_hidden_states = self._run_model(\n"
        "                num_tokens_padded,\n"
        "                attn_metadata,\n"
        "                slot_mappings,\n"
        "                num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode,\n"
        "            )\n"
        "        num_sample = num_reqs * self.num_speculative_steps\n",
        "DSpark _generate_draft sample from _draft_hidden on FULL replay",
    )
    replace_once(
        path,
        "        if batch_desc.cg_mode == CUDAGraphMode.FULL:\n"
        "            assert self.query_cudagraph_manager is not None\n"
        "            self.query_cudagraph_manager.run_fullgraph(batch_desc)\n"
        "        else:\n"
        "            self._generate_draft(\n"
        "                num_reqs,\n"
        "                num_tokens_padded,\n"
        "                draft_attn_metadata,\n"
        "                draft_slot_mappings_by_layer,\n"
        "                num_tokens_across_dp=num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode=batch_desc.cg_mode,\n"
        "            )\n",
        "        if batch_desc.cg_mode == CUDAGraphMode.FULL:\n"
        "            assert self.query_cudagraph_manager is not None\n"
        "            self.query_cudagraph_manager.run_fullgraph(batch_desc)\n"
        "            self._generate_draft(\n"
        "                num_reqs,\n"
        "                num_tokens_padded,\n"
        "                draft_attn_metadata,\n"
        "                draft_slot_mappings_by_layer,\n"
        "                num_tokens_across_dp=num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode=CUDAGraphMode.FULL,\n"
        "            )\n"
        "        else:\n"
        "            self._generate_draft(\n"
        "                num_reqs,\n"
        "                num_tokens_padded,\n"
        "                draft_attn_metadata,\n"
        "                draft_slot_mappings_by_layer,\n"
        "                num_tokens_across_dp=num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode=batch_desc.cg_mode,\n"
        "            )\n",
        "DSpark propose() sample after FULL backbone replay",
    )
    replace_once(
        path,
        "        self.query_cudagraph_manager.capture(\n"
        "            self._generate_draft,\n",
        "        self.query_cudagraph_manager.capture(\n"
        "            self._draft_backbone,\n",
        "DSpark capture target is _draft_backbone",
    )


def patch_dspark_hidden_fix(vllm: Path) -> None:
    """Size DSpark graph output from the draft hidden width, not target hidden."""
    path = vllm / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    if "DSpark _draft_hidden missing during CUDA graph capture" in path.read_text():
        print("skip DSpark hidden fix: already sized from model output")
        return
    replace_once(
        path,
        "        self._draft_hidden = torch.empty(\n"
        "            self.max_num_tokens,\n"
        "            self.hidden_size,\n"
        "            dtype=self.dtype,\n"
        "            device=self.hidden_states.device,\n"
        "        )\n",
        "        self._draft_hidden = None\n",
        "DSpark capture() defer _draft_hidden until first backbone",
    )
    replace_once(
        path,
        "        hs = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        self._draft_hidden[:num_tokens_padded].copy_(hs)\n",
        "        hs = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        if (\n"
        "            getattr(self, \"_draft_hidden\", None) is None\n"
        "            or self._draft_hidden.shape[1] != hs.shape[1]\n"
        "            or self._draft_hidden.shape[0] < hs.shape[0]\n"
        "        ):\n"
        "            if torch.cuda.is_current_stream_capturing():\n"
        "                raise RuntimeError(\n"
        "                    \"DSpark _draft_hidden missing during CUDA graph capture\"\n"
        "                )\n"
        "            self._draft_hidden = torch.empty(\n"
        "                max(self.max_num_tokens, hs.shape[0]),\n"
        "                hs.shape[1],\n"
        "                dtype=hs.dtype,\n"
        "                device=hs.device,\n"
        "            )\n"
        "        self._draft_hidden[: hs.shape[0]].copy_(hs)\n",
        "DSpark _draft_backbone size from model output",
    )


def patch_dspark_disable_graphs(vllm: Path) -> None:
    """Keep DSpark eager. Backbone graphs cut draft accept from ~55% to ~24%."""
    path = vllm / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    replace_once(
        path,
        "        if wants_full and supports_full:\n"
        "            logger.info(\n"
        "                \"%s CUDA graphs: backbone FULL, lm_head eager\",\n"
        "                self._speculator_name,\n"
        "            )\n"
        "            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY\n"
        "        else:\n"
        "            cudagraph_mode = CUDAGraphMode.NONE\n",
        "        logger.info(\n"
        "            \"%s CUDA graphs disabled: draft graphs drop token accept\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        cudagraph_mode = CUDAGraphMode.NONE\n",
        "DSpark disable CUDA graphs (keep draft accept)",
    )
    replace_once(
        path,
        "        logger.info(\n"
        "            \"Capturing %s CUDA graphs (backbone only, lm_head eager)\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        self._draft_hidden = None\n",
        "        logger.info(\n"
        "            \"Skipping %s CUDA graph capture (draft graphs drop accept)\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        return\n"
        "        self._draft_hidden = None\n",
        "DSpark capture() no-op again",
    )


def patch_dspark_backbone_none(vllm: Path) -> None:
    """Graph the DSpark transformer with no nested CUDA graphs.

    Prior backbone capture passed FULL into ``_run_model`` while the outer
    manager was already capturing. Nested graphs plus DSpark's
    ``_generate_draft`` (which ignored ``_draft_hidden`` and ran the backbone
    again) dropped accept from ~55% to ~24%.

    This pass:
    - records transformer ops in the outer graph (``CUDAGraphMode.NONE``)
    - keeps sequential Markov / draft logits eager
    - sizes ``_draft_hidden`` from the draft ``hidden_states`` width
    """
    dflash = vllm / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    dspark = vllm / "v1/worker/gpu/spec_decode/dspark/speculator.py"
    replace_once(
        dflash,
        "        logger.info(\n"
        "            \"%s CUDA graphs disabled: draft graphs drop token accept\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        cudagraph_mode = CUDAGraphMode.NONE\n",
        "        if wants_full and supports_full:\n"
        "            logger.info(\n"
        "                \"%s CUDA graphs: backbone FULL, sample eager\",\n"
        "                self._speculator_name,\n"
        "            )\n"
        "            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY\n"
        "        else:\n"
        "            cudagraph_mode = CUDAGraphMode.NONE\n",
        "DSpark enable backbone FULL graphs",
    )
    replace_once(
        dflash,
        "        hs = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n",
        "        hs = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            CUDAGraphMode.NONE,\n"
        "        )\n",
        "DSpark backbone records ops, no nested graphs",
    )
    replace_once(
        dflash,
        "        logger.info(\n"
        "            \"Skipping %s CUDA graph capture (draft graphs drop accept)\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        return\n"
        "        self._draft_hidden = None\n",
        "        logger.info(\n"
        "            \"Capturing %s CUDA graphs (backbone only, sample eager)\",\n"
        "            self._speculator_name,\n"
        "        )\n"
        "        self._draft_hidden = torch.empty(\n"
        "            self.max_num_tokens,\n"
        "            self.hidden_states.shape[1],\n"
        "            dtype=self.hidden_states.dtype,\n"
        "            device=self.hidden_states.device,\n"
        "        )\n",
        "DSpark capture() backbone into draft-width _draft_hidden",
    )
    replace_once(
        dspark,
        "        # Full draft step (captured under CUDA graph): parallel backbone forward\n"
        "        # then sequential Markov sampling over its hidden state outputs.\n"
        "        head_hidden = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        self._sample_sequential(num_reqs, head_hidden)\n",
        "        # FULL replay already ran _draft_backbone into _draft_hidden.\n"
        "        # Sequential Markov + draft logits stay eager.\n"
        "        if cudagraph_runtime_mode == CUDAGraphMode.FULL:\n"
        "            head_hidden = self._draft_hidden[:num_tokens_padded]\n"
        "        else:\n"
        "            head_hidden = self._run_model(\n"
        "                num_tokens_padded,\n"
        "                attn_metadata,\n"
        "                slot_mappings,\n"
        "                num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode,\n"
        "            )\n"
        "        self._sample_sequential(num_reqs, head_hidden)\n",
        "DSpark _generate_draft samples from _draft_hidden on FULL replay",
    )


def patch_dspark_fullstep_graph(vllm: Path) -> None:
    """Record DSpark sequential sample in the backbone graph.

    DSpark draft logits come from the draft model, not the shared target
    lm_head, so capturing ``_sample_sequential`` does not poison weights.
    The parent FULL path replays ``_draft_backbone`` then calls
    ``_generate_draft``; sampling inside the graph means FULL generate
    must be a no-op or it would sample twice.
    """
    dspark = vllm / "v1/worker/gpu/spec_decode/dspark/speculator.py"
    replace_once(
        dspark,
        "        # FULL replay already ran _draft_backbone into _draft_hidden.\n"
        "        # Sequential Markov + draft logits stay eager.\n"
        "        if cudagraph_runtime_mode == CUDAGraphMode.FULL:\n"
        "            head_hidden = self._draft_hidden[:num_tokens_padded]\n"
        "        else:\n"
        "            head_hidden = self._run_model(\n"
        "                num_tokens_padded,\n"
        "                attn_metadata,\n"
        "                slot_mappings,\n"
        "                num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode,\n"
        "            )\n"
        "        self._sample_sequential(num_reqs, head_hidden)\n",
        "        # FULL replay already ran _draft_backbone, which samples.\n"
        "        if cudagraph_runtime_mode == CUDAGraphMode.FULL:\n"
        "            return\n"
        "        head_hidden = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        self._sample_sequential(num_reqs, head_hidden)\n"
        "\n"
        "    def _draft_backbone(\n"
        "        self,\n"
        "        num_reqs: int,\n"
        "        num_tokens_padded: int,\n"
        "        attn_metadata: dict[str, Any] | None,\n"
        "        slot_mappings: dict[str, torch.Tensor] | None,\n"
        "        num_tokens_across_dp: torch.Tensor | None,\n"
        "        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n"
        "    ) -> None:\n"
        "        super()._draft_backbone(\n"
        "            num_reqs,\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        self._sample_sequential(\n"
        "            num_reqs, self._draft_hidden[:num_tokens_padded]\n"
        "        )\n",
        "DSpark graph sequential sample with backbone",
    )


def patch_dspark_fullstep_revert(vllm: Path) -> None:
    """Undo full-step capture. Graphing sequential sample cut accept and tok/s."""
    dspark = vllm / "v1/worker/gpu/spec_decode/dspark/speculator.py"
    replace_once(
        dspark,
        "        # FULL replay already ran _draft_backbone, which samples.\n"
        "        if cudagraph_runtime_mode == CUDAGraphMode.FULL:\n"
        "            return\n"
        "        head_hidden = self._run_model(\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        self._sample_sequential(num_reqs, head_hidden)\n"
        "\n"
        "    def _draft_backbone(\n"
        "        self,\n"
        "        num_reqs: int,\n"
        "        num_tokens_padded: int,\n"
        "        attn_metadata: dict[str, Any] | None,\n"
        "        slot_mappings: dict[str, torch.Tensor] | None,\n"
        "        num_tokens_across_dp: torch.Tensor | None,\n"
        "        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,\n"
        "    ) -> None:\n"
        "        super()._draft_backbone(\n"
        "            num_reqs,\n"
        "            num_tokens_padded,\n"
        "            attn_metadata,\n"
        "            slot_mappings,\n"
        "            num_tokens_across_dp,\n"
        "            cudagraph_runtime_mode,\n"
        "        )\n"
        "        self._sample_sequential(\n"
        "            num_reqs, self._draft_hidden[:num_tokens_padded]\n"
        "        )\n",
        "        # FULL replay already ran _draft_backbone into _draft_hidden.\n"
        "        # Sequential Markov + draft logits stay eager.\n"
        "        if cudagraph_runtime_mode == CUDAGraphMode.FULL:\n"
        "            head_hidden = self._draft_hidden[:num_tokens_padded]\n"
        "        else:\n"
        "            head_hidden = self._run_model(\n"
        "                num_tokens_padded,\n"
        "                attn_metadata,\n"
        "                slot_mappings,\n"
        "                num_tokens_across_dp,\n"
        "                cudagraph_runtime_mode,\n"
        "            )\n"
        "        self._sample_sequential(num_reqs, head_hidden)\n",
        "DSpark revert full-step graph (sample eager)",
    )


def patch_tp_allreduce_static_workspace(vllm: Path) -> None:
    """NCCL on a default-allocator workspace so FULL graphs do not IMA.

    clone() during capture is served from the CUDA-graph pool. NCCL cannot
    use those pointers. Allocate a persistent workspace outside capture,
    copy in, all-reduce, copy back. PIECEWISE still add_eager as a fallback
    when the workspace is missing (first capture before warmup sized it).
    """
    path = vllm / "distributed/communication_op.py"
    replace_once(
        path,
        "def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:\n"
        "    \"\"\"All-reduce the input tensor across model parallel group.\"\"\"\n"
        "    def _ar(buf: torch.Tensor) -> torch.Tensor:\n"
        "        tmp = buf.detach().clone()\n"
        "        out = get_tp_group().all_reduce(tmp)\n"
        "        buf.copy_(out)\n"
        "        return buf\n"
        "\n"
        "    # Lazy: top-level import of breakable_cudagraph circular-imports\n"
        "    # vllm.config (communication_op is imported from distributed/__init__).\n"
        "    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture\n"
        "    from vllm.config import CUDAGraphMode\n"
        "    from vllm.forward_context import (\n"
        "        get_forward_context,\n"
        "        is_forward_context_available,\n"
        "    )\n"
        "\n"
        "    capture = BreakableCUDAGraphCapture.current()\n"
        "    if capture is None or not capture._capturing:\n"
        "        out = get_tp_group().all_reduce(input_)\n"
        "        if out is not input_:\n"
        "            input_.copy_(out)\n"
        "            return input_\n"
        "        return out\n"
        "    if is_forward_context_available():\n"
        "        mode = get_forward_context().cudagraph_runtime_mode\n"
        "        if mode == CUDAGraphMode.FULL:\n"
        "            return _ar(input_)\n"
        "    if not getattr(tensor_model_parallel_all_reduce, \"_logged_break\", False):\n"
        "        print(\n"
        "            \"TP all-reduce eager-break during capture \"\n"
        "            \"(clone off graph pool)\",\n"
        "            flush=True,\n"
        "        )\n"
        "        tensor_model_parallel_all_reduce._logged_break = True\n"
        "    return capture.add_eager(lambda buf=input_: _ar(buf))\n",
        "def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:\n"
        "    \"\"\"All-reduce the input tensor across model parallel group.\"\"\"\n"
        "    def _workspace(buf: torch.Tensor) -> torch.Tensor:\n"
        "        n = buf.numel()\n"
        "        ws = getattr(tensor_model_parallel_all_reduce, \"_ws\", None)\n"
        "        if (\n"
        "            ws is None\n"
        "            or ws.device != buf.device\n"
        "            or ws.dtype != buf.dtype\n"
        "            or ws.numel() < n\n"
        "        ):\n"
        "            if torch.cuda.is_current_stream_capturing():\n"
        "                return None\n"
        "            tensor_model_parallel_all_reduce._ws = torch.empty(\n"
        "                n, dtype=buf.dtype, device=buf.device\n"
        "            )\n"
        "            ws = tensor_model_parallel_all_reduce._ws\n"
        "        return ws[:n].view_as(buf)\n"
        "\n"
        "    def _ar(buf: torch.Tensor) -> torch.Tensor:\n"
        "        tmp = _workspace(buf)\n"
        "        if tmp is None:\n"
        "            tmp = buf.detach().clone()\n"
        "        else:\n"
        "            tmp.copy_(buf)\n"
        "        out = get_tp_group().all_reduce(tmp)\n"
        "        buf.copy_(out)\n"
        "        return buf\n"
        "\n"
        "    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture\n"
        "    from vllm.config import CUDAGraphMode\n"
        "    from vllm.forward_context import (\n"
        "        get_forward_context,\n"
        "        is_forward_context_available,\n"
        "    )\n"
        "\n"
        "    capture = BreakableCUDAGraphCapture.current()\n"
        "    capturing = capture is not None and capture._capturing\n"
        "    mode = None\n"
        "    if is_forward_context_available():\n"
        "        mode = get_forward_context().cudagraph_runtime_mode\n"
        "    tmp = _workspace(input_)\n"
        "    use_workspace = tmp is not None and (\n"
        "        not capturing or mode == CUDAGraphMode.FULL\n"
        "    )\n"
        "    if use_workspace:\n"
        "        tmp.copy_(input_)\n"
        "        out = get_tp_group().all_reduce(tmp)\n"
        "        input_.copy_(out)\n"
        "        if not getattr(tensor_model_parallel_all_reduce, \"_logged_ws\", False):\n"
        "            print(\n"
        "                \"TP all-reduce static workspace \"\n"
        "                \"(default allocator, graph-safe)\",\n"
        "                flush=True,\n"
        "            )\n"
        "            tensor_model_parallel_all_reduce._logged_ws = True\n"
        "        return input_\n"
        "    if not capturing:\n"
        "        out = get_tp_group().all_reduce(input_)\n"
        "        if out is not input_:\n"
        "            input_.copy_(out)\n"
        "            return input_\n"
        "        return out\n"
        "    if not getattr(tensor_model_parallel_all_reduce, \"_logged_break\", False):\n"
        "        print(\n"
        "            \"TP all-reduce eager-break during capture \"\n"
        "            \"(clone off graph pool)\",\n"
        "            flush=True,\n"
        "        )\n"
        "        tensor_model_parallel_all_reduce._logged_break = True\n"
        "    return capture.add_eager(lambda buf=input_: _ar(buf))\n",
        "TP all-reduce static default-allocator workspace",
    )


def patch_tp_allreduce_piecewise_workspace(vllm: Path) -> None:
    """Use the default-allocator AR workspace inside PIECEWISE capture too.

    NCCL cannot use CUDA-graph-pool pointers. The static workspace is from
    the default allocator, so PIECEWISE can stay in-graph the same way FULL
    already does. Size the workspace for 8192 tokens on first eager alloc.
    """
    path = vllm / "distributed/communication_op.py"
    src = path.read_text()
    if "use_workspace = tmp is not None\n    if use_workspace:" in src:
        print("skip TP all-reduce piecewise workspace: already applied")
        return
    replace_once(
        path,
        "            tensor_model_parallel_all_reduce._ws = torch.empty(\n"
        "                n, dtype=buf.dtype, device=buf.device\n"
        "            )\n",
        "            cap = max(n, 8192 * int(buf.shape[-1]) if buf.ndim > 0 else n)\n"
        "            tensor_model_parallel_all_reduce._ws = torch.empty(\n"
        "                cap, dtype=buf.dtype, device=buf.device\n"
        "            )\n",
        "TP all-reduce workspace sized for prefill",
    )
    replace_once(
        path,
        "    use_workspace = tmp is not None and (\n"
        "        not capturing or mode == CUDAGraphMode.FULL\n"
        "    )\n",
        "    use_workspace = tmp is not None\n",
        "TP all-reduce in-graph workspace for PIECEWISE",
    )


def apply_main(vllm: Path) -> None:
    """Keep/add SM12x overlays for a matched vLLM main tree (docs/PLAN-MAIN.md).

    Do not copy rc2 B12xExperts. Do not blanket-kill DeepGEMM on family 120.
    Skip FlashInfer TOPK 192 if git main already has it. Skip lm_head restore.
    """
    moe = vllm / "model_executor/layers/fused_moe/b12x.py"
    if not moe.is_file():
        raise SystemExit(
            f"main tree missing {moe}; refusing to copy patches/files B12xExperts"
        )
    print(f"ok main already has {moe.relative_to(vllm)}")
    patch_mhc(vllm)
    patch_nvfp4_ds_mla(vllm)
    patch_dsv4_nvfp4_attn(vllm)
    patch_dsv4_sm12x_block_size(vllm)
    patch_cutlass_sm12x_guard(vllm)
    patch_indexer_deepgemm_guard(vllm)
    patch_fp8_einsum_fallback(vllm)
    patch_einsum_sm12x_recipe(vllm)
    patch_einsum_sm12x_scale_upcast(vllm)
    patch_mqa_logits_sm12x_fallback(vllm)
    patch_mqa_paged_cudagraph_safe(vllm)
    patch_mqa_relu_formula(vllm)
    patch_mqa_b12x(vllm)
    patch_sm12x_kv_insert(vllm)
    site = vllm.parent
    if _flashinfer_has_topk192(site):
        print("skip flashinfer DSV4 192: already present")
    else:
        patch_flashinfer_dsv4_dispatch(site)
        patch_flashinfer_dsv4_cu_dispatch(site)
    patch_tp_allreduce_eager_break(vllm)
    patch_dspark_skip_cudagraph(vllm)
    patch_instanttensor_hybrid_draft(vllm)
    patch_b12x_mm_block_fp8_compat(vllm)
    patch_kv_kernel_split_padded_blhnc(vllm)
    patch_kv_zeroer_skip(vllm)
    patch_dsv4_b12x_sparse_backend(vllm)
    copy_sm12x_b12x_kernels(vllm)
    patch_mqa_paged_kernel(vllm)
    patch_mqa_packed_gather(vllm)
    patch_flashinfer_eidx_contig(vllm)
    patch_triton_e8m0_sm12x(vllm)
    patch_indexer_packed_insert(vllm)
    patch_indexer_store_page64(vllm)
    patch_o_proj_b12x(vllm)
    patch_indexer_b12x_schedule(vllm)
    patch_dspark_backbone_cudagraph(vllm)
    patch_tp_allreduce_static_workspace(vllm)
    print(f"main overlays applied under {vllm}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stack",
        choices=["rc2", "main"],
        default="rc2",
        help="rc2: overlay image cherry-picks. main: PLAN-MAIN keep/add only.",
    )
    parser.add_argument(
        "--vllm-dir",
        type=Path,
        default=None,
        help="Installed vllm package dir. Default: import vllm.",
    )
    parser.add_argument(
        "--only",
        choices=[
            "mqa-paged-graph",
            "mqa-relu",
            "dsv4-nvfp4",
            "dsv4-block64",
            "einsum-sm12x",
            "mqa-b12x",
            "sm12x-kv-insert",
            "logit-dump",
            "kv-zeroer",
            "dspark-skip-cg",
            "lm-head-restore",
            "ar-eager-break",
            "instanttensor-hybrid",
            "b12x-mm-block-fp8",
            "kv-kernel-split",
            "indexer-mqa",
            "b12x-sparse",
            "mqa-paged-kernel",
            "indexer-packed-insert",
            "indexer-packed-insert-revert",
            "mqa-packed-gather",
            "flashinfer-eidx-contig",
            "triton-e8m0-sm12x",
            "indexer-store-page64",
            "indexer-b12x-schedule",
            "o-proj-b12x",
            "dspark-backbone-cg",
            "dspark-hidden-fix",
            "dspark-no-cg",
            "dspark-backbone-none",
            "dspark-fullstep-cg",
            "dspark-fullstep-revert",
            "ar-static-ws",
            "ar-piecewise-ws",
        ],
        default=None,
        help="Apply a single overlay (for patching an already-built image).",
    )
    args = parser.parse_args()
    vllm = args.vllm_dir.resolve() if args.vllm_dir else _vllm_dir()
    if not (vllm / "config/kernel.py").is_file():
        raise SystemExit(f"not a vllm package: {vllm}")
    if args.only == "mqa-paged-graph":
        patch_mqa_paged_cudagraph_safe(vllm)
        return 0
    if args.only == "mqa-relu":
        patch_mqa_relu_formula(vllm)
        return 0
    if args.only == "dsv4-nvfp4":
        patch_dsv4_nvfp4_attn(vllm)
        return 0
    if args.only == "dsv4-block64":
        patch_dsv4_sm12x_block_size(vllm)
        return 0
    if args.only == "einsum-sm12x":
        patch_einsum_sm12x_recipe(vllm)
        patch_einsum_sm12x_scale_upcast(vllm)
        return 0
    if args.only == "mqa-b12x":
        patch_mqa_b12x(vllm)
        return 0
    if args.only == "sm12x-kv-insert":
        patch_sm12x_kv_insert(vllm)
        return 0
    if args.only == "logit-dump":
        patch_logit_dump(vllm)
        return 0
    if args.only == "kv-zeroer":
        patch_kv_zeroer_skip(vllm)
        return 0
    if args.only == "dspark-skip-cg":
        patch_dspark_skip_cudagraph(vllm)
        return 0
    if args.only == "lm-head-restore":
        patch_lm_head_restore_after_graphs(vllm)
        return 0
    if args.only == "ar-eager-break":
        patch_tp_allreduce_eager_break(vllm)
        return 0
    if args.only == "instanttensor-hybrid":
        patch_instanttensor_hybrid_draft(vllm)
        return 0
    if args.only == "b12x-mm-block-fp8":
        patch_b12x_mm_block_fp8_compat(vllm)
        return 0
    if args.only == "kv-kernel-split":
        patch_kv_kernel_split_padded_blhnc(vllm)
        patch_kv_zeroer_skip(vllm)
        return 0
    if args.only == "indexer-mqa":
        patch_indexer_deepgemm_guard(vllm)
        return 0
    if args.only == "b12x-sparse":
        patch_dsv4_b12x_sparse_backend(vllm)
        return 0
    if args.only == "mqa-paged-kernel":
        patch_mqa_paged_kernel(vllm)
        return 0
    if args.only == "mqa-packed-gather":
        patch_mqa_packed_gather(vllm)
        return 0
    if args.only == "flashinfer-eidx-contig":
        patch_flashinfer_eidx_contig(vllm)
        return 0
    if args.only == "triton-e8m0-sm12x":
        patch_triton_e8m0_sm12x(vllm)
        return 0
    if args.only == "indexer-packed-insert":
        patch_indexer_packed_insert(vllm)
        return 0
    if args.only == "indexer-packed-insert-revert":
        patch_indexer_packed_insert_revert(vllm)
        return 0
    if args.only == "indexer-store-page64":
        patch_indexer_store_page64(vllm)
        return 0
    if args.only == "indexer-b12x-schedule":
        patch_indexer_b12x_schedule(vllm)
        return 0
    if args.only == "o-proj-b12x":
        patch_o_proj_b12x(vllm)
        return 0
    if args.only == "dspark-backbone-cg":
        patch_dspark_backbone_cudagraph(vllm)
        return 0
    if args.only == "dspark-hidden-fix":
        patch_dspark_hidden_fix(vllm)
        return 0
    if args.only == "dspark-no-cg":
        patch_dspark_disable_graphs(vllm)
        return 0
    if args.only == "dspark-backbone-none":
        patch_dspark_backbone_none(vllm)
        return 0
    if args.only == "dspark-fullstep-cg":
        patch_dspark_fullstep_graph(vllm)
        return 0
    if args.only == "dspark-fullstep-revert":
        patch_dspark_fullstep_revert(vllm)
        return 0
    if args.only == "ar-static-ws":
        patch_tp_allreduce_static_workspace(vllm)
        return 0
    if args.only == "ar-piecewise-ws":
        patch_tp_allreduce_piecewise_workspace(vllm)
        return 0
    if args.stack == "main":
        apply_main(vllm)
        return 0
    apply(vllm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
