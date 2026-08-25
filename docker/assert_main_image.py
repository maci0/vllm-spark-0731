#!/usr/bin/env python3
"""Phase 1 gate for vllm-spark-0731:main-b12x. No overlays, no France."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def nvcc_version() -> str:
    out = subprocess.check_output(["nvcc", "--version"], text=True)
    for line in out.splitlines():
        if "release" in line.lower():
            return line.strip()
    return out.strip()


def find_dsv4_dispatch() -> tuple[Path, str]:
    roots: list[Path] = []
    try:
        import flashinfer

        roots.append(Path(flashinfer.__file__).resolve().parent)
    except Exception:
        pass
    for extra in (
        Path("/opt/flashinfer"),
        Path("/workspace/flashinfer"),
    ):
        if extra.exists():
            roots.append(extra)
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".cu", ".h", ".cuh"}:
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if "_DECODE_DSV4_DISPATCH" in text or "DSV4_DISPATCH" in text:
                if "(8, 192)" in text or "(8,192)" in text or "DSV4_DISPATCH(8, 192)" in text:
                    return path, text
    fail("FlashInfer DSV4 dispatch with (8,192) not found")
    raise AssertionError("unreachable")


def main() -> int:
    nvcc = nvcc_version()
    if "13.3" not in nvcc:
        fail(f"nvcc is not 13.3.x: {nvcc}")
    ok(nvcc)

    try:
        import torch
    except Exception as exc:
        fail(f"import torch: {exc}")

    ver = torch.__version__
    cuda = torch.version.cuda or ""
    if not ver.startswith("2.14"):
        fail(f"torch {ver} does not start with 2.14")
    if not cuda.startswith("13.3"):
        fail(f"torch.version.cuda={cuda} does not start with 13.3")
    ok(f"torch {ver} cuda {cuda} arch {torch.cuda.get_arch_list()}")

    cutlass = metadata.version("nvidia-cutlass-dsl")
    if cutlass != "4.7.0":
        fail(f"nvidia-cutlass-dsl {cutlass} != 4.7.0")
    ok(f"nvidia-cutlass-dsl {cutlass}")

    nvdisasm = metadata.version("nvidia-cuda-nvdisasm")
    if not nvdisasm.startswith("13.3"):
        fail(f"nvidia-cuda-nvdisasm {nvdisasm} is not 13.3.x")
    ok(f"nvidia-cuda-nvdisasm {nvdisasm}")

    for pkg in ("triton", "tilelang", "humming-kernels", "apache-tvm-ffi", "tokenspeed-mla"):
        ok(f"{pkg} {metadata.version(pkg)}")
    importlib.import_module("tilelang")
    ok("import tilelang")

    path, _text = find_dsv4_dispatch()
    ok(f"FlashInfer DSV4 (8,192) in {path}")

    for mod in ("b12x", "flashinfer", "vllm"):
        importlib.import_module(mod)
        ok(f"import {mod}")

    instant = None
    for name in ("instanttensor", "InstantTensor"):
        try:
            instant = importlib.import_module(name)
            break
        except Exception:
            continue
    if instant is None:
        fail("import instanttensor")
    ok(f"import {instant.__name__}")

    importlib.import_module("fastsafetensors")
    ok("import fastsafetensors")
    importlib.import_module("lmcache")
    ok("import lmcache")

    sha_dir = Path(os.environ.get("SPARK_SHA_DIR", "/opt/spark-0731/sha"))
    required = (
        "pytorch",
        "nccl",
        "vllm",
        "deepgemm",
        "b12x",
        "flashinfer",
        "instanttensor",
        "fastsafetensors",
        "lmcache",
        "torchvision",
    )
    for name in required:
        p = sha_dir / name
        if not p.is_file() or not p.read_text().strip():
            fail(f"missing SHA file {p}")
        ok(f"sha {name}={p.read_text().strip()}")

    print("assert_main_image ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
