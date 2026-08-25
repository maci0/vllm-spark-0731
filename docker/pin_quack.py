#!/usr/bin/env python3
"""Rewrite quack-kernels CUTLASS pins to 4.7.0 and install --no-deps."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from pin_cutlass_dsl import pin_text


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} VERSION ARTIFACT_DIR")
    version, artifact_dir = sys.argv[1], Path(sys.argv[2])
    src = next(artifact_dir.glob("quack*"))
    tmp = Path("/tmp/quack-unpacked")
    tmp.mkdir(parents=True, exist_ok=True)
    if src.name.endswith(".whl"):
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        metas = list(tmp.glob("*.dist-info/METADATA")) + list(
            tmp.glob("*.dist-info/requires.txt")
        )
        kind = "wheel"
    elif src.name.endswith(".tar.gz"):
        with tarfile.open(src) as t:
            t.extractall(tmp)
        metas = (
            list(tmp.rglob("METADATA"))
            + list(tmp.rglob("requires.txt"))
            + list(tmp.rglob("pyproject.toml"))
        )
        kind = "sdist"
    else:
        raise SystemExit(f"unexpected quack artifact {src}")

    total = 0
    for meta in metas:
        text, n = pin_text(meta.read_text(), version)
        total += n
        meta.write_text(text)
    if total != 2:
        raise SystemExit(f"quack rewrite expected 2, got {total} in {metas}")

    if kind == "wheel":
        out = Path("/tmp") / src.name
        if out.resolve() == src.resolve():
            out = Path("/tmp/quack-pinned") / src.name
            out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w") as z:
            for path in tmp.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(tmp))
        subprocess.check_call(
            ["uv", "pip", "install", "--reinstall", "--no-deps", str(out)]
        )
    else:
        root = next(path for path in tmp.iterdir() if path.is_dir())
        subprocess.check_call(
            ["uv", "pip", "install", "--reinstall", "--no-deps", str(root)]
        )
    print(f"Pinned quack-kernels CUTLASS DSL to {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
