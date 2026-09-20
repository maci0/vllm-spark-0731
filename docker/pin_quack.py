#!/usr/bin/env python3
"""Rewrite quack-kernels CUTLASS pins to 4.7.0 and install --no-deps.

quack-kernels 0.6.4 declared two exact pins (``nvidia-cutlass-dsl==4.6.2`` and its
``[cu13]`` extra), so both had to be rewritten to the version we install. 0.6.5 declares
``nvidia-cutlass-dsl>=4.7`` instead, which 4.7.0 already satisfies and which
``pin_text`` therefore rewrites zero times. The old ``total != 2`` guard read that as a
failure. Verify the requirement set instead of counting rewrites: a requirement that the
target version does not satisfy still fails loudly, whether or not it was rewritten.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement

from pin_cutlass_dsl import pin_text


def unsatisfied(text: str, version: str) -> list[str]:
    """nvidia-cutlass-dsl requirements in ``text`` that ``version`` does not satisfy."""
    bad = []
    for line in text.splitlines():
        line = line.strip()
        if "nvidia-cutlass-dsl" not in line:
            continue
        if line.startswith("Requires-Dist:"):
            line = line[len("Requires-Dist:"):].strip()
        try:
            req = Requirement(line)
        except Exception:  # noqa: BLE001 - non-requirement metadata line
            continue
        if not req.name.startswith("nvidia-cutlass-dsl"):
            continue
        if not req.specifier.contains(version, prereleases=True):
            bad.append(line)
    return bad


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

    if total not in (0, 2):
        raise SystemExit(f"quack rewrite expected 0 or 2, got {total} in {metas}")

    bad = [f"{meta.name}: {req}" for meta in metas for req in unsatisfied(meta.read_text(), version)]
    if bad:
        raise SystemExit(
            f"quack CUTLASS DSL {version} does not satisfy: " + "; ".join(bad)
        )
    if total == 0:
        print(f"{src.name} already requires CUTLASS DSL {version}; no rewrite needed")

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
