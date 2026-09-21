#!/usr/bin/env python3
"""Per-kernel SASS fingerprint of one or more probe binaries.

Prints, for each kernel in each binary, the instruction count and a hash of the
disassembled text. Identical hashes across arches mean the compiler emitted the
same code for that kernel, so the arch string cannot matter for it.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)", re.M)


def sass_blocks(binary: Path) -> dict[str, list[str]]:
    out = subprocess.run(
        ["cuobjdump", "-sass", str(binary)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    blocks: dict[str, list[str]] = {}
    name: str | None = None
    for line in out.splitlines():
        m = FUNC.match(line)
        if m:
            name = m.group(1)
            blocks.setdefault(name, [])
            continue
        if name is not None:
            blocks[name].append(line)
    return blocks


def short(name: str) -> str:
    # _Z9copy_f4PK6float4PS_ -> copy_f4
    m = re.match(r"_Z\d+([A-Za-z_][A-Za-z0-9_]*)", name)
    return m.group(1) if m else name


INS = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+(.*?);")


def fingerprint(body: list[str]) -> tuple[int, str, str, int]:
    """(instruction count, mnemonic hash, encoding hash, ffma count).

    The mnemonic hash ignores the trailing `/* 0x... */` encoding, so a hit
    there means the compiler chose the same instructions; the encoding hash
    catches the case where only the encoding differs.
    """
    ops: list[str] = []
    full: list[str] = []
    for line in body:
        m = INS.match(line)
        if m:
            ops.append(m.group(1).strip())
            full.append(line.strip())
    return (
        len(ops),
        hashlib.sha256("\n".join(ops).encode()).hexdigest()[:10],
        hashlib.sha256("\n".join(full).encode()).hexdigest()[:10],
        sum(1 for o in ops if o.startswith(("FFMA", "HFMA"))),
    )


def main() -> int:
    binaries = [Path(p) for p in sys.argv[1:]]
    per_binary: dict[str, dict[str, tuple[int, str, str, int]]] = {}
    for b in binaries:
        per_binary[b.name] = {
            short(n): fingerprint(body) for n, body in sass_blocks(b).items()
        }
    kernels = sorted({k for d in per_binary.values() for k in d})
    for k in kernels:
        print(f"kernel {k}")
        for b in binaries:
            n, ops_hash, enc_hash, fma = per_binary[b.name].get(k, (0, "-", "-", 0))
            print(f"  {b.name:<16} insns={n:<5} ops={ops_hash} enc={enc_hash} ffma={fma}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
