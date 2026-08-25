#!/usr/bin/env python3
"""Rewrite exact nvidia-cutlass-dsl pins in b12x or quack-kernels metadata.

PyPI b12x==1.2.6 and git master still require five ==4.6.2 lines
(--expected-count 5). quack-kernels==0.6.4 still requires two
(--expected-count 2). There is no PyPI prerelease that accepts 4.7.0.
Same rewrite eugr uses before `uv pip install --no-deps`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIREMENT = re.compile(
    r"(?m)^(?P<prefix>(?:Requires-Dist:\s+|\s*[\"']?))"
    r"(?P<name>nvidia-cutlass-dsl(?:-libs-(?:base|core|cu12|cu13))?)"
    r"(?P<extra>\[cu13\])?"
    r"(?P<before>\s*)==(?P<after>\s*)"
    r"(?P<version>[^\s,\"';]+)"
)


def pin_text(text: str, version: str) -> tuple[str, int]:
    def replace(match: re.Match[str]) -> str:
        return (
            f"{match.group('prefix')}{match.group('name')}"
            f"{match.group('extra') or ''}{match.group('before')}=="
            f"{match.group('after')}{version}"
        )

    return REQUIREMENT.subn(replace, text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", args.version):
        parser.error(f"invalid CUTLASS DSL version: {args.version}")

    updates: list[tuple[Path, str]] = []
    total = 0
    for path in args.paths:
        original = path.read_text()
        updated, count = pin_text(original, args.version)
        total += count
        updates.append((path, updated))

    if total != args.expected_count:
        raise SystemExit(
            "CUTLASS DSL pin failed: expected "
            f"{args.expected_count} requirements, found {total}"
        )

    for path, updated in updates:
        path.write_text(updated)
    print(f"Pinned {total} CUTLASS DSL requirements to {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
