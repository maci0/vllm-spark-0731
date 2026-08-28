#!/usr/bin/env python3
"""Redact a tool output file or print a safe diagnostic tail."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


TOKEN_ENV_NAMES = (
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "SEARCH_API_KEY",
)

PATTERNS = (
    (
        re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._~+/=-]+"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"\b(?:github_pat_|ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]+"),
        "[REDACTED]",
    ),
    (
        re.compile(r'''(?i)([?&](?:api[_-]?key|token|access[_-]?token)=)[^&\s"'<>]+'''),
        r"\1[REDACTED]",
    ),
)


def redact(text: str) -> str:
    for name in TOKEN_ENV_NAMES:
        value = os.environ.get(name, "")
        if value:
            text = text.replace(value, "[REDACTED]")
    for pattern, replacement in PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="replace the input file with the complete redacted text",
    )
    args = parser.parse_args()
    if args.max_chars < 1:
        parser.error("--max-chars must be positive")

    text = redact(args.input.read_text(encoding="utf-8", errors="replace"))
    if args.in_place:
        args.input.write_text(text, encoding="utf-8")
    else:
        print(text[-args.max_chars :])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
