#!/usr/bin/env python3
"""Validate a reviewed report/chapter pair and build a fixed Copilot prompt."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path, PurePosixPath


REPORT_PATTERN = re.compile(
    r"^docs/research-issue-([1-9][0-9]*)_(\d{4}-\d{2}-\d{2})\.md$"
)
BOOK_PATTERN = re.compile(r"^book/([A-Za-z0-9][A-Za-z0-9._-]*\.md)$")
RESERVED_BOOK_FILES = {"README.md", "TOC.md"}


def _safe_repo_file(root: Path, value: str, pattern: re.Pattern[str]) -> Path:
    if not value or "\\" in value:
        raise ValueError(f"invalid repository path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pattern.fullmatch(value):
        raise ValueError(f"invalid repository path: {value!r}")

    root = root.resolve()
    path = (root / pure).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"repository file does not exist: {value}")
    return path


def validate_inputs(root: Path, report: str, book_file: str) -> tuple[Path, Path]:
    report_path = _safe_repo_file(root, report, REPORT_PATTERN)
    match = REPORT_PATTERN.fullmatch(report)
    assert match is not None
    try:
        dt.date.fromisoformat(match.group(2))
    except ValueError as exc:
        raise ValueError(f"invalid research report date: {match.group(2)}") from exc

    book_path = _safe_repo_file(root, book_file, BOOK_PATTERN)
    if book_path.name in RESERVED_BOOK_FILES:
        raise ValueError(f"reserved book file cannot be drafted: {book_file}")
    return report_path, book_path


def build_prompt(root: Path, report: str, book_file: str) -> str:
    validate_inputs(root, report, book_file)
    base = (root / "prompts/book-draft.md").read_text(encoding="utf-8")
    style = (root / "prompts/fluent-korean.md").read_text(encoding="utf-8")
    return (
        base
        + "\n\n## 공통 한국어 문체\n\n"
        + style
        + "\n\n## 이번 작업의 입력\n\n"
        + f"- 검토된 리서치 보고서: `{report}`\n"
        + f"- 수정할 책 파일: `{book_file}`\n"
        + "\n두 파일을 끝까지 읽은 뒤, 근거가 충분한 내용만 책 파일에 반영한다. "
        + "다른 파일은 수정하지 않는다.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", required=True)
    parser.add_argument("--book-file", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        prompt = build_prompt(args.root, args.report, args.book_file)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.write_text(prompt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
