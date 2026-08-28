#!/usr/bin/env python3
"""Validate the fixed contract for an automated research report."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIRED_HEADINGS = (
    "## 메타데이터",
    "## 결론",
    "## 확인된 사실",
    "## 커뮤니티 주장",
    "## 충돌·미확인 내용",
    "## 책 반영 제안",
    "## 출처 목록",
    "## 보류 사유 및 다음 작업",
)


def fail(message: str) -> None:
    raise SystemExit(f"research report contract failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--research-date", required=True)
    args = parser.parse_args()

    expected_name = f"research-issue-{args.issue_number}_{args.research_date}.md"
    if args.report.name != expected_name:
        fail(f"expected filename {expected_name}, got {args.report.name}")
    if not args.report.is_file():
        fail(f"missing report: {args.report}")

    text = args.report.read_text(encoding="utf-8")
    expected_title = (
        f"# DGX Spark 리서치 기록 — Issue #{args.issue_number} — {args.research_date}"
    )
    if not text.startswith(expected_title + "\n"):
        fail("title, issue number, or research date is missing")

    previous = -1
    for heading in REQUIRED_HEADINGS:
        position = text.find(heading)
        if position < 0:
            fail(f"missing heading: {heading}")
        if position <= previous:
            fail(f"headings are out of order: {heading}")
        previous = position

    required_metadata = (
        f"- 원본 Issue: [Issue #{args.issue_number}](",
        f"- 분석 기준일: `{args.research_date}`",
        "- 분석 실행기: `GitHub Copilot CLI`",
        "- 요청 모델: `",
        "- 현재 상태: `",
        "- 본문 승격: `",
    )
    for fragment in required_metadata:
        if fragment not in text:
            fail(f"missing metadata: {fragment}")

    if "{{" in text or "}}" in text:
        fail("unresolved template placeholder remains")

    empty_fields = (
        "종합 판정",
        "승격 가능한 항목",
        "아직 확정하지 않는 항목",
    )
    for field in empty_fields:
        if re.search(rf"^- {re.escape(field)}:\s*$", text, flags=re.MULTILINE):
            fail(f"empty conclusion field: {field}")

    source_start = text.index("## 출처 목록") + len("## 출처 목록")
    source_end = text.index("## 보류 사유 및 다음 작업", source_start)
    source_section = text[source_start:source_end]
    if not re.search(r"https?://", source_section):
        fail("source list must contain at least one source URL")
    if len(re.findall(r"https?://", text)) < 2:
        fail("report must contain the Issue URL and at least one source URL")

    print(f"Research report OK: {args.report}")


if __name__ == "__main__":
    main()
