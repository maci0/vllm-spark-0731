#!/usr/bin/env python3
"""Restore the deterministic title and metadata of a Copilot research report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from research_candidate_count import candidate_count


def build_metadata(issue: dict, research_date: str, model: str) -> str:
    issue_number = str(issue["number"])
    issue_url = str(issue["url"])
    return f"""# DGX Spark 리서치 기록 — Issue #{issue_number} — {research_date}

## 메타데이터

- 원본 Issue: [Issue #{issue_number}]({issue_url})
- 분석 기준일: `{research_date}`
- 수집 후보 수(이슈 원문 목록 기준): `{candidate_count(issue)}`
- 분석 실행기: `GitHub Copilot CLI`
- 요청 모델: `{model}`
- 현재 상태: `분석`
- 본문 승격: `승격 대기`

"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--issue-json", required=True, type=Path)
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--research-date", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    if not args.report.is_file():
        raise SystemExit(f"research report not found: {args.report}")
    issue = json.loads(args.issue_json.read_text(encoding="utf-8"))
    if str(issue.get("number")) != args.issue_number:
        raise SystemExit(
            f"Issue JSON number does not match --issue-number: {issue.get('number')}"
        )
    if not issue.get("url"):
        raise SystemExit("Issue JSON has no URL")

    text = args.report.read_text(encoding="utf-8")
    conclusion = re.search(r"(?m)^## 결론\s*$", text)
    if conclusion is None:
        raise SystemExit("research report has no ## 결론 heading to normalize")

    normalized = build_metadata(issue, args.research_date, args.model) + text[
        conclusion.start() :
    ]
    args.report.write_text(normalized, encoding="utf-8")
    print(f"Restored deterministic research metadata: {args.report}")


if __name__ == "__main__":
    main()
