#!/usr/bin/env python3
"""Check that Copilot cannot remove the report's deterministic metadata."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
NORMALIZER = ROOT / "tools/normalize_research_report.py"
CHECKER = ROOT / "tools/check_research_report.py"


def expect_contract_failure(
    report_path: Path,
    text: str,
    expected_message: str,
) -> None:
    report_path.write_text(text, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            str(report_path),
            "--issue-number",
            "2",
            "--research-date",
            "2026-08-22",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    if result.returncode == 0 or expected_message not in output:
        raise SystemExit(
            f"expected report contract failure {expected_message!r}, got: {output!r}"
        )


def main() -> None:
    issue = {
        "number": 2,
        "url": "https://github.com/recrack/oh-my-dgx-spark/issues/2",
        "body": """## 수집 개요

- 후보 수: `40`

후보 상세는 자동 생성 댓글에 있습니다.
""",
        "comments": [
            {
                "body": """<!-- generated-research-candidates-part: 001/001 -->

## 후보 자료

### 1. 첫 번째 후보

- URL: https://example.com/one

### 2. 두 번째 후보

- URL: https://example.com/two

### 3. 세 번째 후보

- URL: https://example.com/three
""",
            },
            {"body": "### 999. 일반 댓글에 있는 번호 목록"},
        ],
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        issue_path = root / "issue.json"
        report_path = root / "research-issue-2_2026-08-22.md"
        issue_path.write_text(json.dumps(issue), encoding="utf-8")
        report_path.write_text(
            """# 임의 제목

## 메타데이터

- 모델이 메타데이터를 바꿈

## 결론

- 종합 판정: 분석 대기
- 승격 가능한 항목: 없음
- 아직 확정하지 않는 항목: 수치

## 확인된 사실

- 확인 필요: https://example.com/fact

## 커뮤니티 주장

- 재현 대기: https://example.com/community

## 충돌·미확인 내용

- 조건 부족

## 책 반영 제안

- 별도 승격 PR에서 검토

## 출처 목록

- 예시 출처: https://example.com/source

## 보류 사유 및 다음 작업

- 직접 재현
""",
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                str(NORMALIZER),
                "--report",
                str(report_path),
                "--issue-json",
                str(issue_path),
                "--issue-number",
                "2",
                "--research-date",
                "2026-08-22",
                "--model",
                "auto",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                str(report_path),
                "--issue-number",
                "2",
                "--research-date",
                "2026-08-22",
            ],
            check=True,
        )
        text = report_path.read_text(encoding="utf-8")
        if "- 분석 실행기: `GitHub Copilot CLI`" not in text:
            raise SystemExit("deterministic execution metadata was not restored")
        if "- 수집 후보 수(이슈 원문 목록 기준): `3`" not in text:
            raise SystemExit("candidate count was not derived from the Issue candidate list")
        if not text.startswith("# DGX Spark 리서치 기록 — Issue #2 — 2026-08-22\n"):
            raise SystemExit("deterministic report title was not restored")

        expect_contract_failure(
            report_path,
            text.replace("- 종합 판정: 분석 대기", "- 종합 판정:"),
            "empty conclusion field: 종합 판정",
        )
        expect_contract_failure(
            report_path,
            text.replace(
                "- 예시 출처: https://example.com/source",
                "- 예시 출처",
            ),
            "source list must contain at least one source URL",
        )

    print("Research report normalization OK")


if __name__ == "__main__":
    main()
