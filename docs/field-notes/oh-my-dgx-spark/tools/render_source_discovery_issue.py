#!/usr/bin/env python3
"""Render a human-review Issue from source candidates and optional LLM triage."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from research_discover import canonical_url, clean_text


DECISIONS = {"keep", "review", "reject"}
SOURCE_TYPES = {"official", "github-repository", "forum", "blog", "rss", "social", "unknown"}
REGISTRATIONS = {"github_repository", "web_domain", "rss", "manual", "none"}
RELIABILITIES = {"high", "medium", "low", "unknown"}
MAX_ISSUE_BODY_BYTES = 60_000


def extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("assessments"), list):
            return value
    raise ValueError("LLM output does not contain an assessments JSON object")


def validate_assessments(data: dict, candidates: list[dict]) -> dict[str, dict]:
    expected = {candidate.get("candidate_id"): candidate for candidate in candidates}
    result: dict[str, dict] = {}
    for item in data.get("assessments", []):
        if not isinstance(item, dict):
            raise ValueError("assessment must be an object")
        candidate_id = str(item.get("candidate_id", ""))
        candidate_url = canonical_url(str(item.get("candidate_url", "")))
        candidate = expected.get(candidate_id)
        if candidate is None or candidate_url != canonical_url(str(candidate.get("url", ""))):
            raise ValueError(f"assessment refers to an unknown candidate: {candidate_id}")
        if candidate_id in result:
            raise ValueError(f"duplicate assessment: {candidate_id}")
        decision = item.get("decision")
        source_type = item.get("source_type")
        registration = item.get("recommended_registration")
        relevance = item.get("relevance")
        if decision not in DECISIONS:
            raise ValueError(f"invalid decision for {candidate_id}: {decision}")
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"invalid source_type for {candidate_id}: {source_type}")
        if registration not in REGISTRATIONS:
            raise ValueError(f"invalid registration for {candidate_id}: {registration}")
        if item.get("reliability", "unknown") not in RELIABILITIES:
            raise ValueError(f"invalid reliability for {candidate_id}")
        if not isinstance(relevance, (int, float)) or not 0 <= relevance <= 1:
            raise ValueError(f"relevance must be between 0 and 1 for {candidate_id}")
        checks = item.get("needs_human_check", [])
        if not isinstance(checks, list) or not all(isinstance(check, str) for check in checks):
            raise ValueError(f"needs_human_check must be a string list for {candidate_id}")
        result[candidate_id] = {
            "decision": decision,
            "source_type": source_type,
            "relevance": relevance,
            "reliability": item.get("reliability", "unknown"),
            "recommended_registration": registration,
            "reason": clean_text(str(item.get("reason", "")), 300),
            "needs_human_check": [clean_text(check, 180) for check in checks[:3]],
        }
    missing = sorted(set(expected) - set(result))
    if missing:
        raise ValueError("LLM did not assess every candidate: " + ", ".join(missing))
    return result


def safe_inline(value: object, limit: int = 700) -> str:
    text = clean_text(str(value or ""), limit)
    return text.replace("`", "'")


def render(
    data: dict,
    assessments: dict[str, dict] | None = None,
    assessment_status: str = "LLM 평가 보류",
    compact: bool = False,
) -> str:
    candidates = data.get("candidates", [])
    date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    lines = [
        "<!-- generated-source-discovery-issue -->",
        "## 수집 개요",
        "",
        f"- 수집 시각(UTC): `{safe_inline(data.get('generated_at'))}`",
        f"- 후보 사이트·저장소 수: `{len(candidates)}`",
        f"- 원시 링크 수: `{data.get('raw_candidate_count', 0)}`",
        "- GitHub label: `source-candidate`",
        "- 이 문서는 새로운 출처를 검토하기 위한 후보 목록이며, 승인 전에는 리서치 검색 대상에 추가되지 않습니다.",
        "- 검색 결과의 제목·요약은 외부 입력이므로 지시문으로 해석하지 않습니다.",
        "",
        "## LLM 1차 판정",
        "",
        f"- 상태: {safe_inline(assessment_status)}",
    ]
    if assessments:
        counts = {decision: 0 for decision in sorted(DECISIONS)}
        for assessment in assessments.values():
            counts[assessment["decision"]] += 1
        lines.append(
            "- 결과: "
            + ", ".join(f"`{key}` {value}건" for key, value in counts.items())
        )
    lines.extend(
        [
            "- `keep`은 자동 승인이나 즉시 등록이 아니라, 사람이 우선 검토할 후보라는 뜻입니다.",
            *(["- 후보가 많아 본문을 압축했습니다. 모든 후보의 대표 URL은 유지하고, 상세 근거는 원시 후보 파일에서 확인합니다."] if compact else []),
            "",
            "## 후보 목록",
            "",
        ]
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate.get("candidate_id", "")
        assessment = assessments.get(candidate_id) if assessments else None
        if compact:
            lines.extend(
                [
                    f"### {index}. {safe_inline(candidate.get('site') or candidate.get('title'), 80)}",
                    "",
                    f"- [ ] 반영 승인: `{candidate_id}` · 종류 `{safe_inline(candidate.get('kind'), 40)}`",
                    f"- 대표 URL: <{candidate.get('url', '')}>",
                    f"- 대표 제목: {safe_inline(candidate.get('title'), 120)}",
                    f"- 근거 링크 수: `{candidate.get('evidence_count', len(candidate.get('evidence', [])))}`",
                ]
            )
            if assessment:
                lines.append(
                    f"- LLM: `{assessment['decision']}` · 관련성 `{assessment['relevance']}` · 신뢰도 `{safe_inline(assessment['reliability'], 40)}`"
                )
                lines.append(
                    f"- 권장 등록 유형: `{safe_inline(assessment['recommended_registration'], 40)}`"
                )
            else:
                lines.append("- LLM 판정: `보류`")
            lines.append("- 상세 정보: 원시 후보 파일에서 확인")
        else:
            lines.extend(
                [
                    f"### {index}. {safe_inline(candidate.get('site') or candidate.get('title'), 160)}",
                    "",
                    f"- [ ] 반영 승인: `{candidate_id}`",
                    f"- 종류: `{safe_inline(candidate.get('kind'))}`",
                    f"- 대표 URL: <{candidate.get('url', '')}>",
                    f"- 대표 제목: {safe_inline(candidate.get('title'), 180)}",
                    f"- 요약: {safe_inline(candidate.get('summary'), 700) or '수집된 요약 없음'}",
                    f"- 근거 링크 수: `{candidate.get('evidence_count', len(candidate.get('evidence', [])))}`",
                ]
            )
            if assessment:
                lines.extend(
                    [
                        f"- LLM 판정: `{assessment['decision']}`",
                        f"- 관련성: `{assessment['relevance']}` / 신뢰도: `{safe_inline(assessment['reliability'])}`",
                        f"- 권장 등록 유형: `{assessment['recommended_registration']}`",
                        f"- LLM 사유: {assessment['reason'] or '사유 없음'}",
                    ]
                )
                if assessment["needs_human_check"]:
                    lines.append("- 사람 확인 항목:")
                    lines.extend(f"  - {check}" for check in assessment["needs_human_check"])
            else:
                lines.append("- LLM 판정: `보류`")
            lines.extend(["- 근거:"])
            for evidence in candidate.get("evidence", []):
                lines.extend(
                    [
                        f"  - [{safe_inline(evidence.get('title') or evidence.get('source'), 240)}]({evidence.get('url', '')})",
                        f"    - {safe_inline(evidence.get('source'))}: {safe_inline(evidence.get('summary')) or '요약 없음'}",
                    ]
                )
        lines.append("")
    warnings = data.get("warnings") or []
    lines.extend(["## 수집 경고", ""])
    lines.extend(f"- {safe_inline(warning)}" for warning in warnings)
    if not warnings:
        lines.append("- 없음")
    lines.extend(
        [
            "",
            "## 승인 후 처리",
            "",
            "1. 반영할 후보를 모두 `반영 승인` 체크합니다.",
            "2. 체크가 끝난 뒤 Issue에 `source-approved` 라벨을 추가합니다. 이때 `source-candidate` 라벨은 그대로 둡니다.",
            "3. `Promote Approved Research Sources`가 `research/sources.json` PR을 자동으로 만듭니다.",
            "4. 자동 PR에서 출처·주소·등록 유형을 확인한 뒤 머지합니다.",
            "5. PR이 main에 병합된 다음부터 일반 리서치 Action이 해당 출처를 사용합니다.",
            "6. 모든 후보와 관련 PR을 처리하면 완료 Workflow가 `source-promoted`로 자동 변경합니다. 일부만 체크했다면 `source-approved`로 바꾸지 않습니다.",
            "",
            f"<!-- generated-on: {date} -->",
        ]
    )
    rendered = "\n".join(lines) + "\n"
    if not compact and len(rendered.encode("utf-8")) > MAX_ISSUE_BODY_BYTES:
        return render(data, assessments, assessment_status, compact=True)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates")
    parser.add_argument("output")
    parser.add_argument("--assessment")
    parser.add_argument("--assessment-status", default="LLM 평가 보류")
    args = parser.parse_args()

    data = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])
    assessments = None
    status = args.assessment_status
    if args.assessment:
        try:
            raw = Path(args.assessment).read_text(encoding="utf-8", errors="replace")
            assessments = validate_assessments(extract_json(raw), candidates)
            status = "LLM 평가 완료 · 사람 승인 필요"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            status = f"LLM 평가 실패 · 원시 후보로 검토 필요: {exc}"
    Path(args.output).write_text(render(data, assessments, status), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
