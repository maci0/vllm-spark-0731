#!/usr/bin/env python3
"""Render a bounded research Issue body and chunked candidate comments."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


MAX_ISSUE_BODY_BYTES = 60_000
MAX_COMMENT_BYTES = 60_000
CANDIDATE_CHUNK_BYTES = 56_000
BODY_URL_INDEX_BYTES = 30_000
COMMENT_MARKER_PREFIX = "<!-- generated-research-candidates-part:"


def compact_text(value: object, limit: int) -> str:
    """Collapse untrusted metadata to a single bounded Markdown line."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def byte_length(value: str) -> int:
    return len(value.encode("utf-8"))


def candidate_url(candidate: dict) -> str:
    url = str(candidate.get("url") or "").strip()
    if not url:
        raise ValueError("candidate URL is empty")
    if any(character.isspace() for character in url):
        raise ValueError("candidate URL contains whitespace")
    if len(url) > 8_192:
        raise ValueError("candidate URL exceeds 8,192 characters")
    return url


def render_candidate(candidate: dict, index: int) -> str:
    title = compact_text(candidate.get("title") or "제목 없음", 240)
    source = compact_text(candidate.get("source"), 300)
    kind = compact_text(candidate.get("kind"), 100)
    published_at = compact_text(candidate.get("published_at"), 100)
    url = candidate_url(candidate)
    summary = compact_text(candidate.get("summary") or "수집된 요약 없음", 700)
    return "\n".join(
        [
            f"### {index}. {title}",
            "",
            f"- 출처: `{source}`",
            f"- 종류: `{kind}`",
            f"- 게시·수정 시각: `{published_at or '확인되지 않음'}`",
            f"- URL: {url}",
            f"- 요약: {summary}",
        ]
    )


def chunk_candidates(candidates: list[dict]) -> list[list[tuple[int, str]]]:
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_bytes = 0

    for index, candidate in enumerate(candidates, start=1):
        block = render_candidate(candidate, index)
        block_bytes = byte_length(block) + 2
        if block_bytes > CANDIDATE_CHUNK_BYTES:
            raise ValueError(f"candidate {index} exceeds the safe GitHub comment size")
        if current and current_bytes + block_bytes > CANDIDATE_CHUNK_BYTES:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append((index, block))
        current_bytes += block_bytes

    if current:
        chunks.append(current)
    return chunks


def render_comment(
    chunk: list[tuple[int, str]], part_number: int, part_count: int
) -> str:
    first_index = chunk[0][0]
    last_index = chunk[-1][0]
    blocks = "\n\n".join(block for _, block in chunk)
    return "\n".join(
        [
            f"{COMMENT_MARKER_PREFIX} {part_number:03d}/{part_count:03d} -->",
            f"## 후보 자료 ({part_number}/{part_count})",
            "",
            f"- 전체 목록 기준: `{first_index}~{last_index}`",
            "- 자동 수집된 후보의 링크와 짧은 메타데이터입니다.",
            "",
            blocks,
            "",
        ]
    )


def render_url_index(candidates: list[dict]) -> list[str]:
    lines: list[str] = []
    current_bytes = 0
    for index, candidate in enumerate(candidates, start=1):
        line = f"- {index}: {candidate_url(candidate)}"
        line_bytes = byte_length(line) + 1
        if lines and current_bytes + line_bytes > BODY_URL_INDEX_BYTES:
            break
        if line_bytes > BODY_URL_INDEX_BYTES:
            break
        lines.append(line)
        current_bytes += line_bytes
    return lines


def render_body(data: dict, candidates: list[dict], comment_count: int) -> str:
    warnings = [compact_text(item, 300) for item in (data.get("warnings") or [])]
    visible_warnings = warnings[:15]
    omitted_warning_count = len(warnings) - len(visible_warnings)
    url_index = render_url_index(candidates)
    url_index_complete = len(url_index) == len(candidates)
    date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    lines = [
        "<!-- generated-research-issue -->",
        "<!-- generated-research-url-index: "
        + ("complete" if url_index_complete else "partial")
        + " -->",
        "## 수집 개요",
        "",
        f"- 수집 시각(UTC): `{compact_text(data.get('generated_at'), 100)}`",
        f"- 후보 수: `{len(candidates)}`",
        f"- 후보 상세 댓글: `{comment_count}`개",
        "- 후보 전체 목록은 바로 아래의 자동 생성 댓글에 나누어 기록했습니다.",
        "- 원문 전체를 복사하지 않고 URL과 짧은 메타데이터만 보관합니다.",
        "",
        "## 후보 URL 색인",
        "",
    ]
    lines.extend(url_index)
    if not url_index_complete:
        lines.append(
            f"- 본문에는 `{len(url_index)}/{len(candidates)}`개 URL만 실었습니다. 전체 목록은 후보 상세 댓글에 있습니다."
        )
    lines.extend(
        [
            "",
            "## 자동 수집 경고",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in visible_warnings)
    if omitted_warning_count:
        lines.append(
            f"- 그 밖의 경고 `{omitted_warning_count}`건은 본문 크기 제한으로 생략했습니다."
        )
    if not warnings:
        lines.append("- 없음")
    lines.extend(
        [
            "",
            "## 다음 단계",
            "",
            "- 공식 문서·recipe·재현 조건이 있는 자료를 우선 확인합니다.",
            "- 숫자만 있는 주장은 모델 버전·양자화·런타임·노드 수·동시성을 확인하기 전까지 확정하지 않습니다.",
            "- 분석을 시작하려면 maintainer가 `research-ready` 라벨을 추가합니다.",
            "",
            f"<!-- generated-on: {date} -->",
            "",
        ]
    )
    body = "\n".join(lines)
    if byte_length(body) > MAX_ISSUE_BODY_BYTES:
        raise ValueError("rendered Issue body exceeds the safe GitHub size")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--comments-dir", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    raw_candidates = data.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates must be a JSON list")
    candidates = [item for item in raw_candidates if isinstance(item, dict)]
    if len(candidates) != len(raw_candidates):
        raise ValueError("every candidate must be a JSON object")

    chunks = chunk_candidates(candidates)
    comments = [
        render_comment(chunk, part_number, len(chunks))
        for part_number, chunk in enumerate(chunks, start=1)
    ]
    for part_number, comment in enumerate(comments, start=1):
        if byte_length(comment) > MAX_COMMENT_BYTES:
            raise ValueError(
                f"rendered candidate comment {part_number} exceeds the safe GitHub size"
            )

    output_path = Path(args.output)
    comments_dir = Path(args.comments_dir)
    comments_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in comments_dir.glob("part-*.md"):
        stale_path.unlink()

    output_path.write_text(
        render_body(data, candidates, len(comments)), encoding="utf-8"
    )
    for part_number, comment in enumerate(comments, start=1):
        (comments_dir / f"part-{part_number:03d}.md").write_text(
            comment, encoding="utf-8"
        )

    print(
        f"Rendered {len(candidates)} candidates in {len(comments)} bounded comment part(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
