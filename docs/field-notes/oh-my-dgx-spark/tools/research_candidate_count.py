#!/usr/bin/env python3
"""Count candidates from a research Issue without trusting a stale total."""

from __future__ import annotations

import re


CANDIDATE_HEADING_PATTERN = re.compile(r"(?m)^###\s+(\d+)\.\s+\S")
GENERATED_CANDIDATE_COMMENT_MARKER = "<!-- generated-research-candidates-part:"
DECLARED_COUNT_PATTERNS = (
    re.compile(r"후보\s*수\s*[:：]\s*`?(\d+)", flags=re.IGNORECASE),
    re.compile(r"candidate[_ -]?count\s*[:=]\s*`?(\d+)", flags=re.IGNORECASE),
    re.compile(r"(?<!\d)(\d+)\s*개\s*후보", flags=re.IGNORECASE),
)


def candidate_count(issue: dict) -> str:
    """Return the count in the Issue's numbered candidate list.

    The generated collector Issue has one ``### N.`` heading per candidate in
    its body or marked candidate comments. That list is authoritative when
    present. A declared total is retained as a compatibility fallback for
    hand-written Issues that do not have the generated list; it never
    overrides an actual list count.
    """

    body = str(issue.get("body") or "")
    candidate_texts = [body]
    comments = issue.get("comments") or []
    if isinstance(comments, list):
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            comment_body = str(comment.get("body") or "")
            if GENERATED_CANDIDATE_COMMENT_MARKER in comment_body:
                candidate_texts.append(comment_body)
    candidate_numbers = {
        int(number)
        for text in candidate_texts
        for number in CANDIDATE_HEADING_PATTERN.findall(text)
    }
    if candidate_numbers:
        return str(len(candidate_numbers))

    for pattern in DECLARED_COUNT_PATTERNS:
        match = pattern.search(body)
        if match:
            return match.group(1)
    return "확인 필요"
