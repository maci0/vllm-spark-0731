#!/usr/bin/env python3
"""Promote checked source-discovery candidates into ``research/sources.json``.

The source-discovery Issue is intentionally the approval boundary.  This tool
only reads checked candidate blocks from that Issue; the caller still commits
the resulting configuration on a reviewable pull request.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from pathlib import Path

from discover_source_candidates import canonical_url, clean_text, host_from_url


CANDIDATE_BLOCK_RE = re.compile(
    r"(?ms)^###\s+\d+\..*?(?=^###\s+\d+\.|^##\s|\Z)"
)
CHECKED_CANDIDATE_RE = re.compile(
    r"^-\s*\[[xX]\][^\n]*?(?:`([0-9a-f]{12})`|(?<![0-9a-f])([0-9a-f]{12})(?![0-9a-f]))",
    re.MULTILINE,
)


def first_field(block: str, label: str) -> str:
    match = re.search(rf"^-\s*{re.escape(label)}:\s*(.+?)\s*$", block, re.MULTILINE)
    if not match:
        return ""
    value = match.group(1).strip()
    if value.startswith("<") and value.endswith(">"):  # Markdown autolink.
        value = value[1:-1]
    if value.startswith("`") and value.endswith("`"):
        value = value[1:-1]
    return clean_text(value, 700)


def parse_candidate_block(block: str) -> dict | None:
    checked = CHECKED_CANDIDATE_RE.search(block)
    if not checked:
        return None
    candidate_id = checked.group(1) or checked.group(2)
    url_match = re.search(r"^-\s*대표 URL:\s*<([^>\n]+)>", block, re.MULTILINE)
    if not url_match:
        url_match = re.search(r"^-\s*대표 URL:\s*`?([^\n`]+)`?", block, re.MULTILINE)
    url = canonical_url(url_match.group(1).strip()) if url_match else ""
    if not url:
        raise ValueError(f"checked candidate {candidate_id} has no valid representative URL")

    kind = first_field(block, "종류")
    if not kind:
        compact_kind = re.search(r"종류\s+`([^`]+)`", checked.string[checked.start() :], re.MULTILINE)
        kind = compact_kind.group(1).strip() if compact_kind else ""
    registration = first_field(block, "권장 등록 유형")
    title = first_field(block, "대표 제목") or url
    summary = first_field(block, "요약")
    return {
        "candidate_id": candidate_id,
        "url": url,
        "kind": kind or "unknown",
        "registration": registration or "manual",
        "title": title,
        "summary": summary,
    }


def checked_candidates(issue: dict) -> list[dict]:
    labels = {
        str(item.get("name", ""))
        for item in issue.get("labels", [])
        if isinstance(item, dict)
    }
    if "source-candidate" not in labels:
        raise ValueError("the Issue does not have the source-candidate label")
    body = str(issue.get("body") or "")
    candidates: list[dict] = []
    seen: set[str] = set()
    for block in CANDIDATE_BLOCK_RE.findall(body):
        candidate = parse_candidate_block(block)
        if candidate is None:
            continue
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            raise ValueError(f"duplicate checked candidate: {candidate_id}")
        seen.add(candidate_id)
        candidates.append(candidate)
    return candidates


def candidate_progress(issue: dict) -> tuple[int, int]:
    """Return (total candidate blocks, checked candidate blocks)."""

    body = str(issue.get("body") or "")
    blocks = CANDIDATE_BLOCK_RE.findall(body)
    checked = sum(1 for block in blocks if CHECKED_CANDIDATE_RE.search(block))
    return len(blocks), checked


def github_repository(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host != "github.com":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        return ""
    owner, name = parts
    if name.endswith(".git"):
        name = name[:-4]
    if not owner or not name or any(char in owner + name for char in "<>\"'"):
        return ""
    return f"{owner}/{name}"


def ensure_list(config: dict, key: str) -> list:
    value = config.setdefault(key, [])
    if not isinstance(value, list):
        raise ValueError(f"sources.json field {key!r} must be a list")
    return value


def add_github_repository(config: dict, repository: str) -> bool:
    repositories = ensure_list(config, "github_repositories")
    existing = {str(item).casefold() for item in repositories if isinstance(item, str)}
    if repository.casefold() in existing:
        return False
    repositories.append(repository)
    return True


def add_rss_source(config: dict, candidate: dict, issue_number: int) -> bool:
    feeds = ensure_list(config, "rss")
    url = candidate["url"]
    existing = {
        canonical_url(str(item.get("url", "")))
        for item in feeds
        if isinstance(item, dict)
    }
    if url in existing:
        return False
    feeds.append(
        {
            "name": candidate["title"],
            "url": url,
            "approved_from_issue": issue_number,
        }
    )
    return True


def add_web_domain(config: dict, domain: str) -> bool:
    entries = ensure_list(config, "web_search")
    normalized = domain.lower().removeprefix("www.")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        domains = entry.setdefault("domains", [])
        if not isinstance(domains, list):
            raise ValueError("sources.json web_search.domains must be a list")
        if any(str(item).lower().removeprefix("www.") == normalized for item in domains):
            return False

    # The first general query is the stable place for newly approved domains.
    # If a repository changes the query list later, create a clearly named one.
    target = next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("query") == "DGX Spark GB10 DeepSeek Qwen MiniMax"
        ),
        None,
    )
    if target is None:
        target = {"query": "DGX Spark GB10 DeepSeek Qwen MiniMax", "domains": []}
        entries.append(target)
    target.setdefault("domains", []).append(normalized)
    return True


def add_approved_url(config: dict, candidate: dict, issue_number: int) -> bool:
    sources = ensure_list(config, "approved_sources")
    url = candidate["url"]
    existing = {
        canonical_url(str(item.get("url", "")))
        for item in sources
        if isinstance(item, dict)
    }
    if url in existing:
        return False
    sources.append(
        {
            "name": candidate["title"],
            "url": url,
            "kind": candidate["kind"],
            "candidate_id": candidate["candidate_id"],
            "approved_from_issue": issue_number,
        }
    )
    return True


def promote_candidate(config: dict, candidate: dict, issue_number: int) -> tuple[str, bool]:
    kind = candidate["kind"]
    registration = candidate["registration"]

    if kind == "github-repository":
        repository = github_repository(candidate["url"])
        if repository:
            return "github_repositories", add_github_repository(config, repository)

    if kind == "rss" or registration == "rss":
        return "rss", add_rss_source(config, candidate, issue_number)

    if registration == "web_domain":
        domain = host_from_url(candidate["url"])
        if domain and domain != "github.com":
            return "web_search", add_web_domain(config, domain)

    return "approved_sources", add_approved_url(config, candidate, issue_number)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-json", required=True)
    parser.add_argument("--sources", default="research/sources.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    issue = json.loads(Path(args.issue_json).read_text(encoding="utf-8"))
    issue_number = int(issue.get("number", 0))
    if issue_number <= 0:
        raise SystemExit("Issue number is missing or invalid")
    candidates = checked_candidates(issue)
    config_path = Path(args.sources)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    promoted: list[dict] = []
    already_registered = 0
    changed = False
    for candidate in candidates:
        target, candidate_changed = promote_candidate(config, candidate, issue_number)
        changed = changed or candidate_changed
        if candidate_changed:
            promoted.append({"candidate_id": candidate["candidate_id"], "target": target})
        else:
            already_registered += 1

    if changed:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    result = {
        "issue_number": issue_number,
        "checked_count": len(candidates),
        "promoted_count": len(promoted),
        "already_registered_count": already_registered,
        "changed": changed,
        "promoted": promoted,
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "승인 후보 처리: "
        f"체크 {len(candidates)}건, 신규 반영 {len(promoted)}건, "
        f"기존 등록 {already_registered}건"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
