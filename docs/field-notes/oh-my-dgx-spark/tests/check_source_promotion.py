#!/usr/bin/env python3
"""Validate checkbox-driven source promotion and its research integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import promote_source_candidates as promotion  # noqa: E402
import research_discover  # noqa: E402


def main() -> None:
    workflow = ROOT / ".github/workflows/promote-source-candidates.yml"
    completion_workflow = ROOT / ".github/workflows/complete-source-promotion.yml"
    tool = ROOT / "tools/promote_source_candidates.py"
    renderer = ROOT / "tools/render_source_discovery_issue.py"
    sources_path = ROOT / "research/sources.json"
    for path in (workflow, completion_workflow, tool, renderer, sources_path):
        if not path.is_file():
            raise SystemExit(f"missing source promotion file: {path}")

    workflow_text = workflow.read_text(encoding="utf-8")
    required_fragments = (
        "types:",
        "labeled",
        "source-candidate",
        "source-approved",
        "promote_source_candidates.py",
        "research/sources.json",
        "git push",
        "gh pr create",
        "pull-requests: write",
        "Check editor permission",
        "source-approval",
        "git config user.name",
    )
    if not all(fragment in workflow_text for fragment in required_fragments):
        raise SystemExit("source promotion workflow is missing a required stage")
    if "git push origin main" in workflow_text or "git checkout main" in workflow_text:
        raise SystemExit("source promotion must not write directly to main")

    completion_text = completion_workflow.read_text(encoding="utf-8")
    completion_fragments = (
        "pull_request:",
        "closed",
        "source-approval",
        "source-promoted",
        "candidate_progress",
        "--add-label source-promoted",
        "--remove-label source-candidate",
        "--remove-label source-approved",
    )
    if not all(fragment in completion_text for fragment in completion_fragments):
        raise SystemExit("source completion workflow is missing a required stage")

    issue = {
        "number": 22,
        "labels": [{"name": "source-candidate"}],
        "body": """## 후보 목록

### 1. github.com

- [x] 반영 승인: `aaaaaaaaaaaa` · 종류 `github-repository`
- 대표 URL: <https://github.com/example/new-repo>
- 대표 제목: example/new-repo
- 권장 등록 유형: `github_repository`

### 2. example.org

- [ ] 반영 승인: `bbbbbbbbbbbb` · 종류 `web-search`
- 대표 URL: <https://example.org/post>
- 대표 제목: 선택하지 않은 자료
- 권장 등록 유형: `web_domain`

### 3. github.com

- [x] 반영 승인: `cccccccccccc` · 종류 `github-issue`
- 대표 URL: <https://github.com/example/new-repo/issues/1>
- 대표 제목: 승인된 Issue
- 권장 등록 유형: `manual`
""",
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        issue_path = root / "issue.json"
        sources_path = root / "sources.json"
        output_path = root / "promotion.json"
        issue_path.write_text(json.dumps(issue, ensure_ascii=False), encoding="utf-8")
        sources_path.write_text(
            json.dumps(
                {
                    "github_repositories": ["example/existing"],
                    "rss": [],
                    "web_search": [
                        {
                            "query": "DGX Spark GB10 DeepSeek Qwen MiniMax",
                            "domains": ["nvidia.com"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--issue-json",
                str(issue_path),
                "--sources",
                str(sources_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        config = json.loads(sources_path.read_text(encoding="utf-8"))
        if result["checked_count"] != 2 or result["promoted_count"] != 2:
            raise SystemExit("checked candidates were not promoted exactly once")
        if "example/new-repo" not in config["github_repositories"]:
            raise SystemExit("GitHub repository candidate was not registered")
        approved = config.get("approved_sources", [])
        if len(approved) != 1 or approved[0]["url"] != "https://github.com/example/new-repo/issues/1":
            raise SystemExit("manual candidate was not registered as an approved URL")

        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--issue-json",
                str(issue_path),
                "--sources",
                str(sources_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )
        rerun = json.loads(output_path.read_text(encoding="utf-8"))
        if rerun["promoted_count"] != 0 or rerun["already_registered_count"] != 2:
            raise SystemExit("repeated promotion was not idempotent")

    if len(promotion.checked_candidates(issue)) != 2:
        raise SystemExit("checked candidate parser selected the wrong blocks")
    total, checked = promotion.candidate_progress(issue)
    if (total, checked) != (3, 2):
        raise SystemExit("candidate completion progress was counted incorrectly")
    complete_issue = json.loads(json.dumps(issue))
    complete_issue["body"] = complete_issue["body"].replace("- [ ]", "- [x]")
    if promotion.candidate_progress(complete_issue) != (3, 3):
        raise SystemExit("fully checked Issue was not recognized as complete")
    try:
        promotion.checked_candidates({"labels": [], "body": issue["body"]})
    except ValueError:
        pass
    else:
        raise SystemExit("source-candidate label was not required")

    original_request_bytes = research_discover.request_bytes
    try:
        research_discover.request_bytes = lambda url, headers=None: json.dumps(
            [
                {
                    "labels": [{"name": "source-candidate"}],
                    "title": "candidate",
                    "body": "https://example.org/post",
                },
                {
                    "labels": [{"name": "research-candidate"}],
                    "title": "research",
                    "body": "https://example.net/post",
                },
            ]
        ).encode("utf-8")
        found = research_discover.existing_issue_urls("owner/repo", {}, [])
    finally:
        research_discover.request_bytes = original_request_bytes
    if "https://example.org/post" in found or "https://example.net/post" not in found:
        raise SystemExit("source-candidate URLs were not separated from research URLs")

    original_request_bytes = research_discover.request_bytes
    try:

        def fake_request_bytes(url, headers=None):
            if "/comments" in url:
                return json.dumps(
                    [
                        {
                            "body": """<!-- generated-research-candidates-part: 001/001 -->

### 2. overflow candidate

- URL: https://example.net/overflow
"""
                        },
                        {"body": "일반 댓글 https://example.net/unrelated"},
                    ]
                ).encode("utf-8")
            return json.dumps(
                [
                    {
                        "number": 42,
                        "comments_url": "https://api.github.com/repos/owner/repo/issues/42/comments",
                        "labels": [{"name": "research-candidate"}],
                        "title": "split research",
                        "body": """<!-- generated-research-url-index: partial -->

- 1: https://example.net/indexed
""",
                    }
                ]
            ).encode("utf-8")

        research_discover.request_bytes = fake_request_bytes
        warnings: list[str] = []
        found = research_discover.existing_issue_urls("owner/repo", {}, warnings)
    finally:
        research_discover.request_bytes = original_request_bytes
    if warnings:
        raise SystemExit(f"split Issue deduplication emitted warnings: {warnings}")
    if not {
        "https://example.net/indexed",
        "https://example.net/overflow",
    }.issubset(found):
        raise SystemExit("split Issue URLs were not included in deduplication")
    if "https://example.net/unrelated" in found:
        raise SystemExit("ordinary Issue comments were included in deduplication")

    original_request_bytes = research_discover.request_bytes
    try:
        research_discover.request_bytes = lambda url, headers=None: (
            b'<html><head><title>Approved page</title>'
            b'<meta name="description" content="DGX Spark evidence"></head></html>'
        )
        collected: list[dict] = []
        research_discover.collect_approved_sources(
            {
                "approved_sources": [
                    {"name": "Approved page", "url": "https://example.org/post", "kind": "manual"}
                ]
            },
            research_discover.utc_now(),
            collected,
            set(),
            set(),
            [],
        )
    finally:
        research_discover.request_bytes = original_request_bytes
    if len(collected) != 1 or collected[0]["title"] != "Approved page":
        raise SystemExit("approved URL was not collected by the research workflow")

    print("Source promotion automation wiring OK")


if __name__ == "__main__":
    main()
