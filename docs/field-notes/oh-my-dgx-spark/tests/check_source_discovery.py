#!/usr/bin/env python3
"""Validate source-discovery collection, triage, and Issue rendering."""

from __future__ import annotations

import json
import datetime as dt
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import discover_source_candidates as source_discovery  # noqa: E402
from discover_source_candidates import (  # noqa: E402
    candidate_is_known,
    collect_search_backend,
    configured_domains,
    group_candidates,
    search_urls,
)
from render_source_discovery_issue import (  # noqa: E402
    render,
    validate_assessments,
)


def main() -> None:
    workflow = ROOT / ".github/workflows/research-source-discovery.yml"
    config_path = ROOT / "research/source-discovery.json"
    prompt = ROOT / "prompts/source-discovery.md"
    collector = ROOT / "tools/discover_source_candidates.py"
    prompt_renderer = ROOT / "tools/render_source_discovery_prompt.py"
    issue_renderer = ROOT / "tools/render_source_discovery_issue.py"
    for path in (workflow, config_path, prompt, collector, prompt_renderer, issue_renderer):
        if not path.is_file():
            raise SystemExit(f"missing source-discovery file: {path}")

    workflow_text = workflow.read_text(encoding="utf-8")
    required_fragments = (
        "schedule:",
        "workflow_dispatch:",
        "research/source-discovery.json",
        "tools/discover_source_candidates.py",
        "SEARCH_URL",
        "BRAVE_SEARCH_API_KEY",
        "X_BEARER_TOKEN",
        "SEARCH_API_KEY",
        "COPILOT_GITHUB_TOKEN",
        "SEARCH_URLS",
        "tools/render_source_discovery_prompt.py",
        "tools/render_source_discovery_issue.py",
        "source-candidate",
        "gh issue create",
        "collection_stats",
        "검색 쿼리",
        "수집 후보:",
        'COPILOT_CLI_VERSION: "1.0.80"',
        '@github/copilot@${COPILOT_CLI_VERSION}',
        "SOURCE_DISCOVERY_MAX_AI_CREDITS",
        "DISCOVERY_MODEL: ${{ vars.SOURCE_DISCOVERY_MODEL || 'gpt-5.6-luna' }}",
        '--max-ai-credits "${DISCOVERY_MAX_AI_CREDITS}"',
        "copilot-research-automation",
        "tools/redact_sensitive_output.py",
        "--in-place",
        "--no-custom-instructions",
        "Use the view tool to read the complete source-triage task from ${prompt_file}",
        "--available-tools=view",
        "--allow-tool=read",
        "--deny-tool=url",
    )
    if not all(fragment in workflow_text for fragment in required_fragments):
        raise SystemExit("source-discovery workflow is missing a required stage")
    if "git push" in workflow_text or "git add research/sources.json" in workflow_text:
        raise SystemExit("source discovery must not activate sources without human review")
    if '-p "$(<"${RUNNER_TEMP}/source-discovery-prompt.md")"' in workflow_text:
        raise SystemExit("source-discovery prompt must not be expanded on the command line")
    if '--attachment "${prompt_file}"' in workflow_text:
        raise SystemExit("Copilot CLI 1.0.80 does not accept Markdown prompt attachments")
    if "--available-tools=read" in workflow_text:
        raise SystemExit("Copilot permission kinds must not be used as available tool names")
    if 'tail -c 6000 "${output_file}"' in workflow_text:
        raise SystemExit("source-discovery workflow still writes raw Copilot failures")
    if "vars.RESEARCH_MODEL || 'auto'" in workflow_text:
        raise SystemExit("source discovery must not fall back to the research model or auto")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("github_repository_queries", "web_queries", "search_queries", "search_directory_url", "x_queries"):
        if not config.get(key):
            raise SystemExit(f"source discovery config contains no {key}")

    known = configured_domains(
        {
            "github_repositories": ["recrack/example"],
            "rss": [{"url": "https://forum.example.org/feed.rss"}],
            "web_search": [{"domains": ["docs.example.com"]}],
        }
    )
    if not candidate_is_known("sub.docs.example.com", known):
        raise SystemExit("subdomains must match configured source domains")
    if candidate_is_known("new-source.example", known):
        raise SystemExit("unregistered source domain was treated as known")

    raw = [
        {
            "candidate_id": "ignored-before-grouping",
            "host": "new-source.example",
            "url": "https://new-source.example/a",
            "kind": "web-search",
            "title": "A",
            "summary": "first evidence",
            "source": "Brave",
            "published_at": None,
        },
        {
            "host": "new-source.example",
            "url": "https://new-source.example/b",
            "kind": "search-backend",
            "title": "B",
            "summary": "second evidence",
            "source": "Search backend",
            "published_at": None,
        },
    ]
    candidates = group_candidates(raw)
    if len(candidates) != 1 or candidates[0]["evidence_count"] != 2:
        raise SystemExit("same source domain was not grouped with its evidence")

    candidate = candidates[0]

    backend_raw: list[dict] = []
    backend_seen: set[str] = set()
    backend_stats: dict = {}
    original_request_json = source_discovery.request_json
    source_discovery.request_json = lambda url, headers: {
        "results": [
            {
                "url": "https://another-source.example/post",
                "title": "DGX Spark result",
                "content": "Search backend snippet",
            }
        ]
    }
    try:
        collect_search_backend(
            {"search_queries": ["DGX Spark"]},
            dt.datetime.now(dt.timezone.utc),
            backend_raw,
            backend_seen,
            set(),
            set(),
            set(),
            [],
            base_url="https://search.example",
            collection_stats=backend_stats,
        )
    finally:
        source_discovery.request_json = original_request_json
    if len(backend_raw) != 1 or backend_raw[0]["kind"] != "search-backend":
        raise SystemExit("search backend JSON result was not collected")
    stats = backend_stats.get("search_backend", {})
    if (
        stats.get("source") != "base_url"
        or stats.get("requests_attempted") != 1
        or stats.get("requests_succeeded") != 1
        or stats.get("result_count") != 1
        or stats.get("query_attempts", [{}])[0].get("status") != "success"
    ):
        raise SystemExit("search query execution statistics were not recorded")

    if search_urls({"search_urls": ["https://first.example/", "https://second.example"]}, []) != [
        "https://first.example",
        "https://second.example",
    ]:
        raise SystemExit("search URL list was not normalized")

    failover_raw: list[dict] = []
    failover_seen: set[str] = set()
    original_request_json = source_discovery.request_json

    def fail_first(url: str, headers: dict[str, str]) -> dict:
        if url.startswith("https://first.example/"):
            raise RuntimeError("simulated rate limit")
        return {"results": [{"url": "https://fallback.example/post", "title": "fallback", "content": "ok"}]}

    source_discovery.request_json = fail_first
    failover_warnings: list[str] = []
    try:
        collect_search_backend(
            {"search_queries": ["DGX Spark"], "search_urls": ["https://first.example", "https://second.example"]},
            dt.datetime.now(dt.timezone.utc),
            failover_raw,
            failover_seen,
            set(),
            set(),
            set(),
            failover_warnings,
            base_url="",
        )
    finally:
        source_discovery.request_json = original_request_json
    if len(failover_raw) != 1 or not any("검색 인스턴스 1 제외" in warning for warning in failover_warnings):
        raise SystemExit("search failover did not skip a failed instance")
    if any("first.example" in warning for warning in failover_warnings):
        raise SystemExit("search failover leaked the endpoint URL")

    assessments = validate_assessments(
        {
            "assessments": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_url": candidate["url"],
                    "decision": "review",
                    "source_type": "blog",
                    "relevance": 0.8,
                    "reliability": "unknown",
                    "recommended_registration": "web_domain",
                    "reason": "후보 검토 필요",
                    "needs_human_check": [
                        "원문 접근성",
                        "운영자 신원",
                        "최근 갱신일",
                        "이 항목은 잘려야 함",
                    ],
                }
            ]
        },
        candidates,
    )
    body = render(
        {
            "generated_at": "2026-08-23T00:00:00+00:00",
            "raw_candidate_count": 2,
            "candidates": candidates,
            "warnings": [],
        },
        assessments,
        "LLM 평가 완료 · 사람 승인 필요",
    )
    for fragment in ("source-candidate", "LLM 판정", "new-source.example", "원문 접근성"):
        if fragment not in body:
            raise SystemExit(f"rendered source Issue is missing: {fragment}")
    if "이 항목은 잘려야 함" in body:
        raise SystemExit("source triage retained more than three human checks")

    large_candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "site": f"large-source-{index}.example",
            "url": f"https://large-source-{index}.example/post",
            "kind": "web-search",
            "title": "Long title",
            "summary": "x" * 700,
            "evidence_count": 8,
            "evidence": [
                {
                    "url": f"https://large-source-{index}.example/evidence-{evidence}",
                    "title": "Long evidence title",
                    "source": "search",
                    "summary": "y" * 700,
                }
                for evidence in range(8)
            ],
        }
        for index in range(150)
    ]
    compact_body = render(
        {"generated_at": "2026-08-23T00:00:00+00:00", "candidates": large_candidates},
        assessment_status="LLM 평가 보류",
    )
    if len(compact_body.encode("utf-8")) > 65_536:
        raise SystemExit("large source Issue still exceeds GitHub's body limit")
    if "https://large-source-149.example/post" not in compact_body:
        raise SystemExit("compact source Issue dropped a candidate URL")

    print("Source discovery automation wiring OK")


if __name__ == "__main__":
    main()
