#!/usr/bin/env python3
"""Discover previously unregistered DGX Spark research-source candidates.

This collector searches broadly, but it never edits ``research/sources.json``.
It emits grouped domains and GitHub repositories for a human/LLM review Issue.
External page text is treated as untrusted metadata; the LLM receives snippets
only after this deterministic collection and deduplication step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import urllib.parse
from pathlib import Path

from research_discover import (
    USER_AGENT,
    canonical_url,
    clean_text,
    github_headers,
    parse_date,
    request_json,
    search_headers,
)


IGNORED_HOSTS = {
    "bing.com",
    "brave.com",
    "duckduckgo.com",
    "facebook.com",
    "google.com",
    "instagram.com",
    "t.co",
    "twitter.com",
    "x.com",
    "youtube.com",
}


def host_from_url(value: str) -> str:
    normalized = canonical_url(value)
    if not normalized:
        return ""
    host = urllib.parse.urlsplit(normalized).hostname or ""
    return host.lower().removeprefix("www.")


def host_matches(host: str, domain: str) -> bool:
    host = host.lower().removeprefix("www.")
    domain = domain.lower().removeprefix("www.")
    return bool(host and domain and (host == domain or host.endswith("." + domain)))


def configured_domains(config: dict) -> set[str]:
    domains = {item.lower().removeprefix("www.") for item in config.get("known_domains", [])}
    for item in config.get("web_search", []):
        domains.update(
            str(domain).lower().removeprefix("www.")
            for domain in item.get("domains", [])
        )
    for item in config.get("rss", []):
        domain = host_from_url(str(item.get("url", "")))
        if domain:
            domains.add(domain)
    for item in config.get("approved_sources", []):
        domain = host_from_url(str(item.get("url", "")))
        if domain:
            domains.add(domain)
    if config.get("github_repositories"):
        domains.add("github.com")
    domains.update(IGNORED_HOSTS)
    return domains


def candidate_is_known(host: str, known_domains: set[str]) -> bool:
    return any(host_matches(host, domain) for domain in known_domains)


def candidate_key(item: dict) -> str:
    if item.get("kind") == "github-repository":
        return f"github:{item.get('repository', item.get('url', ''))}"
    return host_from_url(item.get("url", ""))


def append_raw_candidate(
    raw: list[dict],
    seen_urls: set[str],
    existing_urls: set[str],
    existing_hosts: set[str],
    known_domains: set[str],
    *,
    source: str,
    kind: str,
    url: str,
    title: str,
    summary: str = "",
    published_at: str | None = None,
    repository: str = "",
    cutoff: dt.datetime,
) -> None:
    normalized = canonical_url(url)
    host = host_from_url(normalized)
    if not normalized or not host or normalized in seen_urls or normalized in existing_urls:
        return
    if kind != "github-repository" and (
        candidate_is_known(host, known_domains) or candidate_is_known(host, existing_hosts)
    ):
        return
    published = parse_date(published_at)
    if published is not None and published < cutoff:
        return
    seen_urls.add(normalized)
    raw.append(
        {
            "source": clean_text(source, 180),
            "kind": kind,
            "repository": repository,
            "url": normalized,
            "host": host,
            "title": clean_text(title, 240) or normalized,
            "summary": clean_text(summary),
            "published_at": published.isoformat() if published else None,
        }
    )


def existing_source_candidates(
    repo: str, headers: dict[str, str], warnings: list[str]
) -> tuple[set[str], set[str]]:
    """Return URLs and hosts already proposed in source-candidate Issues."""

    if not repo:
        return set(), set()
    urls: set[str] = set()
    hosts: set[str] = set()
    try:
        for page in range(1, 4):
            endpoint = (
                f"https://api.github.com/repos/{repo}/issues"
                f"?state=all&labels=source-candidate&per_page=100&page={page}"
            )
            items = request_json(endpoint, headers)
            if not isinstance(items, list) or not items:
                break
            for issue in items:
                text = f"{issue.get('title', '')}\n{issue.get('body', '')}"
                for token in text.split():
                    normalized = canonical_url(token.strip("<>()[]{}.,`"))
                    if normalized:
                        urls.add(normalized)
                        host = host_from_url(normalized)
                        if host:
                            hosts.add(host)
            if len(items) < 100:
                break
    except Exception as exc:  # pragma: no cover - exercised by live API failures
        warnings.append(f"source-candidate Issue 중복 확인 실패: {exc}")
    return urls, hosts


def collect_github_repositories(
    config: dict,
    token: str,
    cutoff: dt.datetime,
    raw: list[dict],
    seen_urls: set[str],
    existing_urls: set[str],
    existing_hosts: set[str],
    known_domains: set[str],
    warnings: list[str],
) -> None:
    headers = github_headers(token)
    configured = set(config.get("github_repositories", []))
    for query in config.get("github_repository_queries", []):
        try:
            params = urllib.parse.urlencode(
                {
                    "q": f'"{query}" in:name,description',
                    "sort": "updated",
                    "order": "desc",
                    "per_page": "20",
                }
            )
            result = request_json(
                f"https://api.github.com/search/repositories?{params}", headers
            )
            for item in result.get("items", []) if isinstance(result, dict) else []:
                full_name = item.get("full_name", "")
                updated_at = item.get("updated_at") or item.get("pushed_at")
                if not full_name or full_name in configured:
                    continue
                append_raw_candidate(
                    raw,
                    seen_urls,
                    existing_urls,
                    existing_hosts,
                    known_domains,
                    source=f"GitHub repository search: {query}",
                    kind="github-repository",
                    repository=full_name,
                    url=item.get("html_url", ""),
                    title=item.get("full_name") or item.get("name", ""),
                    summary=item.get("description", "") or "",
                    published_at=updated_at,
                    cutoff=cutoff,
                )
        except Exception as exc:  # pragma: no cover - exercised by live API failures
            warnings.append(f"GitHub repository 검색 {query!r} 실패: {exc}")


def collect_brave(
    config: dict,
    cutoff: dt.datetime,
    raw: list[dict],
    seen_urls: set[str],
    existing_urls: set[str],
    existing_hosts: set[str],
    known_domains: set[str],
    warnings: list[str],
) -> None:
    token = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not token:
        if config.get("web_queries"):
            warnings.append("Brave Search 생략: BRAVE_SEARCH_API_KEY가 등록되지 않았습니다")
        return
    for query in config.get("web_queries", []):
        try:
            params = urllib.parse.urlencode(
                {"q": query, "count": "20", "freshness": "pm"}
            )
            result = request_json(
                "https://api.search.brave.com/res/v1/web/search?" + params,
                {"Accept": "application/json", "X-Subscription-Token": token},
            )
            results = (result.get("web") or {}).get("results", [])
            for item in results if isinstance(results, list) else []:
                append_raw_candidate(
                    raw,
                    seen_urls,
                    existing_urls,
                    existing_hosts,
                    known_domains,
                    source=f"Brave Search: {query}",
                    kind="web-search",
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    summary=item.get("description", ""),
                    cutoff=cutoff,
                )
        except Exception as exc:  # pragma: no cover - exercised by live API failures
            warnings.append(f"Brave Search 후보 발굴 {query!r} 실패: {exc}")


def collect_search_backend(
    config: dict,
    cutoff: dt.datetime,
    raw: list[dict],
    seen_urls: set[str],
    existing_urls: set[str],
    existing_hosts: set[str],
    known_domains: set[str],
    warnings: list[str],
    base_url: str = "",
    collection_stats: dict | None = None,
) -> None:
    """Use one search instance at a time and fail over on API errors."""

    stats_root = collection_stats if collection_stats is not None else {}
    backend_stats = stats_root.setdefault(
        "search_backend",
        {
            "source": (
                "SEARCH_URLS"
                if os.environ.get("SEARCH_URLS")
                else "SEARCH_URL"
                if os.environ.get("SEARCH_URL")
                else "base_url"
                if base_url
                else "config-cache-or-directory"
            ),
            "configured_query_count": len(config.get("search_queries", [])),
            "configured_instance_count": 0,
            "instances_attempted": 0,
            "instances_succeeded": 0,
            "instances_failed": 0,
            "requests_attempted": 0,
            "requests_succeeded": 0,
            "result_count": 0,
            "query_attempts": [],
        },
    )
    urls = search_urls(config, warnings, base_url)
    backend_stats["configured_instance_count"] = len(urls)
    if not urls:
        backend_stats["status"] = "no-endpoint"
        return

    max_instances = int(config.get("search_max_instances", 3))
    if max_instances <= 0:
        warnings.append("검색 후보 발굴 생략: search_max_instances가 0 이하입니다")
        backend_stats["status"] = "disabled"
        return
    for index, instance in enumerate(urls[:max_instances], start=1):
        instance_raw: list[dict] = []
        instance_seen: set[str] = set()
        backend_stats["instances_attempted"] += 1
        current_attempt: dict | None = None
        try:
            for query_index, query in enumerate(config.get("search_queries", []), start=1):
                current_attempt = {
                    "instance_index": index,
                    "query_index": query_index,
                    "query": clean_text(str(query), 240),
                    "status": "started",
                }
                backend_stats["query_attempts"].append(current_attempt)
                backend_stats["requests_attempted"] += 1
                params = urllib.parse.urlencode(
                    {
                        "q": query,
                        "format": "json",
                        "categories": "general",
                        "language": "all",
                        "time_range": "month",
                    }
                )
                result = request_json(
                    f"{instance}/search?{params}",
                    search_headers(),
                )
                results = result.get("results", []) if isinstance(result, dict) else []
                result_count = len(results) if isinstance(results, list) else 0
                current_attempt.update(status="success", result_count=result_count)
                backend_stats["requests_succeeded"] += 1
                backend_stats["result_count"] += result_count
                for item in results if isinstance(results, list) else []:
                    append_raw_candidate(
                        instance_raw,
                        instance_seen,
                        existing_urls,
                        existing_hosts,
                        known_domains,
                        source=f"Search backend: {query}",
                        kind="search-backend",
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        summary=item.get("content", "") or item.get("snippet", ""),
                        published_at=item.get("publishedDate"),
                        cutoff=cutoff,
                    )
        except Exception as exc:  # pragma: no cover - exercised by live API failures
            if current_attempt is not None and current_attempt.get("status") == "started":
                current_attempt.update(status="failed", error=type(exc).__name__)
            warnings.append(f"검색 인스턴스 {index} 제외: {type(exc).__name__}")
            backend_stats["instances_failed"] += 1
            continue
        backend_stats["instances_succeeded"] += 1
        backend_stats["status"] = "success"
        raw.extend(instance_raw)
        seen_urls.update(instance_seen)
        return
    backend_stats["status"] = "failed"
    warnings.append("검색 후보 발굴 실패: 시도한 인스턴스가 모두 응답하지 않았습니다")


def search_urls(config: dict, warnings: list[str], base_url: str = "") -> list[str]:
    """Resolve explicit URLs, the last-good cache, then the live directory."""

    explicit = os.environ.get("SEARCH_URLS", "")
    if explicit:
        values = explicit.split(",")
    else:
        single_url = os.environ.get("SEARCH_URL", "")
        if base_url or single_url:
            values = [base_url or single_url]
        elif config.get("search_urls"):
            values = config["search_urls"]
        else:
            cache_path = os.environ.get(
                "SEARCH_ENDPOINTS_CACHE",
                str(config.get("search_endpoints_cache", "research/search-endpoints.json")),
            )
            try:
                cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
                values = [
                    item.get("url", "")
                    for item in cache.get("endpoints", [])
                    if isinstance(item, dict) and item.get("url")
                ]
            except (OSError, json.JSONDecodeError, AttributeError):
                values = []
            if values:
                return normalize_search_urls(values)
            directory_url = os.environ.get(
                "SEARCH_DIRECTORY_URL",
                config.get("search_directory_url", "https://searx.space/data/instances.json"),
            )
            try:
                directory = request_json(
                    directory_url,
                    {"Accept": "application/json", "User-Agent": USER_AGENT},
                )
                values = []
                for url, metadata in (directory.get("instances", {}) if isinstance(directory, dict) else {}).items():
                    if not isinstance(metadata, dict):
                        continue
                    if not str(url).startswith("https://"):
                        continue
                    if (metadata.get("http") or {}).get("status_code") != 200:
                        continue
                    values.append(url)
            except Exception as exc:  # pragma: no cover - exercised by live API failures
                warnings.append(f"검색 인스턴스 목록 조회 실패: {type(exc).__name__}")
                return []

    return normalize_search_urls(values)


def normalize_search_urls(values: list[object]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        normalized = canonical_url(str(value).strip()).rstrip("/")
        if normalized and normalized not in resolved:
            resolved.append(normalized)
    return resolved


def collect_x(
    config: dict,
    cutoff: dt.datetime,
    raw: list[dict],
    seen_urls: set[str],
    existing_urls: set[str],
    existing_hosts: set[str],
    known_domains: set[str],
    warnings: list[str],
) -> None:
    token = os.environ.get("X_BEARER_TOKEN", "")
    if not token:
        if config.get("x_queries"):
            warnings.append("X 검색 생략: X_BEARER_TOKEN이 등록되지 않았습니다")
        return
    for query in config.get("x_queries", []):
        try:
            params = urllib.parse.urlencode(
                {
                    "query": f"({query}) -is:retweet",
                    "max_results": "100",
                    "tweet.fields": "created_at,entities",
                }
            )
            result = request_json(
                f"https://api.x.com/2/tweets/search/recent?{params}",
                {"Authorization": f"Bearer {token}"},
            )
            for tweet in result.get("data", []) if isinstance(result, dict) else []:
                entities = tweet.get("entities") or {}
                for entity in entities.get("urls", []) if isinstance(entities, dict) else []:
                    expanded = entity.get("expanded_url") or entity.get("url", "")
                    append_raw_candidate(
                        raw,
                        seen_urls,
                        existing_urls,
                        existing_hosts,
                        known_domains,
                        source=f"X external link search: {query}",
                        kind="x-link",
                        url=expanded,
                        title=tweet.get("text", ""),
                        summary=tweet.get("text", ""),
                        published_at=tweet.get("created_at"),
                        cutoff=cutoff,
                    )
        except Exception as exc:  # pragma: no cover - exercised by live API failures
            warnings.append(f"X 후보 발굴 {query!r} 실패: {exc}")


def group_candidates(raw: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for item in raw:
        key = candidate_key(item)
        if not key:
            continue
        group = groups.setdefault(
            key,
            {
                "candidate_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
                "site": item.get("host", ""),
                "url": item.get("url", ""),
                "kind": item.get("kind", "new-domain"),
                "repository": item.get("repository", ""),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "evidence": [],
            },
        )
        if item.get("kind") == "github-repository":
            group["kind"] = "github-repository"
            group["repository"] = item.get("repository", "")
        if not group.get("summary") and item.get("summary"):
            group["summary"] = item["summary"]
        group["evidence"].append(
            {
                "source": item.get("source", ""),
                "kind": item.get("kind", ""),
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "published_at": item.get("published_at"),
            }
        )
    candidates = list(groups.values())
    for candidate in candidates:
        candidate["evidence_count"] = len(candidate["evidence"])
        candidate["evidence"] = candidate["evidence"][:8]
    candidates.sort(
        key=lambda item: (-item.get("evidence_count", 0), item.get("site", ""), item.get("url", ""))
    )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="research/source-discovery.json")
    parser.add_argument("--sources", default="research/sources.json")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if args.days is not None and args.days <= 0:
        parser.error("--days must be positive when provided")

    discovery_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_config = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    days = args.days or int(discovery_config.get("days", 30))
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=days)
    warnings: list[str] = []
    raw: list[dict] = []
    seen_urls: set[str] = set()
    collection_stats: dict = {
        "configured_query_counts": {
            "github_repository": len(discovery_config.get("github_repository_queries", [])),
            "search_backend": len(discovery_config.get("search_queries", [])),
            "web": len(discovery_config.get("web_queries", [])),
            "x": len(discovery_config.get("x_queries", [])),
        }
    }
    known_domains = configured_domains(source_config)
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    existing_urls, existing_hosts = existing_source_candidates(
        repo, github_headers(os.environ.get("GITHUB_TOKEN", "")), warnings
    )

    collect_github_repositories(
        discovery_config,
        os.environ.get("GITHUB_TOKEN", ""),
        cutoff,
        raw,
        seen_urls,
        existing_urls,
        existing_hosts,
        known_domains,
        warnings,
    )
    collect_brave(
        discovery_config,
        cutoff,
        raw,
        seen_urls,
        existing_urls,
        existing_hosts,
        known_domains,
        warnings,
    )
    collect_search_backend(
        discovery_config,
        cutoff,
        raw,
        seen_urls,
        existing_urls,
        existing_hosts,
        known_domains,
        warnings,
        collection_stats=collection_stats,
    )
    collect_x(
        discovery_config,
        cutoff,
        raw,
        seen_urls,
        existing_urls,
        existing_hosts,
        known_domains,
        warnings,
    )

    candidates = group_candidates(raw)
    result = {
        "generated_at": now.isoformat(),
        "cutoff": cutoff.isoformat(),
        "known_domain_count": len(known_domains),
        "raw_candidate_count": len(raw),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "warnings": warnings,
        "collection_stats": collection_stats,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        print(encoded, end="")
    else:
        Path(args.output).write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
