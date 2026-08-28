#!/usr/bin/env python3
"""Discover recent DGX Spark research candidates from configured sources.

The collector deliberately stores links and short metadata, not copied
articles. GitHub and RSS work without extra provider credentials. Brave Search
and the X API are optional and become warnings when their secrets are absent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
import xml.etree.ElementTree as ET
from pathlib import Path


USER_AGENT = "oh-my-dgx-spark-research-bot/1.0 (+https://github.com/recrack/oh-my-dgx-spark)"
URL_PATTERN = re.compile(r"https?://[^\s)<>\]]+")
PARTIAL_URL_INDEX_MARKER = "<!-- generated-research-url-index: partial -->"
GENERATED_CANDIDATE_COMMENT_MARKER = "<!-- generated-research-candidates-part:"
TRACKING_KEYS = {
    "fbclid",
    "gclid",
    "ref",
    "ref_src",
    "utm_campaign",
    "utm_medium",
    "utm_source",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_date(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def canonical_url(value: str) -> str:
    value = html.unescape(value.strip())
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [(key, item) for key, item in query if key.lower() not in TRACKING_KEYS]
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, urllib.parse.urlencode(query), "")
    )


def clean_text(value: str | None, limit: int = 700) -> str:
    if not value:
        return ""
    value = html.unescape(value)
    value = re.sub(r"<script\b[^>]*>.*?</script>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<style\b[^>]*>.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def request_bytes(url: str, headers: dict[str, str] | None = None, timeout: int = 25) -> bytes:
    request_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        request_headers.update(headers)
    for attempt in range(3):
        request = urllib.request.Request(url, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            try:
                delay = max(1, min(int(retry_after), 30))
            except ValueError:
                delay = 2 ** attempt
            time.sleep(delay)
    raise RuntimeError(f"request retries exhausted: {url}")


def request_json(url: str, headers: dict[str, str], timeout: int = 25) -> object:
    return json.loads(request_bytes(url, headers=headers, timeout=timeout).decode("utf-8"))


def github_headers(token: str) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def search_headers() -> dict[str, str]:
    """Build headers for the configured JSON search backend.

    ``SEARCH_API_KEY`` is sent only when an explicit search endpoint is
    configured; it is never sent to a public fallback instance.
    """

    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    token = os.environ.get("SEARCH_API_KEY", "")
    endpoint_configured = any(
        os.environ.get(name, "")
        for name in (
            "SEARCH_URL",
            "SEARCH_URLS",
        )
    )
    if token and endpoint_configured:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def add_candidate(
    candidates: list[dict],
    seen: set[str],
    *,
    source: str,
    kind: str,
    title: str,
    url: str,
    summary: str = "",
    published_at: str | None = None,
    cutoff: dt.datetime,
) -> None:
    normalized = canonical_url(url)
    if not normalized or normalized in seen:
        return
    published = parse_date(published_at)
    if published is not None and published < cutoff:
        return
    seen.add(normalized)
    candidates.append(
        {
            "source": source,
            "kind": kind,
            "title": clean_text(title, 240) or normalized,
            "url": normalized,
            "summary": clean_text(summary),
            "published_at": published.isoformat() if published else None,
        }
    )


def add_text_urls(text: str, found: set[str]) -> None:
    for link in URL_PATTERN.findall(text):
        normalized = canonical_url(link.rstrip(".,"))
        if normalized:
            found.add(normalized)


def add_generated_comment_urls(
    issue: dict,
    repo: str,
    headers: dict[str, str],
    found: set[str],
    warnings: list[str],
) -> None:
    """Read overflow candidate URLs only when the Issue index is partial."""

    comments_url = str(issue.get("comments_url") or "")
    issue_number = issue.get("number")
    if not comments_url and issue_number:
        comments_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    if not comments_url:
        return

    try:
        separator = "&" if "?" in comments_url else "?"
        for page in range(1, 11):
            url = f"{comments_url}{separator}per_page=100&page={page}"
            comments = json.loads(request_bytes(url, headers=headers).decode("utf-8"))
            if not isinstance(comments, list) or not comments:
                break
            for comment in comments:
                if not isinstance(comment, dict):
                    continue
                body = str(comment.get("body") or "")
                if GENERATED_CANDIDATE_COMMENT_MARKER in body:
                    add_text_urls(body, found)
            if len(comments) < 100:
                break
    except Exception as exc:
        identifier = f"#{issue_number}" if issue_number else "unknown Issue"
        warnings.append(
            f"GitHub candidate comment deduplication failed for {identifier}: "
            f"{type(exc).__name__}"
        )


def existing_issue_urls(repo: str, headers: dict[str, str], warnings: list[str]) -> set[str]:
    if not repo:
        return set()
    found: set[str] = set()
    try:
        for page in range(1, 4):
            url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page=100&page={page}"
            items = json.loads(request_bytes(url, headers=headers).decode("utf-8"))
            if not isinstance(items, list) or not items:
                break
            for issue in items:
                labels = {
                    str(label.get("name", ""))
                    for label in issue.get("labels", [])
                    if isinstance(label, dict)
                }
                # A source-candidate Issue is an approval queue, not a
                # research result. Its URLs become eligible again after a
                # maintainer promotes them into sources.json.
                if "source-candidate" in labels:
                    continue
                body = str(issue.get("body") or "")
                text = f"{issue.get('title', '')}\n{body}"
                add_text_urls(text, found)
                if PARTIAL_URL_INDEX_MARKER in body:
                    add_generated_comment_urls(issue, repo, headers, found, warnings)
            if len(items) < 100:
                break
    except Exception as exc:
        warnings.append(f"GitHub issue deduplication failed: {exc}")
    return found


def collect_github(
    config: dict,
    token: str,
    cutoff: dt.datetime,
    candidates: list[dict],
    seen: set[str],
    existing: set[str],
    warnings: list[str],
) -> None:
    headers = github_headers(token)
    for repository in config.get("github_repositories", []):
        try:
            releases = request_json(
                f"https://api.github.com/repos/{repository}/releases?per_page=10", headers
            )
            for release in releases if isinstance(releases, list) else []:
                url = release.get("html_url", "")
                if canonical_url(url) in existing:
                    continue
                add_candidate(
                    candidates,
                    seen,
                    source=f"GitHub release: {repository}",
                    kind="github-release",
                    title=release.get("name") or release.get("tag_name") or repository,
                    url=url,
                    summary=release.get("body", ""),
                    published_at=release.get("published_at") or release.get("created_at"),
                    cutoff=cutoff,
                )

            issues = request_json(
                f"https://api.github.com/repos/{repository}/issues?state=all&sort=updated&direction=desc&per_page=10",
                headers,
            )
            for issue in issues if isinstance(issues, list) else []:
                url = issue.get("html_url", "")
                if issue.get("pull_request") or canonical_url(url) in existing:
                    continue
                add_candidate(
                    candidates,
                    seen,
                    source=f"GitHub issue: {repository}",
                    kind="github-issue",
                    title=issue.get("title", ""),
                    url=url,
                    summary=issue.get("body", ""),
                    published_at=issue.get("updated_at"),
                    cutoff=cutoff,
                )
        except Exception as exc:
            warnings.append(f"GitHub repository {repository} failed: {exc}")

    for query in config.get("github_queries", []):
        try:
            encoded = urllib.parse.urlencode({"q": f"{query} sort:updated", "per_page": "10"})
            result = request_json(f"https://api.github.com/search/issues?{encoded}", headers)
            for item in result.get("items", []):
                url = item.get("html_url", "")
                if canonical_url(url) in existing:
                    continue
                add_candidate(
                    candidates,
                    seen,
                    source=f"GitHub search: {query}",
                    kind="github-search",
                    title=item.get("title", ""),
                    url=url,
                    summary=item.get("body", ""),
                    published_at=item.get("updated_at"),
                    cutoff=cutoff,
                )
        except Exception as exc:
            warnings.append(f"GitHub search {query!r} failed: {exc}")


def rss_value(element: ET.Element, names: tuple[str, ...]) -> str:
    for name in names:
        child = element.find(name)
        if child is None:
            child = element.find(f"{{*}}{name}")
        if child is not None:
            if name == "link" and child.attrib.get("href"):
                return child.attrib["href"]
            if child.text:
                return child.text
    return ""


def collect_rss(
    config: dict,
    cutoff: dt.datetime,
    candidates: list[dict],
    seen: set[str],
    existing: set[str],
    warnings: list[str],
) -> None:
    for feed in config.get("rss", []):
        name = feed.get("name", feed.get("url", "RSS"))
        try:
            root = ET.fromstring(
                request_bytes(
                    feed["url"],
                    headers={"Accept": "application/rss+xml, application/atom+xml, application/xml"},
                )
            )
            entries = list(root.findall(".//item")) + list(root.findall(".//{*}entry"))
            for entry in entries:
                link = rss_value(entry, ("link",))
                if canonical_url(link) in existing:
                    continue
                add_candidate(
                    candidates,
                    seen,
                    source=name,
                    kind="rss",
                    title=rss_value(entry, ("title",)),
                    url=link,
                    summary=rss_value(entry, ("description", "summary", "content")),
                    published_at=rss_value(entry, ("pubDate", "published", "updated", "date")),
                    cutoff=cutoff,
                )
        except Exception as exc:
            warnings.append(f"RSS {name} failed: {exc}")


def html_metadata(payload: bytes, fallback_title: str, url: str) -> tuple[str, str]:
    text = payload.decode("utf-8", errors="replace")
    title_match = re.search(r"<title\b[^>]*>(.*?)</title>", text, flags=re.I | re.S)
    description_match = re.search(
        r'<meta\b[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\'](.*?)["\']',
        text,
        flags=re.I | re.S,
    )
    title = clean_text(html.unescape(title_match.group(1)), 240) if title_match else ""
    summary = (
        clean_text(html.unescape(description_match.group(1)), 700)
        if description_match
        else ""
    )
    return title or clean_text(fallback_title, 240) or url, summary


def collect_approved_sources(
    config: dict,
    cutoff: dt.datetime,
    candidates: list[dict],
    seen: set[str],
    existing: set[str],
    warnings: list[str],
) -> None:
    """Collect one-off URLs explicitly approved from a source-candidate Issue."""

    for source in config.get("approved_sources", []):
        if not isinstance(source, dict):
            warnings.append("approved_sources contains a non-object entry")
            continue
        url = canonical_url(str(source.get("url", "")))
        if not url or url in existing:
            continue
        try:
            payload = request_bytes(
                url,
                headers={"Accept": "text/html, application/xhtml+xml, text/plain"},
            )
            title, summary = html_metadata(payload, str(source.get("name", "")), url)
            add_candidate(
                candidates,
                seen,
                source=f"Approved source: {source.get('name', url)}",
                kind=str(source.get("kind", "approved-source")),
                title=title,
                url=url,
                summary=summary,
                cutoff=cutoff,
            )
        except Exception as exc:
            warnings.append(f"Approved source {url} failed: {type(exc).__name__}")


def collect_brave(
    config: dict,
    cutoff: dt.datetime,
    candidates: list[dict],
    seen: set[str],
    existing: set[str],
    warnings: list[str],
) -> None:
    token = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not token:
        if config.get("web_search"):
            warnings.append("Brave Search skipped: BRAVE_SEARCH_API_KEY is not configured")
        return
    for item in config.get("web_search", []):
        try:
            query = item["query"]
            if item.get("domains"):
                domains = " OR ".join(f"site:{domain}" for domain in item["domains"])
                query += f" ({domains})"
            params = urllib.parse.urlencode({"q": query, "count": "10", "freshness": "pw"})
            url = "https://api.search.brave.com/res/v1/web/search?" + params
            result = request_json(
                url,
                {"Accept": "application/json", "X-Subscription-Token": token},
            )
            for item_result in (result.get("web") or {}).get("results", []):
                candidate_url = item_result.get("url", "")
                if canonical_url(candidate_url) in existing:
                    continue
                add_candidate(
                    candidates,
                    seen,
                    source=f"Brave Search: {item['query']}",
                    kind="web-search",
                    title=item_result.get("title", ""),
                    url=candidate_url,
                    summary=item_result.get("description", ""),
                    cutoff=cutoff,
                )
        except Exception as exc:
            warnings.append(f"Brave Search {item.get('query', '')!r} failed: {exc}")


def collect_x(
    config: dict,
    cutoff: dt.datetime,
    candidates: list[dict],
    seen: set[str],
    existing: set[str],
    warnings: list[str],
) -> None:
    token = os.environ.get("X_BEARER_TOKEN", "")
    if not token:
        if config.get("x_search"):
            warnings.append("X search skipped: X_BEARER_TOKEN is not configured")
        return
    for query in config.get("x_search", []):
        try:
            params = urllib.parse.urlencode(
                {
                    # The configured list includes Korean queries as well as
                    # English ones.  Do not force lang:en here.
                    "query": f"({query}) -is:retweet",
                    "max_results": "10",
                    "tweet.fields": "created_at,entities,author_id",
                }
            )
            result = request_json(
                f"https://api.x.com/2/tweets/search/recent?{params}",
                {"Authorization": f"Bearer {token}"},
            )
            for tweet in result.get("data", []):
                tweet_url = f"https://x.com/i/web/status/{tweet.get('id', '')}"
                if canonical_url(tweet_url) in existing:
                    continue
                add_candidate(
                    candidates,
                    seen,
                    source=f"X search: {query}",
                    kind="x-post",
                    title=clean_text(tweet.get("text", ""), 160),
                    url=tweet_url,
                    summary=tweet.get("text", ""),
                    published_at=tweet.get("created_at"),
                    cutoff=cutoff,
                )
        except Exception as exc:
            warnings.append(f"X search {query!r} failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="research/sources.json")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="optional safety cap; omit it to keep every discovered candidate",
    )
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if args.days <= 0 or (args.max_candidates is not None and args.max_candidates <= 0):
        parser.error("--days and --max-candidates must be positive when provided")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    now = utc_now()
    cutoff = now - dt.timedelta(days=args.days)
    warnings: list[str] = []
    candidates: list[dict] = []
    seen: set[str] = set()
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    existing = existing_issue_urls(repo, github_headers(token), warnings)

    collect_github(config, token, cutoff, candidates, seen, existing, warnings)
    collect_rss(config, cutoff, candidates, seen, existing, warnings)
    collect_approved_sources(config, cutoff, candidates, seen, existing, warnings)
    collect_brave(config, cutoff, candidates, seen, existing, warnings)
    collect_x(config, cutoff, candidates, seen, existing, warnings)

    candidates.sort(key=lambda item: item.get("published_at") or "", reverse=True)
    selected_candidates = (
        candidates
        if args.max_candidates is None
        else candidates[: args.max_candidates]
    )
    result = {
        "generated_at": now.isoformat(),
        "cutoff": cutoff.isoformat(),
        "candidate_count": len(selected_candidates),
        "total_candidate_count": len(candidates),
        "candidate_limit": args.max_candidates,
        "candidates": selected_candidates,
        "warnings": warnings,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(encoded)
    else:
        Path(args.output).write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
