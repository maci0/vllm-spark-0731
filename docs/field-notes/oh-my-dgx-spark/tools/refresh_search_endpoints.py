#!/usr/bin/env python3
"""Refresh the last-known-good public search endpoint cache.

The directory is only a candidate registry.  An endpoint is accepted only
after a real ``/search?...&format=json`` request succeeds.  The committed
cache is deliberately stable: timestamps and transient warnings are kept out
of the file so a daily refresh does not create a PR when the endpoint set has
not changed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from research_discover import USER_AGENT, canonical_url, request_json, search_headers, utc_now


DEFAULT_DIRECTORY_URL = "https://searx.space/data/instances.json"
DEFAULT_CACHE_PATH = "research/search-endpoints.json"
DEFAULT_TARGET_COUNT = 5
DEFAULT_PROBE_LIMIT = 20
DEFAULT_PROBE_TIMEOUT = 12


def explicit_endpoint_configured() -> bool:
    return any(
        os.environ.get(name, "")
        for name in (
            "SEARCH_URL",
            "SEARCH_URLS",
        )
    )


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def endpoint_url(value: object) -> str:
    normalized = canonical_url(str(value or "")).rstrip("/")
    if not normalized.startswith("https://"):
        return ""
    return normalized


def split_urls(value: str) -> list[str]:
    result: list[str] = []
    for item in value.split(","):
        normalized = endpoint_url(item.strip())
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def configured_urls(config: dict) -> list[str]:
    many = os.environ.get("SEARCH_URLS", "")
    if many:
        return split_urls(many)
    one = os.environ.get("SEARCH_URL", "")
    if one:
        return split_urls(one)
    result: list[str] = []
    for item in config.get("search_urls", []):
        normalized = endpoint_url(item)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def directory_url(config: dict) -> str:
    return os.environ.get("SEARCH_DIRECTORY_URL", "") or config.get(
        "search_directory_url", DEFAULT_DIRECTORY_URL
    )


def directory_candidates(directory: object) -> tuple[list[dict], object]:
    if not isinstance(directory, dict):
        raise ValueError("검색 인스턴스 목록이 JSON 객체가 아닙니다")
    instances = directory.get("instances")
    if not isinstance(instances, dict):
        raise ValueError("검색 인스턴스 목록에 instances가 없습니다")

    candidates: list[dict] = []
    for raw_url, metadata in instances.items():
        url = endpoint_url(raw_url)
        if not url or not isinstance(metadata, dict):
            continue
        http = metadata.get("http") or {}
        if as_int(http.get("status_code")) != 200 or http.get("error"):
            continue
        timing = metadata.get("timing") or {}
        search_timing = timing.get("search") or {}
        uptime = metadata.get("uptime") or {}
        success = as_float(search_timing.get("success_percentage"), -1)
        week = as_float(uptime.get("uptimeWeek"), -1)
        month = as_float(uptime.get("uptimeMonth"), -1)
        latency = as_float((search_timing.get("all") or {}).get("median"), 9999)
        # The directory metadata is only a ranking hint.  The real probe
        # below remains the acceptance test.
        score = as_float(http.get("score"))
        if success >= 0:
            score += success
        if week >= 0:
            score += week / 10
        if month >= 0:
            score += month / 20
        score -= min(latency, 30) / 10
        candidates.append(
            {
                "url": url,
                "score": round(score, 3),
                "directory_status": as_int(http.get("status_code")),
                "directory_grade": str(http.get("grade") or ""),
                "uptime_week": round(week, 3) if week >= 0 else None,
                "uptime_month": round(month, 3) if month >= 0 else None,
                "search_success_percentage": round(success, 3) if success >= 0 else None,
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["url"]))
    return candidates, directory.get("metadata", {}).get("timestamp") if isinstance(directory.get("metadata"), dict) else None


def load_cache(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "endpoints": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "endpoints": []}
    if not isinstance(value, dict):
        return {"schema_version": 1, "endpoints": []}
    endpoints = []
    for item in value.get("endpoints", []):
        if not isinstance(item, dict):
            continue
        url = endpoint_url(item.get("url"))
        if url:
            endpoints.append({**item, "url": url})
    return {**value, "schema_version": 1, "endpoints": endpoints}


def cached_urls(cache: dict) -> list[str]:
    result: list[str] = []
    for item in cache.get("endpoints", []):
        url = endpoint_url(item.get("url")) if isinstance(item, dict) else ""
        if url and url not in result:
            result.append(url)
    return result


def probe_endpoint(url: str, query: str, timeout: int) -> dict:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "all",
            "time_range": "month",
        }
    )
    started = time.monotonic()
    request = urllib.request.Request(
        f"{url}/search?{params}",
        headers=search_headers(),
    )
    # Health checks must be cheap and independent.  The normal research
    # collector retries transient API failures, but retrying every public
    # instance here would make a daily sweep unnecessarily slow.
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        raise ValueError("검색 응답이 JSON results 배열이 아닙니다")
    return {"latency_ms": elapsed_ms, "result_count": len(result["results"])}


def stable_cache(
    previous: dict,
    selected: list[dict],
    *,
    now: dt.datetime,
    directory: str,
    directory_timestamp: object,
    candidate_count: int,
    probed_count: int,
    warnings: list[str],
) -> dict:
    previous_urls = cached_urls(previous)
    selected_urls = [item["url"] for item in selected]
    # Do not rewrite timestamps or transient warning text when the actual
    # endpoint set is unchanged.  This keeps the daily refresh PR-free.
    if selected_urls == previous_urls and previous.get("endpoints"):
        return previous
    return {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "directory_url": directory,
        "directory_timestamp": directory_timestamp,
        "candidate_count": candidate_count,
        "probed_count": probed_count,
        # Per-instance failures are emitted in the Action log.  Persisting
        # all transient 403/429/timeout messages would create noisy cache PRs.
        "warnings": [],
        "endpoints": selected,
    }


def refresh(config: dict, cache_path: Path, output_path: Path) -> tuple[dict, list[str]]:
    previous = load_cache(cache_path)
    warnings: list[str] = []
    now = utc_now()
    private_endpoint = explicit_endpoint_configured()
    same_cache_output = output_path.resolve() == cache_path.resolve()
    explicit = configured_urls(config)
    directory = directory_url(config)
    candidates: list[dict] = []
    directory_timestamp: object = None

    if explicit:
        candidates = [{"url": url, "score": 1000.0, "source": "configured"} for url in explicit]
    else:
        try:
            metadata = request_json(
                directory,
                {"Accept": "application/json", "User-Agent": USER_AGENT},
                timeout=20,
            )
            candidates, directory_timestamp = directory_candidates(metadata)
        except Exception as exc:
            warnings.append(f"검색 인스턴스 목록 조회 실패: {type(exc).__name__}")

    candidate_by_url = {item["url"]: item for item in candidates}
    # Probe the current directory ranking first. Cached addresses are then
    # appended as a last-known-good fallback, including addresses temporarily
    # absent from the current directory snapshot. An explicitly configured
    # private endpoint is exclusive: never probe a cached public endpoint with
    # the private endpoint's Authorization header.
    ordered_urls: list[str] = []
    for item in candidates:
        if item["url"] not in ordered_urls:
            ordered_urls.append(item["url"])
    if not private_endpoint:
        for url in cached_urls(previous):
            if url not in ordered_urls:
                ordered_urls.append(url)

    target_count = max(1, as_int(config.get("search_endpoint_count"), DEFAULT_TARGET_COUNT))
    probe_limit = max(1, as_int(config.get("search_endpoint_probe_limit"), DEFAULT_PROBE_LIMIT))
    timeout = max(3, as_int(config.get("search_endpoint_probe_timeout"), DEFAULT_PROBE_TIMEOUT))
    query = str(config.get("search_endpoint_probe_query") or "DGX Spark")
    selected: list[dict] = []
    attempted_count = 0
    for index, url in enumerate(ordered_urls):
        # Normally stop after the probe budget has produced enough healthy
        # endpoints.  If the first batch is entirely blocked, continue
        # through the remainder of the directory so the initial bootstrap can
        # still find a working address.
        if index >= probe_limit and len(selected) >= target_count:
            break
        attempted_count += 1
        try:
            result = probe_endpoint(url, query, timeout)
        except Exception as exc:
            warnings.append(f"검색 인스턴스 {index + 1} 제외: {type(exc).__name__}")
            continue
        metadata = candidate_by_url.get(url, {})
        selected.append(
            {
                "url": url,
                "last_success_at": now.isoformat(),
                "latency_ms": result["latency_ms"],
                "result_count": result["result_count"],
                "directory_score": metadata.get("score"),
                "directory_grade": metadata.get("directory_grade", ""),
            }
        )
        if len(selected) >= target_count:
            break

    if not selected:
        if previous.get("endpoints") and not private_endpoint:
            warnings.append("정상 검색 인스턴스가 없어 마지막 성공 캐시를 유지합니다")
            if not same_cache_output:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(previous, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return previous, warnings
        if private_endpoint and previous.get("endpoints"):
            warnings.append("명시한 검색 인스턴스가 응답하지 않아 공개 캐시를 사용하지 않습니다")
        warnings.append("정상 검색 인스턴스를 찾지 못했습니다")
        empty = {
            "schema_version": 1,
            "generated_at": now.isoformat(),
            "directory_url": directory,
            "directory_timestamp": directory_timestamp,
            "candidate_count": len(candidates),
            "probed_count": attempted_count,
            "warnings": warnings,
            "endpoints": [],
        }
        # A failed bootstrap must not create an empty committed cache.  A
        # temporary output is still useful to the caller because the caller
        # can then fall back to the live directory for that run.
        if not same_cache_output:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(empty, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return empty, warnings

    # If fewer than the target number work, keep old entries as fallback so a
    # transient public outage does not erase the only usable address. A
    # private endpoint must never be supplemented with public cache entries.
    selected_urls = {item["url"] for item in selected}
    if not private_endpoint and len(selected) < target_count:
        for old in previous.get("endpoints", []):
            if not isinstance(old, dict):
                continue
            url = endpoint_url(old.get("url"))
            if url and url not in selected_urls:
                selected.append(old)
                selected_urls.add(url)
            if len(selected) >= target_count:
                break

    result = stable_cache(
        previous,
        selected[:target_count],
        now=now,
        directory=directory,
        directory_timestamp=directory_timestamp,
        candidate_count=len(candidates),
        probed_count=attempted_count,
        warnings=warnings,
    )
    # A URL supplied through an environment variable is treated as private.
    # It may be written to a runner-temp file for the current job, but never
    # persisted to the repository cache.
    if not (private_endpoint and same_cache_output):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="research/source-discovery.json")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cache_path = Path(args.cache)
    output_path = Path(args.output or args.cache)
    result, warnings = refresh(config, cache_path, output_path)
    print(
        f"검색 엔드포인트 갱신: {len(result.get('endpoints', []))}개 사용 가능, "
        f"후보 {result.get('candidate_count', 0)}개, 경고 {len(warnings)}개"
    )
    for warning in warnings[:10]:
        print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
