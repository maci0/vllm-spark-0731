#!/usr/bin/env python3
"""Validate endpoint refresh, cache fallback, and PR wiring."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import discover_source_candidates as source_discovery  # noqa: E402
import refresh_search_endpoints as endpoint_refresh  # noqa: E402


def main() -> None:
    workflow = ROOT / ".github/workflows/refresh-search-endpoints.yml"
    discovery_workflow = ROOT / ".github/workflows/research-source-discovery.yml"
    script = ROOT / "tools/refresh_search_endpoints.py"
    config_path = ROOT / "research/source-discovery.json"
    for path in (workflow, discovery_workflow, script, config_path):
        if not path.is_file():
            raise SystemExit(f"missing endpoint refresh file: {path}")

    workflow_text = workflow.read_text(encoding="utf-8")
    for fragment in (
        "schedule:",
        'cron: "5 1 * * 1-5"',
        "workflow_dispatch:",
        "research/search-endpoints.json",
        "tools/refresh_search_endpoints.py",
        "SEARCH_API_KEY",
        "git status --short --untracked-files=all",
        "gh pr create",
        "research-infrastructure",
        "이번에 반영될 검색 주소",
    ):
        if fragment not in workflow_text:
            raise SystemExit(f"endpoint refresh workflow is missing: {fragment}")
    if "git push origin main" in workflow_text:
        raise SystemExit("endpoint refresh must not push directly to main")
    discovery_text = discovery_workflow.read_text(encoding="utf-8")
    for fragment in (
        "Refresh search endpoints for this run",
        "SEARCH_ENDPOINTS_CACHE",
        "steps.endpoints.outputs.cache_file",
    ):
        if fragment not in discovery_text:
            raise SystemExit(f"source discovery is not using endpoint cache: {fragment}")

    directory = {
        "metadata": {"timestamp": 1787440411},
        "instances": {
            "https://healthy.example/": {
                "http": {"status_code": 200, "score": 110, "grade": "A+"},
                "timing": {"search": {"success_percentage": 100, "all": {"median": 0.2}}},
                "uptime": {"uptimeWeek": 100, "uptimeMonth": 100},
            },
            "https://html.example/": {
                "http": {"status_code": 200, "score": 110, "grade": "A+"},
            },
            "https://offline.example/": {
                "http": {"status_code": 503, "score": 0},
            },
        },
    }

    original_request_json = endpoint_refresh.request_json
    original_urlopen = endpoint_refresh.urllib.request.urlopen
    endpoint_env_names = (
        "SEARCH_URL",
        "SEARCH_URLS",
    )
    original_endpoint_env = {name: os.environ.get(name) for name in endpoint_env_names}
    original_api_key = os.environ.get("SEARCH_API_KEY")
    for name in endpoint_env_names:
        os.environ.pop(name, None)

    def fake_request(url: str, headers: dict[str, str], timeout: int = 25) -> object:
        if url == "https://directory.example/instances.json":
            return directory
        if url.startswith("https://healthy.example/search?"):
            return {"results": [{"url": "https://example.org/post"}]}
        if url.startswith("https://html.example/search?"):
            raise ValueError("not JSON")
        raise RuntimeError(f"unexpected endpoint: {url}")

    endpoint_refresh.request_json = fake_request
    os.environ["SEARCH_API_KEY"] = "test-secret"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"results": [{"url": "https://example.org/post"}]}).encode()

    class EmptyResponse(FakeResponse):
        def read(self):
            return json.dumps({"results": []}).encode()

    def fake_urlopen(request, timeout=25):
        if request.full_url.startswith("https://healthy.example/search?"):
            if request.headers.get("Authorization"):
                raise RuntimeError("credential was forwarded to public endpoint")
            return FakeResponse()
        if request.full_url.startswith("https://private.example/search?"):
            if request.headers.get("Authorization") != "Bearer test-secret":
                raise RuntimeError("Bearer header was not forwarded")
            return FakeResponse()
        if request.full_url.startswith("https://empty.example/search?"):
            if request.headers.get("Authorization"):
                raise RuntimeError("credential was forwarded to public endpoint")
            return EmptyResponse()
        if request.full_url.startswith("https://html.example/search?"):
            raise ValueError("not JSON")
        raise RuntimeError(f"unexpected endpoint: {request.full_url}")

    endpoint_refresh.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            cache = temp / "search-endpoints.json"
            config = {
                "search_directory_url": "https://directory.example/instances.json",
                "search_endpoint_count": 2,
                "search_endpoint_probe_limit": 2,
                "search_endpoint_probe_timeout": 3,
                "search_endpoint_probe_query": "DGX Spark",
                "search_urls": [],
            }
            result, warnings = endpoint_refresh.refresh(config, cache, cache)
            if [item["url"] for item in result["endpoints"]] != ["https://healthy.example"]:
                raise SystemExit("healthy endpoint was not cached")
            if not any("검색 인스턴스" in warning for warning in warnings):
                raise SystemExit("failed endpoint was not reported")
            if any("html.example" in warning or "test-secret" in warning for warning in warnings):
                raise SystemExit("private endpoint data leaked into warnings")
            empty_probe = endpoint_refresh.probe_endpoint("https://empty.example", "DGX Spark", 3)
            if empty_probe["result_count"] != 0:
                raise SystemExit("empty but valid JSON response was not counted correctly")

            first_content = cache.read_text(encoding="utf-8")
            endpoint_refresh.request_json = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("directory unavailable")
            )
            endpoint_refresh.urllib.request.urlopen = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("cached endpoint unavailable")
            )
            fallback_output = temp / "fallback-search-endpoints.json"
            fallback, fallback_warnings = endpoint_refresh.refresh(config, cache, fallback_output)
            if [item["url"] for item in fallback["endpoints"]] != ["https://healthy.example"]:
                raise SystemExit("last-known-good cache was not preserved")
            if cache.read_text(encoding="utf-8") != first_content:
                raise SystemExit("fallback rewrote the stable cache")
            if fallback_output.read_text(encoding="utf-8") != first_content:
                raise SystemExit("temporary fallback cache was not emitted")
            if not fallback_warnings:
                raise SystemExit("fallback failure was not reported")

            cache_config = {"search_endpoints_cache": str(cache), "search_urls": []}
            old_env = os.environ.pop("SEARCH_ENDPOINTS_CACHE", None)
            try:
                urls = source_discovery.search_urls(cache_config, [])
            finally:
                if old_env is not None:
                    os.environ["SEARCH_ENDPOINTS_CACHE"] = old_env
            if urls != ["https://healthy.example"]:
                raise SystemExit("source discovery did not read endpoint cache")

            os.environ["SEARCH_URL"] = "https://private.example"
            private_cache = temp / "private-search-endpoints.json"
            private_cache.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "endpoints": [{"url": "https://cached-public.example"}],
                    }
                ),
                encoding="utf-8",
            )
            private_cache_before = private_cache.read_text(encoding="utf-8")
            endpoint_refresh.urllib.request.urlopen = fake_urlopen
            private_result, _ = endpoint_refresh.refresh(config, private_cache, private_cache)
            if [item["url"] for item in private_result["endpoints"]] != ["https://private.example"]:
                raise SystemExit("private endpoint was supplemented with a public cache entry")
            if private_cache.read_text(encoding="utf-8") != private_cache_before:
                raise SystemExit("private endpoint path rewrote the repository cache")

            endpoint_refresh.urllib.request.urlopen = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("private endpoint unavailable")
            )
            failed_private, _ = endpoint_refresh.refresh(config, private_cache, private_cache)
            if failed_private["endpoints"]:
                raise SystemExit("private endpoint failure fell back to a public cache entry")
            if private_cache.read_text(encoding="utf-8") != private_cache_before:
                raise SystemExit("private endpoint failure rewrote the repository cache")
            new_private_cache = temp / "new-private-search-endpoints.json"
            endpoint_refresh.refresh(config, new_private_cache, new_private_cache)
            if new_private_cache.exists():
                raise SystemExit("private endpoint was persisted to a new repository cache")
    finally:
        endpoint_refresh.request_json = original_request_json
        endpoint_refresh.urllib.request.urlopen = original_urlopen
        if original_api_key is None:
            os.environ.pop("SEARCH_API_KEY", None)
        else:
            os.environ["SEARCH_API_KEY"] = original_api_key
        for name, value in original_endpoint_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print("Search endpoint refresh wiring OK")


if __name__ == "__main__":
    main()
