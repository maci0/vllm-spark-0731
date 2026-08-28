#!/usr/bin/env python3
"""Validate the WikiDocs publish workflow's traceable commit message contract."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/publish-wikidocs.yml"


def main() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required_fragments = (
        "branches:",
        "- main",
        "workflow_dispatch:",
        "WIKIDOCS_DEPLOY_TOKEN",
        "source_subject=\"$(git log -1 --format=%s \"${GITHUB_SHA}\")\"",
        "commit_subject=\"docs: sync WikiDocs — ${source_subject}\"",
        "git diff --cached --stat",
        "git diff --cached --name-status",
        "git commit -m \"${commit_subject}\" -m \"${commit_body}\"",
    )
    for fragment in required_fragments:
        if fragment not in text:
            raise SystemExit(f"WikiDocs publish workflow is missing: {fragment}")
    if "chore: sync WikiDocs bundle from ${short_sha}" in text:
        raise SystemExit("WikiDocs publish workflow still uses the opaque short-SHA message")
    print("WikiDocs publish commit message wiring OK")


if __name__ == "__main__":
    main()
