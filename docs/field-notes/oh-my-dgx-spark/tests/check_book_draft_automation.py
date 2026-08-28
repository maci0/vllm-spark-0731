#!/usr/bin/env python3
"""Validate the guarded, manual book-drafting workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prepare_book_draft import build_prompt, validate_inputs  # noqa: E402
from redact_sensitive_output import redact  # noqa: E402


def expect_invalid(report: str, book_file: str) -> None:
    try:
        validate_inputs(ROOT, report, book_file)
    except ValueError:
        return
    raise SystemExit(f"unsafe book-draft input was accepted: {report}, {book_file}")


def main() -> None:
    workflow = ROOT / ".github/workflows/book-draft.yml"
    prompt_path = ROOT / "prompts/book-draft.md"
    helper = ROOT / "tools/prepare_book_draft.py"
    redactor = ROOT / "tools/redact_sensitive_output.py"
    for path in (workflow, prompt_path, helper, redactor):
        if not path.is_file():
            raise SystemExit(f"missing book-draft automation file: {path}")

    workflow_text = workflow.read_text(encoding="utf-8")
    required = (
        "workflow_dispatch:",
        "research_report:",
        "book_file:",
        "ref: main",
        "tools/prepare_book_draft.py",
        "gpt-5.6-terra",
        "BOOK_DRAFT_MODEL: ${{ vars.BOOK_DRAFT_MODEL || 'gpt-5.6-terra' }}",
        "BOOK_DRAFT_MAX_AI_CREDITS",
        '--max-ai-credits "${BOOK_DRAFT_MAX_AI_CREDITS}"',
        'COPILOT_CLI_VERSION: "1.0.80"',
        '@github/copilot@${COPILOT_CLI_VERSION}',
        "copilot-research-automation",
        "Skip an overlapping draft",
        "Use the view tool to read the complete book-editing task from ${prompt_file}",
        "--available-tools=view,edit,create,apply_patch,glob,grep",
        "--allow-tool=read,write",
        "--deny-tool=url",
        "--deny-tool=shell",
        "Enforce the one-file boundary",
        "has_change=false",
        "tools/build_wikidocs_export.py",
        "tests/check_wikidocs_export.py",
        "--draft",
        "book-draft",
        "tools/redact_sensitive_output.py",
        "--in-place",
    )
    if not all(fragment in workflow_text for fragment in required):
        raise SystemExit("book-draft workflow is missing a required guard or stage")
    if "schedule:" in workflow_text:
        raise SystemExit("book drafting must remain manual-only")
    if "WIKIDOCS_DEPLOY_TOKEN" in workflow_text:
        raise SystemExit("book drafting must not receive the WikiDocs deploy token")
    if "--allow-tool=read,write,url" in workflow_text:
        raise SystemExit("book drafting must not browse the web")
    if "--available-tools=read,write" in workflow_text:
        raise SystemExit("Copilot permission kinds must not be used as available tool names")
    if '--attachment "${prompt_file}"' in workflow_text:
        raise SystemExit("Copilot CLI 1.0.80 does not accept Markdown prompt attachments")

    report = "docs/research-issue-2_2026-08-23.md"
    book_file = "book/06-5-deepseek-v4-flash.md"
    report_path, chapter_path = validate_inputs(ROOT, report, book_file)
    if report_path != (ROOT / report).resolve() or chapter_path != (ROOT / book_file).resolve():
        raise SystemExit("valid book-draft paths did not resolve correctly")
    built_prompt = build_prompt(ROOT, report, book_file)
    for fragment in (report, book_file, "공통 한국어 문체", "다른 파일은 수정하지 않는다"):
        if fragment not in built_prompt:
            raise SystemExit(f"book-draft prompt is missing: {fragment}")

    for bad_report, bad_book in (
        ("../docs/research-issue-2_2026-08-23.md", book_file),
        ("docs/research-issue-2_2026-02-30.md", book_file),
        (report, "book/TOC.md"),
        (report, "book/nested/chapter.md"),
        (report, "/tmp/chapter.md"),
    ):
        expect_invalid(bad_report, bad_book)

    old_token = os.environ.get("COPILOT_GITHUB_TOKEN")
    try:
        os.environ["COPILOT_GITHUB_TOKEN"] = "unit-test-secret-92831"
        redacted = redact(
            "unit-test-secret-92831\n"
            "Authorization: Bearer secret-value\n"
            "https://search.example/search?api_key=private-key&q=test"
        )
    finally:
        if old_token is None:
            os.environ.pop("COPILOT_GITHUB_TOKEN", None)
        else:
            os.environ["COPILOT_GITHUB_TOKEN"] = old_token
    for leaked in ("unit-test-secret-92831", "secret-value", "private-key"):
        if leaked in redacted:
            raise SystemExit("diagnostic redaction leaked a credential pattern")

    with TemporaryDirectory() as directory:
        diagnostic = Path(directory) / "copilot-output.txt"
        diagnostic.write_text("Authorization: Bearer secret-value", encoding="utf-8")
        subprocess.run(
            [sys.executable, str(redactor), str(diagnostic), "--in-place"],
            check=True,
        )
        if "secret-value" in diagnostic.read_text(encoding="utf-8"):
            raise SystemExit("in-place diagnostic redaction leaked a credential")

    print("Book draft automation wiring OK")


if __name__ == "__main__":
    main()
