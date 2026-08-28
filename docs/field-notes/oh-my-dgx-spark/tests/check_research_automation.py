#!/usr/bin/env python3
"""Validate the repository's low-risk research automation wiring."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from research_candidate_count import candidate_count  # noqa: E402


def main() -> None:
    workflow = ROOT / ".github/workflows/research.yml"
    template = ROOT / ".github/ISSUE_TEMPLATE/research.md"
    sources = ROOT / "research/sources.json"
    prompt = ROOT / "prompts/research-issue.md"
    report_template = ROOT / "templates/research-issue-report.md"
    issue_renderer = ROOT / "tools/render_research_issue.py"
    report_renderer = ROOT / "tools/render_research_report.py"
    report_checker = ROOT / "tools/check_research_report.py"
    report_normalizer = ROOT / "tools/normalize_research_report.py"
    candidate_counter = ROOT / "tools/research_candidate_count.py"
    discovery = ROOT / "tools/research_discover.py"
    korean_style = ROOT / "prompts/fluent-korean.md"
    required_files = (
        workflow,
        template,
        sources,
        prompt,
        report_template,
        issue_renderer,
        report_renderer,
        report_checker,
        report_normalizer,
        candidate_counter,
        discovery,
        korean_style,
    )
    for path in required_files:
        if not path.is_file():
            raise SystemExit(f"missing research automation file: {path}")

    workflow_text = workflow.read_text(encoding="utf-8")
    renderer_text = report_renderer.read_text(encoding="utf-8")
    discovery_text = discovery.read_text(encoding="utf-8")
    required_fragments = (
        "schedule:",
        "workflow_dispatch:",
        "issues:",
        "research-ready",
        "tools/research_discover.py",
        "tools/render_research_issue.py",
        "--comments-dir",
        "research-issue-comments",
        '--body-file "${comment_files[index]}"',
        "@github/copilot",
        '@github/copilot@${COPILOT_CLI_VERSION}',
        'COPILOT_CLI_VERSION: "1.0.80"',
        "COPILOT_GITHUB_TOKEN",
        "RESEARCH_MAX_AI_CREDITS",
        "RESEARCH_MODEL: ${{ vars.RESEARCH_MODEL || 'gpt-5.6-terra' }}",
        '--max-ai-credits "${RESEARCH_MAX_AI_CREDITS}"',
        "Skip duplicate analysis",
        "copilot-research-automation",
        "tools/redact_sensitive_output.py",
        "--in-place",
        "--no-custom-instructions",
        "--no-remote-export",
        "--available-tools=view,edit,create,apply_patch,glob,grep,web_fetch",
        "--allow-tool=read,write,url",
        "--deny-tool=shell",
        "env -u GH_TOKEN -u GITHUB_TOKEN",
        "RESEARCH_MAX_CANDIDATES",
        "id: prepare",
        'echo "research_date=${research_date}" >> "${GITHUB_OUTPUT}"',
        "RESEARCH_DATE: ${{ steps.prepare.outputs.research_date }}",
        "Research date",
        'prompt_file="${RUNNER_TEMP}/research-prompt.md"',
        "Use the view tool to read the complete research task from ${prompt_file}",
        '--add-dir "${RUNNER_TEMP}"',
        'report_sha256="$(sha256sum "${report_path}"',
        'echo "report_sha256=${report_sha256}" >> "${GITHUB_OUTPUT}"',
        "INITIAL_REPORT_SHA256: ${{ steps.prepare.outputs.report_sha256 }}",
        'current_report_sha256="$(sha256sum "${report_path}"',
        "Copilot CLI completed without updating the required research report",
        "expected_doc=\"docs/research-issue-${ISSUE_NUMBER}_${today}.md\"",
        "tools/render_research_report.py",
        "tools/check_research_report.py",
        "tools/normalize_research_report.py",
        "Restore deterministic research metadata",
        "prompts/fluent-korean.md",
        "Remove changes outside research scope",
        "git restore --source=HEAD",
        "gh pr create",
        "research-pr",
        "labels[]=research-pr",
        "git diff --check",
        "git add docs research",
        'pr_body="## 자동 리서치 제안',
        'public_report_path="appendix-b-research-issue-${ISSUE_NUMBER}_${RESEARCH_DATE}.md"',
        "머지 후 WikiDocs 공개 파일",
        'RESEARCH_DATE: ${{ steps.prepare.outputs.research_date }}',
        "--max-candidates",
    )
    if not all(fragment in workflow_text for fragment in required_fragments):
        raise SystemExit("research workflow is missing a required stage")
    if "research-issue-report.md" not in renderer_text:
        raise SystemExit("research report renderer is not wired to the fixed template")
    if "default=40" in discovery_text:
        raise SystemExit("research discovery must not hard-code a 40-candidate limit")
    if "prompts/fluent-korean.md" not in workflow_text:
        raise SystemExit("research workflow is not wired to the Korean writing guide")
    if '-p "$(<"${RUNNER_TEMP}/research-prompt.md")"' in workflow_text:
        raise SystemExit("research prompt must not be expanded into a command-line argument")
    if '--attachment "${prompt_file}"' in workflow_text:
        raise SystemExit("Copilot CLI 1.0.80 does not accept Markdown prompt attachments")
    if "--available-tools=read,write,url" in workflow_text:
        raise SystemExit("Copilot permission kinds must not be used as available tool names")
    if '--body-file <(tail' in workflow_text:
        raise SystemExit("raw Copilot failure output must not be posted to an Issue")
    if "WIKIDOCS_DEPLOY_TOKEN" in workflow_text:
        raise SystemExit("research workflow must not receive the WikiDocs deploy token")
    if "book/*.md|docs/research-issue-*.md" in workflow_text:
        raise SystemExit("research workflow must not allow arbitrary book chapter edits")
    if workflow_text.count("RESEARCH_MODEL: ${{ vars.RESEARCH_MODEL || 'gpt-5.6-terra' }}") != 3:
        raise SystemExit("every research stage must use the fixed Terra fallback")
    for forbidden in (
        "tools/publish_research_page.py",
        "tools/update_research_log.py",
        "book/appendix-b-research-issue-",
        "Publish dated research subchapter source",
        'cmp -s "${expected_doc}" "${expected_public}"',
        'Automated research proposal for #${ISSUE_NUMBER}.\\n\\n',
    ):
        if forbidden in workflow_text:
            raise SystemExit(f"research workflow still contains obsolete duplicate output: {forbidden}")

    config = json.loads(sources.read_text(encoding="utf-8"))
    for key in ("github_repositories", "github_queries", "rss", "web_search"):
        if not config.get(key):
            raise SystemExit(f"research sources contain no {key}")

    candidates = [
        {
            "source": "large-fixture",
            "kind": "web-search",
            "title": f"DGX Spark 대용량 후보 {index}",
            "url": f"https://example.com/research/{index}?model=deepseek-v4-flash",
            "summary": "긴 요약 " + ("가" * 650),
            "published_at": "2026-08-24T00:00:00+00:00",
        }
        for index in range(1, 181)
    ]
    fixture = {
        "generated_at": "2026-08-24T00:00:00+00:00",
        "candidates": candidates,
        "warnings": [("긴 경고 " + ("나" * 300)) for _ in range(80)],
    }
    with TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        input_path = temporary_root / "candidates.json"
        body_path = temporary_root / "issue.md"
        comments_dir = temporary_root / "comments"
        input_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(issue_renderer),
                str(input_path),
                str(body_path),
                "--comments-dir",
                str(comments_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        body = body_path.read_text(encoding="utf-8")
        comment_paths = sorted(comments_dir.glob("part-*.md"))
        comments = [path.read_text(encoding="utf-8") for path in comment_paths]
        if len(body.encode("utf-8")) > 60_000:
            raise SystemExit("rendered research Issue body exceeds the safe size")
        if len(comments) < 2:
            raise SystemExit("large candidate fixture was not split into comments")
        if any(len(comment.encode("utf-8")) > 60_000 for comment in comments):
            raise SystemExit("rendered candidate comment exceeds the safe size")
        if any(
            "<!-- generated-research-candidates-part:" not in comment
            for comment in comments
        ):
            raise SystemExit("rendered candidate comment has no generated marker")

        rendered_candidates = "\n".join(comments)
        heading_numbers = {
            int(number)
            for number in re.findall(r"(?m)^###\s+(\d+)\.\s+", rendered_candidates)
        }
        if heading_numbers != set(range(1, len(candidates) + 1)):
            raise SystemExit("candidate comment split lost or duplicated a candidate")
        if not all(candidate["url"] in rendered_candidates for candidate in candidates):
            raise SystemExit("candidate comment split lost a source URL")

        stale_body = body.replace(
            f"- 후보 수: `{len(candidates)}`", "- 후보 수: `999`"
        )
        issue_comments = [{"body": comment} for comment in comments]
        issue_comments.append({"body": "### 999. 일반 사용자 댓글의 번호 목록"})
        issue = {"body": stale_body, "comments": issue_comments}
        if candidate_count(issue) != str(len(candidates)):
            raise SystemExit("candidate count did not use generated candidate comments")

    print("Research automation wiring OK")


if __name__ == "__main__":
    main()
