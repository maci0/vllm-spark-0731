#!/usr/bin/env python3
"""Create the deterministic shell for one dated research report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_candidate_count import candidate_count


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "research-issue-report.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--research-date", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--if-missing", action="store_true")
    args = parser.parse_args()

    if args.if_missing and args.output.exists():
        print(f"Research report already exists: {args.output}")
        return

    issue = json.loads(args.issue_json.read_text(encoding="utf-8"))
    values = {
        "{{ISSUE_NUMBER}}": str(issue["number"]),
        "{{ISSUE_URL}}": str(issue["url"]),
        "{{RESEARCH_DATE}}": args.research_date,
        "{{CANDIDATE_COUNT}}": candidate_count(issue),
        "{{RESEARCH_MODEL}}": args.model,
    }
    text = TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in values.items():
        text = text.replace(placeholder, value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Created research report shell: {args.output}")


if __name__ == "__main__":
    main()
