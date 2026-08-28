#!/usr/bin/env python3
"""Build the fixed prompt used to triage source-discovery candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates")
    parser.add_argument("output")
    parser.add_argument("--prompt", default="prompts/source-discovery.md")
    parser.add_argument("--style", default="prompts/fluent-korean.md")
    args = parser.parse_args()

    data = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    style = Path(args.style).read_text(encoding="utf-8")
    prompt += "\n\n## Korean writing style\n\n" + style
    prompt += "\n\n## Candidate JSON\n\n```json\n"
    prompt += json.dumps(data, ensure_ascii=False, indent=2)
    prompt += "\n```\n"
    Path(args.output).write_text(prompt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
