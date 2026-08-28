#!/usr/bin/env python3
"""Derive the corpus document graph from the knowledge base itself.

Parses docs/knowledge/*.md and emits a mermaid "document graph" whose nodes
are the corpus documents and whose edges are the real links between them:

  next     — the footer prev/next reading chain (01 → 02 → … → 11 → glossary)
  related  — `## Related Docs` cross-links between chapters
  evidence — `### Raw evidence (field notes)` links into docs/field-notes/
  defines  — chapter top-nav links to the glossary

The graph is generated, not hand-maintained: re-run after any chapter edit.

Usage:  python3 scripts/knowledge-graph.py [--mermaid | --edges]
"""

from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE = os.path.join(ROOT, "docs", "knowledge")

NODE_ORDER = (
    "00-index", "01-hardware", "02-model", "03-kernels-attention",
    "04-quantization-kv", "05-performance", "06-deployment", "07-gotchas",
    "08-upstream", "09-golden-deepgemm", "10-operations-agents",
    "11-cost-decision", "12-debug-standin", "13-qwenseek", "glossary",
)
LABELS = {
    "00-index": "00-index (hub)",
    "01-hardware": "01-hardware",
    "02-model": "02-model",
    "03-kernels-attention": "03-kernels-attention",
    "04-quantization-kv": "04-quantization-kv",
    "05-performance": "05-performance",
    "06-deployment": "06-deployment",
    "07-gotchas": "07-gotchas",
    "08-upstream": "08-upstream",
    "09-golden-deepgemm": "09-golden-deepgemm",
    "10-operations-agents": "10-operations-agents",
    "11-cost-decision": "11-cost-decision",
    "12-debug-standin": "12-debug-standin",
    "13-qwenseek": "13-qwenseek",
    "glossary": "glossary",
}

FIELD_NOTE_DIRS = ("dgx-spark", "nvfp4", "oh-my-dgx-spark")


def slug(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]


def load_chapters() -> dict[str, str]:
    out = {}
    for f in glob.glob(os.path.join(KNOWLEDGE, "*.md")):
        s = slug(f)
        if s in NODE_ORDER:
            with open(f, encoding="utf-8") as fh:
                out[s] = fh.read()
    return out


def extract_edges(chapters: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return (src, dst, kind) tuples. dst may be a field-note dir or a chapter slug."""
    edges: list[tuple[str, str, str]] = []
    link_re = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

    for src, txt in chapters.items():
        # 1. footer chain
        m = re.search(r"\*\*\[← Prev\]\(([^)]+)\)[^\n]*\[Next\]\(([^)]+)\)", txt)
        if m:
            for t in (m.group(1), m.group(2)):
                d = slug(t)
                if d in NODE_ORDER and d != src:
                    edges.append((src, d, "next"))
        # 2. Related Docs
        sec = re.search(r"## Related Docs\n(.*?)(?:\n### |\n---|\Z)", txt, re.S)
        if sec:
            for lm in link_re.finditer(sec.group(1)):
                d = slug(lm.group(2))
                if d in NODE_ORDER and d != src:
                    edges.append((src, d, "related"))
        # 3. Raw evidence -> field-notes
        ev = re.search(r"### Raw evidence \(field notes\)\n(.*?)(?:\n---|\Z)", txt, re.S)
        if ev:
            for lm in link_re.finditer(ev.group(1)):
                t = lm.group(2)
                for d in FIELD_NOTE_DIRS:
                    if f"field-notes/{d}/" in t:
                        edges.append((src, d, "evidence"))
        # 4. top-nav glossary link
        if "· [Glossary](glossary.md)" in txt.split("\n", 1)[0] and src != "glossary":
            edges.append((src, "glossary", "defines"))

    # dedupe, keep first-seen order
    seen: set[tuple[str, str, str]] = set()
    out = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def render_mermaid(edges: list[tuple[str, str, str]]) -> str:
    nodes = []
    for n in NODE_ORDER:
        nodes.append(f'  {n}["{LABELS[n]}"]')
    lines = ["```mermaid", "graph LR", *nodes, ""]
    # next-chain first (keeps the reading spine visually tight), then rest
    def key(e):
        return 0 if e[2] == "next" else (1 if e[2] == "related" else (2 if e[2] == "defines" else 3))
    for src, dst, kind in sorted(edges, key=key):
        arrow = {"next": "-->", "related": "-.->", "defines": "-.->", "evidence": "-.->"}[kind]
        label = {"next": "next", "related": "related", "defines": "defines", "evidence": "evidence"}[kind]
        lines.append(f"  {src}{arrow}|{label}| {dst}")
    lines.append("```")
    return "\n".join(lines)


def render_edges(edges: list[tuple[str, str, str]]) -> str:
    by_kind: dict[str, list[tuple[str, str, str]]] = {}
    for e in edges:
        by_kind.setdefault(e[2], []).append(e)
    out = []
    for kind in ("next", "related", "defines", "evidence"):
        out.append(f"[{kind}] {len(by_kind.get(kind, []))}")
        for s, d, _ in sorted(by_kind.get(kind, [])):
            out.append(f"  {s} -> {d}")
    return "\n".join(out)


def main() -> int:
    chapters = load_chapters()
    if len(chapters) != len(NODE_ORDER):
        print(f"error: found {len(chapters)}/{len(NODE_ORDER)} chapters", file=sys.stderr)
        return 1
    edges = extract_edges(chapters)
    if "--edges" in sys.argv:
        print(render_edges(edges))
    elif "--spine" in sys.argv:
        spine = [e for e in edges if e[2] == "next"]
        lines = ["```mermaid", "graph LR"]
        for n in NODE_ORDER:
            lines.append(f'  {n}["{LABELS[n]}"]')
        for src, dst, _ in spine:
            lines.append(f"  {src} --> {dst}")
        lines.append("```")
        print("\n".join(lines))
    else:
        print(render_mermaid(edges))
    return 0


if __name__ == "__main__":
    sys.exit(main())
