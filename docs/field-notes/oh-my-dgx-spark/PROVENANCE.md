# Provenance

Vendored verbatim on 2026-08-26 from
<https://github.com/recrack/oh-my-dgx-spark> (commit `8fa0fb6`, "research:
analyze issue #47 (#48)").

A self-contained **DGX Spark lab corpus** (Korean-language): book chapters,
WikiDocs export, research logs, measured results, diagrams, and the research
automation tooling. Not part of the DeepSeek-V4-Flash-0731 experiment lineage
in this repo (`docs/field-notes/`); kept intact as an external reference
corpus because it covers the same hardware (DGX Spark / GB10) and includes
DeepSeek-V4-Flash-0731 material.

- Book + export: `book/`, `wikidocs/`
- Research logs, fact reviews, results: `docs/`
- Diagrams: `assets/`, `docs/diagrams/`
- Automation (research discovery, WikiDocs publishing, book drafting): `tools/`, `tests/`, `prompts/`, `templates/`, `.github/`
- Original README: `README.md` (kept verbatim)

## Scope note for this repo's tooling

This tree is **vendored third-party content, preserved verbatim** — the
verification harness (`scripts/verify-docs.py`) excludes it by design, so
foreign formatting or tooling references do not produce false findings. Its
internal links were checked at merge time (169 markdown files; the only
non-resolving ref is the `{{ISSUE_URL}}` template placeholder).
