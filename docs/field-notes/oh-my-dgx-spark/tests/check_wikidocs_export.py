#!/usr/bin/env python3
"""Validate the generated WikiDocs GitHub integration bundle."""

from __future__ import annotations

import re
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "wikidocs"
TOC_PATTERN = re.compile(r"^\s*\- \[[^]]+\]\((pages/[^)]+)\)$")
sys.path.insert(0, str(ROOT))

from tools import build_wikidocs_export as exporter
from tools.build_wikidocs_export import (
    get_pages,
    parse_research_page_filename,
    parse_research_report_filename,
    rewrite_source_links,
    source_for_page,
    strip_wikidocs_page_links,
)


def check_dynamic_research_pages() -> None:
    original_source_dir = exporter.SOURCE_DIR
    original_research_source_dir = exporter.RESEARCH_SOURCE_DIR
    try:
        with TemporaryDirectory() as directory:
            source_dir = Path(directory) / "book"
            research_source_dir = Path(directory) / "docs"
            source_dir.mkdir()
            research_source_dir.mkdir()
            (source_dir / "README.md").write_text(
                "# Test book\n\nTest subtitle\n\n## 책 소개\n\nTest introduction.\n",
                encoding="utf-8",
            )
            (source_dir / "appendix-b-research-log.md").write_text(
                "# 부록 B. Research index\n", encoding="utf-8"
            )
            (source_dir / "TOC.md").write_text(
                "# 목차\n\n"
                "- [부록 B. Research index](appendix-b-research-log.md)\n",
                encoding="utf-8",
            )
            for filename in (
                "research-issue-2_2026-08-22.md",
                "research-issue-3_2026-08-23.md",
            ):
                (research_source_dir / filename).write_text("# test\n", encoding="utf-8")
            exporter.SOURCE_DIR = source_dir
            exporter.RESEARCH_SOURCE_DIR = research_source_dir
            pages = get_pages()
            filenames = [filename for _, filename, _ in pages]
            parent_index = filenames.index("appendix-b-research-log.md")
            expected = [
                "appendix-b-research-issue-2_2026-08-22.md",
                "appendix-b-research-issue-3_2026-08-23.md",
            ]
            if filenames[parent_index + 1 :] != expected:
                raise SystemExit("dated research pages are not ordered below appendix B")
            toc = exporter.build_toc(pages)
            for index, filename in enumerate(expected, start=1):
                expected_line = (
                    f"  - [B-{index}. 2026-08-{21 + index:02d} Issue #{index + 1} 상세 리서치]"
                    f"(pages/{filename})"
                )
                if expected_line not in toc:
                    raise SystemExit(f"dated research page is missing from TOC: {filename}")
            if source_for_page(expected[0]) != research_source_dir / "research-issue-2_2026-08-22.md":
                raise SystemExit("dated research page does not resolve to the docs source")
    finally:
        exporter.SOURCE_DIR = original_source_dir
        exporter.RESEARCH_SOURCE_DIR = original_research_source_dir

    for invalid in (
        "appendix-b-research-issue-0_2026-08-22.md",
        "appendix-b-research-issue-2_2026-02-30.md",
    ):
        try:
            parse_research_page_filename(invalid)
        except SystemExit:
            continue
        raise SystemExit(f"invalid dated research filename was accepted: {invalid}")

    for invalid in (
        "research-issue-0_2026-08-22.md",
        "research-issue-2_2026-02-30.md",
    ):
        try:
            parse_research_report_filename(invalid)
        except SystemExit:
            continue
        raise SystemExit(f"invalid dated research report filename was accepted: {invalid}")


def check_static_filename_and_title_contract() -> None:
    original_source_dir = exporter.SOURCE_DIR
    try:
        with TemporaryDirectory() as directory:
            source_dir = Path(directory) / "book"
            source_dir.mkdir()
            exporter.SOURCE_DIR = source_dir

            stale_page = source_dir / "17-old-chapter.md"
            stale_page.write_text("# 01. Renamed chapter\n", encoding="utf-8")
            (source_dir / "TOC.md").write_text(
                "# 목차\n\n- [01. Renamed chapter](17-old-chapter.md)\n",
                encoding="utf-8",
            )
            try:
                exporter.load_static_pages()
            except SystemExit as exc:
                if "filename does not match its chapter number" not in str(exc):
                    raise
            else:
                raise SystemExit("export accepted a stale chapter-number filename")

            stale_page.rename(source_dir / "01-renamed-chapter.md")
            (source_dir / "TOC.md").write_text(
                "# 목차\n\n- [01. Current title](01-renamed-chapter.md)\n",
                encoding="utf-8",
            )
            try:
                exporter.load_static_pages()
            except SystemExit as exc:
                if "H1 does not match its TOC title" not in str(exc):
                    raise
            else:
                raise SystemExit("export accepted a page H1 that differs from the TOC")
    finally:
        exporter.SOURCE_DIR = original_source_dir


def check_export_syncs_additions_and_deletions() -> None:
    originals = {
        "SOURCE_DIR": exporter.SOURCE_DIR,
        "RESEARCH_SOURCE_DIR": exporter.RESEARCH_SOURCE_DIR,
        "ASSET_SOURCE_DIR": exporter.ASSET_SOURCE_DIR,
        "OUTPUT_DIR": exporter.OUTPUT_DIR,
        "PAGE_ID_MAP_FILE": exporter.PAGE_ID_MAP_FILE,
    }
    try:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "book"
            research_source_dir = root / "docs"
            asset_source_dir = root / "assets"
            output_dir = root / "wikidocs"
            source_dir.mkdir()
            research_source_dir.mkdir()
            asset_source_dir.mkdir()
            (source_dir / "README.md").write_text(
                "# Test book\n\nTest subtitle\n\n## 책 소개\n\nTest introduction.\n",
                encoding="utf-8",
            )
            (source_dir / "01-static-page.md").write_text(
                "# 01. Static page\n", encoding="utf-8"
            )
            (source_dir / "appendix-b-research-log.md").write_text(
                "# 부록 B. Research index\n", encoding="utf-8"
            )
            (source_dir / "TOC.md").write_text(
                "# 목차\n\n"
                "- [01. Static page](01-static-page.md)\n"
                "- [부록 B. Research index](appendix-b-research-log.md)\n",
                encoding="utf-8",
            )
            report = research_source_dir / "research-issue-2_2026-08-22.md"
            report.write_text("# Research report\n", encoding="utf-8")

            exporter.SOURCE_DIR = source_dir
            exporter.RESEARCH_SOURCE_DIR = research_source_dir
            exporter.ASSET_SOURCE_DIR = asset_source_dir
            exporter.OUTPUT_DIR = output_dir
            exporter.PAGE_ID_MAP_FILE = root / "wikidocs-page-map.json"
            exporter.main()
            public_name = "appendix-b-research-issue-2_2026-08-22.md"
            public_path = output_dir / "pages" / public_name
            static_path = output_dir / "pages" / "01-static-page.md"
            toc = (output_dir / "TOC.md").read_text(encoding="utf-8")
            if not public_path.is_file() or f"pages/{public_name}" not in toc:
                raise SystemExit("export did not add the dated research page")
            if not static_path.is_file() or "pages/01-static-page.md" not in toc:
                raise SystemExit("export did not add the fixed book page")

            orphan = source_dir / "orphan.md"
            orphan.write_text("# Unlisted page\n", encoding="utf-8")
            try:
                exporter.main()
            except SystemExit as exc:
                if "missing from TOC.md" not in str(exc):
                    raise
            else:
                raise SystemExit("export accepted an unlisted fixed book page")
            finally:
                orphan.unlink()
            if not static_path.is_file() or not public_path.is_file():
                raise SystemExit("failed validation damaged the existing export bundle")

            report.unlink()
            exporter.main()
            toc = (output_dir / "TOC.md").read_text(encoding="utf-8")
            if public_path.exists() or f"pages/{public_name}" in toc:
                raise SystemExit("export did not remove the deleted dated research page")

            (source_dir / "01-static-page.md").unlink()
            (source_dir / "TOC.md").write_text(
                "# 목차\n\n"
                "- [부록 B. Research index](appendix-b-research-log.md)\n",
                encoding="utf-8",
            )
            exporter.main()
            toc = (output_dir / "TOC.md").read_text(encoding="utf-8")
            if static_path.exists() or "pages/01-static-page.md" in toc:
                raise SystemExit("export did not remove the deleted fixed book page")
    finally:
        for name, value in originals.items():
            setattr(exporter, name, value)


def check_stale_research_page_ids_are_ignored() -> None:
    original_page_id_map_file = exporter.PAGE_ID_MAP_FILE
    try:
        with TemporaryDirectory() as directory:
            page_map = Path(directory) / "wikidocs-page-map.json"
            page_map.write_text(
                '{"pages": {"static.md": 101, "appendix-b-research-issue-2_2026-08-22.md": 202}}',
                encoding="utf-8",
            )
            exporter.PAGE_ID_MAP_FILE = page_map
            page_ids = exporter.load_wikidocs_page_ids(
                [(0, "static.md", "01. Static")]
            )
            if page_ids != {"static.md": 101}:
                raise SystemExit("stale dated research page ID was not safely ignored")
    finally:
        exporter.PAGE_ID_MAP_FILE = original_page_id_map_file


def check_repository_source_links_are_stable() -> None:
    source = "\n".join(
        [
            "[문체 지침](../prompts/fluent-korean.md)",
            "[보고서 형식](../templates/research-issue-report.md)",
            "[자동화 코드](../tools/build_wikidocs_export.py)",
            "[책 장](../book/01-static-page.md)",
        ]
    )
    rewritten = strip_wikidocs_page_links(
        rewrite_source_links(source),
        pages=[(0, "01-static-page.md", "01. Static page")],
    )
    expected_links = (
        f"{exporter.SOURCE_REPO}/blob/main/prompts/fluent-korean.md",
        f"{exporter.SOURCE_REPO}/blob/main/templates/research-issue-report.md",
        f"{exporter.SOURCE_REPO}/blob/main/tools/build_wikidocs_export.py",
    )
    if not all(link in rewritten for link in expected_links):
        raise SystemExit("repository source link was not rewritten to GitHub")
    if "../" in rewritten or ".md)" in rewritten.splitlines()[-1]:
        raise SystemExit("relative repository or book link remained after export")


def main() -> None:
    check_dynamic_research_pages()
    check_static_filename_and_title_contract()
    check_export_syncs_additions_and_deletions()
    check_stale_research_page_ids_are_ignored()
    check_repository_source_links_are_stable()
    required = [BUNDLE / "README.md", BUNDLE / "TOC.md", BUNDLE / "pages", BUNDLE / "assets"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing WikiDocs bundle entries: " + ", ".join(missing))

    toc_lines = (BUNDLE / "TOC.md").read_text(encoding="utf-8").splitlines()
    page_paths = [match.group(1) for line in toc_lines if (match := TOC_PATTERN.match(line))]
    if not page_paths:
        raise SystemExit("TOC.md contains no pages entries")
    if not any(line.startswith("  - ") for line in toc_lines):
        raise SystemExit("TOC.md contains no nested subchapter entries")

    pages = get_pages()
    titles = {filename: title for _, filename, title in pages}
    expected_page_paths = [f"pages/{filename}" for _, filename, _ in pages]
    if page_paths != expected_page_paths:
        raise SystemExit("generated TOC order does not match book/TOC.md and research pages")
    actual_page_paths = sorted(
        f"pages/{path.name}" for path in (BUNDLE / "pages").glob("*.md")
    )
    if actual_page_paths != sorted(expected_page_paths):
        raise SystemExit("generated pages do not exactly match the TOC page set")
    dynamic_pages = [
        filename
        for _, filename, _ in pages
        if parse_research_page_filename(filename) is not None
    ]
    if dynamic_pages:
        parent_path = "pages/appendix-b-research-log.md"
        parent_index = next(
            index for index, path in enumerate(page_paths) if path == parent_path
        )
        for filename in dynamic_pages:
            path = f"pages/{filename}"
            if path not in page_paths:
                raise SystemExit(f"dated research page missing from TOC: {path}")
            if page_paths.index(path) <= parent_index:
                raise SystemExit(f"dated research page is not under appendix B: {path}")
            toc_line = next(line for line in toc_lines if f"]({path})" in line)
            if not toc_line.startswith("  - "):
                raise SystemExit(f"dated research page is not a level-1 subchapter: {path}")

    readme = (BUNDLE / "README.md").read_text(encoding="utf-8")
    if "![책 표지](assets/book-cover-v1.png)" not in readme:
        raise SystemExit("README.md does not contain the relative cover link")

    errors: list[str] = []
    for relative in page_paths:
        page = BUNDLE / relative
        if not page.is_file():
            errors.append(f"TOC target missing: {relative}")
            continue
        text = page.read_text(encoding="utf-8")
        filename = Path(relative).name
        expected_heading = f"# {titles[filename]}"
        if not text.splitlines() or text.splitlines()[0] != expected_heading:
            errors.append(
                f"page heading does not match source TOC: {relative}: "
                f"expected {expected_heading!r}"
            )
        for forbidden in ("../docs/", "../tests/", "../README.md", "../wikidocs/"):
            if forbidden in text:
                errors.append(f"unrewritten source link {forbidden!r}: {relative}")
        for legacy_navigation in ("상위 장:", "장으로 돌아가기:"):
            if legacy_navigation in text:
                errors.append(
                    f"legacy navigation label remains in published page: "
                    f"{relative}: {legacy_navigation}"
                )

    unstable_page_link = re.compile(r"\]\((?!https?://)[^()\s]+\.md(?:#[^()\s]+)?\)")
    for relative in page_paths:
        text = (BUNDLE / relative).read_text(encoding="utf-8")
        for line in text.splitlines():
            if unstable_page_link.search(line):
                errors.append(f"unstable WikiDocs filename link remains: {relative}")

    mapped = strip_wikidocs_page_links(
        "[벤치마크](05-benchmark.md#측정값의-이름부터-나눈다)",
        {"05-benchmark.md": 410351},
    )
    if mapped != "[벤치마크](https://wikidocs.net/410351#측정값의-이름부터-나눈다)":
        errors.append("WikiDocs page-ID mapping did not produce an absolute URL")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"WikiDocs export OK: {len(page_paths)} pages")


if __name__ == "__main__":
    main()
