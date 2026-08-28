#!/usr/bin/env python3
"""Build the GitHub-integrated WikiDocs book bundle.

Stable book chapters live in ``book/``.  Dated research reports stay in
``docs/`` as their canonical source and are materialized as research
appendix subchapters only in the generated bundle.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "book"
RESEARCH_SOURCE_DIR = ROOT / "docs"
ASSET_SOURCE_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "wikidocs"
PAGE_ID_MAP_FILE = ROOT / "docs" / "wikidocs-page-map.json"
SOURCE_REPO = "https://github.com/recrack/oh-my-dgx-spark"
WIKIDOCS_REPO = "https://github.com/recrack/oh-my-dgx-spark-wikidocs"
WIKIDOCS_PAGE_BASE = "https://wikidocs.net"
REPOSITORY_BLOB_DIRECTORIES = (
    ".github",
    "docs",
    "prompts",
    "research",
    "templates",
    "tests",
    "tools",
)

SOURCE_TOC_PATTERN = re.compile(
    r"^(?P<indent> *)- \[(?P<title>[^]]+)\]\((?P<filename>[^()\s]+\.md)\)$"
)
NUMERIC_TOC_TITLE_PATTERN = re.compile(r"^(?P<number>[0-9]{2}(?:-[0-9]+)?)\.")
APPENDIX_TOC_TITLE_PATTERN = re.compile(r"^부록 (?P<letter>[A-Z])\.")
APPENDIX_SUBCHAPTER_TITLE_PATTERN = re.compile(
    r"^(?P<letter>[A-Z])-(?P<number>[0-9]+)\."
)

RESEARCH_PAGE_PATTERN = re.compile(
    r"^appendix-b-research-issue-(?P<issue>[0-9]+)_(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\.md$"
)
RESEARCH_REPORT_PATTERN = re.compile(
    r"^research-issue-(?P<issue>[0-9]+)_(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\.md$"
)


def _parse_research_filename(
    filename: str,
    pattern: re.Pattern[str],
    description: str,
) -> tuple[int, str] | None:
    match = pattern.fullmatch(filename)
    if match is None:
        return None
    issue_number = int(match.group("issue"))
    research_date = match.group("date")
    if issue_number <= 0:
        raise SystemExit(f"research Issue number must be positive: {filename}")
    try:
        date.fromisoformat(research_date)
    except ValueError as exc:
        raise SystemExit(
            f"{description} date must be a real YYYY-MM-DD date: {filename}"
        ) from exc
    return issue_number, research_date


def parse_research_page_filename(filename: str) -> tuple[int, str] | None:
    return _parse_research_filename(
        filename,
        RESEARCH_PAGE_PATTERN,
        "research page",
    )


def parse_research_report_filename(filename: str) -> tuple[int, str] | None:
    return _parse_research_filename(
        filename,
        RESEARCH_REPORT_PATTERN,
        "research report",
    )


def public_research_page_filename(issue_number: int, research_date: str) -> str:
    return f"appendix-b-research-issue-{issue_number}_{research_date}.md"


def expected_toc_filename_prefix(title: str) -> str:
    """Return the filename prefix required by a reader-facing TOC title."""

    if match := NUMERIC_TOC_TITLE_PATTERN.match(title):
        return f"{match.group('number')}-"
    if match := APPENDIX_TOC_TITLE_PATTERN.match(title):
        return f"appendix-{match.group('letter').lower()}-"
    if match := APPENDIX_SUBCHAPTER_TITLE_PATTERN.match(title):
        return (
            f"appendix-{match.group('letter').lower()}-"
            f"{match.group('number')}-"
        )
    raise SystemExit(f"book TOC title has no supported chapter number: {title}")


def load_static_pages() -> list[tuple[int, str, str]]:
    """Load the reader-facing chapter order from ``book/TOC.md``.

    The source TOC is the single manifest for fixed WikiDocs pages. Every
    Markdown file in ``book/`` must appear exactly once, except README.md and
    TOC.md. This prevents a restored or newly written chapter from silently
    disappearing from the published bundle.
    """

    toc_file = SOURCE_DIR / "TOC.md"
    if not toc_file.is_file():
        raise SystemExit(f"book TOC not found: {toc_file}")

    pages: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    have_parent = False
    for line_number, line in enumerate(
        toc_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip() or line.startswith("# "):
            continue
        match = SOURCE_TOC_PATTERN.fullmatch(line)
        if match is None:
            raise SystemExit(
                f"invalid book TOC entry at {toc_file}:{line_number}: {line!r}"
            )

        indent = match.group("indent")
        if len(indent) % 2:
            raise SystemExit(
                f"book TOC indentation must use pairs of spaces at "
                f"{toc_file}:{line_number}"
            )
        depth = len(indent) // 2
        if depth not in (0, 1):
            raise SystemExit(
                f"book TOC supports chapters and one subchapter level only: "
                f"{toc_file}:{line_number}"
            )
        if depth == 0:
            have_parent = True
        elif not have_parent:
            raise SystemExit(
                f"book TOC subchapter has no parent at {toc_file}:{line_number}"
            )

        title = match.group("title")
        filename = match.group("filename")
        if Path(filename).name != filename or filename in {"README.md", "TOC.md"}:
            raise SystemExit(
                f"book TOC page must be a flat Markdown filename at "
                f"{toc_file}:{line_number}: {filename}"
            )
        expected_prefix = expected_toc_filename_prefix(title)
        if not filename.startswith(expected_prefix):
            raise SystemExit(
                f"book TOC filename does not match its chapter number at "
                f"{toc_file}:{line_number}: {title!r} requires "
                f"a filename beginning with {expected_prefix!r}, got {filename!r}"
            )
        if filename in seen:
            raise SystemExit(f"duplicate book TOC page: {filename}")
        source = SOURCE_DIR / filename
        if not source.is_file():
            raise SystemExit(f"book TOC target not found: {source}")
        source_lines = source.read_text(encoding="utf-8").splitlines()
        expected_heading = f"# {title}"
        if not source_lines or source_lines[0] != expected_heading:
            actual_heading = source_lines[0] if source_lines else "<empty file>"
            raise SystemExit(
                f"book page H1 does not match its TOC title: {source}: "
                f"expected {expected_heading!r}, got {actual_heading!r}"
            )

        seen.add(filename)
        pages.append((depth, filename, title))

    if not pages:
        raise SystemExit(f"book TOC contains no pages: {toc_file}")

    source_pages = {
        path.name
        for path in SOURCE_DIR.glob("*.md")
        if path.name not in {"README.md", "TOC.md"}
    }
    unlisted = sorted(source_pages - seen)
    if unlisted:
        raise SystemExit(
            "book Markdown files are missing from TOC.md: " + ", ".join(unlisted)
        )
    return pages


def get_pages() -> list[tuple[int, str, str]]:
    """Return fixed pages followed by generated research subchapters."""

    pages = load_static_pages()
    dated_pages: list[tuple[str, int, str]] = []
    for source in RESEARCH_SOURCE_DIR.glob("research-issue-*_*.md"):
        parsed = parse_research_report_filename(source.name)
        if parsed is None:
            raise SystemExit(
                "invalid dated research report filename: "
                f"{source.name}; expected research-issue-N_YYYY-MM-DD.md"
            )
        issue_number, research_date = parsed
        dated_pages.append(
            (
                research_date,
                issue_number,
                public_research_page_filename(issue_number, research_date),
            )
        )

    dated_entries = [
        (
            1,
            filename,
            f"B-{index}. {research_date} Issue #{issue_number} 상세 리서치",
        )
        for index, (research_date, issue_number, filename) in enumerate(
            sorted(dated_pages), start=1
        )
    ]
    try:
        parent_index = next(
            index
            for index, (_, filename, _) in enumerate(pages)
            if filename == "appendix-b-research-log.md"
        )
    except StopIteration as exc:
        raise SystemExit(
            "book/TOC.md must include appendix-b-research-log.md"
        ) from exc
    insert_index = parent_index + 1
    while insert_index < len(pages) and pages[insert_index][0] > pages[parent_index][0]:
        insert_index += 1
    pages[insert_index:insert_index] = dated_entries
    return pages


def source_for_page(filename: str) -> Path:
    """Return the canonical source for a bundle page filename."""

    parsed = parse_research_page_filename(filename)
    if parsed is not None:
        issue_number, research_date = parsed
        return RESEARCH_SOURCE_DIR / f"research-issue-{issue_number}_{research_date}.md"
    return SOURCE_DIR / filename


def rewrite_source_links(text: str) -> str:
    """Make links in copied pages work from the WikiDocs ``pages/`` folder."""

    replacements = {
        "../book/": "",
        "../README.md": f"{SOURCE_REPO}/blob/main/README.md",
        "../wikidocs/": f"{SOURCE_REPO}/tree/main/wikidocs/",
    }
    replacements.update(
        {
            f"../{directory}/": f"{SOURCE_REPO}/blob/main/{directory}/"
            for directory in REPOSITORY_BLOB_DIRECTORIES
        }
    )
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def rewrite_page_title(text: str, title: str) -> str:
    """Make the published H1 match the reader-facing source TOC title."""

    lines = text.splitlines()
    title_index = next(
        (index for index, line in enumerate(lines) if line.startswith("# ")), None
    )
    if title_index is None:
        raise SystemExit(f"book page has no top-level heading for TOC title: {title}")
    lines[title_index] = f"# {title}"
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix


def rewrite_legacy_navigation(text: str) -> str:
    """Turn old chapter-number navigation labels into natural Korean prose."""

    text = re.sub(
        r"^상위 장: (\[[^]]+\]\([^)]+\))$",
        r"이 페이지는 \1의 상세 내용입니다.",
        text,
        flags=re.MULTILINE,
    )
    return re.sub(
        r"^[0-9]+장으로 돌아가기: (\[[^]]+\]\([^)]+\))$",
        r"전체 선택 기준은 \1에서 확인할 수 있습니다.",
        text,
        flags=re.MULTILINE,
    )


def load_wikidocs_page_ids(
    pages: list[tuple[int, str, str]] | None = None,
) -> dict[str, int]:
    """Load the optional WikiDocs filename-to-page-ID mapping.

    WikiDocs creates numeric page IDs after the TOC is synchronized, so
    inventing an ID during the first export would create links that look valid
    but are wrong.
    """

    pages = pages if pages is not None else get_pages()
    if not PAGE_ID_MAP_FILE.exists():
        return {}

    try:
        data = json.loads(PAGE_ID_MAP_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid WikiDocs page map: {PAGE_ID_MAP_FILE}: {exc}") from exc

    raw_pages = data.get("pages", {}) if isinstance(data, dict) else {}
    if not isinstance(raw_pages, dict):
        raise SystemExit(f"WikiDocs page map must contain an object at 'pages': {PAGE_ID_MAP_FILE}")

    page_ids: dict[str, int] = {}
    for raw_filename, raw_page_id in raw_pages.items():
        filename = str(raw_filename)
        if filename.startswith("pages/"):
            filename = filename.removeprefix("pages/")
        try:
            page_id = int(raw_page_id)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"WikiDocs page ID must be an integer: {raw_filename}={raw_page_id!r}"
            ) from exc
        if page_id <= 0:
            raise SystemExit(f"WikiDocs page ID must be positive: {raw_filename}={page_id}")
        page_ids[filename] = page_id

    known_filenames = {filename for _, filename, _ in pages}
    unknown_filenames = sorted(
        filename
        for filename in set(page_ids) - known_filenames
        if parse_research_page_filename(filename) is None
    )
    if unknown_filenames:
        raise SystemExit(
            "WikiDocs page map contains unknown filenames: "
            + ", ".join(unknown_filenames)
        )

    # A deleted dated report can leave its old numeric ID in the optional
    # recovery map. It must not keep the page alive or participate in the
    # active duplicate-ID check; the next export will omit it from the bundle.
    page_ids = {
        filename: page_id
        for filename, page_id in page_ids.items()
        if filename in known_filenames
    }

    missing_filenames = sorted(known_filenames - set(page_ids))
    missing_fixed_filenames = [
        filename
        for filename in missing_filenames
        if parse_research_page_filename(filename) is None
    ]
    if missing_fixed_filenames:
        raise SystemExit(
            "WikiDocs page map is incomplete; missing IDs for: "
            + ", ".join(missing_fixed_filenames)
        )

    if len(set(page_ids.values())) != len(page_ids):
        raise SystemExit("WikiDocs page map contains duplicate page IDs")
    return page_ids


def strip_wikidocs_page_links(
    text: str,
    page_ids: dict[str, int] | None = None,
    pages: list[tuple[int, str, str]] | None = None,
) -> str:
    """Use stable WikiDocs IDs, or plain labels until IDs are recovered.

    ``TOC.md`` uses ``pages/filename.md`` because that is the GitHub import
    format. Body links need the numeric WikiDocs page ID instead. Until a
    synchronized TOC has been queried, a plain label is safer than publishing
    a broken filename URL.
    """

    pages = pages if pages is not None else get_pages()
    page_filenames = {filename for _, filename, _ in pages}
    pattern = re.compile(r"\[([^\]]+)\]\(([^()\s]+\.md(?:#[^()\s]+)?)\)")
    page_ids = page_ids or {}

    def replace(match: re.Match[str]) -> str:
        target = match.group(2)
        filename = target.split("#", 1)[0]
        if filename not in page_filenames:
            return match.group(0)
        page_id = page_ids.get(filename)
        if page_id is None:
            return match.group(1)
        anchor = ""
        if "#" in target:
            anchor = "#" + target.split("#", 1)[1]
        return f"[{match.group(1)}]({WIKIDOCS_PAGE_BASE}/{page_id}{anchor})"

    return pattern.sub(replace, text)


def load_book_metadata() -> tuple[str, str, str]:
    """Read the published title, subtitle, and introduction from book/README.md."""

    source = SOURCE_DIR / "README.md"
    lines = source.read_text(encoding="utf-8").splitlines()
    title_index = next(
        (index for index, line in enumerate(lines) if line.startswith("# ")), None
    )
    if title_index is None:
        raise SystemExit(f"book README has no title: {source}")

    title = lines[title_index][2:].strip()
    subtitle = next(
        (line.strip() for line in lines[title_index + 1 :] if line.strip()), ""
    )
    if not subtitle:
        raise SystemExit(f"book README has no subtitle: {source}")

    intro_heading = "## 책 소개"
    try:
        intro_start = lines.index(intro_heading) + 1
    except ValueError as exc:
        raise SystemExit(f"book README has no {intro_heading!r} section: {source}") from exc

    intro_lines: list[str] = []
    for line in lines[intro_start:]:
        if line.startswith("## "):
            break
        intro_lines.append(line)
    introduction = "\n".join(intro_lines).strip()
    if not introduction:
        raise SystemExit(f"book README has an empty {intro_heading!r} section: {source}")
    return title, subtitle, introduction


def build_readme() -> str:
    title, subtitle, introduction = load_book_metadata()
    return f"""# {title}

{subtitle}

GitHub 연동 배포 저장소: [{WIKIDOCS_REPO}]({WIKIDOCS_REPO})

![책 표지](assets/book-cover-v1.png)

{introduction}

현재 원고는 **WikiDocs GitHub 연동 배포판**입니다. 연결된 배포 저장소에 push하면 WikiDocs 책으로 자동 동기화됩니다. 책의 공개 설정과 GitHub 저장소의 공개 설정은 별개입니다. 모델과 runtime이 바뀌면 원문 링크, revision, 실행 조건, raw 결과를 함께 갱신합니다.

## 책의 구성

상위 목차는 00~10장과 부록 A·B로 구성합니다. 상위 장은 판단, 설치, serving, benchmark, 모델 선택, 클러스터, 에이전트, 운영, 비용 판단의 읽기 경로를 안내합니다. 기존 상세 원고와 재현 절차는 해당 장 아래의 서브챕터로 보존하며, 날짜별 공개 리서치는 부록 B 아래에 추가합니다.

- 한 대, 두 대, 세 대, 네 대, 여덟 대 DGX Spark의 역할과 네트워크
- Qwen, DeepSeek, MiniMax 모델의 선택 기준
- vLLM, SGLang, llama.cpp, SparkInfer와 양자화·speculative decoding
- prefill, decode, end-to-end, aggregate throughput의 차이
- tool parser, 로컬 에이전트, 권한 경계와 장애 복구
- GPT-5.6 Sol(`gpt-5.6-sol`, `reasoning_effort=max`)과 로컬 모델을 공정하게 비교하는 방법

## 실험 원칙

성능 숫자에는 hardware, model revision, quant, runtime/image, context, KV dtype, speculative decoding, concurrency, workload와 측정 방법을 함께 적습니다. 공개 레시피의 44~47 tok/s나 370K needle 결과를 이 장비의 모든 요청에 대한 보장값으로 쓰지 않습니다.

## 원본 저장소와 실험 기록

- [원본 GitHub 저장소]({SOURCE_REPO})
- [DeepSeek V4 Flash 0731 성능 리서치]({SOURCE_REPO}/blob/main/docs/deepseek-v4-flash-0731-performance-research-2026-08.md)
- [Qwen3.8-27B 커뮤니티 제작물·활용 사례]({SOURCE_REPO}/blob/main/docs/qwen38-community-builds-2026-08.md)
- [DGX Spark·GB10 벤더 비교 리서치]({SOURCE_REPO}/blob/main/docs/dgx-spark-vendor-comparison-2026-08.md)
- [DGX Spark 모델 선택 리서치]({SOURCE_REPO}/blob/main/docs/dgx-spark-model-selection-research-2026-08.md)
- [GPT-5.6 Sol max 비교 리서치]({SOURCE_REPO}/blob/main/docs/sol-max-comparison-research-2026-08.md)
- [책 집필용 참고문헌]({SOURCE_REPO}/blob/main/docs/dgx-spark-book-references-2026-08.md)
- [기존 원고 보존·재배치 기록]({SOURCE_REPO}/blob/main/docs/book-content-preservation-2026-08-24.md)
- [WikiDocs 배포 계획과 상태]({SOURCE_REPO}/blob/main/docs/wikidocs-deployment-2026-08.md)
- [WikiDocs page ID 회수와 본문 링크 연결]({SOURCE_REPO}/blob/main/docs/wikidocs-page-id-recovery-2026-08.md)

기준일: **2026-08-23**
"""


def build_toc(pages: list[tuple[int, str, str]] | None = None) -> str:
    pages = pages if pages is not None else get_pages()
    lines = ["# 목차", ""]
    lines.extend(
        f"{'  ' * depth}- [{title}](pages/{filename})"
        for depth, filename, title in pages
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"source directory not found: {SOURCE_DIR}")

    pages = get_pages()
    readme_text = build_readme()
    toc_text = build_toc(pages)
    page_ids = load_wikidocs_page_ids(pages)
    if page_ids:
        print(f"loaded {len(page_ids)} WikiDocs page IDs from {PAGE_ID_MAP_FILE}")
    else:
        print("no WikiDocs page-ID map; body page links will be exported as plain labels")

    rendered_pages: dict[str, str] = {}
    for _, filename, title in pages:
        source = source_for_page(filename)
        if not source.is_file():
            raise SystemExit(f"page source not found: {source}")
        rendered_pages[filename] = rewrite_page_title(
            strip_wikidocs_page_links(
                rewrite_legacy_navigation(
                    rewrite_source_links(source.read_text(encoding="utf-8")),
                ),
                page_ids,
                pages,
            ),
            title,
        )

    if OUTPUT_DIR.exists():
        for child in OUTPUT_DIR.iterdir():
            if child.is_dir() and child.name != ".git":
                shutil.rmtree(child)
            elif child.name != ".git":
                child.unlink()
    else:
        OUTPUT_DIR.mkdir()

    pages_dir = OUTPUT_DIR / "pages"
    assets_dir = OUTPUT_DIR / "assets"
    pages_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")
    (OUTPUT_DIR / "TOC.md").write_text(toc_text, encoding="utf-8")
    (assets_dir / ".gitkeep").write_text("", encoding="utf-8")

    if ASSET_SOURCE_DIR.is_dir():
        for asset in ASSET_SOURCE_DIR.iterdir():
            if asset.is_file():
                shutil.copy2(asset, assets_dir / asset.name)

    for filename, text in rendered_pages.items():
        (pages_dir / filename).write_text(text, encoding="utf-8")

    print(f"built {len(pages)} pages in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
