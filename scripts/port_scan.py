#!/usr/bin/env python3
"""Apply `apply_main` to a tree, isolating each overlay's outcome.

Plain `apply_overlays.py --stack main` aborts on the first `SystemExit`, so one
stale needle hides the rest. This imports `apply_overlays`, wraps every
`patch_*` / `copy_*` callable to capture its stdout and exceptions, and runs
`apply_main` once, so a single pass lists every overlay's result in order.

`--with-upstream-patches` reproduces the build's order (`pr-*.diff` first, then
the overlays). Without it the tree is used as-is, which is the mode that
`patch_cutlass_sm12x_guard` is documented to fail in.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "patches"))

import apply_overlays as ao  # noqa: E402


def apply_upstream_patches(vllm: Path) -> list[str]:
    lines: list[str] = []
    for patch in sorted((ROOT / "patches" / "upstream").glob("pr-*.diff")):
        proc = subprocess.run(
            ["patch", "-p1", "--forward", "-N"],
            cwd=vllm.parent,
            stdin=patch.open("rb"),
            capture_output=True,
        )
        lines.append(f"{'applied' if proc.returncode == 0 else 'skip'} {patch.name}")
    return lines


def scan(vllm: Path) -> int:
    results: list[tuple[str, str, str]] = []
    for name in sorted(dir(ao)):
        if not name.startswith(("patch_", "copy_")):
            continue
        original = getattr(ao, name)
        if not callable(original):
            continue

        def wrapper(*args, _name: str = name, _fn=original, **kwargs) -> None:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    _fn(*args, **kwargs)
            except SystemExit as exc:
                results.append((_name, "fail", str(exc)))
                return
            except Exception as exc:  # noqa: BLE001
                results.append((_name, "error", f"{type(exc).__name__}: {exc}"))
                return
            out = buf.getvalue()
            status = "applied" if "ok " in out else "no-op"
            detail = "; ".join(
                line.strip() for line in out.splitlines() if line.strip()
            )
            results.append((_name, status, detail))

        setattr(ao, name, wrapper)

    try:
        ao.apply_main(vllm)
    except SystemExit as exc:
        print(f"apply_main aborted: {exc}", file=sys.stderr)

    width = max(len(n) for n, _, _ in results)
    for i, (name, status, detail) in enumerate(results, 1):
        print(f"{i:3d}  {status:<8}{name:<{width}}  {detail}")
    applied = sum(1 for _, s, _ in results if s == "applied")
    no_op = sum(1 for _, s, _ in results if s == "no-op")
    failed = len(results) - applied - no_op
    print(f"\nFAIL={failed}  applied={applied}  no-op={no_op}  total={len(results)}")
    for name, status, detail in results:
        if status in ("fail", "error"):
            print(f"  {status.upper()} {name}: {detail}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-dir", type=Path, default=None)
    parser.add_argument(
        "--with-upstream-patches",
        action="store_true",
        help="apply patches/upstream/pr-*.diff first, as the image build does",
    )
    args = parser.parse_args()
    vllm = args.vllm_dir.resolve() if args.vllm_dir else ao._vllm_dir()
    if not (vllm / "config/kernel.py").is_file():
        raise SystemExit(f"not a vllm package: {vllm}")
    if args.with_upstream_patches:
        for line in apply_upstream_patches(vllm):
            print(line)
        print()
    return scan(vllm)


if __name__ == "__main__":
    raise SystemExit(main())
