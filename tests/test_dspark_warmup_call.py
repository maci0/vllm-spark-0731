#!/usr/bin/env python3
"""Guard the DSpark gumbel warmup against gumbel_sample signature drift.

The warmup used to pass nine positional arguments. gumbel_sample gained
``is_drafting`` as its seventh parameter, so every later argument shifted one
slot, and the call raised ``AttributeError: 'bool' object has no attribute
'contiguous'`` inside a try/except that only logs. The visible symptom was the
DSpark gumbel triton kernels JIT-compiling during inference instead of at
startup.

Positional binding is what made the failure silent, so the first check is that
the warmup binds by keyword. The second resolves those keywords against the real
signature, and is skipped when no vLLM source tree is present (point
``VLLM_SRC`` at one to enable it, or it will try ``/opt/vllm``).

Set ``WARMUP_OVERLAY`` to run the checks against a different copy of the file.
"""
from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
WARMUP = Path(
    os.environ.get("WARMUP_OVERLAY", HERE.parents[1] / "patches" / "files" / "dsv4_warmup_ext.py")
)
GUMBEL_REL = Path("vllm/v1/worker/gpu/sample/gumbel.py")


def vllm_gumbel_path() -> Path | None:
    override = os.environ.get("VLLM_SRC")
    candidates = (
        [Path(override) / GUMBEL_REL] if override else []
    ) + [Path("/opt/vllm") / GUMBEL_REL, Path("/usr/local/lib/python3.12/dist-packages") / GUMBEL_REL]
    for path in candidates:
        if path.exists():
            return path
    return None


def calls_to(tree: ast.AST, name: str) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            out.append(node)
    return out


class TestDsparkGumbelWarmupCall(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(WARMUP.exists(), f"missing overlay {WARMUP}")
        self.tree = ast.parse(WARMUP.read_text())
        self.calls = calls_to(self.tree, "gumbel_sample")
        self.assertEqual(len(self.calls), 1, "expected exactly one gumbel_sample call")

    def test_binds_by_keyword(self) -> None:
        self.assertEqual(
            self.calls[0].args,
            [],
            "warmup must bind gumbel_sample by keyword; a positional call silently "
            "shifts every argument when the signature changes",
        )

    def test_keywords_match_signature(self) -> None:
        path = vllm_gumbel_path()
        if path is None:
            self.skipTest("no vLLM source tree found; set VLLM_SRC to enable")
        fn = next(
            (
                node
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == "gumbel_sample"
            ),
            None,
        )
        self.assertIsNotNone(fn, f"no gumbel_sample definition in {path}")

        positional = fn.args.posonlyargs + fn.args.args
        known = {a.arg for a in positional + fn.args.kwonlyargs}
        required = {a.arg for a in positional[: len(positional) - len(fn.args.defaults)]}
        required |= {
            a.arg for a, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if default is None
        }

        used = {kw.arg for kw in self.calls[0].keywords if kw.arg is not None}
        self.assertFalse(
            used - known,
            f"warmup passes keywords gumbel_sample does not take: {sorted(used - known)}",
        )
        self.assertFalse(
            required - used,
            f"warmup omits required parameters of gumbel_sample: {sorted(required - used)}",
        )


if __name__ == "__main__":
    unittest.main()
