#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches"))
from apply_overlays import replace_once  # noqa: E402


class TestReplaceOnce(unittest.TestCase):
    def test_replaces_unique(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            p = Path(raw) / "f.py"
            p.write_text("alpha\nbeta\n")
            replace_once(p, "beta\n", "beta\ngamma\n", "t")
            self.assertEqual(p.read_text(), "alpha\nbeta\ngamma\n")

    def test_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            p = Path(raw) / "f.py"
            p.write_text("alpha\nKEEP\n")
            replace_once(p, "KEEP\n", "REPLACED\n", "t")
            replace_once(p, "KEEP\n", "REPLACED\n", "t")
            self.assertEqual(p.read_text(), "alpha\nREPLACED\n")

    def test_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            p = Path(raw) / "f.py"
            p.write_text("alpha\n")
            with self.assertRaises(SystemExit):
                replace_once(p, "nope\n", "x\n", "t")

    def test_not_unique(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            p = Path(raw) / "f.py"
            p.write_text("beta\nbeta\n")
            with self.assertRaises(SystemExit):
                replace_once(p, "beta\n", "x\n", "t")


if __name__ == "__main__":
    unittest.main()
