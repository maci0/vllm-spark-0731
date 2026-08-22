#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches"))
from assert_stack import check  # noqa: E402


class TestAssertStack(unittest.TestCase):
    def test_fp8_b12x_dspark(self) -> None:
        self.assertEqual(check("fp8_ds_mla", "", "b12x", "dspark", 5), "fp8")

    def test_nvfp4_b12x_dspark(self) -> None:
        self.assertEqual(check("nvfp4_ds_mla", "", "b12x", "dspark", 5), "nvfp4")

    def test_eugr(self) -> None:
        self.assertEqual(
            check("fp8_ds_mla", "B12X_MLA_SPARSE", "b12x", "dspark", 5), "eugr"
        )

    def test_refuse_mixed_overlay(self) -> None:
        with self.assertRaises(SystemExit):
            check("nvfp4_ds_mla", "B12X_MLA_SPARSE", "b12x", "dspark", 5)

    def test_refuse_no_dspark(self) -> None:
        with self.assertRaises(SystemExit):
            check("fp8_ds_mla", "", "b12x", "mtp", 5)

    def test_refuse_wrong_k(self) -> None:
        with self.assertRaises(SystemExit):
            check("fp8_ds_mla", "", "b12x", "dspark", 7)

    def test_refuse_b12x_mla_on_028(self) -> None:
        with self.assertRaises(SystemExit):
            check("fp8_ds_mla", "FLASHINFER", "b12x", "dspark", 5)


if __name__ == "__main__":
    unittest.main()
