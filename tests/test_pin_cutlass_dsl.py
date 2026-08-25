#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches"))
from pin_cutlass_dsl import pin_text  # noqa: E402

B12X_PYPROJECT = """\
dependencies = [
  "torch>=2.12.0",
  "cuda-python",
  "nvidia-cutlass-dsl==4.6.2",
  "nvidia-cutlass-dsl-libs-base==4.6.2",
  "nvidia-cutlass-dsl-libs-core==4.6.2",
  "nvidia-cutlass-dsl-libs-cu12==4.6.2",
  "nvidia-cutlass-dsl-libs-cu13==4.6.2",
  "rich>=13",
]
"""


class TestPinCutlassDsl(unittest.TestCase):
    def test_rewrites_five_b12x_pins(self) -> None:
        updated, count = pin_text(B12X_PYPROJECT, "4.7.0")
        self.assertEqual(count, 5)
        self.assertNotIn("==4.6.2", updated)
        self.assertEqual(updated.count("==4.7.0"), 5)
        self.assertIn("nvidia-cutlass-dsl==4.7.0", updated)
        self.assertIn("nvidia-cutlass-dsl-libs-cu13==4.7.0", updated)

    def test_rewrites_two_quack_pins(self) -> None:
        src = (
            "nvidia-cutlass-dsl==4.6.2\n"
            "nvidia-cutlass-dsl[cu13]==4.6.2\n"
        )
        updated, count = pin_text(src, "4.7.0")
        self.assertEqual(count, 2)
        self.assertEqual(updated.count("==4.7.0"), 2)

    def test_rewrites_requires_dist_metadata(self) -> None:
        src = (
            "Requires-Dist: nvidia-cutlass-dsl==4.6.2\n"
            "Requires-Dist: torch\n"
            'Requires-Dist: nvidia-cutlass-dsl[cu13]==4.6.2; extra == "cu13"\n'
        )
        updated, count = pin_text(src, "4.7.0")
        self.assertEqual(count, 2)
        self.assertIn("Requires-Dist: nvidia-cutlass-dsl==4.7.0\n", updated)
        self.assertIn(
            'Requires-Dist: nvidia-cutlass-dsl[cu13]==4.7.0; extra == "cu13"',
            updated,
        )

    def test_leaves_unrelated_pins(self) -> None:
        updated, count = pin_text('torch==2.13.0\n"nvidia-cutlass-dsl==4.6.2"\n', "4.7.0")
        self.assertEqual(count, 1)
        self.assertIn("torch==2.13.0", updated)


if __name__ == "__main__":
    unittest.main()
