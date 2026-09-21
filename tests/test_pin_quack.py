#!/usr/bin/env python3
"""Verification half of the quack-kernels CUTLASS DSL pin.

quack-kernels 0.6.4 declared two exact pins (``nvidia-cutlass-dsl==4.6.2`` plus its
``[cu13]`` extra) and needed both rewritten. 0.6.5 declares ``>=4.7`` instead, so
``pin_quack`` rewrites nothing and must confirm the requirement is already satisfied
rather than fail on a rewrite count.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patches"))
sys.path.insert(0, str(ROOT / "docker"))

from pin_cutlass_dsl import pin_text  # noqa: E402
from pin_quack import unsatisfied  # noqa: E402

EXACT_064 = (
    "Requires-Dist: nvidia-cutlass-dsl==4.6.2\n"
    'Requires-Dist: nvidia-cutlass-dsl[cu13]==4.6.2; extra == "cu13"\n'
)
RANGE_065 = (
    "Requires-Dist: nvidia-cutlass-dsl>=4.7\n"
    'Requires-Dist: nvidia-cutlass-dsl[cu13]>=4.7; extra == "cu13"\n'
)
UNRELATED = 'Requires-Dist: torch\nRequires-Dist: nvidia-matmul-heuristics\n'


class TestPinQuack(unittest.TestCase):
    def test_exact_pins_are_rewritten_and_satisfied(self) -> None:
        updated, count = pin_text(EXACT_064, "4.7.0")
        self.assertEqual(count, 2)
        self.assertEqual(unsatisfied(updated, "4.7.0"), [])

    def test_range_pin_needs_no_rewrite(self) -> None:
        updated, count = pin_text(RANGE_065, "4.7.0")
        self.assertEqual(count, 0)
        self.assertEqual(unsatisfied(updated, "4.7.0"), [])

    def test_unsatisfiable_range_is_reported(self) -> None:
        self.assertEqual(
            unsatisfied("Requires-Dist: nvidia-cutlass-dsl>=4.8\n", "4.7.0"),
            ["nvidia-cutlass-dsl>=4.8"],
        )

    def test_unhandled_pin_form_is_reported(self) -> None:
        # pin_text only rewrites `==`, so the check has to catch the rest.
        updated, count = pin_text("Requires-Dist: nvidia-cutlass-dsl~=4.9\n", "4.7.0")
        self.assertEqual(count, 0)
        self.assertEqual(
            unsatisfied(updated, "4.7.0"), ["nvidia-cutlass-dsl~=4.9"]
        )

    def test_unrelated_requirements_are_ignored(self) -> None:
        self.assertEqual(unsatisfied(UNRELATED, "4.7.0"), [])


if __name__ == "__main__":
    unittest.main()
