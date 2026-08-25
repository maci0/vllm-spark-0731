import glob
import os
import py_compile
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES_DIR = os.path.join(ROOT, "patches", "upstream")
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class TestUpstreamPatches(unittest.TestCase):
    """Validate that all upstream patch diffs parse cleanly and have valid hunk line counts."""

    def test_patch_files_exist(self):
        patch_files = glob.glob(os.path.join(PATCHES_DIR, "*.diff")) + glob.glob(os.path.join(PATCHES_DIR, "*.patch"))
        self.assertGreater(len(patch_files), 0, "No patch diff files found in patches/upstream")

    def test_patch_hunks_parse_and_counts_match(self):
        patch_files = glob.glob(os.path.join(PATCHES_DIR, "*.diff")) + glob.glob(os.path.join(PATCHES_DIR, "*.patch"))
        for pf in patch_files:
            rel_name = os.path.relpath(pf, ROOT)
            with open(pf, "r", encoding="utf-8") as f:
                lines = f.readlines()

            in_hunk = False
            expected_old = 0
            expected_new = 0
            actual_old = 0
            actual_new = 0
            hunk_idx = 0

            for line_no, line in enumerate(lines, 1):
                if line.startswith("@@ "):
                    if in_hunk:
                        self.assertEqual(
                            actual_old,
                            expected_old,
                            f"{rel_name}: hunk {hunk_idx} old count mismatch: expected {expected_old}, got {actual_old} (line {line_no})",
                        )
                        self.assertEqual(
                            actual_new,
                            expected_new,
                            f"{rel_name}: hunk {hunk_idx} new count mismatch: expected {expected_new}, got {actual_new} (line {line_no})",
                        )
                    m = HUNK_HEADER_RE.match(line)
                    self.assertIsNotNone(m, f"{rel_name}:{line_no} invalid hunk header: {line}")
                    old_start, old_cnt, new_start, new_cnt = m.groups()
                    expected_old = int(old_cnt) if old_cnt is not None else 1
                    expected_new = int(new_cnt) if new_cnt is not None else 1
                    actual_old = 0
                    actual_new = 0
                    in_hunk = True
                    hunk_idx += 1
                elif in_hunk:
                    if line.startswith("diff ") or line.startswith("--- "):
                        self.assertEqual(
                            actual_old,
                            expected_old,
                            f"{rel_name}: hunk {hunk_idx} old count mismatch: expected {expected_old}, got {actual_old} (line {line_no})",
                        )
                        self.assertEqual(
                            actual_new,
                            expected_new,
                            f"{rel_name}: hunk {hunk_idx} new count mismatch: expected {expected_new}, got {actual_new} (line {line_no})",
                        )
                        in_hunk = False
                    elif line.startswith("+++ "):
                        pass
                    elif line.startswith("-"):
                        actual_old += 1
                    elif line.startswith("+"):
                        actual_new += 1
                    elif line.startswith(" "):
                        actual_old += 1
                        actual_new += 1
                    elif line.startswith("\\ No newline at end of file"):
                        pass
                    elif line.strip() == "":
                        # Trailing empty or blank context line
                        actual_old += 1
                        actual_new += 1

            if in_hunk:
                self.assertEqual(
                    actual_old,
                    expected_old,
                    f"{rel_name}: hunk {hunk_idx} old count mismatch: expected {expected_old}, got {actual_old}",
                )
                self.assertEqual(
                    actual_new,
                    expected_new,
                    f"{rel_name}: hunk {hunk_idx} new count mismatch: expected {expected_new}, got {actual_new}",
                )
            self.assertGreater(hunk_idx, 0, f"{rel_name} has no valid hunks")

    def test_b12x_utils_main_compiles(self):
        b12x_utils_path = os.path.join(PATCHES_DIR, "b12x-utils-main.py")
        if os.path.exists(b12x_utils_path):
            py_compile.compile(b12x_utils_path, doraise=True)


if __name__ == "__main__":
    unittest.main()
