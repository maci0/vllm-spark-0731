#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches"))
from assert_0731 import check  # noqa: E402


class TestAssert0731(unittest.TestCase):
    def _write(self, tmp: Path, **cfg: object) -> Path:
        blob = {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "dspark_block_size": 5,
            "num_nextn_predict_layers": 1,
            "compress_ratios": [0, 0, 4, 128],
            "num_hidden_layers": 61,
        }
        blob.update(cfg)
        (tmp / "config.json").write_text(json.dumps(blob))
        return tmp

    def test_ok(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            check(self._write(Path(raw)))

    def test_refuse_missing_dspark(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = self._write(Path(raw), dspark_block_size=None)
            with self.assertRaises(SystemExit):
                check(tmp)

    def test_refuse_preview_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="DeepSeek-V4-Flash-") as raw:
            tmp = Path(raw)
            if "0731" in str(tmp):
                return
            (tmp / "config.json").write_text("{}")
            with self.assertRaises(SystemExit):
                check(tmp)


if __name__ == "__main__":
    unittest.main()
