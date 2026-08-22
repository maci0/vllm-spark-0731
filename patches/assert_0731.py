#!/usr/bin/env python3
"""Refuse any checkpoint that is not DeepSeek-V4-Flash-0731."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PIN_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
PIN_NAME = "DeepSeek-V4-Flash-0731"
ARCH = "DeepseekV4ForCausalLM"
MODEL_TYPE = "deepseek_v4"


def load_config(model_dir: pathlib.Path) -> dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"missing {cfg_path}")
    return json.loads(cfg_path.read_text())


def check(model_dir: pathlib.Path) -> None:
    blob = str(model_dir)
    looks_preview = (
        "DeepSeek-V4-Flash" in blob
        and "0731" not in blob
        and "Flash-DSpark" not in blob
    )
    if looks_preview:
        raise SystemExit(
            f"refusing {model_dir}: this recipe is pinned to {PIN_ID}"
        )
    cfg = load_config(model_dir)
    arch = cfg.get("architectures") or []
    if ARCH not in arch:
        raise SystemExit(f"architectures={arch!r}, expected {ARCH}")
    if cfg.get("model_type") != MODEL_TYPE:
        raise SystemExit(f"model_type={cfg.get('model_type')!r}, expected {MODEL_TYPE}")
    if cfg.get("dspark_block_size") != 5:
        raise SystemExit(
            "checkpoint is not 0731: expected dspark_block_size=5 "
            f"(got {cfg.get('dspark_block_size')!r})"
        )
    if cfg.get("num_nextn_predict_layers") != 1:
        raise SystemExit(
            "checkpoint is not 0731: expected num_nextn_predict_layers=1 "
            f"(got {cfg.get('num_nextn_predict_layers')!r})"
        )
    ratios = cfg.get("compress_ratios") or []
    if not ratios or ratios[:4] not in ([0, 0, 4, 128],):
        print(
            f"warn: unexpected Flash compress_ratios prefix {ratios[:6]!r}",
            file=sys.stderr,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "pin": PIN_ID,
                "architecture": ARCH,
                "layers": cfg.get("num_hidden_layers"),
                "experts": cfg.get("n_routed_experts"),
                "index_topk": cfg.get("index_topk"),
                "mtp_layers": cfg.get("num_nextn_predict_layers"),
                "dspark_block_size": cfg.get("dspark_block_size"),
                "expert_dtype": cfg.get("expert_dtype"),
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=pathlib.Path)
    args = parser.parse_args()
    check(args.model_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
