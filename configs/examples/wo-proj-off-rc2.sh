#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — stock O-proj einsum.
#
# Overlay default VLLM_USE_B12X_WO_PROJECTION=1 fuses inv-RoPE FP8 + bmm.
# =0 returns None and the layer uses einsum. stockops2 mixed this with
# sparse-indexer-off. Never isolated on main-030-rc1. Same image, only
# this env. 05-serve.sh already -e the name.
#
#   configs/examples/wo-proj-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nowo-rc2 \
  "VLLM_USE_B12X_WO_PROJECTION=0"
