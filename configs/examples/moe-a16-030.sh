#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — b12x MXFP4 with BF16 activations.
#
# Reference (anemll 0.1.1, b12x 0.15.3) logs Using 'B12X_MXFP4' and maps
# flashinfer_b12x onto that backend (activation_key None = BF16). Our pin
# prefers B12X_MXFP4_MXFP8. VLLM_B12X_MOE_FP4_FORCE_A16=1 selects
# B12X_MXFP4_BF16 on this oracle. proto2-a16 was a wash on the old pin;
# never re-measured on main-030-rc1. Same image, same pin, only this env.
# 05-serve.sh already forwards the name.
#
#   configs/examples/moe-a16-030.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moea16 \
  "VLLM_B12X_MOE_FP4_FORCE_A16=1"
