#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — b12x MXFP4 with BF16 activations.
#
# Same as configs/examples/moe-a16-030.sh, new tag so the log does not
# collide with rc1 `moea16`. VLLM_B12X_MOE_FP4_FORCE_A16=1 selects
# B12X_MXFP4_BF16. Anemll maps flashinfer_b12x onto that backend.
# 05-serve.sh already forwards the name.
#
#   configs/examples/moe-a16-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moea16-rc2 \
  "VLLM_B12X_MOE_FP4_FORCE_A16=1"
