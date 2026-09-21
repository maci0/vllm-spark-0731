#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — BF16 GEMV small-N off.
#
# B12X_DISABLE_BF16_GEMV default unset (GEMV on). Small-N path covers
# N<=1024, K>=1024. DSV4 O-proj is o_lora_rank 1024 vs hidden 4096.
# Force 1 so those matmuls fall back to F.linear. Same image, only
# this env.
#
#   configs/examples/bf16-gemv-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nogemv-rc2 \
  "SERVE_EXTRA_ENV=B12X_DISABLE_BF16_GEMV=1"
