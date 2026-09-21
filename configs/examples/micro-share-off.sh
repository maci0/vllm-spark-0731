#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — disable micro MoE shared input.
#
# B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS default 1. Fires on W4A8 micro when
# activation is silu/relu2, m==1, and a1_gscale is a scalar. Protocol c1
# is m=1 decode. Same image, only this env. SERVE_EXTRA_ENV because
# 05-serve.sh does not -e this name.
#
#   configs/examples/micro-share-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh noshare \
  "SERVE_EXTRA_ENV=B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS=0"
