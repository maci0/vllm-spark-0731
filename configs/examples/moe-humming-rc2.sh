#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — MXFP4 MoE via humming.
#
# Same as configs/examples/moe-humming-030.sh, new tag so the log does
# not collide with rc1 `moehum`. Image humming-kernels is 0.1.15 (was
# 0.1.13 on rc1). Same image, only MOE_BACKEND.
#
#   configs/examples/moe-humming-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moehum-rc2 \
  "MOE_BACKEND=humming"
