#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — dense per-row GS host chain.
#
# B12X_DENSE_PER_ROW_IN_KERNEL default 1. In-kernel per-row scaling is
# bit-identical to the host chain. Force 0 so decode uses the host
# amax/gs path instead of the GEMM-side kernel. Same image, only this
# env.
#
#   configs/examples/dense-per-row-host-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh noprki-rc2 \
  "SERVE_EXTRA_ENV=B12X_DENSE_PER_ROW_IN_KERNEL=0"
