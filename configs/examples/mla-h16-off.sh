#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — DSV4 H16 native decode off.
#
# B12X_MLA_SM120_DSV4_H16_NATIVE unset = auto. On Spark (48 SMs) auto
# turns H16 on for many-chunk / batched-row decode. Overlay
# B12X_MLA_SPARSE decode goes through run_unified_decode. Force 0 to
# keep H8. Never isolated on main-030-rc1. Same image, only this env.
# SERVE_EXTRA_ENV because 05-serve.sh does not -e the name.
#
#   configs/examples/mla-h16-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh noh16 \
  "SERVE_EXTRA_ENV=B12X_MLA_SM120_DSV4_H16_NATIVE=0"
