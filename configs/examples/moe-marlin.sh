#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — MXFP4 MoE via Marlin.
#
# Oracle mapping includes `marlin` -> MARLIN. Device gate is SM75+.
# Never measured on this pin. Same image, same pin, only MOE_BACKEND.
# assert_stack only warns on unknown names.
#
#   configs/examples/moe-marlin.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moemar \
  "MOE_BACKEND=marlin"
