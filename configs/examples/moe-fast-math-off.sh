#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — disable b12x MoE fast-math.
#
# B12X_FAST_MATH default True; keyed into the dynamic W4A8 kernel cache.
# Never measured on this pin. Same image, only this env. SERVE_EXTRA_ENV
# because 05-serve.sh does not -e this name.
#
# Warm FLASHINFER MLA autotune skipped: 0.7.0 cache JSON is metadata-only
# (499 B, no tactic entries). A rerun would still hit the heuristic.
#
#   configs/examples/moe-fast-math-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nofast \
  "SERVE_EXTRA_ENV=B12X_FAST_MATH=0"
