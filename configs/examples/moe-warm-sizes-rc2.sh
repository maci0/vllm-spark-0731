#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — B12X_MOE_WARM_MS=8,24,40,48.
#
# Default MoE warm sizes are 1,2,3,4,5,8 plus inferred CUDA-graph
# capture sizes. Protocol tokens: c1=8, c3=24, c5=40, c6=48. Explicit
# list warms those four Ms before capture even when capture_sizes is
# unset. Comma form so SERVE_EXTRA_ENV word-split stays one kv. Same
# image, only this env.
#
#   configs/examples/moe-warm-sizes-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moewarm-rc2 \
  "SERVE_EXTRA_ENV=B12X_MOE_WARM_MS=8,24,40,48"
