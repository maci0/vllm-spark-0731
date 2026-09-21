#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — disable b12x MoE fast-math.
#
# Same as configs/examples/moe-fast-math-off.sh, new tag so the log does
# not collide with rc1 `nofast`. B12X_FAST_MATH default True; keyed into
# the dynamic W4A8 kernel cache. Same image, only this env.
#
#   configs/examples/moe-fast-math-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nofast-rc2 \
  "SERVE_EXTRA_ENV=B12X_FAST_MATH=0"
