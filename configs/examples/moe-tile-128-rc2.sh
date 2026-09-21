#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — dynamic MoE tile 128x128.
#
# Same as configs/examples/moe-tile-128.sh, new tag so the log does not
# collide with rc1 `tile128`. Auto planner returns (16, 128). 32x128 and
# 64x128 were pin noise. 128x128 is the remaining W4A8 dense-candidate
# ladder step. Same image, only this env.
#
#   configs/examples/moe-tile-128-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh tile128-rc2 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=128x128"
