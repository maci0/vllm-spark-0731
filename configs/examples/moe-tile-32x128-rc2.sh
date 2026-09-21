#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — b12x dynamic MoE tile 32x128.
#
# Same as configs/examples/moe-tile-32x128.sh, new tag so the log does
# not collide with rc1 `moetile32`. Auto planner returns (16, 128) for
# the protocol band. 64x128 and 128x128 already sat in pin noise on this
# pin. M32 is the remaining W4A8 ladder step. Same image, only this env.
#
#   configs/examples/moe-tile-32x128-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moetile32-rc2 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=32x128"
