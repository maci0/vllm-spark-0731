#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — b12x dynamic MoE tile 32x128.
#
# Auto planner returns (16, 128) for the whole protocol band (48-288 routed
# rows, 256 experts). 64x128 was already a wash (`proto2-moetile`). The
# W4A8 ladder's next tactic is M32, which the comments claim reduces
# expert-weight streaming. Same image, same pin, only B12X_DYNAMIC_TILE_MN.
# Must go through SERVE_EXTRA_ENV.
#
#   configs/examples/moe-tile-32x128.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moetile32 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=32x128"
