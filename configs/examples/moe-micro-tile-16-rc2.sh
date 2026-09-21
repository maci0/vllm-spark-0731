#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — micro MoE tile 16x128.
#
# B12X_MOE_TILE_MN gates _select_micro_mma_tiler_mn (protocol c1, below
# the 64-row cutover). Default micro tile is 64x128. 32x128 was pin
# noise; 128x128 died at the KV floor. 16x128 is the remaining smaller
# step (same as the dynamic auto planner). Same image, only this env.
#
#   configs/examples/moe-micro-tile-16-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moetilem16-rc2 \
  "SERVE_EXTRA_ENV=B12X_MOE_TILE_MN=16x128"
