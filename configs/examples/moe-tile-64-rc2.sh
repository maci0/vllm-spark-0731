#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — B12X_DYNAMIC_TILE_MN=64x128.
#
# Auto W4A8 planner on GB10 uses M16 for sparse decode and M32 for the
# DSV4 TP2 verify band. M64 is the remaining ladder step (32x128 and
# 128x128 already measured on this pin). Same image, only this env.
#
#   configs/examples/moe-tile-64-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moetile64-rc2 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=64x128"
