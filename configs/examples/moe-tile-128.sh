#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — dynamic MoE tile 128x128.
#
# Auto planner returns (16, 128) for the protocol band (24-288 routed
# rows) on w4a8_mx. 32x128 (moetile32) and 64x128 (proto2-moetile) were
# pin noise. 128x128 is the remaining ladder step (also in the W4A8
# dense-candidate set). Same image, only this env.
#
#   configs/examples/moe-tile-128.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh tile128 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=128x128"
