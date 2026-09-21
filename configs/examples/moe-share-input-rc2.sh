#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — b12x W4A8 share-input on.
#
# Same as configs/examples/moe-share-input.sh, new tag so the log does
# not collide with rc1 `moeshare`. Default is on only when dense or
# decode candidate is true. Decode candidate requires routed_rows <= 64,
# so only c1. Forcing SHARE_INPUT=1 is the first time that producer is
# measured on this pin's c3-c6 band. Same image, only this env.
#
#   configs/examples/moe-share-input-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moeshare-rc2 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_SHARE_INPUT=1"
