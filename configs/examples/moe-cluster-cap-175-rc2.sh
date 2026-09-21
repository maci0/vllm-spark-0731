#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — b12x dynamic MoE cluster cap 175.
#
# Same as configs/examples/moe-cluster-cap-175.sh, new tag so the log
# does not collide with rc1 `moecap175`. Decode policy is a flat cap of
# 188 up to 640 routed rows. Reference static kernel uses 175 at the c6
# band (288 rows). Same image, only this env.
#
#   configs/examples/moe-cluster-cap-175-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moecap175-rc2 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=175"
