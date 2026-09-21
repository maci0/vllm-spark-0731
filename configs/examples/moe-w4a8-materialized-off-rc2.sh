#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — B12X_DYNAMIC_W4A8_MATERIALIZED=0.
#
# Native W4A8 dense-candidate path defaults MATERIALIZED on when the
# structural predicate is true. Distinct from closed work-source
# persistent_grid. Same image, only this env.
#
#   configs/examples/moe-w4a8-materialized-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh now4mat-rc2 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_MATERIALIZED=0"
