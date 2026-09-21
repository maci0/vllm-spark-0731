#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — b12x dynamic MoE work source.
#
# Same as configs/examples/moe-work-source-grid.sh, new tag so the log
# does not collide with rc1 `moework`. Library default is
# materialized_queue. persistent_grid is documented arithmetic striding.
# ready_queue JIT-dies. Same image, only this env.
#
#   configs/examples/moe-work-source-grid-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moework-rc2 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_WORK_SOURCE=persistent_grid"
