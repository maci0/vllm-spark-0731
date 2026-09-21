#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — b12x dynamic MoE work source.
#
# Production default is materialized_queue. persistent_grid is the library's
# documented A/B (arithmetic striding). Same image, same pin, only this env.
# Must go through SERVE_EXTRA_ENV.
#
#   configs/examples/moe-work-source-grid.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moework \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_WORK_SOURCE=persistent_grid"
