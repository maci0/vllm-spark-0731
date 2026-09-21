#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — B12X_W4A8_CONVERT_SCRATCH_MB=128.
#
# Native W4A8 prepare chunks experts through a 64 MiB scratch. Double
# it. Distinct from now4mat-rc2 (materialized off, a c6 cost). Same
# image, only this env.
#
#   configs/examples/moe-w4a8-convert-scratch-128-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh w4scr128-rc2 \
  "SERVE_EXTRA_ENV=B12X_W4A8_CONVERT_SCRATCH_MB=128"
