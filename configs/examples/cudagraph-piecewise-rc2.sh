#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — CUDAGRAPH_MODE=PIECEWISE.
#
# Standing is FULL_AND_PIECEWISE. FULL-only (cgfull / proto2-dg-cgfull)
# was a large negative. NONE is the price of capture. PIECEWISE-only is
# the leftover that still fires. Same image, only this field.
#
#   configs/examples/cudagraph-piecewise-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh cgpiece-rc2 \
  "CUDAGRAPH_MODE=PIECEWISE"
