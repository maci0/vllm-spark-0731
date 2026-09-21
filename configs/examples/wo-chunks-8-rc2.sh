#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — WO MXFP8 quant chunks 8.
#
# Eager profile (profeager-rc2) prices live c1: FFN 80.3 ms (53 %) and
# WO 57.4 ms (38 %) of a 150.6 ms execute_model. B12X_WO_QUANT_CHUNKS
# _PER_PROGRAM default 16 (legal 1/2/4/8/16/32). Half the default.
# Same image, only this env. WO overlay stays on (nowo-rc2 already
# measured overlay-off as a cost, not a keep).
#
#   configs/examples/wo-chunks-8-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh wochunk8-rc2 \
  "SERVE_EXTRA_ENV=B12X_WO_QUANT_CHUNKS_PER_PROGRAM=8"
