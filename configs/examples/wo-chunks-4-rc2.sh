#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — WO MXFP8 quant chunks 4.
#
# Eager profile priced live c1 WO at 57.4 ms (38 % of 150.6 ms).
# B12X_WO_QUANT_CHUNKS_PER_PROGRAM default 16 (legal 1/2/4/8/16/32).
# Chunks 8 and 32 were pin noise. 4 is the remaining smaller step.
# Same image, only this env.
#
#   configs/examples/wo-chunks-4-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh wochunk4-rc2 \
  "SERVE_EXTRA_ENV=B12X_WO_QUANT_CHUNKS_PER_PROGRAM=4"
