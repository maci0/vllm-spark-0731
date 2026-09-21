#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — MLA num_splits=8.
#
# B12X_MLA_SM120_NUM_SPLITS unset uses the wave-balanced heuristic.
# split1/2/4 sat in pin noise. Pin to 8. Overlay B12X_MLA_SPARSE decode
# goes through run_unified_decode. Dual-cache protocol is ~11 chunks.
# Same image, only this env.
#
#   configs/examples/mla-num-splits-8.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh split8 \
  "SERVE_EXTRA_ENV=B12X_MLA_SM120_NUM_SPLITS=8"
