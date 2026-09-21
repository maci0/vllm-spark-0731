#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — MLA num_splits=1.
#
# B12X_MLA_SM120_NUM_SPLITS unset uses the FlashInfer-ported wave-balanced
# heuristic. Overlay B12X_MLA_SPARSE decode goes through
# compressed_sparse_mla.run -> run_unified_decode, which reads this env
# per call. Pin to 1 (no split-K). Never isolated on main-030-rc1.
# Same image, only this env. SERVE_EXTRA_ENV because 05-serve.sh does
# not -e the name.
#
#   configs/examples/mla-num-splits-1.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh split1 \
  "SERVE_EXTRA_ENV=B12X_MLA_SM120_NUM_SPLITS=1"
