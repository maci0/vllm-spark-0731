#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — b12x dense GEMM 24-atom MMA.
#
# Same as configs/examples/dense-atom-24.sh, new tag so the log does
# not collide with rc1 `atom24`. Default 0. Experimental atom choice;
# changes generated code and is keyed into the persistent compile cache.
# Hits B12xFp8BlockScaled via dense_gemm. Same image, only this env.
#
#   configs/examples/dense-atom-24-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh atom24-rc2 \
  "SERVE_EXTRA_ENV=B12X_DENSE_ATOM_24=1"
