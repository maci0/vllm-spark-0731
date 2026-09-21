#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — b12x dense GEMM 24-atom MMA.
#
# B12X_DENSE_ATOM_24 default 0. Experimental atom choice; changes generated
# code and is keyed into the persistent compile cache. Hits B12xFp8BlockScaled
# via dense_gemm. Same image, same pin, only this env. SERVE_EXTRA_ENV because
# 05-serve.sh does not -e this name.
#
#   configs/examples/dense-atom-24.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh atom24 \
  "SERVE_EXTRA_ENV=B12X_DENSE_ATOM_24=1"
