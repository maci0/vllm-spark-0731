#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — disable dense GEMM split-K turbo.
#
# B12X_DENSE_SPLITK_TURBO default 1. Selects atomic-BF16 reduction when
# split-K slices > 1. Decode policy picks 2-way split-K for m in 2..6
# and k >= 4096 (our FP8 linear decode band). =0 keeps 2-way split-K
# but drops the atomic-BF16 path. Same image, only this env.
# SERVE_EXTRA_ENV because 05-serve.sh does not -e this name.
#
#   configs/examples/splitk-turbo-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh noturbo \
  "SERVE_EXTRA_ENV=B12X_DENSE_SPLITK_TURBO=0"
