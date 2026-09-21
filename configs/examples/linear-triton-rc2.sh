#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — FP8 linear via Triton.
#
# Same as configs/examples/linear-triton.sh, new tag so the log does
# not collide with rc1 `lintri`. Linear family: b12x works, deep_gemm
# worse, humming/marlin die on O-proj. TritonFp8BlockScaledMMKernel
# is_supported returns True on CUDA. Same image, only LINEAR_BACKEND.
# MoE stays b12x.
#
#   configs/examples/linear-triton-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh lintri-rc2 \
  "LINEAR_BACKEND=triton"
