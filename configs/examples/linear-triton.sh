#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — FP8 linear via Triton.
#
# Linear family so far: b12x works, deep_gemm worse/unstable, humming/marlin
# die on O-proj, torch block-scaled is Hopper-only, flashinfer_b12x is
# NVFP4-only. TritonFp8BlockScaledMMKernel.is_supported returns True on
# CUDA. Same image, same pin, only LINEAR_BACKEND. MoE stays b12x.
#
#   configs/examples/linear-triton.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh lintri \
  "LINEAR_BACKEND=triton"
