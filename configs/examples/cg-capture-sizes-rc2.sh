#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — explicit CUDA-graph capture sizes.
#
# Pin max_cudagraph_capture_size=48, capture_sizes unset (vLLM infers).
# Protocol tokens per step: c1=8, c3=24, c5=40, c6=48 (seqs * (k+1)).
# Explicit list so those four graphs are captured even if auto-padding
# skips a protocol size. Same image, only this compilation field.
# Compact JSON so EXTRA word-split stays one assignment.
#
#   configs/examples/cg-capture-sizes-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh cgsizes-rc2 \
  "CUDAGRAPH_CAPTURE_SIZES=[8,24,40,48]"
