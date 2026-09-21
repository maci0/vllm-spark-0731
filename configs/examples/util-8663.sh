#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — restore pre-profiler KV budget.
#
# v0.30 CUDA-graph memory profiling (on by default) maps
# --gpu-memory-utilization=0.8389 to effective 0.8115. The engine itself
# names 0.8663 as the util that keeps the same KV size as 0.8389 without
# the profiler. run-arm.sh hardcodes 0.8389; EXTRA overrides it because
# that assignment is exported into both serve invocations.
#
#   configs/examples/util-8663.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh util8663 \
  "GPU_MEMORY_UTILIZATION=0.8663"
