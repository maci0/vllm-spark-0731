#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — disable CUDA-graph memory profiler.
#
# v0.30 defaults VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=True and maps
# --gpu-memory-utilization=0.8389 to effective 0.8115. util8663 compensated
# by raising the util number (and still sat in pin noise). This turns the
# profiler off at the same 0.8389, which is the actual new default vs the
# proto-era pin. Must go through SERVE_EXTRA_ENV: 05-serve.sh does not
# forward this name on the explicit -e list.
#
#   configs/examples/memprof-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh memprof0 \
  "SERVE_EXTRA_ENV=VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
