#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — disable CUDA-graph memory profiler.
#
# Same as configs/examples/memprof-off.sh, new tag so the log does not
# collide with rc1 `memprof0`. v0.30 defaults
# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=True and maps 0.8389 to
# effective 0.8115. util8663-rc2 raised the util number. This turns the
# profiler off at standing 0.8389. SERVE_EXTRA_ENV because 05-serve.sh
# does not -e the name.
#
#   configs/examples/memprof-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh memprof0-rc2 \
  "SERVE_EXTRA_ENV=VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
