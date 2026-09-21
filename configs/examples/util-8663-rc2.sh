#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — restore pre-profiler KV budget.
#
# Same as configs/examples/util-8663.sh, new tag so the log does not
# collide with rc1 `util8663`. v0.30 CUDA-graph memory profiling maps
# 0.8389 to effective ~0.8115. Engine names 0.8663 as the util that keeps
# the same KV size as 0.8389 without the profiler. rc1 `moemar` served;
# `moemar-rc2` died at 9.48 vs 9.27 GiB. EXTRA overrides run-arm's 0.8389.
#
#   configs/examples/util-8663-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh util8663-rc2 \
  "GPU_MEMORY_UTILIZATION=0.8663"
