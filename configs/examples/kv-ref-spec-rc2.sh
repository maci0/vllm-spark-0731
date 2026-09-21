#!/usr/bin/env bash
# Recipe: reproduce the reference's exact KV spec — MAX_MODEL_LEN=262144 at
# GPU_MEMORY_UTILIZATION=0.86. The reference (ghcr.io/anemll/...:0.1.1) serves
# 262144 at util 0.82, but our layout needs a larger KV pool: 262144 at our
# standing util 0.8389 dies (round 111); at 0.86 it serves (round 113).
# Same one-variable shape as the 245760 capacity arm, at the reference context.
#
#   configs/examples/kv-ref-spec-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh kv262u086-rc2 \
  "MAX_MODEL_LEN=262144 GPU_MEMORY_UTILIZATION=0.86"
