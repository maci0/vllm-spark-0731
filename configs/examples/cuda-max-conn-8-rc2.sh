#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — CUDA_DEVICE_MAX_CONNECTIONS=8.
#
# Same as configs/examples/cuda-max-conn-8.sh, new tag so the log does
# not collide with rc1 `conn8`. env.spark.sh defaults this to 1. CUDA
# default is 8. The anemll recipe does not set it. 05-serve.sh already
# -e the name, so EXTRA overrides.
#
#   configs/examples/cuda-max-conn-8-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh conn8-rc2 \
  "CUDA_DEVICE_MAX_CONNECTIONS=8"
