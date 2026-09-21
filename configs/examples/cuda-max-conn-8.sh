#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — CUDA_DEVICE_MAX_CONNECTIONS=8.
#
# env.spark.sh defaults this to 1. CUDA default is 8. The anemll recipe
# does not set it. 05-serve.sh already -e the name, so EXTRA overrides.
# Same image, only this env.
#
#   configs/examples/cuda-max-conn-8.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh conn8 \
  "CUDA_DEVICE_MAX_CONNECTIONS=8"
