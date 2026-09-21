#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — VLLM_USE_AOT_COMPILE=0.
#
# Standing pin defaults AOT on. proto2-compile tried this on an earlier
# pin and still resolved CompilationMode.NONE. Re-measure on this pin.
# Same image, only this env.
#
#   configs/examples/aot-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh noaot-rc2 \
  "VLLM_USE_AOT_COMPILE=0"
