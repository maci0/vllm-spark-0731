#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — micro MoE compiled-kernel reuse off.
#
# B12X_MICRO_REUSE_COMPILED default 1. Micro owns the tiny tail below the
# 64 routed-row cutover (protocol c1). Force 0 so each launch recompiles
# instead of hitting _MICRO_KERNEL_CACHE. Same image, only this env.
#
#   configs/examples/micro-reuse-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh noreuse-rc2 \
  "SERVE_EXTRA_ENV=B12X_MICRO_REUSE_COMPILED=0"
