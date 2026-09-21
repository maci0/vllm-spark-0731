#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — disable micro MoE shared input.
#
# Same as configs/examples/micro-share-off.sh, new tag so the log does
# not collide with rc1 `noshare`. B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS
# default 1. Fires on W4A8 micro when activation is silu/relu2, m==1,
# and a1_gscale is a scalar. Protocol c1 is m=1 decode. Same image,
# only this env.
#
#   configs/examples/micro-share-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh noshare-rc2 \
  "SERVE_EXTRA_ENV=B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS=0"
