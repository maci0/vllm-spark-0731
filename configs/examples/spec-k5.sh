#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — DSpark k=5.
#
# Checkpoint DSpark block is k=5. Protocol pin uses k=7. proto5 on the
# old pin was rejected because capture missed 5x6=30. This pin captures
# 48, which covers 6*(5+1)=36. EXTRA overrides run-arm's hardcoded
# NUM_SPECULATIVE_TOKENS=7. Same image, only this env.
#
#   configs/examples/spec-k5.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh k5 \
  "NUM_SPECULATIVE_TOKENS=5"
