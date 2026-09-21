#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — DSpark k=5.
#
# Same as configs/examples/spec-k5.sh, new tag so the log does not
# collide with rc1 `k5`. Checkpoint DSpark block is k=5. Pin is k=7.
# Capture 48 covers 6*(5+1)=36. rc1 k5 won c6 but lost tokens/step.
# k6-rc2 held quality; re-measure k=5 on this pin.
#
#   configs/examples/spec-k5-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh k5-rc2 \
  "NUM_SPECULATIVE_TOKENS=5"
