#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — DSpark k=6.
#
# Same as configs/examples/spec-k6.sh, new tag so the log does not
# collide with rc1 `k6`. k=7 is the pin. Capture 48 covers 6*(6+1)=42.
# EXTRA overrides run-arm's hardcoded NUM_SPECULATIVE_TOKENS=7.
#
#   configs/examples/spec-k6-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh k6-rc2 \
  "NUM_SPECULATIVE_TOKENS=6"
