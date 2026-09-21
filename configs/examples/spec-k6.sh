#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — DSpark k=6.
#
# k=5 on this pin won c6 (+18 %) but lost tokens/step (4.031 vs floor
# 4.414) and the sum stayed inside keep-spread. k=7 is the pin. k=6 sits
# between: capture 48 covers 6*(6+1)=42. EXTRA overrides run-arm's
# hardcoded NUM_SPECULATIVE_TOKENS=7. Same image, only this env.
#
#   configs/examples/spec-k6.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh k6 \
  "NUM_SPECULATIVE_TOKENS=6"
