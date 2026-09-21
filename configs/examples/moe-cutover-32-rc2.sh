#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — MoE micro/dynamic cutover 32.
#
# Native b12x MoE is micro vs dynamic only. Dispatch:
#   num_tokens<=8 and routed_rows < B12X_MICRO_DYNAMIC_CUTOVER_PAIRS
# Default cutover 64. Protocol c1 is 8 tok x topk 6 = 48 pairs, so only
# c1 is micro. Cutover 32 sends c1 to dynamic (48 < 32 is false).
# c3/c5/c6 stay dynamic (tokens>8). Same image, only this env.
#
#   configs/examples/moe-cutover-32-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moecut32-rc2 \
  "SERVE_EXTRA_ENV=B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=32"
