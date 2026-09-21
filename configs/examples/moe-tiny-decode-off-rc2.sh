#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — W4A8 tiny-decode off.
#
# Same as configs/examples/moe-tiny-decode-off.sh, new tag so the log
# does not collide with rc1 `notiny`. B12X_W4A8_TINY_DECODE default 1.
# On SM121 DSV4F (k=6144, n=1024) the tiny path is excluded for
# num_tokens>=3, so it only owns c1 (m=1). Same image, only this env.
#
#   configs/examples/moe-tiny-decode-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh notiny-rc2 \
  "SERVE_EXTRA_ENV=B12X_W4A8_TINY_DECODE=0"
