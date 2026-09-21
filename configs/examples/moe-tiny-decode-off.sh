#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — W4A8 tiny-decode off.
#
# B12X_W4A8_TINY_DECODE default 1. On SM121 DSV4F (k=6144, n=1024) the
# tiny path is excluded for num_tokens>=3, so it only owns c1 (m=1).
# Never isolated off on main-030-rc1. Same image, only this env.
# SERVE_EXTRA_ENV because 05-serve.sh does not -e the name.
#
#   configs/examples/moe-tiny-decode-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh notiny \
  "SERVE_EXTRA_ENV=B12X_W4A8_TINY_DECODE=0"
