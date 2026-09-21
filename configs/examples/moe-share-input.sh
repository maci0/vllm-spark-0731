#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — b12x W4A8 share-input on.
#
# Default is on only when dense or decode candidate is true. Decode
# candidate requires routed_rows <= 64, so only c1. c3-c6 (144-288 rows)
# default off. Forcing B12X_DYNAMIC_W4A8_SHARE_INPUT=1 is the first time
# that producer is measured on the protocol's mid band. Same image, same
# pin, only this env. Must go through SERVE_EXTRA_ENV.
#
#   configs/examples/moe-share-input.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moeshare \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_SHARE_INPUT=1"
