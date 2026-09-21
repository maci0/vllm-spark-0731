#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — W4A8 shared-input producer off.
#
# DSV4 MXFP4 experts map to quant_mode=w4a8_mx. Share-input default is on
# when dense or decode candidate is true (`moeshare` forced it on and sat
# in pin noise). This isolates the kill switch. Same image, only this env.
# SERVE_EXTRA_ENV because 05-serve.sh does not -e the name.
#
#   configs/examples/moe-w4a8-share-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh now4sh \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_SHARE_INPUT=0"
