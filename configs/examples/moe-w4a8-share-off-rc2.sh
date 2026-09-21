#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — W4A8 shared-input producer off.
#
# Same as configs/examples/moe-w4a8-share-off.sh, new tag so the log
# does not collide with rc1 `now4sh`. DSV4 MXFP4 experts map to
# quant_mode=w4a8_mx. Share-input default is on when dense or decode
# candidate is true (`moeshare-rc2` forced it on). This isolates the
# kill switch. Same image, only this env.
#
#   configs/examples/moe-w4a8-share-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh now4sh-rc2 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_SHARE_INPUT=0"
