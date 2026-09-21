#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — dynamic MoE down-scale on.
#
# B12X_ENABLE_DYNAMIC_DOWN_SCALE default False. Applied as
# `_dynamic_down_scale_enabled() and not is_w4a8`. MXFP4 experts are
# not W4A8, so this fires. Never isolated on main-030-rc1. Same image,
# only this env. SERVE_EXTRA_ENV because 05-serve.sh does not -e the
# name.
#
#   configs/examples/moe-down-scale.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh downsc \
  "SERVE_EXTRA_ENV=B12X_ENABLE_DYNAMIC_DOWN_SCALE=1"
