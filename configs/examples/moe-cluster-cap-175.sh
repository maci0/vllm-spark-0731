#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — b12x dynamic MoE cluster cap 175.
#
# Our decode policy is a flat cap of 188 for every routed-row count up to 640.
# The reference's static kernel uses 175 at the c6 band (288 routed rows).
# Offline sweep (`outputs/driver/one-off/moe-cluster-cap.log`) measured cap 175
# at 0.941x the 188 time, bit-identical outputs. This is the protocol test of
# that number: same image, same pin, only B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS.
#
# Must go through SERVE_EXTRA_ENV: 05-serve.sh only forwards an explicit -e list.
#
#   configs/examples/moe-cluster-cap-175.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moecap175 \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=175"
