#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — MXFP4 MoE via humming.
#
# Oracle mapping includes `humming` -> HUMMING. Device gate is SM75+.
# Package is in the image (`has_humming()` true). Prior `moe_humming`
# (353.2) was on the old protog pin, never re-measured on main-030-rc1.
# assert_stack only warns on unknown names. Same image, same pin, only
# MOE_BACKEND.
#
#   configs/examples/moe-humming-030.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh moehum \
  "MOE_BACKEND=humming"
