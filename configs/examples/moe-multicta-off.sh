#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — disable dynamic MoE multi-CTA.
#
# B12X_DYNAMIC_ENABLE_MULTICTA default 1. On W4A8 decode, off forces
# effective_mac=1. On, DSV4F TP2 decode (E=256, k=6144, n=1024, 24-48
# routed rows) caps at 24 resident CTAs and may double occupancy for
# compact tile_m<=32. Same image, only this env. SERVE_EXTRA_ENV because
# 05-serve.sh does not -e this name.
#
#   configs/examples/moe-multicta-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nomulti \
  "SERVE_EXTRA_ENV=B12X_DYNAMIC_ENABLE_MULTICTA=0"
