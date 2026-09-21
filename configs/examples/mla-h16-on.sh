#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — DSV4 H16 native decode force-on.
#
# B12X_MLA_SM120_DSV4_H16_NATIVE unset = auto. noh16 forced 0 and sat in
# pin noise. Force 1 so small-row c1 also uses H16 (auto keeps H8 in the
# sub-wave latency regime). Overlay B12X_MLA_SPARSE decode goes through
# run_unified_decode. Same image, only this env.
#
#   configs/examples/mla-h16-on.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh yesh16 \
  "SERVE_EXTRA_ENV=B12X_MLA_SM120_DSV4_H16_NATIVE=1"
