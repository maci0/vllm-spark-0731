#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — GQA6 compact-sync off.
#
# B12X_PAGED_GQA6_COMPACT_SYNC default 1. Overlay B12X_MLA_SPARSE decode
# goes through paged indexer. DSV4 is GQA 6. Force 0 to isolate the
# compact-sync decode fastpath. Same image, only this env.
#
#   configs/examples/gqa6-compact-sync-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nogqa6-rc2 \
  "SERVE_EXTRA_ENV=B12X_PAGED_GQA6_COMPACT_SYNC=0"
