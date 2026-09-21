#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — indexer stream scorer off.
#
# B12X_INDEXER_STREAM_SCORER default True (unset = on). Overlay scores
# DSA indexer logits via logits_paged -> run_paged_logits_kernel. The
# tiled/supertile path gates stream-scorer on this env. Never isolated
# off on main-030-rc1. Same image, only this env. SERVE_EXTRA_ENV
# because 05-serve.sh does not -e the name.
#
#   configs/examples/indexer-stream-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nostream \
  "SERVE_EXTRA_ENV=B12X_INDEXER_STREAM_SCORER=0"
