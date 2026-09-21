#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — indexer direct-K score off.
#
# B12X_INDEXER_DIRECT_K default 1. Kill-switch forces every fused-indexer
# variant back to the staged pipeline (and restores v2 cache keys).
# Overlay scores DSA indexer logits via logits_paged. Same image, only
# this env. Distinct from closed VLLM_B12X_INDEXER_DIRECT_GATHER.
#
#   configs/examples/indexer-direct-k-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh noidxk-rc2 \
  "SERVE_EXTRA_ENV=B12X_INDEXER_DIRECT_K=0"
