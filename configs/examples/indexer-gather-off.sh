#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — packed-indexer gather off.
#
# VLLM_B12X_INDEXER_DIRECT_GATHER default 1 in 05-serve.sh. Proto-era +27 %
# win. Never isolated off on main-030-rc1. Overlay scores logits via
# logits_paged; gather is the packed-sidecar insert path. Same image, only
# this env. 05-serve.sh already -e the name.
#
#   configs/examples/indexer-gather-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nogath \
  "VLLM_B12X_INDEXER_DIRECT_GATHER=0"
