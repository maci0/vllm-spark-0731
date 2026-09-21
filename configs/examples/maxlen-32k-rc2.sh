#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — max_model_len 32768.
#
# Pin is 65536. Anemll uses 262144. Protocol gens 512, so 32k is plenty.
# Smaller max_model_len shrinks the allocated KV page table; overlay
# sparse indexer scores over real_page_table width. Same image, only
# this env. Do not change util.
#
#   configs/examples/maxlen-32k-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh maxlen32k-rc2 \
  "MAX_MODEL_LEN=32768"
