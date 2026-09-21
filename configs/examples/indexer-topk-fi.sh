#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — indexer top-k via FlashInfer.
#
# Auto chain is cooperative (excluded on SM120) -> persistent (topk 512,
# our k) -> per_row. deep_select/flashinfer/torch are opt-in only.
# Overlay scores logits in fp32, which flashinfer.topk.top_k_ragged_transform
# requires. Same image, same pin, only this serve flag. ARM_EXTRA_ARGS
# because run-arm EXTRA is env-only.
#
#   configs/examples/indexer-topk-fi.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--sparse-indexer-topk-backend flashinfer"
exec bash harness/run-arm.sh idxfi ""
