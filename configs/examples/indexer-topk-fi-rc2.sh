#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — indexer top-k via FlashInfer.
#
# Same as configs/examples/indexer-topk-fi.sh, new tag so the log does
# not collide with rc1 `idxfi`. Auto chain is cooperative (excluded on
# SM120) -> persistent (topk 512) -> per_row; flashinfer is opt-in.
# Overlay scores logits in fp32. Same image, only this serve flag.
#
#   configs/examples/indexer-topk-fi-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--sparse-indexer-topk-backend flashinfer"
exec bash harness/run-arm.sh idxfi-rc2 ""
