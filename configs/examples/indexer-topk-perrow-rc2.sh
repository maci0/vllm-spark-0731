#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — indexer top-k via per_row.
#
# Same as configs/examples/indexer-topk-perrow.sh, new tag so the log
# does not collide with rc1 `idxpr`. Auto chain is cooperative (excluded
# on SM120) -> persistent (topk 512) -> per_row. Overlay scores logits
# in fp32. Same image, only this serve flag.
#
#   configs/examples/indexer-topk-perrow-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--sparse-indexer-topk-backend per_row"
exec bash harness/run-arm.sh idxpr-rc2 ""
