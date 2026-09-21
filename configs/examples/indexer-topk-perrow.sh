#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — indexer top-k via per_row.
#
# Auto chain is cooperative (excluded on SM120) -> persistent (topk 512,
# our k) -> per_row. flashinfer was idxfi (pin noise). deep_select needs
# SM100. per_row is leftover opt-in. Overlay scores logits in fp32.
# Same image, only this serve flag. ARM_EXTRA_ARGS because EXTRA is env-only.
#
#   configs/examples/indexer-topk-perrow.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--sparse-indexer-topk-backend per_row"
exec bash harness/run-arm.sh idxpr ""
