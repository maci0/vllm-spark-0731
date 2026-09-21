#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — load-format safetensors.
#
# Pin LOAD_FORMAT=instanttensor is sourced after EXTRA, so env override
# cannot win. Anemll uses load_format=auto. InstantTensor env knobs
# already closed. ARM_EXTRA_ARGS last-flag wins.
#
#   configs/examples/load-safetensors-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--load-format safetensors"
exec bash harness/run-arm.sh loadsaf-rc2 ""
