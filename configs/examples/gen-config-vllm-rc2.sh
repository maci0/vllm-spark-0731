#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — generation_config=vllm.
#
# Same as configs/examples/gen-config-vllm.sh, new tag so the log does
# not collide with rc1 `gencvllm`. Reference recipe sets
# --generation-config vllm. Ours default auto loads checkpoint
# generation_config.json. Protocol requests send temperature, so this
# may be a no-op. ARM_EXTRA_ARGS because EXTRA is env-only.
#
#   configs/examples/gen-config-vllm-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--generation-config vllm"
exec bash harness/run-arm.sh gencvllm-rc2 ""
