#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — generation_config=vllm.
#
# Reference recipe sets --generation-config vllm. Ours default auto, which
# loads the checkpoint generation_config.json (do_sample true, temp 1.0,
# top_p 1.0). Protocol requests send temperature explicitly, so this may
# be a no-op. Same image, only this serve flag. ARM_EXTRA_ARGS because
# EXTRA is env-only.
#
#   configs/examples/gen-config-vllm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--generation-config vllm"
exec bash harness/run-arm.sh gencvllm ""
