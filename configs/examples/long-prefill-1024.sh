#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — long-prefill threshold 1024.
#
# pin.main-029.env sets LONG_PREFILL_TOKEN_THRESHOLD=1024 and the reference
# recipe passes --long-prefill-token-threshold 1024. 05-serve.sh never
# forwards the pin. vLLM default is 0 (no split). This is the first time
# the documented value actually reaches the scheduler on this pin. Same
# image, only this serve flag. ARM_EXTRA_ARGS because EXTRA is env-only.
#
#   configs/examples/long-prefill-1024.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--long-prefill-token-threshold 1024"
exec bash harness/run-arm.sh lpf1024 ""
