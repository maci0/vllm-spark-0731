#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — long-prefill threshold 1024.
#
# Same as configs/examples/long-prefill-1024.sh, new tag so the log does
# not collide with rc1 `lpf1024`. pin.main-029.env sets
# LONG_PREFILL_TOKEN_THRESHOLD=1024 and the anemll recipe passes the
# flag. 05-serve.sh never forwards it. vLLM default is 0. ARM_EXTRA_ARGS
# because EXTRA is env-only.
#
#   configs/examples/long-prefill-1024-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--long-prefill-token-threshold 1024"
exec bash harness/run-arm.sh lpf1024-rc2 ""
