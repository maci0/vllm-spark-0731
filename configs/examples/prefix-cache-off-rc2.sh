#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — prefix cache off.
#
# Same as configs/examples/prefix-cache-off.sh, new tag so the log does
# not collide with rc1 `nopfx`. Pin sets ENABLE_PREFIX_CACHING=1; vLLM
# default is already True. Protocol reuses one prompt across levels.
# BooleanOptionalAction last-flag wins over 05-serve.sh.
#
#   configs/examples/prefix-cache-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--no-enable-prefix-caching"
exec bash harness/run-arm.sh nopfx-rc2 ""
