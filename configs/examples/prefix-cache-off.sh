#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — prefix cache off.
#
# Pin sets ENABLE_PREFIX_CACHING=1; vLLM default is already True.
# Protocol reuses one prompt across levels, so prefix cache can hide
# prefill. Never isolated off on main-030-rc1. BooleanOptionalAction
# last-flag: ARM_EXTRA_ARGS="--no-enable-prefix-caching" wins over
# 05-serve.sh's --enable-prefix-caching. Same image, only this flag.
#
#   configs/examples/prefix-cache-off.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--no-enable-prefix-caching"
exec bash harness/run-arm.sh nopfx ""
