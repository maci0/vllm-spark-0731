#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — disable custom all-reduce.
#
# Anemll recipe / startup dump: disable_custom_all_reduce=True.
# Ours default False. proto2-dg-nocar was a negative on an older pin.
# Eager profile priced 87 all-reduce calls at 81.3 ms gpu_sum (0.93 ms
# avg) on live c1. Same image, only this flag.
#
#   configs/examples/no-custom-ar-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--disable-custom-all-reduce"
exec bash harness/run-arm.sh nocar-rc2 ""
