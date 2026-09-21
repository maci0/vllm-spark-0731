#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_NET=IB.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. NET_PLUGIN=none was pin
# noise. Forcing the IB NET instead of auto is a leftover that still
# fires. Same image, only this env.
#
#   configs/examples/nccl-net-ib-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclnetib-rc2 \
  "SERVE_EXTRA_ENV=NCCL_NET=IB"
