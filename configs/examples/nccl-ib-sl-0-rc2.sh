#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_IB_SL=0.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. IB TC 106 was pin noise.
# Service level 0 is a leftover that still fires on ConnectX-7 QoS.
# Same image, only this env.
#
#   configs/examples/nccl-ib-sl-0-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclsl0-rc2 \
  "SERVE_EXTRA_ENV=NCCL_IB_SL=0"
