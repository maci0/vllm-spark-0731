#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_COLLNET_ENABLE=0.
#
# Standing TP2 all-reduce is PYNCCL over two-host RoCE. CollNet is an
# in-network collective plugin; on 2-node Spark it can still probe.
# Channel/AR/timeout/TC/SL knobs were pin noise or a cost. Same image,
# only this env.
#
#   configs/examples/nccl-collnet-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclcnet0-rc2 \
  "SERVE_EXTRA_ENV=NCCL_COLLNET_ENABLE=0"
