#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_PXN_DISABLE=1.
#
# Standing TP2 all-reduce is PYNCCL over two-host RoCE. PXN (proxy
# NIC) is a NVSwitch/NVLink hop; on two DGX Spark boxes it can add
# a probe. Proto/algo/thread/channel/cuMem/buffer/GROUP/P2P/QPS/
# CROSSNIC/HCA knobs were pin noise or a cost. Same image, only this
# env.
#
#   configs/examples/nccl-pxn-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclpxn0-rc2 \
  "SERVE_EXTRA_ENV=NCCL_PXN_DISABLE=1"
