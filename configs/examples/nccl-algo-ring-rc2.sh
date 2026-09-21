#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_ALGO=Ring.
#
# Standing TP2 all-reduce is PYNCCL. Anemll nsys landed on
# ncclDevKernel_AllReduce_Sum_bf16_RING_LL. Exclusive NCCL_PROTO=LL was
# a large cost; LL,Simple and MAX_NCHANNELS=1 were pin noise. Algorithm
# Ring vs NCCL auto (Tree for small) is a different lever from protocol.
# Same image, only this env.
#
#   configs/examples/nccl-algo-ring-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclring-rc2 \
  "SERVE_EXTRA_ENV=NCCL_ALGO=Ring"
