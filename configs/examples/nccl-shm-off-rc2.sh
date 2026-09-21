#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_SHM_DISABLE=1.
#
# Standing TP2 all-reduce is PYNCCL across two hosts. NCCL SHM is
# intra-node CUDA IPC. Probe can still run on 2-node. Proto/algo/
# thread/channel/cuMem/buffer/GROUP/P2P/QPS/CROSSNIC/HCA/PXN/CHECKS
# knobs were pin noise or a cost. Same image, only this env.
#
#   configs/examples/nccl-shm-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclshm0-rc2 \
  "SERVE_EXTRA_ENV=NCCL_SHM_DISABLE=1"
