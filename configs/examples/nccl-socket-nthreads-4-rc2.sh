#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_SOCKET_NTHREADS=4.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. Proto/algo/thread/channel/
# cuMem/buffer/GROUP/P2P/QPS/CROSSNIC/HCA/PXN/CHECKS/SHM knobs were pin
# noise or a cost. Socket helper threads are a different leftover from
# CUDA NCCL_NTHREADS. Same image, only this env.
#
#   configs/examples/nccl-socket-nthreads-4-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclsock4-rc2 \
  "SERVE_EXTRA_ENV=NCCL_SOCKET_NTHREADS=4"
