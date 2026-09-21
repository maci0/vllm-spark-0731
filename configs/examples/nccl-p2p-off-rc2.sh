#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_P2P_DISABLE=1.
#
# Standing TP2 all-reduce is PYNCCL across two hosts. CUDA P2P / IPC
# does not work across nodes (`proto2-pcie-ar` fell back). NCCL may
# still probe P2P first. Proto/algo/thread/channel/cuMem/buffer/GROUP
# knobs were pin noise or a cost. Same image, only this env.
#
#   configs/examples/nccl-p2p-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclp2p0-rc2 \
  "SERVE_EXTRA_ENV=NCCL_P2P_DISABLE=1"
