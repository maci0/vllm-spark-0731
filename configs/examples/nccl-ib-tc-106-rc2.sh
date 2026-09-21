#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_IB_TC=106.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. Proto/algo/thread/channel/
# cuMem/buffer/GROUP/P2P/QPS/CROSSNIC/HCA/PXN/CHECKS/SHM/SOCKET/TIMEOUT
# knobs were pin noise or a cost. IB traffic class 106 (DSCP 26) is a
# leftover that still fires on ConnectX-7. Same image, only this env.
#
#   configs/examples/nccl-ib-tc-106-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nccltc106-rc2 \
  "SERVE_EXTRA_ENV=NCCL_IB_TC=106"
