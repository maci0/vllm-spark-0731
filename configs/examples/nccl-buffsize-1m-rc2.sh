#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_BUFFSIZE=1048576.
#
# Standing TP2 all-reduce is PYNCCL. Eager profile priced 87 all-reduce
# calls at 81.3 ms gpu_sum on live c1. Proto/algo/thread/channel/cuMem
# knobs were pin noise or a cost. SYMM_MEM needs world_size>=4, TP2
# skips. Default NCCL buffer is 4 MiB; 1 MiB is a different lever for
# small messages. Same image, only this env.
#
#   configs/examples/nccl-buffsize-1m-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclbuf1m-rc2 \
  "SERVE_EXTRA_ENV=NCCL_BUFFSIZE=1048576"
