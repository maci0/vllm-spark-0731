#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_LAUNCH_MODE=GROUP.
#
# Standing TP2 all-reduce is PYNCCL. Eager profile priced 87 all-reduce
# calls at 81.3 ms gpu_sum on live c1. Proto/algo/thread/channel/cuMem/
# buffer knobs were pin noise or a cost. GROUP batches CUDA launches
# instead of PARALLEL. Same image, only this env.
#
#   configs/examples/nccl-launch-group-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclgrp-rc2 \
  "SERVE_EXTRA_ENV=NCCL_LAUNCH_MODE=GROUP"
