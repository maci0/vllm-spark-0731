#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_MAX_NCHANNELS=1.
#
# Standing TP2 all-reduce is PYNCCL. Eager profile priced 87 all-reduce
# calls at 81.3 ms gpu_sum on live c1. NCCL_PROTO=LL was a large cost.
# One channel is a different lever: fewer launches for small messages.
# Same image, only this env.
#
#   configs/examples/nccl-max-nchannels-1-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclch1-rc2 \
  "SERVE_EXTRA_ENV=NCCL_MAX_NCHANNELS=1"
