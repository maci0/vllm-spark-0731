#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_NTHREADS=64.
#
# Standing TP2 all-reduce is PYNCCL. Eager profile priced 87 all-reduce
# calls at 81.3 ms gpu_sum on live c1. NCCL_PROTO exclusive LL was a
# large cost; LL,Simple / MAX_NCHANNELS=1 / ALGO=Ring were pin noise or
# worse at c6. Thread count is a different lever from protocol/algo.
# Same image, only this env.
#
#   configs/examples/nccl-nthreads-64-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclnt64-rc2 \
  "SERVE_EXTRA_ENV=NCCL_NTHREADS=64"
