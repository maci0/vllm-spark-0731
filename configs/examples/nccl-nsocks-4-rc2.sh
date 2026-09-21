#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_NSOCKS_PERTHREAD=4.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. NCCL_SOCKET_NTHREADS=4
# was a c6 cost. NSOCKS_PERTHREAD is the complementary leftover: more
# sockets per helper thread, not more helper threads. Same image, only
# this env.
#
#   configs/examples/nccl-nsocks-4-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclnsock4-rc2 \
  "SERVE_EXTRA_ENV=NCCL_NSOCKS_PERTHREAD=4"
