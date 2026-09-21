#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_IB_QPS_PER_CONNECTION=4.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. Proto/algo/thread/channel/
# cuMem/buffer/GROUP/P2P knobs were pin noise or a cost. Default IB QPS
# per connection is 1. Four QPs is a different lever for 87 small
# all-reduce calls. Same image, only this env.
#
#   configs/examples/nccl-ib-qps-4-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclqps4-rc2 \
  "SERVE_EXTRA_ENV=NCCL_IB_QPS_PER_CONNECTION=4"
