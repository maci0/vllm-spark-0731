#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_IB_RETRY_CNT=1.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. Default retry count is 7.
# Timeout 22 / TC / SL were pin noise. Fewer retries is a leftover that
# still fires on ConnectX-7. Same image, only this env.
#
#   configs/examples/nccl-ib-retry-1-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclretry1-rc2 \
  "SERVE_EXTRA_ENV=NCCL_IB_RETRY_CNT=1"
