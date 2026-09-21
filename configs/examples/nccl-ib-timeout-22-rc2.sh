#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_IB_TIMEOUT=22.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. Default IB timeout is 18
# (~1s). 22 is ~16s, a leftover from socket/thread knobs that were pin
# noise or a cost. Same image, only this env.
#
#   configs/examples/nccl-ib-timeout-22-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclto22-rc2 \
  "SERVE_EXTRA_ENV=NCCL_IB_TIMEOUT=22"
