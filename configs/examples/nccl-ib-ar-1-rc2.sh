#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_IB_AR_ALGORITHM=1.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. NCCL_ALGO=Ring was a c6
# cost. IB adaptive routing is a leftover that still fires on
# ConnectX-7. Same image, only this env.
#
#   configs/examples/nccl-ib-ar-1-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclar1-rc2 \
  "SERVE_EXTRA_ENV=NCCL_IB_AR_ALGORITHM=1"
