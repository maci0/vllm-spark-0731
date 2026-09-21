#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_CHECKS_DISABLE=1.
#
# Standing TP2 all-reduce is PYNCCL. Eager profile priced 87 all-reduce
# calls at 81.3 ms gpu_sum on live c1. Proto/algo/thread/channel/cuMem/
# buffer/GROUP/P2P/QPS/CROSSNIC/HCA/PXN knobs were pin noise or a cost.
# NCCL default still runs argument checks on every collective. Same
# image, only this env.
#
#   configs/examples/nccl-checks-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclchk0-rc2 \
  "SERVE_EXTRA_ENV=NCCL_CHECKS_DISABLE=1"
