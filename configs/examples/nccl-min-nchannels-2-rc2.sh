#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_MIN_NCHANNELS=2.
#
# Standing TP2 all-reduce is PYNCCL. NCCL_MAX_NCHANNELS=1 was pin noise.
# MIN is the complementary leftover: raise the floor instead of capping
# the ceiling. Same image, only this env.
#
#   configs/examples/nccl-min-nchannels-2-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclmin2-rc2 \
  "SERVE_EXTRA_ENV=NCCL_MIN_NCHANNELS=2"
