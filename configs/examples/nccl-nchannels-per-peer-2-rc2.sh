#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_NCHANNELS_PER_NET_PEER=2.
#
# Standing TP2 all-reduce is PYNCCL over two-host RoCE. MAX=1 and MIN=2
# were pin noise / a c6 cost. Per-peer channels is a leftover that
# still fires on 2-node. Same image, only this env.
#
#   configs/examples/nccl-nchannels-per-peer-2-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclpeer2-rc2 \
  "SERVE_EXTRA_ENV=NCCL_NCHANNELS_PER_NET_PEER=2"
