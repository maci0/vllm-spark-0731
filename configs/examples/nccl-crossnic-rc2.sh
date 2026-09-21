#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_CROSSNIC=1.
#
# Standing SPARK_IFACES is enp1s0f1np1 only. Both sparks also have
# enP2p1s0f1np1 UP. CROSSNIC lets NCCL stripe small all-reduces across
# both RoCE ports. Proto/algo/thread/channel/cuMem/buffer/GROUP/P2P/QPS
# knobs were pin noise or a cost. Same image, only this env.
#
#   configs/examples/nccl-crossnic-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclxnic-rc2 \
  "SERVE_EXTRA_ENV=NCCL_CROSSNIC=1"
