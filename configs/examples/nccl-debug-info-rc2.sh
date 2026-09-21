#!/usr/bin/env bash
# Recipe: NCCL transport diagnostic on v0.30.0rc2 — NCCL_DEBUG=INFO.
#
# c6 step is ~203 ms wall but the profiled target forward (~100 ms) plus
# draft+sampler (~21.5 ms) only account for ~122 ms. The region profile puts
# the TP all-reduce at 81.3 ms of device time over 87 calls (0.93 ms avg) for
# an 8 KiB payload, which is ~30x a healthy RoCE all-reduce. Twenty NCCL env
# arms were measured blind; the chosen transport was never inspected.
#
# This arm only turns on the NCCL init/net log so the transport, HCA, GID and
# algorithm are on the record. Diagnostic, not a candidate.
#
#   configs/examples/nccl-debug-info-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclinfo-rc2 \
  "SERVE_EXTRA_ENV=NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING"
