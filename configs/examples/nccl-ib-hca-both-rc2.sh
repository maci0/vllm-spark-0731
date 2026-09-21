#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_IB_HCA both ACTIVE RoCE.
#
# Standing NCCL_IB_HCA is empty (auto). Both sparks have two ACTIVE
# HCAs: rocep1s0f1 and roceP2p1s0f1. CROSSNIC=1 was pin noise; this
# names the devices instead of relying on NCCL auto + SPARK_IFACES.
# Same image, only this env.
#
#   configs/examples/nccl-ib-hca-both-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclhca-rc2 \
  "SERVE_EXTRA_ENV=NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1"
