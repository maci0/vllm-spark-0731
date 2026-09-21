#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_PROTO=LL.
#
# Standing TP2 all-reduce is PYNCCL (FlashInfer MNNVL needs NVSwitch;
# FLASHINFER_PCIE_IPC is same-node CUDA-IPC). Eager profile priced 87
# all-reduce calls at 81.3 ms gpu_sum on live c1. NCCL_PROTO default is
# unset (NCCL picks). LL is the low-latency protocol for small messages.
# Same image, only this env.
#
#   configs/examples/nccl-proto-ll-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nccpll-rc2 \
  "SERVE_EXTRA_ENV=NCCL_PROTO=LL"
