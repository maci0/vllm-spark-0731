#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_CUMEM_ENABLE=0.
#
# Standing TP2 all-reduce is PYNCCL. Proto/algo/thread/channel knobs
# were pin noise or a cost. GB10 is UMA; NCCL default cuMemMap
# allocations can fragment host-device unified memory. Same image,
# only this env.
#
#   configs/examples/nccl-cumem-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclcu0-rc2 \
  "SERVE_EXTRA_ENV=NCCL_CUMEM_ENABLE=0"
