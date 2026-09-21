#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_NET_PLUGIN=none.
#
# Standing TP2 all-reduce is PYNCCL over RoCE. CollNet off was pin
# noise. Forcing no external NET plugin is a leftover that still
# fires. Same image, only this env.
#
#   configs/examples/nccl-net-plugin-none-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh ncclplug0-rc2 \
  "SERVE_EXTRA_ENV=NCCL_NET_PLUGIN=none"
