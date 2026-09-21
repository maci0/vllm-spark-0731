#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — NCCL_PROTO=LL,Simple.
#
# Standing TP2 all-reduce is PYNCCL. Exclusive NCCL_PROTO=LL was a large
# cost (nccpll-rc2 352.3). New hypothesis: enable LL as an option, keep
# Simple for large messages, let NCCL pick per size. Anemll nsys landed
# on RING_LL; exclusive LL is not that. Same image, only this env.
# Comma form so SERVE_EXTRA_ENV word-split stays one kv.
#
#   configs/examples/nccl-proto-ll-simple-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nccllls-rc2 \
  "SERVE_EXTRA_ENV=NCCL_PROTO=LL,Simple"
