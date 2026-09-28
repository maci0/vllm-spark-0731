#!/usr/bin/env bash
# Recipe: 2x1M multi-user probe — two concurrent 1M requests, sequential
# admission. Pool holds both (~11 GiB); the risk is prefill scratch, so the
# loader admits one stream at a time (see scripts/bench-2x1m.py).
#
# Serve is the stripped 1M config (no autotune, no speculation, no graphs,
# one seq slot per... no: MAX_NUM_SEQS=2 so both requests can be resident).
# This is a capacity probe, not a benchmark.
#
#   configs/examples/kv-2x1m.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--no-enable-flashinfer-autotune"
exec bash harness/run-arm.sh kv2x1m \
  "MAX_MODEL_LEN=1048576 GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_SEQS=2 MAX_NUM_BATCHED_TOKENS=4096 DISABLE_DSPARK=1 ENFORCE_EAGER=1"
