#!/usr/bin/env bash
# Recipe: 1M-context capacity probe, minimum-footprint serve.
#
# The 1M KV pool itself fits on paper (1,396,688 tokens against 17.55 GiB at
# util 0.90); the first attempt died AFTER pool alloc, in FlashInfer autotune
# + shm broadcast with host RAM pinned. So strip everything that is not the
# pool: no speculative decode, no CUDA graphs, one sequence, minimal batch.
# This is a capacity probe, not a benchmark — throughput numbers from this
# config mean nothing.
#
#   configs/examples/kv-1m-probe.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--no-enable-flashinfer-autotune"
exec bash harness/run-arm.sh kv2m-probe \
  "MAX_MODEL_LEN=2097152 GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=2048 DISABLE_DSPARK=1 ENFORCE_EAGER=1"
