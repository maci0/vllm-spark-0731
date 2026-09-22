#!/usr/bin/env bash
# Recipe: 2M-token aggregate capacity probe — pool holds 2M concurrent tokens
# across many requests, no request over the 1M native cap.
#
# The 1M pool already holds 5.5M tokens (5.22x), so 2M aggregate fits the
# allocator with room. What needs proving is serving 2M concurrent tokens
# correctly: 4 concurrent streams x 512k-token contexts would exceed per-
# request cap, so instead run 8 streams x 262144-token-equivalent load... in
# practice: max_num_seqs raised to cover concurrent streams, each request at
# <=1M, total live KV ~2M. Meter at c6 with 8 concurrent streams.
#
# This is a capacity probe, not a benchmark.
#
#   configs/examples/kv-2m-aggregate.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export ARM_EXTRA_ARGS="--no-enable-flashinfer-autotune"
exec bash harness/run-arm.sh kv2m-agg \
  "MAX_MODEL_LEN=1048576 GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=12288 DISABLE_DSPARK=1 ENFORCE_EAGER=1"
