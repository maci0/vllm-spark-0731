#!/usr/bin/env bash
# Recipe: serve the max KV context that fits on the rig — MAX_MODEL_LEN=245760
# (KV cache 11.32 GiB). 262144 dies at the memory floor; 65536..245760 all
# serve with sums 446.5-463.3, a band inside the +-9 % session drift, so KV
# depth is performance-neutral up to the frontier. 3.75x the pin context at no
# measured throughput cost.
#
#   configs/examples/kv-max-245760-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh kv245760-rc2 \
  "MAX_MODEL_LEN=245760"
