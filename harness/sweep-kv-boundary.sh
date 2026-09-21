#!/usr/bin/env bash
# Pin the max-KV frontier: probe the two sizes just under the reference's
# 262144 (which died). Capture the worker KV-memory report for each that serves.
set -uo pipefail
cd "$HOME/vllm-spark-0731"
docker rm -f $(docker ps -aq) >/dev/null 2>&1
ssh spark2 "docker rm -f \$(docker ps -aq) >/dev/null 2>&1"
for ml in 229376 245760; do
  echo "== MAX_MODEL_LEN=$ml at $(date -Is)"
  if MAX_MODEL_LEN=$ml PASSES=3 bash harness/run-arm.sh "kv${ml}"; then
    echo "== kv${ml} SERVED"
    docker exec vllm-ds4-0731 python3 -c "import torch; print('KV-report:')" 2>/dev/null
    docker logs vllm-ds4-0731 2>&1 | grep -oE "Current kv cache memory in use is [0-9.]+ GiB|Replace gpu_memory_utilization config with [^;]*" | tail -3
  else
    echo "== kv${ml} FAILED"
  fi
done
echo "== BOUNDARY DONE"
