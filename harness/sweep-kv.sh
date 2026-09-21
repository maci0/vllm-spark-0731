#!/usr/bin/env bash
# KV-capacity / throughput sweep on the clean v0.30.0 build (main-030-0).
# One variable per arm: MAX_MODEL_LEN upward from the pin's 65536 toward the
# reference's 262144. PASSES per the protocol (3). If a size dies (KV floor),
# record it and stop; the frontier is the finding.
set -uo pipefail
cd "$HOME/vllm-spark-0731"
docker rm -f $(docker ps -aq) >/dev/null 2>&1
ssh spark2 "docker rm -f \$(docker ps -aq) >/dev/null 2>&1"

ML=(65536 98304 131072 196608 262144)
for ml in "${ML[@]}"; do
  echo "== MAX_MODEL_LEN=$ml at $(date -Is)"
  if MAX_MODEL_LEN=$ml PASSES=3 bash harness/run-arm.sh "kv${ml}"; then
    echo "== kv${ml} served + median written"
  else
    echo "== kv${ml} FAILED (KV floor / never healthy)"
  fi
done
echo "== KVSWEEP DONE"
