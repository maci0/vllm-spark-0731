#!/usr/bin/env bash
# Probe MoE backends for support on this rig, cheapest first.
#   probe-moe.sh <backend> [<backend> ...]
# Each backend gets a worker+head launch; the probe stops at the first one that
# reaches health, leaving that container running so it can be metered.
# Failures are recorded with the line that names the reason.
set -uo pipefail
cd "$HOME/vllm-spark-0731"
for BE in "$@"; do
  echo "== probing MOE_BACKEND=$BE at $(date -Is)"
  docker rm -f vllm-ds4-0731 >/dev/null 2>&1
  ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1"
  ssh spark2 "cd ~/vllm-spark-0731 && env NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' MOE_BACKEND=$BE nohup bash scripts/05-serve.sh main-029 > ~/probe-$BE-w.log 2>&1 </dev/null &"
  sleep 85
  env NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 \
    SERVE_EXTRA_ARGS='--async-scheduling' MOE_BACKEND="$BE" \
    bash scripts/05-serve.sh main-029 > "$HOME/probe-$BE-h.log" 2>&1
  c=000
  for i in $(seq 1 20); do
    c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true)
    [ "$c" = "200" ] && break
    docker ps --format '{{.Names}}' | grep -q '^vllm-ds4-0731$' || break
    sleep 10
  done
  if [ "$c" = "200" ]; then
    echo "== $BE HEALTHY after $((i*10))s"
    echo "== stopping the probe so it can be metered"
    exit 0
  fi
  echo "== $BE NOT HEALTHY (health=$c)"
  docker logs vllm-ds4-0731 2>&1 | grep -E "does not support|not supported|Unknown MXFP4|ValueError|RuntimeError|AssertionError" | tail -3
  docker rm -f vllm-ds4-0731 >/dev/null 2>&1
  ssh spark2 'docker rm -f vllm-ds4-0731 >/dev/null 2>&1'
done
echo "== no backend in the list came up"
