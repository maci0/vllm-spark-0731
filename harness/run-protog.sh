#!/usr/bin/env bash
# Guarded proto-arm measurement, mirroring the reference recipe's config
# (k=7, capture 48, async scheduling) so the only free variables are the images.
set -uo pipefail
cd "$HOME/vllm-spark-0731"
docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 'docker rm -f vllm-ds4-0731 >/dev/null 2>&1; find /dev/shm -maxdepth 1 \( -name "psm_*" -o -name "nccl-*" -o -name "sem.mp-*" -o -name "mp-*" \) -delete 2>/dev/null; cd ~/vllm-spark-0731 && NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS="--async-scheduling" nohup bash scripts/05-serve.sh main-029 > ~/serve-pg-w.log 2>&1 &'
find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null || true
sleep 85
NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 \
  SERVE_EXTRA_ARGS="--async-scheduling" \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-pg-h.log" 2>&1
c=000
for i in $(seq 1 45); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
  sleep 10
done
echo "== protog health=$c after $((i*10))s"
[ "$c" = "200" ] || { echo "== PROTO NEVER HEALTHY"; exit 1; }
docker logs vllm-ds4-0731 2>&1 | grep -iE "breakable|compil|cudagraph mode|Dynamo|piecewise" | head -20 > "$HOME/protog-engine.log"
~/drive-median.sh protog 3 512 1 3 5 6
