#!/usr/bin/env bash
# Dump the captured CUDA graph's kernel nodes on the shipped (uncompiled) config.
# The 1-byte copy family is invisible to launch-API hooks because replay bypasses
# cudaLaunchKernel, so the graph itself is read via cudaGraphInstantiateWithFlags.
set -uo pipefail
cd "$HOME/vllm-spark-0731"
rm -f "$HOME/.cache/huggingface/inject/dcopy.txt"
docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 'docker rm -f vllm-ds4-0731 >/dev/null 2>&1; find /dev/shm -maxdepth 1 \( -name "psm_*" -o -name "nccl-*" -o -name "sem.mp-*" -o -name "mp-*" \) -delete 2>/dev/null; cd ~/vllm-spark-0731 && GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS="--async-scheduling" SERVE_EXTRA_MOUNTS="$HOME/.cache/huggingface/inject/ld.so.preload:/etc/ld.so.preload:ro" SERVE_EXTRA_ENV="DCOPY_TRACE_OUT=/root/.cache/huggingface/inject/dcopy.txt DCOPY_TRACE_BLOCK_X=0 DCOPY_TRACE_LIMIT=20" nohup bash scripts/05-serve.sh main-029 > ~/serve-gd-w.log 2>&1 &'
find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null || true
sleep 85
GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS="--async-scheduling" \
  SERVE_EXTRA_MOUNTS="$HOME/.cache/huggingface/inject/ld.so.preload:/etc/ld.so.preload:ro" \
  SERVE_EXTRA_ENV="DCOPY_TRACE_OUT=/root/.cache/huggingface/inject/dcopy.txt DCOPY_TRACE_BLOCK_X=0 DCOPY_TRACE_LIMIT=20" \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-gd-h.log" 2>&1
c=000
for i in $(seq 1 70); do
  c=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && { echo "HEALTHY $((i*10))s"; break; }
  docker ps --format '{{.Names}}' | grep -q '^vllm-ds4-0731$' || { echo "container gone"; break; }
  sleep 10
done
echo "health=$c"
curl -s -m 150 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","prompt":"The capital of France is","max_tokens":64,"temperature":0}' -o /dev/null
sleep 3
echo "=== shim loads ==="
docker logs vllm-ds4-0731 2>&1 | grep -oE 'loaded pid=[0-9]+' | sort -u | head -4
echo "=== graph node dump ==="
grep -c 'graph #' "$HOME/.cache/huggingface/inject/dcopy.txt" 2>/dev/null
grep -E 'graph #|node [0-9]+ func=' "$HOME/.cache/huggingface/inject/dcopy.txt" 2>/dev/null | head -30
echo "=== biggest grids among nodes ==="
grep -oE 'grid=\([0-9]+,[0-9]+,[0-9]+\) block=\([0-9]+,[0-9]+,[0-9]+\)' "$HOME/.cache/huggingface/inject/dcopy.txt" 2>/dev/null | sort | uniq -c | sort -rn | head -12
