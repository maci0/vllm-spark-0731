#!/usr/bin/env bash
# Diagnostic: capture-mode decode profiler on v0.30.0rc2.
#
# VLLM_PROFILE_CAPTURE=1 records region marks into the CUDA graph so a
# 1-row replay has gap-free device time. Distorts tok/s; not a candidate.
# SERVE_EXTRA_ENV is the only path that reaches docker -e. Spaces in that
# value cannot go through run-arm EXTRA (word-split), so this launcher
# quotes it the way harness/run-prof6c.sh does.
#
#   configs/examples/profcap-rc2.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
TAG="${TAG:-profcap-rc2}"
PENV="VLLM_PROFILE_DECODE=1 VLLM_PROFILE_CAPTURE=1 VLLM_PROFILE_DECODE_STEPS=12"

docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1; find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null; cd ~/vllm-spark-0731 && env NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV='$PENV' nohup bash scripts/05-serve.sh main-029 > ~/serve-${TAG}-w.log 2>&1 </dev/null &"
find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null || true
sleep 85
env NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 \
  SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV="$PENV" \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-${TAG}-h.log" 2>&1
c=000
for i in $(seq 1 45); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
  docker ps --format '{{.Names}}' | grep -q '^vllm-ds4-0731$' || { echo "== container gone"; break; }
  sleep 10
done
echo "== ${TAG} health=$c after $((i*10))s"
[ "$c" = "200" ] || { echo "== ${TAG} NEVER HEALTHY"; tail -25 "$HOME/serve-${TAG}-h.log"; exit 1; }
echo "== profiler env seen by the container =="
docker exec vllm-ds4-0731 bash -c 'tr "\0" "\n" < /proc/1/environ | grep VLLM_PROFILE'
docker logs vllm-ds4-0731 2>&1 | grep -iE "attention|MoE backend|Experts|KV cache format|indexer|deep_gemm|E8M0|linear|all-reduce|Capturing CUDA|Selected .*Kernel" | head -40 > "$HOME/goal/${TAG}-engine.log"
~/drive-median.sh "$TAG" 1 512 1 6
echo "== region table =="
docker logs vllm-ds4-0731 2>&1 | grep -E "b12x (region|layers|capture profile|step )" | tail -80
