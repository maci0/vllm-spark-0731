#!/usr/bin/env bash
# Diagnostic: eager decode profiler on v0.30.0rc2.
#
# EngineCore drops unregistered VLLM_* so VLLM_PROFILE_DECODE never
# reaches the worker. Alias B12X_PROFILE_* (workers keep B12X_). Bind-mount
# patched sm12x_b12x_kernels.py that reads the alias. Eager so region marks
# fire live. Distorts tok/s; not a candidate.
#
#   configs/examples/profeager-rc2.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
TAG="${TAG:-profeager-rc2}"
HOST_PY="${ROOT}/outputs/driver/one-off/sm12x_b12x_kernels_prof.py"
CTR_PY="/opt/vllm/vllm/utils/sm12x_b12x_kernels.py"
PENV="B12X_PROFILE_DECODE=1 B12X_PROFILE_DECODE_STEPS=12"
MOUNTS="${HOST_PY}:${CTR_PY}"

docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1; find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null; cd ~/vllm-spark-0731 && env NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 ENFORCE_EAGER=1 SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV='$PENV' SERVE_EXTRA_MOUNTS='$MOUNTS' nohup bash scripts/05-serve.sh main-029 > ~/serve-${TAG}-w.log 2>&1 </dev/null &"
find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null || true
sleep 85
env NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 ENFORCE_EAGER=1 \
  SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV="$PENV" SERVE_EXTRA_MOUNTS="$MOUNTS" \
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
echo "== profiler env seen by the worker =="
docker exec vllm-ds4-0731 bash -c 'for p in /proc/[0-9]*; do cmd=$(tr "\0" " " < $p/cmdline 2>/dev/null); case "$cmd" in *Worker*) echo PID ${p#/proc/}; tr "\0" "\n" < $p/environ | grep B12X_PROFILE || echo no_b12x_profile;; esac; done'
~/drive-median.sh "$TAG" 1 64 1
echo "== region table =="
docker logs vllm-ds4-0731 2>&1 | grep -E "b12x (region|layers|capture profile|step )" | tail -120
