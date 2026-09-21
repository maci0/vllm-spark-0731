#!/usr/bin/env bash
# True c6 target-forward device time, with CUDA graphs ON and the protocol
# config (k=7, capture 48).
#
# The earlier phase accounting compared a profiler-window target time against a
# production step time and called the difference host-bound. That run also used
# the pin defaults (k=5, capture 36) so the 48-token target forward was not even
# captured. This one matches `harness/run-arm.sh` exactly.
#
# EngineCore drops unregistered VLLM_* before the worker, so the profiler is
# armed through the B12X_PROFILE_* alias and a bind-mounted patched overlay,
# the same route `profeager-rc2` used.
#
#   configs/examples/stepc6-rc2.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
TAG="${TAG:-stepc6-rc2}"
STEPS="${STEPS:-40}"
HOST_PY="${ROOT}/outputs/driver/one-off/sm12x_b12x_kernels_prof.py"
CTR_PY="/opt/vllm/vllm/utils/sm12x_b12x_kernels.py"
PENV="B12X_PROFILE_DECODE=1 B12X_PROFILE_DECODE_STEPS=${STEPS}"
MOUNTS="${HOST_PY}:${CTR_PY}"

docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1; find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null; cd ~/vllm-spark-0731 && env NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV='$PENV' SERVE_EXTRA_MOUNTS='$MOUNTS' nohup bash scripts/05-serve.sh main-029 > ~/serve-${TAG}-w.log 2>&1 </dev/null &"
find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null || true
sleep 85
env NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 \
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
echo "== worker profile env =="
docker exec vllm-ds4-0731 bash -c 'for p in /proc/[0-9]*; do cmd=$(tr "\0" " " < $p/cmdline 2>/dev/null); case "$cmd" in *Worker*) echo PID ${p#/proc/}; tr "\0" "\n" < $p/environ | grep B12X_PROFILE || echo no_b12x_profile;; esac; done'
echo "== k/capture =="
docker logs vllm-ds4-0731 2>&1 | grep -oE "num_spec_tokens=[0-9]+|cudagraph_capture_sizes.: \[[0-9, ]+\]|max_cudagraph_capture_size.: [0-9]+" | head -4

# c6 only: the window must land in c6 decode, and a warm pass would spend the
# step budget before c6 starts.
python3 scripts/bench-concurrency.py --chat --max-tokens 512 --seed 1234 \
  --levels 6 --no-warm >/dev/null 2>&1
echo "== c6 load done"
echo "== step table =="
docker logs vllm-ds4-0731 2>&1 | grep -E "b12x step|b12x profile window|b12x region|b12x layers" | tail -60
