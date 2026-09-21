#!/usr/bin/env bash
# Profile the decode step at the c6 level.
#
# The step limit must go through SERVE_EXTRA_ENV: that is the only path that
# reaches the container as `docker run -e`. Assigning VLLM_PROFILE_DECODE_STEPS
# in the launching shell does not reach the worker, which is why earlier runs
# silently profiled the default 12 steps (all c1).
set -uo pipefail
cd "$HOME/vllm-spark-0731"
PENV="VLLM_PROFILE_DECODE=1 VLLM_PROFILE_DECODE_STEPS=1250"
docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1; cd ~/vllm-spark-0731 && GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV='$PENV' nohup bash scripts/05-serve.sh main-029 > ~/serve-p6c-w.log 2>&1 </dev/null &"
sleep 85
GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV="$PENV" \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-p6c-h.log" 2>&1
c=000
for i in $(seq 1 45); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
  docker ps --format '{{.Names}}' | grep -q '^vllm-ds4-0731$' || { echo "container gone"; break; }
  sleep 10
done
echo "== p6c health=$c after $((i*10))s"
[ "$c" = "200" ] || { tail -20 "$HOME/serve-p6c-h.log"; exit 1; }
echo "== profiler env seen by the container =="
docker exec vllm-ds4-0731 bash -c 'tr "\0" "\n" < /proc/1/environ | grep VLLM_PROFILE'
~/drive-median.sh p6c 3 512 1 3 5 6
