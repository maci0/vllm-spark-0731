#!/usr/bin/env bash
# True c6 target-forward device time on the standing pin.
#
# The report's phase accounting compared a profiler-window target time
# (~100 ms) against a production step time (203 ms) and called the difference
# "host-bound". That comparison mixed two regimes. This run measures
# `execute_model` and `sample_tokens` CUDA-event time for real c6 steps with
# CUDA graphs on (capture mode off, so the limit actually applies), which is
# the number the step time has to be reconciled against.
#
#   profile-c6-step.sh <tag> [steps]
set -uo pipefail
cd "$HOME/vllm-spark-0731"

TAG="${1:-stepc6}"
STEPS="${2:-40}"
OUT="$HOME/vllm-spark-0731/outputs/driver/one-off"
mkdir -p "$OUT"
LOG="$OUT/$TAG-step.txt"
PENV="VLLM_PROFILE_DECODE=1 VLLM_PROFILE_DECODE_STEPS=$STEPS"

docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1; cd ~/vllm-spark-0731 && GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV='$PENV' nohup bash scripts/05-serve.sh main-029 > ~/serve-$TAG-w.log 2>&1 </dev/null &"
sleep 85
GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' SERVE_EXTRA_ENV="$PENV" \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-$TAG-h.log" 2>&1

c=000
for i in $(seq 1 45); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
  docker ps --format '{{.Names}}' | grep -q '^vllm-ds4-0731$' || { echo "container gone"; exit 1; }
  sleep 10
done
echo "== health=$c after $((i*10))s"
[ "$c" = "200" ] || { tail -30 "$HOME/serve-$TAG-h.log"; exit 1; }
echo "== worker env: $(docker exec vllm-ds4-0731 bash -c 'tr "\0" "\n" < /proc/316/environ 2>/dev/null | grep VLLM_PROFILE || true')"

# c6 only, and no warm pass: the profiler window must land in c6 decode, and a
# warm pass would consume the step budget before c6 starts.
python3 scripts/bench-concurrency.py --chat --max-tokens 512 --seed 1234 \
  --levels 6 --no-warm >/dev/null 2>&1
echo "== c6 load done"

{
  echo "== c6 step profile $TAG $(date -Is)"
  docker logs vllm-ds4-0731 2>&1 | grep -E "b12x step|b12x profile window|b12x region|b12x layers" | tail -60
} 2>&1 | tee "$LOG"
echo "== wrote $LOG"
