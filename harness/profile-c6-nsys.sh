#!/usr/bin/env bash
# nsys kernel profile of a c6 decode window on the standing pin.
#
# The region marks in the overlay only collect under CUDA-graph capture, and
# capture mode dies (`profcap-rc2`: CUDA error invalid argument). nsys attaches
# to the worker instead and traces graph nodes as real kernels, so this is the
# ground-truth kernel table for a graph-replayed c6 step.
#
#   profile-c6-nsys.sh <tag> [seconds]
set -uo pipefail
cd "$HOME/vllm-spark-0731"

TAG="${1:-nsysc6}"
SECS="${2:-12}"
OUT="$HOME/vllm-spark-0731/outputs/driver/one-off"
mkdir -p "$OUT"
REP="/tmp/$TAG-profile"
LOG="$OUT/$TAG-nsys.txt"

docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1; cd ~/vllm-spark-0731 && GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' nohup bash scripts/05-serve.sh main-029 > ~/serve-$TAG-w.log 2>&1 </dev/null &"
sleep 85
GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-$TAG-h.log" 2>&1

c=000
for i in $(seq 1 45); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
  docker ps --format '{{.Names}}' | grep -q '^vllm-ds4-0731$' || { echo "container gone"; exit 1; }
  sleep 10
done
echo "== health=$c after $((i*10))s"
[ "$c" = "200" ] || { tail -20 "$HOME/serve-$TAG-h.log"; exit 1; }

# Warm at the profiled level so the window is steady-state decode.
python3 scripts/bench-concurrency.py --chat --max-tokens 512 --seed 1234 \
  --levels 6 --no-warm >/dev/null 2>&1
echo "== warm c6 done"

W=$(docker exec vllm-ds4-0731 bash -c 'pgrep -f "Worker_TP0|vllm.*Worker" | head -1')
[ -n "$W" ] || W=$(docker exec vllm-ds4-0731 bash -c 'pgrep -f "EngineCore" | head -1')
echo "== worker pid=$W"

docker exec vllm-ds4-0731 bash -c "rm -f ${REP}*"
docker exec vllm-ds4-0731 bash -c \
  "nsys profile --attach-pid $W --output $REP --force-overwrite true \
     --duration $SECS --cuda-graph-trace=node --trace=cuda,nvtx \
     --sample=none --cpuctxsw=none --stats=false" >/dev/null 2>&1 &
NSYS=$!
sleep 3
python3 scripts/bench-concurrency.py --chat --max-tokens 512 --seed 1234 \
  --levels 6 --no-warm >/dev/null 2>&1
wait $NSYS 2>/dev/null
echo "== nsys stopped"

docker exec vllm-ds4-0731 bash -c "ls -l ${REP}* 2>/dev/null"
{
  echo "== nsys c6 kernel summary $TAG $(date -Is)"
  docker exec vllm-ds4-0731 bash -c \
    "nsys stats --report cuda_gpu_kern_sum --format table ${REP}.nsys-rep 2>/dev/null" | head -80
  echo
  echo "== nsys c6 nvtx summary"
  docker exec vllm-ds4-0731 bash -c \
    "nsys stats --report nvtx_sum --format table ${REP}.nsys-rep 2>/dev/null" | head -40
  echo
  echo "== nsys cuda_gpu_sum"
  docker exec vllm-ds4-0731 bash -c \
    "nsys stats --report cuda_gpu_sum --format table ${REP}.nsys-rep 2>/dev/null" | head -30
} 2>&1 | tee "$LOG"
echo "== wrote $LOG"
