#!/usr/bin/env bash
# Interleaved ours-vs-reference at matched KV spec: ours (262144, util 0.86),
# ref, ref, ours. Two interleaved readings per arm, 5 passes each: dense
# enough to hold both keep-rule readings (± pair-scale), spread enough that
# a single-session monotonic drift cancels. Every arm carries the UPSTREAM
# provenance guard (container name + engine version + both-node MoE log).
# Estimated wall: ~80 min; run in background and collect.
set -uo pipefail
cd "$HOME/vllm-spark-0731"
docker rm -f $(docker ps -aq) >/dev/null 2>&1
ssh spark2 "docker rm -f \$(docker ps -aq) >/dev/null 2>&1"
MAX_MODEL_LEN=262144 GPU_MEMORY_UTILIZATION=0.86 PASSES=5 bash harness/run-arm.sh rO-a
bash "$HOME/goal/launch-refbase.sh"
c=000; for i in $(seq 1 45); do c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true); [ "$c" = "200" ] && break; sleep 10; done
[ "$c" = "200" ] || { echo "== REFERENCE NEVER HEALTHY (rR-b)"; exit 1; }
~/drive-median.sh rR-b 5 512 1 3 5 6
bash "$HOME/goal/launch-refbase.sh"
c=000; for i in $(seq 1 45); do c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true); [ "$c" = "200" ] && break; sleep 10; done
[ "$c" = "200" ] || { echo "== REFERENCE NEVER HEALTHY (rR-c)"; exit 1; }
~/drive-median.sh rR-c 5 512 1 3 5 6
MAX_MODEL_LEN=262144 GPU_MEMORY_UTILIZATION=0.86 PASSES=5 bash harness/run-arm.sh rO-d
echo "== PAIR121 DONE"
