#!/usr/bin/env bash
# Guarded arm runner.
#   run-arm.sh <tag> "<env assignments>"
# The assignments are exported into both the worker (spark2) and head (spark1)
# serve invocations, so one arm differs from another only by what is named here.
set -uo pipefail
TAG="${1:?usage: run-arm.sh <tag> \"<env assignments>\"}"
EXTRA="${2:-}"
cd "$HOME/vllm-spark-0731"

# Extra `vllm serve` flags for an arm, e.g. ARM_EXTRA_ARGS="--disable-custom-all-reduce".
# This exists because `${EXTRA}` above is word-split into environment assignments, so a
# value containing a space (which is exactly what a second CLI flag is) cannot be passed
# through it. The default is empty, so every arm that does not set this is unchanged.
ARM_EXTRA_ARGS="${ARM_EXTRA_ARGS:-}"
SERVE_ARGS="--async-scheduling${ARM_EXTRA_ARGS:+ $ARM_EXTRA_ARGS}"

# Speculative depth. Default 6 since 2026-09-21 (round 104), matching the pin;
# it was 7 for every arm up to `wooff-k6-rc2`. Both are overridable so a
# k-sweep arm can move one without editing this file.
K="${NUM_SPECULATIVE_TOKENS:-6}"
CAP="${MAX_CUDAGRAPH_CAPTURE_SIZE:-48}"

# Remove every container on both nodes before launching. A leftover reference
# (`sparkrun_*`) or prior container holds port 8000 and hijacks the measurement
# (round 114: `ctl-hum`, `proto2-030-0b`, `abOTH-d` were silently served by the
# anemll reference this way). `docker rm -f` all, not just our own name.
docker rm -f $(docker ps -aq) >/dev/null 2>&1
# InstantTensor sizes its I/O buffer from free device memory, so a full page cache
# starves it: `buffer_size (1059061760 B) exceeds device memory budget (925720576 B)`
# and the worker dies before health. Drop caches on both nodes first. See HANDOFF.md
# ("Drop the page cache before launching").
ssh spark2 "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" >/dev/null 2>&1 || \
  echo "warn: could not drop spark2 page cache"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || \
  echo "warn: could not drop spark1 page cache"
ssh spark2 "docker rm -f \$(docker ps -aq) >/dev/null 2>&1; find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null; cd ~/vllm-spark-0731 && env NUM_SPECULATIVE_TOKENS=${K} MAX_CUDAGRAPH_CAPTURE_SIZE=${CAP} GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='${SERVE_ARGS}' ${EXTRA} nohup bash scripts/05-serve.sh main-029 > ~/serve-${TAG}-w.log 2>&1 </dev/null &"
find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null || true
sleep 85
env NUM_SPECULATIVE_TOKENS=${K} MAX_CUDAGRAPH_CAPTURE_SIZE=${CAP} GPU_MEMORY_UTILIZATION=0.8389 \
  SERVE_EXTRA_ARGS="${SERVE_ARGS}" ${EXTRA} \
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
docker logs vllm-ds4-0731 2>&1 | grep -iE "attention|MoE backend|Experts|KV cache format|indexer|deep_gemm|E8M0|linear|all-reduce|Capturing CUDA|Selected .*Kernel" | head -40 > "$HOME/goal/${TAG}-engine.log"
# The contract protocol is 3 passes. More passes tighten the median spread,
# which is the bar the keep-rule is measured against, so a re-baseline can ask
# for more without changing anything else.
~/drive-median.sh "$TAG" "${PASSES:-3}" 512 1 3 5 6
