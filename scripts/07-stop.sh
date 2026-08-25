#!/usr/bin/env bash
# docker rm -f does not free a wedged vLLM on Spark. Kill VLLM:: leftovers too.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.main.env"
# shellcheck disable=SC1091
source "${ROOT}/configs/env.spark.sh"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
pkill -9 -f '05-serve.sh' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
pkill -9 -f 'vllm serve' 2>/dev/null || true
echo "stopped ${CONTAINER_NAME} and leftover VLLM/EngineCore procs"

# Clean shared memory left by NCCL / torch multiprocessing
rm -f /dev/shm/nccl-* /dev/shm/cuda_* /dev/shm/vllm_* 2>/dev/null || true
find /dev/shm -maxdepth 1 -user "$(id -u)" -type f -delete 2>/dev/null || true

# Drop filesystem page cache so the next launch profiles clean memory
sync
if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
  echo "cleaned shm and dropped fs cache"
else
  echo "cleaned shm (drop_caches needs sudo, run: sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches')"
fi
