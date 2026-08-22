#!/usr/bin/env bash
# docker rm -f does not free a wedged vLLM on Spark. Kill VLLM:: leftovers too.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.env"
source "${ROOT}/configs/env.spark.sh"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
pkill -9 -f 'EngineCore' 2>/dev/null || true
echo "stopped ${CONTAINER_NAME} and leftover VLLM/EngineCore procs"
