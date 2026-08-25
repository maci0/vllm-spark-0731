#!/usr/bin/env bash
# Phase 2 overlays on vllm-spark-0731:main-b12x (docs/PLAN-MAIN.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.main.env"
TAG="${1:-${IMAGE}}"
export DOCKER_BUILDKIT=1

docker image inspect "${TAG}" >/dev/null
echo "overlay ${TAG}"
docker build \
  --progress=plain \
  --platform "${DOCKER_PLATFORM:-linux/arm64}" \
  -t "${TAG}" \
  -f "${ROOT}/docker/Dockerfile.main-overlays" \
  "${ROOT}"
docker image inspect "${TAG}" --format '{{.Id}} {{.RepoTags}} {{.Size}}'
echo "overlaid ${TAG}"
