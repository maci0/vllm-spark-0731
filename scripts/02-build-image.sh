#!/usr/bin/env bash
# Build the 0.28 + b12x image on linux/arm64 (the Spark).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.env"
source "${ROOT}/configs/env.spark.sh"

TAG="${1:-${IMAGE}}"
PLATFORM="${DOCKER_PLATFORM:-linux/arm64}"

echo "build ${TAG} from vllm/vllm-openai:${VLLM_RELEASE} platform=${PLATFORM}"
docker build \
  --platform "${PLATFORM}" \
  --build-arg "VLLM_RELEASE=${VLLM_RELEASE}" \
  --build-arg "B12X_VERSION=${B12X_VERSION}" \
  --build-arg "RECIPE_VERSION=$(tr -d '[:space:]' < "${ROOT}/VERSION")" \
  -t "${TAG}" \
  -f "${ROOT}/Dockerfile" \
  "${ROOT}"

docker image inspect "${TAG}" --format '{{.Id}} {{.RepoTags}} {{.Architecture}}'
echo "built ${TAG}"
