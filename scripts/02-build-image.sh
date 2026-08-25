#!/usr/bin/env bash
# Build the legacy fallback overlay image (vllm-spark-0731:v0.28.0rc2-b12x) on linux/arm64.
# NOTE: scripts/02-build-main.sh is the primary builder for the matched-main image (vllm-spark-0731:main-b12x).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.env"
# shellcheck disable=SC1091
source "${ROOT}/configs/env.spark.sh"

TAG="${1:-${IMAGE}}"
PLATFORM="${DOCKER_PLATFORM:-linux/arm64}"

echo "build ${TAG} from vllm/vllm-openai:${VLLM_RELEASE} platform=${PLATFORM}"
docker build --no-cache \
  --platform "${PLATFORM}" \
  --build-arg "VLLM_RELEASE=${VLLM_RELEASE}" \
  --build-arg "B12X_VERSION=${B12X_VERSION}" \
  --build-arg "RECIPE_VERSION=$(tr -d '[:space:]' < "${ROOT}/VERSION")" \
  -t "${TAG}" \
  -f "${ROOT}/Dockerfile" \
  "${ROOT}"

docker image inspect "${TAG}" --format '{{.Id}} {{.RepoTags}} {{.Architecture}}'
echo "built ${TAG}"
