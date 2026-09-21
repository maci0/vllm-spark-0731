#!/usr/bin/env bash
# Phase 2 overlays for the current pin (configs/pin.main-029.env).
#
#   scripts/03-apply-main-overlays-029.sh <TAG> [BASE_IMAGE]
#
# With no BASE_IMAGE the overlays are re-applied on top of TAG itself, which is
# how an incremental rebuild on an already-overlaid image is done. Pass the
# phase-1 image to build a fresh one, e.g.
#   scripts/03-apply-main-overlays-029.sh \
#     vllm-spark-0731:main-030-rc2 vllm-spark-0731:main-030-rc2-phase1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.main-029.env"
TAG="${1:-${IMAGE}}"
BASE_IMAGE="${2:-${TAG}}"
export DOCKER_BUILDKIT=1

docker image inspect "${BASE_IMAGE}" >/dev/null
echo "overlay ${TAG} from ${BASE_IMAGE}"
docker build \
  --progress=plain \
  --platform "${DOCKER_PLATFORM:-linux/arm64}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${TAG}" \
  -f "${ROOT}/docker/Dockerfile.main-overlays" \
  "${ROOT}"
docker image inspect "${TAG}" --format '{{.Id}} {{.RepoTags}} {{.Size}}'
echo "overlaid ${TAG}"
