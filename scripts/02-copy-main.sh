#!/usr/bin/env bash
# Copy the Phase 1 image from spark1 to spark2 (docs/PLAN-MAIN.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.main.env"
TAG="${1:-${IMAGE}}"

docker image inspect "${TAG}" >/dev/null
echo "copy ${TAG} -> spark2"
docker save "${TAG}" | ssh spark2 docker load
ssh spark2 docker image inspect "${TAG}" --format '{{.Id}} {{.RepoTags}} {{.Architecture}} {{.Size}}'
echo "copied ${TAG}"
