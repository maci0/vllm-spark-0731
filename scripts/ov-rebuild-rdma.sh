#!/usr/bin/env bash
# Rebuild + resync the v0.28.0 RoCE overlay image after a patches/files edit.
#
# Flow (runs on spark1; requires the repo synced to /tmp/vllm-spark-0731 and
# the nodes up):
#   1. rebuild vllm-spark-0731:main-b12x-028-rdma (FROM the existing rdma
#      image, which carries the rdma-core v54 libmlx5 fix; re-copies the
#      tracked patches/files donors into site-packages)
#   2. docker save | ssh spark2 docker load
#
# Boot afterwards (both nodes):
#   worker (spark2) first, head (spark1) ~90s later:
#     docker rm -f vllm-ds4-0731
#     VLLM_USE_AOT_COMPILE=0 nohup bash scripts/05-serve.sh main-dg
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-vllm-spark-0731:main-b12x-028-rdma}"

[ "$(hostname -s)" = "spark1" ] || { echo "run this on spark1"; exit 1; }
cd "${ROOT}"

docker image inspect "${TAG}" >/dev/null 2>&1 || {
  echo "parent ${TAG} missing - did the repo sync include docker/Dockerfile.ov-rdma?" >&2
  exit 1
}

echo "==> rebuild ${TAG}"
docker build -f docker/Dockerfile.ov-rdma -t "${TAG}" .

echo "==> sync ${TAG} -> spark2"
docker save "${TAG}" | ssh -o StrictHostKeyChecking=no spark2 docker load
ssh -o StrictHostKeyChecking=no spark2 docker image inspect "${TAG}" --format \
  '{{.Id}} {{.RepoTags}} {{.Size}}'
echo "==> done. boot worker (spark2) then head (spark1):"
echo "    docker rm -f vllm-ds4-0731; VLLM_USE_AOT_COMPILE=0 nohup bash scripts/05-serve.sh main-dg"
