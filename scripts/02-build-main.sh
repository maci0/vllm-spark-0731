#!/usr/bin/env bash
# Phase 1 matched-main image (docs/PLAN-MAIN.md). Build on spark1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.main.env"

TAG="${1:-${IMAGE}}"
PLATFORM="${DOCKER_PLATFORM:-linux/arm64}"
export DOCKER_BUILDKIT=1

echo "build ${TAG}"
echo "  base ${FINAL_BASE_IMAGE}"
echo "  torch ${TORCH_REPO} ${TORCH_REF} arch ${TORCH_CUDA_ARCH_LIST} jobs ${MAX_JOBS_TORCH}"
echo "  vllm ${VLLM_REPO} ${VLLM_REF}"
echo "  host $(hostname) $(uname -m)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
free -h | head -2
docker pull --platform "${PLATFORM}" "${FINAL_BASE_IMAGE}"

docker build \
  --progress=plain \
  --platform "${PLATFORM}" \
  --build-arg "CUDA_IMAGE=${FINAL_BASE_IMAGE}" \
  --build-arg "TORCH_REPO=${TORCH_REPO}" \
  --build-arg "TORCH_REF=${TORCH_REF}" \
  --build-arg "TORCHVISION_REPO=${TORCHVISION_REPO}" \
  --build-arg "TORCHVISION_REF=${TORCHVISION_REF:-release/0.29}" \
  --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}" \
  --build-arg "TRITON_VERSION=${TRITON_VERSION}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}" \
  --build-arg "QUACK_KERNELS_VERSION=${QUACK_KERNELS_VERSION}" \
  --build-arg "FLASHINFER_REPO=${FLASHINFER_REPO}" \
  --build-arg "FLASHINFER_REF=${FLASHINFER_REF}" \
  --build-arg "DEEPGEMM_COMMIT=${DEEPGEMM_COMMIT}" \
  --build-arg "INSTANTTENSOR_REPO=${INSTANTTENSOR_REPO}" \
  --build-arg "INSTANTTENSOR_REF=${INSTANTTENSOR_REF}" \
  --build-arg "FASTSAFETENSORS_REPO=${FASTSAFETENSORS_REPO}" \
  --build-arg "FASTSAFETENSORS_REF=${FASTSAFETENSORS_REF}" \
  --build-arg "LMCACHE_REPO=${LMCACHE_REPO}" \
  --build-arg "LMCACHE_REF=${LMCACHE_REF}" \
  --build-arg "TILELANG_VERSION=${TILELANG_VERSION}" \
  --build-arg "HUMMING_KERNELS_VERSION=${HUMMING_KERNELS_VERSION}" \
  --build-arg "APACHE_TVM_FFI_VERSION=${APACHE_TVM_FFI_VERSION}" \
  --build-arg "TOKENSPEED_MLA_VERSION=${TOKENSPEED_MLA_VERSION}" \
  --build-arg "NVIDIA_CUDA_NVDISASM_VERSION=${NVIDIA_CUDA_NVDISASM_VERSION}" \
  --build-arg "NCCL_REPO=${NCCL_REPO}" \
  --build-arg "NCCL_NVCC_GENCODE=${NCCL_NVCC_GENCODE}" \
  --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
  --build-arg "MAX_JOBS_TORCH=${MAX_JOBS_TORCH}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  -t "${TAG}" \
  -f "${ROOT}/docker/Dockerfile.main" \
  "${ROOT}"

docker image inspect "${TAG}" --format '{{.Id}} {{.RepoTags}} {{.Architecture}} {{.Size}}'
echo "built ${TAG}"
