#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731 with b12x kernels and DSpark k=5.
# Usage: 05-serve.sh [fp8|nvfp4|eugr]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

STACK="${1:-fp8}"
case "${STACK}" in
  fp8) PIN="${ROOT}/configs/pin.env" ;;
  nvfp4) PIN="${ROOT}/configs/pin.nvfp4.env" ;;
  eugr) PIN="${ROOT}/configs/pin.eugr-b12x.env" ;;
  *) echo "usage: $0 [fp8|nvfp4|eugr]" >&2; exit 2 ;;
esac

# shellcheck disable=SC1090
source "${PIN}"
# shellcheck disable=SC1091
source "${ROOT}/configs/env.spark.sh"
if [[ -f "${ROOT}/configs/nodes.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/nodes.env"
fi

if [[ "${HF_MODEL_ID}" != "deepseek-ai/DeepSeek-V4-Flash-0731" ]]; then
  echo "refusing ${HF_MODEL_ID}" >&2
  exit 2
fi
if [[ -z "${VLLM_HOST_IP:-}" || -z "${HEAD_IP:-}" ]]; then
  echo "set VLLM_HOST_IP and HEAD_IP (copy configs/nodes.env.example)" >&2
  exit 2
fi

python3 "${ROOT}/patches/assert_stack.py" \
  --kv "${KV_CACHE_DTYPE}" \
  --attn "${ATTENTION_BACKEND:-}" \
  --moe "${MOE_BACKEND}" \
  --spec "${SPEC_METHOD}" \
  --k "${NUM_SPECULATIVE_TOKENS}"

HOST_MODEL="${HOST_MODEL_DIR:-${HOME}/models/ds4-flash-0731}"
python3 "${ROOT}/patches/assert_0731.py" "${HOST_MODEL}"

BLOBS_DIR="$(dirname "$(dirname "${HOST_MODEL}")")/blobs"
BLOBS_ARG=()
if [[ -d "${BLOBS_DIR}" ]]; then
  BLOBS_ARG+=(-v "${BLOBS_DIR}:/blobs:ro")
elif [[ -d "${HOME}/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/blobs" ]]; then
  BLOBS_ARG+=(-v "${HOME}/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/blobs:/blobs:ro")
fi

SPEC_JSON="$(python3 -c 'import json,sys; print(json.dumps({"method":sys.argv[1],"num_speculative_tokens":int(sys.argv[2]),"draft_sample_method":sys.argv[3]}))' \
  "${SPEC_METHOD}" "${NUM_SPECULATIVE_TOKENS}" "${DRAFT_SAMPLE_METHOD}")"

ATTN_ARGS=()
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  ATTN_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi
LINEAR_ARGS=()
if [[ -n "${LINEAR_BACKEND:-}" ]]; then
  LINEAR_ARGS+=(--linear-backend "${LINEAR_BACKEND}")
fi

HEADLESS_ARGS=()
if [[ "${NODE_RANK}" != "0" ]]; then
  HEADLESS_ARGS+=(--headless)
fi

CAPTURE_ARGS=()
if [[ -n "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ]]; then
  CAPTURE_ARGS+=(--compilation-config "{\"max_cudagraph_capture_size\": ${MAX_CUDAGRAPH_CAPTURE_SIZE}}")
fi

IB_ARGS=(--privileged --ulimit memlock=-1 --ulimit stack=67108864)
if [[ -d /dev/infiniband ]]; then
  IB_ARGS+=(--device /dev/infiniband)
fi

echo "serve stack=${STACK} kv=${KV_CACHE_DTYPE} moe=${MOE_BACKEND} spec=${SPEC_JSON} rank=${NODE_RANK} master=${HEAD_IP}"
if [[ "${NODE_RANK}" == "0" ]]; then
  echo "start the worker (rank 1) first, then this head"
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm --name "${CONTAINER_NAME}" --gpus all --ipc=host --network host \
  --shm-size=16g \
  "${IB_ARGS[@]}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_MODEL_ID="${HF_MODEL_ID}" \
  -e CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}" \
  -e VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}" \
  -e VLLM_HOST_IP="${VLLM_HOST_IP}" \
  -e MASTER_ADDR="${HEAD_IP}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  -e NCCL_IB_HCA="${NCCL_IB_HCA}" \
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}" \
  -e NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}" \
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
  -e UCX_NET_DEVICES="${UCX_NET_DEVICES}" \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
  -e TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME}" \
  -e CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}" \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH}" \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${HOST_MODEL}:${MODEL_DIR}:ro" \
  "${BLOBS_ARG[@]}" \
  "${IMAGE}" \
  "${MODEL_DIR}" \
    --host 0.0.0.0 \
    --port "${SERVE_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${HEAD_IP}" \
    --distributed-executor-backend mp \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --moe-backend "${MOE_BACKEND}" \
    "${LINEAR_ARGS[@]}" \
    "${ATTN_ARGS[@]}" \
    "${HEADLESS_ARGS[@]}" \
    --block-size "${BLOCK_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "${CAPTURE_ARGS[@]}" \
    --speculative-config "${SPEC_JSON}" \
    --tokenizer-mode "${TOKENIZER_MODE:-deepseek_v4}" \
    --reasoning-parser "${REASONING_PARSER:-deepseek_v4}" \
    --tool-call-parser "${TOOL_CALL_PARSER:-deepseek_v4}" \
    --enable-auto-tool-choice \
    --served-model-name ${SERVED_MODEL_NAME} \
    --trust-remote-code
