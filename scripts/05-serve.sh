#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731 with b12x kernels and DSpark k=5.
# Usage: 05-serve.sh [fp8|nvfp4|eugr|golden|main]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Capture variables from process environment before configs are sourced
_INITIAL_NODE_RANK="${NODE_RANK:-}"
_INITIAL_VLLM_HOST_IP="${VLLM_HOST_IP:-}"
_INITIAL_HEAD_IP="${HEAD_IP:-}"
_INITIAL_HOST_MODEL_DIR="${HOST_MODEL_DIR:-}"
_INITIAL_SERVE_PORT="${SERVE_PORT:-}"

STACK="${1:-main}"
case "${STACK}" in
  fp8) PIN="${ROOT}/configs/pin.env" ;;
  nvfp4) PIN="${ROOT}/configs/pin.nvfp4.env" ;;
  eugr) PIN="${ROOT}/configs/pin.eugr-b12x.env" ;;
  golden) PIN="${ROOT}/configs/pin.golden.env" ;;
  main) PIN="${ROOT}/configs/pin.main.env" ;;
  main-dg) PIN="${ROOT}/configs/pin.main-dg.env" ;;
  *) echo "usage: $0 [fp8|nvfp4|eugr|golden|main|main-dg]" >&2; exit 2 ;;
esac

# shellcheck disable=SC1090
source "${PIN}"
# shellcheck disable=SC1091
source "${ROOT}/configs/env.spark.sh"

if [[ -f "${ROOT}/configs/nodes.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/nodes.env"
fi

NODE_RANK="${_INITIAL_NODE_RANK:-${NODE_RANK:-0}}"
VLLM_HOST_IP="${_INITIAL_VLLM_HOST_IP:-${VLLM_HOST_IP:-127.0.0.1}}"
HEAD_IP="${_INITIAL_HEAD_IP:-${HEAD_IP:-127.0.0.1}}"
HOST_MODEL_DIR="${_INITIAL_HOST_MODEL_DIR:-${HOST_MODEL_DIR:-${HOME}/models/ds4-flash-0731}}"
SERVE_PORT="${_INITIAL_SERVE_PORT:-${SERVE_PORT:-8000}}"

if [[ "${HF_MODEL_ID}" != "deepseek-ai/DeepSeek-V4-Flash-0731" ]]; then
  echo "refusing ${HF_MODEL_ID}" >&2
  exit 2
fi
if [[ -z "${VLLM_HOST_IP:-}" || -z "${HEAD_IP:-}" ]]; then
  echo "set VLLM_HOST_IP and HEAD_IP (copy configs/nodes.env.example)" >&2
  exit 2
fi
if [[ "${NNODES:-1}" -gt 1 ]]; then
  if [[ "${HEAD_IP}" =~ ^(127\.|0\.0\.0\.0|localhost) || "${VLLM_HOST_IP}" =~ ^(127\.|0\.0\.0\.0|localhost) ]]; then
    echo "ERROR: NNODES=${NNODES} requires cluster fabric IPs for HEAD_IP and VLLM_HOST_IP (got HEAD_IP=${HEAD_IP}, VLLM_HOST_IP=${VLLM_HOST_IP})." >&2
    echo "Please copy configs/nodes.env.example to configs/nodes.env and configure fabric IPs." >&2
    exit 2
  fi
fi

if [[ "${DISABLE_DSPARK:-0}" != "1" ]]; then
  python3 "${ROOT}/patches/assert_stack.py" \
    --kv "${KV_CACHE_DTYPE}" \
    --attn "${ATTENTION_BACKEND:-}" \
    --moe "${MOE_BACKEND}" \
    --spec "${SPEC_METHOD}" \
    --k "${NUM_SPECULATIVE_TOKENS}"
else
  echo "DISABLE_DSPARK=1: isolation boot, speculative decode off"
fi

HOST_MODEL="${HOST_MODEL_DIR:-${HOME}/models/ds4-flash-0731}"
python3 "${ROOT}/patches/assert_0731.py" "${HOST_MODEL}"

BLOBS_DIR="$(dirname "$(dirname "${HOST_MODEL}")")/blobs"
BLOBS_ARG=()
if [[ -d "${BLOBS_DIR}" ]]; then
  BLOBS_ARG+=(-v "${BLOBS_DIR}:/blobs:ro")
elif [[ -d "${HOME}/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/blobs" ]]; then
  BLOBS_ARG+=(-v "${HOME}/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/blobs:/blobs:ro")
fi

SPEC_JSON="$(python3 -c 'import json,sys
d={"method":sys.argv[1],"num_speculative_tokens":int(sys.argv[2]),"draft_sample_method":sys.argv[3]}
attn=sys.argv[4] if len(sys.argv)>4 else ""
if attn:
    d["attention_backend"]=attn
print(json.dumps(d))' \
  "${SPEC_METHOD}" "${NUM_SPECULATIVE_TOKENS}" "${DRAFT_SAMPLE_METHOD}" \
  "${DRAFT_ATTENTION_BACKEND:-${ATTENTION_BACKEND:-}}")"
SPEC_ARGS=()
if [[ "${DISABLE_DSPARK:-0}" != "1" ]]; then
  SPEC_ARGS+=(--speculative-config "${SPEC_JSON}")
fi

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

LOAD_ARGS=()
if [[ -n "${LOAD_FORMAT:-}" ]]; then
  LOAD_ARGS+=(--load-format "${LOAD_FORMAT}")
fi
PP_ARGS=()
if [[ -n "${PP_SIZE:-}" ]]; then
  PP_ARGS+=(--pipeline-parallel-size "${PP_SIZE}")
fi
PREFIX_ARGS=()
if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  PREFIX_ARGS+=(--enable-prefix-caching)
fi

PROFILE_ARGS=()
if [[ "${PROFILE_ENABLE:-0}" == "1" ]]; then
  # Canonical dotted form (the JSON-string form parses on the API frontend
  # but is dropped in the engine -> workers' ProfilerConfig, "Profiling is
  # not enabled"). Exposes POST /start_profile + /stop_profile.
  PROFILE_ARGS+=(--profiler-config.profiler torch)
  PROFILE_ARGS+=(--profiler-config.torch_profiler_dir "${PROFILE_DIR:-/tmp/profile}")
  if [[ "${PROFILE_IGNORE_FRONTEND:-0}" == "1" ]]; then
    PROFILE_ARGS+=(--profiler-config.ignore_frontend true)
  fi
fi

# Serve under the checkpoint's own id unless an alias is explicitly wanted.
SERVED_MODEL_ARGS=()
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
  SERVED_MODEL_ARGS+=(--served-model-name ${SERVED_MODEL_NAME})
fi

HOST_KV_OFFLOAD_DIR="${HOST_KV_OFFLOAD_DIR:-${HOME}/lmcache-0731}"
KV_TRANSFER_ARGS=()
KV_OFFLOAD_DOCKER_ARGS=()
if [[ "${ENABLE_LMCACHE:-0}" == "1" ]]; then
  KV_OFFLOAD="${KV_OFFLOAD:-lmcache}"
fi
if [[ "${KV_OFFLOAD:-}" == "lmcache" ]]; then
  mkdir -p "${HOST_KV_OFFLOAD_DIR}"
  KV_OFFLOAD_DOCKER_ARGS+=(
    -e LMCACHE_CONFIG_FILE=/configs/lmcache.gds.yaml
    -v "${ROOT}/configs/lmcache.gds.yaml:/configs/lmcache.gds.yaml:ro"
    -v "${HOST_KV_OFFLOAD_DIR}:/mnt/nvme/lmcache-0731"
  )
  KV_TRANSFER_ARGS+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
elif [[ "${KV_OFFLOAD:-}" == "native" ]]; then
  mkdir -p "${HOST_KV_OFFLOAD_DIR}"
  KV_OFFLOAD_DOCKER_ARGS+=(
    -v "${HOST_KV_OFFLOAD_DIR}:/mnt/nvme/lmcache-0731"
  )
  KV_TRANSFER_ARGS+=(--kv-transfer-config "$(python3 -c 'import json,pathlib,sys; print(json.dumps(json.loads(pathlib.Path(sys.argv[1]).read_text())))' "${ROOT}/configs/kv-offload.native.json")")
fi

# KV connectors pin/register cache pages. expandable_segments remaps VAs and
# vLLM rejects that unless the cumem allocator is on. The image ENV sets both
# PYTORCH_CUDA_ALLOC_CONF and PYTORCH_ALLOC_CONF; omit is not enough.
ALLOC_CONF_ARGS=()
if [[ -n "${KV_OFFLOAD:-}" ]]; then
  ALLOC_CONF_ARGS+=(
    -e PYTORCH_CUDA_ALLOC_CONF=
    -e PYTORCH_ALLOC_CONF=
  )
else
  ALLOC_CONF_ARGS+=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
fi

CAPTURE_ARGS=()
if [[ "${ENFORCE_EAGER:-0}" != "1" ]]; then
  _cc_parts=()
  if [[ -n "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ]]; then
    _cc_parts+=("\"max_cudagraph_capture_size\": ${MAX_CUDAGRAPH_CAPTURE_SIZE}")
  fi
  if [[ -n "${CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
    _cc_parts+=("\"cudagraph_capture_sizes\": ${CUDAGRAPH_CAPTURE_SIZES}")
  fi
  if [[ -n "${CUDAGRAPH_MODE:-}" ]]; then
    _cc_parts+=("\"cudagraph_mode\": \"${CUDAGRAPH_MODE}\"")
  fi
  if [[ -n "${CUDAGRAPH_COPY_INPUTS:-}" ]]; then
    _cc_parts+=("\"cudagraph_copy_inputs\": ${CUDAGRAPH_COPY_INPUTS}")
  fi
  if [[ ${#_cc_parts[@]} -gt 0 ]]; then
    _cc_joined=$(printf '%s,' "${_cc_parts[@]}")
    CAPTURE_ARGS+=(--compilation-config "{${_cc_joined%,}}")
  fi
fi
EAGER_ARGS=()
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  EAGER_ARGS+=(--enforce-eager)
fi

IB_ARGS=(--privileged --ulimit memlock=-1 --ulimit stack=67108864)
if [[ -d /dev/infiniband ]]; then
  IB_ARGS+=(--device /dev/infiniband)
fi

echo "serve stack=${STACK} kv=${KV_CACHE_DTYPE} moe=${MOE_BACKEND} spec=${SPEC_JSON} disable_dspark=${DISABLE_DSPARK:-0} enforce_eager=${ENFORCE_EAGER:-0} cudagraph_mode=${CUDAGRAPH_MODE:-default} copy_inputs=${CUDAGRAPH_COPY_INPUTS:-} kv_offload=${KV_OFFLOAD:-off} aot=${VLLM_USE_AOT_COMPILE:-0} rank=${NODE_RANK} master=${HEAD_IP}"
if [[ "${NODE_RANK}" == "0" ]]; then
  echo "start the worker (rank 1) first, then this head"
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
mkdir -p \
  "${HOME}/.triton" \
  "${HOME}/.cache/vllm" \
  "${HOME}/.cache/huggingface" \
  "${HOME}/.cache/flashinfer" \
  "${HOME}/.cache/instanttensor" \
  "${HOME}/.cache/b12x" \
  "${HOME}/.cache/torchinductor" \
  "${HOME}/.tilelang"

# Detached: the container must outlive SSH/tmux. --rm is omitted so a crash
# leaves an inspectable container (07-stop.sh still docker rm -f).
# shellcheck disable=SC2086
docker run -d -i --name "${CONTAINER_NAME}" --gpus all --ipc=host --network host \
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
  -e VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-0}" \
  -e VLLM_PRESERVE_SM12X_TARGET="${VLLM_PRESERVE_SM12X_TARGET:-0}" \
  -e DG_JIT_USE_NVRTC="${DG_JIT_USE_NVRTC:-0}" \
  -e INSTANTTENSOR_DRAFT_LOADER="${INSTANTTENSOR_DRAFT_LOADER:-auto}" \
  -e VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-0}" \
  -e VLLM_PROFILE_DECODE="${VLLM_PROFILE_DECODE:-0}" \
  -e B12X_MLA_SM120_UNIFIED="${B12X_MLA_SM120_UNIFIED:-}" \
  -e B12X_MOE_FORCE_A8="${B12X_MOE_FORCE_A8:-}" \
  -e VLLM_USE_B12X_WO_PROJECTION="${VLLM_USE_B12X_WO_PROJECTION:-}" \
  -e VLLM_USE_B12X_MHC="${VLLM_USE_B12X_MHC:-}" \
  -e VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-}" \
  -e VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-}" \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  "${ALLOC_CONF_ARGS[@]}" \
  -v "${HOME}/.triton:/root/.triton" \
  -v "${HOME}/.cache/vllm:/root/.cache/vllm" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "${HOME}/.cache/instanttensor:/root/.cache/instanttensor" \
  -v "${HOME}/.cache/b12x:/root/.cache/b12x" \
  -v "${HOME}/.cache/torchinductor:/root/.cache/torchinductor" \
  -v "${HOME}/.tilelang:/root/.tilelang" \
  "${KV_OFFLOAD_DOCKER_ARGS[@]}" \
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
    "${LOAD_ARGS[@]}" \
    "${PP_ARGS[@]}" \
    "${PREFIX_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${KV_TRANSFER_ARGS[@]}" \
    "${HEADLESS_ARGS[@]}" \
    --block-size "${BLOCK_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "${CAPTURE_ARGS[@]}" \
    "${EAGER_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    --tokenizer-mode "${TOKENIZER_MODE:-deepseek_v4}" \
    --reasoning-parser "${REASONING_PARSER:-deepseek_v4}" \
    --tool-call-parser "${TOOL_CALL_PARSER:-deepseek_v4}" \
    --enable-auto-tool-choice \
    ${SERVED_MODEL_ARGS[@]} \
    --trust-remote-code

echo "started ${CONTAINER_NAME} $(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}")"
if [[ -t 1 ]]; then
  exec docker logs -f "${CONTAINER_NAME}"
fi
echo "follow logs with: docker logs -f ${CONTAINER_NAME}"
