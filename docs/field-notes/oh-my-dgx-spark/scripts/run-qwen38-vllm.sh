#!/usr/bin/env bash
set -euo pipefail

# DGX Spark single-node Qwen3.8 runner.
# The local venv already contains vLLM, PyTorch/cu130, Triton, FlashInfer,
# and ninja. The Python development headers are used only when Triton or
# FlashInfer needs to compile a small launcher on an ARM64 host.

VLLM_VENV="${VLLM_VENV:-/home/grid/vllm-venv}"
MODEL_PATH="${QWEN38_MODEL_PATH:-/home/grid/.cache/huggingface/hub/models--OBLITERATUS--Qwen3.8-27B-OBLITERATED/snapshots/46c3c40faf9d89c692d8e82514cb5fe3d0f7fa83}"
PORT="${QWEN38_PORT:-8083}"
LOCAL_PYTHON_DEV="${DGX_SPARK_PYTHON_DEV_ROOT:-/home/grid/.local/dgx-spark-python-dev}"

if [[ ! -x "${VLLM_VENV}/bin/vllm" ]]; then
  echo "vLLM executable not found: ${VLLM_VENV}/bin/vllm" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Qwen3.8 model snapshot not found: ${MODEL_PATH}" >&2
  exit 1
fi

export PATH="${VLLM_VENV}/bin:${PATH}"

# Prefer system headers when python3.12-dev is installed. Otherwise use the
# non-root header bundle prepared under LOCAL_PYTHON_DEV.
if [[ -f /usr/include/python3.12/Python.h ]]; then
  PYTHON_INCLUDE="/usr/include/python3.12"
  PYTHON_INCLUDE_ROOT="/usr/include"
else
  PYTHON_INCLUDE="${LOCAL_PYTHON_DEV}/usr/include/python3.12"
  PYTHON_INCLUDE_ROOT="${LOCAL_PYTHON_DEV}/usr/include"
fi

if [[ ! -f "${PYTHON_INCLUDE}/Python.h" ]]; then
  echo "Python.h not found. Install python3.12-dev or prepare ${LOCAL_PYTHON_DEV}." >&2
  exit 1
fi

export CPATH="${PYTHON_INCLUDE}:${PYTHON_INCLUDE_ROOT}${CPATH:+:${CPATH}}"

exec "${VLLM_VENV}/bin/vllm" serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name qwen3.8-27b-obliterated \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.50 \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --generation-config vllm \
  --trust-remote-code \
  "$@"
