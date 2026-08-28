# shellcheck shell=bash
# Shared host/container env for DGX Spark (GB10, sm_121, UMA, ConnectX-7).
# Sourced by launch scripts. Safe to source more than once.

export HF_MODEL_ID="${HF_MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
export MODEL_DIR="${MODEL_DIR:-/models/ds4-flash-0731}"
export IMAGE="${IMAGE:-vllm-spark-0731:v0.28.0rc2-b12x}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ds4-0731}"
export SERVE_PORT="${SERVE_PORT:-8000}"
export TP_SIZE="${TP_SIZE:-2}"
export NNODES="${NNODES:-2}"
export NODE_RANK="${NODE_RANK:-}"

# Spark NICs from the official playbook. Override if your QSFP is on the other port.
export SPARK_IFACES="${SPARK_IFACES:-enp1s0f1np1}"

# NCCL / UCX over RoCE (ConnectX-7).
# 2026-08-28: with rdma-core v54 libmlx5 (the NCCL 2.30.7-compatible
# userspace, see docs/knowledge/05-performance.md "NCCL was running on TCP"),
# NCCL auto-selects the working RoCE path. Forcing NCCL_IB_GID_INDEX=3 or
# NCCL_NET_GDR_LEVEL=PHB re-breaks it (back to 0.37ms small-op latency);
# the defaults give 0.031-0.045ms on rocep1s0f1 + roceP2p1s0f1.
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-${SPARK_IFACES}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SPARK_IFACES}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SPARK_IFACES}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${SPARK_IFACES}}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# Per-node fabric IP. Required for 2-node mp. Set before serve.
export VLLM_HOST_IP="${VLLM_HOST_IP:-}"
export HEAD_IP="${HEAD_IP:-}"

export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"
# GB10: the DeepGEMM ue8m0 path needs a working fp8 linear stack; the b12x
# image ships no DeepGEMM and the 8b1392b pin regressed SM12x fp8 (see
# docs/knowledge/09-golden-deepgemm.md). Default 0, matching pin.main.env.
# GB10: ptxas cannot assemble tcgen05 for sm_121a on any CUDA 13.x tested.
# NVRTC was tried as the workaround but its cubins are rejected by the driver
# (CUDA_ERROR_INVALID_IMAGE), so DG_JIT_USE_NVRTC stays 0 (see
# docs/knowledge/09-golden-deepgemm.md → Validation status). pin.main.env
# forces it to 0 for the live stack; keep the default in sync.
export DG_JIT_USE_NVRTC="${DG_JIT_USE_NVRTC:-0}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-4096}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
