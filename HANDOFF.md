# Handoff: 2-Node vLLM for DeepSeek-V4-Flash-0731 on DGX Spark (GB10)

## Overview

Deploy `deepseek-ai/DeepSeek-V4-Flash-0731` across 2x DGX Spark nodes using
vLLM v0.28.0rc2 (installed from source onto v0.27.1 base) + b12x==1.2.6.
fp8_ds_mla KV cache, DSpark k=5 speculative decoding, TP=2 over RoCE.

**Why v0.27.1 base**: v0.28.0rc2 has no arm64 Docker image on Docker Hub.
v0.27.1 is the latest arm64 release. We uninstall v0.27.1 Python code and
install rc2 from source; the base provides PyTorch 2.13, CUDA 13.0, system libs.
Build-time overlays add b12x MoE integration, fp8_einsum SM12x fallback,
mHC TileLang guard, and nvfp4_ds_mla 584-byte page support.

---

## Cluster

| Node | IP (fabric) | IP (mgmt) | Role |
|------|-------------|-----------|------|
| spark1 | 10.0.1.1 | 192.168.0.211 | head (rank 0) |
| spark2 | 10.0.1.2 | 192.168.0.212 | worker (rank 1) |

- **GPU**: NVIDIA GB10, SM12x (capability 12.1, family 120), 128 GiB UMA per node
- **Fabric**: ConnectX-7 RoCE, `enp1s0f1np1`, NCCL IB GID 3
- **Model**: 155.43 GiB safetensors, `/models/ds4-flash-0731`

---

## Docker Image

- **Tag**: `vllm-spark-0731:v0.28.0rc2-b12x`
- **Base**: `vllm/vllm-openai:v0.27.1` (arm64)
- **vLLM**: v0.28.0rc2 installed from source (replaces v0.27.1 Python code)
- **Added**: `b12x==1.2.6` via uv
- **Overlays**: `patches/apply_overlays.py` (build-time string-replace patches)
- **Asserts**: `patches/assert_image.py` (build-time source-level checks)

All patches are baked into the image at build time. No runtime volume mounts needed.

---

## SM12x kernel guards

DeepGEMM and CUTLASS block-FP8 kernels target SM100+. On SM12x they crash.

| Guard | File | Effect |
|-------|------|--------|
| `is_deep_gemm_supported()` | `utils/deep_gemm.py` | Returns False on family 120 |
| `fp8_einsum` fallback | `utils/deep_gemm.py` | Dequant FP8 to bf16 + torch.einsum |
| `cutlass_block_fp8_supported()` | `w8a8_utils.py` | Returns False on family 120 |
| `compute_fp8_einsum_recipe` | `o_proj.py` | Returns ((1,128,128), False) on family 120 |
| Triton e8m0fnu canonicalization | `torch_utils.py` | Maps e8m0fnu to u8 |
| `VLLM_USE_DEEP_GEMM_E8M0=0` | Dockerfile env | Disables E8M0 at env level |

---

## Memory settings

128 GiB UMA per node. Model weights ~77.7 GiB/rank. b12x weight prep adds overhead.

| Setting | Value | Notes |
|---------|-------|-------|
| GPU_MEMORY_UTILIZATION | 0.80 | 102.4 GiB budget. 0.62 gave -6.9 GiB for KV cache |
| MAX_MODEL_LEN | 65536 | |
| MAX_NUM_SEQS | 2 | Conservative for initial bring-up |
| MAX_NUM_BATCHED_TOKENS | 2048 | |
| MAX_CUDAGRAPH_CAPTURE_SIZE | 36 | |

---

## Operating the cluster

### Stop (cleans shm, prompts for fs cache drop)
```bash
ssh spark1 "cd ~/Desktop/trt-llm-ds4/vllm-spark-0731 && bash scripts/07-stop.sh"
ssh spark2 "cd ~/Desktop/trt-llm-ds4/vllm-spark-0731 && bash scripts/07-stop.sh"
# Optional: sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' on each node
```

### Launch (worker first, then head)
```bash
ssh spark2 "cd ~/Desktop/trt-llm-ds4/vllm-spark-0731 && nohup bash scripts/05-serve.sh fp8 > ~/vllm_worker.log 2>&1 &"
# Wait for worker to reach distributed_init, then:
ssh spark1 "cd ~/Desktop/trt-llm-ds4/vllm-spark-0731 && nohup bash scripts/05-serve.sh fp8 > ~/vllm_head.log 2>&1 &"
```

`nodes.env` auto-sets `VLLM_HOST_IP` and `NODE_RANK` from hostname.

### Check logs
```bash
ssh spark1 "tail -20 ~/vllm_head.log"
ssh spark2 "tail -20 ~/vllm_worker.log"
```

---

## Status

- [x] SM12x kernel guards (DeepGEMM, CUTLASS block-FP8, Triton e8m0)
- [x] fp8_einsum SM12x fallback (baked into overlay)
- [x] b12x MoE weight preparation (baked into overlay)
- [x] nodes.env auto-detects hostname for NODE_RANK/VLLM_HOST_IP
- [x] Stop script cleans shm
- [ ] Build image with rc2 from source + overlays
- [ ] fp8_ds_mla serve at GPU_MEMORY_UTILIZATION=0.80
- [ ] Validate with scripts/06-validate.sh
- [ ] Push memory utilization higher, document KV cache token counts
- [ ] Test nvfp4_ds_mla mode
- [ ] Benchmark (tok/s, B/token)

---

## Key learnings

1. **v0.28.0rc2 arm64 image does not exist** on Docker Hub. v0.27.1 is the latest. We install rc2 from source on the v0.27.1 base.
2. **rc2 still lacks**: B12xExperts MoE integration, fp8_einsum SM12x fallback, nvfp4_ds_mla support. All patched by `apply_overlays.py`.
3. **LINEAR_BACKEND must be empty** (auto). b12x linear only covers NVFP4 weight layers, not FP8 attention projections. Auto-select correctly picks TritonFp8BlockScaledMMKernel for linear + B12X_MXFP4_MXFP8 for MoE.
4. **fp8_einsum SM12x fallback**: rc2 has no fallback (calls `_missing()` which raises RuntimeError). The overlay adds dequant-to-bf16 + torch.einsum with correct weight reshape `[h*d, r] -> [h, d, r]`.
5. **B12xExperts.process_weights_after_loading** must be called by the MXFP4 kernel factory. rc2 omits this call. The overlay patches both the factory function and `Mxfp4MoEMethod` caller.
6. **GPU_MEMORY_UTILIZATION=0.62** is too low. Model + b12x prepared weights consume ~86.3 GiB/rank, leaving negative KV cache headroom. Use 0.80+.
