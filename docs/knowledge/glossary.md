[← Index](00-index.md)

# Glossary

Shared vocabulary used across the corpus. Definitions are grounded in the
chapters cited; see those chapters for depth.

| Term | Definition | See |
|------|-----------|-----|
| **b12x** | SM120/121-only CuTe DSL kernel library (linear FP8, MoE MXFP4, compressed MLA). Primary kernel path on Spark. | [03-kernels-attention](03-kernels-attention.md) |
| **B12X_MLA_SPARSE** | b12x compressed-MLA attention backend on the 584 B DSV4 page; live backend on matched-main (no stock vLLM enum — overlay-registered). | [03-kernels-attention](03-kernels-attention.md) |
| **C4A / C128A** | Sparse-attention branches for `compress_ratio` 4 and 128 — the 0731 checkpoint alternates them, so both run. C128A was the eidx-contiguity bug source. | [02-model](02-model.md), [03-kernels-attention](03-kernels-attention.md) |
| **CUTLASS block-FP8** | SM90/SM100 TMA FP8 GEMM path; does **not** run on SM12x (falls back to PyTorch/Triton/TileLang). | [03-kernels-attention](03-kernels-attention.md), [09-golden-deepgemm](09-golden-deepgemm.md) |
| **DeepGEMM** | DeepSeek's FP8/FP4/BF16 GEMM library. SM12x kernels live on the **nv_dev** branch; the pure-FP8 1d1d path regressed at `8b1392b`. | [09-golden-deepgemm](09-golden-deepgemm.md) |
| **DG_JIT_USE_NVRTC** | DeepGEMM JIT flag (NVRTC driver-JIT vs ptxas). NVRTC was tried because ptxas cannot assemble tcgen05 for `sm_121a` on CUDA 13.x, but the driver rejects its cubins — flag stays **0** on the live pin. | [09-golden-deepgemm](09-golden-deepgemm.md) |
| **DSpark** | DeepSeek-V4's native speculative-decoding method (`method=dspark`): k=5 locked, MTP draft on the same checkpoint. | [02-model](02-model.md) |
| **DSV4** | DeepSeek-V4. Also used for the **584 B DSV4 envelope** — the packed per-layer-token MLA page (448 B NoPE + 128 B RoPE + 8 B scale). | [02-model](02-model.md), [04-quantization-kv](04-quantization-kv.md) |
| **eidx** | "Extra sparse indices": the per-token sparse-attention index tensor. Must be contiguous for the SM120 kernel ("eidx must be contiguous" boot crash). | [03-kernels-attention](03-kernels-attention.md), [08-upstream](08-upstream.md) |
| **E8M0 / UE8M0** | Unsigned 8-bit-exponent scale format. SM12x upcasts it to fp32 (native 128×128 UE8M0 unsupported). | [04-quantization-kv](04-quantization-kv.md), [08-upstream](08-upstream.md) |
| **FlashInfer** | Attention/GEMM/MoE kernel library; DSV4 sparse MLA with TOPK=192 dispatch. | [03-kernels-attention](03-kernels-attention.md) |
| **FLASHINFER_MLA_SPARSE_DSV4** | FlashInfer DSV4 sparse-MLA backend; overlay fallback on the same 584 B page. Measured parity with `B12X_MLA_SPARSE`. | [03-kernels-attention](03-kernels-attention.md) |
| **fp8_ds_mla** | Stock FP8 DSV4 KV-cache dtype (576 B page). | [04-quantization-kv](04-quantization-kv.md) |
| **FULL / PIECEWISE** | CUDA graph capture modes (whole graph vs breakable pieces). Matched-main runs `FULL_AND_PIECEWISE`; the rc2 overlay is PIECEWISE-only. | [05-performance](05-performance.md), [07-gotchas](07-gotchas.md) |
| **GDS** | GPUDirect Storage — LMCache's NVMe offload backend (`configs/lmcache.gds.yaml`). | [04-quantization-kv](04-quantization-kv.md), [08-upstream](08-upstream.md) |
| **InstantTensor** | Model loader used on matched-main for fast cold start (hybrid lazy safetensors for the DSpark draft). | [06-deployment](06-deployment.md) |
| **k=5** | DSpark speculative tokens per step — locked by `dspark_block_size=5`; must be a multiple of `n_predict=5`. | [02-model](02-model.md) |
| **LMCache** | KV-cache offload library. Lacks `SupportsHMA`, so DSV4 sparse-MLA + offload fails. | [04-quantization-kv](04-quantization-kv.md), [08-upstream](08-upstream.md) |
| **mHC** | **Multi-head connection** (DeepSeek-V4 layer). Its `mhc_pre_broadcast_tilelang` GEMM is the unguarded DeepGEMM call that aborts on SM12x. | [03-kernels-attention](03-kernels-attention.md), [08-upstream](08-upstream.md) |
| **MLA** | Multi-head latent attention — K/V compressed to a latent vector; the sparse indexer scores KV blocks. | [02-model](02-model.md) |
| **MoE** | Mixture of experts — 256 routed experts in MXFP4 + FP8 shared experts. | [02-model](02-model.md) |
| **MTP** | Multi-token prediction — the DSpark draft head on the same checkpoint. | [02-model](02-model.md) |
| **MXFP4** | Microscaling FP4: FP4 weights with block scales (the routed-expert format; b12x `B12X_MXFP4_MXFP8`). | [04-quantization-kv](04-quantization-kv.md) |
| **NVFP4** | NVIDIA FP4 format. NVFP4 **weights** are a dead end here; the real NVFP4 **KV** writer (anemll golden) is the only measured memory win. | [04-quantization-kv](04-quantization-kv.md), [09-golden-deepgemm](09-golden-deepgemm.md) |
| **nvfp4_ds_mla** | The 584 B DSV4 alias dtype this repo serves — **not** a real NVFP4 writer (same page as fp8). | [04-quantization-kv](04-quantization-kv.md) |
| **nv_dev** | DeepGEMM branch containing the SM12x kernels. Its `8b1392b` tip **regressed** SM12x pure-FP8 linear (aliased `fp8_gemm_nt` → fp4). | [08-upstream](08-upstream.md), [09-golden-deepgemm](09-golden-deepgemm.md) |
| **RoCE** | RDMA over Converged Ethernet — 200 Gb/s CX7 fabric (GID 3) linking spark1/spark2. | [01-hardware](01-hardware.md) |
| **SiTU** | DeepGEMM PR #396 (2026-08-11) — the nv_dev update that moved the pin past eugr's `a6b593d` freeze and landed `8b1392b`. | [08-upstream](08-upstream.md) |
| **SM12x / sm_121a** | Blackwell GB10: compute capability 12.1, SM family 120 (`TORCH_CUDA_ARCH_LIST=12.1a`). | [01-hardware](01-hardware.md) |
| **SWA** | Sliding window attention — `DeepseekV4SWACache`, also on the 584 B DSV4 page. | [02-model](02-model.md), [04-quantization-kv](04-quantization-kv.md) |
| **TP** | Tensor parallelism — TP=2 across the two nodes (128 of 256 experts per node). | [01-hardware](01-hardware.md), [06-deployment](06-deployment.md) |
| **TOPK** | Top-k dispatch entry. DSV4 needs **192** (`ceil(133/64)*64`); stock FlashInfer had only {128, 512, 1024} until #4380. | [03-kernels-attention](03-kernels-attention.md), [08-upstream](08-upstream.md) |
| **UMA** | Unified memory architecture — 128 GiB shared coherent CPU/GPU pool (~273 GB/s). | [01-hardware](01-hardware.md) |
| **util** | `GPU_MEMORY_UTILIZATION` — 0.8 is the safe live ceiling; 0.85 triggers earlyoom on spark2. | [01-hardware](01-hardware.md), [05-performance](05-performance.md) |

---

**[← Prev](09-golden-deepgemm.md) · [Glossary](glossary.md)**
