[← Index](00-index.md)

# Glossary

> **Scope:** Shared vocabulary across the corpus — one-line definitions with chapter links.

Shared vocabulary used across the corpus. Definitions are grounded in the
chapters cited; see those chapters for depth.

| Term | Definition | See |
|------|-----------|-----|
| **Abliterated** | Derivative checkpoint with refusal/safety behavior modified (e.g. `OBLITERATUS/Qwen3.8-27B-OBLITERATED`); original-card quality figures do not apply to it. | [06-deployment](06-deployment.md) |
| **b12x** | SM120/121-only CuTe DSL kernel library (linear FP8, MoE MXFP4, compressed MLA). Primary kernel path on Spark. | [03-kernels-attention](03-kernels-attention.md) |
| **B12X_MLA_SPARSE** | b12x compressed-MLA attention backend on the 584 B DSV4 page; live backend on matched-main (no stock vLLM enum — overlay-registered). | [03-kernels-attention](03-kernels-attention.md) |
| **C4A / C128A** | Sparse-attention branches for `compress_ratio` 4 and 128 — the 0731 checkpoint alternates them, so both run. C128A was the eidx-contiguity bug source. | [02-model](02-model.md), [03-kernels-attention](03-kernels-attention.md) |
| **CSA / HCA** | DeepSeek-V4 attention modes: **C**ompressed **S**parse **A**ttention (CSA, compress_ratio 4) and **H**eavily **C**ompressed **A**ttention (HCA, ratio 128) — the paper-level names for the C4A/C128A alternation; KV cached in blocks of lcm(4,128)=128 original tokens. | [02-model](02-model.md), [04-quantization-kv](04-quantization-kv.md) |
| **CAPEX / OPEX / TCO** | One-time purchase cost / running cost / total cost of ownership over a fixed horizon (`CAPEX + Y×(power + maintenance + cooling) − residual`). | [11-cost-decision](11-cost-decision.md) |
| **CUTLASS block-FP8** | SM90/SM100 TMA FP8 GEMM path; does **not** run on SM12x (falls back to PyTorch/Triton/TileLang). | [03-kernels-attention](03-kernels-attention.md), [09-golden-deepgemm](09-golden-deepgemm.md) |
| **DSA** | DeepSeek Sparse Attention (V3.2): lightning indexer (score Σ w·ReLU(q·k)) + top-k token selection, O(L²) → O(L·k). V4's CSA = DSA + sequence compression — the efficiency baseline for the 27%/10% (Pro) and 10%/7% (Flash) FLOPs/KV claims at 1M ctx. | [04-quantization-kv](04-quantization-kv.md) |
| **DeepGEMM** | DeepSeek's FP8/FP4/BF16 GEMM library. SM12x kernels live on the **nv_dev** branch; the pure-FP8 1d1d path regressed at `8b1392b`. | [09-golden-deepgemm](09-golden-deepgemm.md) |
| **DFlash / DFlash2** | Speculative-decoding draft family (DeepSpec/DeepSeek; pooled-Laguna variants): standalone draft model line vs MTP/DSpark. Acceptance transfers only when trained on the target's base weights (e.g. RedHatAI DFlash 11.8% vs 41.6% on 0731); engine support varies (SGLang DFlash2, llama.cpp, vLLM forks). | [05-performance](05-performance.md), [06-deployment](06-deployment.md) |
| **DG_JIT_USE_NVRTC** | DeepGEMM JIT flag (NVRTC driver-JIT vs ptxas). NVRTC was tried because ptxas cannot assemble tcgen05 for `sm_121a` on CUDA 13.x, but the driver rejects its cubins — flag stays **0** on the live pin. | [09-golden-deepgemm](09-golden-deepgemm.md) |
| **DSpark** | DeepSeek-V4's native speculative-decoding method (`method=dspark`): k=5 locked, MTP draft on the same checkpoint. | [02-model](02-model.md) |
| **DSV4** | DeepSeek-V4. Also used for the **584 B DSV4 envelope** — the packed per-layer-token MLA page (448 B NoPE + 128 B RoPE + 8 B scale). | [02-model](02-model.md), [04-quantization-kv](04-quantization-kv.md) |
| **eidx** | "Extra sparse indices": the per-token sparse-attention index tensor. Must be contiguous for the SM120 kernel ("eidx must be contiguous" boot crash). | [03-kernels-attention](03-kernels-attention.md), [08-upstream](08-upstream.md) |
| **EAGLE-3** | Speculative-decoding draft with a built-in drafting head (no separate draft model) — used for GPT-OSS-120B (NVIDIA playbooks) and MiniMax-M3 k=2 (fp8-KV) recipes on GB10. | [01-hardware](01-hardware.md), [06-deployment](06-deployment.md) |
| **E8M0 / UE8M0** | Unsigned 8-bit-exponent scale format. SM12x upcasts it to fp32 (native 128×128 UE8M0 unsupported). | [04-quantization-kv](04-quantization-kv.md), [08-upstream](08-upstream.md) |
| **EXL3** | ExLlama-family ultra-low-bit quantization. The DeepSeek one-Spark recipe uses EXL3 **3.0 bpw** with non-uniform bit allocation (Trellis) on surviving tensors — not uniform round-to-nearest, and 3.0 bpw ≠ Q3 GGUF. | [06-deployment](06-deployment.md), [05-performance](05-performance.md) |
| **FlashInfer** | Attention/GEMM/MoE kernel library; DSV4 sparse MLA with TOPK=192 dispatch. | [03-kernels-attention](03-kernels-attention.md) |
| **FLASHINFER_MLA_SPARSE_DSV4** | FlashInfer DSV4 sparse-MLA backend; overlay fallback on the same 584 B page. Measured parity with `B12X_MLA_SPARSE`. | [03-kernels-attention](03-kernels-attention.md) |
| **fp8_ds_mla** | Stock FP8 DSV4 KV-cache dtype (576 B page). | [04-quantization-kv](04-quantization-kv.md) |
| **FULL / PIECEWISE** | CUDA graph capture modes (whole graph vs breakable pieces). Matched-main runs `FULL_AND_PIECEWISE`; the rc2 overlay is PIECEWISE-only. | [05-performance](05-performance.md), [07-gotchas](07-gotchas.md) |
| **GB10 OEM systems** | NVIDIA-certified GB10 workstations beyond the Founders Edition: Acer Veriton GN100, ASUS Ascent GX10, Dell Pro Max with GB10, GIGABYTE AI TOP ATOM, HP ZGX Nano, Lenovo ThinkStation PGX, MSI EdgeXpert — same SoC/UMA/CX-7, differing cooling/storage/adapter/support. | [01-hardware](01-hardware.md), [11-cost-decision](11-cost-decision.md) |
| **GDS** | GPUDirect Storage — LMCache's NVMe offload backend (`configs/lmcache.gds.yaml`). | [04-quantization-kv](04-quantization-kv.md), [08-upstream](08-upstream.md) |
| **GPT-5.6 Sol** | OpenAI hosted API model `gpt-5.6-sol` (alias `gpt-5.6`); "Sol max" = `reasoning.effort=max`, not a separate model. Context 1,050,000 tokens, max output 128,000. | [11-cost-decision](11-cost-decision.md) |
| **InstantTensor** | Model loader used on matched-main for fast cold start (hybrid lazy safetensors for the DSpark draft). | [06-deployment](06-deployment.md) |
| **k=5** | DSpark speculative tokens per step — locked by `dspark_block_size=5`; must be a multiple of `n_predict=5`. | [02-model](02-model.md) |
| **LMCache** | KV-cache offload library. Lacks `SupportsHMA`, so DSV4 sparse-MLA + offload fails. | [04-quantization-kv](04-quantization-kv.md), [08-upstream](08-upstream.md) |
| **MCDMA** | Community experimental prototype mapping Metal unified memory ↔ CUDA memory over USB-C (~939 MB/s single link, ~24 µs RTT claimed); no official NVIDIA documentation or public reproduction. | [01-hardware](01-hardware.md) |
| **mHC** | **Multi-head connection** (DeepSeek-V4 layer). Its `mhc_pre_broadcast_tilelang` GEMM is the unguarded DeepGEMM call that aborts on SM12x. | [03-kernels-attention](03-kernels-attention.md), [08-upstream](08-upstream.md) |
| **MLA** | Multi-head latent attention — K/V compressed to a latent vector; the sparse indexer scores KV blocks. | [02-model](02-model.md) |
| **MoE** | Mixture of experts — 256 routed experts in MXFP4 + FP8 shared experts. | [02-model](02-model.md) |
| **MTP** | Multi-token prediction — the DSpark draft head on the same checkpoint. | [02-model](02-model.md) |
| **MXFP4** | Microscaling FP4: FP4 weights with block scales (the routed-expert format; b12x `B12X_MXFP4_MXFP8`). | [04-quantization-kv](04-quantization-kv.md) |
| **NemoClaw** | NVIDIA's local-agent stack on DGX Spark: NemoClaw orchestrator + OpenShell sandbox/gateway + OpenClaw (multi-channel agent) + a Nemotron model via Ollama/NIM; one-command installer, WebUI on `127.0.0.1:18789`. | [01-hardware](01-hardware.md), [06-deployment](06-deployment.md) |
| **E2M1 / UE4M3 (NVFP4 sub-byte)** | The NVFP4 operand format: E2M1 4-bit floats (micro-block 16, one UE4M3 scale per 16 K-elements). The only SM12x NVFP4 blockscaled MMA is `m16n8k64` with `.scale_vec::4X` (see the Colfax tutorial). | [03-kernels-attention](03-kernels-attention.md) |
| **mma.sync vs tcgen05** | SM12x uses warp-level `mma.sync` (SM8x-style, register fragments) — **not** SM10x `tcgen05.mma`/TMEM. SM10x kernels don't run on SM12x; SM8x kernels do. | [03-kernels-attention](03-kernels-attention.md) |
| **NVFP4** | NVIDIA FP4 format. NVFP4 **weights** are a dead end here; the real NVFP4 **KV** writer (anemll golden) is the only measured memory win. | [04-quantization-kv](04-quantization-kv.md), [09-golden-deepgemm](09-golden-deepgemm.md) |
| **nvfp4_ds_mla** | The 584 B DSV4 alias dtype this repo serves — **not** a real NVFP4 writer (same page as fp8). | [04-quantization-kv](04-quantization-kv.md) |
| **nv_dev** | DeepGEMM branch containing the SM12x kernels. Its `8b1392b` tip **regressed** SM12x pure-FP8 linear (aliased `fp8_gemm_nt` → fp4). | [08-upstream](08-upstream.md), [09-golden-deepgemm](09-golden-deepgemm.md) |
| **packed-at-store** | The sparse-indexer K storage layout: K-then-scale packed per 64-token page (not interleaved). Gather of this layout is numerically wrong; only the b12x paged kernel reads it correctly. | [03-kernels-attention](03-kernels-attention.md) |
| **REAP** | Expert pruning applied to DeepSeek weights: the one-Spark EXL3 checkpoint keeps **216 of 256 experts** (REAP-K216), stored at non-uniform bit width via Trellis. | [06-deployment](06-deployment.md), [05-performance](05-performance.md) |
| **RoCE** | RDMA over Converged Ethernet — 200 Gb/s CX7 fabric (GID 3) linking spark1/spark2. | [01-hardware](01-hardware.md) |
| **SiTU** | DeepGEMM PR #396 (2026-08-11) — the nv_dev update that moved the pin past eugr's `a6b593d` freeze and landed `8b1392b`. | [08-upstream](08-upstream.md) |
| **SM12x / sm_121a** | Blackwell GB10: compute capability 12.1, SM family 120 (`TORCH_CUDA_ARCH_LIST=12.1a`). | [01-hardware](01-hardware.md) |
| **SparkInfer** | Serving engine (recipe fork + patches) for the DeepSeek one-Spark EXL3 path; OpenAI-format endpoint but different quant/draft/KV internals than stock vLLM configs. | [06-deployment](06-deployment.md), [10-operations-agents](10-operations-agents.md) |
| **SWA** | Sliding window attention — `DeepseekV4SWACache`, also on the 584 B DSV4 page. | [02-model](02-model.md), [04-quantization-kv](04-quantization-kv.md) |
| **TP** | Tensor parallelism — TP=2 across the two nodes (128 of 256 experts per node). | [01-hardware](01-hardware.md), [06-deployment](06-deployment.md) |
| **TOPK** | Top-k dispatch entry. DSV4 needs **192** (`ceil(133/64)*64`); stock FlashInfer had only {128, 512, 1024} until #4380. | [03-kernels-attention](03-kernels-attention.md), [08-upstream](08-upstream.md) |
| **TTFT** | Time to first token — the prefill latency component of decode latency. | [05-performance](05-performance.md), [11-cost-decision](11-cost-decision.md) |
| **UMA** | Unified memory architecture — 128 GiB shared coherent CPU/GPU pool (~273 GB/s). | [01-hardware](01-hardware.md) |
| **util** | `GPU_MEMORY_UTILIZATION` — 0.8 is the safe live ceiling; 0.85 triggers earlyoom on spark2. | [01-hardware](01-hardware.md), [05-performance](05-performance.md) |
| **zswap / swappiness** | Linux memory-compression layer and VM swap aggressiveness; `zstd/zsmalloc` with `vm.swappiness=10` is the measured-safe Spark config (100 + disk swap caused decode stalls). | [01-hardware](01-hardware.md) |

---

**[← Prev](13-qwenseek.md) · [Glossary](glossary.md)**
