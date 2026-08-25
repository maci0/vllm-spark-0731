# Comprehensive Technical & Codebase Audit: Architecture Specifications, NVIDIA GB10 Hardware Execution, Upstream PR Lineage, and Empirical Benchmarks for vllm-spark-0731

## 1. Executive Summary & Audit Verdict

This document presents a rigorous, publication-grade paper and codebase audit of **vllm-spark-0731** (`DeepSeek-V4-Flash-0731` served on a cluster of 2× NVIDIA DGX Spark workstations equipped with GB10 Grace Blackwell Superchips) [1, 2, 3].

### Audit Verdict
The serving implementation achieves verified numerical correctness, robust multi-stream scaling, and deterministic low-latency execution under the **matched-main** software stack (`vllm-spark-0731:main-b12x`, commit `e25c586b9`, CUDA 13.3.1, PyTorch 2.14 `12.1a`, `B12X_VERSION=1.2.6`, `CUTLASS_DSL_VERSION=4.7.0`) [4, 10, 22].

Key findings verified across the hardware, kernel, and model layers include:
1. **Mathematical Soundness**: The Dynamic Sparse Attention (DSA) Lightning Indexer strictly preserves the non-linear $\text{ReLU}$ activation during per-head scoring, preventing catastrophic cache eviction and text drift [5, 21].
2. **Hardware Constraints & Workarounds**: While the NVIDIA GB10 Grace Blackwell Superchip delivers dense FP4 tensor compute up to 1 PFLOP and 128 GB coherent LPDDR5x Unified Memory Architecture (UMA) at ~273 GB/s, it lacks datacenter-grade asynchronous Tensor Memory Accelerator (TMA) hardware [1, 2, 4]. As a consequence, CUTLASS FP8 block-scaled GEMMs and unpatched DeepGEMM pure-FP8 kernels fail or corrupt output, necessitating exact kernel routing to `B12xFp8BlockScaledMMKernel` and the grouped `torch.bmm` output projection workaround (`try_b12x_wo_proj`) [6, 11, 15, 21].
3. **Speculative Decoding Alignment**: FlashInfer Sparse Attention requires exact TOPK=192 dispatch arithmetic ($\lceil (128+5)/64 \rceil \times 64 = 192$) for $k=5$ speculative decoding lookahead, unlocked upstream via FlashInfer PR #4380 [6, 20].
4. **Memory Stability**: The 128 GiB UMA boundary requires maintaining `GPU_MEMORY_UTILIZATION <= 0.80` to prevent triggering the `earlyoom` process killer (<8% `MemAvailable` cliff), supported by the unmasking of the 16 GiB `/swap.img` on worker node `spark2` and tuning `vm.swappiness=10` [4, 9].
5. **Quality & Throughput Validation**: The cluster passes the greedy France prompt quality gate ($\log p(\text{' Paris'}) \in [-0.27, -0.24]$, $n_{\text{tie}}=1$) and achieves scaling from 25.8 tok/s at concurrency $c=1$ up to 172.0 tok/s at $c=32$ [8].

---

## 2. System Architecture & Model Specification Audit

### 2.1 Multi-Head Latent Attention (MLA) Page Geometry
The `DeepSeek-V4-Flash-0731` architecture compresses Key/Value activations into a low-dimensional latent subspace before storing them in paged memory [3, 5]:
- **Latent Dimension**: $d_c = 512$ total latent key dimensions, decomposed into $448\text{ (FP8 NoPE)} + 64\text{ (BF16 RoPE)} = 512\text{ dimensions}$ [5].
- **Per-Layer Token Envelope ($584\text{ B}$)**:
  - Non-Rotary Key/Value latent projections ($\text{NoPE}$): $448\text{ B}$ ($d_{\text{NoPE}} = 448$ in 8-bit FP8 precision: $448 \times 1\text{ B} = 448\text{ B}$) [5].
  - Decoupled Rotary Position Embedding Key (RoPE): $128\text{ B}$ ($d_{\text{RoPE}} = 64$ dimensions in 16-bit BF16 precision: $64 \times 2\text{ B} = 128\text{ B}$) [5].
  - Quantization Scale Factors: $8\text{ B}$ (two FP32/FP8 block scales) [5].
  $$\text{Envelope per Layer per Token} = 448\text{ B (NoPE)} + 128\text{ B (RoPE)} + 8\text{ B (Scale)} = 576\text{ B (Tensor Data)} + 8\text{ B (Scale)} = 584\text{ B}$$
- **Architectural Divergence from Generic Formats**:
  - Stock vLLM Generic MLA (`fp8_ds_mla`): $576\text{ B}$ ($512\text{ B}$ latent $+ 64\text{ B}$ RoPE) [5, 7].
  - GLM NVFP4: $432\text{ B}$ or $368\text{ B}$ with incompatible `scale_format=2` layout [5, 7].
  - Matched-main vLLM treats the $584\text{ B}$ layout as an FP8 page geometry alias rather than a dense NVFP4 writer [7].

### 2.2 Dynamic Sparse Attention (DSA) Indexer Scoring & Mathematical Proof
The Dynamic Sparse Attention (DSA) indexer (Lightning Indexer) computes dynamic routing scores to identify top-k KV cache blocks [5, 21]:
$$\text{score}[h, m, n] = (q[m, h] \cdot k[n]) \cdot \text{scale}[n]$$
$$\text{logits}[m, n] = \sum_{h=1}^{H} w[m, h] \cdot \text{ReLU}(\text{score}[h, m, n])$$
where $m$ is the query token index, $h$ indexes the indexer head ($H=8$), $n$ indexes historical context tokens, $w[m, h]$ is the dynamic head routing weight, and $\text{scale}[n]$ is the dequantization scale factor [5, 21].

#### Mathematical Proof: Why Linear Weighted-Q Contraction Drops ReLU and Degrades Accuracy
In standard linear attention optimizations, developers frequently perform an upfront contraction of query vectors across heads to conserve memory bandwidth:
$$q_{\text{eff}}[m] = \sum_{h=1}^{H} w[m, h] \, q[m, h]$$
If the indexer scoring function were strictly linear, the summation over attention heads would commute with the dot product:
$$\sum_{h=1}^{H} w[m, h] \left( (q[m, h] \cdot k[n]) \cdot \text{scale}[n] \right) = \left( \left( \sum_{h=1}^{H} w[m, h] \, q[m, h] \right) \cdot k[n] \right) \cdot \text{scale}[n] = (q_{\text{eff}}[m] \cdot k[n]) \cdot \text{scale}[n]$$
However, because $\text{ReLU}(x) = \max(0, x)$ is a non-linear activation operator:
$$\sum_{h=1}^{H} w[m, h] \cdot \text{ReLU}(\text{score}[h, m, n]) \neq \text{ReLU}\left( \sum_{h=1}^{H} w[m, h] \cdot \text{score}[h, m, n] \right)$$
Dropping the per-head $\text{ReLU}$ evaluation eliminates the negative-alignment threshold filter [5, 9]. As a result, irrelevant or negatively correlated KV tokens receive artificial positive scores, corrupting block selection, causing acceptance rate collapse, and inducing immediate text degeneration during decode [5, 9, 21].

### 2.3 Aggregate Whole-Model KV Footprint & Memory Capacity Math
Across the complete transformer architecture:
- **Total Layers**: $L = 61$ transformer layers [3, 5].
- **Whole-Model Token Footprint**:
  $$\text{Footprint per Token} = 61 \times 584\text{ B} = 35,624\text{ B/token} \approx 35.624\text{ KB/token}$$
- **Amortized Baseline MLA Comparison**:
  - Stock vLLM Generic MLA (`fp8_ds_mla`): 576 B unpadded base latent representation; amortizes to 11,317 B/token (11,317 / 61 ≈ 185.5 B/layer/token) across 61 layers under hierarchical compressed indexer sharing [5, 7].
- **KV Capacity Math at `GPU_MEMORY_UTILIZATION=0.80`**:
  - Total Unified Physical Memory across 2 nodes: $2 \times 128\text{ GB} = 256\text{ GB}$ (usable CUDA pool: ~121.7 GiB per node) [4, 5].
  - Model Weights with Tensor Parallelism (TP=2): $155.43\text{ GiB} / 2 \approx 77.71\text{ GiB}$ per node [5].
  - Usable budget at 0.80 utilization (~97.36 GiB allocatable ceiling per node minus ~77.7 GiB weights and runtime activation/graph buffers) yields an active pool of **97,737 tokens** (~97.7k tokens) [5, 8].
- **Comparison to anemll Golden Image NVFP4 Writer**:
  - The anemll image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`) implements a true NVFP4 KV cache writer requiring only **7,650 B/token** across all 61 layers [7, 8].
  - This allows a pool of **2,047,170 tokens** (~2.05M tokens at 0.82 utilization) [8].

### 2.4 DSpark Speculative Decoding Configuration Locked in `assert_0731.py`
In `patches/assert_0731.py` (lines 10–54), strict programmatic assertions guard against configuration drift [3]:
- `PIN_ID`: `"deepseek-ai/DeepSeek-V4-Flash-0731"` [3]
- `ARCH`: `"DeepseekV4ForCausalLM"` [3]
- `MODEL_TYPE`: `"deepseek_v4"` [3]
- `dspark_block_size`: `5` (enforcing $k=5$ speculative decoding lookahead) [3]
- `num_nextn_predict_layers`: `1` (1-layer Multi-Token Prediction draft head) [3]
- `compress_ratios`: `[0, 0, 4, 128, ...]` (hierarchical compression ratios alternating C4A and C128A branches) [3]
Any discrepancy triggers an immediate `SystemExit` at process startup [3].

---

## 3. NVIDIA GB10 Grace Blackwell Superchip Datasheet & Hardware Execution Audit

### 3.1 Hardware Specifications Summary

| Subsystem / Feature | Official Hardware Specification | Architectural Grounding / Verification Source |
|---------------------|--------------------------------|----------------------------------------------|
| **SoC Architecture** | NVIDIA Grace Blackwell Superchip | High-bandwidth NVLink-C2C interconnect unifying Grace CPU and Blackwell GPU [1, 2]. |
| **GPU Architecture** | Blackwell SM12x (`sm_121a` / Compute Capability 12.1) | Low-power workstation class Blackwell family 120 architecture [1, 4]. |
| **Streaming Multiprocessors** | **48 SMs** | Equivalent to **6,144 CUDA cores** [1]. |
| **Tensor Cores** | 5th Generation Tensor Cores | Native hardware execution for dense NVFP4, MXFP4, FP8 (E4M3/E5M2), BF16, TF32, and FP16 arithmetic [1]. |
| **AI Performance** | Up to **1 PFLOP (1,000 TOPS)** | Peak dense FP4 tensor compute ceiling [1, 2]. |
| **Ray Tracing Cores** | 4th Generation RT Cores | Hardware-accelerated BVH traversal [1]. |
| **CPU Subsystem** | 20-core Armv9 CPU | **10× Cortex-X925** high-performance cores + **10× Cortex-A725** high-efficiency cores (Armv9.2-A with SVE2, BF16, and i8mm) [1]. |
| **Unified Memory (UMA)** | **128 GB coherent LPDDR5x** | 256-bit memory bus interface @ 4266 MHz, providing **~273 GB/s peak bandwidth** [1]. |
| **CUDA Allocatable Pool** | **~121.7 GiB usable memory** | Dynamically shared across CPU, GPU, operating system kernel, page tables, and buffer caches [1, 4]. |
| **Network Interface** | Integrated **NVIDIA ConnectX-7 NIC** | 100/200 GbE QSFP port configured for RoCE v2 on network device `enp1s0f1np1` with NCCL IB GID index 3 (`NCCL_IB_GID_INDEX=3`) [1, 4, 22]. |

### 3.2 DeepGEMM `nv_dev` (`8b1392b`) Compilation Status & TMA Assertions
- **Main Branch Incompatibility**: DeepGEMM upstream `main` (`e21c821`) strictly targets Hopper and Blackwell datacenter GPUs, asserting `arch_major == 9 || arch_major == 10` at `csrc/apis/attention.hpp:122` [10, 11]. This crashes on GB10 (`arch_major == 12`) [10, 11].
- **`nv_dev` Branch Status**: DeepGEMM `nv_dev` (`8b1392b`) enables SM120 compilation, causing `is_deep_gemm_supported()` to evaluate to `True` [10, 11].
- **Pure-FP8 Dispatch Bug on SM12x**: In `nv_dev 8b1392b`, `csrc/apis/gemm.hpp:851` erroneously aliases pure FP8 GEMMs to the FP4 dispatch entrypoint:
  ```cpp
  m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt");
  ```
  On `arch_major == 12`, this executes `sm120_fp8_fp4_gemm_1d1d` over FP8 weight tensors, resulting in numerical corruption (yielding `' Septy Septy...'` gibberish at 4.4 tok/s) [11].
- **Hopper/Blackwell Page Layout Mismatch**: DeepGEMM paged MQA metadata helpers assert 32 or 64 states per page, whereas DSV4 compress-128 pages contain 2 states [11, 18].

### 3.3 DeepGEMM PR #403 (`deepgemm-pr-403.diff`)
Patch `patches/upstream/deepgemm-pr-403.diff` modifies `csrc/apis/layout.hpp` (lines 49, 57, 112) to include SM120/SM121 architectures in Scale-Factor (SF) transformation routines [10, 12]:
```cpp
// (FP32, x, gran_k) on SM100/SM120: transform to (INT, 1, gran_k), TMA-aligned and MN-major
if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and (arch_major == 10 or arch_major == 12)) {
    DG_HOST_ASSERT(not disable_ue8m0_cast);
    ...
```
This guarantees valid TMA-aligned packed UE8M0 scale tensor generation on SM120/SM121 systems [10, 12].

### 3.4 CUTLASS FP8 Block-Scaled GEMM Exclusion (PR #53055) & Routing to `B12xFp8BlockScaledMMKernel`
- **Hardware Barrier**: CUTLASS block-scaled FP8 depends on asynchronous TMA descriptor hardware present on SM90 and SM100, which is absent on SM12x [4, 6, 15].
- **PR #53055 (`pr-53055.diff`)**: Amends `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py` so that `CutlassFP8ScaledMMLinearKernel.is_supported()` verifies compute capability with `ops.cutlass_scaled_mm_supports_fp8(compute_capability)`, rejecting SM12x (`family 120`) [15].
- **Live Routing**: Linear layers route to `--linear-backend b12x`, executing `B12xFp8BlockScaledMMKernel` with a cosine similarity of 0.9999986 against PyTorch reference implementations [6, 7].

### 3.5 Attention Output Projection Workaround (`try_b12x_wo_proj`)
Located in `patches/files/sm12x_b12x_kernels.py` (lines 826–896) [21]:
- **Failure Mode**: Executing `wo_proj.run()` under native MXFP8 enters infinite generation loops on greedy prompts [6, 9, 21].
- **Execution Flow**:
  1. Calls `fused_inv_rope_fp8_quant` to generate `o_fp8` and non-TMA `o_scale` [21].
  2. Executes `_dequant_grouped_fp8` to dequantize activations without TMA dependencies [21].
  3. Allocates persistent workspace buffers (`a_ws`, `z_ws`, `flat_ws`) [21].
  4. Executes grouped batch matrix multiplication via `torch.bmm(a_ws, w_bmm, out=z_ws)` using cached `wo_a` weights [21].
  5. Evaluates output through `wo_b(flat_ws)` [21].

---

## 4. FlashInfer Sparse Attention & Upstream PR Matrix

### 4.1 TOPK=192 Dispatch Arithmetic
In DSpark speculative decoding ($k=5$), the sparse attention mechanism must span both the Sliding Window Attention context ($W = 128$) and the $k = 5$ speculative draft tokens, requiring an active span of $128 + 5 = 133$ tokens [5, 6, 20]:
$$\text{TOPK} = \left\lceil \frac{128 + 5}{64} \right\rceil \times 64 = \left\lceil \frac{133}{64} \right\rceil \times 64 = 3 \times 64 = 192$$
- **FlashInfer PR #4380**: Upstream FlashInfer supported only TOPK $\in \{128, 512, 1024, 2048\}$. PR #4380 (merged 2026-08-08) added explicit kernel instantiations for TOPK 192 and 256, eliminating the need to pad to 512 and delivering a 13–16% throughput speedup [6, 10, 20].

### 4.2 Comprehensive Audit of Upstream Patches (`patches/upstream/`)

| Patch File | Target Upstream PR | Affected Subsystems / Files | Exact Functional Mechanism |
|------------|-------------------|-----------------------------|----------------------------|
| **`deepgemm-pr-403.diff`** | [deepseek-ai/DeepGEMM#403](https://github.com/deepseek-ai/DeepGEMM/pull/403) [12] | `csrc/apis/layout.hpp` | Extends `transform_sf_into_required_layout` condition `arch_major == 10` to `(arch_major == 10 or arch_major == 12)`, enabling SM120/SM121 Scale-Factor transformations [10, 12]. |
| **`pr-47988.diff`** | [vllm-project/vllm#47988](https://github.com/vllm-project/vllm/pull/47988) [13] | `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`<br>`vllm/model_executor/layers/quantization/utils/fp8_utils.py` | Removes platform gating to unconditionally upcast E8M0 scale tensors to FP32 in `w8a8_triton_block_scaled_mm` (resolving `KeyError: 'float8_e8m0fnu'`); enforces weight dimension 128-divisibility in CUTLASS `can_implement` for SM12x [10, 13]. |
| **`pr-52499.diff`** | [vllm-project/vllm#52499](https://github.com/vllm-project/vllm/pull/52499) [14] | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | Reshapes decode query and metadata tensors to 4D `[num_decodes, next_n, ...]` during speculative decoding; routes prefill chunks with $\le 64$ tokens to decode kernels to satisfy SM120 prefill assertion `num_tokens > 64` [10, 14]. |
| **`pr-53055.diff`** | [vllm-project/vllm#53055](https://github.com/vllm-project/vllm/pull/53055) [15] | `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`<br>`vllm/model_executor/kernels/mhc/tilelang.py`<br>`tests/kernels/test_mhc_kernels.py` | Guards `tf32_hc_prenorm_gemm` invocation with `is_deep_gemm_supported()` in `mhc_pre_tilelang`; adds compute capability check to `CutlassFP8ScaledMMLinearKernel.is_supported()` rejecting SM12x [10, 15]. |
| **`pr-53425.diff`** | [vllm-project/vllm#53425](https://github.com/vllm-project/vllm/pull/53425) [16] | `vllm/models/deepseek_v4/sparse_mla.py`<br>`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`<br>`vllm/v1/attention/backends/mla/indexer.py`<br>`tests/v1/attention/test_dsv4_kernel_block_size.py` | Implements `dsv4_supported_kernel_block_sizes()` returning `[64]` when `is_device_capability_family(120)` is True, matching SM120 FlashInfer 64-token page compilation while keeping manager block size at 256 [10, 16]. |
| **`pr-53521.diff`** | [vllm-project/vllm#53521](https://github.com/vllm-project/vllm/pull/53521) [17] | `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`<br>`tests/models/test_deepseek_v4_fp8_einsum_recipe.py` | Forces `compute_fp8_einsum_recipe()` on SM12x (`cap.major == 12`) to return Hopper FP32 recipe `(1, 128, 128)` and `tma_aligned_scales=False`, avoiding SM100 packed INT32 corruption [10, 17]. |
| **`pr-53522.diff`** | [vllm-project/vllm#53522](https://github.com/vllm-project/vllm/pull/53522) [18] | `vllm/v1/attention/backends/mla/indexer.py`<br>`tests/v1/attention/test_indexer_paged_mqa_metadata.py` | Introduces `_should_build_paged_mqa_logits_metadata(num_states)` requiring `is_deep_gemm_supported()` and `num_states in (32, 64)`, bypassing DeepGEMM scheduling on 2-state DSV4 compress-128 pages [10, 18]. |
| **`pr-53574.diff`** | [vllm-project/vllm#53574](https://github.com/vllm-project/vllm/pull/53574) [19] | `vllm/models/deepseek_v4/sparse_mla.py` | Applies `.contiguous()` to `global_decode.view(num_decode_tokens, 1, -1)` in `_build_c128a_metadata()`, resolving C++ `eidx.IsContiguous()` assertion failure in `sparse_mla_sm120.cu` during speculative decode warmup [10, 19]. |

---

## 5. Memory Limits, Quality Gates & Empirical Benchmarks

### 5.1 128 GiB UMA Bounds & Operating System Configuration
- **Unified Memory Architecture**: The 128 GB physical LPDDR5x RAM is dynamically shared between CPU, GPU, OS kernel, page tables, and filesystem buffers [1, 4].
- **Safe Utilization Threshold**: `GPU_MEMORY_UTILIZATION=0.80` is the verified operational limit [4, 9, 22]. Increasing utilization to 0.85 crosses an allocation threshold where Linux `earlyoom` issues an uncatchable `SIGTERM` when `MemAvailable < 8%` (~10 GiB available memory remaining) [4, 9].
- **Swap Recovery on Worker Node `spark2`**:
  - A 16 GiB `/swap.img` file is defined in `/etc/fstab` across both nodes [4].
  - On worker node `spark2`, the systemd unit `swap.img.swap` was previously masked (`/etc/systemd/system/swap.img.swap -> /dev/null`), resulting in 0 MiB total swap at boot [4, 9].
  - Unmasking on 2026-08-24 restored swap activation (`Activating swap swap.img.swap` → `Adding 16777212k swap` → `Activated`) [4].
- **Swappiness & zswap Tuning**:
  - `vm.swappiness=10` is persisted in `/etc/sysctl.d/99-dgx-spark-swap.conf` (reduced from default 100 to eliminate token decode jitter caused by aggressive disk paging) [4, 9].
  - Configured with `zswap` using `zstd` compression and `zsmalloc` allocator (`max_pool_percent=5`) [4, 9].

### 5.2 Greedy France Validation Quality Gate
Validation script `scripts/06-validate.sh` executes the precision quality gate against `127.0.0.1:8000` [8]:
- **Prompt**: `"The capital of France is"` [8]
- **Parameters**: `max_tokens=32`, `temperature=0`, `logprobs=5` [8]
- **Expected Text Prefix**: `' Paris. The capital of Spain is Madrid...'` [8]
- **Logprob Verification**:
  $$\log p(\text{' Paris'}) \in [-0.27, -0.24] \quad (\text{measured: } -0.24) \quad [8]$$
- **Tie Count**:
  $$n_{\text{tie}} = 1 \quad (\text{distinct probability maximum, zero ambiguity}) \quad [8]$$
Any divergence (e.g. repetitive generation `' Septy Septy...'`) signals dequantization or scale layout corruption [8, 9, 11].

### 5.3 Concurrency Scaling Benchmarks
Empirically measured via `scripts/bench-concurrency.py` on the 2× GB10 cluster (TP=2, DSpark $k=5$, greedy decode, 128 output tokens per stream) [8]:

| Concurrency Level | Aggregate Throughput | Per-Stream Throughput | Operational Notes |
|-------------------|----------------------|-----------------------|-------------------|
| **c1 (Single Stream)** | **~25.8 tok/s** | 25.8 tok/s | France prompt greedy decode; latency-bound on kernel dispatch [8]. |
| **c8 (8 Streams)** | **~95.0 tok/s** | 11.9 tok/s | Multi-stream batching; utilizes multi-stream CUDA graph capture [8]. |
| **c16 (16 Streams)** | **~116.0 tok/s** | 7.25 tok/s | Intermediate concurrency saturation [8]. |
| **c32 (32 Streams)** | **~172.0 tok/s** | 5.38 tok/s | Full saturation at `MAX_NUM_SEQS=32` (verified stable) [8]. |

*(Note: Setting `MAX_NUM_SEQS=48` exceeds boot memory ceilings and causes startup hangs [8, 9]).*

---

## 6. Codebase vs Documentation Gap Analysis & Resolved Items

| Subsystem | Documented Expectation | Codebase / Hardware Reality | Resolution / Verification Status |
|-----------|------------------------|-----------------------------|----------------------------------|
| **DeepGEMM Support** | Generic Blackwell support implies SM12x is fully functional [10]. | Upstream `main` crashes at `attention.hpp:122` (asserts arch 9/10) [10, 11]. `nv_dev` `8b1392b` contains FP8-to-FP4 aliasing bug at `gemm.hpp:851` [11]. | PR #403 backported (`deepgemm-pr-403.diff`) [12]. Pure-FP8 linear GEMMs routed to `B12xFp8BlockScaledMMKernel` [6, 15]. |
| **KV Cache Compression** | Full NVFP4 compression across all layers [5, 7]. | Matched-main uses 584 B FP8 page geometry alias (35,624 B/token) [7]. True NVFP4 writer (7,650 B/token) exists only in anemll image [7, 8]. | Accurately documented in memory budgets and capacity math (97.7k token active pool) [5, 8]. |
| **Sliding Window Dispatch** | Speculative decoding evaluates within standard 128 SWA window [5, 6]. | Speculative $k=5$ expands span to 133 tokens, requiring TOPK=192 dispatch ($\lceil 133/64 \rceil \times 64 = 192$) [6, 20]. | Unlocked upstream via FlashInfer PR #4380; backported in runtime overlays [6, 20]. |
| **CUTLASS Block Scaling** | CUTLASS handles all FP8 scaled linear operations [10, 15]. | CUTLASS block-scaled FP8 relies on TMA hardware absent on SM12x [4, 6, 15]. | PR #53055 backported to reject SM12x in CUTLASS and route to b12x [15]. |
| **Attention Output Projection** | Standard `wo_proj.run()` with MXFP8 quantization [6, 21]. | Direct `wo_proj.run()` enters infinite generation loops on greedy prompts [6, 9, 21]. | Resolved via `try_b12x_wo_proj` grouped `torch.bmm` fallback in `sm12x_b12x_kernels.py` [21]. |
| **Swap Memory on `spark2`** | 16 GiB swap available on all cluster nodes [4]. | `swap.img.swap` systemd unit was masked on `spark2` (0 MiB swap) [4, 9]. | Unmasked on 2026-08-24, restoring full 16 GiB active swap pool [4]. |

---

## 7. Reproduction Risks & Recommendations

1. **Do Not Increase `GPU_MEMORY_UTILIZATION` Above 0.80**: Setting utilization to 0.85 triggers earlyoom SIGTERM kills due to UMA memory contention [4, 9].
2. **Do Not Enable Datacenter TMA Kernels on SM12x**: SM12x lacks physical TMA descriptors; attempting to bypass checks will result in illegal instruction crashes (Xid 13) or numerical corruption [4, 6, 11].
3. **Maintain Pinned Speculative Lookahead ($k=5$)**: Modifying $k$ requires re-evaluating the FlashInfer TOPK tile arithmetic and verifying alignment with C128A/C4A indexers [3, 6, 20].
4. **Preserve E8M0 Unconditional Upcast**: Ensure `pr-47988.diff` is maintained to prevent `KeyError: 'float8_e8m0fnu'` during Triton block-scaled matrix multiplications [10, 13].
5. **Ensure Tensor Contiguity in C128A Metadata**: Maintain `pr-53574.diff` to prevent C++ `eidx.IsContiguous()` assertion crashes in `sparse_mla_sm120.cu` [10, 19].

---

## 8. Sources

[1] NVIDIA Corporation. *Hardware Overview — DGX Spark User Guide*. NVIDIA Technical Documentation, 2025/2026. Available at: https://docs.nvidia.com/dgx/dgx-spark/hardware.html  
[2] NVIDIA Corporation. *NVIDIA DGX Spark Official Datasheet*. NVIDIA Asset Library, 2025/2026. Available at: https://dam-cdn.nvd.orangelogic.com/AssetLink/3lhuar5pc56pn7se4c7ahsskw20xw8h5.pdf  
[3] vLLM-Spark Project. *DeepSeek-V4-Flash-0731 Runtime Architecture & Speculative Decoding Assertion Guard*. Repository file: `patches/assert_0731.py`.  
[4] vLLM-Spark Project. *Hardware & Memory Constraints on DGX Spark (GB10, SM12x)*. Repository documentation: `docs/knowledge/01-hardware.md`.  
[5] vLLM-Spark Project. *DeepSeek-V4-Flash Model Architecture & MLA Specifications*. Repository documentation: `docs/knowledge/02-model.md`.  
[6] vLLM-Spark Project. *Kernel Stack & Attention Backends on SM12x*. Repository documentation: `docs/knowledge/03-kernels-attention.md`.  
[7] vLLM-Spark Project. *Quantization and KV Cache Geometry*. Repository documentation: `docs/knowledge/04-quantization-kv.md`.  
[8] vLLM-Spark Project. *Performance Benchmarks & Concurrency Scaling*. Repository documentation: `docs/knowledge/05-performance.md`.  
[9] vLLM-Spark Project. *Operational Gotchas and Failure Modes on SM12x*. Repository documentation: `docs/knowledge/07-gotchas.md`.  
[10] vLLM-Spark Project. *Upstream Gaps and PR Lineage Tracker*. Repository documentation: `docs/knowledge/08-upstream.md` and `docs/UPSTREAM.md`.  
[11] vLLM-Spark Project. *Golden DeepGEMM Analysis and nv_dev Native Dispatches*. Repository documentation: `docs/knowledge/09-golden-deepgemm.md`.  
[12] DeepSeek AI & vLLM Community. *DeepGEMM SM120/SM121 Scale-Factor Layout Patch*. Patch diff: `patches/upstream/deepgemm-pr-403.diff` (Upstream PR: https://github.com/deepseek-ai/DeepGEMM/pull/403).  
[13] vLLM Community. *Triton E8M0 Upcast & CUTLASS Divisibility Backport*. Patch diff: `patches/upstream/pr-47988.diff` (Upstream PR: https://github.com/vllm-project/vllm/pull/47988).  
[14] vLLM Community. *FlashInfer DSV4 Speculative Decode Shapes Backport*. Patch diff: `patches/upstream/pr-52499.diff` (Upstream PR: https://github.com/vllm-project/vllm/pull/52499).  
[15] vLLM Community. *CUTLASS FP8 SM12x Exclusion & TileLang mHC Guard Backport*. Patch diff: `patches/upstream/pr-53055.diff` (Upstream PR: https://github.com/vllm-project/vllm/pull/53055).  
[16] vLLM Community. *DSV4 Kernel Block Size 64 Configuration Backport*. Patch diff: `patches/upstream/pr-53425.diff` (Upstream PR: https://github.com/vllm-project/vllm/pull/53425).  
[17] vLLM Community. *Hopper FP32 Einsum Recipe on SM12x Backport*. Patch diff: `patches/upstream/pr-53521.diff` (Upstream PR: https://github.com/vllm-project/vllm/pull/53521).  
[18] vLLM Community. *Indexer Paged MQA Metadata DeepGEMM State Guard Backport*. Patch diff: `patches/upstream/pr-53522.diff` (Upstream PR: https://github.com/vllm-project/vllm/pull/53522).  
[19] vLLM Community. *FlashInfer C128A Metadata Contiguity Backport*. Patch diff: `patches/upstream/pr-53574.diff` (Upstream PR: https://github.com/vllm-project/vllm/pull/53574).  
[20] FlashInfer AI Team. *Add TOPK 192 and 256 Instantiations for DSV4 Sparse MLA*. Upstream Pull Request: https://github.com/flashinfer-ai/flashinfer/pull/4380 (Merged Aug 8, 2026).  
[21] vLLM-Spark Project. *SM12x b12x Kernel Implementations and Attention Workarounds*. Repository file: `patches/files/sm12x_b12x_kernels.py`.  
[22] vLLM-Spark Project. *DGX Spark Cluster Environment Configuration*. Repository files: `configs/env.spark.sh` and `configs/pin.main.env`.
