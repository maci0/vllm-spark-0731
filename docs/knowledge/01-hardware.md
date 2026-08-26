[← Index](00-index.md) · [Glossary](glossary.md)

# Hardware: DGX Spark (GB10, SM12x)

## NVIDIA GB10 Grace Blackwell Superchip (SoC) Datasheet

| Component / Subsystem | Official Specification | Architectural Notes |
|-----------------------|------------------------|---------------------|
| **SoC Architecture** | NVIDIA Grace Blackwell | Fused Arm CPU + Blackwell GPU on unified substrate |
| **GPU Architecture** | Blackwell (SM12x / `sm_121a` / SM family 120) | Compute Capability 12.1 |
| **Streaming Multiprocessors (SMs)** | 48 SMs | 6,144 CUDA cores |
| **Tensor Cores** | 5th Generation Tensor Cores | Native NVFP4, MXFP4, FP8, BF16, TF32, FP16 |
| **AI Tensor Performance** | Up to 1 PFLOP (1,000 TOPS) dense FP4 | Optimized for dense FP4/NVFP4 matrix operations |
| **Ray Tracing Cores** | 4th Generation RT Cores | Hardware-accelerated BVH traversal |
| **CPU Subsystem** | 20-core Armv9 CPU | 10× Cortex-X925 high-performance + 10× Cortex-A725 high-efficiency cores |
| **Unified System Memory** | 128 GB coherent LPDDR5x (UMA) | 256-bit memory bus, ~273 GB/s peak bandwidth, NVLink-C2C interconnect |
| **Usable CUDA Pool** | ~121.7 GiB allocatable | Shared dynamically between CPU, OS, page cache, and GPU |
| **Network Interface** | NVIDIA ConnectX-7 NIC | 100/200 GbE QSFP, RoCE v2 (`enp1s0f1np1`, GID 3) |
| **Form Factor & OS** | Compact Desktop Workstation | NVIDIA DGX OS 7.5.0 (Ubuntu 24.04 LTS aarch64 base) |

### Key Implications for Kernels

- **DeepGEMM**: Compiled for SM12x in matched-main (`main-b12x`) via nv_dev (`a6b593d`, pinned back from `8b1392b` 2026-08-25), where `is_deep_gemm_supported()` is True. However, specific unsupported shapes and ops (like 2-state MQA or mHC broadcast) require fallbacks/guards.
  > ⚠️ **CRITICAL WARNING — Pure-FP8 Linear Dispatch**: In upstream `nv_dev 8b1392b`, `csrc/apis/gemm.hpp:851` aliases `fp8_gemm_nt` to `fp8_fp4_gemm_nt`. On SM12x (`arch_major == 12`), that dispatcher calls `sm120_fp8_fp4_gemm_1d1d` unconditionally, feeding FP8 weights to an FP4 kernel and producing silent output corruption (e.g. `' Septy…'`). The matched-main pin-back to `a6b593d` removes the aliasing, but GB10 validation is still blocked at the JIT toolchain level — **`LINEAR_BACKEND=deep_gemm` stays off**; linear layers remain pinned to `--linear-backend b12x` (see [09-golden-deepgemm.md](09-golden-deepgemm.md)).
- **CUTLASS block-FP8**: SM90/SM100 only. **Does not run on SM12x**. Falls back to PyTorch/Triton/TileLang.
- **b12x**: Purpose-built for SM120/SM121. **This is the primary kernel path** on Spark.
- **FlashInfer**: Has SM12x support for DSV4 MLA (TOPK 192 added in #4380).

### TORCH_CUDA_ARCH_LIST
```bash
TORCH_CUDA_ARCH_LIST=12.1a
```
The `a` suffix enables PTX for forward compatibility.

---

## Memory: 128 GiB UMA

- **No separate VRAM** — GPU and CPU share the same physical memory pool.
- **16 GiB `/swap.img` enabled on both nodes** (fstab `sw` entry). spark2's swap was absent at boot because the `swap.img.swap` unit was **masked** (`/etc/systemd/system/swap.img.swap -> /dev/null`); a mask suppresses the fstab-generated unit, so earlyoom logged `swap total: 0 MiB` at boot. Unmasked 2026-08-24 — unit state is now `generated` and it auto-activates at boot (journal chain: `Activating swap swap.img.swap` → `Adding 16777212k swap` → `Activated` → `swap.target reached`).
- **`vm.swappiness=10`** persisted in `/etc/sysctl.d/99-dgx-spark-swap.conf` on both nodes (was 100; with disk swap live, swappiness=100 caused decode stalls). zswap sits on top (`zstd/zsmalloc`, `max_pool_percent=5`).
- **GPU_MEMORY_UTILIZATION=0.8** is the safe ceiling. 0.85 allocated 20 GiB KV then earlyoom SIGTERM'd the container.
- **KV offload to NVMe** via LMCache GDS or vLLM OffloadingConnector is the path to >0.8 util.

---

## Network: 2-Node RoCE over ConnectX-7

| Node | Fabric IP | Mgmt IP | Role |
|------|-----------|---------|------|
| spark1 | 10.0.1.1 | 192.168.0.211 | head (rank 0) |
| spark2 | 10.0.1.2 | 192.168.0.212 | worker (rank 1) |

### NCCL / UCX Config (from `configs/env.spark.sh`)

```bash
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export TP_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3        # GID 3 = RoCE v2 (IPv4 UDP encapsulation) on ConnectX-7
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Required at Serve Time
```bash
export VLLM_HOST_IP=10.0.1.1  # or .2
export HEAD_IP=10.0.1.1
```

---

## Software Stack Pins

| Component | Version | Source |
|-----------|---------|--------|
| CUDA | 13.3.1 | `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04` |
| PyTorch | 2.14 (source) | `release/2.14` branch, built for `12.1a` |
| NCCL | source | built for `sm_121` |
| Triton | 3.7.1 | |
| Rust | stable | for CUTLASS DSL |
| b12x | master | `local-inference-lab/b12x` |
| cutlass-dsl | **4.7.0** | metadata rewrite (not 4.6.2 from vLLM cuda.txt) |
| FlashInfer | main | DSV4 TOPK 192 present |
| DeepGEMM | nv_dev `a6b593d` | pinned back from `8b1392b` 2026-08-25 (SM12x fp8 regression; see [09-golden-deepgemm.md](09-golden-deepgemm.md)) |
| DeepEP | main | for EP if needed |

---

## Cluster Provisioning Checklist

- [ ] Both nodes on same RoCE fabric (QSFP connected)
- [ ] `nvidia-smi` shows GB10, driver ≥ 570, CUDA 13.3
- [ ] `ibv_devinfo` shows `mlx5` ports UP
- [ ] `ping 10.0.1.2` from spark1 works (and vice versa)
- [ ] `nccl-test` all-reduce passes at line rate
- [ ] Docker + nvidia-container-toolkit installed
- [ ] `configs/nodes.env` copied and sourced on both nodes
- [ ] Model downloaded to `${HOME}/models/ds4-flash-0731` on both (or `$HOST_MODEL_DIR`)

---

## Related Docs

- [02-model.md](02-model.md) — Model architecture implications for this hardware
- [03-kernels-attention.md](03-kernels-attention.md) — Why b12x is the only viable kernel path
- [06-deployment.md](06-deployment.md) — Build and run procedures
- [08-upstream.md](08-upstream.md) — SM12x gaps tracked upstream

### Raw evidence (field notes)

- [`../field-notes/dgx-spark/README.md`](../field-notes/dgx-spark/README.md) — cluster topology, 1M recipe, measured ceilings
- [`../field-notes/nvfp4/MHC_DEEPGEMM_SM121.md`](../field-notes/nvfp4/MHC_DEEPGEMM_SM121.md) — the SM121 mHC assertion that motivates the guards

---

**[← Index](00-index.md) · [Glossary](glossary.md) · [Next](02-model.md) →**
