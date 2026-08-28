[← Index](00-index.md) · [Glossary](glossary.md)

# Hardware: DGX Spark (GB10, SM12x)

> **Scope:** GB10 hardware, cluster networking (RoCE), thermal/power, vendor landscape, node setup — the physical layer everything else runs on.

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

## GB10 Vendor Landscape & Official Systems

NVIDIA lists the DGX Spark Founders Edition and **seven other NVIDIA-certified
GB10 systems**; all share the same GB10 SoC, 128 GB LPDDR5x (273 GB/s) and
ConnectX-7, and differ in storage SKU, cooling, firmware, power adapter,
initial image and support policy. **Same spec ≠ same sustained performance** —
cooling path/fan curve, SSD model, GB10 clock/power/firmware state,
ConnectX-7 temperature and the real RDMA/NCCL path all differ per unit.


| System (NVIDIA-certified GB10) | Notable spec / differentiator |
|---|---|
| **NVIDIA DGX Spark (Founders Edition)** | Baseline for NVIDIA recipes; 1/4 TB NVMe SKU; 240 W adapter; DGX OS; CX-7 |
| **Acer Veriton GN100** | 128 GB, up to 4 TB; PyTorch/Jupyter/Ollama; 2-unit direct + up-to-4-unit switch guidance |
| **ASUS Ascent GX10** | 1/2/4 TB series; QuietFlow 3-fan + dual vapor chamber; 240 W peak; most community DeepSeek field material |
| **Dell Pro Max with GB10** | 2 TB QLC / 4 TB SED; 2×200G QSFP; **280 W adapter**; Desktop vs Network Appliance mode |
| **GIGABYTE AI TOP ATOM** | Up to 4 TB; CX-7; AI TOP Utility (model download / inference / RAG / ML) |
| **HP ZGX Nano** | 2/4 TB SED; 2×200G QSFP; 240 W; DGX OS / Ubuntu only (no Windows); rack mount not supported |
| **Lenovo ThinkStation PGX** | 1/4 TB SED; 2×CX-7 QSFP; 240 W; TPM 2.0 / Secure Boot / NVIDIA FW Recovery |
| **MSI EdgeXpert** | 1/4 TB SED; 2×QSFP CX-7; 240 W; Docker Compose + private-CA edge-appliance direction |

- **[MEASURED] (StorageReview, 5 units, same room)**: sustained-load cooling
  differs — Acer lowest (CPU 74.6 °C / GPU 68 °C), FE/Dell/Gigabyte ~87–88 °C
  CPU / ~80–82 °C GPU, ASUS in between. This is a **cooling/power comparison,
  not a tok/s leaderboard** (NVMe models and GPU rail power differ per unit).
- Different-architecture "alternatives" (AMD Ryzen AI Halo, Apple Mac Studio)
  can match memory size but are **not** GB10 cluster nodes — no CUDA / DGX OS /
  ConnectX-7 / NCCL recipe carries over; do not mix them into a Spark cluster.
- NVIDIA's "single up to 200B, dual 405B" is *model-size positioning*, not a
  guarantee a given checkpoint generates stably at a given context/runtime,
  and not a benchmark result.
- Verify post-purchase: `ethtool`, `ibdev2netdev`, `NCCL_DEBUG=INFO` — a vendor
  page saying 4 TB / 200 Gb/s does not guarantee the purchased SKU or real
  inference payload.

## Thermal, Power & Clock Capping

- **Power numbers (official docs)**: the included adapter is **240 W**; the
  GB10 SoC TDP is **140 W**, the rest of the system ~100 W; regulatory figures
  are **max 233.2 W AC, idle 38.0 W, off-mode 4.1 W**. Use the supplied 240 W
  adapter — other PSUs can cause performance degradation, boot failure or
  unexpected shutdown.
- `nvidia-smi` device / GPU-rail power ≠ wall AC (CPU, unified memory, SSD,
  ConnectX-7, fans, adapter losses differ). Never merge `nvtop`/GPU-rail values
  and AC wall-meter values into one power column.
- **Clock capping** (`sudo nvidia-smi -lgc 0,2200` to apply, `sudo nvidia-smi -rgc`
  to remove) is an *experiment*, not an optimization rule: decode is
  memory-bound and holds up under a cap, while compute-bound cold prefill
  loses speed. **[MEASURED] (agjs/gb10-clock-cap, 2×GB10 reference)**: 2200 MHz
  cap → peak temp 90→78 °C, GPU rail ~63.1→40.1 W/node, decode 73.3→72.5 tok/s;
  the 2000/1800 MHz sweep cost cold prefill +8.1% / +13.0%. Apply the same cap
  on every node and record actual per-host clock.
- **Low-clock / low-power state that looks healthy** **[CLAIMED, Blackwellboy X]**:
  GPU util ~96%, P-state P0, no throttle reason, but SM clock ~799 MHz and load
  power ~19.5 W; decode ~44 tok/s. After a full power disconnect: SM clock
  ~2.3–2.5 GHz, ~92 W, decode 73.9 tok/s. Community workaround (not confirmed
  root cause): stop workload → save logs → normal shutdown → unplug power brick
  AND AC → wait a few minutes → reconnect. `P0 + high utilization` alone does
  not mean healthy — read `clocks.sm`, `power.draw`, temperatures and real tok/s
  together.
- Community thermal data: thermal shutdown near **95 °C** under sustained
  inference is reported; dual-Spark ducted-cooling cage builds exist but are
  DIY, not official designs. On rising temperature, stop the workload first,
  then check airflow/ambient/dust/spacing; never bypass thermal protection.


## Unified-Memory Details

- **No dedicated framebuffer** (official Known Issues): memory display differs
  from normal GPUs — do not judge actually allocatable memory by
  `cudaMemGetInfo` alone. A 128 GiB SKU shows `MemTotal` ≈ **121.6 GiB**; judge
  fit by `MemAvailable` / KV / workspace / OS headroom, not labeled capacity.
- Budget checklist: weights + KV cache (grows with context × concurrency) +
  CUDA workspace/graph + tokenizer/runtime/framework + OS/desktop/agent
  sidecars + **safety headroom**. "Weight < 128 GiB" is only a start; if
  weights fit but no KV room remains, it is a *load demo*, not "runnable".
- Memory-full failures can take down SSH/HDMI/OS, not just one request — treat
  headroom as an ops resource. Do not raise `gpu-memory-utilization` when
  memory keeps shrinking; stop the server and check logs/processes first.
- MoE memory notes: total ≠ active parameters — small active params do not
  remove all expert weights, KV cache, or runtime buffers.


## Storage & Port Smoke Tests

- Smoke order: `df -h /` → `ss -ltnp` → `curl -fsS http://127.0.0.1:PORT/v1/models`.
  A responding `/v1/models` means the endpoint is alive — **not** verified
  model quality or tool parser.
- Then one short text request; save the response's `model`, `finish_reason`
  and `usage` fields. On error, keep the request JSON, server log and runtime
  version together.
- Failure classification: port connection failure → process/address/firewall;
  model-list failure → server startup or route; generation failure →
  weight/template/memory; structured-output failure → parser/schema/chat
  template. Long-context / concurrency tests only after the first smoke passes.
- Ports are cheap and explicit: existing servers on 8082/8083, DeepSeek recipe
  on 8888 — pre-record endpoint host+port, served model id, other GPU users,
  memory cap and log location; check `ss -ltnp` and `ps -ef | grep -E 'vllm|sglang|llama|spark'`.


## Mac + Spark (MCDMA / RDMA) Research

Community work only: **MCDMA** (Metal unified memory ↔ CUDA memory over USB-C)
is an experimental prototype (~939 MB/s single link, ~24 µs RTT claimed) with
no public reproduction and no official NVIDIA documentation of the memory
registration — it is not a Spark↔Mac TP path. Official split: Spark↔Spark =
CX-7/QSFP/RoCE/NCCL compute net; Mac = control host / API router / MLX worker
over Ethernet. Mac as a CUDA/NCCL rank alongside Sparks has no verified path.


## Node Setup & Preflight (additions)

- Preflight record: `hostnamectl`, `uname -a`, `nvidia-smi`, `free -h`,
  `df -h`; then DGX OS/Ubuntu version, kernel + NVIDIA driver, CUDA/PyTorch/
  vLLM versions, model file size + revision, other running inference processes,
  idle memory + temperature.
- First boot: NVIDIA separates display-attached and network-appliance modes;
  initial setup needs stable internet + the included adapter and may auto-reboot
  (up to 10+ minutes).
- **ASUS GX10 field gotcha**: `apt autoremove --purge` removed NVIDIA/DGX
  platform packages. Safe order: save package state → mark NVIDIA/DGX/CUDA/
  Docker packages protected → dry-run with `apt-get -s` → abort if a protected
  package appears → re-check `/proc/cmdline`, `nvidia-smi`, Docker, failed
  units. Don't copy ASUS drivers / firmware / interface names to other GB10
  units.
- **ConnectX-7 socket-direct**: one physical QSFP port can appear as MULTIPLE
  Linux interfaces — inspect `ip -br link`, `/sys/class/net/*/phys_switch_id`,
  `/sys/class/net/*/phys_port_name`, `ibdev2netdev` before bonding or picking
  an interface by name.
- **200 G negotiated ≠ payload**: cases capped at 12–13 Gbps recovered to
  ~109–111 Gbps after a full power drain (field workaround, not official).
  Validate in order: link up → IP/MTU → RDMA/`ib_write_bw` → NCCL collective →
  model request; expect `NET/IB` in `NCCL_DEBUG=INFO`, never `NET/Socket`.
- Don't A/B models on day one by swapping only the model name — pin image
  digest, model revision, engine commit, and record the shared-GPU state.


---


## Official DGX Spark spec & release baseline (linked NVIDIA docs, 2026-08-26)

From the DGX Spark Hardware Overview / User Guide / Release Notes / Porting Guide / Known Issues / Spark Clustering pages ([REFERENCES.md](../../REFERENCES.md) → NVIDIA official):

- **SoC**: Grace Blackwell; CPU = 2 clusters (5× Cortex-X925, 16 MB L3 + 5× Cortex-A725, 8 MB L3); GPU = 6,144 CUDA cores, 5th-gen tensor cores (FP4), up to 1,000 TOPS / 1 PFLOP FP4-sparse. Memory: 128 GB LPDDR5x @ 4266 MHz, 16 channels, **273 GB/s**.
- **Power**: GB10 SoC TDP **140 W**; supplied **240 W PSU mandatory** (lower-rated PSU → reduced perf / boot failure); measured IEC 62623: **233.2 W max AC, 38.0 W idle, 4.1 W off**.
- **Software baseline (Founders Edition)**: DGX OS **7.5.0** (Ubuntu 24.04-based), driver **580.159.03**, CUDA **13.0.2**, kernel **6.17**. GB10 partner systems may lag.
- **Networking**: ConnectX-7 with 2× QSFP ports, **200 Gb/s Ethernet-only**; NIC↔SoC via two independent PCIe Gen5 ×4 links; two cables = **4 Ethernet + 4 RoCE interfaces** (inspect with `ibdev2netdev`). Approved cables only: Amphenol NJAAKK-N911 / Luxshare LMTQF022-SD-R. Official cluster topologies: **2–3 nodes direct (3 = ring), up to 4 via switch**; one cable per link (two cables between a pair does not help); link-speed acceptance floor **184 Gbit/s**.
- **Known issues with official backing**: `nvidia-smi` shows "Memory-Usage: Not Supported" (no dedicated framebuffer — expected); `cudaMemGetInfo` under-reports allocatable memory (CPU can reclaim DRAM via swap) — read `/proc/meminfo` (MemAvailable + SwapFree) instead.
- **Clock reality [RECIPE — agjs/gb10-clock-cap]**: stock GPU clock is power-limited to **~2455 MHz** (not the 3003 MHz spec sheet); a **2200 MHz cap is the knee**: −1% decode, −36% GPU power, −12 °C peak temp, zero throttling (8.2 s → 0 s); decode is bandwidth-bound so a cap barely touches it, prefill takes the whole cost. `nvidia-smi -lgc 0,2200` (host root only), revert `-rgc`.
- Default-open ports: 22/tcp (OpenSSH), 80/tcp (HTTP); DGX Dashboard on localhost:11000.

### Official NVIDIA benchmark & ops additions (developer blogs, User Guide, playbooks, 2026-08-26)

Sources: developer.nvidia.com blog posts (performance, agents & multi-node, CES 2026, Computex 2026, secure-agent), docs.nvidia.com DGX Spark User Guide (system-overview / system-recovery / os-and-component-update), docs.nvidia.com certified-systems, NVIDIA dgx-spark-playbooks performance guide ([REFERENCES.md](../../REFERENCES.md) → NVIDIA official).

- **Official inference figures [OFFICIAL]**: at ISL|OSL 2048|128, BS=1 — Qwen3 14B NVFP4/TRT-LLM **5,929 prefill / 22.7 gen tok/s**; GPT-OSS-20B MXFP4/llama.cpp 3,670 / 82.7; GPT-OSS-120B MXFP4/llama.cpp 1,725 / 55.4; Llama 3.1 8B NVFP4/TRT-LLM 10,257 / 38.7; Qwen2.5-VL-7B NVFP4 65,832 / 41.7; **Qwen3 235B NVFP4 on dual Spark 23,477 / 11.7**. Backends: TRT-LLM, llama.cpp, vLLM; NVFP4 = Blackwell 4-bit FP (near-FP8, <1% degradation) vs MXFP4 = OCP microscaling FP4.
- **Agent-context figures [OFFICIAL]** (128K|1K ISL/OSL, BS=1): Nemotron 3 Super 120B NVFP4/TRT-LLM 2,855 prefill / 18 gen / 99 s e2e; Qwen3.5 35B-A3B FP8/vLLM 3,080 / 35.75 / 73 s; Qwen3 Coder Next 80B FP8/vLLM 2,390 / 28.95 / 89 s; 1→4 concurrent: 2.6× time, prefill 3,261 → 9,616 tok/s. Agent contexts commonly 30K–120K tokens, up to 250K.
- **TP scaling [OFFICIAL]**: TPOT ~2× at TP2, ~4× at TP4 (Llama 3.3 70B NVFP4 TRT-LLM, 32K|1K: TTFT 33,415 / 21,384 / 15,552 ms; TPOT 269 / 133 / 72 ms). Cluster sizing: 1 node = large-context inference + fine-tune ≤120B; 2 nodes = ≤400B; 3-ring = fine-tune; 4 nodes via RoCE 200 GbE switch = ≤700B, "local AI factory" — measured on the dual-Spark TP=2 setup this repo runs.
- **Software optimizations [OFFICIAL]** (CES 2026): Qwen 235B on dual Spark — NVFP4 + speculative decoding up to **2.6× vs FP8** (NVFP4 −40% memory at near-FP8 accuracy); llama.cpp MoE updates ≈ +35%; 128 GB runs GPT-OSS-120B / FLUX 2 (90 GB) at full precision; **EAGLE-3** spec-decoding with a built-in drafting head (GPT-OSS-120B — no separate draft model). OEM GB10 systems were entering NVIDIA-Certified Systems testing at that post.
- **NVIDIA-Certified Systems [OFFICIAL]** (docs.nvidia.com/certification-programs): exactly 7 certified GB10 systems — Acer Veriton GN100-UD11, ASUS Ascent GX10, Dell Pro Max with GB10, GIGABYTE AI TOP ATOM ATAGB10-9000, HP ZGX Nano AI Station, Lenovo ThinkStation PGX, MSI EdgeXpert. Program tests performance/functionality/scalability/security with NGC software; desktop-class units tested standalone.
- **Connectivity & form factor [OFFICIAL]** (User Guide System Overview): Wi-Fi 7 + 10 GbE + ConnectX-7; 150×150×50.5 mm; access modes local (KVM) / network (SSH, NVIDIA Sync, remote desktop) / hybrid; training/fine-tune positioning "up to 200 billion parameters"; primary tutorial hub = build.nvidia.com/spark.
- **Docker/container baseline [OFFICIAL]** (secure-agent blog): `sudo nvidia-ctk runtime configure --runtime=docker`; **`default-cgroupns-mode: host` required in `/etc/docker/daemon.json` for DGX Spark containers**; Ollama must run under systemd with an `OLLAMA_HOST=0.0.0.0` override (`/etc/systemd/system/ollama.service.d/override.conf`) — a manual start won't pick it up.
- **OS & recovery ops [OFFICIAL]** (User Guide): DGX OS is Ubuntu-based with Ubuntu Pro (10-year Canonical support); manual update fallback = `sudo apt update && sudo apt dist-upgrade && sudo fwupdmgr refresh && sudo fwupdmgr upgrade`; **system recovery is Founders-Edition-only** — recovery media is a tar.gz (not ISO), USB ≥16 GB, wired USB keyboard required in UEFI; June 2026 DGX OS: OTA updates no longer installed by default during initial setup.
- **Benchmark harness & fabric ceiling [RECIPE]** (playbook performance guide): TRT-LLM `trtllm-bench`/`trtllm-serve` (1.2.0rc6), vLLM `vllm bench throughput|serve` (25.12-py3), SGLang `bench_offline_throughput`; measured dual-Spark `ib_write_bw` ≈ 92.6 + 97.3 ≈ **189.9 Gbps aggregate (~97 Gbps/link, not 200)**; host CUDA is fixed-cadence — run NGC containers for newer CUDA.
- **Multi-node switch requirements [OFFICIAL]** (Computex blog + multi-Spark playbook): ≥4 QSFP56-DD ports, breakout to 25/50/100/200/400 G, 200–400 G recommended max port speed, 1× 1/10GbE mgmt port, RoCE v2, ≥0.8–1.6 Tbps switching; Sync cluster assistant automates 2–4-node readiness (OTA, sudo), CX-7 topology probe (LLDP/BPDU + interface/IP checks), IP planning/netplan and `ib_write_bw`/`ib_write_lat` validation.

## Related Docs

- [02-model.md](02-model.md) — Model architecture implications for this hardware
- [03-kernels-attention.md](03-kernels-attention.md) — Why b12x is the only viable kernel path
- [06-deployment.md](06-deployment.md) — Build and run procedures
- [08-upstream.md](08-upstream.md) — SM12x gaps tracked upstream
- [11-cost-decision.md](11-cost-decision.md) — GB10 vendor selection, CAPEX/OPEX/TCO

### Raw evidence (field notes)

- [`../field-notes/dgx-spark/README.md`](../field-notes/dgx-spark/README.md) — cluster topology, 1M recipe, measured ceilings
- [`../field-notes/nvfp4/MHC_DEEPGEMM_SM121.md`](../field-notes/nvfp4/MHC_DEEPGEMM_SM121.md) — the SM121 mHC assertion that motivates the guards

---

**[← Index](00-index.md) · [Glossary](glossary.md) · [Next](02-model.md) →**
