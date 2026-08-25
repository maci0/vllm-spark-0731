[← Back to Knowledge Index](00-index.md)

# Deployment & Images

## Lineages Measured on This Cluster (One Harness, 2026-08-22)

| Image | vLLM | KV | KV Pool | c1/c5/c6 tok/s |
|-------|------|-----|---------|----------------|
| anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | 0.25.2 | nvfp4 (7650 B/t) | **2.00M** | 51.4 / 126.2 / 157.9 |
| eugr `dgx-vllm-eugr-nightly-b12x:2026081903` | 0.27.x | fp8 (11317) | 1.77M | 54.3 / 127.4 / ~109 |
| tonyd2wild `dspark-nvfp4-stage-c` | 0.21.1rc1 | NVFP4 (~11900) | 1.44M | 56.1 / 116.0 / 141.1 |
| eugr `spark-vllm-b12x` (≤512K, spec off) | main | FP8 UE8M0 | — | ~326 tok/s @ c48 |
| **main-b12x (this repo)** | main `e25c586b9` | nvfp4_ds_mla (584 B) | 97,737 | ~25.8 / — / 172 @ c32 |

### Golden re-measured 2026-08-24 (after the fix backlog)

Deployed stock on this cluster and validated (see HANDOFF Status → Golden):

- France temp 0, 128 tok: **c1 65.2, c6 216.8, c16 183.9, c32 186.2** —
  2.5× our c1. Plateau past c6 by design (`max_num_seqs=6`).
- Golden harness (BST coding, temp 0.7): c1 54.6–58.1, c3 108.5, c5 124.2,
  c6 155.0 — reproduces the 2026-08-22 table.
- KV **2,047,170 tokens** @ 7,650 B/token.

### Golden deploy procedure (anemll)

```bash
# On spark1: sparkrun drives both nodes. Copy the recipe OUT of a path
# containing "sparkrun" first (spark-launch.sh pkill -9 -f '[s]parkrun'
# matches its own argv if the recipe path contains that string).
ssh spark1 'cp ~/tonyd2wild/sparkrun/anemll-nvfp4.yaml /tmp/anemll-nvfp4-golden.yaml \
  && bash ~/spark-launch.sh /tmp/anemll-nvfp4-golden.yaml ~/anemll.log'
# Phases: image check (18.8 GB, cached) -> model distribute ~21 min (81
# files) -> weights 79.17 GiB / ~253 s -> warmup -> health 200 (~14 min
# total after the fetch). Containers: sparkrun_<id>_node_0 / _node_1.
# Liveness = a tiny generation, NOT /health (a wedged worker keeps 200).
```

---

## The Live Pin (`configs/pin.main.env`)

```bash
# Core stack
B12X_MLA_SPARSE + nvfp4_ds_mla + b12x linear/MoE
# Speculative
DSpark k=5, FULL_AND_PIECEWISE
# Runtime
util 0.8, max_num_seqs 32 (was 8), capture 192
# Loading
InstantTensor + hybrid lazy draft
```

**Fallback:** overlay rc2 (`PIECEWISE`, `FLASHINFER_MLA_SPARSE_DSV4`) via `05-serve.sh nvfp4`.

---

## Build Procedure

```bash
# On spark1 (head node)
./scripts/00-prereq.sh              # Validate dependencies (docker, nvidia-container-toolkit, etc.)
./scripts/01-download-0731.sh       # Download model to ${HOME}/models/ds4-flash-0731 (or $HOST_MODEL_DIR)
./scripts/02-build-main.sh          # Build matched-main image (takes ~45 min)
./scripts/03-apply-main-overlays.sh # Apply SM12x overlays
./scripts/02-copy-main.sh           # Copy image to spark2 (docker save/load)
```

### Build Details (`scripts/02-build-main.sh`)

- **Base**: `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04`
- **PyTorch**: Built from source (`release/2.14`, `TORCH_CUDA_ARCH_LIST=12.1a`)
- **NCCL**: Built from source for `sm_121`
- **vLLM**: Git `main`, `--no-build-isolation`
- **b12x**: Git master + **cutlass-dsl 4.7.0** (metadata rewrite, not 4.6.2)
- **FlashInfer**: Git main (DSV4 TOPK 192)
- **DeepGEMM**: nv_dev commit `8b1392b`
- **InstantTensor**: For fast cold start

---

## Serve Procedure

### Prerequisites (Both Nodes)

```bash
# Source environment
source configs/env.spark.sh
source configs/nodes.env  # Sets VLLM_HOST_IP and HEAD_IP

# Model must exist at ${HOME}/models/ds4-flash-0731 (or $HOST_MODEL_DIR)
```

### Start Worker First (spark2), Then Head (spark1)

```bash
# 1. Start worker on spark2 (rank 1 via non-interactive SSH)
ssh -o ControlPath=none spark2 'cd /tmp/vllm-spark-0731 && bash scripts/05-serve.sh main </dev/null'

# 2. Start head on spark1 (rank 0 - locally or via remote SSH)
# Local execution on spark1:
./scripts/05-serve.sh main
# Or remote SSH execution to spark1:
# ssh -o ControlPath=none spark1 'cd /tmp/vllm-spark-0731 && bash scripts/05-serve.sh main </dev/null'
```

> **Deployment Warnings:**
> - **Process Isolation:** Never chain `07-stop.sh` and `05-serve.sh` in a single command string or SSH invocation because `07-stop.sh` sends `pkill -9` to leftover processes and cleans `/dev/shm`, which can race or abort the newly launching serve process.
> - **Docker Commit Entrypoint Restoration:** When committing a debugging container in-place with `docker commit`, always explicitly restore the entrypoint flags: `--change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD []'`.

### Serve Script Options (`scripts/05-serve.sh`)

```bash
./scripts/05-serve.sh [fp8|nvfp4|eugr|main|golden]
```

| Stack | Pin File | Image Tag | Attention Backend | Linear / MoE | KV Cache | Status / Role |
|-------|----------|-----------|-------------------|--------------|----------|---------------|
| `fp8` | `configs/pin.env` | `vllm-spark-0731:v0.28.0rc2-b12x` | `FLASHINFER_MLA_SPARSE_DSV4` | `b12x` / `b12x` | `fp8_ds_mla` | Legacy fallback (576 B) |
| `nvfp4` | `configs/pin.nvfp4.env` | `vllm-spark-0731:v0.28.0rc2-b12x` | `FLASHINFER_MLA_SPARSE_DSV4` | `b12x` / `b12x` | `nvfp4_ds_mla` | Overlay fallback (584 B) |
| `eugr` | `configs/pin.eugr-b12x.env` | `dgx-vllm-eugr-nightly-b12x:2026081903` | `B12X_MLA_SPARSE` | `b12x` / `b12x` | `fp8_ds_mla` | Upstream comparison |
| `main` | `configs/pin.main.env` | `vllm-spark-0731:main-b12x` | `B12X_MLA_SPARSE` | `b12x` / `b12x` | `nvfp4_ds_mla` | **Live Production** (matched-main `e25c586b9`) |
| `golden` | `configs/pin.golden.env` (or sparkrun `anemll-nvfp4.yaml`) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | `FLASHINFER_MLA_SPARSE_DSV4` | Stock / `flashinfer_b12x` | `nvfp4_ds_mla` (real NVFP4 writer) | Reference benchmark (2.05M pool) |

---

## Validation

```bash
# Quality gate (must pass)
VALIDATE_STACK=main ./scripts/06-validate.sh

# Tests: /models, greedy France, 8-way aggregate, DSpark acceptance
```

### Expected Output

```
GET http://10.0.1.1:8000/v1/models
deepseek-v4-flash 65536

POST completions greedy France (32 tok)
' Paris. The capital of Spain is Madrid. The capital of Italy is Rome. ...'
usage {...}
first_token ' Paris' logprob -0.245 n_tie=1
✓ France green
```

---

## Image Management

### Tagging

- **Main**: `vllm-spark-0731:main-b12x`
- **Overlay fallback**: `vllm-spark-0731:v0.28.0rc2-b12x`

### Docker Commit (If Needed)

```bash
# After container runs successfully, preserve the serve entrypoint
docker commit --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD []' <container_id> vllm-spark-0731:main-b12x
# Verify entrypoint
docker inspect vllm-spark-0731:main-b12x | grep -A2 Entrypoint
# Must show: "Entrypoint": ["vllm", "serve"], "Cmd": []
```

---

## KV Offload (Optional, Experimental)

```bash
# LMCache GDS
ENABLE_LMCACHE=1 ./scripts/05-serve.sh main

# vLLM Native
KV_OFFLOAD=native ./scripts/05-serve.sh main
```

**Currently faults on this model** — hybrid multi-group cache vs flat transfer path.

---

## Stop Procedure

```bash
# Stop head on spark1 (locally or via SSH) and worker on spark2 via SSH
./scripts/07-stop.sh
ssh spark2 "cd /tmp/vllm-spark-0731 && ./scripts/07-stop.sh"
```

---

## Related Docs

- [00-index.md](00-index.md) — Quick links
- [01-hardware.md](01-hardware.md) — Cluster setup
- [03-kernels-attention.md](03-kernels-attention.md) — Backend differences
- [04-quantization-kv.md](04-quantization-kv.md) — KV dtype details
- [05-performance.md](05-performance.md) — Expected numbers
- [07-gotchas.md](07-gotchas.md) — Common failure modes