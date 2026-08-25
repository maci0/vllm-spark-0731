# vllm-spark-main-b12x

Matched **vLLM main** (Python + `.so` from the same commit) for
`deepseek-ai/DeepSeek-V4-Flash-0731` on **2x DGX Spark** (GB10, SM 12.1,
128 GiB UMA, TP=2 RoCE).

Image tag: `vllm-spark-0731:main-b12x`.

| Piece | Pin |
|---|---|
| Base | `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04` |
| PyTorch | `release/2.14` from source, `TORCH_CUDA_ARCH_LIST=12.1a` |
| NCCL | source, `sm_121` |
| vLLM | git `main`, `--no-build-isolation` |
| FlashInfer | git `main` (DSV4 TOPK 192) |
| b12x | git master + cutlass-dsl **4.7.0** metadata rewrite |
| Load | InstantTensor + hybrid lazy safetensors for DSpark draft |
| KV | `nvfp4_ds_mla` (584 B DSV4 envelope, not GLM 432/368) |
| Linear / MoE | `--linear-backend b12x`, `--moe-backend b12x` |
| Attention | `B12X_MLA_SPARSE` (target and DSpark draft). Stock main has no enum; overlay `patches/files/dsv4_b12x_sparse.py`. |
| Spec | DSpark k=5. Backbone FULL, sample eager. |
| Graphs | PIECEWISE 11/11 + FULL 7/7. TP AR in-graph. |
| Util | **0.8** (do not use 0.85 on spark2 — earlyoom 8% + decode stalls) |

Plan, pins, and provenance: [docs/PLAN-MAIN.md](docs/PLAN-MAIN.md). Comprehensive knowledge base: [docs/knowledge/00-index.md](docs/knowledge/00-index.md). Canonical Architecture & Codebase Audit: [outputs/vllm-spark-0731-docs-audit.md](outputs/vllm-spark-0731-docs-audit.md) ([Plan](outputs/.plans/vllm-spark-0731-docs.md)).

> **Speed / real-NVFP4 note (2026-08-24):** this matched-main image is the
> upstream/PR track. For maximum speed with real NVFP4 KV + DSpark, the
> golden image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, stock) serves
> **France c1 65.2 / c6 216.8 tok/s, KV 2.05M tokens @ 7,650 B/token** on
> this cluster (vs ~25.8 / ~95 here). See
> [docs/knowledge/05-performance.md](docs/knowledge/05-performance.md) and
> [HANDOFF.md](HANDOFF.md) → Status → Golden.
> 
> ⚠️ **Kernel Warning:** `LINEAR_BACKEND=deep_gemm` is currently blocked on SM12x (produces silent output corruption due to upstream `fp8_fp4_gemm_nt` aliasing in `nv_dev 8b1392b`). Keep `--linear-backend b12x` pinned until `deepgemm-fp8-1d1d-port.diff` is upstreamed.
Open upstream PRs: [docs/UPSTREAM.md](docs/UPSTREAM.md).

## History

This repo absorbed its three predecessors on 2026-08-25 — `vllm-spark-main-b12x`
(already fully contained here), `vllm-spark-nvfp4`, and
`dgx-spark-deepseek-v4-flash-0731` (field notes). The raw, unedited write-ups
live in [docs/field-notes/](docs/field-notes/README.md); their patches,
Dockerfiles, prod recipes, and scripts moved into `patches/`, `docker/`,
`configs/examples/`, and `scripts/` (see the field-notes index).

The older overlay image (`vllm/vllm-openai:v0.27.1` + rc2 Python) is the
fallback. Ops notes: [HANDOFF.md](HANDOFF.md). Do not overlay main Python
onto v0.27.1.

## Quality gate

Greedy `"The capital of France is"` (`temperature=0`, `max_tokens=32`) is
coherent English. First token `' Paris'`, `n_tie=1`. Chat answers `Paris.`.

```bash
VALIDATE_STACK=main ./scripts/06-validate.sh
```

Live on this image (2026-08-24 01:40 UTC, util 0.8, paged indexer, WO bmm,
DSpark backbone FULL):

| Check | Result |
|-------|--------|
| Greedy | `' Paris. ...'` n_tie=1 |
| Chat | `Paris` |
| 1-way | median 26.90 tok/s (128). Gate 17.81. Gather pin ~30.6. |
| 8-way | ~95 tok/s wall (c8); ~172 tok/s @ c32. Gate 52.12. |
| Indexer | b12x paged, page_size 64, packed-at-store, `sched=False` on 48-row capture |
| WO | `torch.bmm` dequant. Not MXFP8 `wo_proj.run()`. |
| KV | `nvfp4_ds_mla`, 97,737 tokens at util 0.8 |

2026-08-23 Phase 3 baseline (gather indexer, DSpark graphs off): 17.81 /
52.12 tok/s, KV 93,401.

`nvfp4_ds_mla` is still a 584 B alias of the fp8 DSV4 page. It is not a
memory win versus `fp8_ds_mla`.

## Build (spark1)

```bash
./scripts/00-prereq.sh
./scripts/01-download-0731.sh ~/models/ds4-flash-0731
./scripts/02-build-main.sh
./scripts/03-apply-main-overlays.sh
./scripts/02-copy-main.sh          # docker save | ssh spark2 docker load
```

## Serve (both nodes)

Copy `configs/nodes.env.example` to `configs/nodes.env`. Fabric IPs, not LAN.
Start the **worker first**, then the head.

Worker (rank 1):

```bash
NODE_RANK=1 VLLM_HOST_IP=<worker fabric> HEAD_IP=<head fabric> \
  ./scripts/05-serve.sh main </dev/null
```

Head (rank 0):

```bash
NODE_RANK=0 VLLM_HOST_IP=<head fabric> HEAD_IP=<head fabric> \
  ./scripts/05-serve.sh main </dev/null
```

Optional KV offload (native `OffloadingConnector` FS; LMCache GDS does not
support DSV4 HMA on this tree):

```bash
KV_OFFLOAD=native ./scripts/05-serve.sh main </dev/null
```

`ENFORCE_EAGER=1` is the graph fallback. Do not boot `FULL_AND_PIECEWISE` on
this pair; FULL collapsed France on the overlay pin. After a b12x-sparse
reapply, pass `--vllm-dir /opt/vllm/vllm` so apply does not `import vllm`
(a duplicated `B12X_MLA_SPARSE` enum crashes that import).

```bash
./scripts/07-stop.sh    # docker rm -f is not enough; also pkill VLLM::
```

Run stop and serve in **separate** SSH calls. `07-stop.sh` matches `05-serve.sh`
in the process list.

## Layout

```
configs/    pin.main.env (this image) plus overlay / eugr / golden pins
docker/     Dockerfile.main, overlays, pin_quack.py, asserts
patches/    SM12x keep/add overlays applied onto main
scripts/    02-build-main, 02-copy-main, 03-apply-main-overlays, 05-serve, 06-validate
docs/       PLAN-MAIN.md, UPSTREAM.md, LINEAGE.md, knowledge/ (comprehensive operational docs)
tests/      pytest suite for stack assertions and kernel helpers
```

## Testing

Run unit tests locally (skips GPU-dependent tests when PyTorch is not present):

```bash
pytest tests/
```
