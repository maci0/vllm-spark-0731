# vllm-spark-0731

Official **vLLM v0.28.0rc2** + **b12x 1.2.6** kernels + **DSpark k=5** for
`deepseek-ai/DeepSeek-V4-Flash-0731` on **2x DGX Spark** (GB10, sm_121a).

**Resume from [HANDOFF.md](HANDOFF.md).** The image has not been built on GB10 yet.

One image, two KV dtypes:

| `--kv-cache-dtype` | Kernels | Spec | Notes |
|---|---|---|---|
| `fp8_ds_mla` | `--linear-backend b12x --moe-backend b12x` | DSpark k=5 | default, stock V4 layout |
| `nvfp4_ds_mla` | same | DSpark k=5 | 584-byte DSV4 page, not GLM 432 |

Do not set `--attention-backend B12X_MLA_SPARSE` on this image. That flag is
the eugr rollback stack (`configs/pin.eugr-b12x.env`) and mixing it with
`nvfp4_ds_mla` is the failed 432-vs-584 overlay.

Checkpoint is **0731 only**. Preview `DeepSeek-V4-Flash` is refused.

## What is in the image

| Piece | Source |
|---|---|
| Base | `vllm/vllm-openai:v0.28.0rc2` (build `--platform linux/arm64`) |
| Wheel | `b12x==1.2.6` (PyPI, Luke Alonso) |
| `#52016` b12x linear | already in rc2 |
| `#52018` b12x MXFP4 MoE | merged 2026-08-21, **after** rc2, cherry-picked |
| `#50645` mHC TileLang guard | open, cherry-picked (SM12x has no DeepGEMM prenorm) |
| `B12xWarmupUnit` / `get_b12x_fused_moe` | on main, not in rc2 |
| `nvfp4_ds_mla` | 584-byte DSV4 page + MLA guard narrowed to exact `nvfp4` |
| DSpark | stock in rc2; serve flags lock `method=dspark`, `k=5` |

`--moe-backend b12x` is the MXFP4 path DeepSeek-V4 experts need.
`flashinfer_b12x` is a different CuteDSL NVFP4-weight path. Use `b12x`.

This repo does **not** vendor a DeepSeek-V4 NVFP4 CUDA writer. The 584-byte
page is the DSV4 envelope (same billed size as fp8 content). Real ~32% B/token
saving was measured on `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, not on a Python
dtype alias. Build asserts reject a 432-byte GLM page.

## Build (on the Spark)

```bash
./scripts/00-prereq.sh
./scripts/01-download-0731.sh ~/models/ds4-flash-0731
./scripts/02-build-image.sh
```

## Serve (both nodes)

Copy `configs/nodes.env.example` to `configs/nodes.env` and set QSFP fabric IPs, not the LAN.

Start the **worker first**, then the head. The worker is `--headless`; only rank 0 binds `:8000`. Dist init uses `--master-addr ${HEAD_IP}` (default without it is `127.0.0.1` and the cluster never forms).

Worker (rank 1):

```bash
NODE_RANK=1 VLLM_HOST_IP=<worker fabric> HEAD_IP=<head fabric> \
  ./scripts/05-serve.sh fp8
```

Head (rank 0):

```bash
NODE_RANK=0 VLLM_HOST_IP=<head fabric> HEAD_IP=<head fabric> \
  ./scripts/05-serve.sh fp8
```

NVFP4 KV, same kernels and DSpark:

```bash
./scripts/05-serve.sh nvfp4
```

Confirm in the log:

```
Using DeepSeek V4 padded nvfp4_ds_mla KV cache format   # nvfp4 stack only
method=dspark
GPU KV cache size: N tokens
```

k is locked at 5 (0731 `n_predict=5`). 7 is rejected at boot, 10 crashes.

```bash
./scripts/06-validate.sh
./scripts/07-stop.sh    # docker rm -f is not enough; this also pkill VLLM::
```

## Prebuilt rollbacks

| Pin | Image | KV | Attention |
|---|---|---|---|
| `pin.golden.env` | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | nvfp4_ds_mla (measured 7,650 B/tok) | default |
| `pin.eugr-b12x.env` | eugr nightly | fp8_ds_mla | `B12X_MLA_SPARSE` |

```bash
./scripts/02-pull-image.sh golden
```

Field measurements: [maci0/dgx-spark-deepseek-v4-flash-0731](https://github.com/maci0/dgx-spark-deepseek-v4-flash-0731).

## Layout

```
configs/    pin.env (fp8) pin.nvfp4.env pin.eugr-b12x.env pin.golden.env
patches/    apply_overlays.py, asserts, files copied into the image
scripts/    00 prereq .. 07 stop
tests/      stack / checkpoint / overlay unit tests
```
