# Handoff: vllm-spark-0731

Last updated: 2026-08-22.
Repo: https://github.com/maci0/vllm-spark-0731
Local: `/home/maci/Desktop/trt-llm-ds4/vllm-spark-0731` (nested git, not the parent TRT-LLM tree).

Read this, then [README.md](README.md). Field measurements live in
[maci0/dgx-spark-deepseek-v4-flash-0731](https://github.com/maci0/dgx-spark-deepseek-v4-flash-0731)
(`GOLDEN.md`). Failed envelope work lives in
[maci0/vllm-spark-nvfp4](https://github.com/maci0/vllm-spark-nvfp4).

## Status

**Paper recipe, not a measured serve.** The 0.28 image has never been built on
GB10. Dual-Spark mp flags were wired after the first push. Next owner starts at
step 1 below, on a Spark, not on a laptop.

| Goal | State |
|---|---|
| Pin `deepseek-ai/DeepSeek-V4-Flash-0731` | done (`assert_0731.py`) |
| Dockerfile FROM `v0.28.0rc2` + `b12x==1.2.6` via uv | written, unbuilt |
| `--linear-backend b12x --moe-backend b12x` | overlay written (#52016 in rc2, #52018 cherry-pick) |
| `--kv-cache-dtype fp8_ds_mla` | stock rc2 + DSpark k=5 locked |
| `--kv-cache-dtype nvfp4_ds_mla` | Python 584 B DSV4 page only. No CUDA writer. |
| DSpark `method=dspark` k=5 | locked in pins and `assert_stack.py` |
| 2-node RoCE serve | `--master-addr`, `--headless`, privileged IB. Untested. |
| GHCR image | not published |

HEAD on `main` after the serve-flag commit. Confirm with `git log -1 --oneline`.

## What the image is supposed to be

```
vllm/vllm-openai:v0.28.0rc2     # 74a6576, 2026-08-21 06:47 UTC. Need linux/arm64.
+ uv pip install b12x==1.2.6
+ patches/apply_overlays.py
    #52018 fused_moe/b12x.py + mxfp4 oracle + MoEBackend "b12x"
    B12xWarmupUnit + get_b12x_fused_moe
    #50645 mhc_pre_broadcast_tilelang DeepGEMM guard
    nvfp4_ds_mla CacheDType, 584/576/512 page ladder, indexer untouched
    MLA guard narrowed from startswith("nvfp4") to == "nvfp4"
+ patches/assert_image.py          # build dies if the dtype is a silent alias
```

Serve default: `./scripts/05-serve.sh fp8`
Same image, NVFP4 name: `./scripts/05-serve.sh nvfp4`

`--moe-backend b12x` is the MXFP4 expert path. `flashinfer_b12x` is a different
CuteDSL NVFP4-weight path. Do not swap them because the name looks similar.

## Do not do

- Mix `B12X_MLA_SPARSE` with `nvfp4_ds_mla`. That is the 432-vs-584 eugr overlay.
- Drop `maci0/vllm-spark-nvfp4` `eugr-nvfp4.patch` or the 191-line 0.27.1 envelope
  onto 0.28.
- Widen `DeepseekV4IndexerCache` to 584. FlashInfer then dies after a full load.
- Serve without DSpark, or with k other than 5. 0731 `n_predict=5`; 7 rejects, 10 crashes.
- `docker rm -f` alone. Always `./scripts/07-stop.sh` (`pkill -9 -f 'VLLM::'`).
- SSD KV offload. Illegal memory access on this model, every dtype.
- Preview checkpoint `DeepSeek-V4-Flash` (no `-0731`).
- Trust a green `assert_image.py` as "NVFP4 is real". It only proves the **name**
  and the **584 B page**. Packing is a CUDA writer.

## Next steps (do these in order)

### 1. Confirm linger and a quiet box

On **both** Sparks:

```
loginctl show-user "$USER" | grep Linger    # must be yes
# if not: loginctl enable-linger "$USER"
```

Stop other GPU tenants (`llama-server`, gpustack, leftover `VLLM::`). Clock cap
2200 MHz is already the production default on these boxes; leave it.

### 2. Arm64 base tag

```
docker manifest inspect vllm/vllm-openai:v0.28.0rc2 | grep -E 'architecture|variant'
```

Need `arm64`. If the tag is amd64-only, change `VLLM_RELEASE` in `configs/pin.env`
and the Dockerfile `FROM` to the aarch64 nightly (or NGC Spark vLLM) and record
the digest in this file. Do not build amd64 and expect it to run on GB10.

### 3. Build on one Spark

```
cd /home/maci/Desktop/trt-llm-ds4/vllm-spark-0731
./scripts/00-prereq.sh
./scripts/02-build-image.sh
```

If `apply_overlays.py` exits "missing needle" or "not unique", rc2 drifted.
Fix the needle in `patches/apply_overlays.py` against the installed
`python3 -c 'import vllm,os; print(os.path.dirname(vllm.__file__))'` tree.
`patches/v0.28/*.diff` are provenance; the applicator is what the Dockerfile runs.

If `uv pip install b12x==1.2.6` fails, check PyPI has an aarch64 wheel. Do not pip.

### 4. Serve fp8 + DSpark on 2 nodes (the real test)

Checkpoint: `~/models/ds4-flash-0731` (or `./scripts/01-download-0731.sh`).
Must pass `patches/assert_0731.py` (`dspark_block_size=5`).

`configs/nodes.env` from `nodes.env.example`. QSFP fabric IPs, not LAN.

**Worker first, then head:**

```
# spark2
NODE_RANK=1 VLLM_HOST_IP=<worker fabric> HEAD_IP=<head fabric> ./scripts/05-serve.sh fp8

# spark1
NODE_RANK=0 VLLM_HOST_IP=<head fabric> HEAD_IP=<head fabric> ./scripts/05-serve.sh fp8
```

Pass criteria (log + `./scripts/06-validate.sh`):

- `method=dspark` / spec k=5
- `--moe-backend b12x` actually selected (`B12X_MXFP4_*`, not "not supported for MXFP4 MoE")
- mHC does not die in `tf32_hc_prenorm_gemm` / `hyperconnection.hpp`
- `GPU KV cache size: N tokens` and a short completion returns
- Dist init is not stuck on 1/2 clients (if it is: linger, leftover `VLLM::`, wrong `HEAD_IP`)

Record: image id, KV tokens, B/token, c1 and c6 tok/s on the existing `c5.py` harness.

If this step fails, stop. Do not debug nvfp4 on a stack that cannot serve fp8.

### 5. Same image, nvfp4 flag

```
./scripts/07-stop.sh    # both nodes
# then worker, then head:
./scripts/05-serve.sh nvfp4
```

Pass criteria:

- log: `Using DeepSeek V4 padded nvfp4_ds_mla KV cache format`
- no `Expected packed SM120 DSV4 swa_kv_cache head dim 584, got 512`
- no GLM 432 page-size assert
- **B/token vs the fp8 run from step 4.** If it is ~11k (fp8 was 11,317 on eugr),
  the flag is an envelope. If it is ~7.6k (anemll GOLDEN), a writer is present
  in this FlashInfer. Write the number in GOLDEN.md / here. Do not claim NVFP4
  from the log line alone.

### 6. Only if step 5 is envelope (expected)

Do **not** invent a 416 B layout. Options, pick one:

1. Stay on `fp8_ds_mla` + b12x + DSpark (this image) for the 0.28 path.
2. Keep serving GOLDEN anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
   (`./scripts/02-pull-image.sh golden`) when you need the 2.00M token pool.
3. Port the anemll DSV4 writer (17 sites under `models/deepseek_v4/`) onto this
   0.28 tree. That is a CUDA/FlashInfer job, not another Python CacheDType patch.

### 7. Optional follow-ups (after a healthy fp8 serve)

- Wire `#52018` MoE warmup units into rc2 `b12x_warmup.py` (file shape differs;
  first MoE shapes JIT at request time today).
- Confirm `--compilation-config max_cudagraph_capture_size=36` is accepted on rc2.
- `LONG_PREFILL_TOKEN_THRESHOLD=1024` is in the pin and not passed to `vllm serve`.
  Add it if chunked-prefill starves decode at c5/c10 (see GOLDEN / arena notes).
- Shared-expert 0731 loader (tonyd2wild Patch 4) if 12 tensors drop at load.
- Publish `ghcr.io/maci0/vllm-spark-0731:v0.28.0rc2-b12x` once step 4 is measured.

## Related pins (do not mix)

| Pin | When to use |
|---|---|
| `configs/pin.env` | this 0.28 image, fp8 KV, b12x, DSpark |
| `configs/pin.nvfp4.env` | same image, nvfp4 **name** |
| `configs/pin.golden.env` | anemll 0.1.1, measured real NVFP4 |
| `configs/pin.eugr-b12x.env` | eugr nightly, `B12X_MLA_SPARSE` + fp8 only |

## Cluster gotchas that already cost time

- `Linger=no` + SSH-started worker: systemd deletes POSIX semaphores, head hangs
  on the next collective, looks like a b12x deadlock. Worker has no `Worker_TP`.
- Other GPU containers on spark2 (llama.cpp, gpustack) steal UMA and wedge NCCL.
- Fabric IP must be the QSFP (`enp1s0f0np0` / `enp1s0f1np1`), GID index 3 (RoCE v2).
- `HEAD_IP` unused used to default `--master-addr 127.0.0.1`. Fixed in 05-serve.
- Host-owned JIT caches from root docker runs break later uid-mapped containers.

## Done when

Step 4 has a measured fp8 + DSpark + b12x serve on 2x Spark, and step 5 has a
B/token number next to it. Until both exist, this repo is a patchset, not a
deployment.
