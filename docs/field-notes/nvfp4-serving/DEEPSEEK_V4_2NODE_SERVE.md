# DeepSeek-V4-Flash NVFP4 — 2-node high-performance serve on DGX Spark (via beam)

Serve `sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4` (~164GB NVFP4 MoE, 256 experts /
6 active, MLA + DeepSeek Sparse Attention, `deepseek_v4`) across **two GB10 DGX Sparks** over the
**200Gb RoCE** link at the **full 1M-token context**, with **expert-parallel + DeepEP**, MLA
(`fp8_ds_mla`), DSA, and NVFP4 experts — all on **stock `vllm/vllm-openai:latest`**. Speculative
decoding (DSpark/MTP) is **not** included: it needs a patched image on GB10, see gotcha #5.

One GB10 = ~122GB usable unified memory; ~78GB of weights load per node (TP=2), so it needs both.

Uses **[beam](https://github.com/maci0/beam)** (a ~1.5k-line Ray drop-in for vLLM distributed
inference) bind-mounted into the stock `vllm/vllm-openai` image — the stock image no longer ships
ray. beam carries only the control plane; NCCL/RoCE + DeepEP carry the tensors and expert routing.

> **Verified working 2026-08-04** on vLLM **0.26.0**. This model is bleeding-edge (`deepseek_v4` +
> UE8M0 microscaling fp8 + DSA + MTP); the recipe below reflects what actually loaded and served,
> not the theoretical ideal. Read the **Hard-won gotchas** section before deviating.

### Hard-won gotchas (each one cost a full reload; do not skip)

1. **Pin BOTH nodes to the same vLLM version.** Floating `:latest` pulled 0.24.0 on one node and
   0.26.0 on the other → the head crashed with `AttributeError: 'ModelConfig' has no attribute
   'model_class_overrides'` (a stale-`.pyc` leaking a newer method). **0.26.0 is required** anyway:
   0.24.0 lacks `dspark`, mis-handles UE8M0 in DeepGEMM, and its cutlass can't dispatch UE8M0.
   `docker pull` on both, confirm `docker exec ds4 python3 -c "import vllm;print(vllm.__version__)"`
   matches before serving.
2. **`VLLM_USE_DEEP_GEMM=1` is mandatory.** The fp8 weights are UE8M0 microscaling; CUTLASS
   `scaled_mm` cannot dispatch UE8M0 on sm_121 (GB10) → `RuntimeError: dispatch_scaled_mm,
   .../w8a8/cutlass/c3x/scaled_mm_helper.hpp:17`. DeepGEMM is the only UE8M0 GEMM path.
3. **all2all is a SERVE FLAG now, not an env.** vLLM 0.26 **removed `VLLM_ALL2ALL_BACKEND`**
   (and `VLLM_ATTENTION_BACKEND`); both log "Unknown vLLM environment variable" and are ignored.
   Use `--all2all-backend deepep_low_latency` on `vllm serve`. MLA is auto-selected for
   `deepseek_v4` (`fp8_ds_mla`), no env needed.
4. **Free the worker node's other GPU services first.** GB10 unified memory means GPU allocations
   come out of the same 121 GiB as system RAM. With gpustack + llama-server (~13 GiB) still running
   on the worker, the 78 GiB of weights + warmup spike pushed it to 109/121 GiB and the kernel
   **SIGKILL'd the worker rank (exit 137) mid-warmup** — which looks like a cluster hang from the
   head. Stop them: `docker stop gpustack-worker docker-llama-gen-1 docker-llama-embed-1 docker-app-1`.
5. **MTP / DSpark speculative CANNOT run on stock vllm-openai** (researched + reproduced, not a
   config miss). `--speculative-config '{"method":"dspark","num_speculative_tokens":7}'` loads the
   draft head, then dies in warmup: the DSA sparse-MLA Blackwell kernel asserts
   `sparse_mla_sm120_paged_attention: num_tokens > 64 (7 vs 64)` — the spec-verify batch (1-7 tokens)
   routes to the prefill kernel instead of `sparse_mla_sm120_decode_dsv4`. Two independent blockers,
   both **code-level, not knobs**:
   - The SM12x sparse-decode fix (`DeepseekV4FlashInferSM120Attention`, forces the compressed decode
     index contiguous) is **vLLM PR #41834 — still OPEN**, only in the `jasl/vllm` fork tags.
   - This NVFP4 checkpoint's **MTP draft experts are MXFP4** (group_size 32, e8m0 block scale) while
     the main experts are NVFP4 (group_size 16, e4m3); the draft must route to `Mxfp4MoEMethod` or
     the exponent bytes are destroyed. That routing fix is a **code patch** in custom images only
     (hikarioyama, tonyd2wild Stage-C). DSpark also needs CUDA graphs (bug #40969: those hang on
     GB10 with chunked prefill anyway).
   Every practitioner who got MTP working on GB10/SM120 used a **patched image**. To get MTP you must
   drop the stock constraint (options: `jasl/vllm` preview, hazyumps sm_121 fork, hikarioyama/
   tonyd2wild B12X images). On stock: **no spec-decode.** (+38% single-stream is the prize if you switch.)
6. **CUDA graphs: keep `--enforce-eager`.** DSA/MLA uses custom TileLang/flashinfer kernels;
   `FULL_AND_PIECEWISE` hangs after ~6 requests on GB10 (vLLM issue #40969). `PIECEWISE` is the only
   graph mode that survives, and it's needed for spec-decode — moot here since spec is off.
7. **KV dtype: `fp8` -> auto `fp8_ds_mla` (MLA-native).** Stock 0.26 also lists `nvfp4` and the
   `turboquant_*` asymmetric-KV dtypes, but **`nvfp4` KV is unsupported on the sparse-MLA path** and
   turboquant is untested with MLA. `fp8_ds_mla` is the proven one; the aggressive NVFP4 KV
   (`nvfp4_ds_mla`) that fits 1M with room is fork-only (tonyd2wild Stage-C).

## Topology

| Node | Role | mgmt IP | RoCE IP (fast) |
|---|---|---|---|
| spark1 | beam head + vLLM | 192.168.0.211 | **10.0.1.1** |
| spark2 | beam worker | 192.168.0.212 | **10.0.1.2** |

- **Fast fabric: RoCE v2, 200Gb/s (2X NDR)** on `enp1s0f1np1`; two cabled HCAs **`rocep1s0f1`,
  `roceP2p1s0f1`**; subnet `10.0.1.0/24`. `ztklhw5udj` (10.10.10.x) is a 10G tun overlay — never NCCL.
- GB10 = Grace-Blackwell sm_121, 128GB unified, driver 580.159.03. vLLM supports `DeepseekV4ForCausalLM`.

## 0. Model download (both nodes, ~hours)

Each rank loads its shards from **local** disk (no shared FS), so both nodes need the full 176GB.

```bash
# on EACH node
docker run -d --name dl-ds4 -v /home/maci/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 --entrypoint bash vllm/vllm-openai:latest -c \
  'pip install -q hf_transfer; python3 -c "from huggingface_hub import snapshot_download; \
   snapshot_download(\"sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4\")"'
```

## 1. Deploy beam to both nodes

```bash
git clone https://github.com/maci0/beam /tmp/beam
for ip in 211 212; do
  ssh maci@192.168.0.$ip 'mkdir -p ~/beam'
  rsync -a /tmp/beam/python/   maci@192.168.0.$ip:~/beam/python/
  rsync -a /tmp/beam/examples/ maci@192.168.0.$ip:~/beam/examples/
done
```

beam self-bootstraps in the container: `PYTHONPATH=/opt/beam/python` makes `import ray` resolve to
the shim; on `ray start` it writes `ray`/`beam` launchers into `/usr/local/bin`.

## 2. Bring up the cluster (head + worker)

First **free the worker node's GPU services** (gotcha #4) and confirm both nodes are on the same
vLLM (gotcha #1):

```bash
ssh maci@192.168.0.212 'docker stop gpustack-worker docker-llama-gen-1 docker-llama-embed-1 docker-app-1'
for ip in 211 212; do ssh maci@192.168.0.$ip \
  'docker pull vllm/vllm-openai:latest; docker run --rm vllm/vllm-openai:latest python3 -c "import vllm;print(vllm.__version__)"'; done
# both must print 0.26.0 (or matching)
```

The run args carry the **RDMA passthrough**, the **GB10 GPU override**, and **NCCL/RoCE tuning**.
Note: the all2all backend and MLA kernel are NOT env vars in 0.26 (gotcha #3) — all2all is a serve
flag, MLA is automatic:

```bash
RDMA="--cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband"
NCCL="-e NCCL_DEBUG=INFO -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
      -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
      -e NCCL_IB_GID_INDEX=3 -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=8"
COMMON="--network host --gpus all --ipc host --shm-size 16g $RDMA \
  -v /home/maci/beam:/opt/beam:ro -v /home/maci/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONPATH=/opt/beam/python -e BEAM_NUM_GPUS=1 $NCCL --entrypoint python3"

# --- head: spark1 (advertise the RoCE IP) ---
ssh maci@192.168.0.211 "docker rm -f ds4 2>/dev/null; docker run -d --name ds4 \
  -e VLLM_HOST_IP=10.0.1.1 -e BEAM_NODE_IP=10.0.1.1 $COMMON \
  vllm/vllm-openai:latest -m ray start --head --port 6379 --block"

# --- worker: spark2 ---
ssh maci@192.168.0.212 "docker rm -f ds4 2>/dev/null; docker run -d --name ds4 \
  -e VLLM_HOST_IP=10.0.1.2 -e BEAM_NODE_IP=10.0.1.2 $COMMON \
  vllm/vllm-openai:latest -m ray start --address 10.0.1.1:6379 --block"

ssh maci@192.168.0.211 'docker exec ds4 python3 -m ray status'   # -> 2 nodes, 2 GPUs
```

`--device /dev/infiniband` grants the device-cgroup permission for the RDMA verbs devices (a `-v`
mount exposes them but the cgroup still blocks — which is what made `--privileged` look necessary).
Without it NCCL logs `Unable to open device rocep*` and drops to TCP; with it:
`NCCL INFO NET/IB : Using rocep1s0f1/RoCE roceP2p1s0f1/RoCE`. Verify the RoCE-v2 GID once:
`docker run --rm --net host vllm/vllm-openai:latest show_gids | grep -i v2` (use that index, usually 3).

## 3. Serve (verified high-performance default)

Put this in `/home/maci/beam/serve.sh` (bind-mounted at `/opt/beam/serve.sh`) so it runs inside the
container with `import ray` resolving to beam:

```bash
#!/bin/bash
export VLLM_USE_DEEP_GEMM=1                     # mandatory: UE8M0 fp8 needs DeepGEMM (gotcha #2)
vllm serve sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4 \
  --served-model-name deepseek-v4-flash \
  --distributed-executor-backend ray --tensor-parallel-size 2 \
  --enable-expert-parallel --all2all-backend deepep_low_latency \
  --trust-remote-code \
  --reasoning-parser deepseek_v4 \
  --gpu-memory-utilization 0.82 --max-model-len 1048576 --enforce-eager \
  --kv-cache-dtype fp8 --block-size 256 \
  --enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 6 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000
```

**Verified: this serves the full 1M context** (`max_model_len 1048576`) on stock 0.26.0. It works
because 1M is native (config `max_position_embeddings=1048576` + YaRN factor 16 over 65536 — no
override needed) and **DSA sparse+SWA KV is sublinear in context**: at util 0.82 the load left
**18.3 GiB KV**, which the sparse structure stretches to a full 1M single request (a naive dense KV
would need hundreds of GiB). Both nodes sit ~110/121 GiB with ~7-12 GiB headroom — stable, but this
is the edge; do not push util past 0.82. For higher concurrency instead of max context, drop
`--max-model-len` (256K/512K) and raise `--max-num-seqs`.

Launch it (detached) on the head, then poll health (weight load + warmup ~13 min):

```bash
ssh maci@192.168.0.211 "docker exec -d ds4 bash -c 'bash /opt/beam/serve.sh > /tmp/vllm.log 2>&1'"
ssh maci@192.168.0.211 'curl -sf http://localhost:8000/health && echo UP'
```

Confirm the perf paths in `docker logs ds4` (all present in a healthy run):
`Expert parallelism is enabled ... Local/global number of experts: 128/256` (EP),
`Using DeepSeek's fp8_ds_mla KV cache format` (MLA), `FP8 indexer cache for Lightning Indexer` +
`DEEPSEEK_SPARSE_SWA` (DSA), `FLASHINFER_CUTLASS NvFp4 MoE backend` (NVFP4).

Why each knob:
- **`--enable-expert-parallel` + `--all2all-backend deepep_low_latency`** — shard the 256 experts
  128/node; route tokens with DeepSeek's RDMA-aware **DeepEP** low-latency all-to-all (best for
  interactive decode). `deepep_high_throughput` if prefill/batch-bound. This is a **serve flag** in
  0.26 (the old `VLLM_ALL2ALL_BACKEND` env is gone, gotcha #3).
- **MLA is automatic** for `deepseek_v4` (`fp8_ds_mla`); no attention-backend env. MLA's 1-KV-head
  compressed latent is the whole long-context KV win.
- **`--kv-cache-dtype fp8`** — the model auto-selects MLA-native `fp8_ds_mla`. DSA sparse+SWA KV is
  sublinear in context, so 1M fits in the 18.3 GiB the load leaves at util 0.82 (gotcha #7 for the
  other dtypes and why `nvfp4` KV is off-limits here).
- **`--gpu-memory-utilization 0.82`** — after freeing the worker's other services (gotcha #4), both
  ranks sit ~110/121 GiB with ~7-12 GiB headroom at 1M. 0.82 is the practical GB10 max (matches the
  NVIDIA-forum guidance); 0.74 was too low (KV 0.19 GiB short even at 8192), 0.85+ risks the warmup
  SIGKILL.
- **`--max-model-len 1048576` / `--max-num-seqs 6`** — the 1M profile: max context, low concurrency.
  For throughput instead, lower max-model-len (256K/512K) and raise max-num-seqs; the KV pool is
  shared, not `seqs x len` pre-reserved.
- **batching** (`--max-num-batched-tokens 8192`, chunked prefill, `--block-size 256`) — DSA makes
  long prefills cheap. Lower batched-tokens (512-2048) frees KV if you push concurrency at long ctx.
- **`--enable-prefix-caching`** — near-free KV reuse for shared prefixes (agents/system prompts).
- **`--enforce-eager`** — required (gotcha #6).
- **MTP / DSpark: impossible on stock** (gotcha #5) — needs the unmerged PR #41834 + an MXFP4
  draft-expert routing patch; switch to a patched image if you want it (+38% single-stream).

## Bring-up fallback (if EP/DeepEP/MLA won't init)

Strip to the minimum to isolate the failure, then re-add one at a time:
```bash
# plain TP, eager: drop --enable-expert-parallel --all2all-backend --enable-prefix-caching
#   keep VLLM_USE_DEEP_GEMM=1 (always needed for UE8M0)
```
- worker "hangs" mid-warmup -> it was SIGKILL'd for memory (gotcha #4). Free worker services; lower
  `--gpu-memory-utilization` (0.80 -> 0.76). Check `docker inspect ds4 --format '{{.State.ExitCode}}'`
  (137 = killed).
- DeepEP won't init -> `--all2all-backend allgather_reducescatter` (NCCL-only) or `naive`, or drop
  `--enable-expert-parallel`.
- KV-cache-too-small ValueError -> raise `--gpu-memory-utilization` or lower `--max-model-len` to the
  suggested value in the error.
- leftover `VLLM::Worker_TP`/`EngineCore` procs holding GPU after a kill -> `pkill -f "vllm serve"`
  misses them; recreate the container (`docker rm -f ds4` + rerun) for a clean ray + GPU.

## 4. Test

```bash
curl http://192.168.0.211:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4-flash",
  "messages":[{"role":"user","content":"Explain MoE routing in two sentences."}],
  "max_tokens":128}'
# benchmark: vllm bench serve ... (compare decode tok/s + TTFT); raise --max-model-len while it fits.
```

## Troubleshooting

- **NCCL on TCP not RoCE**: log must show `NET/IB : Using rocep1s0f1/RoCE`. `Unable to open device
  rocep*` -> `--device /dev/infiniband` missing. Picks the tun -> fix `NCCL_SOCKET_IFNAME`. Hang at
  init -> wrong `NCCL_IB_GID_INDEX` (check `show_gids | grep v2`).
- **beam sees 0 GPUs**: `BEAM_NUM_GPUS=1` missing (GB10 device nodes aren't `/dev/nvidia*`).
- **ray worker won't join**: `BEAM_NODE_IP`/`VLLM_HOST_IP` must be each node's own RoCE IP;
  `10.0.1.1:6379` reachable from spark2 (RoCE subnet, yes).

## Teardown

```bash
for ip in 211 212; do ssh maci@192.168.0.$ip 'docker rm -f ds4 dl-ds4'; done
# restore the worker node's GPU services stopped in step 2:
ssh maci@192.168.0.212 'docker start gpustack-worker docker-llama-gen-1 docker-llama-embed-1 docker-app-1'
```
