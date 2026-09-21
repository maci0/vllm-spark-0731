# DeepSeek-V4-Flash 0731 — DSpark speculative decoding on 2x DGX Spark (B12X path)

Companion to `DEEPSEEK_V4_2NODE_SERVE.md`. That runbook serves our **NVFP4 abliterated** model at
**1M context, no spec**, on stock `vllm/vllm-openai`. This one gets **DSpark speculative decoding
working** — which stock vLLM cannot do on GB10 — using eugr's **B12X** image and an **FP8** checkpoint.

**Verified working 2026-08-04**, 2x GB10, TP=2, DSpark k=5, FULL_AND_PIECEWISE cudagraphs:
- `deepseek-ai/DeepSeek-V4-Flash-0731` (FP8) — dspark graphs captured, coherent, mean accept len 2.80.
- **`apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8`** — **abliterated + spec both confirmed**:
  answers refusal-adjacent prompts (no refusal), coherent, DSpark mean acceptance length 2.56
  (per-position 0.69/0.46/0.20/0.13/0.09), ~2.5x decode. **This is the abliterated+spec target.**

## What's missing from upstream `vllm/vllm-openai` (the short answer)

Nothing you can patch in. Stock `vllm-openai:latest` (0.26.0) **and** `:nightly` (main) both lack:
1. **PR #41834** — SM12x sparse-MLA decode + DSpark. **Unmerged into main**, so absent from every
   stock tag. Without it, dspark dies at `sparse_mla_sm120: num_tokens>64`.
2. **The sparkinfer (ex-b12x) kernel library** — compiled CuTe-DSL SM120/121 kernels (B12X MoE,
   MLA-sparse attention, MHC, FP8 GEMM) + the `B12X_MLA_SPARSE` attention backend. Lives in the
   `local-inference-lab/vllm` fork, not main.
3. **Patched FlashInfer (PR 3817) + DeepGEMM nv_dev branch + torch 2.12** — all source-built.

These are a **built image**, not a diff: the version gap from 0.26.0 to Aug-`main` plus a compiled
kernel package. That's why eugr ships a 23GB prebuilt (`eugr/spark-vllm-b12x:latest`) instead of a
patchset, and why "just apply the 2 patches to 0.26.0" is not viable — #41834 is written against a
much newer main than 0.26.0. **Pulling eugr's image is the shortcut.**

## Performance & operating modes (abliterated FP8, 2x GB10, measured 2026-08-04)

Spec and high-concurrency throughput are **mutually exclusive** — pick one:

| Mode | Flags | single-stream | peak aggregate | Use for |
|---|---|---|---|---|
| **Latency** | spec ON, `--max-num-seqs 8` | **37 tok/s** | 106 @ c16 | interactive, agents, ExploitGym |
| **Throughput** | spec OFF, `--max-num-seqs 48` | 28 tok/s | **298 @ c48** | batch, many users |
| ~~spec ON + seqs 48~~ | — | 9.8 ❌ | 299 | never (worst of both) |

- DSpark: mean acceptance length ~2.6-2.8/depth 5, ~35% avg per-draft. TTFT ~300ms.
- **NVFP4 is eager-only** (can't compile -> no cudagraphs, no B12X sparse attn, no spec):
  single-stream 18.5, peak 215. Slower than FP8 despite 4-bit — **compilability beats bit-width here**.
  NVFP4's only edge is a smaller memory footprint (more KV/context).
- Single-stream is LPDDR5X-bandwidth-bound; ~37 tok/s is near the ceiling for a 164B-active MoE.

## Why stock vLLM can't do spec (and NVFP4 can't at all)

Established by reproduction across many attempts:

1. **PR #41834** (SM12x sparse-MLA decode + DSpark) is **unmerged into vLLM main** → absent from
   `vllm-openai:latest` (0.26.0) AND `:nightly`. Stock dspark dies at
   `sparse_mla_sm120_paged_attention: num_tokens > 64 (got 7)`.
2. **Spec needs CUDA graphs**, not `--enforce-eager`: dspark captures the draft+verify in one graph.
   Eager can't route the small spec-verify batch to the decode kernel.
3. **NVFP4 can't do spec at all — even on the B12X image.** vLLM won't `torch.compile` the ModelOpt
   mixed-NVFP4 layout (`torch.compile is turned on, but the model does not support it`), so the
   compiled DSpark runner never sets `block_tables` → `AttributeError`. Independent of MoE backend
   and `swiglu_limit` (tested both). This is why **every NVFP4 card ships eager + no spec**, and why
   our `sakamakismile` NVFP4 (synthetic `INPUT_AMAX=6.0` scales) fails. FP8 compiles → spec works.

The enablers (unmerged PR #41834 + the sparkinfer/B12X CuTe-DSL SM120/121 kernels + FlashInfer PR
3817 + DeepGEMM nv_dev + torch 2.12) are a **built image**, not a patchset — hence eugr's prebuilt.

## Image (pull, ~23GB, both nodes)

```bash
for ip in 211 212; do ssh maci@192.168.0.$ip 'docker pull eugr/spark-vllm-b12x:latest'; done
```
Ships **real ray 2.56.1** + vLLM main (`0.1.dev19023+...d20260804`, a setuptools_scm dev string =
today's main, NOT "version 0.1") + B12X kernels compiled in (there is no importable `b12x` module;
the kernels live inside vllm). No beam needed.

## Model (FP8 required; NVFP4 won't compile)

- **`deepseek-ai/DeepSeek-V4-Flash-0731`** (167GB) — official, non-abliterated. Proven.
- **`apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8`** (167GB) — abliterated the *right* way:
  rank-1 refusal projection on attention residual-writers (`o_proj`), layers 10-42 + MTP stages,
  native FP8 requant (real scales). Being FP8, it compiles → **abliterated + spec**.

Each rank loads local shards; download the full model on **both** nodes.

## Cluster (real ray, our RoCE; NOT eugr's launch-cluster tooling)

```bash
RDMA="--cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband"
NCCL="-e NCCL_DEBUG=WARN -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
      -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_GID_INDEX=3"
UCX="-e UCX_MEM_MMAP_HOOK_MODE=none -e UCX_RCACHE_MAX_UNRELEASED=1024"
# B12X switches (mandatory: the b12x deepseek_v4 model + MHC TileLang need these, else torch.compile
# trips on mhc_pre_broadcast_tilelang):
B12X="-e CUTE_DSL_ARCH=sm_121a -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=-1 -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e B12X_MLA_SM120_UNIFIED=1 -e B12X_MOE_FORCE_A8=1"
COMMON="--network host --gpus all --ipc host --shm-size 16g $RDMA \
  -v /home/maci/beam:/opt/beam:ro -v /home/maci/.cache/huggingface:/root/.cache/huggingface $NCCL $UCX $B12X"

# head (real ray; --num-gpus 1 forces GB10 GPU detection)
ssh maci@192.168.0.211 "docker rm -f ds4; docker run -d --name ds4 -e VLLM_HOST_IP=10.0.1.1 $COMMON \
  eugr/spark-vllm-b12x:latest ray start --head --port 6379 --num-gpus 1 --block"
# worker
ssh maci@192.168.0.212 "docker rm -f ds4; docker run -d --name ds4 -e VLLM_HOST_IP=10.0.1.2 $COMMON \
  eugr/spark-vllm-b12x:latest ray start --address 10.0.1.1:6379 --num-gpus 1 --block"
ssh maci@192.168.0.211 'docker exec ds4 ray status'   # -> 2 GPUs
```

## Serve (eugr 0731 B12X recipe + DSpark)

```bash
#!/bin/bash   # /home/maci/beam/serve.sh
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --distributed-executor-backend ray --tensor-parallel-size 2 \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --gpu-memory-utilization 0.85 --max-model-len 262144 \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --moe-backend b12x --linear-backend b12x --attention-backend B12X_MLA_SPARSE \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}' \
  --load-format safetensors \
  --host 0.0.0.0 --port 8000
```
```bash
ssh maci@192.168.0.211 "docker exec -d ds4 bash -c 'bash /opt/beam/serve.sh > /tmp/vllm.log 2>&1'"
# healthy log: "DeepSeek V4 b12x mHC enabled" -> "DSpark draft model loaded" ->
#   "Capturing dspark CUDA graphs (FULL): 100%" -> "Application startup complete"
```

For the **abliterated** variant, swap the model line to
`apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8` — everything else identical.

## Gotchas (each cost a reload)

- **`swiglu_limit` -> use flashinfer MoE.** If the checkpoint sets `swiglu_limit` (our NVFP4 did,
  =10.0), `--moe-backend b12x` errors ("does not apply the SwiGLU clamp"); use
  `--moe-backend flashinfer_cutlass`. The official FP8 doesn't set it -> b12x MoE is fine.
- **Plain flags on the b12x image fail.** Without the `VLLM_USE_B12X_*` env set, the b12x
  `deepseek_v4` model trips `mhc_pre_broadcast_tilelang` under torch.compile. Image and B12X flags
  are a package deal.
- **Real ray, not beam.** This image ships ray; do NOT set `PYTHONPATH=/opt/beam/python` or
  `BEAM_NUM_GPUS`. Force `ray start --num-gpus 1` (GB10 device nodes aren't `/dev/nvidia*`).
- **Zombie workers after a kill.** `pkill -f "vllm serve"` misses `VLLM::Worker_TP`/`EngineCore`;
  recreate the container to free the GPU + reset ray (avoids `placement group needs more GPUs`).
- **Memory.** Both ranks ~112/122 GiB at util 0.85 with FP8 + draft + graphs; freed the worker's
  gpustack stack first. 15 GiB KV. Push `--max-model-len` toward 1M only if headroom allows.

## Getting abliterated + NVFP4 + spec (future)

No clean 0731 NVFP4 exists (nvidia's is the pre-0731 preview). To get all three:
1. Start from FP8 0731 (or `apetersson` abliterated FP8).
2. Quantize experts to NVFP4 with `nvidia-modelopt` using **REAL calibration** (true amax), never
   synthetic scales — synthetic is what broke `sakamakismile` (won't compile).
3. Keep attention/shared/router/MTP FP8 (nvidia layout). Result compiles -> spec works.
Simplest interim: `apetersson` **abliterated FP8** already gives abliterated + spec (just larger
than NVFP4).
