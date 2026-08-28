[← Index](00-index.md) · [Glossary](glossary.md)

# Deployment & Images

> **Scope:** Images and how to run them — lineages, engine selection, recipes (EXL3, Qwen, MiniMax), build/serve/stop, two-Spark topology.

## Lineages Measured on This Cluster (One Harness, 2026-08-22)

| Image | vLLM | KV | KV Pool | c1/c5/c6 tok/s |
|-------|------|-----|---------|----------------|
| anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | 0.25.2 | nvfp4 (7650 B/t) | **2.00M** | 51.4 / 126.2 / 157.9 |
| eugr `dgx-vllm-eugr-nightly-b12x:2026081903` | 0.27.x | fp8 (11317) | 1.77M | 54.3 / 127.4 / ~109 |
| tonyd2wild `dspark-nvfp4-stage-c` | 0.21.1rc1 | NVFP4 (~11900) | 1.44M | 56.1 / 116.0 / 141.1 |
| eugr `spark-vllm-b12x` (≤512K, spec off) | main | FP8 UE8M0 | — | ~326 tok/s @ c48 |
| **main-b12x (this repo)** | main `e25c586b9` | nvfp4_ds_mla (584 B) | 97,737 | ~25.8 / — / 172 @ c32 |

### Golden re-measured 2026-08-24 (numbers in [05-performance.md](05-performance.md))

Deployed stock on this cluster and validated (see HANDOFF Status → Golden).
Measured numbers live in the performance chapter: c1 **65.2** / c6 **216.8**,
KV **2,047,170 tokens** @ 7,650 B/token — 2.5× our c1, plateau past c6 by
design (`max_num_seqs=6`).

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
- **DeepGEMM**: nv_dev commit `a6b593d` (pinned back 2026-08-25 — `8b1392b` regressed SM12x pure-FP8 linear; see [09-golden-deepgemm.md](09-golden-deepgemm.md))
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
| `fp8` | `configs/pin.env` | `vllm-spark-0731:v0.28.0rc2-b12x` | `FLASHINFER_MLA_SPARSE_DSV4` | — / `b12x` | `fp8_ds_mla` | Legacy fallback (576 B); linear backend unset in pin |
| `nvfp4` | `configs/pin.nvfp4.env` | `vllm-spark-0731:v0.28.0rc2-b12x` | `FLASHINFER_MLA_SPARSE_DSV4` | `b12x` / `b12x` | `nvfp4_ds_mla` | Overlay fallback (584 B) |
| `eugr` | `configs/pin.eugr-b12x.env` | `dgx-vllm-eugr-nightly-b12x:2026081903` | `B12X_MLA_SPARSE` | `b12x` / `b12x` | `fp8_ds_mla` | Upstream comparison |
| `main` | `configs/pin.main.env` | `vllm-spark-0731:main-b12x` | `B12X_MLA_SPARSE` | `b12x` / `b12x` | `nvfp4_ds_mla` | **Live Production** (matched-main `e25c586b9`) |
| `golden` | `configs/pin.golden.env` (or sparkrun `anemll-nvfp4.yaml`) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | unset (image default; v0.25.2 has no `FLASHINFER_MLA_SPARSE_DSV4`) | — / `flashinfer_b12x` | `nvfp4_ds_mla` (real NVFP4 writer) | Reference benchmark (2.05M pool) |

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

## Engine Selection (vLLM / SGLang / llama.cpp / SparkInfer)

Purpose-first selection distilled from `book/04-1` + `book/04-2`: find an
engine that can run the *same* model and workload before hunting "the fastest
engine". A checkpoint behaves differently under each engine — supported
features and measurements differ, so don't switch engines just because one is
newer; measure functional and speed baselines on the same model/prompt/context
first.


| Engine | Pick for | Weakness |
|--------|----------|----------|
| **vLLM** | OpenAI-compatible API, broad model/API ecosystem, official Spark playbook, TP path | GB10/SM121 version/fork/kernel dependencies; per-model parsers |
| **SGLang** | Qwen3.8 single/dual speculative decode (DFlash2/DSpark), prefix/radix cache, structured output, agent workloads | Docker memory cap + RDMA passthrough + sm_121 kernels must all align |
| **llama.cpp** | Quick GGUF tests, single-user experiments, simple endpoints | Complex multi-node, special MoE, tool parser need separate verification |
| **SparkInfer (+ EXL3)** | DeepSeek V4 Flash 0731 single-Spark long single session | Derived quant, dedicated runtime; quality/parser verification burden |

- Decision order (6 questions): (1) weight + quant in the engine's support
  table? (2) GB10/SM121 image/commit/kernel conditions public? (3) endpoint +
  tokenizer pass smoke? (4) does KV fit the needed context? (5) tool parser +
  structured output match the real request format? (6) memory/temperature
  stable over long runs? Any "no"/unknown → record `unknown`, don't advance.
- **[CLAIMED, forum 380257]** Qwen3.8 comparison on the same GB10: llama.cpp
  ~27 tok/s, vLLM NVFP4+MTP ~24.5, SGLang+NVFP4+DSpark ~34–38 tok/s — quote
  with the thread and its conditions, not as a universal ranking.
- **Never** put the vLLM BF16 baseline and the DeepSeek EXL3 3.0 bpw + DSpark
  result in the same ranking table — different quant/engine/KV/speculation
  paths. Title such pairs "configuration comparison", never "speed winner".
- "OpenAI-compatible" = serving `/v1/chat/completions` ≠ full OpenAI API
  parity. Verify separately: model name + endpoint path; streaming + usage
  fields; `temperature`/`max_tokens`/thinking handling; structured output +
  JSON schema; tool choice + arguments format; image input/vision adapter.
- Fix before comparing engines: hardware (Spark count, GB10/SM121, UMA, QSFP/
  switch), software (DGX OS, driver, CUDA, image/digest, engine commit), model
  (repo, revision, tokenizer revision), quant, KV (dtype, cache format, max
  context, memory fraction), speculation (method, draft model, draft tokens,
  acceptance), serving (TP/PP/DP, batch token limit, prefix cache, parser),
  workload (prompt/output tokens, thinking, tool loop, image). Any difference
  → mark the result title "conditions differ".

## Single-Spark DeepSeek-V4 EXL3 Recipe (MiaAI-Lab) — Verified Profile

The community one-Spark path for DeepSeek V4 Flash 0731
(`MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark`): **EXL3 3.0 bpw** weights (0xSero
`REAP-K216`, 216 of 256 experts kept, surviving tensors stored via Trellis,
weight ≈ 99.5 GiB) + **SparkInfer** + **DSpark K5/K64** draft + native
**NVFP4** KV, TP=1 on 128 GiB GB10. It is a derived-quant serving artifact —
**not** the official full-FP8/full-expert checkpoint.


**Profile record (verified conditions — the profile is only valid together):**

| Field | Value |
|-------|-------|
| Model / endpoint | `deepseek-v4-flash-0731`, served on `http://127.0.0.1:8888` (`/health`, `/v1/models`) |
| Parallelism | **TP=1**, single box (GB10/SM121) |
| Context | `MAX_MODEL_LEN=384000` (a *setting ceiling*, not a per-request guarantee) |
| KV | native NVFP4 (`kv-cache-dtype=nvfp4_ds_mla`, `VLLM_DSV4_PADDED_NVFP4=0`), attention/moe/linear backends `B12X_MLA_SPARSE`/`b12x`/`b12x` |
| Speculation | DSpark `MODE=dspark`, `DSPARK_TOKENS=5` (K5 draft) |
| Concurrency | `MAX_NUM_SEQS=1` (raising it shrinks the KV pool — c1/deep-context profile) |
| Memory | `GPU_MEMORY_UTILIZATION=0.94` (aggressive, experiment condition, not a general default) |
| Thinking | default **on**, `reasoning_effort=max`; benchmark requests override to `thinking=false` |
| Loading | InstantTensor TP1 coalesce, `load-format=instanttensor`, `VERIFY_MODEL_CHECKSUMS=1`, `KV_OFFLOAD_GB=0` |
| Pin | recipe commit `d1dc9e7` (upstream `main` `5ba18b7` differs only in one README label); image `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489…` |

**[MEASURED]** here (2026-08-21/22, one DGX Spark): first boot ≈1h10m incl.
~106 GB weight download + TP1 coalesce + CUDA graph capture (model load itself
≈47 s); KV pool 469,175 tokens in one boot (boot-dependent); model memory
95.39 GiB; non-streaming decode **mean 31.34 tok/s** (3 runs); C1 code-decode
median 41.358 tok/s (gate 35); cold prefill 985.377 tok/s — **failed** the
1,000 tok/s gate; `DS4_TEST_OK` smoke + one mock `lookup_weather` tool loop
passed. **The claimed 44–47 tok/s was not reproduced here.**
**[CLAIMED]** (recipe README): structured decode 44–47 tok/s, ~439,622-token
KV pool, 370,104-token needle exact recall, initial prefill ~1,024 tok/s.
Numbers belong to the stated conditions — never scale 44–47 or the needle
result to multi-user service or general long-doc quality.

**Reproduction start**: `git clone …` → `git log -1 --oneline` → `free -h`,
`df -h .` → `./start.sh compose-gen` → `./start.sh` → `curl -sS http://127.0.0.1:8888/health`
+ `/v1/models`. Use the compose the repo
generates; don't hand-edit `start.sh`-generated configs. Record weight ·
bpw · KV dtype · context · draft model · single stream · prompt tokens ·
output tok/s · exact recall; missing any field → label the result "case with
missing conditions", not "DeepSeek speed".

**Not proven by the recipe**: full-expert/original-FP8 quality equivalence;
c4/c8 multi-stream performance; general long-document retrieval/reasoning;
long agent loops + tool-call recovery.

## Qwen3.8 & MiniMax Recipes (concise)


- **Qwen3.8-27B** (dense 27B, native 262K context + YaRN 1M ceiling, thinking
  on by default, Apache-2.0, **has a vision encoder** in the original). It is
  built on the Qwen3.5 architecture — `Qwen3_5ForConditionalGeneration` in
  logs is a lineage artifact, not a wrong load. **[MEASURED]** BF16 vLLM
  baseline (this repo, derivative `OBLITERATUS/Qwen3.8-27B-OBLITERATED`,
  `max-model-len 32768`, util 0.50, `qwen3` reasoning parser, `qwen3_xml`
  tool parser — smoke-test setting, NOT an optimal-speed recommendation).
  **[CLAIMED]** optimized paths: SGLang NVFP4 + DFlash2/DSpark ~34–38 tok/s;
  vLLM NVFP4+MTP ~20–30; llama.cpp Q4_K_M ~11.5–11.8. The derivative modifies
  refusal behavior — original-card quality numbers do not apply. Runner +
  smoke harness live in the archive (`scripts/run-qwen38-vllm.sh`,
  `tests/qwen38_smoke.py`); record reached state `loaded`/`generates`/
  `serves`/`tool-tested` separately.
- **MiniMax M2.7** (230B total / 10B active MoE, 256 experts / 8 active,
  204,800 context, text-only; license = NVIDIA + MiniMax **non-commercial** —
  verify before commercial use). No reliable single-Spark recipe confirmed;
  **[CLAIMED]** 2× ASUS GX10 NVFP4: ~196K context, `tg128` ~24.3 tok/s,
  `pp2048` ~2,074 tok/s prefill. The card's SGLang example is B200 TP=8 — do
  not copy to Sparks.
- **MiniMax M3** (428B / 23B active, 1,048,576 max context, DSpark path).
  **[CLAIMED]** Spark cases: 2× llama.cpp RPC UD-IQ4_XS ~10.7 tok/s @ 65K
  context; 4× custom vLLM c1 ~9–10 tok/s; 4× `MiniMax-M3-NVFP4` ~31 tok/s with
  1M KV profile + native-vision/tool claims (all condition-embedded — do not
  compare directly to Qwen/DeepSeek). 1 node is not a default choice; 4+ is
  the service-candidate tier.
- **MiniMax-H3** (33B audio+video Omni-DiT via ComfyUI — **not a language
  model**; measure s/step, never tok/s). Single-GB10 `sm_121` recipe exists
  (aarch64, CUDA 13, ComfyUI 0.30.1, ~41 GB weights, ~45 GB disk, full-stack
  ~8.39 s/step claim). `sm_121` patches must not be applied to `sm_120`
  (RTX 5090); install script only runs on GB10.
- **Recommended default**: buy/operate Qwen3.8 or DeepSeek; add MiniMax only
  with multi-Spark hardware + an experiment budget.

## Two-Spark Topology & Preflight

Two Sparks are NOT "+128 GiB GPU memory" — node-to-node communication,
synchronization and KV splitting become new bottlenecks; link and NCCL must be
verified before running models.


- **Topology choice** (official playbooks): 2 nodes = QSFP/RoCE direct
  (less gear; hand-tune cable, interface names, MTU, RDMA); 3 = QSFP ring
  (3 cables, no switch — not always faster than 2; TP=3 divisibility breaks
  for many models, prefer PP=3/DP=3/2+1); 4+ = QSFP switch (port count AND
  real RDMA compatibility must be verified); plain Ethernet only for SSH/
  management — TP comm can fall back to socket.
- **Pre-connection parity checklist** (identical on both nodes): DGX OS,
  kernel, NVIDIA driver, CUDA, NCCL lineage; container image + digest;
  vLLM/SGLang commit; model/tokenizer revision; time, hostname, node rank;
  MTU, HCA, interface names; firewall/ports/SSH. Record per node:
  `hostnamectl`, `uname -a`, `nvidia-smi`, `ip -br addr`, `ip -br link`,
  `ethtool <qsfp-interface>`, `rdma link`, `ibstat` (errors suppressed).
- **Verification ladder** (strictly ordered): physical link up → IP/MTU/
  interface → RDMA device → NCCL communicator → all-reduce/bandwidth pass →
  small-model TP request → long context, concurrency, soak. If an earlier
  stage fails, do not record model numbers from later stages. `Link detected: yes`
  only means the physical link is up — NCCL logs must show `NET/IB`, not
  `NET/Socket`; in containers verify `/dev/infiniband` is passed through.
  **[CLAIMED]**: SGLang Docker without `/dev/infiniband` dropped GLM-4.7 FP8
  on 4× Spark from 8.2 → 25.1 tok/s (~2.5×) purely from enabling RDMA.
- **Socket-direct gotcha**: one physical QSFP port can appear as MULTIPLE
  Linux interfaces — don't bond or discard by interface name alone; inspect
  `phys_switch_id`/`phys_port_name`/`ibdev2netdev`. A link negotiating 200G is
  not high payload — validate link → IP → RDMA bandwidth → NCCL collective →
  model request as separate stages.
- **DeepSeek TP=2 profiles**: 256K (262,144 ctx; coding/general service,
  higher concurrency) vs 1M (1,048,576 ctx; long single documents, low
  concurrency). "1M" is a setting ceiling — check the KV pool and
  `max_num_seqs`; one long request or many short requests, not both.
- **UCX / memory under long load** **[CLAIMED, recipe-specific]**:
  `UCX_MEM_MMAP_HOOK_MODE=none`, `UCX_RCACHE_MAX_UNRELEASED=1024` are leak
  mitigations from one public reproduction, NOT an official fix — soak-test
  before/after and keep a rollback method.
- **2×2 pool (4 nodes)**: DeepSeek TP=2 supervisor + Qwen3.8 TP=2 worker are
  TWO independent TP=2 services with separate memory/fault domains — not a
  TP=4 model, no unified-memory merge. With only two Sparks, pick either
  DeepSeek TP=2 OR Qwen3.8 TP=2; running both as independent TP=2 services on
  the same two nodes is experimental.

## OpenAI-Endpoint Tool-Parser Notes

Agents depend on the endpoint contract more than the model name — fix
`/v1/models`, `/v1/chat/completions`, tool schema and error body first.


- Basic checks: `curl -fsS http://127.0.0.1:PORT/v1/models` and a
  chat-completions request with `"max_tokens": 64`.
- **Qwen3.8 vLLM path**: do NOT omit `--enable-auto-tool-choice` and
  `--tool-call-parser qwen3_xml`. Enabling the parser does not verify
  tool-call quality — inspect arguments JSON and function names separately.
  **[MEASURED]** gotcha: a tool request on the local Qwen3.8 smoke server
  without a parser returned **HTTP 400** — a serving-configuration gap, not a
  model-ability gap. Tool parser = (1) server accepts the schema, (2) model
  emits valid arguments, (3) client returns results as a `tool` message —
  test each part; never bolt the parser onto the functional-test server
  (separate port).
- **Four operations criteria** so a model swap doesn't become an agent-wide
  outage: retry count when the endpoint dies; timeout + context-exceeded
  response handling; separation of the thinking field from the final answer;
  how tool results are fed into the next turn. Pin in client code/config:
  `base_url · model · timeout · max_output_tokens · retry policy` and
  `tool schema · allowed tools · log redaction · health check`; add request
  IDs + idempotency so auto-retry never re-executes the same tool when the
  server is down.
- **Model swap procedure**: send the same prompt to both endpoints; compare
  HTTP status, JSON schema and finish reason BEFORE comparing answer quality.
  Retry policy: timeout → limited retry allowed; wrong tool arguments and
  permission errors → return to the agent immediately, never retry
  unconditionally.

---


## External serving recipes (linked repos, 2026-08-26)

The recipe landscape these images belong to (details + links in [REFERENCES.md](../../REFERENCES.md) → GitHub repositories & recipes):

| Recipe | Stack | Key facts [RECIPE] |
|---|---|---|
| MiaAI 2x (anemll base) | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, official 0731 pinned `9e165c30…`, TP=2 | `nvfp4_ds_mla` block 256, util 0.835 → **2,493,464-token KV**; DSpark k=5, capture 36; single chat ~62–83 tok/s, 6-chat ~160–190 agg; thinking=max can emit ~50k reasoning chars (size `max_tokens` big) |
| MiaAI one-Spark EXL3 | SparkInfer/b12x `272a84bd`, EXL3 3.0 bpw REAP-K216 | TP=1, 384K ctx, `nvfp4_ds_mla` (432 B native records → 439,622 pool @ util 0.94); decode 44–47 tok/s; host needs ≥114.3 GiB free RAM; `MAX_NUM_BATCHED_TOKENS=8224` load-bearing; ABLATE opt-in refusal-direction ablation |
| hazyumps fork | jasl/vllm fork (PR #41834 base), NCCL 2.30.4 LD_PRELOAD | GA (156–167 GB, 48 shards) not drop-in over beta; TP=2+EP, **RDMA passthrough mandatory** (else 12→30+ tok/s); 384K, fp8 KV 1,187,206; ~47–60 tok/s single, prefill ~1.6–1.8k tok/s |
| Qwen3.8-27B | SGLang DFlash2, NVFP4 (`RadixArk`), draft `z-lab` | 78.6 tok/s single (draft 16), 480.7 agg (16 streams); draft 6→16 = +28% single-stream; mamba-slot concurrency model (5 slots/req) |
| MiniMax M2.7 NVFP4 / M3 DSpark | NVIDIA-packaged HF cards | M2.7: 230B/10B, 204,800 ctx, R&D-only, Blackwell-only; M3: 428B/23B, 1M ctx, **block-size 128 required**, DSpark k=8, commercial |
| Nemotron 3 Super 120B / 3.5 Lightning 30B | NVIDIA cards, `vllm/vllm-openai:v0.27.1` recipes | Both run on **1× DGX Spark** (Marlin W4A16 — no FP4 tensor-core MoE on GB10); Lightning: `--moe-backend marlin --kv-cache-dtype fp8`, DSpark draft (3 tokens on Spark), 1M ctx |
| nologik vLLM for GB10 | community image (`avarok/vllm-dgx-spark:v11`) | Measured eager ~42 → CUDA-graphs **~66–67 tok/s** (+60%); `VLLM_FLASHINFER_MOE_BACKEND=latency` required on SM12.1 |

### NVIDIA playbooks & NGC containers (2026-08-26)

Sources: fetched playbook READMEs (github.com/NVIDIA/dgx-spark-playbooks: vllm, sglang, llama-cpp, nvfp4-quantization, nccl, connect-two/three-sparks, multi-sparks-through-switch, performance guide) + NGC container tags ([REFERENCES.md](../../REFERENCES.md) → GitHub recipes / NVIDIA official).

- **vLLM playbook [RECIPE]**: containers `nvcr.io/nvidia/vllm:26.05.post1-py3` (latest) / `26.05-py3` / `26.02-py3`; tested matrix incl. Nemotron-3-Super-120B NVFP4, GPT-OSS-20B/120B MXFP4, Llama-3.1-8B & 3.3-70B NVFP4/FP8, Qwen3 8/14/32B FP8+NVFP4, Qwen2.5-VL-7B NVFP4, Nemotron-3-Nano BF16/FP8. Agent-ready Qwen3.6-35B-A3B-NVFP4 flags: `--kv-cache-dtype fp8 --attention-backend flashinfer --moe-backend marlin --gpu-memory-utilization 0.4 --max-model-len 262144 --max-num-seqs 4 --enable-chunked-prefill --enable-prefix-caching --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`.
- **2-node vLLM cluster [RECIPE]**: `run_cluster.sh` from vLLM (pinned `51c1ee9b`); **26.05-py3 ships without Ray** — `pip install ray[default]>=2.9` first; env `VLLM_HOST_IP` / `UCX_NET_DEVICES` / `NCCL_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` (+ `RAY_memory_monitor_refresh_ms=0`); high-speed iface `enp1s0f1np1`. 405B AWQ-INT4 on 2 Sparks only at `--max-model-len 64 --max-num-seqs 1 --max-num-batched-tokens 64` (insufficient headroom); UMA pressure fix `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.
- **SGLang / llama.cpp / NVFP4-quant playbooks [RECIPE]**: SGLang `lmsysorg/sglang:latest-cu130`, port 30000, `--tp 1 --attention-backend flashinfer --mem-fraction-static 0.75`; NVFP4 models need `--quantization modelopt_fp4`. llama.cpp build `-DCMAKE_CUDA_ARCHITECTURES=121a-real` + `-DGGML_CUDA=ON` (~5–10 min); MTP spec-decode `--spec-type draft-mtp --spec-draft-n-max 3`. NVFP4 quant recipes `huggingface/qwen3_5_moe/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast` (recommended on Spark, same layout as `nvidia/Qwen3.6-35B-A3B-NVFP4`) + `general/ptq/nvfp4_experts_only-kv_fp8_cast`; Model-Optimizer branch 0.45.0, needs `transformers>=5.2`, runs inside the NGC vLLM container ≥26.05.
- **NCCL / multi-node wiring playbooks [RECIPE]**: build NCCL v2.30.7-1 with `NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`; nccl-tests need a **16 GB buffer to saturate 200 Gbps**; one QSFP cable already gives full bandwidth; 2-node netplan `40-cx7.yaml` — `enp1s0f1np1`/`enP2p1s0f1np1` on different subnets; 3-node ring P0↔P1 with 4 IPs/node (same physical CX-7 port across nodes); switch: ≥4 QSFP56-DD @ ≥200 G, single L2 bridge, `ethtool` must show 200000 Mb/s (auto-neg often lands at 100G — set 200G-baseCR4 manually).
- **NGC containers [OFFICIAL]**: TRT-LLM `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6` (`trtllm-bench` / `trtllm-serve`); PyTorch `25.12-py3` / `25.10-py3`; NGC catalog pages are JS-empty — container tags recovered from playbooks.
- **NemoClaw one-command install [OFFICIAL]**: `curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash` (Node.js + OpenShell + CLI + wizard); WebUI `http://127.0.0.1:18789/#token=<TOKEN>` — must use `127.0.0.1`, not `localhost` (origin check); `NEMOCLAW_AGENT=openclaw|hermes|langchain-deepagents-code`; `NEMOCLAW_PROVIDER=ollama|vllm|install-vllm|build|openrouter|openai|anthropic|gemini|routed|custom` (`NEMOCLAW_VLLM_PORT` default 8000); Express install auto-sets Ollama + Qwen3.6-35B.

### Newly-crawled community recipes (2026-08-26)

- **eugr/spark-vllm-docker [RECIPE]**: the community reference vLLM stack most other recipes pin against — Ray or native mp (default since 2026-07-01), 3-node mesh, autodiscovery, `--apply-vllm-pr`, earlyoom `-M 524288,102400 -s 100 -r 60`, `--load-format instanttensor`; 4×-Spark Qwen3.5-397B INT4 ~37 single / ~103 agg; **driver 580.x required — 590.x has a CUDAGraph capture deadlock on GB10**; sudden-shutdown workaround `sudo nvidia-smi -lgc 200,2150`.
- **MiaAI GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks [COMMUNITY]**: 3× Spark TP3, hybrid NVFP4-hot + AQLM-2bit-cold (~272 GB + 1 GB vision), fork `vllm-glm52-sm120` (TP3 head pad 64→66). Two KV paths: `nvfp4_ds_mla` 11 GiB pin → ~348–380k ctx / ~21 tok/s structured; `fp8_ds_mla` 12 GiB → ~235k ctx / ~25–26 tok/s. Vision tower needs `MM_ENCODER_TP_MODE=data` (16 heads ∤ 3). **Disable earlyoom on all nodes** (kills Ray workers at 11–12 GiB KV pins). Image `ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks:k12l1-vision`.
- **mark-ramsey-ri vllm-dgx-spark / sglang-dgx-spark [RECIPE]**: 1-to-N cluster scripts; vLLM `nvcr.io/nvidia/vllm:26.04-py3` (vLLM 0.19.0, CUDA 13.2.1) + Ray 2.55.1 pip-installed at startup (26.04 no longer ships Ray), 41 model presets, GPT-OSS-120B 50–100 tok/s out / 400–700 batch; SGLang `lmsysorg/sglang:v0.5.10.post1-cu130` (multi-arch arm64), **cross-node TP needs `--enable-dp-attention`** (FlashInfer AllReduce Fusion uses CUDA IPC and doesn't cross nodes).
- **ArgentAIOS dgx-spark-cluster [RECIPE]**: 2-node direct-attach for distributed training + EXO + NVMe-TCP storage: 93.5% DDP efficiency, 1.87× epoch speedup, NCCL ~23–24 GB/s; RDMA `safe` (CPU-staged 18–20 GB/s) vs `dmabuf` (zero-copy 22–23 GB/s, needs driver 580+/kernel 6.x); **nvidia-peermem fails on ARM64**; NCCL env `NCCL_IB_HCA=mlx5_0 NCCL_IB_GID_INDEX=3 NCCL_ALGO=Ring NCCL_IB_TIMEOUT=22`.
- **antirez/ds4 + DwarfStar [RECIPE]**: native V4-Flash engine (`make cuda-spark` for GB10), asymmetric 2-bit quants (routed IQ2_XXS/Q2_K, rest untouched); DSpark via separate ~5.6 GiB 0731 support GGUF (`--mtp … --dspark`, confidence gate 0.7); `--power N` throttles GPU for heat/noise; SSD streaming for >RAM models; **`--cuda-tensor-parallel` must NOT be used on Spark** (prefill-only speedup, L40S-oriented).
- **Y-Computer/recipes [RECIPE]**: llama.cpp DSpark-draft path on one Spark — `--spec-type draft-dspark --spec-draft-n-max 3` with an IQ3_M sidecar draft (7.95 GiB vs upstream 10.15 GiB Q8_0-named-but-mixed-MXFP4): 28.29 vs 16.93 tok/s target-only (1.67×).
- **FlyCockpit DeepSeek-V4-Vision-2x [RECIPE]**: MoonViT-style vision tower (~865 MB + adapter) grafted onto FP8 0731 backbone (~167 GB — needs 2× Spark TP), API :8899, cold start ~6 min; **0731-only** — old-line NVFP4 produces garbage (different embedding space).
- **joeynyc MiniMax-H3-2x [RECIPE]**: H3 FL2VA video across 2 Sparks (Ray + Ulysses SP over RoCE) ~2.3× faster client-side (68.8 vs 155 s); H3 Community License excludes US/EU/UK/KR.

## Related Docs

- [00-index.md](00-index.md) — Quick links
- [01-hardware.md](01-hardware.md) — Cluster setup
- [03-kernels-attention.md](03-kernels-attention.md) — Backend differences
- [04-quantization-kv.md](04-quantization-kv.md) — KV dtype details
- [05-performance.md](05-performance.md) — Expected numbers
- [07-gotchas.md](07-gotchas.md) — Common failure modes
- [10-operations-agents.md](10-operations-agents.md) — Agent endpoint contract, role split, failure-runbook summary

### Raw evidence (field notes)

- [`../field-notes/dgx-spark/PRODUCTION.md`](../field-notes/dgx-spark/PRODUCTION.md) — the shipped eugr fp8 prod config, every value measured
- [`../field-notes/dgx-spark/EUGR_B12X_PROD.md`](../field-notes/dgx-spark/EUGR_B12X_PROD.md) — eugr b12x production path and why nvfp4_ds_mla is closed there
- [`../field-notes/nvfp4/EUGR_NVFP4.md`](../field-notes/nvfp4/EUGR_NVFP4.md) — 89-line NVFP4-on-eugr experiment (works, not competitive)

---

**[← Prev](05-performance.md) · [Glossary](glossary.md) · [Next](07-gotchas.md) →**
