# Handoff: 2-Node vLLM for DeepSeek-V4-Flash-0731 on DGX Spark (GB10)

## Overview

Deploy `deepseek-ai/DeepSeek-V4-Flash-0731` across 2x DGX Spark nodes.

**Audit Artifact:** [outputs/vllm-spark-0731-docs-audit.md](outputs/vllm-spark-0731-docs-audit.md) — Comprehensive architecture & codebase audit ([Plan](outputs/.plans/vllm-spark-0731-docs.md)).  
**Knowledge Base:** [docs/knowledge/00-index.md](docs/knowledge/00-index.md) — Definitive 10-chapter reference guide.

**Live (2026-08-26, re-served):** `vllm-spark-0731:main-b12x` (matched vLLM
`v0.1.dev1+ge25c586b9.d20260823`, CUDA 13.3.1, torch 2.14 `12.1a`). b12x
linear + MoE + `B12X_MLA_SPARSE` (target and DSpark draft), `nvfp4_ds_mla`
584 B DSV4 envelope, DSpark k=5, `FULL_AND_PIECEWISE`, DSpark backbone FULL
(sample eager), util **0.8**, `MAX_NUM_SEQS=32` (was 8), capture 192,
TP=2 over RoCE. Pin: `scripts/05-serve.sh main`.
Build plan: [docs/PLAN-MAIN.md](docs/PLAN-MAIN.md).

**DeepGEMM fp8 scale story — FINAL RESOLUTION (2026-08-27):**
After full boot-free + E2E investigation, the einsum "misread" was a chain
of misdiagnoses. The truth, all measured on GB10 with the real shared `w1`
(F8_E4M3 + F8_E8M0) and a bf16 reference:
1. **Linear path (`fp8_gemm_nt`) with packed-UE8M0 scales: CORRECT**
   (mean_rel=0.000000). No fix needed.
2. **Einsum kernel with packed E8M0 scales + recipe `(1,1,128)`
   (stock upstream SM12x config): CORRECT** (mean_rel=0.000000). No fix
   needed — the kernel never misread the real scales.
3. The earlier "290% einsum error" was the **packer mantissa-leak** hit by
   synthetic FP32 scales with mantissa (upstream deepseek-ai/DeepGEMM
   **#337** fixes it; real checkpoint scales are E8M0 = zero mantissa).
4. The "~2³² wrong" was **our own dequant-fallback overlay** reading
   packed int32 as fp32 — the fallback was the actual E2E-garbage source.

**Final stack: `main-b12x-mn2`** = stock upstream einsum kernel path
(passthrough `fp8_einsum`, no fallback, no SM12x recipe override) + packed
scales. Verified E2E: France coherent, 9×8="72", `max_model_len=65536`,
KV 9.38 GiB. `pin.main-dg.env` points at mn2.
**PR fallout:** vLLM #53898 (fallback) and #53521 (recipe override) both
CLOSED as not-needed (analysis posted on each). The only genuine upstream
fix in this area is DeepGEMM #337 (packer mantissa mask). The mn image
(fallback variant) measured 46.3 tok/s vs mn2's 37.1 in one cold 4-parallel
run — inconclusive, worth a clean re-benchmark if einsum decode perf matters.
NOTE: the phase-1 `02-build-main.sh` rebuild does NOT apply the vLLM pr-*
backports — use the layered `main-b12x-mn2` image (or the overlay build).

**Debugging the deep_gemm path:** a6-wheelled image + `--linear-backend
deep_gemm` boots (health 200) but France is garbage — the a6 kernels do not
match vLLM main's scale producers (einsum proven: FP32 scales correct,
packed UE8M0 ~2³² wrong; `fp8_gemm_nt` suspected). Validate per-op with the
tiny 1:1 model `yujiepan/deepseek-v4-tiny-random` (downloaded on spark1,
`/home/maci/models/deepseek-v4-tiny-random`) — see
[docs/knowledge/09-golden-deepgemm.md](docs/knowledge/09-golden-deepgemm.md).

**Cluster state 2026-08-27 (RESTORED):** both nodes were power-cycled
(spark1 08-27 ~09:55, spark2 ~10:30); `/tmp` on both is cleared per-reboot
(re-sync the repo from the local checkout before booting — `tar czf - . |
ssh … tar xzf - -C /tmp/vllm-spark-0731`). Swap is now ACTIVE on both
(`/swap.img` 16G, systemd `swap.img.swap`, fstab entry present — the
earlier "no swap at boot" issue is fixed). **Boot hygiene learned this
session:** (a) a fresh image with no AOT/inductor cache + `AOT=1` can wedge
a host into a memory-thrash where sshd accepts TCP but never sends a banner
— boot `VLLM_USE_AOT_COMPILE=0` until caches warm; (b) if one node
reboots/dies mid-rendezvous, the peer can wedge the same way — cycle it;
(c) `gb10-clockcap` does not always auto-restart on spark1 — `docker start
gb10-clockcap` after a cycle. Verified stack (a6fix, TP=2, deep_gemm) is
left RUNNING for interactive testing; stop with `07-stop.sh` on both.

**Boot runbook (2026-08-28, Phase 4 — retry pattern so benchmark cycles
aren't wasted):** the 2-node NCCL rendezvous fails intermittently (~50%,
spark2 worker logs `Connection reset by peer` during init). Do NOT burn a
benchmark cycle on a half-booted stack:
1. Pre-flight: both nodes pingable; `docker images` on spark1 shows the
   target tag; repo synced to `/tmp/vllm-spark-0731` on spark1.
2. Clean slate on BOTH nodes: `docker rm -f vllm-ds4-0731` (plus
   `scripts/07-stop.sh` if a stack is up).
3. Boot worker FIRST (spark2), then head ~90s later (spark1):
   `VLLM_USE_AOT_COMPILE=0 nohup bash scripts/05-serve.sh main-dg`.
   `VLLM_USE_AOT_COMPILE=0` until the inductor/AOT caches are warm (a fresh
   image with AOT=1 can wedge the host in memory-thrash).
4. Wait for `/health` 200 on the head (~200 s), then verify BOTH worker and
   head logs converged (TP init succeeded — `NCCL` all-reduce microbench
   ~0.03-0.045 ms in the log, no `NET/IB: No device found`, no
   `Connection reset by peer`).
5. If the worker died mid-rendezvous: `docker rm -f` on BOTH nodes, wait
   30 s (let the SHM/NCCL segments drain — see `spark-launch.sh` shm_sweep),
   repeat from step 3. Do not restart only one side.
6. Benchmark only after the health gate passes; warmup compiles (mHC
   broadcast ~30-120 s, gumbel ~seconds) run during model load — the
   `Warming up ... finished in X.XX seconds` lines must appear BEFORE the
   first request, or the first measurement includes the JIT stall.

Re-served 2026-08-26 (spark2 worker then spark1 head): `/health` 200 in
~200s, France greedy coherent (`" Paris. The capital of Italy is Rome…"`),
c1 ~21.8 tok/s incl. TTFT (decode-only ~26, matches the 25.8 baseline).
Swap verified active on both nodes (see the swap note below).

France is green (`' Paris'` logprob -0.25..-0.26, n_tie=1, chat `Paris`).
Measured 2026-08-24 (France, temp 0, 128 tok): 1-way ~25.8, c8 ~95, c16 ~116,
**c32 ~172** tok/s (the max_num_seqs 8→32 lift). KV **97,737**.
FLASHINFER_MLA_SPARSE_DSV4 was A/B'd after the eidx-contiguity fix: ~parity
(c32 179, c1 22.6 — slightly worse single-stream), so `B12X_MLA_SPARSE`
stays live. The ~2× gap to the anemll/eugr images is a whole-stack
difference (older vLLM, real NVFP4 writer), not this image's config. Do not
mark the 1-way gap closed.

**o_proj (WO) decode bmm — ROOT CAUSE FOUND + FIXED (2026-08-28, commit
0d52159, pending rebuild/deploy):** `try_b12x_wo_proj` fell back to the slow
einsum on every decode call because `_cached_wo_a_bmm_weight` only handled
the 2D checkpoint layout `[G*R, D]`, while vLLM's DSV4 `wo_a` post-loads via
`deepgemm_post_process_fp8_weight_block` with `is_bmm=True` (set in
`attention.py`) into the **3D** local shard `[G, R, D]` (TP=2: `[4, 1024,
4096]`, scale `[4, 8, 32]`). The function misread rows/cols from the 3D
weight, expanded the scale to a garbage shape and threw
`size of tensor a (4096) must match size of tensor b (1024) at non-singleton
dimension 2` — the bmm path never ran. Fix handles both layouts (3D:
`transpose(1,2)` for `bmm(a[G,T,D], w[G,D,R])`; 2D: existing view). Also made
`_ensure_bmm_ws` require the group dim to match exactly (was `>=` — stale
8-group workspaces could be reused for 4-group calls). Expect ~8-10 ms/step
saved at c1 if the estimate holds; rebuild via
`scripts/ov-rebuild-rdma.sh`, verify `DBG wo_proj OK#N` + `b12x wo_proj bmm
ok` in the worker log, then re-bench.

**Phase 2 warmup — JIT gap found + FIXED in code (2026-08-28, commit
2e92fa5, pending rebuild/deploy):** upstream `deepseek_v4_mhc_warmup` only
warms the 3D per-layer mHC path; the first layer's
`mhc_pre_broadcast_tilelang` (2D residual + `fn_broadcast`, runs every
decode step) and the DSpark `gumbel_sample` triton kernels were never
warmed → first request after boot pays TileLang (~30-120 s) + DeepGEMM
per-M + triton compiles, the c16-collapse suspect. New
`vllm/model_executor/warmup/dsv4_warmup_ext.py` warms both (called from
`kernel_warmup` after the upstream warmup). Verify at boot: worker log
`Warming up DSv4 mHC broadcast ...` + `DSpark gumbel sampler kernels ...`
+ `finished in X.XX seconds` before health 200, then re-run the c8/c16/c24
concurrency sweep.

**Fallback:** `vllm-spark-0731:v0.28.0rc2-b12x` (v0.28.0rc2 Python on
v0.27.1 arm64 base). Same greedy string. Attention is FlashInfer DSV4,
not b12x. Pin: `scripts/05-serve.sh nvfp4`. Rest of this file is overlay
ops unless a section says main-b12x.

Greedy `"The capital of France is"` (`temperature=0`, `max_tokens=32`) is
coherent: `" Paris. The capital of Spain is Madrid. The capital of Italy is
Rome. ..."`. Chat answers `"Paris."`. Do not raise spark2 to util 0.85.
spark2 swap is now **enabled** (the `swap.img.swap` unit was masked
`-> /dev/null`; unmasked 2026-08-24 — it auto-enables at boot; fstab already
had the `sw` entry, the mask suppressed the generated unit, hence earlyoom
logging `swap total: 0 MiB` at boot). Verified 2026-08-26: both nodes have
`/swap.img` 16 GiB active (unit `generated` + active, ~835 MiB used), fstab
entry present, `vm.swappiness=10` live, zswap `Y/zstd/zsmalloc`. spark1
activates swap automatically at boot (Aug-22 boot log: unit activated +
`swap.target` reached); spark2's Aug-22 boot was the masked-unit case.
**swappiness=10** (was 100): swappiness=100
+ disk swap caused decode stalls/hitches once swap was live; 10 keeps zswap
reclaim without the disk-swap stalls. Persisted in
`/etc/sysctl.d/99-dgx-spark-swap.conf` on both nodes (root-owned, written via
docker-as-root). zswap: `zstd/zsmalloc`, `max_pool_percent=5`. Do not graph DSpark
`_sample_sequential` (shared `lm_head`).
Do not call b12x MXFP8 `wo_proj.run()`. Do not gather packed-at-store
indexer K. Do not add CUDA graph size 6. Do not feed the 1-row scheduled
scorer into 8-row decode.

---

## Cluster

| Node | IP (fabric) | IP (mgmt) | Role |
|------|-------------|-----------|------|
| spark1 | 10.0.1.1 | 192.168.0.211 | head (rank 0) |
| spark2 | 10.0.1.2 | 192.168.0.212 | worker (rank 1) |

- **GPU**: NVIDIA GB10, SM12x (capability 12.1, family 120), 128 GiB UMA per node
- **Fabric**: ConnectX-7 RoCE, `enp1s0f1np1`, NCCL IB GID 3
- **Model**: 155.43 GiB safetensors, `~/models/ds4-flash-0731` on each node
- **Repo clone**: `/tmp/vllm-spark-0731` on each node (includes `configs/nodes.env`)

---

## Docker Image (overlay fallback)

- **Tag**: `vllm-spark-0731:v0.28.0rc2-b12x`
- **Base**: `vllm/vllm-openai:v0.27.1` (arm64)
- **vLLM**: v0.28.0rc2 Python code overlaid onto v0.27.1 compiled extensions
- **Added**: `b12x==1.2.6` via uv (SM12x MoE + attention kernels)
- **Overlays**: `patches/apply_overlays.py` (20 build-time source patches)
- **Asserts**: `patches/assert_image.py` (build-time source-level verification)
- **Build**: `bash scripts/02-build-image.sh` on each node
- **Live images**: patched in-place with `docker commit` (MQA graph-safe + ReLU
  + DSV4 `nvfp4_ds_mla` accept). Always restore
  `ENTRYPOINT ["vllm", "serve"]` and `CMD []`.

All patches are baked into the image at build time. No runtime volume mounts
needed for code.

Matched-main image: `vllm-spark-0731:main-b12x`. Build:
`scripts/02-build-main.sh` then `scripts/03-apply-main-overlays.sh`. Copy
with `scripts/02-copy-main.sh`. Reapply attention with `--only b12x-sparse
--vllm-dir /opt/vllm/vllm` (see Status).

**Why v0.27.1 base (overlay only):** v0.28.0rc2 has no arm64 Docker image on
Docker Hub. v0.27.1 is the latest arm64 release. Overlay rc2 Python onto
v0.27.1 for the fallback pin. Main-b12x does not use this base.

---

## Kernel stack on SM12x (GB10)

SM12x cannot run SM90/SM100 TMA-based DeepGEMM routines or CUTLASS block-FP8 without granular guards and fallbacks.
The kernel selection for each operation:

| Operation | Kernel | Notes |
|-----------|--------|-------|
| Linear (FP8) | B12xFp8BlockScaledMM | `--linear-backend b12x`. Real `wq_a` GEMM cosine 0.9999986 vs torch dequant. Triton dies on `float8_e8m0fnu`. |
| MoE (MXFP4) | b12x B12X_MXFP4_MXFP8 | Explicit MOE_BACKEND=b12x |
| Attention (MLA) | b12x compressed MLA | Live main: `B12X_MLA_SPARSE`. Overlay fallback: FLASHINFER_MLA_SPARSE_DSV4. Same 584 B page. |
| MQA logits (prefill) | PyTorch dequant + ReLU | `_sm12x_fp8_mqa_logits`, matches `fp8_mqa_logits_torch` |
| MQA logits (decode) | b12x paged, page_size 64 | Packed-at-store. Gather of packed pages is wrong. Fallback gather is unpacked-only. |
| fp8_einsum | PyTorch dequant fallback | Patched in `utils/deep_gemm.py` |
| mHC prenorm | TileLang fallback | Guard on `is_deep_gemm_supported()` |
| Speculative decode | DSpark k=5 | FlashInfer TOPK=192 dispatch added |

---

## SM12x kernel guards and fallbacks

DeepGEMM and CUTLASS block-FP8 kernels assert SM90/SM100 at the C++ level.
On SM12x (arch_major=12), these crash with `attention.hpp:122` or similar.
Every call path that reaches DeepGEMM C++ code needs a Python-level guard.

### Guards (prevent calling into incompatible C++)

| Guard | File | Effect |
|-------|------|--------|
| `is_deep_gemm_supported()` | `utils/deep_gemm.py` | Returns False on family 120 (legacy rc2 overlay; matched-main compiles nv_dev 8b1392b where `is_deep_gemm_supported()` is True with granular operation guards) |
| CUTLASS FP8 exclusion | `kernels/linear/scaled_mm/cutlass.py` | `is_supported()` returns False on family 120 (routes linear to b12x) |
| Indexer build guard | `v1/attention/backends/mla/indexer.py` | Gates DeepGEMM paged MQA metadata on `num_states in (32, 64)` and SM12x fallbacks (PR #53522) |
| mHC TileLang guard | `kernels/mhc/tilelang.py` | Uses TileLang instead of DeepGEMM for prenorm GEMM |

### Fallbacks (pure-PyTorch replacements)

| Fallback | File | What it replaces |
|----------|------|------------------|
| `fp8_fp4_mqa_logits` | `utils/deep_gemm.py` | DeepGEMM MQA logits (prefill). `sum_h w_h * relu((q_h·k) * scale)` |
| `fp8_fp4_paged_mqa_logits` | `utils/deep_gemm.py` | DeepGEMM paged MQA logits (decode). Same formula, gather from paged cache |
| `fp8_einsum` | `utils/deep_gemm.py` | DeepGEMM fp8 einsum. Dequants FP8 to bf16 + torch.einsum with correct weight reshape |
| `VLLM_USE_DEEP_GEMM_E8M0=0` | `configs/env.spark.sh` | Disables E8M0 at env level (DeepGEMM-specific scale format) |

### MQA logits fallback detail

The MQA logits functions are called from `sparse_attn_indexer.py` during
the sparse attention indexing phase. DeepGEMM's C++ implementation
(`attention.hpp:122`) hard-asserts `arch_major in {9, 10}`.

DSA indexer scores are not a linear weighted-Q contraction. DeepGEMM and
vLLM's `fp8_mqa_logits_torch` (`v1/attention/ops/rocm_aiter_mla_sparse.py`)
compute:

```
score[h, m, n] = (q[m,h] · k[n]) * scale[n]
logits[m, n]   = sum_h  w[m,h] * relu(score[h,m,n])
```

A first overlay used `q_w = einsum('...hd,...h->...d', Q, w)` then `q_w @ K.T`.
That equals `sum_h w_h * (q_h · k)` and drops ReLU, so it is the wrong
kernel. Overlay 17 (`patch_mqa_relu_formula`) replaced both prefill and
paged fallbacks with the ReLU formula. Prefill always masks with
`cu_seqlen_ks/ke`. Paged gather length is
`min(max_blocks * block_size, max_model_len)` from Python shapes (no
`.item()`), chunked, invalid positions set to `-inf`.

The paged layout is `[num_blocks, block_size, 1, D+4]`: first D bytes are
`float8_e4m3fn`, last 4 bytes are an fp32 dequant scale.

---

## FlashInfer DSV4 dispatch for TOPK=192

DSpark k=5 speculative decoding requires `top_k = ceil(133/64) * 64 = 192`.
v0.27.1 FlashInfer only supports TOPK in {128, 512, 1024}.

**Three-part fix:**

1. **Python dispatch table** (`flashinfer/mla/_sparse_mla_sm120.py`):
   Added `(H, 192)` entries for all H in {8, 16, 32, 64, 128} to
   `_DECODE_DSV4_DISPATCH` frozenset.

2. **C++ dispatch table** (`flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu`):
   Added `DSV4_DISPATCH(H, 192)` macro calls for all H values. The template
   `launch_decode_dsv4_impl` is generic over NUM_HEADS and TOPK, so adding
   dispatch entries instantiates new template specializations.

3. **JIT cache invalidation**: Deleted pre-compiled `sparse_mla_sm120.so`
   from `flashinfer_jit_cache` package to force JIT recompilation from
   patched C++ source. First startup will be slower due to JIT compilation.

---

## Build-time overlay system

`patches/apply_overlays.py` applies 20 source-level patches in order:

1. `copy_new_modules` -- b12x MoE integration files
2. `patch_moe_backend` -- MoEBackend + LinearBackend type literals
3. `patch_envs` -- VLLM_B12X_MOE_FP4_FORCE_A16 env var
4. `patch_utils_b12x` -- B12xWarmupUnit, get_b12x_fused_moe imports
5. `patch_mxfp4_oracle` -- MXFP4 oracle for b12x backend selection
6. `patch_mhc` -- mHC TileLang fallback for SM12x
7. `patch_nvfp4_ds_mla` -- nvfp4_ds_mla KV cache dtype (584B DSV4 page)
8. `patch_dsv4_nvfp4_attn` -- FLASHINFER_MLA_SPARSE_DSV4 accepts nvfp4_ds_mla on SM12x
9. `patch_deep_gemm_sm12x_guard` -- `is_deep_gemm_supported()` excludes SM12x
10. `patch_cutlass_sm12x_guard` -- CUTLASS FP8 excludes SM12x
11. `patch_indexer_deepgemm_guard` -- indexer build() skips DeepGEMM on SM12x
12. `patch_fp8_einsum_fallback` -- fp8_einsum SM12x dequant fallback
13. `patch_einsum_sm12x_recipe` -- o_proj SM90 FP32 128x128 scales on SM12x (not SM100 packed INT32)
14. `patch_einsum_sm12x_scale_upcast` -- UE8M0 scale bytes to 2^(e-127) in einsum
15. `patch_mqa_logits_sm12x_fallback` -- MQA logits SM12x dequant fallback (prefill + paged decode)
16. `patch_mqa_paged_cudagraph_safe` -- paged MQA gather uses Python shapes, no `.item()`
17. `patch_mqa_relu_formula` -- DSA ReLU formula (replaces weighted-Q)
18. `patch_mxfp4_process_weights` -- process_weights_after_loading call for b12x
19. `patch_flashinfer_dsv4_dispatch` -- Python DSV4 dispatch +TOPK=192
20. `patch_flashinfer_dsv4_cu_dispatch` -- C++ DSV4 dispatch +TOPK=192, jit cache delete

Each patch is idempotent (skips if already applied). Incremental image
patches: `--only mqa-paged-graph`, `--only mqa-relu`, `--only dsv4-nvfp4`,
`--only einsum-sm12x`.
`assert_image.py` checks SM12x guards, no `.item()` in paged MQA, ReLU
in both MQA fallbacks, DSV4 `nvfp4_ds_mla` accept, and SM12x o_proj FP32 scales.

---

## Memory settings

128 GiB UMA per node. Model weights ~77.7 GiB/rank. b12x weight prep adds overhead.

Live **main-b12x** pin (`configs/pin.main.env`): util **0.8**, `MAX_NUM_SEQS=8`,
`MAX_NUM_BATCHED_TOKENS=8192`, default capture `[1, 2, 4, 8, 16, 24, 32, 40,
48, 56, 64]` (do not add 6), KV **97,737** tokens. b12x MLA scratch is
~0.61 GiB (do not plan 8192×16). spark2 MemAvailable ~12.4 GiB after this
boot, SwapTotal 0. Do not raise util.

Overlay fallback (`pin.nvfp4.env`):

| Setting | Value | Notes |
|---------|-------|-------|
| GPU_MEMORY_UTILIZATION | 0.81 | spark2 earlyoom SIGTERM at MemAvailable <8%. 0.85 died. PIECEWISE live: 561,703 KV tokens. |
| MAX_MODEL_LEN | 65536 | |
| MAX_NUM_SEQS | 2 | |
| MAX_NUM_BATCHED_TOKENS | 2048 | |
| ENFORCE_EAGER | 0 | PIECEWISE + TP all-reduce eager-break. FULL still forbidden. |
| CUDAGRAPH_MODE | PIECEWISE | Breakable graphs. DSpark capture skipped. |
| MAX_CUDAGRAPH_CAPTURE_SIZE | 36 | vLLM truncates to 32. |
| BLOCK_SIZE | 256 | Manager block. SWA/FlashInfer kernel page is 64 on family 120. Do not set 64. |

---

## Operating the cluster

### Build (on spark1, then copy or build on each node)
```bash
ssh spark1 "cd /tmp/vllm-spark-0731 && git pull && bash scripts/02-build-main.sh && bash scripts/03-apply-main-overlays.sh"
# Copy image to spark2 via scripts/02-copy-main.sh
```

### Stop (cleans shm, prompts for fs cache drop)
```bash
ssh spark1 "cd /tmp/vllm-spark-0731 && bash scripts/07-stop.sh"
ssh spark2 "cd /tmp/vllm-spark-0731 && bash scripts/07-stop.sh"
```

### Launch (worker first, then head)

`05-serve.sh` starts `docker run -d`. Do not `nohup` it. Never chain
`07-stop.sh` and `05-serve.sh` in one SSH session. Use a separate SSH per
node (`ControlPath=none`). Do not reboot the Sparks.

```bash
# Worker first, then head. Scripts are on the nodes at /tmp/vllm-spark-0731.
ssh -o ControlPath=none spark2 'cd /tmp/vllm-spark-0731 && bash scripts/05-serve.sh main </dev/null'
ssh -o ControlPath=none spark1 'cd /tmp/vllm-spark-0731 && bash scripts/05-serve.sh main </dev/null'
```

Overlay fallback uses `nvfp4` instead of `main`. `fp8` pin is parked.

Main pin is `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` in `configs/pin.main.env`.
Do not set `CUDAGRAPH_CAPTURE_SIZES` on that pin (default list, no size 6).
Overlay fallback stays `PIECEWISE` with TP all-reduce eager-break. Do not
copy overlay `FULL` onto the rc2 image.

`configs/nodes.env` (gitignored, manually copied) sets `VLLM_HOST_IP`, `HEAD_IP`,
and `NODE_RANK` per node. The workstation hostname is not a Spark hostname,
so `scripts/06-validate.sh` from the laptop hits the wrong `HEAD_IP`. Run
validate and benches on spark1 (`http://127.0.0.1:8000`):

```bash
ssh -o ControlPath=none spark1 'cd /tmp/vllm-spark-0731 && VALIDATE_STACK=main bash scripts/06-validate.sh'
```

### Check logs
```bash
ssh spark1 "tail -40 /tmp/vllm-serve.log"
ssh spark2 "tail -40 /tmp/vllm-serve.log"
# or
ssh sparkN "docker logs --tail 40 vllm-ds4-0731"
```

### Test inference
```bash
curl -s http://spark1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

---

## Status

### v0.28.0 rebase — DEPLOYED (2026-08-28)

Upstream released **vLLM v0.28.0** (2026-08-26, tag `2cf0a6915ce5`),
which includes DSV4 sparse-MLA E2E fixes (#51538), b12x linear backends
(#52016), JIT-warmup infra, and pins DeepGEMM at `8b1392b` (the same tree
our fp8-1d1d port was staged against). Rebase status:

- **Patches rebased + verified applying to v0.28.0**: `pr-53522` (renamed
  `num_states` -> `storage_block_size`), `pr-53425`, `pr-53574`,
  `pr-47988`, `pr-53055` (test hunks dropped - refactored upstream;
  **cuda.py sm121 exclusion dropped** - deep_gemm verified working on GB10),
  `kv-offload-bounds-check` (headers fixed), `b12x-moe` (envs/warmup hunks
  rebuilt), `0003-nvfp4-ds-mla` (paths fixed + kv_cache_interface section
  dropped - superseded by the v0.28.0 `state_content_bytes` mechanism).
- **Dropped**: `pr-53521`, `pr-53898` (closed - einsum misdiagnosis),
  `pr-52499`, `b12x-linear-52016` (merged as #52016), `deepgemm-pr-403`
  (already in the pinned 8b1392b), `mhc-guard-50645` (dup of 0002).
- **apply_overlays.py adapted**: the 3 einsum overlays (fallback / recipe /
  scale-upcast) removed from both apply flows; `patch_b12x_mm_block_fp8_compat`
  retargeted to `b12x_block.py`; obsolete-guard skips for the upstream-present
  SM12x KV insert, the removed BLHNC split, and the reworked KVBlockZeroer;
  indexer b12x-schedule needles updated to the `_should_build_paged_mqa_logits_metadata`
  gate; eager_scratch getattr now skips assignment sites (was a SyntaxError
  on v0.28.0); schedule-pass overlay made tolerant; **v0.28.0 integration
  fixes**: `patch_envs` + MoE overlays added to apply_main (rc2-only before),
  `patch_mxfp4_process_weights` wired (modular-MoE weight prep), b12x MoE
  weight-prep call for GPU, donor `nsa_indexer` -> `dsa_indexer` (b12x
  master drift), donor `compressed_mla` -> `compressed_sparse_mla`.
- **Verified**: full pipeline (5 pr-* patches + donors + ~26 overlays) exits 0
  on a clean v0.28.0 tree; `compileall` passes; **E2E boot on both nodes**:
  France coherent, health 200, CUDA graphs 27/27 + DSpark 21/21 captured.
- **Pins**: `VLLM_REF=2cf0a6915ce5` (pin.main.env + pin.main-dg.env).
  `DEEPGEMM_COMMIT` stays `a6b593d` (the verified wheel; do NOT switch to
  the v0.28.0-pinned 8b1392b, which removed the SM12x fp8 1d1d path).
- **Images**: `main-b12x-028` (v0.28.0 + overlays, committed 97e4a6a);
  `main-b12x-028-rdma` (added rdma-core v54 libmlx5 - see 05-performance.md
  NCCL/RoCE finding; benchmark pending).
- **Benchmarks (golden methodology, shared coding prompt, natural EOS)**:
  c1 17.2, c3 32.4, c5 60.4, c6 49.6 tok/s (before the rdma fix).
  NOTE: benchmark prompts matter ~1.7x (see GOLDEN.md); unique-prose +
  forced-continuation (ignore_eos) is the WORST case.
- **b12x hazard**: B12X_REF=master is rolling - the phase-1 build pulled a
  version that renamed `nsa_indexer`->`dsa_indexer` and
  `compressed_mla`->`compressed_sparse_mla`; consider pinning B12X_REF.

### Golden
 (anemll, real NVFP4) — deployed 2026-08-24

`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (stock, zero patches) via sparkrun
with `examples/anemll-nvfp4-golden.yaml` + the abliterated checkpoint
`drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32`.
**Copy the recipe to /tmp first**: `spark-launch.sh` runs
`pkill -9 -f '[s]parkrun'`, which kills the script itself when the recipe
path contains "sparkrun". Boot: model distribute ~21 min (81 files), then
weights 79.17 GiB / 252.8 s, then warmup. Health 200 at ~17:14 +08.
Containers `sparkrun_<id>_node_0` / `_node_1`.

- **KV pool 2,047,170 tokens** (16.07 GiB) — real `nvfp4_ds_mla`, 7,650
  B/token. Log: `Using DeepSeek V4 padded nvfp4_ds_mla KV cache format.`
- France greedy (temp 0, 32 tok): `' Paris. The capital of Spain is
  Madrid…'` — same string as main-b12x.
- France temp 0, 128 tok: **c1 65.2** (main-b12x 25.8, 2.5×), c6 **216.8**,
  c16 183.9, c32 186.2. Plateau past c6 is by design (`max_num_seqs=6`,
  capture truncated 36→32).
- Golden harness (BST coding, temp 0.7, 128 tok): c1 54.6–58.1 (GOLDEN
  51.4), c3 108.5 (112.7), c5 124.2 (126.2), c6 155.0 (157.9) — reproduces
  GOLDEN.md on this hardware.
- Bench: `scripts/bench-concurrency.py` (stdlib-only; run on spark1). The
  2.5× is whole-stack (v0.25.2 core + their kernels + real NVFP4 writer),
  not the attention backend (our A/B was parity). Serving conclusion: golden
  for max speed + real NVFP4; matched-main stays the upstream/PR track.

### Live: main-b12x (2026-08-24 08:13 UTC)

`scripts/05-serve.sh main` (`configs/pin.main.env`). Image
`vllm-spark-0731:main-b12x`. API: `http://spark1:8000` (from spark1:
`http://127.0.0.1:8000`). Container `vllm-ds4-0731`. Repo on nodes:
`/tmp/vllm-spark-0731`. vLLM `v0.1.dev1+ge25c586b9.d20260823`.

`--only` overlays on the live image: `o-proj-b12x`, `indexer-store-page64`,
`indexer-b12x-schedule`, `ar-piecewise-ws`, `dspark-backbone-none`,
`mqa-packed-gather` (packed-layout-correct MQA gather fallback, unit-tested
0.0 diff), `flashinfer-eidx-contig` (`.contiguous()` on extra_sparse_indices
so FLASHINFER_MLA_SPARSE_DSV4 boots — root cause is the C128A builder
publishing a width-sliced view of the persistent `global_decode_buffer`,
upstream #53574, backported as `patches/upstream/pr-53574.diff`; the C4A
branch was verified already contiguous, so there is no C4A bug). Helper
`patches/files/sm12x_b12x_kernels.py` is copied to
`vllm/utils/sm12x_b12x_kernels.py`. `_B12X_SCHEDULE_MAX_Q_ROWS = 1`.
`MAX_NUM_SEQS=32` (was 8), `MAX_CUDAGRAPH_CAPTURE_SIZE=192` (was 64).

`VALIDATE_STACK=main scripts/06-validate.sh` on spark1 at 08:35 UTC: first
token `' Paris'` logprob -0.245, n_tie=1, chat `Paris`. 1-way / 8-way were
not re-benched after the 08:12 restart (last benches 01:40 UTC).

| Check | Result |
|-------|--------|
| Linear | `--linear-backend b12x` |
| MoE | `--moe-backend b12x` |
| Attn | `B12X_MLA_SPARSE` target + DSpark draft (`b12x.attention.compressed_mla`) |
| WO | decode `torch.bmm` after fused inv-RoPE fp8 dequant. Not MXFP8 `wo_proj.run()`. |
| Indexer | b12x paged MQA, page_size 64, packed-at-store. Capture: `table=(48, 1024) sched=False q_rows=48`; 1-way path `table=(8, 1023) sched=False q_rows=8`; 1-row `table=(1, 1024) sched=True`. |
| KV | `nvfp4_ds_mla` (584 B DSV4 envelope). **97,737** tokens at util 0.8. spark2 swap **enabled** (was masked; see Overview). |
| Greedy France | `' Paris. ...'` n_tie=1, logprob -0.25..-0.26 |
| Chat | `Paris` |
| 1-way 128 | bench (2026-08-24, temp 0) **~25.8** tok/s. Gate 17.81. Gather pin ~30.6. |
| 8-way 128 | bench (2026-08-24) **~95** tok/s (was 85.98 at 01:40). Gate 52.12. |
| c16 / c32 | **~116 / ~172** tok/s aggregate (max_num_seqs 8→32 lift) |
| DSpark | k=5. Backbone FULL 6/6, sample eager. Do not graph `_sample_sequential`. |
| Graphs | PIECEWISE 11/11, FULL 7/7, DSpark FULL 6/6. Capture up to 192 (32 seqs × 6). TP all-reduce in-graph. |
| MHC | TileLang. `VLLM_USE_B12X_MHC` is unused. |
| spark2 | util 0.8. Do not raise to 0.85. |

Phase 3 baseline on the same image (2026-08-23 16:29 UTC), before paged
indexer / WO bmm / DSpark backbone / in-graph AR: 1-way 17.81, 8-way 52.12,
KV 93,401, DSpark capture skipped, FlashInfer still owned indexer gather.

**1-way gap (~4 tok/s vs gather ~30.6).** DSpark 1-way is 1+5=6 tokens.
`CudagraphDispatcher.uniform_decode_query_len` is 6. The default capture
list has no 6, so the step pads to 8. FULL uniform decode requires
`num_tokens_padded % 6 == 0`, so 8 (`8 % 6 != 0`) uses the size-8 mixed
graph, not a FULL size-6. Adding 6 to the capture list did capture
`table=(6, 1023)` and was a net loss (1-way 23.98, 8-way 71.52). Leave the
default list.

b12x `uses_paged_schedule` is true for `q_rows<=8` and `max_pages>=1024`.
The helper only consumes vLLM `plan_paged_schedule` when `q_rows==1`, then
trims to 1023 pages so the 8-row DSpark path stays unscheduled. Overlay
`indexer-b12x-schedule` also skips the GPU `plan_paged_schedule` kernel
unless `q_rows==1`. 8-way capture is 48 rows, already unscheduled.

The remaining hole is the paged 8-row 1023-page indexer versus interleaved
gather (~0.14 ms × 43 layers is the working guess). Safe overlay knobs
above are exhausted. Not tried: a faster paged kernel/grid for short
seqlens (not Python schedule consume); interleaved store for 1-way gather
with packed paged still on for 8-way, without a dual-layout spark2 OOM.

b12x has a multi-row scheduled scorer for `q_rows` 2–8. It is not live.
Feeding the 1-row scheduled kernel (or a zero DeepGEMM buffer) into 8-row
decode dropped 8-way to 16.29 tok/s. Consuming the vLLM schedule for
`q_rows` 2–8 made 1-way 25.36 (`sched=True q_rows=8`). Do not remove the
`q_rows==1` gate without a correct per-step schedule and a re-measure of
both 1-way and 8-way.

**KV.** Manager tables on this pin are already 1024-wide page64
(`width*64 >= max_model_len`, expand skipped). `indexer-b12x-schedule` now
skips allocating `page64_block_table_buffer` in that case. KV recovered
94,516 → **97,737**.

Tried and reverted (do not repeat):

- MXFP8 `wo_proj.run()`: France loops.
- DSpark full-step graph including `_sample_sequential`: accept 66.7% → 57.4%.
- Pack-every-decode or padded 48×1024 pack: 1-way below 17.81.
- Packed sidecar at insert: ~3.7 GiB, spark2 OOM.
- `plan_paged_schedule` inside CUDA graphs: frozen warmup seqlens, accept 47%, 8-way 50.62.
- Extra CUDA graph sizes `[1,2,3,...,8,16,...,64]`: KV 97k→92k, no 1-way win.
- Expand already-page64 1024-wide tables `*4` → 4096: garbage page ids, KV 92k, 8-way collapse.
- Live schedule into 8-row decode (1-row kernel or empty DeepGEMM buffer): 8-way 16.29 tok/s.
- Multi-row scheduled paged scorer (`q_rows` 2–8 consume vLLM schedule): 1-way median 25.36.
- CUDA graph size 6 (so DSpark 1+5 does not pad to 8): 1-way 23.98, 8-way 71.52, KV 92k–94k.
- `preinitialize_invalid_logits=False`: France still Paris but text drifted, logprob -0.335; 1-way 19.18 then 15–17.
- Skip paged kernel when `m_rows<=8`, FlashInfer gather of packed-at-store: warm 1-way median 29.88 (best 30.90), 8-way 89.34; 9-run 1-way median 28.70 with hitches 16.29 / 17.40; DSpark accept 38–70% (HANDOFF pin ~73%). Gather of packed-at-store is numerically wrong.

b12x compressed MLA is 584 B/token (same DSV4 page as `fp8_ds_mla`). It is
not the GLM 432/368 NVFP4 writer. Do not pass `scale_format=2`.

Kernel contracts in `patches/files/dsv4_b12x_sparse.py`:

1. Indexed/SWA page size is the cache tensor (64 slots → 37376 B), not C4
   `attn_metadata.block_size // 4` (16). Indices are raw slot ids.
2. Prefill MG (and Spark decode at 16+ tokens with ≤10 chunks) rejects
   DSpark's padded SWA width 192. SWA-only pads to 512. Dual-cache prefill
   clips SWA to 128. Small-batch decode keeps 192.

Reapply on a committed image (both nodes). `--vllm-dir` is required; a
duplicated `B12X_MLA_SPARSE` enum makes `import vllm` raise `TypeError`.
For the helper + schedule overlay:

```bash
docker rm -f overlay-b12x
docker run --name overlay-b12x --user root --entrypoint python3 \
  -v /tmp/vllm-spark-0731/patches:/patches:ro \
  vllm-spark-0731:main-b12x \
  /patches/apply_overlays.py --only indexer-b12x-schedule --vllm-dir /opt/vllm/vllm
docker commit --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD []' \
  overlay-b12x vllm-spark-0731:main-b12x
docker rm overlay-b12x
```

Attention enum (once per image): `--only b12x-sparse --vllm-dir /opt/vllm/vllm`.
Same `ENTRYPOINT` / `CMD` restore.

Worker first, then head: `bash scripts/05-serve.sh main </dev/null`.

Still open on this image: 1-way at/above gather ~30.6 with paged indexer
left on and France/8-way intact; util above 0.8; a real DSV4 NVFP4 writer;
KV offload (Phase 6). FULL target graphs are captured; do not treat that
as the old overlay FULL crash. Do not feed the 1-row scheduled kernel into
8-row decode. Do not gather packed-at-store. Do not add capture size 6.

### Fallback overlay (v0.28.0rc2 on v0.27.1)

Active overlay pin: `scripts/05-serve.sh nvfp4` (`pin.nvfp4.env`). B12x
linear, B12x MoE, `nvfp4_ds_mla` KV, FLASHINFER_MLA_SPARSE_DSV4, DSpark k=5,
PIECEWISE graphs. `fp8_ds_mla` is parked.

### Overlay evidence (PIECEWISE + AR eager-break, 2026-08-23 03:08 UTC)

| Check | Result |
|-------|--------|
| Greedy France | ` Paris. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of Germany is Berlin. The capital of Portugal is Lisbon. The capital` |
| First token | `' Paris'` logprob -0.244, n_tie=1 (not -ln(96), not -9999) |
| Chat | `content="Paris."`; English in `message.reasoning` |
| tok/s | 15.64 greedy (32 tokens / 2.05 s). Earlier request 14.14. First cold request 8.95. |
| KV | 561,703 tokens at util 0.81 |
| DSpark | k=5 on, 110/275 draft tokens accepted (40%), mean length 2.0 |
| Linear | `B12xFp8BlockScaledMMKernel` |
| MoE | `B12xExperts` |
| Attn | FLASHINFER_MLA_SPARSE_DSV4, `sparse_mla_sm120_decode_dsv4` cache hit |
| CUDA graphs | PIECEWISE 7/7. Log: `TP all-reduce eager-break during capture (clone off graph pool)`. DSpark capture skipped. |
| KVBlockZeroer | `ratio=1` on indexer/SWA/C128 (`(16679, 64, 132)`, `(16679, 64, 584)`, `(16679, 2, 584)`) |
| spark2 MemAvailable | ~10.7 GiB after warmup (just above earlyoom 8% / ~9.7 GiB) |

`scripts/06-validate.sh` passed (greedy France starts with Paris + Spain/Madrid; chat `Paris.`).

Prior eager boot (2026-08-22 21:55 UTC) was the same France string, `' Paris'` ≈ -0.275, 14–16 tok/s, KV 538,937–540,857, DSpark 40–49% / mean 3.0–3.46. Keep that as the fallback pin (`ENFORCE_EAGER=1`).

CUDA graph isolations that failed before the AR clone overlay:

- FULL + DSpark FULL: after capture, `lm_head` dump went to inf / logits ~1e33. Greedy was `的超` / `LeanCAN` / `buffalo` with tied logprobs -4.564 = -ln(96). DSpark accept ~5%.
- FULL, DSpark graphs skipped, `lm_head` restored from CPU after capture: `w_finite=True` but dummy logits still ~1e33. spark2 then died in `CUDAGraph::replay()` / `cudaGraphLaunch`.
- PIECEWISE only, DSpark capture no-op: `' Actors'` x32, `' Paris'` logprob -9999.
- PIECEWISE + `cudagraph_copy_inputs=true`: `ligands`/`Actors`, `' Paris'` still -9999.
- PIECEWISE + AR eager-break via `@eager_break_during_capture` at import of `communication_op.py`: circular import (`VllmConfig` from partially initialized `vllm.config`).
- PIECEWISE + AR eager-break, NCCL on weak-ref graph-pool buffer: capture 7/7, dummy `l_max≈22` `n_tie=1`, then warmup replay `ncclAllReduce` / illegal memory access.

### Done
- [x] Docker image: v0.27.1 base + rc2 Python overlay + b12x==1.2.6
- [x] SM12x kernel guards (DeepGEMM, CUTLASS block-FP8)
- [x] fp8_einsum SM12x dequant fallback
- [x] MQA logits SM12x dequant fallback (prefill + paged decode)
- [x] Paged MQA fallback is CUDA-graph-safe (no `.item()` host sync)
- [x] MQA fallback uses DSA ReLU formula (not weighted-Q)
- [x] b12x MoE weight preparation + process_weights_after_loading
- [x] FlashInfer DSV4 TOPK=192 dispatch (Python + C++ + jit cache invalidation)
- [x] nvfp4_ds_mla 584-byte page support (`patch_nvfp4_ds_mla`)
- [x] DSV4 SM12x `supports_combination` accepts `nvfp4_ds_mla`
      (`patch_dsv4_nvfp4_attn`). Without this, `05-serve.sh nvfp4` never
      selects FLASHINFER_MLA_SPARSE_DSV4.
- [x] mHC TileLang guard for SM12x
- [x] nodes.env auto-detects hostname for NODE_RANK/VLLM_HOST_IP
- [x] Stop script cleans shm
- [x] Images docker-commit patched on both nodes. Always restore
      `ENTRYPOINT ["vllm", "serve"]` and `CMD []`.
- [x] Coherent greedy France on eager nvfp4 + B12x + DSpark k=5 (see live
      evidence). Isolated kernel cosines (mHC TileLang, CuteDSL C4,
      FlashInfer DSV4, B12x GEMM/MoE, dsv4_topk) already matched torch.
- [x] B12x FP8 linear vs torch dequant on real `layers.0.attn.wq_a`:
      cosine 0.99999857. Synthetic 128x128 UE8M0 same cosine. B12x GEMM
      is not the `的超` source.
- [x] B12x MXFP4 MoE vs `moe_reference_w4a8_mx` (expert 0, silu,
      `swiglu_limit=10`, `w13_layout="w31"`): cosine 0.998960. Wrong
      `w13` layout is cosine 0.636. Shared experts are FP8 128x128, same
      path as `wq_a`. MoE math is not the `的超` source.

### Graphs vs eager (rc2 overlay image)

Matched-main numbers are in the live table above. This table is the
**v0.28.0rc2 overlay** only. Do not copy "never FULL" onto main-b12x.

| Mode | France | Notes |
|------|--------|-------|
| `--enforce-eager` | coherent `" Paris."` | 16.25 tok/s, DSpark 45.7%, KV 540,857 |
| PIECEWISE + AR eager-break, clone off graph pool | coherent `" Paris."` | 15.64 tok/s, DSpark 40%, KV 561,703. Overlay pin. |
| FULL + DSpark FULL | `的超` / -ln(96) | logits collapse after capture |
| FULL, DSpark graphs off | crash | `cudaGraphLaunch` on spark2 |
| PIECEWISE, DSpark graphs off | `Actors` loop | `' Paris'` at -9999 |
| PIECEWISE + `cudagraph_copy_inputs=true` | `ligands`/`Actors` loop | `' Paris'` still -9999; dummy dumps mix 1e33 and ~20 |
| PIECEWISE + AR `@eager_break` at import | boot fail | circular import `vllm.config` |
| PIECEWISE + AR break, NCCL on graph-pool weak-ref | crash | capture ok, replay `ncclAllReduce` IMA |

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` is on in `env.spark.sh`. Cross-node GB10
NCCL is host-staged PYNCCL. On the overlay image, PIECEWISE does not break
at NCCL; RowParallelLinear `down_proj` captured it inside GEMM segments.
Overlay runs `tensor_model_parallel_all_reduce` via `capture.add_eager`,
clones to the default allocator, then `copy_` into the caller buffer.
FULL mode on that image still leaves NCCL inside the graph. Do not enable
FULL on the overlay pin.

Matched-main uses `ar-piecewise-ws` (in-graph TP AR) and keeps France with
`FULL_AND_PIECEWISE`.

### Still open
- [x] PIECEWISE CUDA graphs that keep France as `" Paris"` (AR eager-break
      + clone off graph pool). Overlay FULL still crashes or collapses.
- [x] Matched-main `FULL_AND_PIECEWISE` with in-graph TP AR (`ar-piecewise-ws`).
      Overlay FULL is still forbidden.
- [ ] Overlay FULL graphs. Overlay does not break AR when `cudagraph_runtime_mode`
      is FULL. Do not boot FULL on the rc2 pair.
- [ ] GPU_MEMORY_UTILIZATION 0.83–0.85 only if spark2 stays above earlyoom 8%.
      Swap is now enabled (16 GiB, unmasked, swappiness=10) but the earlyoom
      threshold still governs. Live main stays 0.8.
- [x] KVBlockZeroer: unaligned `num_blocks % ratio` no longer skips the
      tensor. Live log `KVBlockZeroer ratio=1` on indexer/SWA/C128 pages
      (shapes `(16679, 64, 132)`, `(16679, 64, 584)`, `(16679, 2, 584)`).
      `block_dim` is still the num_blocks axis by design.
- [x] Skip unused indexer `page64_block_table_buffer` when manager tables
      are already 1024-wide. KV 94,516 → 97,737.
- [ ] 1-way decode at/above gather ~30.6 tok/s with paged indexer on,
      France green, 8-way not collapsed. Re-bench 2026-08-26 (restored
      main-b12x): c1 26.3 tok/s, c8 agg 73.4, c16 115.3, **c32 agg 174.7**,
      France green. The correctness
      half of item 18 is fixed (`packed_gather_mqa_logits`, 0.0-diff unit
      test); this bullet is now purely a perf gap on main-b12x. The speed
      goal itself is served by the golden image (c1 65.2, c6 216.8 —
      neither reaches 300 aggregated; see docs/knowledge/05-performance.md).
- [x] Real NVFP4 KV: resolved by adopting the golden image. Porting the
      NVFP4 writer into matched-main is not needed for the speed goal; this
      image's `nvfp4_ds_mla` stays the 584-byte fp8 envelope alias.
- [x] Validate with `scripts/06-validate.sh` (greedy France + chat Paris)
- [x] Benchmark (tok/s, B/token) on real NVFP4: golden deployed 2026-08-24,
      KV 2,047,170 tokens @ 7,650 B/token, France c1 65.2 / c6 216.8, golden
      harness reproduces GOLDEN.md (c1 54.6–58.1, c6 155.0). Bench script:
      `scripts/bench-concurrency.py` (stdlib-only, run on spark1).
- [ ] If more DeepGEMM or SM12x crashes appear, add more fallbacks
      (e.g. `tf32_hc_prenorm_gemm` at lines 806-823 of deep_gemm.py has no
      SM12x guard, but should be unreachable via the mHC TileLang guard.
      Monitor for crashes from `sm90_tf32_hc_prenorm_gemm.cuh` or
      `sm100_tf32_hc_prenorm_gemm.cuh`.)

### Upstream PRs

Matched **vLLM main** build (`nvidia/cuda:13.3.1-cudnn-devel`, source
PyTorch **2.14** `12.1a`, source NCCL sm_121, git heads for b12x /
FlashInfer / InstantTensor / LMCache, cutlass **4.7.0**, util 0.8):
[docs/PLAN-MAIN.md](docs/PLAN-MAIN.md).
Draft pin: [configs/pin.main.env](configs/pin.main.env). Overlay
`v0.28.0rc2-b12x` is the fallback if main-b12x is down.

Tracker vs **v0.28.0rc2 `74a6576`** and current main: [docs/UPSTREAM.md](docs/UPSTREAM.md).

Already in the rc2 **Python** tag: DSpark, `FLASHINFER_MLA_SPARSE_DSV4` (fp8
layouts), linear `--linear-backend b12x` (#52016), MoE `flashinfer_b12x`,
guarded mHC siblings (not broadcast).

Not in rc2, merged on main later: MoE `--moe-backend b12x` (#52018, 8h after
the tag).

Not in rc2 **or** main yet (still OPEN 2026-08-26): SM12x einsum recipe
([#53521](https://github.com/vllm-project/vllm/pull/53521); #52357 closed —
**updated 8284955**: SM12x `(1,128,128)` + `tma_aligned_scales=True` (packed
INT32 UE8M0) + 3D `wo_a` views; kitch2400 review applied, verified on SM121a
T=10/96/8192),
mHC broadcast + CUTLASS SM12x ([#53055](https://github.com/vllm-project/vllm/pull/53055)),
DSV4 kernel block 64 ([#53425](https://github.com/vllm-project/vllm/pull/53425) —
**import-cycle fix ed71de5**: `indexer → deepseek_v4.sparse_mla` module-level
import broke `vllm._aiter_ops` cold start (kitch2400 report); lazy import),
indexer DeepGEMM gate ([#53522](https://github.com/vllm-project/vllm/pull/53522) —
**ivanusto reviewed**: test passed, gate scoped correctly),
C128A eidx contiguity ([#53574](https://github.com/vllm-project/vllm/pull/53574)),
Triton E8M0 upcast ([#47988](https://github.com/vllm-project/vllm/pull/47988)).
The last two are backported as `patches/upstream/pr-53574.diff` /
`pr-47988.diff` (applied before overlays in the build); evidence comments
posted, no duplicate PRs (the C4A eidx branch was verified already
contiguous). Offload flat-layout root cause → issue
[#53607](https://github.com/vllm-project/vllm/issues/53607).
FlashInfer TOPK 192 is on flashinfer-ai main (#4380) but not in the overlay
image's `0.6.16.post3` wheel until the overlay. Matched-main FlashInfer is
git main (192 present).

DeepGEMM: rc2 cmake pins **nv_dev** `8b1392b` (SM12x); **8b1392b regressed
the pure-fp8 path** (removed `sm100_fp8_gemm_1d1d.{hpp,cuh}`; the
`fp8_gemm_nt` alias predates it — corrected 2026-08-26 in
[DeepGEMM#417](https://github.com/deepseek-ai/DeepGEMM/issues/417)). vLLM PR
[#53680](https://github.com/vllm-project/vllm/pull/53680) re-pins cmake to
`a6b593d`; local port `deepgemm-fp8-1d1d-port.diff` covers 8b-era builds.
This image still runs the v0.27.1 **main** `.so` (`e21c821`,
`attention.hpp:122`). eugr rebuilds nv_dev but freezes at `a6b593d` because
of an SM121 MXFP4 grouped scale-factor regression at `f8e8fb5` (DeepGEMM PR
#384). See [docs/UPSTREAM.md](docs/UPSTREAM.md).

---

## Key learnings

1. **v0.28.0rc2 arm64 image does not exist** on Docker Hub. v0.27.1 is the
   latest. We overlay rc2 Python code onto v0.27.1's compiled `.so` extensions.

2. **This image's DeepGEMM `.so` is SM90/SM100 only.** It is DeepGEMM
   **main** `e21c821` from the v0.27.1 arm64 build. MQA hits
   `attention.hpp:122` (`arch_major` in {9, 10}). v0.28.0rc2 cmake already
   wants DeepGEMM **nv_dev** `8b1392b` (SM12x). eugr rebuilds nv_dev but
   pins `a6b593d` to dodge an SM121 MXFP4 grouped-scale regression. Until
   this image rebuilds DeepGEMM, every C++ call path needs a Python guard.
   In contrast, matched-main (`main-b12x`) compiles `nv_dev 8b1392b` (`is_deep_gemm_supported() == True`) with granular operation guards on TMA attention (`attention.hpp:122`), 2-state MQA pages (PR #53522), and FP8 linear GEMM (`gemm.hpp:851`).

3. **FP8 linear kernel priority**: DeepGEMM > CUTLASS > B12x > Triton.
   While legacy rc2 forced a blanket `is_deep_gemm_supported() == False`, matched-main compiles `nv_dev 8b1392b` where DeepGEMM FP8 linear GEMM (`gemm.hpp:851`) and CUTLASS FP8 (`cutlass.py`) are guarded on SM12x, so auto-select selects `B12xFp8BlockScaledMMKernel` (`--linear-backend b12x`), avoiding Triton's `KeyError: 'float8_e8m0fnu'`. B12x linear does cover FP8
   block-scaled MM on SM12x (K128); it upcasts UE8M0 scales to fp32 because
   native 128x128 UE8M0 is not supported yet. Forcing `LINEAR_BACKEND=triton`
   selects TritonFp8BlockScaledMMKernel, then crashes:
   `KeyError: 'float8_e8m0fnu'` in Triton's `canonicalize_dtype` during
   `w8a8_triton_block_scaled_mm` (attention fused wqa/wkv GEMM). Pin nvfp4
   to `LINEAR_BACKEND=b12x`. Do not chase Triton for quality.

4. **FlashInfer JIT cache**: The `flashinfer_jit_cache` package ships
   pre-compiled `sparse_mla_sm120.so`. When patching the C++ source to add
   TOPK=192, the pre-compiled `.so` must be deleted to force JIT recompilation
   from the patched source. First startup is slower due to JIT compilation.

5. **DSpark k=5 needs TOPK=192**: `top_k = ceil(133/64) * 64 = 192`.
   The original dispatch tables (Python and C++) only have {128, 512, 1024}.
   Both must be extended for DSpark to work.

6. **B12xExperts.process_weights_after_loading** must be called by the MXFP4
   kernel factory. rc2 omits this call. The overlay patches both the factory
   function and `Mxfp4MoEMethod` caller.

7. **GPU_MEMORY_UTILIZATION=0.62 is too low**. Model + b12x prepared weights
   consume ~86.3 GiB/rank, leaving negative KV cache headroom. Use 0.80+.

8. **DSA MQA is ReLU, not weighted-Q.** `sum_h w_h * relu(q_h · k * scale)`.
   Weighted-Q (`einsum('...hd,...h->...d', Q, w)` then `@ K.T`) drops ReLU
   and is the wrong kernel. vLLM reference: `fp8_mqa_logits_torch`.

9. **CUDA graph capture forbids host sync.** `_sm12x_fp8_paged_mqa_logits`
   used `ctx.max().item()` to size the gather. Piecewise graphs captured;
   FULL graphs died with `cudaErrorStreamCaptureUnsupported`. Size the
   gather from Python tensor shapes (`block_tables.shape[1] * block_size`)
   and mask with `context_lens` instead.

10. **`docker commit` inherits `--entrypoint`.** Committing a container
    started with `--entrypoint bash` or `python3` overwrites `vllm serve`.
    Always `docker commit --change 'ENTRYPOINT ["vllm", "serve"]' --change 'CMD []'`.

11. **B12x linear and MoE match torch on this checkpoint.** Real `wq_a`
    FP8 GEMM cosine 0.99999857. MXFP4 expert 0 cosine 0.998960 with
    `w13_layout="w31"` (vLLM stacks gate-first; b12x then flips to
    `[up; gate]`). Wrong layout cosine 0.636. `的超` is not B12x math.

12. **DSV4 SM12x rejects `nvfp4_ds_mla` until patched.**
    `DeepseekV4FlashInferMLASparseBackend.supports_combination` only listed
    `fp8` / `fp8_e4m3` / `fp8_ds_mla`. Overlay 8 adds `nvfp4_ds_mla`.
    The SM120 kernels still consume the packed 584-byte uint8 page. This
    image has no separate NVFP4 CUDA writer.

13. **HF shards are blobs.** `~/models/ds4-flash-0731` files are relative
    symlinks into `~/.cache/huggingface/hub/.../blobs`. Docker must mount
    both the model dir and the blobs dir.

14. **SM12x is not SM100 for fp8_einsum recipe.** `compute_fp8_einsum_recipe`
    used `cap.major >= 10` to pick packed INT32 UE8M0 TMA scales. GB10 is
    major 12, so `fused_inv_rope_fp8_quant` emitted int32 packed scales and
    the Python `fp8_einsum` fallback did `.to(float32)` on those ints.
    o_proj was noise. SM12x must use SM90 FP32 128x128 scales
    (    `tma_aligned_scales=False`, recipe `(1,128,128)`). UE8M0 bytes are
    `2^(e-127)` (byte 0 is 0.0).

15. **CUDA graphs on the rc2 overlay (2-node GB10).** Host-staged PYNCCL
    cannot live inside a breakable CUDA graph. PIECEWISE does not break at
    NCCL; `RowParallelLinear.down_proj` captured `tensor_model_parallel_all_reduce`
    inside GEMM segments. That produced `Actors`/`ligands` with `' Paris'`
    at -9999, or 1e33 dummy logits. FULL capture also exploded `lm_head`
    (`w_rms=inf`, `l_max~1e33`, sampler -ln(96)) and later `cudaGraphLaunch`.
    DSpark FULL graphs wrap `_generate_draft` on the shared target `lm_head`;
    overlay skips DSpark capture. Overlay fix: lazy `capture.add_eager`
    around TP all-reduce, clone the buffer off the CUDA graph pool, `copy_`
    back. Weak-ref into graph-pool memory died in `ncclAllReduce` on replay.
    Top-level import of `breakable_cudagraph` from `communication_op.py`
    circular-imports `vllm.config`. Overlay pin:
    `ENFORCE_EAGER=0` `CUDAGRAPH_MODE=PIECEWISE`. Fallback `ENFORCE_EAGER=1`.
    Matched-main is learning 23.

16. **spark2 earlyoom** prefers `comm=vllm` and SIGTERM at MemAvailable <8%
    (~10 GiB). SwapTotal is 0 (`/swap.img` exists, not enabled, no
    passwordless sudo). util 0.85 died; 0.81 eager sits around 10–11%
    after warmup.

17. **Never chain `07-stop.sh` and `05-serve.sh` in one SSH.** Stop,
    confirm containers gone, then serve. `05-serve.sh` is `docker run -d`.
    Separate SSH per node (`ControlPath=none`). Do not reboot the Sparks.

18. **Gather of packed-at-store indexer K is numerically wrong.** Store
    overlay packs four 64-token pages per 256-token manager block. FlashInfer
    gather of that layout still prints Paris but DSpark accept swings
    38–70% (pin ~73%) and 1-way hitch to 16 tok/s. Keep the paged kernel.
    Dual packed+interleaved sidecars OOM spark2 (~3.7 GiB). **FIXED
    2026-08-24:** `packed_gather_mqa_logits`
    (`patches/files/sm12x_b12x_kernels.py`, overlay `mqa-packed-gather`) reads
    the packed K-then-scale offsets; unit test passes 0.0 diff (verified
    inside the production image). The paged kernel stays the live decode
    path; the 1-way ≥30.6 gap is now a perf gap, not a correctness bug.

19. **DSpark 1-way is 6 tokens padded to 8.** Capture sizes skip 6.
    `uniform_decode_query_len = 1 + num_speculative_tokens` (6). Padding
    6→8 makes `8 % 6 != 0`, so 1-way uses the size-8 mixed graph. Adding
    6 to the list captured `table=(6, 1023)` and lost both 1-way and 8-way.
    `configs/pin.main.env` must not set `CUDAGRAPH_CAPTURE_SIZES`.

20. **b12x scheduled paged scorer is a 1-row win only on this pin.**
    `q_rows` 2–8 scheduled was slower than unscheduled 1023-page (1-way
    25.36). Planning inside CUDA graphs freezes warmup seqlens. Feeding
    the 1-row scheduled kernel into 8-row decode dropped 8-way to 16.29.
    Helper `_B12X_SCHEDULE_MAX_Q_ROWS = 1`. Overlay plans only `q_rows==1`.

21. **Unused page64 workspace steals KV.** Manager tables are already
    1024-wide (`width*64 >= max_model_len`). Allocating
    `max_num_batched_tokens x width*4` anyway dropped KV to 94,516.
    Skip that buffer: 97,737.

22. **Validate from spark1.** Workstation `nodes.env` uses Spark hostnames.
    `06-validate.sh` on the laptop hits the wrong `HEAD_IP`. Use
    `VALIDATE_STACK=main` on spark1 (`127.0.0.1:8000`).

23. **Matched-main FULL_AND_PIECEWISE keeps France.** Overlay FULL still
    dies (`Actors` / `的超` / `cudaGraphLaunch`). Main image uses
    `ar-piecewise-ws` (in-graph TP AR) plus DSpark backbone FULL with
    sample eager. Do not graph `_sample_sequential` (shared `lm_head`).
    Do not copy overlay "never FULL" onto the main pin.

24. **`preinitialize_invalid_logits=False` is not a 1-way win.** France
    stayed Paris but the string drifted (logprob -0.335) and 1-way fell
    to 15–19 tok/s. Leave the default.

---

## File map

```
vllm-spark-0731/
  HANDOFF.md                     # This file (live ops + fallback overlay)
  README.md                      # Matched-main quality gate
  configs/
    pin.main.env                 # live: main-b12x, FULL_AND_PIECEWISE, util 0.8
    pin.nvfp4.env                # overlay fallback: nvfp4_ds_mla, PIECEWISE
    pin.env                      # fp8_ds_mla parked
    env.spark.sh                 # Shared DGX Spark env (NCCL, UCX, RoCE)
    nodes.env                    # Per-node IPs and ranks (gitignored)
    nodes.env.example            # Template for nodes.env
    kv-offload.native.json       # Phase 6 native FS offload (not live)
    lmcache.gds.yaml             # Phase 6 LMCache GDS (not live)
  docker/                        # Matched-main Dockerfile
  docs/
    PLAN-MAIN.md                 # Build plan for main-b12x
    UPSTREAM.md                  # rc2 vs main vs open PRs
    LINEAGE.md                   # nvfp4_ds_mla / b12x lineage
  patches/
    apply_overlays.py            # rc2 + main overlays; --only incremental
    assert_image.py              # Build-time verification
    assert_0731.py               # Model checkpoint validation
    assert_stack.py              # Runtime stack consistency check
    pin_cutlass_dsl.py           # Rewrite b12x/quack cutlass 4.6.2 → 4.7.0
    files/
      dsv4_b12x_sparse.py        # B12X_MLA_SPARSE on 584 B DSV4 page
      sm12x_b12x_kernels.py      # paged indexer + WO bmm helper
      fused_moe_b12x.py          # B12xExperts (rc2 only)
      b12x_moe.py                # b12x weight prep (rc2 only)
  scripts/
    02-build-main.sh             # Matched-main image on current node
    02-copy-main.sh              # docker save | ssh spark2 docker load
    03-apply-main-overlays.sh    # Bake main overlays into the image
    05-serve.sh                  # Launch (fp8|nvfp4|eugr|main)
    06-validate.sh               # France + chat; VALIDATE_STACK=main
    07-stop.sh                   # Stop containers + clean shm
```

---

## Git history (recent)

```
c4c7e18 Add SM12x dequant fallbacks for MQA logits (prefill + paged decode)
d176361 Patch FlashInfer C++ DSV4 dispatch to add TOPK=192 for DSpark k=5
3f97e68 Fix assert class name: DeepseekV32IndexerMetadataBuilder
30655ab Guard indexer paged MQA logits against SM12x
9425f2f Exclude CUTLASS FP8 block-scaled MM on SM12x
5755346 Exclude SM12x from is_deep_gemm_supported()
b87f167 Fix dispatch table patch: disambiguate needle from DSV3_2 table
4efaeee Add (32,192) to FlashInfer DSV4 dispatch for DSpark k=5
bb790b7 Rename attention backend to FLASHINFER_MLA_SPARSE_DSV4 (rc2 rename)
359e50f Overlay rc2 on top of v0.27.1 instead of replacing (preserves third_party)
```

---

## Crash reference

The crash that drove the MQA logits fallback work:

```
RuntimeError: Assertion error (/workspace/.deps/deepgemm-src/csrc/apis/attention.hpp:122):
  (arch_major == 10 and (num_heads == 8 or num_heads == 16 or num_heads == 32 or num_heads == 64))
  or (arch_major == 9 and (num_heads == 32 or num_heads == 64))
```

Call path: `cudagraph_utils.py:capture` -> model forward -> `attention.py:617`
-> `sparse_attn_indexer.py:500` -> `fp8_fp4_mqa_logits` (deep_gemm.py:570 in
OLD image) -> DeepGEMM C++.

Fix: Python guard at the top of `fp8_fp4_mqa_logits` and
`fp8_fp4_paged_mqa_logits` checks `is_device_capability_family(120)` and
returns a pure-PyTorch dequant fallback. In the new image, the function is
at line 673 (shifted due to added helper functions).

The next crash, after that guard, during FULL cudagraph capture:

```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
  cudaErrorStreamCaptureUnsupported
```

Call path: `cudagraph_utils.py:capture` (FULL) -> `sparse_attn_indexer.py:603`
-> `fp8_fp4_paged_mqa_logits` -> `_sm12x_fp8_paged_mqa_logits` ->
`int(ctx.max().item())`.

Fix: gather length is `min(max_blocks * block_size, max_model_len)` from
Python shapes. Context is applied as a mask, never as a host-side size.
