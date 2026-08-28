[← Index](00-index.md) · [Glossary](glossary.md)

# Golden Image Analysis: Speed Gap, Root Cause & Lift Plan

> **Scope:** Why the anemll golden image is ~2.5× faster, and the DeepGEMM pin-back (`a6b593d`) that fixes the SM12x fp8 regression.

Why `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` serves France c1 at **65.2 tok/s**
while our matched-main (vLLM `e25c586b9`, Aug main) serves ~25.8 — and what
to port upstream. All claims below were verified on this cluster, not quoted.

## Measured kernel stacks

| Component | Golden (anemll v0.25.2) | matched-main (Aug main) |
|-----------|--------------------------|-------------------------|
| vLLM core | `0.25.2.dev0+g752a3a504` (2026-07-14) | `e25c586b9` (2026-08-23) |
| torch | 2.11.0+cu130 | 2.14 source (12.1a) |
| **FP8 linear** | **DeepGEMM** (`Selected DeepGemmFp8BlockScaledMMKernel`, UE8M0 + PDL) | b12x `B12xFp8BlockScaledMMKernel` (pinned) |
| MoE | B12xExperts (b12x **0.15.3**) | B12xExperts (b12x 1.2.6) |
| Attention | FlashInfer sparse MLA + **real NVFP4 writer** | B12X_MLA_SPARSE / FlashInfer (fp8 alias) |
| mHC | TileLang 0.1.9 | TileLang |
| Triton | tokenspeed fork 3.8.10 | triton 3.7.x |
| DSpark accept | ~71% mean len 4.5 (Eagle3 aux layers 41,42,43) | ~73% (stock draft) — **not the difference** |
| KV | 7,650 B/token NVFP4 | 584 B fp8 envelope |

## What the golden's DeepGEMM actually is

The golden ships a **fork of DeepGEMM, self-labeled `2.5.0`**, that has no
upstream release (deepseek-ai/DeepGEMM tags end at `v2.1.1.post3`, 2025-10;
`__version__ == 2.5.0` only in the golden wheel). Evidence:

- Its kernel headers (61 files) differ from the vLLM-vendored `nv_dev
  8b1392b` tree (55 files): it **adds** `sm100_fp8_gemm_1d1d.cuh`,
  `sm100_fp8_mqa_logits.cuh`, `sm100_fp8_paged_mqa_logits.cuh`,
  `sm100_fp4_*` variants, and `common/` files (`scheduler.cuh`,
  `sm90_utils.cuh`, `sm100_utils.cuh`, `tma_utils.cuh`,
  `epilogue_utils.cuh`, `reduction.cuh`, `types.hpp`).
- None of those files exist upstream (GitHub code search: zero hits).
- The golden's 1,762-kernel linear warmup runs in **<1 s (2,800 it/s =
  precompiled cubins)**; ours takes **35 s (nvcc JIT at boot)** — the
  golden compiled its kernels at image build.

## Why DeepGEMM linear is broken on SM12x in *newer* upstream

Experiment log (all on matched-main, France temp 0, 128 tok):

| Config | c1 tok/s | France output |
|--------|----------|---------------|
| `LINEAR_BACKEND=b12x` (pin) | 25.8 | `' Paris…'` coherent |
| `LINEAR_BACKEND=deep_gemm`, E8M0=0 | 4.4 | `' Septy Septy…'` garbage |
| `LINEAR_BACKEND=deep_gemm`, E8M0=1 | 4.4 | garbage |

The vLLM-side Python integration is **byte-identical** between 0.25.2 and
Aug main (`DeepGemmFp8BlockScaledMMKernel`, the `fp8_gemm_nt` wrapper, and
`fp8_utils.py` differ only in our own SM12x fallbacks). The bug is native,
and it is precise:

1. **`nv_dev 8b1392b` has no pure-fp8 1d1d kernel for SM100/SM120**
   (impls/ only has `sm90_fp8_gemm_1d1d.hpp`; the golden's fork adds
   `sm100_fp8_gemm_1d1d.cuh`).
2. **The export aliases pure-fp8 to the fp4 dispatcher**:
   `csrc/apis/gemm.hpp:851` → `m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt")`.
   On `arch_major == 12` that dispatcher runs `sm120_fp8_fp4_gemm_1d1d`
   regardless of B's dtype — an fp8 weight tensor fed to an fp4 kernel =
   **silent corruption** (repro: DSV4-Flash linear on SM121a, France
   `' Septy…'`, 4.4 tok/s).
3. The SM120 SF-layout fix ([#403](https://github.com/deepseek-ai/DeepGEMM/pull/403))
   is **already in nv_dev 8b1392b** (`layout.hpp:49/57/112` have
   `arch_major == 12`) — not the bug on the vLLM pin (comment posted
   correcting our earlier field report). Related: PR #415 (SM90 k-grouped
   silent corruption) is a separate path.

Wrong linear output → every DSpark draft rejected → decode collapses to
~1 token/step → the 4.4 tok/s and garbage text.

A wholesale swap of the golden's headers into our wheel **fails to compile**
(`NVCC compilation failed`, `compiler.hpp:234`) — the golden's headers and
its `.so` are a matched pair; its full binary is built against torch 2.11.

## Lift / upstream plan

1. **Port `sm100_fp8_gemm_1d1d`** from the golden wheel's headers (they are
   the JIT sources) into `nv_dev` as
   `csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp` + its `common/` deps
   (`scheduler`, `sm100_utils`, `epilogue_utils` + the shared `utils`
   diffs), mapped from the golden's `include/deep_gemm/**` layout.
2. **Fix the dispatch** in `csrc/apis/gemm.hpp`: give `fp8_gemm_nt` a
   dtype-aware path so pure fp8×fp8 on SM100/SM120 runs the new kernel
   instead of the fp8×fp4 alias.
3. **PR both** to deepseek-ai/DeepGEMM `nv_dev` (crediting the anemll build
   as the source), then a **vLLM PR** bumping the cmake pin from `8b1392b`
   to the commit that lands them (no vLLM Python changes needed).
4. Validation loop on this cluster: build via `DEEPGEMM_SRC_DIR` (patches
   applied in `docker/Dockerfile.main`, idempotent loop), boot
   `LINEAR_BACKEND=deep_gemm`, France must be coherent and ≥ b12x's 25.8 c1.

**Not liftable as binaries:** the golden's `.so` (torch 2.11 ABI, no
source). **Not the lever:** attention backend (our A/B was parity), DSpark
acceptance (parity ~71-73%), config (`max_num_seqs`).

## The actual difference (resolved 2026-08-25)

The golden image's build is public:
[github.com/Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10).
Its Dockerfile is `FROM vllm/vllm-openai:v0.25.1` + a **Python-only overlay**
(no DeepGEMM rebuild, no custom toolchain). vLLM v0.25.1's cmake pins
DeepGEMM nv_dev **`a6b593d`** — the "last known good" commit eugr froze, and
it **has** `sm100_fp8_gemm_1d1d` + a correct `fp8_gemm_nt` dispatch.

Our matched-main pins **`8b1392b`** (2 commits later), which regressed SM12x
fp8: it dropped the pure-fp8 1d1d kernels and aliased
`fp8_gemm_nt = fp8_fp4_gemm_nt` (`gemm.hpp:851`) — an fp8 weight tensor fed
to an fp4 kernel = the silent corruption we measured (`' Septy'`, 4.4 tok/s).

The "golden fork 2.5.0" was a red herring: the wheel's `__version__` label
and its extra headers are just a6b593d's tree as packaged by vLLM 0.25.1.

## The fix (minimal, applied)

1. `docker/Dockerfile.main`: `DEEPGEMM_COMMIT=a6b593d` (done).
2. NVRTC JIT mode was tried (`DG_JIT_USE_NVRTC=1`) because ptxas cannot
   assemble tcgen05 for sm_121a on any CUDA 13.x tested (13.0 and 13.3).
   **It did not land** — see Validation status below: NVRTC cubins were
   rejected by the driver (`CUDA_ERROR_INVALID_IMAGE`). `DG_JIT_USE_NVRTC`
   stays **0** (`configs/pin.main.env`, `docs/PLAN-MAIN.md` §4.4).
3. `VLLM_USE_DEEP_GEMM_E8M0=1` (matches the golden).

No kernel porting, no vendored headers, no wrapper surgery needed. The
kernel-port work done earlier (`dg25_fp8`, wrapper, dispatch patch) served to
isolate the regression and is preserved as analysis, not shipped
(`patches/upstream/deepgemm-fp8-1d1d-port.diff` is analysis-only).

## Upstream legs (shipped 2026-08-25)

- **vLLM PR [#53680](https://github.com/vllm-project/vllm/pull/53680)**: pin
  the cmake DeepGEMM tag from `8b1392b` back to `a6b593d` (branch
  `fix-deepgemm-sm12x-fp8-regression` on maci0/vllm).
- **DeepGEMM issue [deepseek-ai/DeepGEMM#417](https://github.com/deepseek-ai/DeepGEMM/issues/417)**:
  the `8b1392b` SM12x fp8 regression (removed kernels + fp4 alias).
- In-repo: `DEEPGEMM_COMMIT=a6b593d` in `docker/Dockerfile.main` +
  `configs/pin.main.env`.

## Validation status (2026-08-25)

The pin change + a6b593d rebuild were delivered, but **boot validation on GB10
remains blocked at the JIT toolchain level**: with the stock CUDA 13.3 runtime,
no route produces a loadable tcgen05 kernel image for the fp8 path —

- nvcc/ptxas: `tcgen05.mma`/`.cta_group::1`/`.block32` rejected for every
  sm_12x target tested (13.0 and 13.3 toolchains).
- NVRTC: compiles clean after cccl/STL/device-runtime fixes, but the driver
  rejects the resulting cubins (`CUDA_ERROR_INVALID_IMAGE`), and further
  NVRTC-frontend constexpr failures appear in the a6b593d SM120 swizzle
  templates under 13.3.
- The golden image works only on its self-consistent v0.25.1 + torch 2.11 +
  CUDA 13.0 stack; its compiled `.so` is torch-2.11-ABI and cannot load in
  our torch 2.14 runtime.

Cluster restored to the known-good serving state (`main-b12x-orig`, b12x
linear, France coherent, ~25.8 tok/s). The production fix is the pin change
(the regression is upstream); the GB10 JIT toolchain gap is a separate
upstream issue (see docs/UPSTREAM.md DeepGEMM section).

Rollback tags on the nodes: `vllm-spark-0731:main-b12x-orig` = known-good
(b12x linear) image; `main-b12x-dg25h` = header-swap experiment (broken).

## Probe findings (2026-08-26, SM121a, main-b12x image + a6 wheel 2.5.0)

A/B of the a6 wheel (2.5.0) vs the 8b-era wheel (2.6.1) on the same image:

- **`fp8_einsum` "bhr,hdr->bhd" (DSV4 o_proj): works on BOTH wheels.** Real
  DeepSeek-V4-Flash-0731 shapes (G=8, D=4096, o_lora_rank=1024), T=10/96/8192
  → host layout checks pass, output finite. This is NOT the regression surface.
- **`fp8_gemm_nt` pure-fp8 pair path (DSV4 linear): broken on BOTH.**
  a6 wheel 2.5.0 → NaN output; ported 2.6.1 (`dg25_fp8` pure-fp8 kernel) →
  `CUDA error: unspecified launch failure`. Called exactly like vLLM
  (`deepgemm_post_process_fp8_weight_block` + `fp8_gemm_nt(..., disable_ue8m0_cast=False)`).
  M=N=512, K=4096, block scales (128,128).
- **NVRTC 13.3 cannot compile the `sm100_fp8_gemm_1d1d` TU at all** — every
  variant (with/without math.cuh preamble fix, `--std=c++20`, orig TU) returns
  rc=6 with a log containing only warnings, no error text. The earlier
  "NVRTC compiles" note does not hold for this TU on 13.3.
- **The a6 pure-fp8 headers are latently broken**: `sm100_fp8_gemm_1d1d.cuh`
  never includes `common/math.cuh`, so `align`/`swap` (used unqualified in
  `common/scheduler.cuh:192`, `common/sm100_utils.cuh:127`) are undefined, and
  `math.cuh`'s `cast_into_bf16_and_pack` needs `cuda_bf16.h`. The TU is only
  fixable by fixing the headers (include math.cuh + `using namespace
  deep_gemm::math` before the common includes + `#include <cuda_bf16.h>`).
  Our local port (`deepgemm-fp8-1d1d-port.diff`) does exactly this by
  re-homing the kernel under `deep_gemm/dg25_fp8/` with fixed headers.
- **Regression mechanism (corrected 2026-08-26)**: `fp8_gemm_nt =
  fp8_fp4_gemm_nt` is NOT new (exists in a6b593d, gemm.hpp:792). 8b1392b
  removed `sm100_fp8_gemm_1d1d.{hpp,cuh}` and rewrote the `fp8_fp4_mqa_logits`
  dispatch + `fp8_fp4_gemm_nt_sm120` (AB-swap). Full-stack measurement remains
  the ground truth: 8b pin → France `' Septy…'` ~4.4 tok/s; a6 pin → coherent.

Implication for the upstream DeepGEMM PR: restoring the pure-fp8 path means
re-adding the 1d1d kernel WITH the header fixes (the `dg25_fp8` port shape),
not a verbatim a6b593d resurrection — the a6 headers do not compile as-is.

**Upstreamed 2026-08-26: [deepseek-ai/DeepGEMM#419](https://github.com/deepseek-ai/DeepGEMM/pull/419)**
(restore patchset against nv_dev: dg25_fp8 kernel module + gemm.hpp
fp8xfp8 routing + TU header fixes + NVRTC `cute::is_same_v`).

**Driver-JIT PTX route — tested and blocked (2026-08-26)**: with the fixed
headers, NVRTC 13.3 compiles the 1d1d TU to PTX for `compute_121a` (rc=0,
~231 KB). Every load route fails on the driver/toolkit: ptxas 13.3 rejects
`tcgen05.mma`/`.cta_group::1`/`.block32` for `sm_121a`; `nvrtcGetCUBIN`
cubins → `CUDA_ERROR_INVALID_IMAGE` (200); driver-JIT `cuModuleLoadDataEx`
of the compute_121a PTX → `CUDA_ERROR_INVALID_PTX` (218); `--ptx` is not a
valid NVRTC option and `sm_121a` targets run ptxas inside NVRTC (rc=6). The
golden works because its stack precompiles the kernels (build-time toolchain
or cache); on stock CUDA 13.x runtime, runtime JIT of tcgen05 kernels for
sm_121a is impossible → the a6b593d pin (#53680) / build-time compilation is
the working path.

## Full-boot validation (2026-08-26): deep_gemm linear on GB10 — garbage

`vllm-spark-0731:main-b12x-a6dg/a6fp32` (current image + a6b593d wheel +
the #53521-fixed o_proj) booted with `--linear-backend deep_gemm` (a6dg:
packed scales; a6fp32: FP32 scales): **health 200, but France is garbage**
(`'�carecarecare…'`). Layer-by-layer numeric validation (full
`deep_gemm_fp8_o_proj` vs an fp32 torch reference, real DSV4-Flash-0731
shapes, T=16):

| activation scales | max abs err | verdict |
|---|---|---|
| FP32 (`tma_aligned_scales=False`) | 9e-5 | **correct** |
| INT32-packed UE8M0 (`True`) | 5.7e9 (~2³²) | **wrong** (misread lanes) |

→ The a6 einsum is numerically correct with **FP32** scales (the a6fp32
image), yet the full model is still garbage — so **another layer is also
wrong** (prime suspect: `fp8_gemm_nt` linear scales, the same
producer/kernel scale-layout mismatch class). Conclusion: the a6 pin restores
the **dispatch** (no fp4 misread) but vLLM **main's scale producers do not
match what the a6 kernels expect** — the golden's 0.25.1-era producers did.
The remaining work is producer-layout alignment, validated layer-by-layer.

### Exhaustive boot matrix (2026-08-26, full 0731, deep_gemm linear)

| config | health | France |
|---|---|---|
| E8M0=0, instanttensor, packed o_proj | 200 | `'�carecarecare…'` |
| E8M0=0, instanttensor, FP32 o_proj (a6fp32) | 200 | `'�carecarecare…'` |
| E8M0=1, instanttensor | 200 | `' Septy Septy Septy…'` (known-bad) |
| E8M0=1, raw `--load-format safetensors` | 200 | `' Septy Septy…'` |

→ Load format (instanttensor vs safetensors) is **not** the cause. Both E8M0
modes are wrong.

### Tiny-model A/B (decisive): deep_gemm ≠ b12x on the same model

`yujiepan/deepseek-v4-tiny-random` served with `--linear-backend b12x` vs
`deep_gemm` (same prompt, greedy): outputs **diverge** after the first token
(`ójój…` common prefix, then different text). **CORRECTION (deep-dive
2026-08-26): this divergence is NOT proof of a kernel bug** — the tiny model
has random weights, so fp8 rounding differences amplify chaotically through
7 layers and produce different greedy tokens even when both paths are within
fp8 tolerance. The per-op replays are the valid measure:

- **fp8_gemm_nt with the REAL producer layouts (captured from a live tiny
  deep_gemm serve via a sitecustomize hook): CORRECT** (0.24 % err vs an fp32
  reference). Scales: `sfa/sfb [M,1] int32` packed ue8m0 (K=256), MN-major
  strides `(1, M)`, `disable_ue8m0_cast=False`.
- **fp8_gemm_nt with packed `[M,8]` int32 scales at K=4096** (the full-model
  granularity, real producer strides `(1, M)`): matches the reference.
- **o_proj einsum with FP32 act scales (the fd67094 fix): CORRECT**; with the
  GROUPED packed-UE8M0 case (`[T,G,8]` int32, the producer's
  `tma_aligned_scales=True`): WRONG (~2³²) — the one proven kernel-level
  mismatch, covered by the FP32 fix.

So every individually-validated op is correct, yet the full 0731 model with
deep_gemm is still garbage in all 4 boot configs. The remaining suspect is
full-model integration (DSpark k=5 draft path, or the b12x-MoE × deep_gemm
linear interaction), which needs **layer-level bisection** (capture each
layer's output in the deep_gemm boot vs the b12x boot and find the first
divergence) — next step once the cluster is back (spark2 was down
2026-08-26 evening).

### E8M0/transform assert (raw safetensors load)

With a raw safetensors load (any backend), the weight post-process calls the
DeepGEMM SF transform with `disable_ue8m0_cast=True` (E8M0=0) → a6 wheel
asserts `layout.hpp:49/50: not disable_ue8m0_cast`. `VLLM_USE_DEEP_GEMM_E8M0=1`
is required for raw-load boots (the full model's instanttensor load skips the
post-process, which is why the E8M0=0 full boot got past it).

### Cluster restore note (2026-08-26)

Restoring after the test cycles: **strictly** `07-stop.sh` on both nodes →
boot worker (spark2) and let it settle ~60 s → boot head (spark1). Out-of-sync
restarts cause NCCL `DistNetworkError` pairing failures and a stuck head;
`gb10-clockcap` also died (137) during the cycles — restart with
`docker start gb10-clockcap` on both nodes.

### Layer bisection (2026-08-26, tiny model, spark1-only)

Captured per-layer hidden states + MoE gate logits in the tiny deep_gemm vs
b12x boots (sitecustomize hooks on `DeepseekV4DecoderLayer.forward` and
`DeepseekV4MoE.forward`):

- **Layer outputs diverge from L0** (~50 % relative at every layer). L0's
  gate logits are **identical** (top-6 expert selection 100 % match) → the
  L0 hidden-state difference comes from the **attention projections or the
  (identically-routed) experts**, not the routing. From L1 on the gate
  logits differ proportionally to their (diverged) inputs and the top-6
  selection flips (0-15 % match) — the random-weight tiny model is
  routing-sensitive, so fp8-tolerance projection differences amplify to
  garbage. This makes the earlier "kernel-level divergence" reading of the
  tiny A/B doubly wrong: ops are correct; the divergence is amplification.
- **fp8_gemm_nt packed [M,8] int32 at K=4096 (clean reference): CORRECT**
  (0.24 %) — the multi-int32-per-row packed path works; the full-model
  K=4096 linear scales are read correctly. The granularity hypothesis is
  refuted for fp8_gemm_nt.

**Remaining question**: the full 0731 model's garbage with deep_gemm is NOT
explained by any individually-validated op (all correct within fp8
tolerance) — it needs the **full-model layer bisection** (same hooks, full
checkpoint, dg vs b12 boot) to find the first really-diverging layer. That
is blocked until spark2 is back (down since 2026-08-26 evening; needs a
manual reset).

### Shared-experts localization (2026-08-26, tiny model, fixed hooks)

Hook corrections: the layer input must be `args[0]` (earlier captures used
`args[1]` = positions → zero records). With correct captures:

- **L0 input IDENTICAL, L0 post-attention (`moe_in`) IDENTICAL, L0 gate
  IDENTICAL, yet the MoE output differs** — the divergence is **inside the
  MoE**, not attention or routing.
- Splitting the MoE (hook `DeepseekV4MLP.forward` = shared experts):
  **the SHARED experts' output differs ~60-100 %** (0.11/0.05/0.14 abs on
  0.12-0.18 mag) and exactly matches the layer-output deltas. The routed
  experts path shows no difference.
- Captured ALL deep_gemm ops in the dg boot (fp8_gemm_nt / bf16 / einsum /
  m_grouped): every call is `fp8_gemm_nt`, all replay **CORRECT** vs an
  fp32 reference using the ue8m0 convention 2^(e-127). The shared MLP's
  `down_proj` (K=512) is **not** a deep_gemm op (identical in both boots).
- **Conclusion**: with identical inputs, a replay-correct gate_up, and an
  identical down_proj, the shared-expert output should match — it does not.
  The residual difference is a **scale-convention incompatibility between
  the deep_gemm fp8 kernels and the b12x/checkpoint ue8m0 semantics** that
  an fp32 reference with the standard bias cannot distinguish (deep_gemm and
  the reference agree; b12x differs; the b12x full model is the coherent
  one). This is the localized root cause of the deep_gemm-path garbage:
  the deep_gemm ue8m0 scale decode must be aligned to the checkpoint/b12x
  convention (a DeepGEMM-kernel or vLLM-producer fix).

### Convention verdict (2026-08-26, torch-reference comparison, tiny model)

Same-pass `moe_in` is identical across boots. Computed the shared-MLP output
exactly in torch (e8m0fnu dequant; verified torch's `e8m0fnu → float` is
`2^(e-127)` and deep_gemm's decode matches it) and compared both backends:

| backend | vs torch reference (L0 shared output) | verdict |
|---|---|---|
| deep_gemm | err 0.0055 (3 %, fp8 noise) | **matches the torch/standard convention** |
| b12x | err 0.1024 (59 %) | differs |

So deep_gemm decodes ue8m0 with the **standard** convention and is *right*
on the tiny model; b12x uses a different convention and is *wrong* on the
tiny model — yet b12x is coherent on the **full** 0731 model. Conclusion:
the **full checkpoint's scales were quantized with the b12x convention, not
the torch-standard** — deep_gemm (standard) decodes the real scales wrong →
garbage. The fix target is the ue8m0 scale convention (bias/lane order) the
deep_gemm kernels use vs what the real checkpoint encodes.

### Convention CONFIRMED on the real checkpoint (2026-08-26, boot-free)

Loaded the full 0731 checkpoint's `layers.0.ffn.shared_experts.w1`
([2048,4096] fp8 + [16,32] e8m0fnu, bytes 0x73 = 2⁻¹², standard layout —
no swizzle), ran vLLM's post-process (`deepgemm_post_process_fp8_weight_block`
→ packed [2048,8] int32) + `fp8_gemm_nt` vs the exact torch dequant:

```
full shared w1 replay: err 0.268 vs ref ~1.78 -> WRONG (~15 %)
```

**deep_gemm misdecodes the REAL checkpoint's shared-expert scales by ~15 %**
— the convention mismatch is confirmed directly on the real model, no boot
needed. (The tiny model's scales happened to be standard → deep_gemm right
there; the real checkpoint's differ.) Fix target: deep_gemm's packed-UE8M0
scale decode (sm120 fp8 kernel) vs the checkpoint's semantics — a
DeepGEMM-kernel or vLLM-producer-side alignment, reproducible via the
boot-free replay above.

**Lane-verdict (2026-08-26)**: with a clean fp32 scale pattern (block (0,1) =
2¹, rest 2⁻¹⁵) packed to `0x70708070` (2¹ correctly in lane 1), the kernel
applies the **correct lane** (block-1-only input matches the 2¹ reference on
samples) — the ~15 % error is a **magnitude attenuation on peak entries**
(kernel output max ~26 % below the fp32 reference), not a lane-order bug.
The tiny model's uniform scales masked it entirely. Mechanism: the producer's packed-UE8M0 pre-transform layout is misread by
`fp8_gemm_nt` for varied scales. **FIX (verified boot-free 2026-08-26)**:
SM12x skips the packed pre-transform in
`deepgemm_post_process_weight_scale_block` and returns raw FP32 block
scales — the full shared-w1 replay drops from 15% WRONG to **0.27% CORRECT**.
Landed as vLLM **#53898** (backport `pr-53898.diff`).

**One-spark end-to-end validation (2026-08-27)**: the `main-b12x-a6fix`
image (a6fp32 + patched `fp8_utils.py`) boots the tiny model with
`--linear-backend deep_gemm` on a single spark (TP=1) and the captured
L0 shared-experts output matches the exact torch reference: err 0.0055 vs
~0.175 = **3% CORRECT** (fp8 noise) — the fix works through the real
serving image. The full 2-node 0731 boot remains pending a stable spark2.

## Validation vehicle: tiny 1:1 DSV4 model

**`yujiepan/deepseek-v4-tiny-random`** (7 layers, hidden 256; downloaded to
`/home/maci/models/deepseek-v4-tiny-random` on spark1, 277 MB) — layout
fidelity with DeepSeek-V4-Flash-0731:

- `model_type=deepseek_v4`, `DeepseekV4ForCausalLM`, `quant_method=fp8`,
  **`scale_fmt=ue8m0`, `weight_block_size=[128,128]`**, `expert_dtype=fp4`
- MLA: `q_lora_rank=128`, `o_lora_rank=128`, `head_dim=512`,
  `qk_rope_head_dim=64` (448/64 nope/rope split — same as 0731), `o_groups=2`
- sparse attention (`sliding_window=128`), DSpark (`num_nextn_predict_layers=1`)

Use it to numerically validate each DeepGEMM op (einsum, `fp8_gemm_nt`,
MLA logits) against torch references with real model tensors — the producer
must be aligned until every op matches. Boot cycle is ~1-2 min (vs ~4 min +
155 GB for the full checkpoint). NOTE: base is V4-Pro, not Flash-0731 — the
quant/MLA geometry matches; verify `nvfp4_ds_mla` envelope + DSpark shapes
per op before trusting a pass.

---

## Related Docs

- [00-index.md](00-index.md) — Knowledge Base index & overview
- [03-kernels-attention.md](03-kernels-attention.md) — b12x vs FlashInfer vs DeepGEMM kernels
- [05-performance.md](05-performance.md) — Concurrency scaling & throughput benchmarks
- [06-deployment.md](06-deployment.md) — Image build, lineage, and golden runbooks

### Raw evidence (field notes)

- [`../field-notes/dgx-spark/GOLDEN.md`](../field-notes/dgx-spark/GOLDEN.md) — the shipped golden deployment: recipe, harness, three-lineage comparison

---

**[← Prev](08-upstream.md) · [Glossary](glossary.md) · [Next](10-operations-agents.md) →**
