[← Back to Knowledge Index](00-index.md)

# The golden image's speed: exact difference + lift plan (2026-08-24)

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
2. NVRTC JIT mode (`DG_JIT_USE_NVRTC=1` in `configs/env.spark.sh`): ptxas
   cannot assemble tcgen05 for sm_121a on any CUDA 13.x tested (13.0 and
   13.3), but NVRTC → driver-JIT works — the golden (a runtime image without
   nvcc) uses NVRTC implicitly.
3. `VLLM_USE_DEEP_GEMM_E8M0=1` (matches the golden).

No kernel porting, no vendored headers, no wrapper surgery needed. The
kernel-port work done earlier (`dg25_fp8`, wrapper, dispatch patch) served to
isolate the regression and is preserved as analysis, not shipped
(`patches/upstream/deepgemm-fp8-1d1d-port.diff` is analysis-only).

## Upstream legs

- **vLLM PR**: pin the cmake DeepGEMM tag from `8b1392b` back to `a6b593d`
  (branch `fix-deepgemm-sm12x-fp8-regression` on maci0/vllm, commit c43c627).
- **DeepGEMM issue**: [deepseek-ai/DeepGEMM#417](https://github.com/deepseek-ai/DeepGEMM/issues/417)
  filed (2026-08-25) with the regression evidence.

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

---

## Related Docs

- [00-index.md](00-index.md) — Knowledge Base index & overview
- [03-kernels-attention.md](03-kernels-attention.md) — b12x vs FlashInfer vs DeepGEMM kernels
- [05-performance.md](05-performance.md) — Concurrency scaling & throughput benchmarks
- [06-deployment.md](06-deployment.md) — Image build, lineage, and golden runbooks
