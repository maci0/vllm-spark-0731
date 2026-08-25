# Plan: 0731 on vLLM main (matched build)

Research date: 2026-08-23.

This is the build plan. It is not a migration of the live 0.27.1 overlay
image. Overlay France/Paris quality is documented in [HANDOFF.md](../HANDOFF.md).
Matched-main Phase 3 passed 2026-08-23 on `vllm-spark-0731:main-b12x`
(`B12X_MLA_SPARSE` + `nvfp4_ds_mla`, util 0.8, PIECEWISE).

Tracker for open PRs: [UPSTREAM.md](UPSTREAM.md).
Live overlay ops for the v0.27.1 + rc2 fallback: [HANDOFF.md](../HANDOFF.md).

## 1. Goal

Serve `deepseek-ai/DeepSeek-V4-Flash-0731` on 2x DGX Spark (GB10, SM 12.1,
family 120, 128 GiB UMA, TP=2 RoCE) from a **matched** vLLM main tree:
Python and compiled `.so` from the same commit, DeepGEMM `nv_dev`, CUDA
**13.3.1** devel, PyTorch **2.14 built from source** for `12.1a`, latest b12x,
surgical SM12x patches only.

Features in scope:

- DSpark k=5
- Full b12x (linear FP8 + MXFP4 MoE + `B12X_MLA_SPARSE` attention)
- KV `nvfp4_ds_mla` (584 B DSV4 envelope) and stock `fp8_ds_mla`
- Fast cold start: InstantTensor (primary) / fastsafetensors GDS (fallback)
- No full reload on every iteration (caches + hybrid draft load)
- KV offload to NVMe via GDS (LMCache) or vLLM `OffloadingConnector` FS tier
- Optional later: runtime MoE expert paging to SSD
- Test util **0.8**

Out of scope for the first green image: FULL CUDA graphs, util 0.85,
DeepGEMM MoE as the expert kernel, overlaying main Python onto the
v0.27.1 `.so`. `B12X_MLA_SPARSE` is now the live attention backend on
the matched-main image (Phase 3).

## 2. Success bar (do not weaken)

Greedy `"The capital of France is"`, `temperature=0`, `max_tokens=32` is
coherent English. Proven string on the overlay image:

` Paris. The capital of Spain is Madrid. The capital of Italy is Rome. ...`

First token `' Paris'` logprob about -0.24 to -0.27, n_tie=1. Chat answers
`Paris.`. `scripts/06-validate.sh` is the gate. Matched-main live numbers
are in [HANDOFF.md](../HANDOFF.md) (2026-08-23 16:29 UTC).

Do not declare the main image done on tok/s or KV tokens alone.

## 3. Hardware facts that drive every flag

| Fact | Consequence |
|------|-------------|
| 128 GiB **UMA** per node (HBM = DRAM) | CPU "offload" of KV or experts does **not** free memory. Only NVMe (or another node) does. |
| Weights ~77.7 GiB/rank, 155 GiB checkpoint | 0731 **fits** at TP=2 without expert SSD. Expert paging is extra KV headroom, not a load requirement. |
| spark2 earlyoom SIGTERM at MemAvailable <8% (~10 GiB), prefers `comm=vllm` | util 0.85 died. Overlay live 0.81. **Test at 0.8.** SwapTotal 0. |
| ConnectX-7 RoCE `10.0.1.1/2`, NCCL IB GID 3 | Keep `PP_SIZE=1`. fastsafetensors + DSpark + **PP>1** hangs (WORLD broadcast). |
| SM 12.1, **CUDA 13.3.1 devel in the image** | Compile torch, NCCL, vLLM, FlashInfer, DeepGEMM with `12.1a`. Official Hub `nightly` is still cu129 + arch `12.0`. Factory DGX OS is CUDA **13.0** / R580. See section 4.1. |

## 4. Pins (2026-08-23)

"Latest" means git default branch, or newest PyPI / prerelease that still
has DSpark TOPK 192 and SM12x. Re-pin every SHA at build start.

| Component | Pin | Why this, not a different head |
|-----------|-----|--------------------------------|
| vLLM | `main` (HEAD 2026-08-23 `a3561ef8e49d`) | Matched Python + `.so`. Re-pin SHA at build start. |
| CUDA toolkit (image) | **13.3.1** (`nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04`) | Latest CUDA on Hub/NGC for ubuntu24.04 (arm64). Not vLLM default 13.0.3. No `nvidia/cuda:13.4-*` tag. Not ubuntu26.04. |
| PyTorch | **source** `release/2.14` (`2.14.0` RC, GA 2026-09-02) | Compile in the CUDA image with `TORCH_CUDA_ARCH_LIST=12.1a`. Official wheels stop at cu132 and ship `12.0+PTX`, not `12.1a` tensor-core kernels ([forum](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744)). No NGC bundled torch. Record SHA. |
| torchvision / torchaudio / triton | **vision `release/0.29` from source** / skip audio if it fights / **triton 3.7.1** pip + `TRITON_PTXAS_PATH` | Must match the local torch ABI. PyPI has no `triton==3.8.0` (nightly-only). Do not pip cu132 torch wheels over the source torch. |
| Build method | Custom Dockerfile on Spark, eugr-shaped: CUDA devel → NCCL → torch → `use_existing_torch.py` → vLLM / FlashInfer / DeepGEMM | Not vLLM two-stage manylinux. Not NGC pytorch. Not `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0). |
| DeepGEMM | **nv_dev** HEAD `8b1392b978f5` (2026-08-11, vLLM cmake tag) | Already nv_dev tip. DeepGEMM **main** has no SM12x. |
| DeepGEMM A/B | eugr freeze `a6b593d28267` (2026-06-29) | MXFP4 grouped-scale regression at `f8e8fb5` / [#384](https://github.com/deepseek-ai/DeepGEMM/pull/384). Measure 0731 experts before trusting cmake. |
| DeepEP | git **main** `01dc3aaac820` (2026-08-04) if the Dockerfile builds it | vLLM default `DEEPEP_COMMIT_HASH=d4f41e4e93` is stale. 0731 is TP=2, not EP; skip the compile if the image does. |
| b12x | git **master** `local-inference-lab/b12x` (HEAD `36bce2c1552b`, package **1.2.6**) | No PyPI prerelease. Rewrite cutlass metadata. `--no-deps`. Record SHA. |
| nvidia-cutlass-dsl | **4.7.0** (`[cu13]`) | PyPI latest. Rewrite b12x **and** quack-kernels pins (`patches/pin_cutlass_dsl.py`). |
| quack-kernels | **0.6.4** + same 4.7.0 rewrite | Latest. Still `nvidia-cutlass-dsl==4.6.2` in metadata (vLLM comment: "Required for CUTLASS DSL 4.6"). `--no-deps` after rewrite. expected-count **2**. |
| FlashInfer | git **main** (HEAD 2026-08-22 `fb28d7242b35`), not PyPI **0.6.17** | #4380 TOPK 192/256 is on main. Latest tag `v0.6.18rc7` (`3b3e688`) is **behind** main. vLLM docker still 0.6.17 (`{128,512,1024}`). Verify `_DECODE_DSV4_DISPATCH` has `(8,192)`. |
| InstantTensor | git **main** `scitix/InstantTensor` (HEAD `d8b2d707ab58`; package still 0.1.9) | PyPI 0.1.9 is 2026-05-27. Main is **16 commits** ahead, including aarch64 wheels (#16) and GDS/io_uring loader work. |
| fastsafetensors | git **main** `foundation-model-stack/fastsafetensors` (HEAD 2026-08-20 `11f88b88ced9`) | PyPI **0.3.3** is 2026-07-07. Main is ahead (Spark-tested #100). Fallback loader. PP=1. |
| LMCache | git **`dev`** `LMCache/LMCache` (HEAD 2026-08-22 `f9addd2e4e07`) | Default branch is `dev`, not `main`. PyPI **0.5.4** (2026-08-20) is the latest tag; `dev` is newer. GDS: `LMCACHE_LOCAL_CPU=False`. |
| NCCL | **source** `NVIDIA/nccl` with `NVCC_GENCODE=sm_121` | Same as eugr. Then `USE_SYSTEM_NCCL=1` for the torch build. Do not pip `nvidia-nccl-cu13` over it. Keep Spark RoCE env. |
| tilelang | **0.1.13** | Latest. vLLM cuda.txt still 0.1.12. mHC TileLang fallback. |
| humming-kernels | **0.1.13** (`[cu13]`) | Latest (2026-08-21). vLLM cuda.txt still 0.1.12. |
| apache-tvm-ffi | **0.1.12** | tilelang 0.1.13 requires `>=0.1.11,<0.1.13`. tokenspeed-mla 0.2.5 wants 0.1.13 exactly, so install tokenspeed `--no-deps`. |
| tokenspeed-mla | **0.2.5** | Latest. vLLM cuda.txt still 0.1.8. Spec-decode MLA helper. |
| nvidia-cuda-nvdisasm | **13.3.73** | Cutlass-dsl 4.7.0 wheel dep (`>=13.3,<14`). Not the image toolkit. Not `13.4.46rc1`. |

Env for DeepGEMM JIT: `DG_JIT_USE_NVRTC=0` (eugr Dockerfile; nv_dev vs NVRTC).

### 4.1 CUDA 13.3.1 devel (not NGC pytorch, not Hub nightly)

Base: `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04` (Hub and NGC, arm64,
ubuntu24.04). Latest CUDA tag on that distro. Devel + cuDNN: we compile
torch, NCCL, vLLM, FlashInfer, DeepGEMM. Do not start from `-base` or
`-runtime`.

| Pin | Value |
|-----|--------|
| Base / runner | `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04` |
| `CUDA_VERSION` | `13.3.1` |
| `torch_cuda_arch_list` | `12.1a` |
| Python | distro 3.12 + `uv` (`UV_SYSTEM_PYTHON=1`) |
| Pattern | eugr `Dockerfile` (CUDA devel, not their `13.0.2` / cu130 wheels) |

Do not use vLLM `docker/Dockerfile` two-stage (manylinux builder + pip torch).
Do not use `nvcr.io/nvidia/pytorch:26.07-py3` or `nvcr.io/nvidia/vllm:26.07-py3`.
Do not use ubuntu26.04 CUDA tags. There is no `nvidia/cuda:13.4-*` on Hub.

**Host vs container.** Factory Spark is CUDA **13.0** + R580. eugr moved
**to** `nvidia/cuda:13.0.2-devel-ubuntu24.04` for host compatibility.
13.3.1 needs forward-compat on factory hosts.

| Host `nvidia-smi` CUDA | What to do |
|------------------------|------------|
| 13.3+ | Run as-is. |
| 13.0 / R580 (factory) | Keep the **container** on 13.3.1. Install `cuda-compat-13-3` if the image does not already have it. Set `VLLM_ENABLE_CUDA_COMPATIBILITY=1`. Do not flash an unofficial host driver as a prerequisite. |
| Older than 13.0 | Stop. Spark GB10 is not that box. |

CUDA 13.x CMake still drops `12.1` from `CUDA_SUPPORTED_ARCHS` (family `12.0`).
Keep `VLLM_PRESERVE_SM12X_TARGET=1` and `12.1a`.

Fallback if 13.3.1 will not load on factory R580: same compile recipe on
`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` (eugr's host-compat base).

### 4.2 b12x from git + CUTLASS DSL 4.7.0

PyPI `b12x` latest is **1.2.6** (2026-08-20). Git `master` is the same
package version. Both declare five exact pins:

`nvidia-cutlass-dsl{,-libs-base,-libs-core,-libs-cu12,-libs-cu13}==4.6.2`

There is no 1.2.7, no `1.3.0a`, no `>=4.6.2` on PyPI or on master
`pyproject.toml`. `pip install b12x==1.2.6` cannot take 4.7.0.

Do what eugr already ships (`patches/pin_cutlass_dsl.py`,
`CUTLASS_DSL_VERSION=4.7.0`, `B12X_REPO` + `--no-deps`):

```
git clone --depth 1 --branch master \
  https://github.com/local-inference-lab/b12x.git /tmp/b12x-source
# eugr still clones https://github.com/lukealonso/b12x.git ; same repo.
python3 patches/pin_cutlass_dsl.py 4.7.0 --expected-count 5 \
  /tmp/b12x-source/pyproject.toml
uv pip install nvidia-cutlass-dsl[cu13]==4.7.0
uv pip install --reinstall --no-deps /tmp/b12x-source
printf '%s\n' "$(git -C /tmp/b12x-source rev-parse HEAD)" > .b12x-source-commit
```

`--no-deps` is required. A normal install of the rewritten pyproject would
still try to pull b12x's other pins; vLLM already provides torch. Record the
SHA. Re-pin at build start if master moved.

`nvidia-cutlass-dsl-libs-cu13` 4.6.2 and 4.7.0 both require pip
`nvidia-cuda-nvdisasm>=13.3,<14`. That is a **wheel**, not a second toolkit.
Image CUDA is already **13.3.1**. Install `nvidia-cuda-nvdisasm==13.3.73`
(latest non-rc). Do not take `13.4.46rc1`.

Fallback if 4.7.0 JIT fails France: same git tree, rewrite back to 4.6.2
(or skip the rewrite). Do not `pip install b12x==1.2.6` as the main path.

Live overlay image stays PyPI `b12x==1.2.6` + 4.6.2 until this plan's
Phase 3 passes.

`quack-kernels==0.6.4` has the same exact pin (`nvidia-cutlass-dsl==4.6.2`
and `[cu13]==4.6.2`). Run `pin_cutlass_dsl.py 4.7.0 --expected-count 2` on
its metadata (or `--no-deps` after a checkout) so it cannot pull 4.6.2
over the image-wide 4.7.0.

### 4.3 Rest of the stack (git / newest wheel, not vLLM cuda.txt)

vLLM `requirements/cuda.txt` on main is a **cu130 / 4.6.2 / 0.6.17** freeze.
Override it after the vLLM wheel install.

| vLLM cuda.txt today | We install |
|---------------------|------------|
| `flashinfer-python==0.6.17` + `flashinfer-cubin==0.6.17` | FlashInfer **git main** (cubin from that tree / flashinfer.ai matching SHA) |
| `nvidia-cutlass-dsl[cu13]==4.6.2` | **4.7.0** |
| `quack-kernels==0.6.4` (cutlass 4.6 pin) | 0.6.4 + metadata rewrite |
| `instanttensor >= 0.1.9` | InstantTensor **git main** |
| `fastsafetensors >= 0.3.3` | fastsafetensors **git main** |
| `tilelang==0.1.12` | **0.1.13** |
| `humming-kernels[cu13]==0.1.12` | **0.1.13** |
| `apache-tvm-ffi==0.1.11` | **0.1.13.post3** |
| `tokenspeed-mla==0.1.8` | **0.2.5** |
| `NCCL_VERSION=2.30.7` | **source** NCCL `sm_121` (eugr), then `USE_SYSTEM_NCCL=1` |
| `torch==2.13.0` | **source** `release/2.14` via `use_existing_torch.py` |
| `DEEPEP_COMMIT_HASH=d4f41e4e93` | DeepEP **main** `01dc3aa` if compiled |

LMCache is not in cuda.txt. Install from git `dev`.

transformers stays a floor (`>= 5.10.4` in vLLM common.txt). Take the
newest release that satisfies it at build time. Do not freeze an old 4.x.

### 4.4 Compile torch, then vLLM (no wheels, no NGC)

Official aarch64 wheels are cu126 / cu130 / cu132 and list Blackwell as
`12.0+PTX`. We already compile NCCL, FlashInfer, DeepGEMM, vLLM, b12x.
Compile torch against this image's nvcc with `12.1a` as well.

eugr currently **does not** do this: `nvidia/cuda:13.0.2-devel` +
`torch==2.13.0` from `whl/cu130`. They tried NGC pytorch for sm_121
([issue 23](https://github.com/eugr/spark-vllm-docker/issues/23)), then
came back to CUDA 13.0.2 for host compatibility. We take their Dockerfile
shape (devel, uv, ccache, RDMA, source NCCL, `use_existing_torch.py`) and
swap in 13.3.1 + a source torch.

Order inside `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04`:

```
# 0. apt: python3-dev cmake ninja-build ccache git
#    libibverbs1 libibverbs-dev rdma-core
#    pip install uv
#    UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1
#    TORCH_CUDA_ARCH_LIST=12.1a
#    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
#    DG_JIT_USE_NVRTC=0
#    MAX_JOBS=8 for torch (128 GiB UMA; eugr uses 16 for vLLM)

# 1. NCCL (eugr)
git clone https://github.com/NVIDIA/nccl.git
make src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
make pkg.debian.build && apt install the debs
# --allow-change-held-packages

# 2. PyTorch 2.14
git clone --recursive --branch release/2.14 https://github.com/pytorch/pytorch.git
export USE_CUDA=1 USE_CUDNN=1 USE_SYSTEM_NCCL=1 USE_DISTRIBUTED=1
export USE_MKLDNN=0 BUILD_TEST=0
export TORCH_CUDA_ARCH_LIST=12.1a CUDA_HOME=/usr/local/cuda
python3 setup.py install
# expect 2-6 hours. ccache on. do not serve 0731 during this.

# 3. torchvision 0.29.0 against that torch (source). triton==3.8.0 pip.

# 4. vLLM main
python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
uv pip install --no-build-isolation -e .
```

`use_existing_torch.py` strips vLLM's `torch == 2.13.0` so the resolver
cannot pull a wheel over the source install. CMake only **warns** if
`Torch_VERSION` is not exactly `2.13.0`. The 2.13-only stable-ABI header
patch is skipped on 2.14 (`>=2.13 AND <2.14`).

Gate after torch:

```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"
nvcc --version
```

Expect CUDA **13.3.x**. `torch.cuda.get_arch_list()` may only show `sm_120`
even after a `12.1a` build (tensor-core `12.1a` is a small set of files and
often does not appear in that list). Record `torch.__version__` and the
git SHA instead.

PyTorch's published matrix (2.13, and 2.14 wheels) is CUDA **13.0.2** stable
and **13.2.1** experimental. **13.3.1 is not listed.** nvcc 13.3.1 can still
compile 2.14; that pairing is the first thing that can fail.

Disable PyTorch's own flash-attn / mem-efficient attention in the torch
build (`USE_FLASH_ATTENTION=0`, `USE_MEM_EFF_ATTENTION=0`). vLLM uses
FlashInfer. Those flags save hours.

vLLM's builder also needs **rustc** (eugr installs rustup for the frontend)
and a **separate FlashInfer wheel** (`uv build --no-build-isolation`,
`FLASHINFER_CUDA_ARCH_LIST=12.1a`, `UV_PYTHON_DOWNLOADS=never`).

Do **not** copy eugr's runner stage blindly. Their runner `FROM` CUDA again
and `uv pip install torch==2.13.0` from `whl/cu130`, which would throw away
the source torch. Stay on one devel image, or copy `site-packages` from the
builder.

[#52183](https://github.com/vllm-project/vllm/pull/52183) is the wheel bump.
Track it. Do not import its lockfiles.

Fallback if `release/2.14` will not compile against nvcc 13.3.1: same image,
`v2.13.0` source. Fallback if 13.3.1 will not run on factory R580:
`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` (eugr's host-compat base) and
the same source recipe. Optional last resort: pip `2.14.0+cu132` on
`13.2.1-cudnn-devel` (loses the source-torch goal).

Do not overlay this Python/torch onto the live v0.27.1 `.so`.

### 4.5 Doability (what is proven vs extra)

**Proven on this hardware (eugr `Dockerfile` today):**
`nvidia/cuda:13.0.2-devel-ubuntu24.04`, pip `torch==2.13.0` from `whl/cu130`,
source NCCL `sm_121`, FlashInfer git wheel, DeepGEMM `DEEPGEMM_SRC_DIR`,
`use_existing_torch.py`, `12.1a`, cutlass-dsl 4.7.0 rewrite, b12x `--no-deps`.
They moved the default CUDA image **to** 13.0.2 for host compatibility.

**This plan adds two unproven pieces on top of that shape:**

1. Toolkit **13.3.1** on a factory **13.0 / R580** host (`cuda-compat-13-3`).
2. **Source** PyTorch `release/2.14` with `12.1a`. Official aarch64 wheels
   never list 13.3. Compiling torch is 2–6 hours, `MAX_JOBS=8` or spark2
   earlyoom. The kernels that won France (FlashInfer DSV4, DeepGEMM, b12x,
   NCCL) are compiled anyway. Torch source is the weakest ROI in the stack;
   it is still the user pin.

Everything after Phase 1 (overlays, PIECEWISE AR-break, France gate) is the
same work as before. A green compile does not skip Phase 2–3.

Time order of failure, most likely first: R580 vs 13.3.1, then torch 2.14
vs nvcc 13.3.1, then torchvision ABI vs source torch, then triton 3.8 pip
vs source torch, then FlashInfer/vLLM compile, then France.

## 5. Feature list: how to get each

### 5.1 DSpark k=5

**Need:** `speculative_config.method=dspark`, `num_speculative_tokens=5`.
0731 `dspark_block_size=5`, `num_nextn_predict_layers=1`.

**Upstream:** in vLLM since #51538 / #52288 (in rc2, still on main).

**Keep from us:** FlashInfer TOPK 192 (or vendor FlashInfer main). Skip DSpark
CUDA graph capture until PIECEWISE + AR-break is re-proven on the new `.so`.
[#52499](https://github.com/vllm-project/vllm/pull/52499) (spec shapes) is
open; we did not need it after 192. Comment already posted. Do not block.

**eugr:** same k=5. Their recipe sets draft `attention_backend` to
`B12X_MLA_SPARSE`. Live main pin does the same (`DRAFT_ATTENTION_BACKEND`
in `configs/pin.main.env`). Stock vLLM main has no `B12X_MLA_SPARSE` enum;
`patches/files/dsv4_b12x_sparse.py` registers it onto the 584 B DSV4 page.

### 5.2 Full b12x kernels

| Op | Flag | Upstream | Proven |
|----|------|----------|--------|
| Linear FP8 | `--linear-backend b12x` | #52016, in rc2 and main | `B12xFp8BlockScaledMMKernel`, wq_a cosine 0.9999986 |
| MoE MXFP4 | `--moe-backend b12x` | #52018, **in main**, not in rc2 | expert 0 cosine 0.99896 (`w13_layout=w31`) |
| Attention | `--attention-backend B12X_MLA_SPARSE` | overlay module; eugr nightly has the name | Live 2026-08-24: France green, 26.90 / 85.98 tok/s (gated 17.81 / 52.12) |

On stock main, eugr's `VLLM_USE_B12X_WO_PROJECTION` / `_MHC` / `_FP8_GEMM` /
`_MOE` / `_SPARSE_INDEXER` are **not** the API. Use CLI backends.

CUTLASS block-FP8 `is_supported()` still ignores SM12x ([#53055](https://github.com/vllm-project/vllm/pull/53055) open).
Keep `patch_cutlass_sm12x_guard` so auto-select cannot pick a crashing CUTLASS
ahead of b12x.

Do not use Triton linear (`float8_e8m0fnu` KeyError on this stack).

### 5.3 Attention + KV dtypes

**Attention:** `--attention-backend B12X_MLA_SPARSE` (target and DSpark
draft). Decode/prefill kernels are `b12x.attention.compressed_mla`.
Indexer decode is b12x paged (page_size 64, packed-at-store). FlashInfer
SM120 still owns KV insert / DSV4 metadata. `FLASHINFER_MLA_SPARSE_DSV4`
remains a fallback backend on the same 584 B page. Do not pass
`scale_format=2` (GLM NVFP4 432/368).

**Kernel block 64:** SM12x FlashInfer page is 64. vLLM main still
`return [256]` in sparse MLA / indexer. We opened
[#53425](https://github.com/vllm-project/vllm/pull/53425). Keep
`patch_dsv4_sm12x_block_size`. Manager `--block-size 256` (C128 = block/128).
Do not set manager block 64.

**fp8_ds_mla:** stock V4 layout. Parked on the overlay only because isolation
ran on the nvfp4 alias. On main, this is **Phase 3 primary** (fewer dtype
patches).

**nvfp4_ds_mla:** 584 B DSV4 envelope, same billed size as fp8 content. Main
cache dtypes: `fp8_ds_mla`, `nvfp4`, `nvfp4_4over6`. No `nvfp4_ds_mla`.
`validate_nvfp4_kv_cache_with_mla` uses `startswith("nvfp4")` and would
reject a DSV4 name. Keep `patch_nvfp4_ds_mla` + `patch_dsv4_nvfp4_attn`.
This image still has **no NVFP4 CUDA writer**. Real B/token saving was
anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (~7,650 B/token whole-model across 61 layers, $61 \times 125.4\text{ B}$,
`configs/pin.golden.env`). Do not claim memory wins from the alias.

**MQA indexer:** DSA scores are `sum_h w_h * relu((q_h·k)*scale)`, not
weighted-Q. Overlay 17 (`patch_mqa_relu_formula`) + no `.item()` gather.
[#41834](https://github.com/vllm-project/vllm/pull/41834) `sm12x_mqa.py` is
the landing zone (needs-rebase). Keep our fallbacks until DeepGEMM nv_dev
MQA matches `fp8_mqa_logits_torch` on GB10.

**einsum o_proj:** `compute_fp8_einsum_recipe` still `major>=10` → SM100
packed INT32. GB10 is major 12. [#52357](https://github.com/vllm-project/vllm/pull/52357)
open. Keep SM90 FP32 128x128 recipe + dequant fallback. Enabling DeepGEMM
einsum **with the wrong recipe** is worse than PyTorch dequant.

### 5.4 Cold start: InstantTensor / fastsafetensors GDS

Two different jobs people collapse into "GDS":

1. **Load weights once per process** (this section).
2. **Offload KV while serving** (section 5.6).

vLLM main `LoadFormats` includes `fastsafetensors` and `instanttensor`.

| Loader | Pin | Spark notes | Provenance |
|--------|-----|-------------|------------|
| **InstantTensor** (primary) | git main, `--load-format instanttensor` | Direct I/O, pipelined prefetch, GDS when present. aarch64 wheels on main (#16). | vLLM docs; eugr 0731 recipe; [scitix/InstantTensor](https://github.com/scitix/InstantTensor) |
| fastsafetensors (fallback) | git main, `--load-format fastsafetensors` | UMA copier. eugr: skip if weights >0.85 of RAM. DSpark+PP hang: keep **PP=1**. | IBM paper; eugr README; NVIDIA forum Kimi-K3/PP report |

**Hybrid draft load (required with DSpark + InstantTensor):**
DSpark loads a second model from the **same** 0731 checkpoint. A second
InstantTensor pass restreams the full 155 GiB. eugr
`mods/instanttensor-hybrid-draft-loader` keeps InstantTensor for the target
and lazy safetensors for the same-path draft (`INSTANTTENSOR_DRAFT_LOADER=auto`).
Port that mod. Provenance:
[eugr README](https://github.com/eugr/spark-vllm-docker/blob/main/mods/instanttensor-hybrid-draft-loader/README.md).
Remove when vLLM has a per-draft load-format.

### 5.5 Do not reload every iteration

"Iteration" here is restart / rebuild / next request, not token decode.

| Layer | What to persist | Where |
|-------|-----------------|-------|
| Checkpoint | 0731 safetensors on NVMe, not re-downloaded | `~/models/ds4-flash-0731` (already) |
| HF / InstantTensor metadata | tokenizer, index | bind-mount cache |
| FlashInfer JIT | `sparse_mla_sm120.so` after TOPK 192 instantiate | volume `flashinfer_jit_cache` |
| b12x / CuTe DSL | first-use JIT | `~/.cache` / `CUTE_DSL_*` cache dir |
| torch inductor / AOT | eugr sets `VLLM_USE_AOT_COMPILE=1` | persist compile cache between serves |
| Docker image | do not `docker build --no-cache` | layer cache on each Spark |
| Prefix / KV | `--enable-prefix-caching` + LMCache GDS | NVMe `gds_path` |

Do **not** persist CUDA graph capture across process death. Recapture each
serve. That is cheap compared to 155 GiB weight copy.

Drop-caches (`mods/drop-caches` in eugr, commented out on 0731) fights
page-cache double-buffering. InstantTensor O_DIRECT makes it less necessary.
Do not drop caches as a default.

### 5.6 KV offload: GDS / LMCache / native connector

On UMA, **CPU KV offload consumes the same 128 GiB**. Native vLLM
`OffloadingConnector` always has a CPU primary tier; FS secondary **stages
through CPU** (`docs/features/kv_offloading_usage.md` on main).

**Preferred on Spark:** LMCache GDS, CPU RAM offload **off**.

```
local_cpu: false
chunk_size: 256
gds_path: /mnt/nvme/lmcache-0731
cufile_buffer_size: 2048
```

`cufile_buffer_size` is VRAM (UMA). Start small (2 GiB), util 0.8.
`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`.
Docs: https://docs.lmcache.ai/kv_cache/gds.html
vLLM main already vendors `lmcache_connector.py`.

**Phase order:** quality at util 0.8 **without** KV offload, then add GDS
so KV tokens can grow without raising util. Prefix caching stays on either
way (`--enable-prefix-caching`).

Native `OffloadingConnector` + `type: fs` is a backup if LMCache GDS
misbehaves on GB10. Keep `cpu_bytes_to_use` tiny (staging only), `root_dir`
on NVMe. Do not allocate tens of GiB CPU cache on Spark.

### 5.7 SSD streaming of experts (runtime, optional)

Not required to **load** 0731 at TP=2.

| Mechanism | Status 2026-08-23 | Spark note |
|-----------|-------------------|------------|
| Weight prefetch offload v2 `--offload-params` / `--offload-prefetch-step` | Merged [#29941](https://github.com/vllm-project/vllm/pull/29941) (2026-02-26). Tests: `test_prefetch_offload.py`. Aimed at **CPU** onload hide (GB200 C2C). | UMA: copies do not free the pool. Useful only if dest is not UMA. |
| `--moe-expert-cache-size` LFRU GPU cache | [#37190](https://github.com/vllm-project/vllm/pull/37190) **OPEN** | CPU-pinned experts. Same UMA trap. |
| `--moe-gpu-prefetch` pager | [#41447](https://github.com/vllm-project/vllm/issues/41447) OPEN | Overlaps #37190. |
| InstantTensor | Load-time stream, not decode-time expert pager | Already in 5.4. |
| SSD MoE energy | arXiv 2508.06978: SSD expert decode can be ~12× energy vs HBM | Prefetch hides latency, not energy. |

**Sequence:** after France passes and InstantTensor load is fast, measure
whether KV is the limiter at util 0.8. If yes, prototype NVMe expert
paging (not CPU). Do not block the main image on #37190.

## 6. Surgical patchset on main

Compared to `patches/apply_overlays.py` on rc2-overlay.

### Drop (in main already, or overlay-only because of v0.27.1 `.so`)

| Overlay | Why drop |
|---------|----------|
| `copy_new_modules`, `patch_moe_backend`, `patch_envs`, `patch_utils_b12x`, `patch_mxfp4_oracle`, `patch_mxfp4_process_weights` | #52018 is on main |
| `patch_deep_gemm_sm12x_guard` | **Conditional.** Drop only after nv_dev MQA/mHC run without `attention.hpp:122`. If grouped MXFP4 scales fail, keep DeepGEMM off for MoE; do not blanket-kill family 120 if MQA works. |
| `patch_kv_zeroer_skip` | Main rewrote the zeroer past #49704. Confirm 64-vs-256 pages, then drop. |
| Overlaying Python onto a foreign `.so` | The whole point of this plan |

### Keep (still open on main, or local quality)

| Patch | Provenance | Upstream |
|-------|------------|----------|
| `patch_dsv4_sm12x_block_size` | this repo; SM12x page 64 | [#53425](https://github.com/vllm-project/vllm/pull/53425) OPEN (ours) |
| `patch_einsum_sm12x_recipe` + `patch_fp8_einsum_fallback` + scale upcast | this repo; GB10 major=12 | [#52357](https://github.com/vllm-project/vllm/pull/52357) OPEN |
| `patch_mhc` (broadcast only) | this repo; sibling kernels already guarded | [#53055](https://github.com/vllm-project/vllm/pull/53055) OPEN; older [#50645](https://github.com/vllm-project/vllm/pull/50645) |
| `patch_cutlass_sm12x_guard` | this repo | #53055 |
| `patch_indexer_deepgemm_guard` | this repo | #41834 / #53055 orbit |
| `patch_mqa_logits_sm12x_fallback` + ReLU + no `.item()` + b12x MQA | this repo; wrong first formula was weighted-Q | [#41834](https://github.com/vllm-project/vllm/pull/41834) `sm12x_mqa.py` |
| `patch_sm12x_kv_insert` | this repo | local / #41834 |
| `patch_nvfp4_ds_mla` + `patch_dsv4_nvfp4_attn` | this repo; anemll measured real NVFP4 | skip upstream as a fake writer |
| `patch_flashinfer_dsv4_dispatch` (+ cu) | this repo; DSpark ceil(133/64)*64=192 | FlashInfer [#4380](https://github.com/flashinfer-ai/flashinfer/pull/4380) **merged**; **not in 0.6.17**. Prefer vendor FlashInfer main, then drop this overlay. |
| `patch_tp_allreduce_eager_break` | this repo; PIECEWISE captured host-staged PYNCCL | watch, not a clean one-liner |
| `patch_dspark_skip_cudagraph` | this repo; shared `lm_head` | local |
| `patch_lm_head_restore_after_graphs` | diagnostic | skip in production image |

### Add (new on main)

| Patch / config | Provenance | Notes |
|----------------|------------|-------|
| InstantTensor hybrid draft loader | eugr `mods/instanttensor-hybrid-draft-loader` | Required with `--load-format instanttensor` + DSpark |
| CUDA 13 keep `12.1` in `CUDA_SUPPORTED_ARCHS` | eugr `patch_vllm_preserve_sm12x_target.py` | Opt-in `VLLM_PRESERVE_SM12X_TARGET=1` |
| b12x git + cutlass metadata 4.7.0 | eugr `pin_cutlass_dsl.py`; `patches/pin_cutlass_dsl.py` | Not a vLLM overlay. `--no-deps` git install. |
| `DG_JIT_USE_NVRTC=0` | eugr Dockerfile | env, not a source patch |
| LMCache GDS yaml | LMCache docs | `configs/lmcache.gds.yaml` |
| Persist JIT/AOT volumes | ops | `scripts/05-serve.sh` bind mounts |

Do not take eugr's Gemma4 / DiffusionGemma / AutoGPTQ / MiniMax Dockerfile
patches. Not 0731.

Do not take eugr `cooperative_topk` patch unless MoE leaves b12x.

## 7. Graphs and NCCL (keep the hard-won isolation)

Failed (do not retry blindly): FULL graphs, FULL + DSpark capture, PIECEWISE
with DSpark capture no-op, `cudagraph_copy_inputs=true`, AR break via
import-time `eager_break` (circular import), AR break without clone (NCCL
into graph-pool).

Landed: PIECEWISE + lazy `BreakableCUDAGraphCapture` + clone off graph pool
+ strong-ref + DSpark graphs off.

FULL still forbidden. Fallback `ENFORCE_EAGER=1`.

eugr 0731 uses `FULL_AND_PIECEWISE` and `VLLM_USE_BREAKABLE_CUDAGRAPH=0`.
Different kernel stack. Do not copy those flags until France is green.

## 8. Target serve flags (test)

Live pin: [../configs/pin.main.env](../configs/pin.main.env).

```
--tensor-parallel-size 2
--pipeline-parallel-size 1
--gpu-memory-utilization 0.8
--kv-cache-dtype nvfp4_ds_mla
--linear-backend b12x
--moe-backend b12x
--attention-backend B12X_MLA_SPARSE
--block-size 256
--max-model-len 65536
--max-num-seqs 8
--max-num-batched-tokens 8192
--load-format instanttensor
--enable-prefix-caching
--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}
--compilation-config {"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":64}
```

Isolation order if France dies: `fp8_ds_mla` + `FLASHINFER_MLA_SPARSE_DSV4`
+ `ENFORCE_EAGER=1`. Then PIECEWISE. Then nvfp4 + b12x attn. Do not enable
LMCache or FULL graphs on the first boot.

## 9. Work sequence

Each phase has a gate. Do not skip gates.

### Phase 0 (this document)

Plan + pins + provenance. Gate: this file and `pin.main.env` exist.

### Phase 1: matched image on Spark

Build on spark1 (copy the image to spark2). Custom Dockerfile, eugr-shaped
(section 4.1 / 4.4). Do not start from vLLM `docker/Dockerfile`. Do not
start from NGC pytorch.

- `FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04` (one image, not eugr's
  runner that re-pips torch)
- rustc (eugr rustup) before the vLLM build
- FlashInfer as its own `uv build` wheel, `FLASHINFER_CUDA_ARCH_LIST=12.1a`
- `MAX_JOBS=8` for torch, 16 later for vLLM / FlashInfer (eugr default 16)
- ccache, RDMA (`libibverbs`, `rdma-core`)
- NCCL from source, `sm_121` (section 4.4)
- PyTorch `release/2.14` from source, `TORCH_CUDA_ARCH_LIST=12.1a`, `USE_SYSTEM_NCCL=1`
- torchvision 0.29.0 from source; `triton==3.7.1`; `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`
- `VLLM_PRESERVE_SM12X_TARGET=1`
- `DEEPGEMM_SRC_DIR` at `8b1392b` (record `.deepgemm-commit`)
- `DG_JIT_USE_NVRTC=0`
- clone vLLM `main`, `python use_existing_torch.py`, `--no-build-isolation`
- FlashInfer from git **main** (HEAD `fb28d7242b35`; not 0.6.17; verify `_DECODE_DSV4_DISPATCH` has `(8,192)`)
- b12x from git master + `patches/pin_cutlass_dsl.py 4.7.0 --expected-count 5` + `--no-deps` (section 4.2)
- quack-kernels 0.6.4 + `pin_cutlass_dsl.py 4.7.0 --expected-count 2` + `--no-deps`
- InstantTensor, fastsafetensors, LMCache from **git** (section 4.3)
- sequential `uv pip`: cutlass-dsl 4.7.0, tilelang 0.1.13, humming 0.1.13, apache-tvm-ffi **0.1.12**, tokenspeed-mla 0.2.5 `--no-deps`, nvidia-cuda-nvdisasm 13.3.73. `BUILD_NVEP=0` so FlashInfer does not pip `nvidia-nccl-cu13` over source NCCL.
- do **not** pip-install torch / `nvidia-nccl-cu13` over the source builds
- If host driver reports CUDA 13.0: `cuda-compat-13-3` + `VLLM_ENABLE_CUDA_COMPATIBILITY=1`

Gate: `nvcc --version` is 13.3.x; `torch.version.cuda` starts with `13.3`;
`torch.__version__` starts with `2.14`; pytorch git SHA recorded;
cutlass 4.7.0; FlashInfer 192 in the frozenset;
b12x / InstantTensor / fastsafetensors / lmcache import;
SHA files for PyTorch, NCCL, vLLM, DeepGEMM, b12x, FlashInfer, InstantTensor, fastsafetensors, LMCache.
Tag `vllm-spark-0731:main-b12x`. `ENTRYPOINT` / `CMD` for `vllm serve`.

### Phase 2: surgical overlays only

Port keep/add patches onto the new tree. Rewrite `apply_overlays.py` so
rc2-only cherry-picks are gone. `assert_image.py` for the new layout.

Gate: asserts pass. No `B12xExperts` copy from `patches/files/` if main
already has `fused_moe/b12x.py`.

### Phase 3: quality (eager, then PIECEWISE)

**Done 2026-08-23** on `vllm-spark-0731:main-b12x`, util 0.8, PIECEWISE,
InstantTensor, DSpark k=5, `nvfp4_ds_mla` + `B12X_MLA_SPARSE` (not the
fp8-first order below). `VALIDATE_STACK=main scripts/06-validate.sh` green:
`' Paris'` logprob -0.265, n_tie=1, chat `Paris.`. 1-way 17.81 tok/s,
8-way 52.12 tok/s. KV 93,401 tokens.

**Decode follow-on 2026-08-24 01:40 UTC** (same image, `--only`
`o-proj-b12x`, `indexer-store-page64`, `indexer-b12x-schedule`,
`ar-piecewise-ws`, `dspark-backbone-none`): France still green. 1-way
median 26.90 tok/s (gather pin ~30.6). 8-way 85.98 tok/s. KV 94,516.
WO is bmm, not `wo_proj.run()`. DSpark backbone FULL, sample eager.
Paged indexer is packed-at-store page 64; vLLM schedule is passed only
for `q_rows==1`, so DSpark 1-way (padded to 8) stays on the unscheduled
1023-page path. Do not gather packed storage. Do not expand already-1024
page tables.

Original order (still the isolation path if France dies):

1. `ENFORCE_EAGER=1`, `fp8_ds_mla`, util 0.8, InstantTensor, DSpark, no
   LMCache. `06-validate.sh`.
2. If DeepGEMM MQA crashes, re-enable indexer/MQA fallbacks (keep). If
   grouped MXFP4 experts are wrong, keep MoE on b12x (already the plan)
   and pin DeepGEMM A/B.
3. PIECEWISE + AR clone + DSpark graphs off. France again.
4. Only then `nvfp4_ds_mla`.

Gate: same greedy string / `' Paris'` logprob / chat `Paris.`.
FULL_AND_PIECEWISE is still out. Util stays 0.8.

### Phase 4: persist caches

Bind-mount JIT, AOT, HF, InstantTensor. Restart serve twice. Gate: second
start does not re-JIT FlashInfer DSV4 192 and does not re-download 0731.

### Phase 5: hybrid draft loader

Port eugr InstantTensor hybrid. Gate: logs show lazy safetensors for draft;
target still InstantTensor; France still green; load time down vs double
stream.

### Phase 6: KV GDS offload

LMCache GDS to NVMe, `local_cpu: false`, small cufile buffer. Gate: process
stays under earlyoom; KV tokens ≥ eager baseline; France still green.
If GDS fails, native `OffloadingConnector` FS with tiny CPU staging.

### Phase 7: optional expert SSD

Only if KV is the limiter at 0.8. Do not merge #37190 blindly (CPU/UMA).

### Phase 8: raise util

0.80 → 0.81 only with swap or earlyoom `-m` change. Never 0.85 on spark2
as it is today.

## 10. Do not

- Overlay main Python on `vllm/vllm-openai:v0.27.1`.
- Pull Hub `nightly` (cu129) or stale `cu130-nightly` as the production base.
- Use `nvcr.io/nvidia/pytorch:26.07-py3` or `nvcr.io/nvidia/vllm:26.07-py3`.
- Use `pytorch/pytorch:*` Hub images (amd64 CUDA tags).
- Use ubuntu26.04 CUDA tags, or a manylinux `cuda13.4` builder as the serve image.
- Pip-install cu130 / cu132 / nightly torch wheels over the source build.
- Pip-install `nvidia-nccl-cu13` over the source NCCL.
- Compile torch with `MAX_JOBS` high enough to earlyoom spark2 (start at 8).
- Default vLLM `CUDA_VERSION=13.0.3` via their Dockerfile.
- Leave vLLM `cuda.txt` FlashInfer 0.6.17, cutlass 4.6.2, tilelang 0.1.12, humming 0.1.12, tvm-ffi 0.1.11, tokenspeed-mla 0.1.8.
- `pip install b12x==1.2.6` or `instanttensor==0.1.9` / `fastsafetensors==0.3.3` / `lmcache==0.5.4` as the main-image path (git is newer).
- Leave b12x's or quack's `==4.6.2` cutlass lines in place.
- Copy eugr `FULL_AND_PIECEWISE` + util 0.85 together with `B12X_MLA_SPARSE`.
  Live main uses `B12X_MLA_SPARSE` + PIECEWISE + util 0.8.
- Pass GLM `scale_format=2` / 432/368 NVFP4 into DSV4 584 B pages. The legal
  mix is `B12X_MLA_SPARSE` + `nvfp4_ds_mla` as a 584 B fp8 envelope.
- Enable DeepGEMM MoE for 0731 MXFP4 experts without the `a6b593d` vs
  `8b1392b` A/B.
- Upstream a blanket `is_deep_gemm_supported()=False` on family 120.
- Open duplicate PRs for #52357, #53055, #41834, #52499, FlashInfer #4380.
- Use `PP>1` with fastsafetensors + DSpark.
- Treat `nvfp4_ds_mla` as a memory optimization without a CUDA writer.
- `pip install --break-system-packages`; use uv in the image.

## 11. Memory budget at util 0.8 (test)

Per node 128 GiB UMA. util 0.8 → 102.4 GiB vLLM.

| Piece | Rough |
|-------|--------|
| Weights / rank | ~77.7 GiB |
| Graphs + activations + InstantTensor staging | several GiB |
| LMCache cufile buffer | start 2 GiB |
| KV remainder | tens of GiB, not the 561k tokens of util 0.81 PIECEWISE |

Expect **fewer** KV tokens than the overlay 0.81 run. That is accepted for
the test pin.

## 12. Provenance index

| Idea | Source |
|------|--------|
| Matched main + 12.1a + nv_dev DeepGEMM | this plan; eugr Dockerfile; vLLM cmake |
| CUDA 13.3.1 cudnn-devel ubuntu24.04 | NVIDIA Hub / NGC; latest 13.x on 24.04; no 13.4 tag |
| Host 13.0 + `cuda-compat-13-3` | Spark porting guide; vLLM `VLLM_ENABLE_CUDA_COMPATIBILITY`; eugr chose 13.0.2 for host-compat |
| Source torch `release/2.14` + `12.1a` | wheels are `12.0+PTX`; eugr issue 23; forum sm_121a tensor cores |
| Source NCCL sm_121 | eugr Dockerfile `NVIDIA/nccl` + `NVCC_GENCODE` |
| b12x git + cutlass 4.7.0 metadata rewrite | eugr Dockerfile `pin_cutlass_dsl.py`; this repo `patches/pin_cutlass_dsl.py`; PyPI b12x 1.2.6 still `==4.6.2` |
| quack-kernels 0.6.4 + same rewrite | vLLM cuda.txt; quack still pins cutlass 4.6.2 |
| FlashInfer git main (not 0.6.17 / not only rc7) | flashinfer-ai main `fb28d724`; tag v0.6.18rc7 behind main; #4380 |
| InstantTensor git main | scitix/InstantTensor 16 commits past PyPI 0.1.9; aarch64 #16 |
| fastsafetensors git main | foundation-model-stack HEAD `11f88b88` (Spark #100); PyPI 0.3.3 older |
| LMCache git `dev` | LMCache default branch; PyPI 0.5.4 is the last tag |
| NCCL / torch | source NCCL sm_121 + PyTorch `release/2.14`; wheels / NGC are fallback only |
| tilelang 0.1.13 / humming 0.1.13 / tvm-ffi 0.1.13.post3 / tokenspeed-mla 0.2.5 | PyPI latest vs vLLM cuda.txt |
| DeepGEMM freeze `a6b593d` | eugr Dockerfile comment; commit `3fba416` |
| `DG_JIT_USE_NVRTC=0` | eugr Dockerfile |
| InstantTensor 0731 + hybrid draft | eugr `recipes/deepseek-v4-flash-0731.yaml` + mod README |
| fastsafetensors UMA / GDS | fastsafetensors 0.3.3; eugr README; vLLM docs |
| LMCache GDS CPU=off | https://docs.lmcache.ai/kv_cache/gds.html |
| Native FS KV tier | vLLM `docs/features/kv_offloading_usage.md` |
| Prefetch offload v2 | vLLM #29941 |
| Expert GPU cache | vLLM #37190 OPEN |
| PIECEWISE + AR clone | this repo HANDOFF, live 2026-08-23 |
| MQA ReLU formula | this repo vs `fp8_mqa_logits_torch` |
| TOPK 192 | FlashInfer #4380; DSpark k=5 arithmetic |
| Kernel block 64 | this repo #53425 |
| 584 B envelope vs real NVFP4 | this repo LINEAGE; anemll golden |
| earlyoom / util 0.85 | this cluster; eugr issues #349 |
| Do not Hub-nightly | [UPSTREAM.md](UPSTREAM.md) nightly section |

## 13. Open A/B (do not guess)

1. DeepGEMM `8b1392b` vs `a6b593d` on 0731 MXFP4 grouped scales.
2. DeepGEMM SM12x MQA vs our ReLU PyTorch fallback (cosine vs
   `fp8_mqa_logits_torch`).
3. InstantTensor vs fastsafetensors load time and peak UMA on 2-node 0731.
4. LMCache GDS vs native `OffloadingConnector` FS on GB10 NVMe.
5. Whether family `12.0f` Hub binaries even run DeepGEMM on 12.1 (we compile
   `12.1a` and skip the experiment unless time appears).
6. b12x JIT on cutlass 4.7.0 vs rewrite-back to 4.6.2 if France fails.
7. CUDA 13.3.1 + `cuda-compat-13-3` on factory R580 vs same recipe on 13.0.2-devel.
8. PyTorch `release/2.14` source vs `v2.13.0` source if 2.14 will not compile on nvcc 13.3.1.
