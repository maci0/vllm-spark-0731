# Versions

Authoritative provenance for the current serving pin. The pin file
(`configs/pin.main-029.env`) is the version source of truth while `v0.31.0`
is the pinned tag; this document is a single-summary copy of exactly what the
`vllm-spark-0731:main-031-0` image contains and what the protocol serves.

Status of this pin: the image is **built and self-verified in-build** (the
overlay stage runs `apply_overlays` then `assert_image`, which imports the
DSV4 model tree). The France/Paris gate and the protocol re-baseline have
**not** run against it, because the rig was occupied by other jobs at build
time. Treat every serving number in this document as belonging to the
previous `v0.30.0` pin until a re-baseline is recorded.

## Pinned commit / image

| | |
|---|---|
| vLLM tag | `v0.31.0` (`db9527a46873454610df6dbedf79a36d6bf1a7f6`) |
| image | `vllm-spark-0731:main-031-0` sha `f0fbbc4bc931`, 29.4 GB (overlay on `main-031-0-phase1` sha `cad0c9f17698`) |
| in-image `vllm.__version__` | not read (no serve run on this pin yet) |

## Runtime versions inside the image

| | |
|---|---|
| torch | `2.14.0a0+git2b3ec34` (git `2b3ec34829036a65cd9d1398ea72a0167dc37470`, branch `release/2.14`) |
| triton | `3.7.1` |
| CUDA toolkit | `13.3.1` (base `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04`) |
| flashinfer | `0.7.0rc3` |
| b12x | `1.2.6` from source `B12X_REF=3a437ab516…` (code identical to the 1.3.0 wheel; see pin comment) |
| cutlass-dsl | `4.7.0` (pinned override; do not move to 4.6.2/4.7.1) |
| quack-kernels | `0.6.5` |
| humming-kernels | `0.1.15` |
| tilelang | `0.1.14` |
| tokenspeed-mla | `0.2.8` |
| DeepEP / fastsafetensors / InstantTensor / LMCache | per pin: `v1.2.1`, `0.4.0`, `v0.2.0`, `v0.5.5` |

## Overlay set

- `scripts/port_scan.py --with-upstream-patches` on a clean `v0.31.0` tree:
  **FAIL=0, applied=43, no-op=12, total=55** (same shape as the `v0.30.0` scan).
- Upstream patch files applied by the overlay Dockerfile (`pr-*.diff`):
  `pr-47988.diff`, `pr-53425.diff`, `pr-53522.diff`. `pr-53055.diff` returns
  rc=1 on v0.31.0 because its `mhc/tilelang.py` hunk is already upstream;
  its `cutlass.py` hunk still applies.
- `pr-53425.diff` was regenerated against `v0.31.0`. The previous copy applied
  only its import hunk there, which dropped `MultipleOf` while leaving the
  method that annotates with it, so the overlay build failed with `NameError`.
- Our tracked upstream PRs: `#53425`, `#53522`, `#53271`, `#46716` — all open,
  mergeable-blocked behind `pre-run-check`, none merged into `v0.31.0`.

## Serving protocol (what `drive-median.sh <tag> 3 512 1 3 5 6` measures)

Validated default (rounds 103-112, each confirmed by interleaved same-session
A/B):

| | |
|---|---|
| MoE backend | `MOE_BACKEND=humming` (validated +6.4 % vs b12x) |
| speculative depth | `NUM_SPECULATIVE_TOKENS=6` (validated +6.0 % sum / +9.0 % c6 vs k=7) |
| WO projection | `VLLM_USE_B12X_WO_PROJECTION=0` (0 % vs overlay, kept for simplicity) |
| capture size | `MAX_CUDAGRAPH_CAPTURE_SIZE=48` (set by `harness/run-arm.sh`) |
| util | `GPU_MEMORY_UTILIZATION=0.8389` (set by `harness/run-arm.sh`, overridden per arm when a run says so) |
| KV | `nvfp4_ds_mla`, block 256, `max_model_len` from the pin (`262144`), prefix cache on |
| workload | chat, `max_tokens 512`, seed 1234, levels 1 3 5 6, 3 passes, median |

## Measured standing (decisive interleaved, round 110, `v0.30.0` pin)

| arm | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|
| ours (`abOTH-a/d`, mean) | 63.9 | 111.8 | 142.4 | 155.8 | 474.0 |
| reference (`abREF-b/c`, mean) | 62.5 | 113.4 | 142.3 | 157.3 | 475.5 |

Parity: `-0.3 %` sum, `-1.0 %` c6. Rig session drift `±9 %` exceeds every
achievable one-variable effect; the win bar (more than the larger median
spread) is not reachable. See `docs/GOAL-BEAT-ANEMLL.md` and `docs/UPSTREAM.md`.

## Port recipe

`configs/examples/port-v0.31.0.sh` carries the steps that produced this pin:
`check`, `scan`, `phase1`, `overlay`, `copy`, `arm`. The `arm` step is the one
not yet run.
