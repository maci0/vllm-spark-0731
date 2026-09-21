# Versions

Authoritative provenance for the current serving pin. The pin file
(`configs/pin.main-029.env`) is the version source of truth while `v0.30.0`
is the pinned tag; this document is a single-summary copy of exactly what the
`vllm-spark-0731:main-030-0` image contains and what the protocol serves.

## Pinned commit / image

| | |
|---|---|
| vLLM tag | `v0.30.0` (lightweight tag, 2026-09-21T03:14:00Z) |
| vLLM ref | `9ed533eb4adfe48aef7e569a08daeccd2a773fed` |
| diff vs rc2 | 1 commit / 2 files: `[Build] Fix DeepGEMM CUDA 12.9 release builds (#57554)` |
| image | `vllm-spark-0731:main-030-0` sha `f2f9e43a7088` (19-layer overlay on `main-030-0-phase1`) |
| in-image `vllm.__version__` | `0.30.1.dev0+g9ed533eb4.d20260921` (post-tag dev stamp; ref is `9ed533eb4adf`) |

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

- `scripts/port_scan.py --with-upstream-patches` on a clean `v0.30.0` tree:
  **FAIL=0, applied=43, no-op=12, total=55** (identical to the rc2 scan).
- Upstream patch files applied by the overlay Dockerfile (`pr-*.diff`):
  `pr-47988.diff`, `pr-53055.diff`, `pr-53425.diff`, `pr-53522.diff`.
- Our tracked upstream PRs: `#53425`, `#53522`, `#53271`, `#46716` — all open,
  mergeable-blocked behind `pre-run-check`, none merged into `v0.30.0`.

## Serving protocol (what `drive-median.sh <tag> 3 512 1 3 5 6` measures)

Validated default (rounds 103-112, each confirmed by interleaved same-session
A/B):

| | |
|---|---|
| MoE backend | `MOE_BACKEND=humming` (validated +6.4 % vs b12x) |
| speculative depth | `NUM_SPECULATIVE_TOKENS=6` (validated +6.0 % sum / +9.0 % c6 vs k=7) |
| WO projection | `VLLM_USE_B12X_WO_PROJECTION=0` (0 % vs overlay, kept for simplicity) |
| capture size | `MAX_CUDAGRAPH_CAPTURE_SIZE=48` (set by `harness/run-arm.sh`) |
| util | `GPU_MEMORY_UTILIZATION=0.8389` (set by `harness/run-arm.sh`) |
| KV | `nvfp4_ds_mla`, block 256, `max_model_len 65536`, prefix cache on |
| workload | chat, `max_tokens 512`, seed 1234, levels 1 3 5 6, 3 passes, median |

## Measured standing (decisive interleaved, round 110)

| arm | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|
| ours (`abOTH-a/d`, mean) | 63.9 | 111.8 | 142.4 | 155.8 | 474.0 |
| reference (`abREF-b/c`, mean) | 62.5 | 113.4 | 142.3 | 157.3 | 475.5 |

Parity: `-0.3 %` sum, `-1.0 %` c6. Rig session drift `±9 %` exceeds every
achievable one-variable effect; the win bar (more than the larger median
spread) is not reachable. See `docs/GOAL-BEAT-ANEMLL.md` and `docs/UPSTREAM.md`.
