
### Option 1 first step: the reference's b12x generation is not reachable, 2026-09-13

Built `docker/Dockerfile.kernels-ref-b12x` as a derivative image, `FROM
vllm-spark-0731:main-029-proto` with the reference's `b12x-0.15.3` copied over ours. b12x is the
right library to start with because it ships as a pure-Python wheel, so it cannot break on torch or
CUDA ABI the way flashinfer would.

First serve attempt failed for a mundane reason worth recording: the derivative image existed only on
spark1, so spark2's worker could not start (`Unable to find image ... pull access denied`) and the
head then blocked forever waiting for rank 1. `scripts/02-copy-main.sh` (`docker save | ssh spark2
docker load`) is the repo's distribution path and the image is 28.9 GB.

The API question does not need the copy, and the answer is no:

| what our code imports from b12x | against 1.2.6 | against 0.15.3 |
|---|---|---|
| `attention.compressed_sparse_mla` (Caps, plan, bind, run) | ok | `ModuleNotFoundError` |
| `attention.dsa_indexer.plan_paged_schedule` | ok | `ModuleNotFoundError` |
| `gemm.wo_projection` (Caps, plan, bind, run) | ok | module present, all four names missing |

Six failures, no module usable. Checked with `check_b12x_api.py`, which is worth keeping as a gate
for any future pin move.

The functionality is not gone, it moved and was renamed. 0.15.3 exposes
`attention.mla.compressed_mla_decode_forward`, `attention.mla.sparse_mla_decode_forward`,
`MLASparseDecodeMetadata`, `clear_mla_caches`, and a `b12x.attention.indexer` package of
`IndexerPaged*` bindings. So a port is a re-expression of our three overlay call sites against a
class and metadata API, not an impossibility. It is still one more surface than the overlays: our
vLLM's own b12x integration would have to move too, namely the mxfp4 MoE oracle and the
`b12x_sparse` attention backend, both written against 1.2.6.

### Instrument note: the region table is the target's

The two docstrings in `apply_overlays.py` disagree, so this needed settling, because it decides
whether the region table is evidence at all. The gate is `_b12x_region_active`, which returns True
only while `_PROFILING_STEP[0]` is set, and that flag is set by the `execute_model` wrapper alone. The
draft runs inside `sample_tokens`, where the flag is clear. Capture mode arms the marks separately and
only for `tokens == 1`. So the layer and region events accumulate during the target forward, and the
region table describes the target. The earlier reading stands: per-layer 2.83 ms, ffn/MoE 1.84 ms at
65 %, attn 0.94, wo 0.60, mla 0.20, allreduce 0.20.

Caveat to keep: `b12x layers` printed `n=3`, not 43, so only three layer events are recorded per
print. The per-layer average remains valid, and 43 x 2.83 ms does land on the measured c1 forward
(98 to 118 ms), but the sample is smaller than it should be and the why is not established.
