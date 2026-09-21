
### `stockops`, 2026-09-13: our overlays are load-bearing, and the config space is exhausted

`VLLM_USE_B12X_WO_PROJECTION=0 VLLM_USE_B12X_SPARSE_INDEXER=0` with `MAX_MODEL_LEN=32768` gives
33.1 / 59.9 / 74.7 / 81.2 against `protog` 38.4 / 80.8 / 105.2 / 122.3. Dropping our two per-layer
substitutions costs about 34 % at c6, so they are a large win over the stock paths and not a cost.

The first attempt at this arm died for a different reason worth keeping: with the stock paths the
engine needs 9.48 GiB of KV at 65536 context against 9.32 GiB available, so the overlays also save
memory, and the retry lowered `MAX_MODEL_LEN` to 32768 to fit.

B12X 0.15.3, the reference's, does not export any name our overlays import: no `dsa_indexer`, no
`logits_paged`, no `logits_contiguous`, no `prepare_paged_metadata`, no `uses_paged_schedule`, no
`plan_paged_schedule`. Its `b12x/moe/` holds `fused` and `tuning` where ours holds `_lib` and `comm`.
So the reference's kernel generation is unreachable from our overlays without a port.

Put together, every knob this repo owns is already at its best measured value:

- attention backend is a wash (`attnfi` 37.8 / 78.5 / 104.6 / 119.8)
- the fp8 linear path cannot leave b12x (`deeplinear` crashes, `dglinear` goes numerically dead)
- FULL cudagraph mode loses and is unstable (`cgfull` 38.9 / 76.0 / 87.7 / 109.2)
- the spec token count is level (`proto5` 38.9 / 82.4 / 95.4 / 133.0)
- both remaining overlay knobs are wins that have to stay on (`stockops2` 33.1 / 59.9 / 74.7 / 81.2)

The remaining 26 % on the sum, and the 40 ms of target-forward time at c1, sit in libraries the two
images do not share: b12x (1.2.6 against 0.15.3), flashinfer (0.7.0 against 0.6.15), tilelang (0.1.14
against 0.1.9), and the torch and NCCL builds (NCCL 2.31.2 against 2.30.7). Closing it is a port, or
an extraction of specific kernels from the older b12x, not a configuration change.
