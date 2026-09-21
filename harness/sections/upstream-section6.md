
### Region attribution, 2026-09-13: the MoE is the dominant region and the collectives are not

`VLLM_PROFILE_DECODE=1 VLLM_PROFILE_CAPTURE=1 VLLM_PROFILE_DECODE_STEPS=12`, reading the region
marks the overlay already installs. Per recorded layer:

| region | n | gpu_sum | gpu_avg | gpu_max |
|---|---|---|---|---|
| ffn (MoE) | 3 | 5.5 ms | 1.84 ms | 1.93 ms |
| attn | 3 | 2.8 ms | 0.94 ms | 1.09 ms |
| wo | 3 | 1.8 ms | 0.60 ms | 0.69 ms |
| mla | 3 | 0.6 ms | 0.20 ms | 0.26 ms |
| allreduce | 7 | 1.4 ms | 0.20 ms | 0.67 ms |

and `b12x layers: n=3 sum=8.5ms avg=2.83ms top: L0=2.99 L1=2.79 L2=2.70`, i.e. per-layer cost is
flat across layers, not concentrated in the first dense ones.

Two conclusions follow, and they are the first region-level facts this investigation has had:

- 2.83 ms per layer over 43 layers is about 122 ms, which is the measured c1 target forward (98 to
  118 ms). So a c1 decode step is 43 sequential layers at roughly 2.8 ms each. The step is a latency
  chain, and the reference's chain is about 35 % shorter.
- the MoE accounts for 1.84 ms of that 2.83 ms, about 65 %. Attention is 0.94 ms, and the TP
  all-reduce is 0.20 ms, which is why lowering the collective cost is not worth pursuing: 43 of them
  are under 9 ms and the measured region total is 1.4 ms for the ones recorded.

Instrument caveats, recorded rather than smoothed: only three layer events and three to seven region
events are recorded per step, so these are per-recorded-unit averages rather than whole-forward sums;
and the window summary's `fwd_gpu/step=137.9ms` exceeds `wall/step=75.3ms` because async scheduling
overlaps the events, so the region averages are the trustworthy part and that total is not.

What this changes: the MoE is 65 % of the layer that dominates the step, and our MoE does not come
from our own overlays, it comes from b12x, whose version is 1.2.6 against the reference's 0.15.3.
Better still, the mxfp4 oracle exposes a whole menu of MoE implementations behind the already
env-overridable `MOE_BACKEND`, none of which has been measured:

    b12x (current), deep_gemm, flashinfer_trtllm, flashinfer_trtllm_afp8, flashinfer_cutlass,
    flashinfer_cutlass_afp8, triton, triton_unfused, humming, marlin, aiter variants

So the dominant region has eleven selectable implementations and we have measured exactly one. That
is the next A/B series, one backend per arm against `protog` 38.4 / 80.8 / 105.2 / 122.3.
