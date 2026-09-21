
### MoE backend sweep, 2026-09-13

The mxfp4 oracle's menu probed one at a time at the standard config (k=7, capture 48, async, guarded):

| `MOE_BACKEND` | outcome |
|---|---|
| `b12x` (current) | 38.4 / 80.8 / 105.2 / 122.3 |
| `flashinfer_trtllm` | dies at worker init: `Mxfp4 MoE backend 'FLASHINFER_TRTLLM_MXFP4_MXFP8' does not support the deployment configuration since kernel does not support current device cuda` |
| `flashinfer_cutlass` | never healthy, reason not captured |
| `humming` | healthy in 190 s, metered below |
| `deep_gemm`, `marlin`, `triton` | not reached; the probe stops at the first healthy backend |

`MOE_BACKEND=humming` meters 42.0 / 79.3 / 109.1 / 122.8 (spreads 3.8 / 1.5 / 7.5 / 4.6 %) against
`protog` 38.4 / 80.8 / 105.2 / 122.3: +9.4 % at c1, -1.9 % at c3, +3.7 % at c5, +0.4 % at c6, +1.9 %
on the sum. Gates pass.

That result is more informative than the numbers look. The MoE is 65 % of the per-layer *latency* at
c1, and swapping its implementation for a different vendor's moves c1 by 9 % but c6 by 0.4 %. So at
c6 the step is not MoE-latency-bound, and the c1 region split does not describe the c6 case. Sweeping
the rest of the menu (deep_gemm, marlin, triton) is therefore unlikely to find the 28 % that c6
needs, and two of the four probed backends do not run on this device at all.

`humming` is kept as the better c1 setting (its c1 spread of 3.8 % is also tighter than `b12x`'s
9.4 %), but it is a 2 % change on the sum, not the change this goal needs.
