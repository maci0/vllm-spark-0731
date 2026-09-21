
### Arm ledger, 2026-09-13 (all guarded, k=7, capture 48, `--async-scheduling`, 3 passes, 512 tokens, levels 1 3 5 6)

| arm | change against `protog` | c1 | c3 | c5 | c6 | verdict |
|---|---|---|---|---|---|---|
| `refg` | reference image, anemll 0.1.1 | 63.1 | 112.1 | 140.9 | 156.0 | comparator |
| `protog` | none, base | 38.4 | 80.8 | 105.2 | 122.3 | - |
| `attnfi` | `ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` = `FLASHINFER_MLA_SPARSE_DSV4` | 37.8 | 78.5 | 104.6 | 119.8 | wash, rejected |
| `deeplinear` | `LINEAR_BACKEND=auto VLLM_USE_DEEP_GEMM_E8M0=1` | - | - | - | - | crash, see below |
| `dglinear` | `LINEAR_BACKEND=auto` | 10.0 | 27.8 | - | - | broken, see below |
| `cgfull` | `CUDAGRAPH_MODE=FULL` | 38.9 | 76.0 | 87.7 | 109.2 | rejected, see below |

What the three negatives establish:

- `attnfi` at 37.8 / 78.5 / 104.6 / 119.8 is inside the spread of `protog`, so the attention
  implementation is not the gap. Our `B12X_MLA_SPARSE` overlay is not what costs us the step time,
  and the reference's 24 tuned FlashInfer configs are not what earns it its step time.
- `deeplinear` dies at startup: `DeepGEMM` assert `sf.size(-2) == ceil_div(mn, gran_mn)` at
  `csrc/utils/layout.hpp:97`, which is the ue8m0 scale layout. The pin's own comment warned that
  this path was not working, and this is the concrete failure.
- `dglinear` starts but is numerically broken: `accept_rate 0.0 %`, `tokens_per_step 1.002`, 10 tok/s
  at c1, because every draft token is rejected. So the fp8 linear layers cannot move from `b12x` to
  DeepGEMM on this pin without either a DeepGEMM bump or a layout fix, and the measured cost of
  trying is a dead arm.
- `cgfull` is worse at every level and unstable: pass-to-pass spread of 74 / 18 / 18 / 11 %, and
  generations truncate (1381 tokens at c3 where 1536 were asked for). So removing piecewise
  cudagraphing in favour of FULL is not a route.

Why step time is the whole game, stated arithmetically: at c6 `protog` does 4.585 tokens per step at
25.87 steps/s, and `refg` does 4.676 tokens per step at 33.99 steps/s. Holding our own tokens per
step and adopting the reference's step rate would give 4.585 x 33.99 = 155.8 tok/s against their
156.0. Acceptance and per-position acceptance are already level, so nothing needs to be won back on
the speculative side: only 9.25 ms of step time at c6, and about 40 ms at c1.
