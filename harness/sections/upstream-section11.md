
### The fp8 linear axis is closed, and it exposes a real bug, 2026-09-13

`LINEAR_BACKEND=flashinfer_cutedsl` starts and then falls back, from its own log:

    WARNING --linear-backend=flashinfer_cutedsl has no kernel for this linear layer type; using automatic selection for that layer type.
    INFO    Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod
    WARNING --linear-backend=flashinfer_cutedsl requested FlashInfer mm_bf16 backend 'cute-dsl', but it is unavailable on the current hardware or environment; using automatic selection for unquantized linear layers.

So on this image the automatic selection lands on DeepGEMM, and DeepGEMM's fp8 block-scaled MM is wrong here. Both gates fail on both
`LINEAR_BACKEND=flashinfer_cutedsl` and `LINEAR_BACKEND=auto`:

    == gate_france '<?>carecarecarecarecare'
    == gate_9x8    '<?>carecarecarecarecare'

with `accept_rate 0.0 %` and `tokens_per_step 1.002`, so this breaks the target model's own output and
not only the speculative decode.

Two conclusions:

- `LINEAR_BACKEND=b12x` is not a preference, it is the only working option on this pin, and
  `Fp8LinearMethod`'s DeepGEMM kernel, which is what the reference uses, is broken in our image. That
  is a correctness bug in the fp8 linear path, worth its own tracker entry, and it is the concrete
  form of the ue8m0 lead that the earlier `deeplinear` crash pointed at.
- It is not obviously a performance opportunity either. The broken arm's c1 did 511 steps in 53.2 s,
  which is 104 ms per step, against our healthy 114 ms per step at batch 8. So even a working DeepGEMM
  linear is unlikely to be the missing 25 %.

The harness did its job: the gate check fails closed and would have caught this without any numbers
being read.

### Closing the axes

Every axis this repo can move has now been measured, each with the guarded protocol:

| axis | measure | verdict |
|---|---|---|
| attention implementation | `ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4`, target and draft | wash |
| attention tuning | the arm that passes the sparse-MLA autotune gate | wash |
| MoE implementation | `MOE_BACKEND` probed over the oracle menu, humming healthy | +9.4 % c1, +0.4 % c6 |
| fp8 linear | `auto`, `flashinfer_cutedsl` | numerically broken |
| cudagraph mode | `CUDAGRAPH_MODE=FULL` | worse and unstable |
| spec token count | k=5 against k=7 | level on the sum |
| our own overlays | WO projection and sparse indexer off | large loss, they must stay |
| collective cost | allreduce region 0.20 ms avg | negligible, not a lever |

And the two levels agree on the shape: the target forward is 85 % of a c1 step and 82 % of a c6 step,
so speculative decode is not where the gap is, at either level. What remains is per-op performance
inside third-party kernel libraries, where our generation differs from the reference's (b12x 1.2.6
against 0.15.3, flashinfer 0.7.0 against 0.6.15, tilelang 0.1.14 against 0.1.9), and the two regions
that could be swapped individually have both been measured and are washes. Reaching the older
generation means porting our overlays onto an API that does not export the names they import, and
that port would target exactly the two regions just measured as washes.
