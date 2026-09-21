
### The c6 split, measured properly this time, 2026-09-13

After fixing the plumbing (`VLLM_PROFILE_DECODE_STEPS` has to go through `SERVE_EXTRA_ENV` to reach
the container as `docker run -e`; assigning it in the launching shell does not reach the worker, which
is why every earlier profiling run silently used the default 12 steps and profiled c1), the window is
1250 steps and the rows immediately before the summary are c6 steps:

| call | host wall | gpu |
|---|---|---|
| `execute_model` (target forward) | 2.7 ms | 99.1 to 104.5 ms |
| `sample` (sampler) | 0.6 ms | 2.5 ms |
| `sample_tokens` (draft + sampler) | 117 to 123 ms | 21.2 to 21.9 ms |

So at c6 the target is about 82 % of the per-forward GPU time and the draft about 18 %, which is the
same shape as c1 (roughly 100 ms against 25 ms). The draft is not the c6 bottleneck, and the claim
that the c1 split does not describe c6 was itself wrong: the shape is the same at both levels, only
the absolute scale differs. Two hypotheses are retired by this: that the draft carries c6, and that
c6 is explained by something the c1 profile cannot see.

The host and device numbers invert, which is what made this confusing to read. `execute_model`
returns to the host in 2.7 ms while its GPU work is about 100 ms, and `sample_tokens` blocks the host
for about 119 ms while its own GPU work is 21 ms. So the wall column is not per-step latency and must
not be compared against the meter's step interval.

Where this leaves the search: the target dominates at both c1 and c6, and the target's two largest
regions have both been swapped and both measured a wash, the MoE across vendors and the attention
across implementations and with its autotune active. The remaining difference is inside those kernels,
that is, in the library generation, not in which implementation is selected.
