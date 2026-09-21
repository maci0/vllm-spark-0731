
### Step attribution at last, and the whole kernel stack differs, 2026-09-13

Raised the decode profiler's window (`VLLM_PROFILE_DECODE_STEPS=900`) so it prints in steady state
rather than on the first twelve cold steps. Reading the last samples, which are single-sequence
(`tok=8`, i.e. 1 x (7+1)) c1 steps:

| region | gpu time |
|---|---|
| `execute_model` (target forward) | 98 to 118 ms |
| `sample_tokens` (draft + sampler) | 23 to 40 ms |
| `sample` (sampler alone) | 2.5 to 3.9 ms |

So at c1 the target forward is about 85 % of the step; the sampler is noise. The reference's whole
c1 step is 74 ms, so if its split has a similar shape its target is around 63 ms against our 100 ms,
which accounts for the entire 40 ms gap. The draft is not the story and neither is sampling. That
retires the previous turn's guess that the DSpark draft was the leading suspect.

Also from that log, per-layer detail on our side:

- our WO projection runs as a batched `bmm` with a batch of 4 on 42 rows
  (`b12x wo_proj bmm ok tgd=(42, 4, 4096) z=(4, 42, 1024) out=(42, 4096)`), four small GEMMs per
  layer instead of one.
- the B12X MLA scratch is 810 MB, sized for `rows=12288`, the prefill bound
  (`B12X_MLA_SPARSE compressed MLA scratch 810028032 bytes (heads=32 rows=12288 width=704 page=64
  chunks=2)`); decode allocation only, not steady-state cost, but it is why the KV pool is tight.

The two images share almost no kernel library, which is why every A/B above came back a wash: none
of them changes the code that actually dominates.

| library | ours | reference |
|---|---|---|
| b12x | 1.2.6 (source at `3a437ab5`, the 1.3.0 release state) | 0.15.3 |
| flashinfer | 0.7.0 | 0.6.15 (+ cubin 0.6.13, jit_cache) |
| tilelang | 0.1.14 | 0.1.9 |
| humming_kernels | 0.1.13 | 0.1.10 |
| quack_kernels | 0.6.5 | 0.5.0 |
| tokenspeed_mla | 0.2.8 | 0.1.2 |
| NCCL | 2.31.2 | 2.30.7 |

b12x is the one that matters most: it supplies the MoE experts, the compressed MLA and the indexer,
and 0.15.3 and 1.2.6 are different series with different module layouts (`cute`, `distributed`,
`quant` against `_lib`, `comm`, `norm`), so the kernels behind `Using B12xExperts` are not the same
kernels. The pin's own comment already recorded the version difference; what is new here is that the
target forward is where the gap lives, so this is now a measured reason to care rather than a note.

`dspark.py` differs too but not in algorithm: ours is the later upstream file (confidence head and
sequence-parallel paths the reference lacks), 570 lines against 490.
