
### Correction to the previous section: that "c6 split" was c1 data, 2026-09-13

The section above labels the `execute_model` gpu 97 to 108 ms against `sample_tokens` gpu 22.7 to
37.3 ms split as the c6 level. It is not. Those samples carry `tok=8`, which is one sequence times
(k+1), so they are c1 steps, and they agree with the c1 numbers measured earlier rather than adding
anything at c6.

The reason the window never moved: `VLLM_PROFILE_CAPTURE=1` routes the step wrapper to
`_b12x_capture_step`, which prints its own bounded window and ignores
`VLLM_PROFILE_DECODE_STEPS`. That is also why the window summary reports `steps=12` in both the 900
and 1250 runs. So the step limit silently does nothing whenever capture mode is on.

The inference I drew from it, that c6 is throughput-bound with about three engine steps in flight,
does not hold either: a single engine step whose target forward takes 100 ms of GPU time cannot be
completing every 38.65 ms, so that arithmetic was comparing a c1 forward against a c6 interval.

What stands: the c1 split (target ~85 %, draft 22 to 40 ms, sampler 2.5 to 3.9 ms), the c1 region
table, the per-layer 2.83 ms, and the fact that humming moves c1 by 9 % while moving c6 by 0.4 %. The
c6 split is not measured. The re-measurement runs with `VLLM_PROFILE_CAPTURE` off so the step limit
applies, which gives the step-level target against draft split at c6; it gives up the region detail,
because region marks inside a replayed graph are only recorded when capture mode armed them.
