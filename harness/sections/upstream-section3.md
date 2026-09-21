
### `proto5` (k=5) against `protog` (k=7), 2026-09-13

| arm | k | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|---|
| `refg` | 7 | 63.1 | 112.1 | 140.9 | 156.0 | 472.1 |
| `protog` | 7 | 38.4 | 80.8 | 105.2 | 122.3 | 346.7 |
| `proto5` | 5 | 38.9 | 82.4 | 95.4 | 133.0 | 349.7 |

Gates pass in all three (`gate_france` starts " Paris", `gate_9x8` is "72, 9x9"). k=5 buys 8.7 % at
c6 and a hair at c1 and c3, and gives back 9.3 % at c5, where k=7's capture fits 5 x 8 = 40 inside
the captured size list and k=5's 5 x 6 = 30 does not. Sum is level within the spreads. So the
speculative token count is not a lever worth pulling; both are about 26 % behind the reference on
the sum and 15 % behind at c6.

### What the decode profiler showed

`VLLM_PROFILE_DECODE=1 VLLM_PROFILE_CAPTURE=1` (the profiler already in
`patches/files/sm12x_b12x_kernels.py`) prints, for the first steps of a run, per-step region and
per-layer figures. Read from the `prof2` container:

- the DSpark draft's per-step cost in steady state is 3.2 to 5.3 ms (`b12x layers: n=3 sum=11.0ms
  avg=3.67ms`), against a target step of 38.65 ms at c6. With k=7 the draft runs seven of those in
  series, so the draft is a large share of the step and is not covered by the attention arm, which
  changed the draft's backend too and measured a wash.
- one early step showed `b12x step sample: wall=78.1ms gpu=77.3ms` and `sample_tokens: gpu=181.1ms`
  as cold-start figures; the printed window is the first 12 non-dummy steps, so these are not steady
  state and are recorded here only as an observation, not as a finding.
- per-layer figures are one to three layers per print and read 3.2 to 5.5 ms, with the maximum
  always at `L0`.

The profiler cannot be pointed at the reference, which is a different build, so it attributes our
step but cannot attribute the difference. That remains the open problem.
