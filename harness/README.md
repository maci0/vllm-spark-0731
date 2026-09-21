# harness/

The measurement tooling that produced [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md). These lived at
`~/` on the rig; they are copied here so the protocol travels with the record.

## The protocol

| File | Role |
|---|---|
| `drive-median.sh` | the entry point. `drive-median.sh <tag> <passes> <max_tokens> [levels...]` runs N passes of `drive-meter.sh` and reports the per-level median. Writes the guard lines, so this is the file that makes an arm evidence |
| `drive-meter.sh` | one pass: health poll, gates, warm pass, then each level measured through `bench-concurrency.py` |
| `median_levels.py` | reads the per-pass `*-<N>.meter.log` files and prints the median table |

`drive-meter.sh` calls `scripts/bench-concurrency.py` inside the repo, which is what actually drives
the workload (chat mode, seed 1234, 512-token generations).

## Launch and arm runners

| File | Role |
|---|---|
| `run-arm.sh` | candidate arm, end to end: stop the old container, start the worker on spark2, start the head on spark1 about 85 s later, poll health, save the engine config, then meter. Usage: `run-arm.sh <tag> "<env assignments>"`. **The assignments only reach the container if they go through `SERVE_EXTRA_ENV`**, which is the trap that silently defeated the profiler three times |
| `launch-refbase.sh` | the reference arm, through `sparkrun` with `ref-base0731.yaml` |
| `ref-base0731.yaml` | the reference recipe: anemll image, base checkpoint, k=7, capture 48, breakable cudagraph off. Staged here, not in a path containing "sparkrun", because `spark-launch.sh` runs `pkill -9 -f '[s]parkrun'` and would kill the caller |
| `run-refg.sh`, `run-protog.sh` | the two specific runs behind the guarded baseline |
| `run-prof6c.sh` | the c6 profile, and the one that passes the profiler settings through `SERVE_EXTRA_ENV` so the step window actually applies |
| `run-graphdump.sh` | the CUDA launch-shim run; superseded, see `../instruments/` |
| `probe-moe.sh` | probes a list of `MOE_BACKEND` values and stops at the first that serves |

## Instruments and gates

| File | Role |
|---|---|
| `check_b12x_api.py` | asserts the exact b12x surface this repo imports. Against 1.2.6 it passes; against the reference's 0.15.3 it reports six failures. Keep it as a gate for any b12x pin move |
| `check_b12x_side.py` | asserts the side-by-side install: 1.2.6 still works as `b12x`, 0.15.3 reachable as `b12x_ref` |
| `pin.patch.sed` | the one-line sed that made `DRAFT_ATTENTION_BACKEND` overridable in `configs/pin.main-029.env` |
| `test-ccompile.sh` | the torch.compile experiment driver, parked; see `docs/EXPERIMENTS.md` |

## `logs/`

Engine and launcher output kept because the arms that failed left no artifact in `outputs/`:
`deeplinear` (DeepGEMM ue8m0 assert), `b12x015` (image missing on spark2), `stockops` (KV too small),
plus the `*-engine.log` captures of the guarded arms' backend selections, the `ccompile-run*.log`
series, `graphdump.log`, and `reference-anemll-launch.log`, which is the reference arm's own config
dump.

## `sections/`

The `upstream-section*.md` fragments written during the 2026-09-13 session. They are already appended
verbatim to [`docs/UPSTREAM.md`](../docs/UPSTREAM.md) in order; they are kept here as the source
fragments rather than deleted, and are not a second version of the record.
