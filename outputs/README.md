# outputs/

Measurement artifacts. Nothing here is a source file; everything is either produced by the harness or
generated from it.

## Layout

| Path | What it is |
|---|---|
| `driver/` | one arm per file group. Flat files are the **15 valid arms** (guarded, see below) |
| `driver/superseded/` | 28 arms measured **before the guard existed**. Kept as history, not evidence |
| `driver/one-off/` | ad-hoc instrument output: torch profiler traces (`*.tprof.log`), nsys runs, memory histograms, accept-rate probes, `snap_a/b.json`, `util-c6.csv`, and older single-shot logs from earlier naming generations |
| `experiments.json` | machine-readable copy of the record, written by `scripts/experiment-ledger.py` |
| `verify-findings.md`, `vllm-spark-0731-docs-audit.md` | older audits, dated 2026-08-25; read with that date in mind |
| `.drafts/`, `.plans/` | working directories from earlier sessions |

## Naming inside `driver/`

For an arm tagged `<tag>`:

| File | Content |
|---|---|
| `<tag>.median.log` | the arm's summary: the guard lines, every pass echoed, and the per-level median table at the end |
| `<tag>-<N>.meter.log` | pass `N`, full output |
| `<tag>-<N>.meter.txt` | pass `N`, the per-level numbers plus the gate answers and the acceptance table |

`~/drive-median.sh <tag> <passes> <max_tokens> [levels...]` writes these, and
`~/drive-meter.sh` writes the per-pass files.

## The guard, and why it exists

The first lines of every `.median.log` are:

```
== <tag>: <passes> passes at max_tokens=<n>, levels <...>, start <timestamp>
== port holder: <who was listening on 8000>
== container: <which container answered>
== engine: <engine version string, if the log carried one>
```

They are there because they were missing once. A still-running reference container answered an entire
series, so five candidate arms were in fact the reference. Every arm without these lines is not
evidence; they live in `superseded/` and are excluded from comparisons.

Read `== container:` first. `vllm-ds4-0731` is a candidate arm; anything named `sparkrun_...` is the
reference.

## Reading a result

The per-level table reports the median across passes and the spread. The spread matters as much as the
median: one 128-token pass swings up to 18 % on this rig, so a change is kept only if it beats the
larger of the two arms' spreads. The gates (`gate_france`, `gate_9x8`) come first for a reason: they
have caught two arms that were numerically dead while still reporting plausible throughput.

## Index and provenance

[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) is the curated record: every arm, what it changed, its
numbers, and whether it was kept, generated from these files by
[`scripts/experiment-ledger.py`](../scripts/experiment-ledger.py). Regenerate rather than hand-copy:

```sh
uv run scripts/experiment-ledger.py
```
