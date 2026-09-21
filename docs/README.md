# Docs index

Start at [`../HANDOVER.md`](../HANDOVER.md) if you are picking this work up, then come back here for
the depth.

| File | What it is | Status |
|---|---|---|
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Every measured arm: what it changed, its per-level medians and spreads, its gates, the verdict, and the arms that failed. Generated from `outputs/driver/` by `scripts/experiment-ledger.py` | **current, generated** |
| [`LINEAGE.md`](LINEAGE.md) | What the base is, which upstream PRs are in it and which are patched, and the DeepGEMM pins that are not interchangeable | current |
| [`PLAN-MAIN.md`](PLAN-MAIN.md) | The matched-main build plan: toolchain versions, what is built from source, why | current |
| [`knowledge/`](knowledge/00-index.md) | The distilled chapters, 01 to 14: hardware, model, kernels, quantization, performance, deployment, gotchas, upstream, golden DeepGEMM, operations, cost, debugging, QwenSeek | current, start at `00-index.md` |
| [`field-notes/`](field-notes/README.md) | Raw, unedited archives from the predecessor projects, plus the DeepSeek-V4 serving notes added 2026-09-13 | history, verbatim |
| [`UPSTREAM.md`](UPSTREAM.md) | The chronological work log: every measurement, every dead end, in the order it happened. 2873 lines | **history, not current state**; the current summary is `EXPERIMENTS.md` |
| [`../HANDOFF.md`](../HANDOFF.md) | The accumulated long-form handoff from earlier sessions, 8634 lines | **history** |

## Which one answers which question

- "What is the project and where does it stand?" -> [`../HANDOVER.md`](../HANDOVER.md).
- "What has been tried, and did it work?" -> [`EXPERIMENTS.md`](EXPERIMENTS.md).
- "How do I run a measurement, and what will bite me?" -> [`../HANDOVER.md`](../HANDOVER.md) and
  [`../harness/README.md`](../harness/README.md).
- "How do I rebuild the image, and with what?" -> [`PLAN-MAIN.md`](PLAN-MAIN.md), then
  `scripts/02-build-main-029.sh`.
- "Which port and pin is the base?" -> [`LINEAGE.md`](LINEAGE.md).
- "Why is this kernel or dtype the way it is?" -> the matching `knowledge/` chapter.
- "Where did the predecessor work come from?" -> [`field-notes/README.md`](field-notes/README.md).

## Two files that are easy to misread

`UPSTREAM.md` and `HANDOFF.md` are append-only history. Both contain claims that later work corrected,
including several presented as settled that are not, and `UPSTREAM.md` opens with a note saying so.
Read them for provenance and for the reasoning behind a decision, never for the current state.

## Backups that used to look like sources

Until 2026-09-13, eleven files under `configs/`, `patches/` and `docker/`, plus
`docs/UPSTREAM.md.pre-port`, carried `.pre-port`, `.pre-proto`, `.pre-reanchor`, `.bak` or a
timestamp suffix: `patches/apply_overlays.py.pre-proto-port`,
`patches/apply_overlays.py.20260910-211635.bak`, `patches/assert_image.py.pre-proto`,
`docker/assert_main_image.py.20260910-211635.bak`, `docker/pin_quack.py.pre-proto`,
`configs/pin.main.env.b12x-sparse.bak`, `configs/pin.main-029.env.bak-tvmffi` and four more of the
same shape.

They were snapshots taken before a port or a re-anchor, and they have been **deleted**. They were
redundant: the state each one preserved is superseded, the git history that is tracked already holds
it, and nothing in the build or the harness read them. `docs/UPSTREAM.md` still refers to two of them
by name in its log entries, which is expected of an append-only history and is flagged in its header.

