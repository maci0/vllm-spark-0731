# Field Notes (raw archives from predecessor repos)

Raw, verbatim write-ups from the two GitHub repositories that this repo
(`vllm-spark-0731`) absorbed, plus the older `vllm-spark-main-b12x` lineage
(which was already fully contained here as an earlier snapshot — nothing was
left behind).

`oh-my-dgx-spark/` is different: it is a **third-party corpus vendored
verbatim** (a Korean-language DGX Spark lab book + research logs). It is kept
unlinked — its raw files are not referenced from the knowledge chapters; the
distilled knowledge itself lives in chapters 01, 05, 06, 07, 10 and 11.

These are the **original source documents**, kept unedited for provenance. The
consolidated, cross-linked version of this knowledge lives in
[`docs/knowledge/`](../knowledge/00-index.md); the two are linked below.

| Archive | Source repo (deleted after merge) | Content |
|---|---|---|
| [`nvfp4/`](nvfp4/README.md) | `maci0/vllm-spark-nvfp4` | NVFP4 MLA KV cache lineage: patch write-ups, DeepGEMM call-site gaps, KV offload root cause, MHC/DeepGEMM SM121 failure, eugr NVFP4 experiment |
| [`dgx-spark/`](dgx-spark/README.md) | `maci0/dgx-spark-deepseek-v4-flash-0731` | Field notes: full quant × framework sweep, golden recipe, KV ceiling, tuning, troubleshooting, session/test logs, prod configs, deadlock bug report |

## Where each document maps in the knowledge base

| Field note | Knowledge base chapter |
|---|---|
| `nvfp4/README.md`, `nvfp4/KV_OFFLOAD_MLA.md` | [04-quantization-kv](../knowledge/04-quantization-kv.md) |
| `nvfp4/DEEPGEMM_CALL_SITES.md`, `nvfp4/MHC_DEEPGEMM_SM121.md` | [03-kernels-attention](../knowledge/03-kernels-attention.md), [08-upstream](../knowledge/08-upstream.md) |
| `nvfp4/EUGR_NVFP4.md` | [06-deployment](../knowledge/06-deployment.md), [04-quantization-kv](../knowledge/04-quantization-kv.md) |
| `dgx-spark/GOLDEN.md`, `dgx-spark/PRODUCTION.md`, `dgx-spark/EUGR_B12X_PROD.md` | [06-deployment](../knowledge/06-deployment.md), [09-golden-deepgemm](../knowledge/09-golden-deepgemm.md) |
| `dgx-spark/KV_CEILING.md`, `dgx-spark/TUNING.md` | [05-performance](../knowledge/05-performance.md) |
| `dgx-spark/TROUBLESHOOTING.md`, `dgx-spark/BUG_REPORT_b12x_2node_deadlock.md` | [07-gotchas](../knowledge/07-gotchas.md) |
| `dgx-spark/UPSTREAM_GAPS.md` | [08-upstream](../knowledge/08-upstream.md) |
| `dgx-spark/TEST_LOG.md`, `dgx-spark/SESSION_2026-08-20.md`, `dgx-spark/ARENA_HANDOFF.md` | raw experiment logs (not re-summarized) |
| `dgx-spark/CLIENT_INTEGRATION.md`, `dgx-spark/MODEL_VARIANTS.md` | [02-model](../knowledge/02-model.md), [06-deployment](../knowledge/06-deployment.md) |
| `dgx-spark/PROD_C5_SSD.md` | [04-quantization-kv](../knowledge/04-quantization-kv.md) (SSD offload; two claims disproven — see `nvfp4/KV_OFFLOAD_MLA.md`) |

## Related artifacts that moved into the live tree

- `vllm-spark-nvfp4` patches → [`patches/upstream/`](../../patches/upstream/) (`*-vllm-only.diff`, `kv-offload-bounds-check.patch`) and [`patches/v0.27.1/`](../../patches/v0.27.1/) (`combined-v0.27.1.patch`, `eugr-nvfp4.patch`)
- `vllm-spark-nvfp4` Dockerfiles → [`docker/Dockerfile.nvfp4`](../../docker/Dockerfile.nvfp4), [`docker/Dockerfile.eugr-nvfp4`](../../docker/Dockerfile.eugr-nvfp4)
- `dgx-spark` prod recipes → [`configs/examples/`](../../configs/examples/)
- `dgx-spark` ops scripts → [`scripts/`](../../scripts/) (`clean-restart.sh`, `spark-launch.sh`, `sweep.sh`, `sitecustomize-nvfp4-mla-guard.py`); the original aiohttp harness stays archived at `dgx-spark/scripts/bench.py` (superseded by `scripts/bench-concurrency.py`)
- `dgx-spark` patchset design doc → `dgx-spark/patchset/` (anemll NVFP4 reference extract)

## Why `vllm-spark-main-b12x` has no archive here

`../vllm-spark-main-b12x` (also `maci0/vllm-spark-main-b12x` on GitHub) was a
superset-subset sibling of this repo: every file it contained already existed
here, and every differing file was an older revision (checked file-by-file on
2026-08-25). Its unique history is two commits
(`7d7e66f raise serve concurrency to 8 seqs`, `748c59b matched vLLM main image
for 0731 on DGX Spark`), both reflected in this repo's later state.
