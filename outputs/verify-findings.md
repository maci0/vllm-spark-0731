# Verification Findings — vllm-spark-0731 docs audit

Date: 2026-08-25. Scope: accuracy, truthfulness, and completeness of every doc
claim (README, HANDOFF, `docs/`, knowledge base, `patches/*.md`) against the
actual repo state (configs, scripts, Dockerfiles, patches, tests).

**Verifier:** `scripts/verify-docs.py` — exits 0 when all six checks pass:

1. markdown links resolve (except the 2 verbatim-archive exceptions below)
2. backticked repo-path references resolve (except documented exceptions)
3. config pin matrix matches the pin files and `docker/Dockerfile.main`
4. key numbers and arithmetic identities hold and appear in the corpus
5. every documented `--only` overlay name exists in `patches/apply_overlays.py`
6. `pytest tests/` passes (22 passed / 12 skipped, GPU-dependent skipped)

Run: `python3 scripts/verify-docs.py`

## Allowed exceptions (checked by design, documented)

| # | Where | Ref | Why |
|---|-------|-----|-----|
| 1 | `docs/field-notes/dgx-spark/PROD_C5_SSD.md` | `examples/prod-c5-ssd.yaml` | Broken in the source repo too; field-notes kept verbatim (archive README flags the doc's disproven claims) |
| 2 | `docs/field-notes/dgx-spark/TROUBLESHOOTING.md` | `examples/prod-c5-ssd.yaml` | Same verbatim-archive policy |
| 3 | `docs/field-notes/dgx-spark/UPSTREAM_GAPS.md` | `` `examples/prod-c5-ssd.yaml` `` | Same verbatim-archive policy (backticked prose) |
| 4 | `docs/PLAN-MAIN.md` | `` `docker/Dockerfile` `` | vLLM upstream repo's file, not ours (docs say "vLLM's …") |
| 5 | `docs/UPSTREAM.md` | `` `docker/versions.json` `` | vLLM upstream repo's file, not ours (docs say "vLLM's …") |

## Ledger

### FIXED — clear-cut doc-vs-reality mismatches (this audit + prior pass)

| # | Doc | Issue | Fix |
|---|-----|-------|-----|
| 1 | `configs/env.spark.sh`, `docs/knowledge/09`, `glossary.md` | `DG_JIT_USE_NVRTC` claimed as "the working path" (default 1); contradicts `pin.main.env` (0), `PLAN-MAIN` (0), and 09's own validation (NVRTC cubins rejected: `CUDA_ERROR_INVALID_IMAGE`) | Default → 0; comment states the rejection; 09 "fix" step 2 → "tried, did not land"; glossary updated |
| 2 | `README.md` | Kernel-warning remedy said "until `deepgemm-fp8-1d1d-port.diff` is upstreamed" — port is analysis-only; real fix is the pin-back | Remedy → pin-back (vLLM #53680 / DeepGEMM #417) + JIT toolchain gap, links 09 |
| 3 | `patches/README.md` | "Do not use" listed patches in the deleted `maci0/vllm-spark-nvfp4` repo; both now live here | Reworded to local paths with per-patch rationale |
| 4 | `patches/README.md` | `patches/v0.28/` (7 diffs) undocumented and unreferenced (orphaned since the first commit) | Documented in "Reference files" as superseded by `apply_overlays.py --stack rc2` |
| 5 | `docs/knowledge/06` serve table | `golden` row claimed `FLASHINFER_MLA_SPARSE_DSV4` + "Stock/flashinfer_b12x"; pin leaves both unset (v0.25.2 predates the backend name); `fp8` row claimed linear `b12x`, pin leaves it unset | Cells corrected to match pins |
| 6 | `docs/knowledge/08` gap table | mHC row listed `patch_mhc`/`patch_cutlass_sm12x_guard` as if `--only` flags (they are `--stack main` functions) | Clarified "(applied in `--stack main`, not standalone `--only`)" |
| 7 | `docs/knowledge/{00-index,01,06}` | DeepGEMM pin still `8b1392b` after the 2026-08-25 pin-back to `a6b593d` | Pins updated; 01's critical warning rewritten for the pin-back + JIT gap |
| 8 | `docs/UPSTREAM.md`, `docs/knowledge/08`, `patches/upstream/README.md` | `` `Dockerfile.main` `` refs missing the `docker/` prefix (file is `docker/Dockerfile.main`) | 6 refs → `docker/Dockerfile.main` |
| 9 | `docs/PLAN-MAIN.md`, `docs/UPSTREAM.md`, `docs/LINEAGE.md` | Refers to vLLM-upstream files (`docker/Dockerfile`, `docker/versions.json`) and the TRT-LLM-tree note with local-looking paths | Marked external ("vLLM's …", "(external repo)") |
| 10 | `docs/field-notes/dgx-spark/EUGR_B12X_PROD.md` | Backticked prose ref `examples/stagec-nvfp4-prod.yaml` still pointed at the pre-merge location | → `../../../configs/examples/…` |
| 11 | `docs/knowledge/{README,00-index,…}` | Prior structure pass: dual entry points, missing provenance, no glossary | Single landing page, bidirectional field-note links, glossary, prev/next nav (see git diff) |

### DOCUMENTED — judgment calls, no behavior changed (by design)

| # | Item | Status | Note |
|---|------|--------|------|
| 1 | `configs/pin.env` (fp8 legacy stack) has no `LINEAR_BACKEND`; its sibling `pin.nvfp4.env` sets `b12x` | DOCUMENTED — table in `06-deployment` now states "linear unset in pin" | Possibly an oversight in the pin; changing it alters runtime behavior → out of scope; revisit if the fp8 stack is ever served again |
| 2 | `HANDOFF.md` says `VLLM_USE_B12X_MHC` "is unused" while `pin.main.env` defaults it to 1 and `05-serve.sh` passes it through | VERIFIED — no repo code reads the var (grep: only pin + serve); it is container-side pass-through | The claim is accurate at the repo level; image-side behavior is out of scope |
| 3 | `patches/v0.28/` files exist but nothing applies them | DOCUMENTED — kept for history, now described in `patches/README.md` | Optionally deletable later; kept to preserve the original rc2 recipe mechanism |

### NOT-VERIFIABLE — cluster-only claims (recorded, not blocking)

| # | Claim | Where | Why not verifiable here |
|---|-------|-------|------------------------|
| 1 | Kernel behavior on SM12x (b12x paged MQA, DeepGEMM guards, FlashInfer dispatch) | `docs/knowledge/03`, HANDOFF | Requires the 2× DGX Spark cluster + built images |
| 2 | Measured throughput / KV pool numbers (25.8, 95, 172, 97,737, 65.2, 216.8, 2,047,170, …) | README, HANDOFF, `docs/knowledge/05` | Measured on the cluster; cross-doc consistency verified, provenance is HANDOFF/field-notes |
| 3 | Golden image internals (anemll v0.25.2 attention backend name, DeepGEMM 2.5.0 fork details) | `docs/knowledge/09`, `06` | Requires image inspection / GPU; docs already note the pin leaves the backend unset |
| 4 | Upstream PR states (#53055, #53425, … open/merged) | `docs/UPSTREAM.md` | Needs live GitHub checks at run time; the tracker records its own "last verified" date |

## Completion

Verifier green (`0 finding(s) total`, exit 0) on 2026-08-25. No OPEN ledger
items; every discrepancy found was either fixed, documented as a deliberate
judgment call, or recorded as NOT-VERIFIABLE (cluster-only).
