#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc1 — FlashInfer sparse MLA DSV4.
#
# Reference auto-picks FLASHINFER_MLA_SPARSE_DSV4 on SM12x. Our pin forces
# B12X_MLA_SPARSE. proto2-attnfi / proto2-attnfi-warm were negative on the
# old pin (cold then warm autotune). Never re-measured on main-030-rc1.
# FlashInfer 0.7.0 SM120 DSV4 specialization is present. Same image, only
# target + draft attention backend (one family).
#
#   configs/examples/attn-fi-dsv4.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh attnfi030 \
  "ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4 DRAFT_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4"
