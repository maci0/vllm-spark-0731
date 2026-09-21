#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — FlashInfer sparse MLA DSV4.
#
# Same as configs/examples/attn-fi-dsv4.sh, new tag so the log does not
# collide with rc1 `attnfi030`. This image pins FlashInfer v0.7.0rc3
# (rc1 used floating main). Reference auto-picks this backend on SM12x.
#
#   configs/examples/attn-fi-dsv4-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh attnfi-rc2 \
  "ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4 DRAFT_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4"
