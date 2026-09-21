#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — MSA scheduled page-max decode off.
#
# B12X_MSA_DECODE_PAGEMAX default 1. Overlay scores DSA indexer logits
# via logits_paged; when uses_paged_mqa_schedule is true the scorer
# takes the scheduled page-max path. Force 0 so decode stays on the
# unscheduled logits_paged kernel. Same image, only this env.
#
#   configs/examples/msa-pagemax-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nopagemax-rc2 \
  "SERVE_EXTRA_ENV=B12X_MSA_DECODE_PAGEMAX=0"
