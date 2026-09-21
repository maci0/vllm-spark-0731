#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — VLLM_KV_CACHE_LAYOUT=BLNHC.
#
# Standing layout is auto BLHNC. kvlbnhc-rc2 died: LBNHC is illegal;
# valid layouts are ['BLHNC', 'BLNHC']. This is the other legal layout.
# Same image, only this env.
#
#   configs/examples/kv-layout-blnhc-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh kvblnhc-rc2 \
  "SERVE_EXTRA_ENV=VLLM_KV_CACHE_LAYOUT=BLNHC"
