#!/usr/bin/env bash
# Recipe: same-day re-baseline of standing arm on v0.30.0rc2.
#
# Later same-image arms (nowo-rc2 442.2, moehum-rc2 445.5, k6-rc2 445.6,
# moea16-rc2 453.7) sit well above proto2-030-rc2 417.5. That is inside
# the 14.0 % keep-rule, but it is also consistent with a low first sample.
# Re-run the pin with no extra env under a new tag. Keep-rule still uses
# the original standing until this median lands.
#
#   configs/examples/proto2-030-rc2-rebaseline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh p030-rc2b ""
