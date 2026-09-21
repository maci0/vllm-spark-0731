#!/usr/bin/env bash
# Recipe: same-day re-baseline of standing arm on v0.30.0rc1.
#
# Later same-image arms (moecap175, moework, util8663, moetile32, moeshare)
# cluster at 421-429 while proto2-030 is 412.1. That is inside the 12.8 %
# keep-rule, but it is also consistent with a low first sample. Re-run the
# pin with no extra env under a new tag so the log does not append onto
# proto2-030. Keep rule still uses the original standing until this median
# lands.
#
#   configs/examples/proto2-030-rebaseline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh p030b ""
