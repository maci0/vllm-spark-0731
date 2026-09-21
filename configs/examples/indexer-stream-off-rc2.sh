#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — indexer stream scorer off.
#
# Same as configs/examples/indexer-stream-off.sh, new tag so the log
# does not collide with rc1 `nostream`. B12X_INDEXER_STREAM_SCORER
# default True. Overlay scores DSA indexer logits via logits_paged.
# Same image, only this env.
#
#   configs/examples/indexer-stream-off-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh nostream-rc2 \
  "SERVE_EXTRA_ENV=B12X_INDEXER_STREAM_SCORER=0"
