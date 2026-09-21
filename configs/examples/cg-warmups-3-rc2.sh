#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — cudagraph_num_of_warmups=3.
#
# Standing compilation_config has cudagraph_num_of_warmups: 0.
# Three warmups before capture can stabilize the 87 piecewise
# all-reduce graphs. Same image, only this field.
#
#   configs/examples/cg-warmups-3-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh cgwarm3-rc2 \
  "CUDAGRAPH_NUM_OF_WARMUPS=3"
