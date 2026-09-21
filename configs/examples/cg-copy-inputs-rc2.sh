#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — cudagraph_copy_inputs=true.
#
# Standing compilation_config has cudagraph_copy_inputs: False.
# Copying graph inputs on replay avoids aliasing captured pointers
# across 87 piecewise all-reduce graphs. Same image, only this field.
#
#   configs/examples/cg-copy-inputs-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh cgcopy-rc2 \
  "CUDAGRAPH_COPY_INPUTS=true"
