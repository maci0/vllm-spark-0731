#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — use_inductor_graph_partition=true.
#
# Standing compilation_config has use_inductor_graph_partition: None
# (EngineCore later False). Partitioning inductor graphs around the 87
# piecewise all-reduce breaks is a different lever from capture sizes /
# copy_inputs / warmups. Same image, only this field.
#
#   configs/examples/inductor-graph-partition-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
exec bash harness/run-arm.sh igp-rc2 \
  "USE_INDUCTOR_GRAPH_PARTITION=true"
