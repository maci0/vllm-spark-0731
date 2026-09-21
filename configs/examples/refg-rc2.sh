#!/usr/bin/env bash
# Recipe: same-day anemll reference on the current pin.
#
# Image is ghcr.io/anemll/dspark-vllm-gx10:0.1.1, launched via
# ~/goal/launch-refbase.sh (spark-launch + ~/goal/ref-base0731.yaml).
# Meter with drive-median.sh so the comparison is the repo protocol.
#
#   configs/examples/refg-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
TAG="${TAG:-refg-rc2}"
bash "${HOME}/goal/launch-refbase.sh"
exec bash "${HOME}/drive-median.sh" "${TAG}" 3 512 1 3 5 6
