#!/usr/bin/env bash
# Recipe: same-day anemll reference on 2026-09-20.
#
# Same as configs/examples/refg-rc2.sh, new tag so the log does not
# collide with yesterday's `refg-rc2`. Image is
# ghcr.io/anemll/dspark-vllm-gx10:0.1.1 via ~/goal/launch-refbase.sh.
#
#   configs/examples/refg-rc2b.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
TAG="${TAG:-refg-rc2b}"
bash "${HOME}/goal/launch-refbase.sh"
exec bash "${HOME}/drive-median.sh" "${TAG}" 3 512 1 3 5 6
