#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — nvidia-cutlass-dsl 4.7.1.
#
# Latest tagged cutlass-dsl is 4.7.1 (2026-08-26). Pin image is 4.7.0.
# One variable: pip bump of nvidia-cutlass-dsl[cu13] + libs-base +
# libs-cu13, committed as vllm-spark-0731:main-030-rc2-dsl471. No
# overlay, no other dep, no rebuild of vLLM/b12x.
#
#   configs/examples/cutlass-dsl-471-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export IMAGE=vllm-spark-0731:main-030-rc2-dsl471
exec bash harness/run-arm.sh dsl471-rc2 ""
