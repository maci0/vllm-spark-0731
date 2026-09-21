#!/usr/bin/env bash
# Recipe: port the matched-main pin to vLLM v0.30.0, then rebuild.
#
# Latest upstream serving tag as of 2026-09-21. It is one commit ahead of
# v0.30.0rc2 (`fa6ff060667f`): #57554 "[Build] Fix DeepGEMM CUDA 12.9 release
# builds", two files, build-only. One-variable change: VLLM_REF, nothing else.
#
#   configs/examples/port-v0.30.0.sh check   # pin + FAIL=0, no GPU
#   configs/examples/port-v0.30.0.sh phase1  # spark1: 02-build-main-029.sh
#   configs/examples/port-v0.30.0.sh overlay # phase 2 over phase-1
#   configs/examples/port-v0.30.0.sh copy    # spark1 -> spark2
#   configs/examples/port-v0.30.0.sh arm     # protocol on the new image
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.main-029.env"

WANT_REF=9ed533eb4adfe48aef7e569a08daeccd2a773fed
WANT_IMAGE=vllm-spark-0731:main-030-0
PHASE1="${PHASE1:-${WANT_IMAGE}-phase1}"

check() {
  grep -q "^VLLM_REF=${WANT_REF}$" "${ROOT}/configs/pin.main-029.env"
  grep -q "main-030-0" "${ROOT}/configs/pin.main-029.env"
  test "${VLLM_REF}" = "${WANT_REF}"
  test "${IMAGE}" = "${WANT_IMAGE}"
  echo "pin ok VLLM_REF=${VLLM_REF} IMAGE=${IMAGE}"
}

scan() {
  check
  exec bash "${ROOT}/scripts/port_scan.py" --with-upstream-patches
}

phase1() {
  check
  exec env MAX_JOBS="${MAX_JOBS:-16}" bash "${ROOT}/scripts/02-build-main-029.sh" "${PHASE1}"
}

overlay() {
  check
  docker image inspect "${PHASE1}" >/dev/null
  exec bash "${ROOT}/scripts/03-apply-main-overlays-029.sh" "${WANT_IMAGE}" "${PHASE1}"
}

copy() {
  check
  docker image inspect "${WANT_IMAGE}" >/dev/null
  exec bash "${ROOT}/scripts/02-copy-main.sh" "${WANT_IMAGE}"
}

arm() {
  check
  IMAGE="${WANT_IMAGE}" exec bash "${ROOT}/harness/run-arm.sh" proto2-030-0 ""
}

cmd="${1:-check}"
case "${cmd}" in
  check) check ;;
  scan) scan ;;
  phase1) phase1 ;;
  overlay) overlay ;;
  copy) copy ;;
  arm) arm ;;
  *) echo "usage: $0 check|scan|phase1|overlay|copy|arm" >&2; exit 2 ;;
esac
