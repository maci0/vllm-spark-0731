#!/usr/bin/env bash
# Recipe: port the matched-main pin to vLLM v0.31.0, then rebuild.
#
# Latest upstream serving tag (db9527a46873). Four overlay needles moved with
# it and are fixed in patches/apply_overlays.py; the pin change is VLLM_REF
# plus IMAGE, nothing else.
#
#   configs/examples/port-v0.31.0.sh check   # pin + FAIL=0, no GPU
#   configs/examples/port-v0.31.0.sh scan    # overlay scan against the tag
#   configs/examples/port-v0.31.0.sh phase1  # spark1: 02-build-main-029.sh
#   configs/examples/port-v0.31.0.sh overlay # phase 2 over phase-1
#   configs/examples/port-v0.31.0.sh copy    # spark1 -> spark2
#   configs/examples/port-v0.31.0.sh arm     # protocol on the new image
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.main-029.env"

WANT_REF=db9527a46873454610df6dbedf79a36d6bf1a7f6
WANT_IMAGE=vllm-spark-0731:main-031-0
PHASE1="${PHASE1:-${WANT_IMAGE}-phase1}"

check() {
  grep -q "^VLLM_REF=${WANT_REF}$" "${ROOT}/configs/pin.main-029.env"
  grep -q "main-031-0" "${ROOT}/configs/pin.main-029.env"
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
  IMAGE="${WANT_IMAGE}" exec bash "${ROOT}/harness/run-arm.sh" proto2-031-0 ""
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
