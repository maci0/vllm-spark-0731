#!/usr/bin/env bash
# Pull the prebuilt vLLM image for the selected stack. No source rebuild.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STACK="${1:-fp8}"
case "${STACK}" in
  fp8) PIN="${ROOT}/configs/pin.env" ;;
  nvfp4) PIN="${ROOT}/configs/pin.nvfp4.env" ;;
  eugr) PIN="${ROOT}/configs/pin.eugr-b12x.env" ;;
  golden) PIN="${ROOT}/configs/pin.golden.env" ;;
  *) echo "usage: $0 [fp8|nvfp4|eugr|golden]" >&2; exit 2 ;;
esac
# shellcheck disable=SC1090
source "${PIN}"
# shellcheck disable=SC1091
source "${ROOT}/configs/env.spark.sh"

python3 "${ROOT}/patches/assert_stack.py" \
  --kv "${KV_CACHE_DTYPE}" \
  --attn "${ATTENTION_BACKEND:-}" \
  --moe "${MOE_BACKEND}"

echo "pull ${IMAGE}"
docker pull "${IMAGE}"
docker image inspect "${IMAGE}" --format '{{.Id}} {{.RepoTags}}'
