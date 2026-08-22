#!/usr/bin/env bash
# Download only deepseek-ai/DeepSeek-V4-Flash-0731. No other Flash tag.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.env"
source "${ROOT}/configs/env.spark.sh"

DEST="${1:-${HOME}/models/ds4-flash-0731}"
mkdir -p "${DEST}"

if [[ "${HF_MODEL_ID}" != "deepseek-ai/DeepSeek-V4-Flash-0731" ]]; then
  echo "refusing to download ${HF_MODEL_ID}; pin is DeepSeek-V4-Flash-0731" >&2
  exit 2
fi

if command -v hf >/dev/null; then
  hf download "${HF_MODEL_ID}" --local-dir "${DEST}" --token "${HF_TOKEN:-}"
elif command -v huggingface-cli >/dev/null; then
  huggingface-cli download "${HF_MODEL_ID}" --local-dir "${DEST}"
else
  docker run --rm \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${DEST}:/data" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    python:3.12-slim \
    bash -lc 'pip install -q huggingface_hub && hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir /data'
fi

python3 "${ROOT}/patches/assert_0731.py" "${DEST}"
echo "0731 ready at ${DEST}"
