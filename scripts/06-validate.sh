#!/usr/bin/env bash
# Health + DSpark smoke against the head node.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.env"
source "${ROOT}/configs/env.spark.sh"
if [[ -f "${ROOT}/configs/nodes.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/nodes.env"
fi

BASE="http://${HEAD_IP:-127.0.0.1}:${SERVE_PORT}/v1"
MODEL="${1:-deepseek-v4-flash}"

echo "GET ${BASE}/models"
curl -fsS "${BASE}/models" | python3 -c 'import sys,json; m=json.load(sys.stdin)["data"][0]; print(m["id"], m.get("max_model_len"))'

echo "POST chat.completions (32 tok)"
curl -fsS "${BASE}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in five words.\"}],\"max_tokens\":32}" \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); c=r["choices"][0]; print(c["message"].get("content")); print("usage", r.get("usage"))'

echo "validate ok"
