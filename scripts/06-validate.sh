#!/usr/bin/env bash
# Health + DSpark smoke against the head node.
# VALIDATE_STACK=main (or passing 'main' as $1) sources pin.main.env; default is the live main-b12x pin.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [STACK] [MODEL]"
  echo "STACK: main (default) | nvfp4 | fp8 | golden | eugr"
  echo "MODEL: deepseek-v4-flash (default)"
  echo "Environment variables: HEAD_IP, SERVE_PORT"
  exit 0
fi

_INITIAL_HEAD_IP="${HEAD_IP:-}"
_INITIAL_SERVE_PORT="${SERVE_PORT:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

STACK="${VALIDATE_STACK:-}"
if [[ -z "${STACK}" ]]; then
  if [[ "${1:-}" =~ ^(nvfp4|main|fp8|golden|eugr)$ ]]; then
    STACK="$1"
    shift
  else
    STACK="main"
  fi
fi

case "${STACK}" in
  nvfp4) PIN="${ROOT}/configs/pin.nvfp4.env" ;;
  main) PIN="${ROOT}/configs/pin.main.env" ;;
  fp8) PIN="${ROOT}/configs/pin.env" ;;
  eugr) PIN="${ROOT}/configs/pin.eugr-b12x.env" ;;
  golden) PIN="${ROOT}/configs/pin.golden.env" ;;
  *) echo "VALIDATE_STACK must be nvfp4, main, fp8, eugr, or golden" >&2; exit 2 ;;
esac
# shellcheck disable=SC1090
source "${PIN}"
# shellcheck disable=SC1091
source "${ROOT}/configs/env.spark.sh"
if [[ -f "${ROOT}/configs/nodes.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/nodes.env"
fi
HEAD_IP="${_INITIAL_HEAD_IP:-${HEAD_IP:-127.0.0.1}}"
SERVE_PORT="${_INITIAL_SERVE_PORT:-${SERVE_PORT:-8000}}"

BASE="http://${HEAD_IP}:${SERVE_PORT}/v1"
MODEL="${1:-deepseek-v4-flash}"

echo "GET ${BASE}/models"
curl -fsS "${BASE}/models" | python3 -c 'import sys,json; m=json.load(sys.stdin)["data"][0]; print(m["id"], m.get("max_model_len"))'

echo "POST completions greedy France (32 tok)"
curl -fsS "${BASE}/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0,\"logprobs\":5}" \
  | python3 -c '
import sys, json
r = json.load(sys.stdin)
c = r["choices"][0]
text = c.get("text") or ""
print(repr(text))
print("usage", r.get("usage"))
lp = c.get("logprobs") or {}
tokens = lp.get("tokens") or []
tlogp = lp.get("token_logprobs") or []
top = lp.get("top_logprobs") or []
if tokens:
    first = tokens[0]
    flp = tlogp[0] if tlogp else None
    n_tie = None
    if top:
        first_top = top[0] if isinstance(top[0], dict) else {}
        if first_top:
            mx = max(first_top.values())
            n_tie = sum(1 for v in first_top.values() if v >= mx - 1e-3)
    print(f"first_token {first!r} logprob {flp} n_tie={n_tie}")
if "Paris" not in text:
    raise SystemExit("greedy missing Paris: " + repr(text))
'

echo "POST chat.completions (32 tok)"
curl -fsS "${BASE}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":32,\"temperature\":0}" \
  | python3 -c '
import sys, json
r = json.load(sys.stdin)
m = r["choices"][0]["message"]
content = m.get("content") or ""
print("content", repr(content))
print("reasoning", repr((m.get("reasoning") or m.get("reasoning_content") or "")[:240]))
print("usage", r.get("usage"))
if "Paris" not in content:
    raise SystemExit("chat missing Paris: " + repr(content))
'

echo "validate ok"
