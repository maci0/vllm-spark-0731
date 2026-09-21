#!/usr/bin/env bash
# Per-concurrency measurement with a step-rate readout.
#
#   drive-meter.sh <tag> [max_tokens] [levels...]
#
# For each level: snapshot the counters, run the golden bench at that level,
# snapshot again, and report wall time, steps/s, step ms, tokens/step and the
# acceptance rate. That separates the two halves of the throughput gap (how many
# tokens a step returns versus how long a step takes) instead of reporting one
# aggregate number.
set -uo pipefail

TAG="${1:?usage: drive-meter.sh <tag> [max_tokens] [levels...]}"
MAXTOK="${2:-128}"
shift 2 2>/dev/null || shift 1
LEVELS=("${@:-1 3 5 6}")
BASE=http://127.0.0.1:8000/v1
ROOT="$HOME/vllm-spark-0731"
OUT="$ROOT/outputs/driver"
mkdir -p "$OUT"
RES="$OUT/$TAG.meter.txt"
LOG="$OUT/$TAG.meter.log"
exec > >(tee -a "$LOG") 2>&1
: > "$RES"
echo "== $TAG metered start $(date -Is)"

health=""
for i in $(seq 1 130); do
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8000/health || true)
  if [ "$code" = "200" ]; then health=$code; break; fi
  sleep 10
done
echo "== health=$health after $i polls at $(date -Is)"
[ "$health" = "200" ] || { echo "== never healthy"; exit 1; }

req() {
  curl -s -m 300 "$BASE/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v4-flash\",\"prompt\":\"$1\",\"max_tokens\":$2,\"temperature\":0}" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(repr(d["choices"][0]["text"]))' 2>/dev/null
}
FR="$(req 'The capital of France is' 6)"
MUL="$(req '9x8=' 6)"
{
  echo "== tag $TAG"
  echo "== gate_france $FR"
  echo "== gate_9x8 $MUL"
} | tee -a "$RES"

# One warm pass so the first measured level is not paying JIT and cache warmup.
python3 "$ROOT/scripts/bench-concurrency.py" --chat --max-tokens 64 --levels 1 \
  --seed 1234 >/dev/null 2>&1

for L in "${LEVELS[@]}"; do
  python3 "$ROOT/.scratch/stepmeter.py" snap "$OUT/snap_a.json" >/dev/null
  t0=$(date +%s.%N)
  python3 "$ROOT/scripts/bench-concurrency.py" --chat --max-tokens "$MAXTOK" --seed 1234 \
    --levels "$L" --no-warm 2>&1 | tail -2 | tee -a "$RES"
  t1=$(date +%s.%N)
  python3 "$ROOT/.scratch/stepmeter.py" snap "$OUT/snap_b.json" >/dev/null
  echo "-- level $L" | tee -a "$RES"
  python3 "$ROOT/.scratch/stepmeter.py" report "$OUT/snap_a.json" "$OUT/snap_b.json" \
    "$(echo "$t1 - $t0" | bc)" | tee -a "$RES"
done
echo "== wrote $RES"
echo "== $TAG metered done $(date -Is)"
