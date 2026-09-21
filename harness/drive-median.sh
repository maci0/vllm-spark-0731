#!/usr/bin/env bash
# Run the metered levels N times and report the per-level median.
#
#   drive-median.sh <tag> <passes> <max_tokens> [levels...]
#
# One pass at 128 tokens swings by up to 18% on this rig, which is larger than any
# effect the k sweep was trying to measure. Use 512-token generations and at least
# three passes for anything that has to be believed.
set -uo pipefail

TAG="${1:?usage: drive-median.sh <tag> <passes> <max_tokens> [levels...]}"
PASSES="${2:?}"
MAXTOK="${3:?}"
shift 3 || true
LEVELS=("${@:-1 3 5 6}")

ROOT="$HOME/vllm-spark-0731"
OUT="$ROOT/outputs/driver"
mkdir -p "$OUT"
LOG="$OUT/$TAG.median.log"
exec > >(tee -a "$LOG") 2>&1
echo "== $TAG: $PASSES passes at max_tokens=$MAXTOK, levels ${LEVELS[*]}, start $(date -Is)"

# Record which engine actually answers before trusting a single number. Every result file
# carries the port holder and the engine version, so a run that reached the wrong container
# is visible in the artefact rather than inferred later. Add an assertion on the version
# string here when measuring a specific arm.
echo "== port holder: $(ss -ltnp 2>/dev/null | grep -E "[:.]8000[[:space:]]" | head -1)"
_cont="$(docker ps --format '{{.Names}}' | head -1)"
echo "== container: ${_cont:-none} $(docker ps --format '{{.Status}}' | head -1)"
echo "== engine: $(docker logs --tail 4000 "${_cont}" 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+\.dev0\+g[0-9a-f]+' | tail -1)"

for p in $(seq 1 "$PASSES"); do
  echo "-- pass $p at $(date -Is)"
  ~/drive-meter.sh "$TAG-$p" "$MAXTOK" "${LEVELS[@]}"
done

python3 "$ROOT/.scratch/median_levels.py" "$TAG" "$PASSES" "${LEVELS[@]}"
echo "== $TAG done $(date -Is)"
