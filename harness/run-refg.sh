#!/usr/bin/env bash
# Guarded reference measurement: launch the anemll arm, wait for health, meter 3 passes.
set -uo pipefail
cd "$HOME/vllm-spark-0731"
bash "$HOME/goal/launch-refbase.sh"
c=000
for i in $(seq 1 45); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
  sleep 10
done
echo "== refg health=$c after $((i*10))s"
[ "$c" = "200" ] || { echo "== REFERENCE NEVER HEALTHY"; exit 1; }
~/drive-median.sh refg 3 512 1 3 5 6
