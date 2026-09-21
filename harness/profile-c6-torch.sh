#!/usr/bin/env bash
# In-process torch profile of a c6 decode window on the standing pin.
#
# nsys `--attach-pid` cannot enable CUDA tracing on a process that was not
# launched under it (no CUPTI injection), so the kernel table has to come from
# vLLM's own torch profiler, which 05-serve.sh already plumbs behind
# PROFILE_ENABLE=1 (POST /start_profile, /stop_profile).
#
#   profile-c6-torch.sh <tag>
set -uo pipefail
cd "$HOME/vllm-spark-0731"

TAG="${1:-torchc6}"
OUT="$HOME/vllm-spark-0731/outputs/driver/one-off"
mkdir -p "$OUT"
LOG="$OUT/$TAG-torch.txt"
PROF_DIR=/tmp/prof-$TAG

docker rm -f vllm-ds4-0731 >/dev/null 2>&1
# PROFILE_ENABLE is host-side (05-serve.sh), so the worker node needs it too.
# Without it the rank-1 worker has ProfilerConfig unset and /start_profile
# fails with "Profiling is not enabled".
ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1; cd ~/vllm-spark-0731 && PROFILE_ENABLE=1 PROFILE_DIR=$PROF_DIR GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' nohup bash scripts/05-serve.sh main-029 > ~/serve-$TAG-w.log 2>&1 </dev/null &"
sleep 85
PROFILE_ENABLE=1 PROFILE_DIR="$PROF_DIR" GPU_MEMORY_UTILIZATION=0.8389 \
  SERVE_EXTRA_ARGS='--async-scheduling' \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-$TAG-h.log" 2>&1

c=000
for i in $(seq 1 45); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
  docker ps --format '{{.Names}}' | grep -q '^vllm-ds4-0731$' || { echo "container gone"; exit 1; }
  sleep 10
done
echo "== health=$c after $((i*10))s"
[ "$c" = "200" ] || { tail -30 "$HOME/serve-$TAG-h.log"; exit 1; }

python3 scripts/bench-concurrency.py --chat --max-tokens 512 --seed 1234 \
  --levels 6 --no-warm >/dev/null 2>&1
echo "== warm c6 done"

echo "== start_profile: $(curl -s -m 20 -X POST http://127.0.0.1:8000/start_profile)"
python3 scripts/bench-concurrency.py --chat --max-tokens 512 --seed 1234 \
  --levels 6 --no-warm >/dev/null 2>&1
echo "== stop_profile: $(curl -s -m 60 -X POST http://127.0.0.1:8000/stop_profile)"

echo "== traces in container:"
docker exec vllm-ds4-0731 bash -c "find $PROF_DIR -type f 2>/dev/null | head"
mkdir -p "$OUT/$TAG-traces"
docker cp "vllm-ds4-0731:$PROF_DIR" "$OUT/$TAG-traces/" 2>&1 | tail -2
ls -l "$OUT/$TAG-traces" 2>/dev/null

{
  echo "== torch c6 kernel table $TAG $(date -Is)"
  python3 - "$OUT/$TAG-traces" <<'PY'
import glob, json, gzip, os, sys, collections

root = sys.argv[1]
files = []
for pat in ("**/*.pt.trace.json", "**/*.json", "**/*.json.gz"):
    files += glob.glob(os.path.join(root, pat), recursive=True)
files = sorted(set(files), key=os.path.getsize, reverse=True)
print("trace files:", len(files))
if not files:
    sys.exit(0)
path = files[0]
print("reading", path, os.path.getsize(path), "bytes")
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt", errors="ignore") as fh:
    data = json.load(fh)
ev = data.get("traceEvents", data if isinstance(data, list) else [])
kern = collections.Counter()
kcount = collections.Counter()
cat = collections.Counter()
for e in ev:
    c = e.get("cat", "")
    cat[c] += 1
    if c == "kernel" and e.get("dur"):
        name = e.get("name", "?")
        kern[name] += e["dur"]
        kcount[name] += 1
print("categories:", cat.most_common(8))
tot = sum(kern.values())
print(f"total kernel us={tot:.0f} kernels={sum(kcount.values())} unique={len(kern)}")
for name, us in kern.most_common(40):
    print(f"{us/1000.0:9.2f} ms  {100.0*us/tot:5.1f} %  n={kcount[name]:6d}  {name[:110]}")
PY
} 2>&1 | tee "$LOG"
echo "== wrote $LOG"
