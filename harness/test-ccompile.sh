#!/usr/bin/env bash
# Test the torch.compile overlay without a rebuild: apply it into a running container, commit
# that as a derived image, sync it to the worker, then try the reference's combination
# (breakable cudagraphs off + VLLM_COMPILE), which our image could not enter before.
#
# The derived image is an experiment artefact, not the image under measurement: it exists so
# the inductor question can be answered in minutes instead of a multi-hour rebuild.
set -uo pipefail
cd "$HOME/vllm-spark-0731"

TAG=vllm-spark-0731:main-029-1rc0-ccompile
BASE=vllm-spark-0731:main-029-1rc0   # what scripts/05-serve.sh uses by default now

sweep_shm() {
  find /dev/shm -maxdepth 1 \
    \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete 2>/dev/null || true
}

# 1. clean slate on both nodes: only containers we own may run
docker ps --format '{{.Names}}' | while read -r n; do
  case "$n" in vllm-ds4-0731) ;; *) echo "removing $n"; docker rm -f "$n" >/dev/null 2>&1 ;; esac
done
docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 'docker ps --format "{{.Names}}" | while read -r n; do case "$n" in vllm-ds4-0731) ;; *) docker rm -f "$n" >/dev/null 2>&1 ;; esac; done; docker rm -f vllm-ds4-0731 >/dev/null 2>&1'
sweep_shm

# 2. bring the unpatched image up on the working config. The worker must go first: this is a
# 2-node config, so a head launched alone blocks forever waiting for rank 1 and never reaches
# health, which is what an earlier version of this script did by omission.
ssh spark2 "cd ~/vllm-spark-0731 && GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' nohup bash scripts/05-serve.sh main-029 > ~/serve-cc3-w.log 2>&1 &"
sleep 85
GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS="--async-scheduling" \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-cc3-h.log" 2>&1
c=000
for i in $(seq 1 70); do
  c=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && break
  sleep 10
done
echo "unpatched container health=$c"
[ "$c" = "200" ] || { echo "ABORT: base image did not reach health"; exit 1; }

# 3. apply the overlay inside the running container
docker cp "$HOME/vllm-spark-0731/patches/apply_overlays.py" \
  vllm-ds4-0731:/opt/spark-0731/patches/apply_overlays.py
docker exec vllm-ds4-0731 python3 /opt/spark-0731/patches/apply_overlays.py \
  --stack main --only torch-compile-nvidia --vllm-dir /opt/vllm/vllm
# The decorator alone reaches Dynamo, which then rejects our all-reduce workspace probe for
# returning a bool. Both overlays are needed before the reference combination can start.
docker exec vllm-ds4-0731 python3 /opt/spark-0731/patches/apply_overlays.py \
  --stack main --only allreduce-dynamo-safe --vllm-dir /opt/vllm/vllm
# Third break: the mHC forwards call is_deep_gemm_supported(), which reaches a ctypes function
# pointer Dynamo cannot trace. The constant is read at import instead.
docker exec vllm-ds4-0731 python3 /opt/spark-0731/patches/apply_overlays.py \
  --stack main --only mhc-deep-gemm-static --vllm-dir /opt/vllm/vllm
# Fourth break: the traced forward reaches tf32_hc_prenorm_gemm -> _lazy_init() ->
# has_deep_gemm() -> importlib.import_module, which Dynamo skips (importlib is a skip dir).
docker exec vllm-ds4-0731 python3 /opt/spark-0731/patches/apply_overlays.py \
  --stack main --only deep-gemm-eager-init --vllm-dir /opt/vllm/vllm
# Fifth break: tf32_hc_prenorm_gemm itself is a pybind11 op Dynamo cannot trace, and the local
# import that precedes each call reaches importlib. Both go behind torch.compiler.disable.
docker exec vllm-ds4-0731 python3 /opt/spark-0731/patches/apply_overlays.py \
  --stack main --only mhc-tf32-uncaptured --vllm-dir /opt/vllm/vllm
# The wrapper alone is not enough: the call site must be redirected to it. Overlay 5's call-site
# match assumed a different local-import spelling and matched nothing, so this is its companion.
docker exec vllm-ds4-0731 python3 /opt/spark-0731/patches/apply_overlays.py \
  --stack main --only mhc-tf32-redirect --vllm-dir /opt/vllm/vllm
echo "=== overlays applied inside the container? ==="
docker exec vllm-ds4-0731 grep -c "_tf32_hc_prenorm_gemm_uncaptured" \
  /opt/vllm/vllm/model_executor/kernels/mhc/tilelang.py
docker exec vllm-ds4-0731 grep -n "tf32_hc_prenorm_gemm(" \
  /opt/vllm/vllm/model_executor/kernels/mhc/tilelang.py | head -5
docker exec vllm-ds4-0731 grep -n "support_torch_compile" \
  /opt/vllm/vllm/models/deepseek_v4/nvidia/model.py | head -3
docker exec vllm-ds4-0731 grep -n "is_compiling" \
  /opt/vllm/vllm/distributed/communication_op.py | head -3
docker exec vllm-ds4-0731 grep -n "_USE_DEEP_GEMM" \
  /opt/vllm/vllm/model_executor/kernels/mhc/tilelang.py | head -4
docker exec vllm-ds4-0731 grep -n "_EAGER_INIT_DONE" \
  /opt/vllm/vllm/utils/deep_gemm.py | head -3

# 4. commit the patched container and give the worker the same image
docker commit vllm-ds4-0731 "$TAG" >/dev/null 2>&1 && echo "committed $TAG"
docker save "$TAG" | ssh spark2 docker load 2>&1 | tail -2

# 5. try the reference combination on the derived image
docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 'docker rm -f vllm-ds4-0731 >/dev/null 2>&1'
sweep_shm
ssh spark2 "find /dev/shm -maxdepth 1 \\( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \\) -delete 2>/dev/null; cd ~/vllm-spark-0731 && IMAGE=$TAG VLLM_USE_BREAKABLE_CUDAGRAPH=0 COMPILATION_MODE=3 VLLM_USE_AOT_COMPILE=0 GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' nohup bash scripts/05-serve.sh main-029 > ~/serve-cc3-w.log 2>&1 &"
sleep 85
IMAGE="$TAG" VLLM_USE_BREAKABLE_CUDAGRAPH=0 COMPILATION_MODE=3 VLLM_USE_AOT_COMPILE=0 \
  GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS="--async-scheduling" \
  bash scripts/05-serve.sh main-029 > "$HOME/serve-cc3b-h.log" 2>&1

c=000
for i in $(seq 1 80); do
  c=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && { echo "HEALTHY after $((i*10))s"; break; }
  docker ps --format '{{.Names}}' | grep -q '^vllm-ds4-0731$' || { echo "container gone"; break; }
  sleep 10
done
echo "health=$c"

if [ "$c" = "200" ]; then
  docker logs --tail 8000 vllm-ds4-0731 2>&1 \
    | grep -oE 'v0\.1\.1\.dev0\+g[0-9a-f]+|"mode": [0-9]|cudagraph_mode.: [A-Z_]+' | sort -u | head -4
  ~/drive-median.sh proto-ccompiled 3 512 1 3 5 6 2>&1 | tail -9
else
  echo "=== why it did not start ==="
  docker logs --tail 3000 vllm-ds4-0731 2>&1 \
    | grep -iE 'RuntimeError:|ValueError:|Cannot|not supported|inductor' | tail -5
fi
