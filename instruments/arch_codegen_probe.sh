#!/usr/bin/env bash
# Build arch_codegen_probe.cu for sm_120, sm_120f and sm_121a, then compare
# their SASS fingerprints and runtime on the GPU.
#
# This answers whether the arch string changes anything for kernels shaped like
# the ones vLLM compiles into its own extension, which is the only thing
# VLLM_PRESERVE_SM12X_TARGET can affect. Run it inside the pinned image (CUDA
# toolkit plus the GB10):
#
#   IMAGE=vllm-spark-0731:main-029-1rc0
#   docker run --rm --gpus all -v "$PWD/instruments":/probe -w /probe \
#     --entrypoint bash "$IMAGE" arch_codegen_probe.sh
set -euo pipefail
cd "$(dirname "$0")"

ARCHES="sm_120 sm_120f sm_121a"
for a in $ARCHES; do
    nvcc -O3 -std=c++17 -arch="$a" -o "probe_$a" arch_codegen_probe.cu
done

echo "=== SASS fingerprint per kernel (equal hashes mean identical code)"
python3 sass_fingerprint.py probe_sm_120 probe_sm_120f probe_sm_121a

echo
echo "=== GPU state during the run"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu \
    --format=csv,noheader

echo
echo "=== runtime (median of 20 samples, each sample inner launches)"
for a in $ARCHES; do
    echo "--- $a"
    "./probe_$a"
done

echo
echo "=== interleaved repeats, med_ms per kernel (rmsnorm is the one that moved)"
for round in 1 2 3; do
    for a in $ARCHES; do
        printf "round%s %-9s " "$round" "$a"
        "./probe_$a" | awk '/^(copy|rmsnorm|quant_fp8|gemm_f32)/ {printf "%s=%.3f ", $1, $3}'
        echo
    done
done

echo
echo "=== which ELF archs each binary carries"
for a in $ARCHES; do
    printf "%-9s %s\n" "$a" \
        "$(cuobjdump --list-elf "probe_$a" | grep -oE 'sm_[0-9]+[af]?' | sort -u | tr '\n' ' ')"
done
