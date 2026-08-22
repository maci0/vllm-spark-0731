#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/pin.env"
source "${ROOT}/configs/env.spark.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }

command -v docker >/dev/null || fail "docker not on PATH"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not on PATH"
nvidia-smi >/dev/null || fail "nvidia-smi failed"

python3 - <<'PY' || fail "not aarch64 GB10/Spark (or SM parse failed)"
import platform, subprocess, sys
if platform.machine() not in ("aarch64", "arm64"):
    print("WARN: this recipe is for DGX Spark aarch64; machine is", platform.machine(), file=sys.stderr)
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
    text=True,
)
print(out.strip())
if "12.1" not in out and "GB10" not in out:
    print("WARN: expected GB10 / compute cap 12.1", file=sys.stderr)
PY

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is unset; gated downloads will fail"
fi

echo "prereq ok"
echo "  image pin: ${IMAGE}"
echo "  checkpoint pin: ${HF_MODEL_ID}"
echo "  stack: ${STACK:-golden}"
