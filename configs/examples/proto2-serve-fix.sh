#!/usr/bin/env bash
# Recipe: reproduce the proto2 serve fix, then measure the arm against the reference.
#
# Two failures stopped `vllm-spark-0731:main-029-proto2` from serving at all. Neither was
# in the new base, and both are checked here in the order they bite.
#
#   1. DeepGEMM rejected our own overlay's fp8_einsum recipe. `patch_einsum_sm12x_recipe`
#      forced `(1,128,128)` with `tma_aligned_scales=False` on SM12x; the SM120 kernel
#      reads packed UE8M0 and refuses the call at `csrc/utils/layout.hpp:113`
#      (`sf.size(-2) == ceil_div(mn, gran_mn)`) during `_initialize_kv_caches`, so
#      `VllmWorker-0` died and the engine never started. Upstream's own value,
#      `(1, 1, block_size)` with `tma_aligned_scales=True`, is the accepted layout.
#      `apply_main` no longer calls that overlay, and `assert_image.py` now asserts the
#      override is ABSENT rather than present.
#
#   2. InstantTensor sizes its I/O buffer from free device memory, so a warm page cache
#      starves it: `buffer_size (1059061760 B) exceeds device memory budget (925720576 B)`.
#      `harness/run-arm.sh` drops the caches on both nodes itself.
#
# Usage:
#   configs/examples/proto2-serve-fix.sh check    # static checks, no GPU
#   configs/examples/proto2-serve-fix.sh build    # phase 2 over the phase-1 base
#   configs/examples/proto2-serve-fix.sh arm [util]  # our arm + meter (default 0.86)
#   configs/examples/proto2-serve-fix.sh ref      # reference arm + meter, same day
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/vllm-spark-0731}"
PHASE1="${PHASE1:-vllm-spark-0731:main-029-proto2-phase1}"
TAG="${TAG:-vllm-spark-0731:main-029-proto2}"
REFERENCE="${REFERENCE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"
UTIL="${2:-0.86}"

# `apply_main` must not call any of the three einsum overlays.
check_static() {
  echo "== overlays that must stay off"
  python3 - "$ROOT/patches/apply_overlays.py" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
body = src[src.index("def apply_main("):]
body = body[: body.index("\ndef ", 10)]
for name in ("patch_einsum_sm12x_recipe", "patch_fp8_einsum_fallback",
             "patch_einsum_sm12x_scale_upcast"):
    live = [l for l in body.splitlines()
            if re.match(rf"\s+{name}\(", l)]
    assert not live, f"apply_main still calls {name}"
    print(f"  off: {name}")
PY
  echo "== assert_image must require the override ABSENT"
  grep -q 'assert "cap.major == 12" not in recipe_src' "$ROOT/patches/assert_image.py" \
    && echo "  ok" || { echo "  FAIL: stale recipe assertion"; return 1; }
  echo "== run-arm must drop the page cache on both nodes"
  grep -q 'drop_caches' "$ROOT/harness/run-arm.sh" \
    && echo "  ok" || { echo "  FAIL: page cache not dropped"; return 1; }
}

# Phase 2 on top of phase 1. The incremental form (no BASE_IMAGE) is NOT idempotent:
# it fails on `patch_o_proj_einsum_e8m0` with `missing needle in .../ops/o_proj.py`.
build_image() {
  cd "$REPO_DIR"
  scripts/03-apply-main-overlays-029.sh "$TAG" "$PHASE1"
  docker run --rm --entrypoint python3 "$TAG" -c '
import inspect
from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe as r
src = inspect.getsource(r)
assert "cap.major == 12" not in src, "SM12x override is still in the image"
assert "tma_aligned_scales = cap.major >= 10" in src, "upstream recipe not intact"
print("image ok: upstream recipe, no SM12x override")' 2>/dev/null | tail -1
  scripts/02-copy-main.sh "$TAG" || echo "warn: copy to spark2 reported an error (its --format ssh quoting is broken)"
}

case "${1:-check}" in
  check) check_static ;;
  build) check_static; build_image ;;
  arm)   cd "$REPO_DIR"; bash harness/run-arm.sh proto2 "GPU_MEMORY_UTILIZATION=$UTIL" ;;
  ref)   cd "$REPO_DIR"
         docker image inspect "$REFERENCE" >/dev/null
         ssh spark2 "docker image inspect $REFERENCE" >/dev/null
         bash harness/run-refg.sh ;;
  *)     echo "usage: $0 check|build|arm [util]|ref" >&2; exit 2 ;;
esac
