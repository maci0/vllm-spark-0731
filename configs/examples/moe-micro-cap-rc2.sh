#!/usr/bin/env bash
# Recipe: one-variable arm on v0.30.0rc2 — extend native micro MoE to c6.
#
# Native b12x has no static kernel (FlashInfer static IMA'd on live MXFP4).
# Micro owns num_tokens<=8 and routed_rows<64, so only protocol c1
# (8 tok * topk 6 = 48). c3/c5/c6 are dynamic. One overlay: raise
# _MICRO_MAX_TOKENS 8→48 and _MICRO_DYNAMIC_CUTOVER_PAIRS_DEFAULT 64→320
# so protocol c6 (48 tok * 6 = 288) stays micro. Same image, only this
# bind-mount. New hypothesis vs wrapper-size retries.
#
#   configs/examples/moe-micro-cap-rc2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

HOST_IMPL="${ROOT}/outputs/driver/one-off/b12x_fused_moe_impl.py"
CTR_IMPL="/usr/local/lib/python3.12/dist-packages/b12x/moe/fused_moe/_impl.py"
IMAGE="${IMAGE:-vllm-spark-0731:main-030-rc2}"
mkdir -p "$(dirname "${HOST_IMPL}")"
docker run --rm --entrypoint cat "${IMAGE}" "${CTR_IMPL}" > "${HOST_IMPL}"
python3 - "${HOST_IMPL}" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
t = p.read_text()
t = t.replace(
    "_MICRO_DYNAMIC_CUTOVER_PAIRS_DEFAULT = 64",
    "_MICRO_DYNAMIC_CUTOVER_PAIRS_DEFAULT = 320",
)
t = t.replace("_MICRO_MAX_TOKENS = 8", "_MICRO_MAX_TOKENS = 48")
if "_MICRO_MAX_TOKENS = 48" not in t or "_MICRO_DYNAMIC_CUTOVER_PAIRS_DEFAULT = 320" not in t:
    raise SystemExit("micro-cap patch did not apply")
p.write_text(t)
PY
# spark2 needs the same host path for the bind-mount.
rsync -a "${HOST_IMPL}" spark2:"${HOST_IMPL}"

exec bash harness/run-arm.sh micromax-rc2 \
  "SERVE_EXTRA_MOUNTS=${HOST_IMPL}:${CTR_IMPL}"
