#!/usr/bin/env bash
# Apply upstream backport patches (patches/upstream/pr-*.diff) onto a vLLM
# source tree. Use in the build before apply_overlays.py, or standalone to
# refresh a checkout.
#
# Usage: scripts/apply-upstream-patches.sh <vllm-src-dir>
set -uo pipefail
SRC="${1:?usage: apply-upstream-patches.sh <vllm-src-dir>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for p in "$ROOT"/patches/upstream/pr-*.diff; do
  name="$(basename "$p")"
  # Test-file hunks (tests/) are only present in a full repo checkout;
  # source hunks (vllm/) apply on both repo checkouts and installed trees.
  if (cd "$SRC" && patch -p1 --forward --dry-run < "$p" >/dev/null 2>&1); then
    (cd "$SRC" && patch -p1 --forward < "$p" >/dev/null)
    echo "applied $name"
  else
    echo "skip  $name (already applied or context mismatch)"
  fi
done
