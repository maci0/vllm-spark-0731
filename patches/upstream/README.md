# Upstream backport patches

Clean git-diff patches fetched from vLLM PRs, for applying into our build
(vLLM `a00a3544b93e`, the `v0.30.0rc1` tag commit) via `git apply` /
`patch -p1` before the string-replace overlays in `apply_overlays.py` run.

Order matters. An overlay needle is written against the post-patch text, so the patches
run first and the overlays key on what they leave behind. `patch_cutlass_sm12x_guard`
matches `is_supported` in `scaled_mm/cutlass.py` as `pr-53055.diff` leaves it (typed
signature, `"CUTLASS block FP8 is not supported."`), not as the pristine base has it;
keying it on the base text fails the overlay with `missing needle`.

## Open-PR backports (fetched 2026-08-24 via `gh pr diff`)

| Patch | PR | Fix | Equivalent overlay (`--only`) |
|---|---|---|---|
| `pr-53055.diff` | #53055 (OPEN) | guard DeepGEMM in mHC pre-broadcast + exclude CUTLASS FP8 on SM12x | `patch_mhc`, `patch_cutlass_sm12x_guard` (applied in `--stack main`) |
| `pr-53425.diff` | #53425 (OPEN) | DSV4 sparse MLA kernel block size 64 on SM12x; lazy `sparse_mla` import in the indexer (kills the `vllm._aiter_ops` cold-start import cycle); **re-anchored 2026-09-14** for `7ee8a6dd`: the base added a comment above the indexer's `return [256]` and moved the method to line 246, so the hunk was regenerated from that base (the overlay's early-return would otherwise have skipped the indexer half, leaving the indexer at `[256]` against sparse MLA's `[64]`) | `dsv4-block64` |
| `pr-53521.diff` | #53521 **CLOSED** 2026-08-27 | Hopper `fp8_einsum` recipe on SM12x — NOT NEEDED: stock `(1,1,128)` + packed E8M0 scales verified correct on GB10 (mean_rel 0.000000; E2E France coherent on `main-b12x-mn2`). **Drop this backport.** | `einsum-sm12x` |
| `pr-53522.diff` | #53522 (OPEN) | gate indexer paged MQA metadata on `is_deep_gemm_supported()` | `indexer-mqa` |
| `pr-53898.diff` | #53898 **CLOSED** 2026-08-27 | SM12x fp8_einsum dequant fallback + unpack — NOT NEEDED: the einsum kernel is correct with packed scales; the fallback itself was the E2E-garbage source (packed-int32-as-fp32). Real upstream fix: deepseek-ai/DeepGEMM #337 (packer mantissa mask). **Drop this backport; mn2 uses the stock path.** | `einsum-sm12x` family |
| `pr-52499.diff` | #52499 (OPEN) | DSV4 sparse-MLA spec-decode query shapes | comment-only (we didn't need it after TOPK=192) |
| `pr-47988.diff` | #47988 (OPEN, head `e1dbe81c` 2026-09-12; **CUTLASS hunks only**, refetched 2026-09-14) | CUTLASS SM12x `can_implement` N%128 fall-through + E8M0 weight-scale upcast at load. The Triton unconditional-upcast hunk is **gone**: `v0.29.1rc0` contains `e77daef89e` (#56214), which landed it upstream, so the hunk no longer applies and `patch_triton_e8m0_sm12x` skips. The head's `_upcast_e8m0_to_fp32` rewrite is also not carried: it differs from the base's bit-shift helper at exponent bytes 0 and 255 (measured 2026-09-14) and neither fix needs it. | `triton-e8m0-sm12x` (family-120 gate variant) |

`pr-41834` (SM12x umbrella) is **not** fetched: diff exceeds the 20k-line
GitHub limit and the PR needs-rebase — comment only, per `docs/UPSTREAM.md`.

## DeepGEMM patches

None, and `docker/Dockerfile.main` no longer selects DeepGEMM itself: vLLM's cmake FetchContent
pin (`cmake/external_projects/deepgemm.cmake`) fetches the fork, which is what this base needs
(its `_hc_prenorm` path passes `activation_alpha=`, and the cmake globs
`third-party/deep_jit/include`; neither exists in `a6b593d`). Our `DEEPGEMM_SRC_DIR` checkout and
the `DEEPGEMM_COMMIT` build arg were removed 2026-09-17. A DeepGEMM source patch would now be a
patch against that FetchContent tree; the upstream work the old staged port came from is
[DeepGEMM#419](https://github.com/deepseek-ai/DeepGEMM/pull/419), still open.

## Merged-fix patches (already in the build)

| Patch | PR | Status | Applied as |
|---|---|---|---|
| `0002-pr-50645-mhc-tilelang.diff` | #50645 | superseded by #53055 | `patch_mhc` |
| `0003-nvfp4-ds-mla-v0.27.1.patch` | local | local | `patch_nvfp4_ds_mla` |
| `b12x-utils-main.py` | #52018 | merged | copied by `patch_utils_b12x` |

## v0.27.1-only variants (from `maci0/vllm-spark-nvfp4`, merged 2026-08-25)

Trimmed variants of the same PRs, scoped to what applies on a pristine
`vllm/vllm-openai:v0.27.1` base (the rc2 overlay fallback). Not used by the
main track; kept for the historical v0.27.1 build (`docker/Dockerfile.nvfp4`,
`patches/v0.27.1/combined-v0.27.1.patch`).

| Patch | PR / origin | Relation to the files above |
|---|---|---|
| `mhc-guard-50645-vllm-only.diff` | #50645 | superseded by #53055 (`pr-53055.diff`) |
| `kv-offload-bounds-check.patch` | ours, #53271 (open) | diagnosability only; see `docs/field-notes/nvfp4/KV_OFFLOAD_MLA.md` |

## Apply (in `scripts/apply-upstream-patches.sh` and `docker/Dockerfile.main-overlays`)

```bash
cd "$VLLM_SRC"  # fresh vLLM main checkout (build step)
for p in /opt/spark-0731/patches/upstream/pr-*.diff; do
  patch -p1 --forward -N < "$p" >/dev/null 2>&1 && echo "applied $(basename "$p")" || echo "skip $(basename "$p")"
done
```

Notes:
- Patches are against vLLM `main` at the PRs' merge bases (~2026-08-23/24) unless a
  row says they were re-anchored to our pin; re-fetch if our pinned vLLM commit drifts.
  Re-fetch `pr-47988.diff` as CUTLASS hunks only: the PR's own diff no longer contains
  the Triton upcast hunk, because main gained it in `e77daef89e` (#56214), which is in
  `v0.29.1rc0`.
- The glob is `pr-*.diff` on purpose. `pr54631.diff` and `pr47988.diff` have no hyphen
  and are never applied by the image build; both touch only `tests/`, which the
  installed tree does not need.
- Verified against `a00a3544b93e` (`v0.30.0rc1`) on 2026-09-18: `patch -p1 --forward -N`
  applies `pr-47988` / `pr-53425` / `pr-53522` with rc=0 (`pr-53055` skips as already
  applied), and `scripts/port_scan.py --with-upstream-patches` reports FAIL=0 over the
  whole overlay stack. The `mhc-tf32` pair that used to abort it is parked; see
  `patches/README.md`. Same three-apply / one-skip pattern as `f37c550bf635`
  (`proto-v0.2.0`) on 2026-09-17.
- #53574 merged 2026-08-31 (`699e180df4`, an ancestor of the pin), so `pr-53574.diff`
  and the `flashinfer-eidx-contig` overlay it fed are retired. `build_c128a_topk_metadata`
  now returns the full-width buffer slice on family 120 (contiguous at the builder), and
  the C4A path is `empty_like`-contiguous, so the consumer `.contiguous()` was a no-op.
- The equivalent overlays are idempotent, so applying a patch AND the overlay
  is safe (the overlay skips when already applied); prefer the patch and keep
  the overlay as the fallback for the rc2 overlay image.
