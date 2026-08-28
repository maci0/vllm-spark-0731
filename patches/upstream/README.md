# Upstream backport patches

Clean git-diff patches fetched from vLLM PRs, for applying into our build
(vLLM main `e25c586b9`) via `git apply` / `patch -p1` before the string-replace
overlays in `apply_overlays.py` run.

## Open-PR backports (fetched 2026-08-24 via `gh pr diff`)

| Patch | PR | Fix | Equivalent overlay (`--only`) |
|---|---|---|---|
| `pr-53055.diff` | #53055 (OPEN) | guard DeepGEMM in mHC pre-broadcast + exclude CUTLASS FP8 on SM12x | `patch_mhc`, `patch_cutlass_sm12x_guard` (applied in `--stack main`) |
| `pr-53425.diff` | #53425 (OPEN) | DSV4 sparse MLA kernel block size 64 on SM12x; refreshed 2026-08-26 (ed71de5): lazy `sparse_mla` import in indexer (kills the `vllm._aiter_ops` cold-start import cycle) | `dsv4-block64` |
| `pr-53521.diff` | #53521 **CLOSED** 2026-08-27 | Hopper `fp8_einsum` recipe on SM12x — NOT NEEDED: stock `(1,1,128)` + packed E8M0 scales verified correct on GB10 (mean_rel 0.000000; E2E France coherent on `main-b12x-mn2`). **Drop this backport.** | `einsum-sm12x` |
| `pr-53522.diff` | #53522 (OPEN) | gate indexer paged MQA metadata on `is_deep_gemm_supported()` | `indexer-mqa` |
| `pr-53898.diff` | #53898 **CLOSED** 2026-08-27 | SM12x fp8_einsum dequant fallback + unpack — NOT NEEDED: the einsum kernel is correct with packed scales; the fallback itself was the E2E-garbage source (packed-int32-as-fp32). Real upstream fix: deepseek-ai/DeepGEMM #337 (packer mantissa mask). **Drop this backport; mn2 uses the stock path.** | `einsum-sm12x` family |
| `pr-52499.diff` | #52499 (OPEN) | DSV4 sparse-MLA spec-decode query shapes | comment-only (we didn't need it after TOPK=192) |
| `pr-53574.diff` | #53574 (OPEN) | C128A eidx contiguity at the builder (`_build_c128a_metadata`); root cause of the "eidx must be contiguous" DSV4+spec-decode boot crash | `flashinfer-eidx-contig` (consumer-side) |
| `pr-47988.diff` | #47988 (OPEN) | unconditional E8M0→fp32 upcast in `w8a8_triton_block_scaled_mm` + CUTLASS SM12x `can_implement`/weight-scale handling (source hunks only) | `triton-e8m0-sm12x` (family-120 gate variant) |

`pr-41834` (SM12x umbrella) is **not** fetched: diff exceeds the 20k-line
GitHub limit and the PR needs-rebase — comment only, per `docs/UPSTREAM.md`.

## DeepGEMM backports (applied in `docker/Dockerfile.main` via `*deepgemm*.diff`)

| Patch | PR | Fix | Status |
|---|---|---|---|
| `deepgemm-pr-403.diff` | [deepseek-ai/DeepGEMM#403](https://github.com/deepseek-ai/DeepGEMM/pull/403) | SM120/SM121 SF layout transformation in `csrc/apis/layout.hpp` | Merged in `nv_dev 8b1392b978f5`; applied idempotently in `docker/Dockerfile.main` before vLLM compilation |
| `deepgemm-fp8-1d1d-port.diff` | anemll 2.5.0 port | Port of golden anemll 2.5.0 `sm100_fp8_gemm_1d1d` kernel and dispatch to `nv_dev` | Staged local port for SM120 pure-FP8 GEMM; applied in `docker/Dockerfile.main` before compilation. **Upstreamed 2026-08-26 as [deepseek-ai/DeepGEMM#419](https://github.com/deepseek-ai/DeepGEMM/pull/419)** — refreshed to the reviewed state (44d9d2e: pure-fp8 excluded from AB-swap, `allow_swap_ab` layout filtering, arch-10 mixed-dtype routing, epilogue/math.cuh fixes) |

## Merged-fix patches (already in the build)

| Patch | PR | Status | Applied as |
|---|---|---|---|
| `0001-pr-52018-b12x-moe-v0.27.1.diff` | #52018 | merged | `copy_new_modules` + `patch_moe_backend` + … |
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
| `b12x-linear-52016-vllm-only.diff` | #52016 (merged) | vllm-only hunks for the v0.27.1 base |
| `b12x-moe-52018-vllm-only.diff` | #52018 (merged) | vllm-only hunks; 4 hunks dropped (target code absent in 0.27.1) |
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
- Patches are against vLLM `main` at the PRs' merge bases (~2026-08-23/24);
  re-fetch if our pinned vLLM commit drifts.
- The equivalent overlays are idempotent, so applying a patch AND the overlay
  is safe (the overlay skips when already applied); prefer the patch and keep
  the overlay as the fallback for the rc2 overlay image.
