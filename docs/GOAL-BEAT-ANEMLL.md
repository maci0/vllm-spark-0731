# Goal: beat the anemll reference

Standing contract for every continuation of this work. Numbers and pin SHAs live
in `configs/pin.main-029.env`, `docs/EXPERIMENTS.md`, and `docs/UPSTREAM.md`.
This file is the rules. Do not copy a snapshot into here.

## Win

Same rig, same day, both arms:

    ~/drive-median.sh <tag> 3 512 1 3 5 6

Beat `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` at **c6** and on the **c1+c3+c5+c6
sum**, by more than the larger of the two arms' median spreads. Every pass on
both arms must pass `gate_france` (`' Paris...'`) and `gate_9x8` (`'72, 9x9'`).
Our acceptance rate and tokens per step must not fall.

If that is not reachable: stop with the best measured arm on the current pin,
plus written attribution (layer family, library, measurement) in
`docs/UPSTREAM.md`.

## Always-current base

The image is never a stock tag. It is:

1. the latest **upstream tagged serving** vLLM (`gh api repos/vllm-project/vllm/tags`)
2. the latest **tagged** version of each dep in `configs/pin.main-029.env`
3. this repo's overlay set on top (`patches/upstream/pr-*.diff`, then
   `patches/apply_overlays.py --stack main`, modules in `patches/files/`)

Source of truth for versions is the pin file, not this document.

**On every continuation, before any measurement:**

```
gh api repos/vllm-project/vllm/tags?per_page=3
```

plus the latest GitHub/PyPI tag for every `*_REF` / `*_VERSION` in the pin.
If any is newer than the pin:

1. Update the pin (one tag family at a time if they conflict).
2. `scripts/port_scan.py --with-upstream-patches` on a **clean** tree until
   `FAIL=0`. Dirty-tree FAILs after a first apply are mutation artifacts; reset
   and scan once.
3. Rebuild phase1 + overlays + copy to spark2. Recipe pattern:
   `configs/examples/port-<tag>.sh`.
4. Re-baseline the protocol. Do not A/B a new pin against a previous-pin number.
5. Same-day anemll `refg` on the new pin before claiming a beat.

Do not follow untagged `main` / `dev` HEADs. Do not bump a dep the current
stack cannot import until that conflict is its own one-variable arm. Known
holds (re-check on each port; drop when the conflict is gone):

- b12x 1.3.0 pins cutlass-dsl 4.6.2; image is 4.7.0. Isolate before bump.
- triton 3.8 is not torch 2.14's 3.7.1.
- tilelang 0.1.14 requires `apache-tvm-ffi<0.1.13`; tokenspeed-mla 0.2.9
  requires 0.1.13.post3.

## Overlay policy

`pr-*.diff` first, overlays second. Needles are written against post-patch
text. Drop a backport the day its PR is in the tag we pin. Re-anchor a hunk
when a new tag moves the needle; do not rewrite the intent.

Local measurement hooks (profilers, `SERVE_EXTRA_ENV`) stay local. Everything
else that is not a hook must have or get a proper upstream PR (vLLM,
DeepGEMM, FlashInfer, or b12x). Comment on the existing PR; do not open a
duplicate.

## Upstream PRs

Check every round. Record in `docs/UPSTREAM.md`. Act only on rebase, review
comments, or merge.

Ours (`maci0`), still behind the `pre-run-check` **label** gate (not a code
failure). Never request that label as an AI agent. Never open a new PR
without the owner:

| PR | what |
|---|---|
| [#53425](https://github.com/vllm-project/vllm/pull/53425) | SM12x DSV4 block size 64 (`pr-53425.diff`) |
| [#53522](https://github.com/vllm-project/vllm/pull/53522) | indexer MQA DeepGEMM gate (`pr-53522.diff`) |
| [#53271](https://github.com/vllm-project/vllm/pull/53271) | KV-offload pointer check |
| [#46716](https://github.com/vllm-project/vllm/pull/46716) | CPU SHM all-reduce deadlock |

Carried, not ours. Comment; do not take over:

| PR | patch |
|---|---|
| [#47988](https://github.com/vllm-project/vllm/pull/47988) | `pr-47988.diff` |
| [#53055](https://github.com/vllm-project/vllm/pull/53055) | `pr-53055.diff` (skip on current base; already in) |
| [#52499](https://github.com/vllm-project/vllm/pull/52499) | comment-only |
| [#41834](https://github.com/vllm-project/vllm/pull/41834) | umbrella, needs-rebase, not fetched |

If one of ours merges: drop the matching overlay / `pr-*.diff` on the next
tag port. If GitHub asks for a rebase, rebase. If a reviewer asks for a code
change, make it. If the only remaining gate is `pre-run-check`, record and
wait.

## Round loop

Do these in order. Skip a step only when its check is already current this
round.

1. **Tags.** Newer vLLM or dep tag → port (above) and stop the round at
   re-baseline. Do not mix a port with a kernel A/B.
2. **PRs.** `gh pr view` on the eight PRs above. Rebase / reply / drop
   overlay if merged. Never `pre-run-check`.
3. **One variable.** One recipe under `configs/examples/`, one
   `harness/run-arm.sh` tag, logs in `outputs/driver/`, one
   `scripts/experiment-ledger.py` row, one `docs/UPSTREAM.md` note.
4. **Keep-rule.** Keep only if c6 **and** the sum beat the **standing arm on
   this pin** by more than the larger median spread, both gates, acceptance
   and tokens/step not lower. Standing arm = latest same-pin control in
   `docs/EXPERIMENTS.md` (today `proto2-030-rc2` on `main-030-rc2`). Noise
   center is that control, not a previous-pin first sample.
5. **Same-day `refg`.** Any kept change, and any claim vs anemll, needs a
   fresh anemll run on this pin.

## What to try (and what is closed)

Still open on the current pin:

- Missing `static` MoE on the **live** MXFP4 path. Image FlashInfer has
  `MoEStaticKernel`; dummy zeros compiled `static_m8`; live weights did not
  serve (`fistat` IMA, `fistat2`/`fistat4` hang, `fistat3` profile 12288 vs
  wrapper 48). Re-open only with a new hypothesis, not another wrapper-size
  A/B.
- Real NVFP4 KV writer (today `nvfp4_ds_mla` is the 584 B DSV4 envelope).
- NVFP4 per layer family. This checkpoint's experts are MXFP4 (`w4a8_mx`);
  MoE-NVFP4 is unreachable here.

Closed. Do not re-run unless the pin's kernel or overlay actually changed:

MULTICTA, SPLITK_TURBO, indexer gather, overlay sparse indexer, MG prefill,
chunked prefill (all load-bearing). WO overlay (`nowo`: cost vs einsum).
NUM_SPLITS ladder, H16 on/off, MATERIALIZED, TRITON MoE on SM120, fused
indexer, TWO_LEVEL_FOLD, SWAP_AB, DOWN_SCALE, FLASHINFER AR, InstantTensor
knobs, MHC env, CUTOVER, b12x 1.3.0 bump without isolating cutlass-dsl.

Load-bearing defaults stay on unless an arm turns one off as the single
variable.

## Hard stops

- Workload, prompt, seed, token counts: do not change.
- Do not delete images. Do not disturb llama.cpp.
- One variable per measurement. Two changes is two arms.
- KV floor: hungrier than standing pin at util 0.8389 / max_model_len 65536
  dies at 9.48 GiB need. Do not add a second variable to make it fit.
- Scope: `patches/`, `docker/`, `scripts/`, `configs/`, `harness/`,
  `instruments/`, `docs/`, spark1, spark2.
- Ask owner before any new upstream PR.

## Pointers

| what | where |
|---|---|
| versions | `configs/pin.main-029.env` |
| port recipe | `configs/examples/port-*.sh` |
| overlay scan | `scripts/port_scan.py --with-upstream-patches` |
| ledger | `scripts/experiment-ledger.py` → `docs/EXPERIMENTS.md` |
| long record | `docs/UPSTREAM.md` |
| backports | `patches/upstream/README.md` |
| build plan | `docs/PLAN-MAIN.md` |
