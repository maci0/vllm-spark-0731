# HANDOFF-CURRENT

Current entry point. Supersedes `HANDOVER.md` (written 2026-09-13, pin `v0.30.0rc2`) and
`HANDOFF.md` (historical, append-only, 8634 lines) for the state of the pin and the open work.
The long-form record is `docs/UPSTREAM.md`; the measured ledger is `docs/EXPERIMENTS.md`.

Written 2026-10-12.

## Where the pin stands

| | |
|---|---|
| vLLM tag | `v0.31.1rc0`, ref `e37e51dd246cf421c554e7d6f53e178e3fd29085` (466 commits past the `v0.31.0` final) |
| image | `vllm-spark-0731:main-031-1rc0` sha `f0d20a27dfc5`, 28.8 GB, on spark1 and spark2 |
| phase 1 | `vllm-spark-0731:main-031-1rc0-phase1` sha `764865bd8423` |
| previous pin | `v0.31.0` `db9527a46873`, image `main-031-0` sha `f0fbbc4bc931` |
| overlay scan | `scripts/port_scan.py --with-upstream-patches` on a clean tree: **FAIL=0, applied=43, no-op=12, total=55** |
| recipe | `configs/examples/port-v0.31.1rc0.sh check\|scan\|phase1\|overlay\|copy\|arm` |

**The image is built and self-verified in-build, and nothing has been served on it.** The overlay
stage runs `apply_overlays` then `assert_image` inside the image, and both passed, so the DSV4 model
tree imports and the overlays landed. The France/Paris gate and the protocol re-baseline have **not**
run. `docs/versions.md` says this too; treat every serving number there as the previous pin's.

## What blocks the remaining work

Both nodes are held by other jobs of the same user: `screen train4096` on spark1 and `screen
train2048` on spark2 (`~/adec`, `scratch_dgx.py`, started 2026-10-10 22:53). Measured at block time:
spark1 **28 GiB free**, spark2 **40 GiB free**, while the serve needs about **102 GiB** free per node
at `GPU_MEMORY_UTILIZATION=0.8389` and the model alone loads **79.3 GiB**. No utilization value fits.
Do not reboot and do not kill those jobs.

One related trap, already paid for once: `earlyoom` runs with `-m 8,4 --prefer '^(python3?|vllm|java|node)$'`,
which matches both a build's python and that training job. An unthrottled `02-build-main-029.sh` drove
spark1 to **8.32 % available** and was killed deliberately. Rebuild with `MAX_JOBS=4`, which held
available memory near 48 % for the whole compile. Wall-clock roughly doubles.

## To resume

```bash
# 1. rig must be free: ~100 GiB available on each node
ssh spark1 'free -g | head -2'   # and spark2

# 2. gate + 3-pass re-baseline on the built image
ssh spark1 'cd ~/vllm-spark-0731 && PASSES=3 bash configs/examples/port-v0.31.1rc0.sh arm'
#    drive-median.sh writes outputs/driver/proto2-031-1rc0.median.log and logs gate_france / gate_9x8
```

Then record the result in `docs/UPSTREAM.md`, refresh `docs/versions.md` with the measured rows, and
regenerate the ledger with `python3 scripts/experiment-ledger.py`.

## What moved in this port, and why

Four overlay needles and one diff, all in `patches/`:

- **Indexer page64 buffer** (`patches/apply_overlays.py`). v0.31.1rc0, via upstream #59211, moved the
  `# KV compression` comment and the `compress_ratio` assignment earlier in the builder's `__init__`
  and put a DCP interleave check where they used to be. Both page64 needles were anchored on that
  comment. Added `_INDEXER_PAGE64_BUFFER_RC_{OLD,NEW,SKIP}` beside the `v0.31.0` pair, sharing an
  `RC_TAIL` constant. `replace_one_of` tries each in turn, so both tags still work.
- **`pr-53425.diff`** was regenerated against `v0.31.0` in the previous port and still applies to the
  RC. Keep it regenerated whenever the tag moves: on `v0.31.0` the old copy applied only its import
  hunk, which dropped `MultipleOf` while leaving the method annotated with it, so the *overlay* build
  failed at `assert_image` with `NameError`, not at patch time. A `patch` on a stale diff can half
  apply and still look like a skip.
- **`pr-53055.diff`** returns rc=1 on both new tags. Benign and pre-existing: its `mhc/tilelang.py`
  hunk is already upstream, its `cutlass.py` hunk applies. It leaves a `.rej`/`.orig` in `/opt/vllm`.
- **Dockerfile needed no change this time.** `setup.py` and `tools/use_existing_torch.py` are still
  where `docker/Dockerfile.main` expects. On the `v0.31.0` port the latter had moved (upstream #58572)
  and had to be repointed.

## Provenance facts that are not what the docs used to say

- **Torch is not reproducible from the pin.** `TORCH_REF=release/2.14` and `TORCHVISION_REF=release/0.29`
  are moving branches, so torch drifted from `2.14.0a0+git2b3ec34` (v0.30.0 image) to
  `2.14.1a0+git5c48869` (this image) with no pin change. Pinning both to commits closes it and needs a
  rebuild to verify.
- **DeepEP is not installed in any of our images.** The pin declares `DEEPEP_REPO`/`DEEPEP_REF=v1.2.1`
  but nothing passes or reads them, and `import deep_ep` fails in `main-031-1rc0`, `main-031-0` and
  `main-030-0` alike. Either wire it up or drop the variables.
- In-image `vllm.__version__` is `0.31.1rc1.dev0+ge37e51dd2.d20261011`; upstream stamps `rc1.dev0`
  ahead of the `v0.31.1rc0` tag. CUDA is `13.3` (`V13.3.73`).
- `~/.ssh/config` gained `HostName` blocks for `spark1`/`spark2`. The NVIDIA Sync include that used to
  supply them is gone, and `known_hosts` is keyed by IP, so bare names failed host key verification.
  The live fingerprints were checked against the stored ones before adding them.

## Tracked upstream PRs

All four are open, `MERGEABLE`, and blocked behind the `pre-run-check` label, which is never requested.

| PR | state | notes |
|---|---|---|
| #53425 | head `32042c72e9` | rebased onto main in the v0.31.0 port; the stale `needs-rebase` label cleared itself. All four fixes here are still unimplemented upstream in `v0.31.1rc0`. |
| #53522 | head `fc493b1f90` | rebased onto main `c85616efc6f0` and extended with the decode-site gate (see below). |
| #53271 | head `f8e1fad3da` | maintainer Etelis asked for the repro config and would rather fix the root cause than add validation. Answered with the reachability condition (`blocks_per_chunk > 1`) and offered the core fix as a follow-up. |
| #46716 | head `ddc0860168` | no reviewer activity. |

**Owed on #53522.** michael-gpu's review produced two findings; the first is fixed (decode-path twin
of the gated site, added upstream by #57443, now uses `_should_build_paged_mqa_logits_metadata`), and
the second was answered with source evidence. Two things remain:

1. They are writing an `update_draft_decode_metadata` gate test against our branch and will send it as
   a PR. Fold it in when it arrives.
2. We owe them a **companion guard** PR, scoped to the divergent-predicate path only: the producer is
   env-aware (`is_deep_gemm_supported()` includes `VLLM_USE_DEEP_GEMM`) and the consumer wrapper is
   not, so with the kill switch off on supported hardware the kernel schedules from an unfilled
   buffer. Prefer one shared predicate over two that must agree. Asked which shape they would accept,
   because it touches every caller including GLM5.

## Open items, deliberately not done

- **Seven overlays report "already upstream"** and are deletion candidates: `patch_envs`,
  `patch_moe_backend`, `patch_utils_b12x`, `patch_mxfp4_oracle`, `patch_mxfp4_process_weights`,
  `patch_mhc`, `patch_triton_e8m0_sm12x`. Deleting them changes the build and can only be shown safe
  by a rebuild plus the gate, so it does not belong inside a port.
- **`_usable_b12x_schedule` checks shape, not contents** (`patches/files/sm12x_b12x_kernels.py`). A
  zeroed `(num_sms+1, 2)` buffer passes it, so a declined gate can hand b12x a schedule it treats as
  valid. It does **not** trigger in the production pin (family 120 is in `support_deep_gemm()`,
  `VLLM_USE_DEEP_GEMM` defaults on, indexer group `num_states` is 64), but it is the same failure
  shape the reviewer found upstream, one layer down. A content check would drop into the existing
  trim-and-skip branch; it sits on the decode hot path and needs a throughput run.
- **KV capacity results from earlier rounds** are in `docs/UPSTREAM.md`: 1M context serves
  (`kv1m-probe`, 8.3/9.5/9.6/9.6, gates passing) with `--no-enable-flashinfer-autotune` as the fix for
  the init-time wedge, and a 2M-token pool fits (2,072,844 tokens at 1.98x). Nothing was re-measured
  on the RC pin.

## Measurement traps already paid for

- `port_scan.py` mutates the tree it scans. **Reset before every scan** or the FAIL counts are
  meaningless (`git -C <tree> reset --hard && git clean -fdx`).
- `pgrep -f "<script>.sh"` in an ssh one-liner matches its own shell, so a watcher written that way
  never sees the process exit. Use a bracket pattern (`02-build-main-029[.]sh`).
- uv buffers the vLLM compile, so `#21 Building vllm @ file:///opt/vllm` can sit unchanged for the
  better part of an hour while `cicc`/`ptxas` churn. Check the compiler processes, not the log line.
- The rig's own session drift is ±9 %, larger than any one-variable effect. The keep bar is always the
  reference's c5 median spread (17 to 25 %), which no measured delta here has approached. Best
  same-day pooled result is +2.0 % sum / +1.9 % c6, every ours arm ahead of every ref arm, no keep.
- Never re-run `port_scan` on a tree a previous run already patched; the second scan reports a
  different and wrong FAIL count.
