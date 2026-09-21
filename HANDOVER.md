# HANDOVER

Written 2026-09-13 at the end of a session that took the proto-v0.1.0 work from an untrustworthy
ledger to a measured, closed diagnosis. Read this first; the long-form record is
`docs/UPSTREAM.md` and the accumulated history is `HANDOFF.md`.

**Base moved 2026-09-19**: the pin is now `v0.30.0rc2` (`fa6ff060667f`) and the image tag is
`vllm-spark-0731:main-030-rc2`. `scripts/port_scan.py --with-upstream-patches` reports `FAIL=0`
(applied=43, no-op=12). One commit ahead of `v0.30.0rc1` (`#57570` NIXL). Everything below
that says `main-030-rc1` / `v0.30.0rc1` is the previous serving pin. Earlier bases:
`main-029-proto2` (`proto-v0.2.0` / `f37c550bf635`), `main-029-1rc0`
(`v0.29.1rc0` / `7ee8a6dd0138`), `main-029-proto` (`proto-v0.1.0`).

## Status in one paragraph

The goal, making our arm measurably faster than the anemll reference
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` on the 2x DGX Spark rig, is **not met, and the remaining gap is
now attributed to one thing: the compilation regime.**

The best arm is `proto2-dg` -- plain `proto2` plus `VLLM_B12X_INDEXER_DIRECT_GATHER=1`, now promoted into
`configs/pin.main-029.env` and `scripts/05-serve.sh`. Three runs: 424.3, 422.6, 437.2 on the
c1+c3+c5+c6 sum, against the same-day reference `refg` at 66.4 / 118.4 / 139.0 / 161.0 (sum 484.8). The
honest run-to-run range is **c6 139.6-150.4 and sum 422.6-437.2**, so the gap is **9.8-12.8 % on the sum
and 6.6-13.3 % at c6**, down from 27.3 % / 31.0 % before the indexer-gather win. Gates, acceptance
(50.8-53.6 %) and tokens per step hold.

**The gap has two halves with different causes, and both resolve to the same thing.** Measured in step
time (`harness/step-time-gap.py`, medians of three passes, same day): we are **+13.1 ms at c1, +17.1 at c3,
+13.9 at c5, +20.9 at c6**. Cudagraph capture is worth **-56.5 ms/step at c1** but is inside pass-to-pass
spread at c5/c6 (`proto2-dg-cgnone`), so at c5/c6 there is no launch overhead left to recover and the
~14-21 ms there must be kernel time. At c1/c3 the step is launch-dominated and our breakable-cudagraph
capture is less complete than the reference's compiled full graph. **Both halves are the compiled regime:**
fusion is what removes the remaining per-kernel latency at c1/c3, and fusion is also what the c5/c6 kernel
time would have to come from. The config-difference list is now exhausted -- `disable_custom_all_reduce`
was the last item and it is a measured negative (`proto2-dg-nocar`, sum 429.1 against 437.2, worse at c6).
**See "The gap, attributed (consolidated)" and "Both shape modes fail, in different places, inside the
same machinery" in `docs/UPSTREAM.md` for the full evidence chain.**

**The compile port is closed, and it correctly identified why we cannot follow the reference.** Rounds 55-72
cleared 13 Dynamo graph breaks, built a complete Inductor graph, and executed it; it then failed in two
different places depending only on how shapes were handled -- `assert_size_stride` on the symbolic token dim
with dynamic shapes, `KeyError: 't0'` in `piecewise_backend.py:266 compile_all_ranges` with static ones.
Both are inside vLLM's piecewise machinery. **That vindicates upstream rather than contradicting it:** our
base deliberately puts `DeepseekV4ForCausalLM` in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`
(`config/vllm.py:77`) and forces `CompilationMode.NONE`, and the reference compiles only because its DSv4
stack is structurally different -- no `jit_warmup_triton_helper.py`, a plain `fused_q_kv_rmsnorm`, no lazy
`wo_a` repack, no b12x overlays. **No compile arm ever served and none produced a tok/s number.**

**Re-measured as a tight back-to-back pair (2026-09-18: ours finished 02:18, the reference 02:27)**, so
that "the rig drifted between the two runs" cannot be raised: ours 53.9 / 98.6 / 129.9 / 144.0, sum
**426.4**; reference 65.3 / 110.1 / 138.6 / 159.1, sum **473.1**. **Sum gap -9.9 %, c6 gap -9.5 %,
step-time delta median +17.8 ms (range +11.5 .. +19.2) over a step time that more than doubles.** The
looser same-day pair gave -9.8 % and +15.5 ms, so both the headline gap and the flat per-step delta
reproduce. Both arms came in below their earlier runs, which means the honest ranges are ours
**422.6-437.2** and the reference's **473.1-484.8** -- the gap is what is stable, not either absolute.

**Why, in one line:** the reference runs `CompilationMode.VLLM_COMPILE` and we run `CompilationMode.NONE`
-- both engine logs are quoted in `docs/UPSTREAM.md` ("The gap is attributed") -- and our newer base
selects `NONE` deliberately for `DeepseekV4ForCausalLM` via `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`
(`config/vllm.py:77`), a per-architecture policy the reference's older July build does not have.
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` lifts it, and then the DSv4 b12x stack has to be Dynamo-clean, which it
is not: **12 graph breaks, 10 cleared**, and each break fatal because `fullgraph=True` is unconditional
(`fullgraph=False` dies on `VllmBackend can only be called once`, so there is no supported graph-break
configuration in any of the four `CompilationMode` values). A single enumeration boot now finds them all
at once instead of one per boot (`proto2-enum`, 23 sites logged;
`outputs/driver/one-off/proto2-enum-breaks.log`). **The residue is overlayable, and rounds 59-61 wrongly called it out of scope.** The contract's scope
boundary is about which directories this repo may modify and where it may run, not about which files
inside the image may be bind-mounted -- this repo has overlaid upstream vLLM files since break 5
(`mhc/tilelang.py`). Only *contacting* upstream needs the owner's agreement. `deep_gemm.py:1153`
(`tf32_hc_prenorm_gemm`) is already cleared as of round 62 by merging break 5's overlay with the parked
`patch_mhc_tf32_customop`; the remaining three are:

| site | owner |
|---|---|
| `b12x/attention/_shared/mla/merge.py:84,85,92` | b12x -- third-party, needs the owner's agreement |
| `deep_gemm.py:1153` `tf32_hc_prenorm_gemm` | **cleared round 62** (merged overlay) |
| `multi_stream_utils.py:56` `torch.fx.traceback.annotate` | vLLM |
| `attention.py:852`, `import_utils.py:427` | vLLM |

Every layer family reachable *by configuration* had already been closed as a negative: linear
(`deep_gemm`, worse), attention (FlashInfer, cold and autotune-warm, worse), MoE activation format
(`B12X_MXFP4_BF16`, a wash with 40-50 % spreads).

## The remaining gap is the compilation regime (round 54)

Read this before touching any kernel. The reference is not faster because of a kernel; it is faster
because it compiles and we do not.

| | reference `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | ours `main-029-proto2` |
|---|---|---|
| vLLM | `0.25.2.dev0+g752a3a504` (2026-07-14) | `0.2.1.dev0+gf37c550bf` (2026-09-17) |
| code path | `/usr/local/lib/python3.12/dist-packages/vllm` | `/opt/vllm/vllm` |
| `compilation_config.mode` | `CompilationMode.VLLM_COMPILE` | `CompilationMode.NONE` |
| `cudagraph_capture_sizes` | `[1, 2, 4, 8, 16, 24, 32, 40, 48]` | `[1, 2, 4, 8, 16, 24, 32, 40, 48]` |
| `is_current_stream_capturing` sites | 16 | 42 |
| spec tokens / capture size | 7 / 48 | 7 / 48 |

Logs: `spark1:~/refbase-run.log` (reference), `spark1:~/goal/proto2-dg-engine.log` (ours). The protocol
matches; the mode does not.

**Why ours is `NONE`.** Our base ships a per-architecture policy the reference build predates:
`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (`config/vllm.py:77`) contains `DeepseekV4ForCausalLM`, so
`_maybe_enable_breakable_cudagraph` sets `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and then forces
`CompilationMode.NONE` (`config/vllm.py:796`). The reference's `config/vllm.py` gates only on the env var
(line 1133), with no architecture list. `configs/env.spark.sh:47` follows the upstream default rather
than fighting it -- that is a *correct* reading of upstream, and it is also what costs us the gap.

**What it costs to opt out.** `VLLM_USE_BREAKABLE_CUDAGRAPH=0` works, and then every graph break is fatal:
`fullgraph=True` is hardcoded in `compilation/wrapper.py:150` and `VllmBackend.__call__` asserts it runs
exactly once, so `fullgraph=False` is not merely slower, it is `AssertionError: VllmBackend can only be
called once`. Nine breaks so far, cleared one boot at a time:

| # | site | class | status |
|---|---|---|---|
| 5 | `mhc/tilelang.py` `mhc_pre_broadcast_tilelang` | `hasattr(TritonKernelVariable, ...)` | cleared, custom op |
| 6 | `ops/fused_qk_rmsnorm.py` `fused_q_kv_rmsnorm` | `TritonKernelVariable` | cleared, custom op |
| 7 | `b12x_sparse.py:158` `logger.info_once` | logging method | cleared, `is_compiling()` guard |
| 8 | `sm12x_b12x_kernels.py` `DBG wo_proj` prints | `bool` into graph | cleared: prints deleted, and `torch.cuda.is_current_stream_capturing` folded once in `vllm/__init__.py` |
| 9 | `utils/deep_gemm.py:536` via `ops/o_proj.py:99` | ctypes `_FuncPtr` | cleared, pre-packed before `profile_run` |
| 10 | `ops/o_proj.py:98` `cached[2] != weight.data_ptr()` | data pointer comparison | cleared, guard restructured |
| 11 | `utils/deep_gemm.py:501` `fp8_einsum` | `Attempted to call function marked as skipped` | cleared, `allow_in_graph` |
| 12 | same call | `allow_in_graph` leaf runs under FakeTensors | **open, needs an opaque custom op** |

Break 9 was a one-time weight repack running lazily inside `forward`, hit during `profile_run` (12288 rows
> 256, so `try_b12x_wo_proj` declines and the DeepGEMM fallback runs). It now happens before the first
trace, and the log confirms all 43 layers: `DSv4 wo_a DeepGEMM einsum pre-pack: 43 packed, 0 skipped`.
**`kernel_warmup` is too late for this** -- it runs from `compile_or_warm_up_model`, after
`determine_available_memory` has already compiled inside `profile_run` -- so the hook is at the top of
`determine_available_memory` instead.

**The mechanism is now settled, and it is the important result.** `allow_in_graph` is not a shortcut
anywhere in this stack. It converts `Attempted to call function marked as skipped` into
`Cannot access data pointer of Tensor (FakeTensor)` -- because it makes the call a graph *leaf* and Dynamo
then executes that leaf under FakeTensors to derive metadata. That is the second independent confirmation,
after `patch_mhc_tf32_customop` died the same way on TileLang's `unhashable type: non-nested SymInt`.
**Every third-party kernel on this path needs `direct_register_custom_op` plus an explicit fake impl** --
the pattern that cleared breaks 5 and 6. **Next step: wrap `fp8_einsum` that way.** Its signature is
printed in the failure, so the fake impl is `torch.empty_like(out)` with the tensor pairs flattened and
`mutates_args` declared. How many more entry points follow is open: the round-55 registration touched every
resolved `_impl` global at once, so the fake-tensor failure may simply move to the next one.

Artifacts, all regenerable and all staged on both nodes under `~/fix10/`:
`.scratch/patch_qk_op.py`, `.scratch/patch_capturing_const.py`, `.scratch/patch_nofullgraph.py`,
`.scratch/patch_mhc_op1.py`, `.scratch/arm-cg.sh`.

## The FFN priced by removal, and the batch-independent floor (round 41)

Replacing `self.ffn(x, input_ids)` with `x = x` in the layer forward (bind-mounted `model.py`), with
spec decode off so per-stream tok/s is 1/step-time:

| level | unmodified | FFN removed | FFN cost |
|---|---|---|---|
| c1 | 105.3 ms | **96.2 ms** | +9.1 ms |
| c3 | 128.2 ms | 97.7 ms | +30.5 ms |
| c5 | 145.3 ms | 97.8 ms | +47.5 ms |
| c6 | 148.1 ms | **98.0 ms** | **+50.1 ms** |

Spreads 0.0-0.3 %. **The FFN carries essentially all of the batch dependence** -- with it removed the
step is flat at 96-98 ms from 8 to 48 rows -- and **the reference's entire c1 step (69 ms) is below that
96 ms floor even with the MoE deleted.** So the c1 deficit is not the MoE; it is in ~96 ms of
batch-independent per-layer work. The FFN figure is an upper bound on the MoE, because removing it also
degenerates the residual stream.

The matching attention removal is **not usable** and is recorded as such: it collapses the residual
stream, shortens generations (601/761/75/1230 tokens against an expected 1536), blows spreads to
9.7-15.1 % and implies ~17 ms/step, which is not credible. The MoE router reads what attention produces,
so that arm prices two things at once. **Removal is clean only when the removed stage does not feed
another stage's selector** -- the all-reduce, indexer and FFN skips satisfy that; attention does not.

## The target forward, isolated (round 40)

`DISABLE_DSPARK=1` produces one token per engine step, so per-stream tok/s *is* 1/step-time and the
target model's forward can be measured directly, with no profiler:

| level | no DSpark (1 tok/step) | with DSpark (4.6 tok/step) | DSpark overhead |
|---|---|---|---|
| c1 | **105.3 ms** | 130 ms | +24.7 ms |
| c3 | 128.2 ms | 175 ms | +46.8 ms |
| c5 | 145.3 ms | 223 ms | +77.7 ms |
| c6 | **148.1 ms** | 234 ms | +86.2 ms |

Spreads 0.0-0.6 %, the tightest arm on this rig. Three consequences:

- **The target forward barely moves with batch** -- 105.3 ms at 8 rows, 148.1 ms at 48. That is the
  flat per-step penalty seen directly: 43 layers at 2.4 ms each.
- **The DSpark machinery is +24.7 ms at c1**, so the draft is not the problem; it buys 4.6 tokens for
  130 ms against 105.3 ms for one.
- **The gap is the target forward.** The reference's *whole* c1 step is 69 ms including its own draft;
  our target forward alone at batch 8 is 105.3 ms -- 1.5x the reference's entire step, before our draft.

What is still open is which half of the forward carries it, `attn` (80.2 ms region sum) or `ffn`
(64.2 ms); the region *split* is the unreliable part. Pricing those by removal, as the all-reduce was
priced, is the next measurement.

## The all-reduce is not the gap, and the profiler cannot be trusted here (round 39)

Three separate profiler readings implicated the TP all-reduce: the region marks (`allreduce gpu_sum`
5.7 ms), round 30 (21.4 % of captured CUDA time) and round 37 (54.9 ms). It was priced by removal this
round -- a `skip_allreduce` marker under `VLLM_SKIP_FLAG_DIR` against a bind-mounted
`communication_op.py` -- and the skip is verified to have fired (`grep -c skip_allreduce` returns 1 in
the container; the region counter falls from 5.7 ms to 0.1 ms):

| measurement | control | all-reduce skipped |
|---|---|---|
| `b12x layers: n=43 sum` | **148.0 ms** | **147.7 ms** |
| region `allreduce` gpu_sum | 5.7 ms | 0.1 ms |

**All 87 collectives per step became free and the layer total moved 0.3 ms.** So the collective is
~0.2 % of layer time -- every profiler number for it was an artifact -- and with `attn` + `ffn` =
144.4 of 147.7 ms, those two stages are what the layer total is made of.

Two method conclusions, both earned:

- **The profiler has now failed on this rig three distinct ways**: region marks measure queue-drain
  time (round 26), an `--enforce-eager` capture distorts the regime (round 30), and a graph-mode capture
  sees only the ~6 % of a step outside graph replay (round 38). `capture_torch_profiler`, the documented
  remedy, does not engage here at all -- it is accepted in the config and graphs are captured, but no
  `capture_traces` appear and neither of the two log lines its branch emits is present (round 39).
- **Intervention is the method that works**: a skip A/B was decisive the first time, and the enclosing
  per-layer total is additive where the region marks are not. The same switch extends to `ffn`, which is
  the next thing to price.

## NEW BEST ARM, CONFIRMED (2026-09-17, round 45)

`VLLM_B12X_INDEXER_DIRECT_GATHER=1`. Two identical runs:

| level | `proto2` | run 1 | run 2 | reference |
|---|---|---|---|---|
| c1 | 35.4 | 51.9 | **55.1** | 66.4 |
| c3 | 78.9 | 101.4 | **98.2** | 118.4 |
| c5 | 103.0 | 127.6 | **129.7** | 139.0 |
| c6 | 117.0 | 143.4 | **139.6** | 161.0 |
| sum | 334.3 | 424.3 | **422.6** | 484.8 |

Gates pass in every pass; acceptance 50.8-53.6 % and tokens per step 4.538-4.732, both unchanged from
`proto2`. **+27 % on the sum and +22 % at c6 over the old arm.** The switch is now in
`configs/pin.main-029.env` and the serve forward list, so this is the standing configuration.

**Where the criterion stands, with three runs of the kept configuration** (range is honest run-to-run
variation, not spread within a run):

| level | kept arm | reference | gap |
|---|---|---|---|
| c1 | 51.9 - 55.5 | 66.4 | 16 - 22 % behind |
| c3 | 98.2 - 101.4 | 118.4 | 14 - 17 % behind |
| c5 | 127.6 - 131.7 | 139.0 | 5 - 8 % behind |
| c6 | **139.6 - 150.4** | **161.0** | **6.6 - 13.3 % behind** |
| sum | **422.6 - 437.2** | **484.8** | **9.8 - 12.8 % behind** |

**Not met**, but the gap is now single digits at c5 and c6 and largest at c1 -- the batch-independent
floor.

**Confirmed at the kernel level.** Re-running the round-45 instrument on the kept arm shows `aten::copy_`
falling from **433.03 ms to 22.33 ms** (19x) and the `direct_copy` elementwise kernel dropping out of the
top list, which is exactly the full-cache indexer gather the switch removes. **The new top item is the
MoE at 42.55 % of kernel time** -- and that is the one item where every lever is already spent: our
kernel is 10-13x faster than FlashInfer's head-to-head, the A16 activation format is a wash, and the
tile-shape override is inside the run-to-run range.

**Where it came from.** Not a new idea: `patches/files/sm12x_b12x_kernels.py`'s `_indexer_direct_gather`
docstring already said the default packed-indexer gather "materialises a full-cache copy on every layer
of every decode step (~57 ms/step at 1 row)" and that the direct read "measured 10.1 -> 25.5 tok/s on the
no-spec arm, but draft acceptance drops from ~60 % to ~41-50 %, so it stays off until that is
understood". Round 44's shapes-recorded trace named that copy independently
(`aten::copy_ [17177, 64, 132]`, 1344 calls; `64 x 132 = 8448` matches the packed sidecar width in our own
boot log), and this round measured the switch end to end with DSpark on -- the configuration that had
never been tested. **The acceptance penalty does not reproduce on proto2.** The arm is kept, and
promoting the switch into `configs/pin.main-029.env` is the follow-up.

## Per-gap attribution (2026-09-17, rounds 40-42)

The goal's fallback clause asks for the best measured arm and a written attribution of each remaining
gap: which layer family, which library, which measurement. This is it. **Every number below comes from
an intervention -- an arm, a skip, an A/B, or a kernel head-to-head. None of it is a profiler share**,
because the profiler was shown to fail three different ways on this rig.

### The two arms

| level | `proto2` (ours) | `refg` (reference) |
|---|---|---|
| c1 | 35.4 | 66.4 |
| c3 | 78.9 | 118.4 |
| c5 | 103.0 | 139.0 |
| c6 | **117.0** | **161.0** |
| sum | **334.3** | **484.8** |

Gates pass in every pass on both arms; acceptance ~52 % against 53.7 %; tokens per step 4.571 against
4.733. **Not met**: 27.3 % behind at c6, 31.0 % behind on the sum, against a larger spread of 26.0 %.

### Step-time decomposition, by intervention

**Note (round 45): this decomposition was taken on `proto2`, before the indexer direct-gather switch was
kept. The ~44 ms/step `direct_copy_kernel_cuda` item recorded below as "residual" is exactly what that
switch removes, which is why the new arm is ~27 % faster.**

| component | c1 | c6 | instrument |
|---|---|---|---|
| full step (with DSpark) | 130 ms | 234 ms | the meter |
| target forward, batch = concurrency | **105.3 ms** | **148.1 ms** | `DISABLE_DSPARK=1` |
| ~~DSpark draft + verify~~ | ~~+24.7 ms~~ | ~~+86.2 ms~~ | **withdrawn, round 48** |
| FFN (MoE) | +9.1 ms | +50.1 ms | FFN -> `x = x` (upper bound) |
| TP all-reduce | +0.3 ms | -- | `skip_allreduce` marker |
| indexer | ~10.6 ms | -- | `skip_indexer_all` (round 26) |
| LM head | ~5.3 ms | -- | 324 kernels x median 0.284 ms |
| mHC prenorm GEMM (impl choice) | ~1 ms | ~2 ms | tf32 vs TileLang fallback A/B |
| **residual: attention + norms/glue** | **~79 ms** | **~96 ms** | remainder |

**Correction (round 48): the "DSpark draft + verify" row above is withdrawn.** With speculation off a
step processes `concurrency` rows; with DSpark k=7 it processes `concurrency x 8` (6 vs 48 at c6). So that
difference is the draft's work **plus the target's own scaling from 6 to 48 rows**, and it cannot be
separated inside the protocol, which fixes the levels at 1 3 5 6. The no-DSpark curve is steep at tiny
batch (+8.6 ms/row from 1 to 6 rows) and then flattens, and the DSpark points sit close to where it would
continue -- so the draft looks near-free in step time, and no draft-versus-target split should be quoted.
The **removal** numbers (FFN, all-reduce, indexer, LM head, mHC) are unaffected: each held the
configuration fixed and moved one stage. Also checked: the draft backbone is already graphed
(`FULL_DECODE_ONLY`, `lm_head` eager) in the built image, so there is no launch-overhead lever there.

**The residual is launch overhead, and its size is explained.** With the FFN removed the step is flat at
96-98 ms from 8 to 48 rows, which cannot be memory traffic. The round-37 trace holds 139353 kernel
events and 5194.8 ms of kernel time across ~35 graph replays, i.e. **~3981 launches per step and ~92.6
per layer**; at a conservative 25 us per launch that is ~2.3 ms per layer, ~99 ms per step -- the
residual almost exactly. So the thing to attack is **kernel count per layer**, not a percentage, and
that is what a compiled forward reduces.

### Each remaining gap, with its family, library and closing measurement

**1. The attention half of the target layer -- ~79 ms at c1, the largest residual.**
Family: attention. Library: ours is `B12X_MLA_SPARSE` (b12x). There is no reachable lever here, and that
is measured, not assumed: `ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4` measured **328.1** (cold) and
**320.9** (with its autotune cache populated) against `B12X_MLA_SPARSE`'s **334.3**, so the reference's
own attention is worse on our stack; and the kernel head-to-head has our MLA at **0.224 ms against
`b12x_ref`'s 0.614 ms**. It cannot be priced by removal because removing attention collapses the MoE's
routing (recorded under `proto2-skip-attn`). **Gap unexplained; family identified; no lever found.**

**2. The MoE -- +9.1 ms at c1, +50.1 ms at c6.**
Family: MoE. Library: b12x `B12X_MXFP4_MXFP8`. Closed three ways: our kernel is **10-13x faster** than
FlashInfer's in the head-to-head (10.07 ms against 127.7 ms at 288 routed rows); the activation format
A8 vs A16 measured a **wash** (`proto2-a16`, sum 334.0 against 334.3, spreads 40-50 %); and the policy
ladder that an older b12x generation had is not drop-in (the 0.15.3 arm dies on `ScaledMM`).

**3. DSpark draft + verification -- +24.7 ms at c1, +86.2 ms at c6.**
Ours: our engine's speculative path. It is not shown to be worse than the reference's, and it earns its
keep -- 4.6 tokens for 130 ms against 105.3 ms for one. The reference's *entire* c1 step is 69 ms, which
is **below our 105.3 ms target forward** and below the **96 ms floor with the MoE deleted**.

**4. The TP all-reduce -- 0.3 ms.** Not a gap. Both engines select PYNCCL; b12x's own PCIe transport is
CUDA-IPC based and cannot open a peer handle across hosts.

**5. The linear family.** All four reachable backends tried: `b12x` best, `deep_gemm` (the reference's
own pick) worse and unstable (spreads to 48.6 %), `humming` cannot load this model, `cutlass`/`marlin`
rejected by the rig.

**6. NVFP4 KV.** Goal item 1 was closed in round 25: the reference's `nvfp4_ds_mla` is also a 584 B/token
padded envelope, so there is no KV-quantisation gap to win.

### What is left, stated plainly

The gap is **~105.3 ms of target forward at batch 8 against a reference whose whole step is 69 ms**, and
~79 ms of that is the attention half plus the per-layer glue, which no configuration we can reach has
moved. The remaining lever with a mechanism is torch.compile, and its case rests on an eager-mode
artifact (~13 % of the real step is fusable, not 30 %), behind a Dynamo port of unknown depth in a
third-party library -- five custom ops with fake impls, with a new break behind every one fixed so far.

**Recommendation: end the goal here with this attribution**, as the contract's fallback clause provides.
Continuing is possible; it is no longer the best use of the rig.

## Standing, and the recommendation (2026-09-17, round 37)

Best measured arm, `proto2`, against the same-day reference:

| level | `proto2` | `refg` |
|---|---|---|
| c1 | 35.4 | 66.4 |
| c3 | 78.9 | 118.4 |
| c5 | 103.0 | 139.0 |
| c6 | **117.0** | **161.0** |
| sum | **334.3** | **484.8** |

Gates pass in every pass on both arms; acceptance ~52 % against 53.7 %; tokens per step 4.571 against
4.733. The criterion is not met: 27.3 % behind at c6 and 31.0 % behind on the sum, against a larger
spread of 26.0 %.

The measured attribution of the serving arm is **linear ~44 %, MoE 30 %, all-reduce 8 %, everything else
below 8 %**, and all three are closed by experiment:

| consumer | how it is closed |
|---|---|
| linear ~44 % | all four reachable backends tried: `b12x` best, `deep_gemm` worse and unstable, `humming` cannot load this model, `cutlass`/`marlin` rejected by the rig |
| MoE 30 % | our kernel is 10-13x faster than FlashInfer's in the head-to-head; the activation format was tested and is a wash; the policy-ladder gap belongs to an older b12x generation that is not drop-in |
| all-reduce 8 % | both engines select PYNCCL; b12x's PCIe transport is CUDA-IPC and cannot cross hosts; disabling CUDA graphs (removing our per-layer break) is worse |

The compile path is the only lever with a mechanism left, but **round 30's case for it was an
eager-mode artifact** -- on the real arm the fusable work is ~13 %, not 30 % -- and it is a porting
effort of unknown depth (five custom ops with fake impls; every Dynamo break fixed so far revealed
another).

**Recommendation: take the goal's fallback clause.** The goal is "not reachable" in the sense its own
contract allows for -- the best arm, its numbers, and a per-gap attribution naming the layer family, the
library and the measurement that closed it, are all recorded. Continuing is possible but no longer
obviously wise.

## The completion criterion

Same-rig, same-day protocol, both nodes restarted clean:

    ~/drive-median.sh <tag> 3 512 1 3 5 6

The proto arm's median aggregate tok/s must be higher than the anemll arm's at c6 **and** across the
c1+c3+c5+c6 sum, by more than the larger of the two arms' median spreads at those levels; every pass
in both arms must pass `gate_france` (' Paris...') and `gate_9x8` ('72, 9x9'); and the proto arm's
acceptance rate and tokens-per-step must not be lower.

## Where everything is

Both machines hold the repo at `~/vllm-spark-0731`. This local copy at
`/home/maci/Desktop/vllm-spark-0731` was merged from `spark1` on 2026-09-13 and additionally holds:

| path | what |
|---|---|
| **`docs/EXPERIMENTS.md`** | **the measured record**: every arm, what it changed, its numbers, its gates, its verdict, and the arms that failed. Generated from `outputs/driver/` by `scripts/experiment-ledger.py` |
| `docs/README.md` | the documentation index, and which file answers which question |
| `harness/README.md` | the protocol and run harness, one line per script; `harness/logs/` holds the engine captures and `harness/sections/` the source fragments of the tracker entries |
| `instruments/` | `dcopy_trace.c` + `libcudart.vers` (CUDA launch interposer), `dcopy.txt` (its capture), `fix_indexer_reshape.py`, `ld.so.preload`, `dcopy_selftest.py`, `arch_codegen_probe.{cu,sh}` + `sass_fingerprint.py` (compiles four vLLM-shaped kernels at `sm_120`/`sm_120f`/`sm_121a` and diffs their SASS and runtime) |
| `outputs/README.md` | the artifact map: what is valid, what is superseded, and how the guard works |
| `outputs/driver/` | the raw logs. Flat files are the 15 valid arms; `superseded/` is pre-guard history; `one-off/` is ad-hoc instrument output |
| `docs/UPSTREAM.md` | the chronological log of the whole investigation, 2854 lines. History, not current state |
| `docs/field-notes/nvfp4-serving/` | DeepSeek-V4 serving notes folded in on 2026-09-13 from the predecessor `nvfp4` project |

Note the local copy has the `.git` that the DGX copies lack, so the DGX merge landed as 19 modified
plus 25 untracked paths against commit `c4acb2d`, uncommitted.

## How to run a measurement

Nodes: `spark1` = 192.168.0.211 (head, rank 0), `spark2` = 192.168.0.212 (worker, rank 1), user
`maci`. **Launch the worker first, then the head about 85 s later.** A head launched alone on a
2-node config blocks forever and looks like a broken image.

Candidate arm, on both nodes:

    ssh spark2 "cd ~/vllm-spark-0731 && NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 \
      GPU_MEMORY_UTILIZATION=0.8389 SERVE_EXTRA_ARGS='--async-scheduling' \
      nohup bash scripts/05-serve.sh main-029 > ~/serve-w.log 2>&1 &"
    # wait ~85 s
    NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48 GPU_MEMORY_UTILIZATION=0.8389 \
      SERVE_EXTRA_ARGS='--async-scheduling' bash scripts/05-serve.sh main-029 > ~/serve-h.log 2>&1

Reference arm:

    bash ~/goal/launch-refbase.sh          # sparkrun recipe at ~/goal/ref-base0731.yaml

Then, once healthy:

    ~/drive-median.sh <tag> 3 512 1 3 5 6

`harness/run-arm.sh <tag> "<env assignments>"` already does the rest: it clears `vllm-ds4-0731`
on both nodes, deletes the `/dev/shm` `psm_*`/`nccl-*`/`sem.mp-*` leftovers, queues the worker 85 s
ahead of the head, polls health for up to 450 s, writes the engine log to `~/goal/<tag>-engine.log`,
and ends by running `~/drive-median.sh <tag> 3 512 1 3 5 6`. The standard flags (k=7, capture 48,
util 0.8389, `--async-scheduling`) are baked in, so the second argument is the only thing that
differs between arms.
stagger, the health poll, saving the engine config to `~/goal/<tag>-engine.log`, and the meter.

## Upstream, as of 2026-09-17 (round 51)

All nine tracked vLLM PRs are **OPEN, none merged**; `#53055` and `#47988` and `#53425`/`#53522`/`#53271`/
`#46716`/`#52499` are MERGEABLE, `#47988`/`#50645` UNKNOWN, `#41834` CONFLICTING. DeepGEMM `#417`/`#419`/
`#403` all open. `eugr/spark-vllm-docker#348` open. Ours carry only the bot welcome comment, so there is
no review to answer, and no upstream write was made.

**Watch `#53055`** -- "Guard DeepGEMM in `mhc_pre_broadcast_tilelang` with a torch fallback". It is the
only tracked item that has moved recently (2026-09-16) and it is the upstream fix for the call site that
blocks the compile port: our parked `patch_mhc_tf32_*` overlays exist because their needle predates
`pr-53055.diff`. If it merges, re-evaluate break 4's custom-op work against it instead of carrying a local
overlay.

## Measurement traps, all paid for once already

- **A new host environment variable does not reach the container unless it is forwarded.**
  `scripts/05-serve.sh` builds `docker run` with a hand-written `-e` list of about twenty names plus
  `SERVE_EXTRA_ENV`; anything else exported on the host is silently dropped and the arm runs stock. This
  voided two arms before it was caught: `VLLM_B12X_MOE_FP4_FORCE_A16` was set through the arm's extra
  env, was absent from the list, and `proto2-a16` therefore ran stock `proto2` until it was re-run
  through `SERVE_EXTRA_ENV`. Check with `docker exec vllm-ds4-0731 env | grep <VAR>` before believing any
  env-driven arm. CLI-carried knobs (`MOE_BACKEND`, `LINEAR_BACKEND`, `ATTENTION_BACKEND`,
  `NUM_SPECULATIVE_TOKENS`, `GPU_MEMORY_UTILIZATION`) are unaffected -- `05-serve.sh` passes those as vLLM
  arguments. The variable is now in the list.
- **`B12X_MOE_FORCE_A8` is a dead knob.** It reaches the container, but no code in vLLM or b12x reads it,
  so any arm that moved it was a no-op. This is what `proto2-moea4` actually was.

- **No pre-2026-09-13T03:15 number is valid.** Before the port guard existed, a still-running
  reference container silently answered an entire measurement series, and five "proto" sets were
  the reference. `scripts/05-serve.sh` now refuses to start when port 8000 is held, and
  `drive-median.sh` records the port holder and container in every result log. Read those lines
  before trusting any number.
- **A derivative image must exist on both nodes.** Otherwise spark2's worker cannot start and the
  head hangs, which looks like a configuration bug. `scripts/02-copy-main.sh <tag>`
  (`docker save | ssh spark2 docker load`) is the path; the b12x015 image is 28.9 GB.
- **Do not put a GPU measurement on this rig while the host is compiling.** The MoE cluster sweep on
  2026-09-17 ran concurrently with the proto2 build (`ninja -j 16`, load average 16.0, sixteen `cicc`
  processes at 100 %). Its first rep came back uniformly inflated, 13.121 ms at cap 188 against 11.557
  and 11.761 in the two later reps, a 13 % error on the same configuration; the later reps agreed with
  each other to 0.5 %. That rep had to be discarded. GPU work and a CPU-bound build share this
  unified-memory part, so the build perturbs it.
- **Container environment only arrives through `SERVE_EXTRA_ENV`.** Assigning `VLLM_*` in the
  launching shell reaches the serve script but not the worker. This silently defeated
  `VLLM_PROFILE_DECODE_STEPS` and made three profiling runs measure the default 12 steps, all of
  which are c1.
- **`VLLM_PROFILE_CAPTURE=1` overrides the step limit**: the wrapper routes to `_b12x_capture_step`,
  which prints its own bounded window and ignores `VLLM_PROFILE_DECODE_STEPS`. Profile steady state
  with capture off.
- **128-token passes swing up to 18 %** on this rig, larger than most effects being chased. Use 512
  tokens and at least three passes, and keep a change only if it beats the larger spread.
- **The reference recipe path must not contain the string "sparkrun"**: `spark-launch.sh` runs
  `pkill -9 -f '[s]parkrun'`, which kills the caller. It is staged at `~/goal/ref-base0731.yaml`.
- `scripts/05-serve.sh golden` does not work (DeepGEMM assertion plus a tilelang `cudart_stub`
  symbol error); use `launch-refbase.sh`.
- `scripts/07-stop.sh` stops `vllm-ds4-0731` and drops caches but does not remove leftover
  `sparkrun_*` containers, which must be removed by hand before a candidate arm.
- `GPU_MEMORY_UTILIZATION=0.8389` is needed on the proto base; at 0.8 there is not enough KV for a
  65536 context because CUDA-graph memory profiling now reserves more.

## Measured standing

All guarded, 3 passes at 512 tokens, levels 1 3 5 6, chat, thinking=false, seed 1234, k=7, capture
48, `--async-scheduling` unless stated. This is the summary; the full record, including per-pass
values, spreads, tokens-per-step, acceptance, the container that answered each arm and every failed
arm, is [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

| tag | change | c1 | c3 | c5 | c6 | sum |
|---|---|---|---|---|---|---|
| `refg` | reference image | 63.1 | 112.1 | 140.9 | 156.0 | 472.1 |
| `moe_humming` | `MOE_BACKEND=humming` | 42.0 | 79.3 | 109.1 | 122.8 | **353.2** |
| `proto5` | k=5 | 38.9 | 82.4 | 95.4 | 133.0 | 349.7 |
| `protog` | k=7, base | 38.4 | 80.8 | 105.2 | 122.3 | 346.7 |
| `attnfi` | FlashInfer sparse MLA, target and draft | 37.8 | 78.5 | 104.6 | 119.8 | 340.7 |
| `cgfull` | `CUDAGRAPH_MODE=FULL` | 38.9 | 76.0 | 87.7 | 109.2 | 311.8 |
| `stockops2` | our WO and indexer overlays off | 33.1 | 59.9 | 74.7 | 81.2 | 248.9 |
| `deeplinear` | `LINEAR_BACKEND=auto VLLM_USE_DEEP_GEMM_E8M0=1` | crash | | | | |
| `dglinear`, `lin_cutedsl` | automatic linear selection | numerically dead, gates fail | | | | |

`humming` is the best sum measured but note it is within the c1 spread of `protog` (9.4 % against
9.4 %), so by the criterion's own rule it has not earned its place yet; it needs a longer run.

## The layer profiler was reporting the draft model, and the region table is not summable

**Resolved 2026-09-17, and the answer supersedes the question.** The draft/target mix-up is fixed
(`b12x draft layers: n=3` on its own list). The target breakdown now prints too --
`b12x layers: n=43 sum=148.0ms avg=3.44ms` -- but **only under `--enforce-eager`**, and the reason it
never printed before is structural: a decode step with `FULL_AND_PIECEWISE` CUDA graphs *replays a
captured graph*, so the Python-level layer wrapper never runs and the list is empty at print time. The
module's own docstring already said so; `VLLM_PROFILE_CAPTURE=1` is the existing route for
graph-replayed work. The layers are **flat** -- slowest L2 at 4.61 ms against a 3.44 ms average -- so no
layer family is the gap at this batch.

**The per-region numbers must not be summed, and that supersedes every region percentage in this file.**
With `skip_indexer_all` armed, the `indexer` region goes 52.0 ms -> **0.0 ms** while `ffn` *rises*
65.2 ms -> 86.2 ms and the enclosing layer total falls only 148.0 -> 137.4 ms. The marks record CUDA
events on a stream still draining queued work, with no synchronize between them, so `gpu_sum` is
queue-drain time, not that region's compute. The indexer's real net cost is **~10.6 ms per step (7 %)**,
not 52 ms (35 %). This is a second, independent reason the old "ffn/MoE is 65 %" attribution is
unusable. Only the per-layer total is additive. Artifacts:
`outputs/driver/one-off/proto2-region-control.log` and `...-skip-indexer.log`.

Root-caused 2026-09-17. `HANDOVER.md` used to say "the profiler prints `n=3` layer events, not 43, and
that why is not established". It is established: the profiler was timing the **wrong model**.

`b12x_profile_decode_once` is applied to the DSpark **draft's** `DFlashSpeculator._run_model`
(`patches/apply_overlays.py:287-308`). It sets `_PROFILING_STEP[0] = True`, resets the shared
`_LAYER_EVENTS[0] = []`, runs the draft, and then **prints the layer summary from that list** - the
list the target's 43 `DeepseekV4DecoderLayer.forward` calls write into. So the printed `n=3` is the
draft's three MTP blocks, and the target's 43 events are discarded before the summary is printed. A
second defect in the same function: its docstring says "One-shot timing for the first real DSpark
decode step" but there is no one-shot guard, only the `is_current_stream_capturing()` early return, so
on the eager path it prints on every draft forward, up to `k+1` times per engine step.

**What this invalidates.** The per-layer and per-region shares in this file - `ffn/MoE 1.84 ms of a
2.83 ms layer`, WO `0.60 ms`, MLA `0.20 ms`, and therefore the claim that ffn/MoE is 65 % of a target
layer - cannot be assumed to describe the target model at all. That is a second, independent reason
they did not reconcile with the corrected step basis, and it means the remaining attribution has been
leaning on an unverified number.

**What it does not touch.** The direct kernel measurements: the WO and MLA head-to-heads, the FlashInfer
0.6.15 against 0.7.0 regression, the expert-skipping curve, and the bandwidth figures. None of them
used this profiler.

**Fix written, not yet active.** `patches/files/sm12x_b12x_kernels.py` now keeps the draft's events in
their own list: `_DRAFT_PHASE` is set around the draft's `_run_model`, `b12x_profile_layer` routes each
`(e0, e1)` pair to `_DRAFT_LAYER_EVENTS` or `_LAYER_EVENTS` accordingly, the draft hook no longer resets
or prints the target's list, and `_DECODE_PROFILED` gives it the one-shot behaviour the docstring
already claimed. Its prints are relabelled `b12x draft step` / `b12x draft layers` so the two models are
distinguishable in a log. **Not synced to the rig on purpose** - the proto2 build is running from this
exact `patches/` tree and editing it mid-build would desync the image or force another build. It lands
in the next rebuild, and the check is the profile print itself: the target summary must read `n=43` with
the draft's three reported separately.

## Where the gap is, in step time (2026-09-17)

From the two same-day median logs, with ~4.6 tokens per step on both arms
(`step_ms = tokens_per_step / (agg / concurrency) * 1000`):

| level | ours | reference | ratio | excess |
|---|---|---|---|---|
| c1 | 130 ms | 69 ms | 1.88x | +61 ms |
| c3 | 175 ms | 117 ms | 1.50x | +58 ms |
| c5 | 223 ms | 165 ms | 1.35x | +58 ms |
| c6 | 236 ms | 171 ms | 1.38x | +65 ms |

**The penalty is almost constant in absolute terms across a 1x-to-6x batch range -- about 60 ms per
step.** That is a fixed per-step cost, not a per-token or per-request one. It is spread evenly over all
43 layers (3.44 ms average, slowest 4.61 ms, no hot layer), and the only stage with a measured price is
the indexer at ~10.6 ms of it. Both direct kernel head-to-heads found our own kernels *faster* than the
alternatives -- MoE 10.07 ms against FlashInfer's 127.7 ms, MLA 0.224 ms against `b12x_ref`'s 0.614 ms
-- so this is not a slow kernel in a hot region. It is a constant added to every step.

Two consequences for the next round. First, the search should be for something that runs **once per
step, 43 times, or once per layer**, not for a stage that scales with tokens: candidates are the
per-layer host/launch path, the sparse-MLA and indexer plumbing that runs on every layer regardless of
batch, and the TP all-reduce count (87 calls per step measured). Second, any fix must be judged on
**step time at c1**, where the penalty is proportionally largest and least diluted by the batch, not
only on c6 aggregate.

### The lead, and it was already measured

Three candidates for that fixed cost are now eliminated. The all-reduce transport is identical (both
engines select `['PYNCCL']`). Turning the CUDA graphs off is **worse** at every level -- `proto2-eager`
28.4 / 68.6 / 98.2 / 112.9, sum 308.1, against `proto2` 334.3 -- so our per-layer all-reduce graph break
is not the cost. And `vllm._deepselect_C` is missing in **both** images, so the fast top-k extension
explains nothing.

What is left is a missing kernel. `outputs/driver/one-off/moe-tuning-coverage.log` has held this since
the MoE sweep:

| | ours (`b12x 1.2.6`) | reference (`b12x 0.15.3`) |
|---|---|---|
| decode policies | `dynamic`, `dynamic_w4a8_decode`, `micro` | `dynamic`, `micro`, **`static`** |
| cheap-policy coverage | micro ends at 20 routed rows | **static covers 20 < rows <= 640** |
| 48 routed rows (= c1) | 188 | **149** |
| 144 (= c3) | 188 | ~130-166 |
| 240 (= c5) | 188 | ~130-166 |
| 288 (= c6) | 188 | ~141-175 |

`routed_rows = q_rows * topk`, `q_rows = c * (k+1)`, topk 6, so the protocol's whole range is 48-288
routed rows: inside the reference's `static` band, outside our `micro` band, so we run a flat `dynamic`
at 188 while the reference runs a `static` kernel 7-31 % faster. Where both libraries have a policy the
times are **identical** (107/63/84 at 8/16/20 rows), so this is a missing kernel generation, not a
tuning difference.

**Do not chase that generation.** The `vllm-spark-0731:main-029-proto-b12x015` arm -- the reference's
`b12x 0.15.3` under our stack -- was finally run (its image had only ever existed on spark1) and dies at
worker init with `ValueError: Failed to find a kernel that can implement the ScaledMM linear layer`, with
`LINEAR_BACKEND=auto` as well. Both arms are retired. **The rule here is the current base, the current
`b12x 1.2.6`, and patches on top** -- never a regression to an older library to recover one kernel. The
question is which fast paths the *current* library ships and our stack does not use, and `b12x 1.2.6`
reads 90-plus `B12X_*` flags.

### The strongest unexplored lever came back negative: b12x's all-reduce is intra-host

The 87-per-step all-reduce count made b12x's own `b12x/comm/pcie` transport the best-fit candidate for
the fixed ~61 ms: it is the only thing the profiler counts once per layer, it is batch-independent, and
our overlay already has to break PYNCCL out of the CUDA graph. It was wired in as an overlay on
`tensor_model_parallel_all_reduce` and **it cannot work on this rig**:

| step | result |
|---|---|
| construction, default call | `AttributeError('PCIeOneshotAllReducePool' object has no attribute 'should_allreduce')` |
| construction, `single_channel=False` | `RuntimeError('distributed PCIe oneshot eager use requires an explicit semantic channel_id shared by every rank')` |
| construction, `single_channel=True` | **succeeds**, `algorithm=oneshot` |
| first collective | `RuntimeError('PCIe shared buffer CUDA IPC import open failed: failed to open CUDA IPC handle for peer')` on both ranks |

The runtime is CUDA-IPC based, and CUDA IPC handles are host-local -- rank 0 on spark1 cannot open rank
1's handle on spark2. `b12x/comm` contains exactly one module and its docstring states the scope:
"``pcie``: collectives for consumer PCIe fabrics (no NVLink) -- one-shot and DMA/CE-ring all-reduce",
i.e. several consumer cards in one box. b12x ships no inter-node alternative. **The 87 per-step
all-reduces stay on PYNCCL and the fixed-cost search has to move on-node.** Two library defects were
found on the way and are worth reporting to the owner: the broken `should_allreduce` delegation, and the
undocumented `single_channel=True` requirement for eager use. The patch lived only as a bind-mount and is
not in `apply_overlays.py`.

The `B12X_*` flag surface was surveyed and mostly closes too -- `B12X_W4A8_TINY_DECODE`,
`B12X_FUSED_INDEXER`, `B12X_INDEXER_DIRECT_K` and `B12X_PAGED_MSA` are **on by default**,
`B12X_PAGED_INDEX_SUPERTILE_K` is already 32768, `B12X_TURBO_ATTN` is gated on a paged fp8 path that our
`B12X_MLA_SPARSE` backend does not use, and the W4A16 small-M flags are for a different MoE dtype. What
remains genuinely unset are the MoE tile-shape overrides -- `B12X_MOE_TILE_MN` (`"64x128"` syntax),
`B12X_DYNAMIC_TILE_MN`, `B12X_DYNAMIC_SWAP_AB` -- which are documented benchmarking knobs and are
single-node microbenchmarks rather than full arms. That is the next measurement.

Also corrected: `B12X_REF=3a437ab5...` in `configs/pin.main-029.env` **is** the 1.3.0 release state, so the
image already runs the newest b12x. "ours 1.2.6 against the reference's 0.15.3" is a version-string
comparison, not a newness one.

## The kernel composition, and the one lever left (2026-09-17, round 30)

**Corrected again 2026-09-17 (round 38, itself corrected in round 43): the round-37 table below is
unusable and its "linear ~44 %" reading is withdrawn.** The reason first recorded -- "only ~6 % of a step
appears as individual kernel events" -- is **wrong**. Round 43 counted the same trace: 139353 kernel
events and **5194.8 ms of kernel CUDA time** across ~35 graph replays, i.e. ~3981 launches per step and
**~92.6 per layer**. The kernels are all there. What covers only 675.3 ms is attribution to a *Python
frame*, because most run inside Inductor's `execute_context_*` regions. So the shares below are of the
Python-attributed subset, not of the step -- the conclusion stands, the recorded reason did not. What survives with numbers: the
top row is the **LM head**, unquantized BF16 (`logits_processor._apply_head` ->
`default_unquantized_gemm` -> `F.linear`), 324 kernels, median 0.284 ms, **~5.3 ms per step = ~2.2 % of
the c6 step**. The correct instrument is `capture_torch_profiler: true`, which profiles the graph at
capture time. **Neither existing trace may be used to quote a per-stage share of the step** -- the eager
one distorts the regime, the graph one hides 94 % of it.

**Corrected 2026-09-17 (rounds 36-37): the numbers below are from an `--enforce-eager` run and do not
describe the arm that actually serves.** Re-profiled on the normal graph-mode arm with
`torch_profiler_with_stack: true`, attributing every kernel to its innermost Python frame:

| ms | share | innermost Python frame |
|---|---|---|
| 242.79 | **36.0 %** | `<built-in function linear>` |
| 203.66 | **30.2 %** | `tp_moe_dynamic_launch` (our b12x MoE) |
| 55.65 | 8.2 % | `blockscaled_serialized` (b12x block-scaled linear) |
| 54.94 | 8.1 % | `all_reduce` |
| 44.72 | 6.6 % | `reshape` |
| 27.73 | 4.1 % | `bmm` |
| 6.92 | 1.0 % | `copy_` |
| 0.96 | 0.1 % | `sm12x_b12x_kernels.py(255): sync_packed_indexer_k` (ours) |

Absolute totals are not comparable between the two captures (stack recording inflates them), but the
rank order is, and it says the opposite of the table below: **the pointwise block is ~13 %, not 30 %,
and the linear family at ~44 % is the biggest consumer.** Round 30's "30 % unfused pointwise" was an
eager-mode artifact, and it is what made compilation look like the lever. Our own overlay file is
exonerated -- its two hot sites are 0.96 ms and ~2 ms.


The region marks were proven useless for attribution (they measure queue-drain), so this round used
vLLM's own torch profiler -- `--profiler-config {"profiler":"torch","torch_profiler_dir":...}` with
`POST /start_profile` / `POST /stop_profile`, no rebuild needed. Under `--enforce-eager`, one request:

| ms | % | calls | kernel |
|---|---|---|---|
| 2088.22 | **30.0 %** | 7140 | `at::native::elementwise_kernel<...gpu_kernel_impl_nocast<...direct...` |
| 1490.73 | **21.4 %** | 3196 | `ncclDevKernel_AllReduce_bf16_RING` |
| 1419.60 | 20.4 % | 1564 | `...b12xmoe...siluMoEDynamicKernelSilu...` |
| 285.01 | 4.1 % | 1521 | `nvjet_sm121_tst_mma_112x64x64...` |
| 252.01 | 3.6 % | 3567 | `cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2` |
| 205.05 | 2.9 % | 3042 | `...b12x_libdense_gemmDenseGemmKernel...` |
| 57.88 | 0.8 % | 7752 | `per_token_group_quant_8bit_kernel<BFloat16, Float8_e4m3fn...` |
| 37.53 | 0.5 % | 1353 | `...b12xattention_sharedmlakernelUnifiedDecodeKernel...` |
| 18.55 | 0.3 % | 1462 | `_dsv4_topk_kernel` |

**Pointwise work, the all-reduce and the MoE carry the step; everything this repo has tuned is noise.**
The linear GEMMs together are 10.6 %, the sparse MLA decode kernel 0.5 %, the indexer top-k 0.3 % -- which
is why every family swap measured as a wash or worse.

Two corrections follow. The all-reduce works out to **~0.47 ms per call**, so two per layer is ~0.93 ms
against the ~1.45 ms per layer the flat step penalty implies: the collective is plausibly **most** of the
fixed per-layer cost, and the 5.7 ms the region marks reported for `allreduce` was low by 260x. Round 27's
answer to that -- b12x's `PCIeAllReduce` -- cannot help, being CUDA-IPC based and intra-host only.

And the 30 % pointwise block (7140 calls of one template, plus 7752 quantisation kernels) is what an
**uncompiled** forward looks like. Our model declares `@support_torch_compile` (model.py:1367, via our own
overlay) and then never compiles, because `configs/env.spark.sh:47` forces
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, and `config/vllm.py:786` says:

```python
enabled = is_breakable_cudagraph_enabled()
if enabled:
    self.compilation_config.mode = CompilationMode.NONE
```

All four of this repo's own example recipes set that variable to `0`. With `0`, the config really does
resolve to `CompilationMode.VLLM_COMPILE` (`proto2-break0`), compilation starts, and it stops at break 4
-- `tf32_hc_prenorm_gemm`, a pybind op Dynamo refuses to trace, reproduced identically on proto2. The two
overlays for it are parked from `apply_main` for a documented reason (stale needle after `pr-53055.diff`),
and their `torch.compiler.disable` route is one this torch rejects anyway.

**The compile path is the only remaining lever with a mechanism behind it, and break 4 is now
cleared (round 31).** `direct_register_custom_op` for `tf32_hc_prenorm_gemm` -- all **three** call sites,
plus hoisting the **three** local import blocks, because a local `from ... import` is itself a graph
break -- removes that error from the traceback entirely and compilation advances. It is
`patches/apply_overlays.py --only mhc-tf32-customop`, tested and idempotent, parked from `apply_main`
like its siblings. The `@torch.compiler.disable` route those siblings use is rejected outright by this
torch.

Compilation then dies at **break 5, TileLang**: `_MHC_PRE_BIG_FUSE_TILELANG_KERNEL` ->
`tilelang/jit/__init__.py:527 __call__` -> `_infer_jit_mode` -> a TVM source introspection
(`has_internal_prim_func` -> `inspect.getsource`) that lives in a Dynamo skip directory. The guard at
`jit/__init__.py:523` (`if self.mode == "auto"`) looked like the fix, and **two candidate fixes are now
ruled out by measurement**:

- **Pre-resolving the mode at import is impossible.** `VllmTileLangJitKernel.kernel` is a plain
  *function* (`has .mode: False`), and the object that owns `mode` is produced by calling it:
  `self.launch(self.kernel(*kernel_args), ...)`. It exists only at call time, inside the traced frame.
- **Warmup does not cover this kernel.** `launch` sends warmup through `compile_tilelang(...)` ->
  `.compile()`, never `__call__`. Binding a mode resolution into `launch` just moves the same call
  inside Dynamo, where the traceback names our own line and the `Unsupported` is not catchable from user
  code.

**The un-patching route is also closed, and closed by the standard library (round 33).**
`inspect.getfile` arrives at TileLang patched twice -- `torch.package.package_importer` patches it and
keeps the original as `_orig_getfile`, then bundled TVM patches it again and stashes *torch's* version --
so restoring from TVM's stash just swaps one patch for the other. Reloading `inspect` does clear the
Dynamo skip (0 occurrences) but re-executes the module and breaks pydantic mid-import
(`TypeError: 'BeforeValidator' object is not iterable`). And with the chain removed by the milder route,
Dynamo simply moves on to the next frame:

```
qualname: checkcache
skip reason: file is under skip directory (/usr/lib/python3.12/linecache.py
```

`inspect.getsource` bottoms out in `linecache`, and `/usr/lib/python3.12/` is **itself** in torch's skip
list. So no repair of `inspect.getfile` can work -- Dynamo will not trace source introspection at all.

**And the third route is closed too (round 34), so TileLang must be kept out of Dynamo entirely.**
`_is_lazy_style`'s fast path is `has_internal_prim_func(self.orig_func)`, which scans the AST; guarding
it to return False under compilation is a safe substitution (it only skips the fast path -- the
authoritative test is the body call that follows, and `builder.py:1525` is its only caller). It works:
the skip error drops to 0. And that is what proves the point, because Dynamo then traces the kernel
body:

```
torch._dynamo.exc.Unsupported: Unsupported method call
  Explanation: Dynamo does not know how to trace method `__new__` of class `type`
  Developer debug context: call_method UserDefinedClassVariable(<class 'tvm_ffi.core.CObject'>) __new__
    [UserDefinedClassVariable(<class 'tvm.tirx.expr.Var'>), 'num_tokens', ...]
```

TVM IR construction, which Dynamo refuses. So both paths through `_is_lazy_style` are untraceable --
source scan (`linecache`, stdlib skip directory) and body call (TVM IR) -- and there is no third. The
guard was removed again: it clears one error only to land on an equally hard one.

What is left is to **keep Dynamo out of TileLang entirely**, and `allow_in_graph` is not a shortcut for
it. `model.py` reaches TileLang through five entry points, so one `torch._dynamo.allow_in_graph(...)` per
function looked like the cheap fix; it applies cleanly and does clear the skip error, but it still *runs*
the function -- on FakeTensors, to infer outputs -- and TileLang's kernel cache then dies with
`TypeError('unhashable type: non-nested SymInt')`. A real `direct_register_custom_op(..., fake_impl=...)`
is required precisely because Dynamo calls the **fake impl** and never runs the kernel while tracing.

So break 5's remaining work is fully specified and is a porting effort, not a config change: for each of
the five mhc entry points (`mhc_pre_broadcast_tilelang`, `mhc_pre_tilelang`,
`mhc_fused_post_pre_tilelang`, `mhc_post_tilelang`, `hc_head_fused_kernel_tilelang`) -- signature, fake
shapes, registration, and rewriting `model.py`'s five call sites through `torch.ops.vllm.*`. Rounds 31-35
cleared break 4 and mapped break 5 completely, but every fix so far has revealed another break, and the
repo stalled at this same point once before; the goal's fallback clause (best arm plus per-gap
attribution) is the alternative if this does not converge.

**Also this round:** `VLLM_USE_AOT_COMPILE=0` measured 318.9 against 334.3, but the mode stayed NONE, so
that number is the cost of losing the AOT artifacts, not of compiling -- keep AOT on. And a dead-knob
audit found `VLLM_USE_B12X_MOE`, `VLLM_USE_B12X_MHC`, `B12X_MLA_SM120_UNIFIED`,
`B12X_MOE_FORCE_A8` and `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` are all forwarded and read by nothing.

## The leading attribution: we stream ~31 % more expert bytes per layer than the reference

### Downgrade: the 31 % is a step difference, not yet a proven expert-byte count

Both mechanisms that could be tested directly came back negative, so the expert-byte decomposition needs
to be held more loosely than when it was written.

- The decomposition assumed every layer is MoE and that the step is the target pass alone. Its
  per-layer budget is therefore an upper bound, and the 28 % / 37 % fractions move once the draft
  model, attention and collectives are given their share.
- **Routing is weakened as the last candidate.** Both arms run the same checkpoint, so they run the
  same router; for the same token stream they should select nearly the same experts, and a difference
  would need the two engines' numerics to diverge enough to change expert selection.
- What is solid: the step difference itself (176.5 ms against 231.8 ms at c6, a 6x-corrected basis), and
  that the kernel skips and streams efficiently (88 % of the bandwith bound) with EP = 1 on both sides.

**So read 31 % as an undecomposed difference in the step, and settle it by re-running the region
profiler on the corrected basis** rather than by inference. That needs a serve, which needs the rig.

**Prepared for the routing measurement anyway**, since it is cheap and self-contained:
`--enable-return-routed-experts` is a real flag (`vllm/engine/arg_utils.py:900-903`, backed by
`ModelConfig.enable_return_routed_experts`, default False) and it adds a `routed_experts` field to the
chat response, a base64-encoded numpy array (`vllm/entrypoints/openai/chat_completion/protocol.py:119-124`
documents the decode as `np.load(io.BytesIO(base64.b64decode(s)))`). `harness/capture_routing.py` sends
one request and reports distinct experts touched, load distribution and singleton count. Both images
support the flag, **but the reference's configuration is read-only by contract**, so the symmetric
comparison is not available and any routing number would be ours alone.

### And so is the sharding half: both arms run EP = 1

vLLM's own `FusedMoEParallelConfig.make` docstring, identical in both images, gives **EP = {1, 0} per
device** when TP = 2, DP = 1 and EP = False. Neither arm passes `--enable-expert-parallel` (it appears
nowhere in `scripts/`, `configs/` or `patches/`, and the comparator recipe does not have it). So both
shard inside each expert along TP rather than splitting the expert set, and every rank holds the same
expert set. **Expert parallelism is not the ~31 %.** Both candidate mechanisms for the expert-byte gap
are now closed, which leaves **routing distribution** - which experts the router picks and how many
rows land per step - neither of which is measured today.

That one is cheap to get: `enable_return_routed_experts` is **False** in this arm, and flipping it is a
config change rather than a rebuild. It reports the routed expert ids per request, so the
experts-touched-per-rank becomes a measured number instead of an inferred one, and it is a goal item 3
measurement needing no reference contact.

Useful configuration facts recorded with it, from `harness/logs/attnfi-engine.log`: the MoE backend
resolves to `B12X_MXFP4_MXFP8` and then `B12xExperts`; the KV dtype really does resolve to the padded
`nvfp4_ds_mla` envelope; the model quantisation is `deepseek_v4_fp8`; graph capture sizes are
[1, 2, 4, 8, 16, 24, 32, 40, 48]; and `enable_return_routed_experts` is off.

### The skipping half of that attribution is closed

Tested 2026-09-17 with weights, shapes and 288 routed rows fixed and only the number of distinct
experts targeted varying: 174 experts 13.29 ms, 116 at 9.67, 61 at 6.13, 32 at 4.19, 8 at 2.72 - a
straight line of 0.0637 ms per touched expert plus ~2.21 ms fixed. One expert is 12.6 MB of fp4
weights, 0.0564 ms at the measured 223 GB/s, so **the kernel streams touched experts at 88 % of the
bound and does skip the rest**. So the ~31 % expert-byte gap is not a skipping failure and not kernel
inefficiency: it has to be **how many experts each rank touches**, i.e. expert parallelism or routing
distribution, which is goal item 3 and not item 4. Limits: one layer, synthetic weights, near-uniform
routing where the served routing skews.



Established 2026-09-17 from measured bandwidth plus the corrected step times. This supersedes the
region-share attribution below, which rested on the wrong basis.

Streaming bandwidth measured three ways: device copy 223.2 GB/s, GEMV (a weight matrix streamed once,
which is the MoE's pattern) 201.0 GB/s at 128 MiB and 198.9 GB/s at 512 MiB. So ~200-223 GB/s is
achievable and the 273 GB/s on the spec sheet is not.

Expert traffic is 3.22 GB per layer, 138.5 GB over 43 layers. Against the corrected c6 steps:

| arm | c6 step ms | ms/layer | GB streamed per rank at 223 GB/s | fraction of a full expert layer |
|---|---|---|---|---|
| reference `refg` | 176.5 | 4.10 | 0.916 | **28.4 %** |
| ours `protog` | 231.8 | 5.39 | 1.202 | 37.3 % |

**The reference's entire step is accounted for by expert streaming.** With experts sharded across the
two ranks and empty-expert skipping, the predicted fraction touched is 1 - exp(-144/128) = 67.5 % of
128 local experts, i.e. 33.6 % of the full 256, close to the 28.4 % the timing implies.

**And the same arithmetic prices the gap: we move ~31 % more expert bytes per layer for the same
work**, which is the whole 55.3 ms difference. The kernel is not the suspect - at 128 local experts our
MoE measured 7.03 ms against FlashInfer 0.6.15's 7.67 ms. The candidates are a different expert
sharding (more experts read per rank) or a failure to skip experts that receive no tokens. Both are
testable without touching the reference, and a fix belongs to goal item 3.

Caveat: the fraction touched is predicted from assumed uniform routing, and real routing skews, so
28 % and 31 % are the right order rather than exact.

## CORRECTION: the step-time basis in this file is 6x too small

Found 2026-09-17. The paragraph below says "Step time is 38.65 ms against 29.4 ms at c6". Those are
`drafts_per_req / wall`, and `drafts_per_req` is counted **per sequence** while the forward pass serves
all six, so both figures are too small by exactly the concurrency: 29.4 x 6 = 176.4 and 38.65 x 6 =
231.9. From the meter logs in `outputs/driver/`:

| level | tag | wall s | tokens/step | steps per sequence | **step ms** | quoted |
|---|---|---|---|---|---|---|
| c6 | `refg` | 19.33 | 4.676 | 109.5 | **176.5** | 29.4 |
| c6 | `protog` | 25.89 | 4.585 | 111.7 | **231.8** | 38.65 |
| c1 | `refg` | 7.79 | 4.923 | 104.0 | **74.9** | 13.35 |

Two checks agree: the factor is exactly 6, and the corrected ratio 176.5/231.8 = 0.761 matches the
measured throughput ratio 118.64/158.93 = 0.747, whereas the quoted pair gives its inverse.

**So every per-layer budget in this file and in `docs/knowledge/05-performance.md` was computed from a
basis that is 6x too small and must be redone.** That includes "ffn/MoE 1.84 ms of a 2.83 ms layer",
the WO 0.60 ms and MLA 0.20 ms shares, and the statement that the target forward is 82-85 % of the
step. The corrected step sits at the same order as the machine's expert-weight traffic (>=320 ms at
TP=2 once empty experts are accounted for), so **the step is memory-bound on expert weights and MoE is
essentially the whole game** - which makes the MoE leads the important ones and the small-kernel
budgets suspect. The kernel comparisons themselves (WO 0.4 ms, MLA 0.22 ms against 0.61 ms) are direct
measurements and are unaffected.

## The diagnosis, and what is ruled out

The gap is decode step time, not speculative decode: at c6 we do 4.585 tokens per step against 4.676
and acceptance is level, so adopting the reference's step rate alone would give 155.8 tok/s against
its 156.0. Step time is 38.65 ms against 29.4 ms at c6, and about 114 ms against 74 ms at c1.

Step attribution: the **target forward is 82 to 85 %** of the step at both c1 and c6; the draft plus
sampler is 21 to 22 ms at c6 and the sampler alone 2.5 to 3.9 ms. Within a target layer, 2.83 ms
total, the regions are ffn/MoE 1.84 ms (65 %), attn 0.94, wo 0.60, mla 0.20, allreduce 0.20, flat
across layers. TP collectives are negligible, so the interconnect is not the lever.

Ruled out, each by a guarded arm: attention implementation and attention autotuning, MoE vendor, the
fp8 linear path (alternatives are either unavailable or numerically broken), FULL cudagraph mode,
the spec token count, and every one of our own overlay knobs (turning those off costs 34 %).

Stack difference against the reference: b12x 1.2.6 against 0.15.3, flashinfer 0.7.0 against 0.6.15,
tilelang 0.1.14 against 0.1.9, humming 0.1.13 against 0.1.10, quack 0.6.5 against 0.5.0,
tokenspeed_mla 0.2.8 against 0.1.2, NCCL 2.31.2 against 2.30.7, torch 2.14 built from source against
2.11.0+cu130, CUDA 13.3.1 against 13.0, vLLM `0.1.1.dev0+g69db1c26b` against
`0.25.2.dev0+g752a3a504`. Both images run uncompiled: the reference's config says `VLLM_COMPILE` but
its own log shows ``torch.compile is turned on, but the model ... does not support it``. Both use
PYNCCL and prefix caching.

## Next step, in order

1. **DONE 2026-09-17: the `b12x_ref` WO head-to-head was measured, and it is not a lever.** Both
   generations coexist in `vllm-spark-0731:main-029-proto-b12xref`, which keeps our `b12x` and adds
   the reference's 0.15.3 as `b12x_ref`. `harness/bench_b12x_wo.py` drove both native WO-A/WO-B
   kernels at the DSV4-Flash shapes (hidden 4096, groups 8, group_width 4096, rank 1024) with the
   same weights and the same input; the two agree to 1e-6 relative at 1 token and exactly at 8 and
   64, so the timings are comparable. Median of 50 CUDA-event timings, ms:

   | tokens | `b12x` (ours) | `b12x_ref` | delta |
   |---|---|---|---|
   | 1 | 0.374 | 0.331 | ref -0.042 |
   | 8 | 0.381 | 0.344 | ref -0.037 |
   | 16 | 0.381 | 0.396 | ours -0.015 |
   | 32 | 0.410 | 0.426 | ours -0.015 |
   | 48 | 0.405 | 0.446 | ours -0.042 |
   | 64 | 0.417 | 0.479 | ours -0.063 |

   The older generation wins only at or below 8 tokens, by at most 0.050 ms; a c6 decode step runs
   about 48 rows, where ours is 0.042 ms ahead and at 64 rows 0.063 ms ahead. Best case for a port is
   0.05 ms against a 38.65 ms c6 step, 0.13 %. **Do not port the WO call site.** Caveat recorded with
   the number: the 0.60 ms the profiler attributes to the WO region is the live path (fused
   inverse-RoPE quant, dequant, grouped `bmm`, `wo_b` linear), not this native kernel, so this closes
   the generation question and does not show the WO region itself is optimal.

   **The MLA head-to-head is now measured too, and it points the other way.** `harness/bench_b12x_mla.py`
   drives both `compressed_mla_decode_forward` entry points on the same DSV4-Flash decode contract
   (48 query rows, 32 local q heads at TP=2, head_dim 512, SWA window 128, indexed topk 512 pages,
   584 B/token page), with inputs built by the reference package's own
   `pack_compressed_mla_kv_cache_reference` and the pure-torch `compressed_sparse_mla_reference` as
   ground truth. Three runs, median of 30 CUDA-event timings each:

   | library | run 1 | run 2 | run 3 | median | vs pure-torch reference |
   |---|---|---|---|---|---|
   | `b12x` (ours) | 0.2236 | 0.2238 | 0.2257 | **0.224** | 0.50 % |
   | `b12x_ref` | 0.6138 | 0.6105 | 0.6319 | 0.614 | 0.50 % |

   Both sit 0.50 % from the pure-torch reference and agree with each other to 0.10 %, so they compute
   the same thing; ours is **2.7x faster**. Nothing to port, and the older generation is now closed on
   both surfaces that have been measured (WO and MLA).

   **That is the useful part: it relocates the gap.** If the reference's attention kernel is 2.7x
   slower than ours, attention is not how its engine earns its step time, so the missing milliseconds
   have to be in the region the profiler already prices highest: ffn/MoE at 1.84 ms of the 2.83 ms
   target layer, 65 %, flat across layers. The next attribution work belongs there, not in attention.
   Caveat carried with the number: this compares each stack's public decode entry point, and
   `compressed_mla_decode_forward` splits into chunks and merges where ours is a single fused call, so
   part of the 2.7x is entry-point design rather than raw kernel throughput.

   The indexer head-to-head is still open.

   **The MoE region now has a leading attribution, and it is structural.** The profiler prices 65 % of
   a target layer in ffn/MoE (1.84 ms of 2.83 ms), and the MLA result above rules attention out as the
   reference's source of speed, so the gap has to live in ffn/MoE. `harness/probe_moe_tuning.py`
   compares the two generations' decode MoE policy coverage:

   | routed rows | ours micro | ours dynamic | ref micro | ref **static** | ref dynamic |
   |---|---|---|---|---|---|
   | 20 | 84 | 188 | 84 | 148 | 188 |
   | 48 | - | 188 | - | 149 | 188 |
   | 144 | - | 188 | - | 130 | 188 |
   | 240 | - | 188 | - | 141 | 188 |
   | 288 | - | 188 | - | 175 | 188 |
   | 640 | - | 188 | - | 188 | 188 |
   | 1024 | - | 147 | - | - | 147 |

   **We have no `static` MoE backend at all** — no `static` module in `b12x.moe.fused_moe` and no
   `decode/static` policy in the tuning registry — while the reference ships `MoEStaticKernel` plus a
   generated ladder whose cap is tuned per row count. Every level of the protocol lands in that band
   (256 routed experts, 6 per token, `q_rows = c * 8`: c1 = 48 rows, c3 = 144, c5 = 240, c6 = 288), so
   we run the `dynamic` kernel with a flat cap of 188 at all four while the reference runs a different
   kernel with a tuned cap.

   **A first attempt to convert it into milliseconds came back inconclusive, and is recorded as
   inconclusive.** `harness/bench_moe_clusters.py` drives our own MoE at the DSV4-Flash decode shape
   (48 tokens x topk 6 = 288 routed rows, 256 experts, hidden 4096, intermediate 2048) with weights
   packed in the real contract, and sweeps only `max_active_clusters`: 188 (our flat default), 175,
   149, 141, 130, 96. With the work held identical the sweep is **flat within 1 %** — 10.48 / 10.39 /
   10.47 / 10.39 / 10.48 / 10.38 ms — so a cap-only change is not the lever. An earlier run showed
   -5.1 % at cap 149; that was input re-randomisation between caps in my own harness, not the cap,
   and it disappeared once `make_case` was seeded.

   **The red flag is resolved: it was my bug, not the kernel's.** `harness/diag_moe_divergence.py`
   showed repeated `run` calls at a fixed cap perfectly stable, two rebuilds at the same cap
   disagreeing, inputs bit-identical, and each binding still reading its own buffer. The cause is that
   `fused_moe.prepare_weights` repacks the packed weight tensors **in place**
   (`_logical_weight_to_w4a8_rp_inplace`, `_e8m0_scale_to_w4a8_sfb_inplace`), so my loop feeding the
   same `PackedWeights` into a second build handed it already-repacked data. Only the first build in a
   process was correct. With a fresh weight copy per cap the output sums agree to **0.008 %** across all
   six caps, which is FP32 atomic-accumulation order — the kernel is right.

   On clean inputs a real effect does show up, and it is in our favour to chase:

   | cap | median ms | vs 188 | output abs-sum |
   |---|---|---|---|
   | 188 (our flat default) | 10.8387 | 1.000 | 516334.9 |
   | 175 | 10.3170 | 0.952 | 516318.8 |
   | 149 | 10.8124 | 0.998 | 516325.5 |
   | 141 | 10.2148 | 0.942 | 516328.1 |
   | 130 | 10.7469 | 0.992 | 516358.5 |
   | 96 | 10.1690 | 0.938 | 516327.4 |

   Our default 188 is the slowest cell; the caps the reference tunes for this band are 4.8-5.8 % faster,
   and the ordering reproduced across two runs. **It is not earned**: the sweep is non-monotonic, and
   cap 188 itself measured 10.4787 in one run and 10.8387 in the next, a 3.4 % swing on the same
   configuration — the same order as the 5 % effect, so by this repo's own rule it has not beaten the
   spread. It also cannot be projected onto the protocol yet: 10.4 ms for one MoE with 288 routed rows
   is about an order of magnitude above what a 43-layer step of 38.65 ms can contain, so this
   single-GPU 256-expert execution plan is not the path the engine runs.

   Next MoE step, in order: repeat the sweep to bound run-to-run noise, then reshape the probe to the
   served configuration (TP sharding, real routing skew, tuned plan) before calling anything a win.

2. **The base move to `proto-v0.2.0` was ported but never reached the rig.** `VLLM_REF` in
   `configs/pin.main-029.env` is `f37c550bf635` and the pin names `vllm-spark-0731:main-029-proto2`,
   but spark1's repo copy was still the pre-port `main-029-proto` state (`VLLM_REF=69db1c26b4`, no
   `scripts/port_scan.py`). The desktop copy was synced to spark1 on 2026-09-17 and the phase-1 build
   of `main-029-proto2` started there (`~/proto2-build.log`, `MAX_JOBS=16`, host idle). **No protocol
   measurement on the new base has been run yet**; every number in `docs/EXPERIMENTS.md` is still
   proto-v0.1.0-era and must be re-baselined on the new image before any A/B is believed.
3. If more granular target attribution is needed, fix the layer sample: the profiler prints `n=3`
   layer events, not 43, and that why is not established.

## The proto2 build failed once, on a missing package

Recorded 2026-09-17 so it is not paid for twice. Step #21 (the vLLM wheel) died after 1392 s; the real
error was `deep_jit/utils/exception.hpp:3:10: fatal error: elfutils/libdwfl.h: No such file or
directory`. **This repo had already recorded the requirement and never applied it**: the 2026-09-16
DeepGEMM #447 entry in `docs/UPSTREAM.md` says the build needs "`libdw-dev` (DeepJIT's `exception.hpp`
dlopens `libdw.so.1`)". The proto2 port moved DeepGEMM to the upstream fork with `deep_jit` as a
submodule, which is the code path that needs it, and `docker/Dockerfile.main` was never updated. Fixed
by adding `libdw-dev` to the base apt layer; it pulls `libdw1`, which the `dlopen` needs at runtime
too. The failed log is preserved at spark1:`~/proto2-build-failed.log`, and BuildKit caching means the
restart resumes at #21 rather than rebuilding the base.

### Phase 1 is not the arm image: phase 2 applies the overlays

Corrected 2026-09-17 after serving the phase-1 image failed with `Unknown attention backend:
'B12X_MLA_SPARSE'`. `docker/Dockerfile.main` is **phase 1 only** and says so in its header; it never runs
`apply_overlays.py`, and the image has no `vllm/utils/sm12x_b12x_kernels.py` at all. The overlays are a
second build: `bash scripts/03-apply-main-overlays-029.sh <tag> <phase1-tag>`. **The runbook below has
been corrected** - the arm image is the phase-2 one, and the pin's `IMAGE` must name it.

Phase 2 then failed four times on four real defects, all now fixed:

1. `replace_once` matched a needle's leading indentation as the *tail* of a deeper line, writing
   `compressor(...)` at the wrong indent and killing the build on an `IndentationError`. It now matches
   only at a line boundary, so this class of bug fails loudly instead of corrupting silently - and
   `port_scan.py` had counted it as applied, which is why the port looked clean.
2. The compressor needle could not pin its own indentation; it is now anchored on the guarding
   `if not skip_compressor:` line.
3. The same bug in the lambda variant (12 spaces where the base has 16).
4. `assert_image.py` asserted `is_deep_gemm_supported` is *present* in an mHC forward and that
   `_tilelang_hc_prenorm_gemm` exists. The overlay's job is to remove the first, and the second no
   longer exists anywhere in the module, so both are now the real invariants: `_USE_DEEP_GEMM` present,
   no bare `is_deep_gemm_supported()` left.

Verified in the phase-1 image with the fixed patches: `apply_rc=0`, `assert_rc=0`, `image OK (main)`.
**spark2's repo was also stale** (pin still `main-029-proto`), so the worker had run the old base while
the head ran proto2; both nodes are synced now.

## The proto2 build is done and verified (2026-09-17)

`vllm-spark-0731:main-029-proto2-phase1`, ID `b5f2b5fdbf8e`, 29.2 GB, is the phase-1 base with zero
error lines. Verified from inside the image: **`vllm 0.2.1.dev0+gf37c550bf.d20260917`**, i.e. the
pinned `proto-v0.2.0` commit. Phase 1 is not the arm image; phase 2 applies the overlays.

`vllm-spark-0731:main-029-proto2`, ID `965ff6b4c681` after the fix described in the section below,
29.2 GB, is the overlaid arm image. Rebuild it with the documented fresh path, which is the one that
works:

    scripts/03-apply-main-overlays-029.sh \
      vllm-spark-0731:main-029-proto2 vllm-spark-0731:main-029-proto2-phase1

**The incremental form of that command does not work.** `03-apply-main-overlays-029.sh <TAG>` with no
base rebuilds on top of `TAG` itself, and that fails on `patch_o_proj_einsum_e8m0` with
`missing needle in .../ops/o_proj.py`: the overlay is not idempotent against its own output. Always
pass the phase-1 image.

## The proto2 serve blocker: a stale overlay the new DeepGEMM rejects

`main-029-proto2` died during `_initialize_kv_caches` with

    Assertion error (.../deepgemm-src/csrc/utils/layout.hpp:113): sf.size(-2) == ceil_div(mn, gran_mn)

and `Worker proc VllmWorker-0 died unexpectedly`. The stack is `attention.py:532` ->
`nvidia/flashinfer_sparse.py:585 _o_proj` -> `nvidia/ops/o_proj.py:116 deep_gemm_fp8_o_proj` ->
`utils/deep_gemm.py:501 fp8_einsum`, i.e. the FlashInfer sparse layer's O projection, not our b12x one.

**Cause: our own `patch_einsum_sm12x_recipe`, applied by `apply_main`.** It makes SM12x return recipe
`(1, 128, 128)` with `tma_aligned_scales=False` instead of upstream's `(1, 1, block_size)` with
`tma_aligned_scales=True`. Upstream's value is the packed UE8M0 layout the SM120 einsum kernel reads;
the override feeds it SM90-style fp32 scales, and DeepGEMM refuses the call.

We had already measured this and written it down. `docs/UPSTREAM.md` (DeepGEMM #447) records that the
DSv4 o_proj shape is **accepted at T=10 and T=256 with packed ue8m0 scales at `recipe=(1,1,128)`**, and
that the `layout.hpp` rejection at `recipe=(1,128,128)` with pre-packed per-row scales "is correct by
design, not a defect". The overlay's own recorded reason was that the Python `fp8_einsum` fallback
upcast packed int32 -- but that fallback is not applied by `apply_main` any more, and
`utils/deep_gemm.py` binds `_fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)`, DeepGEMM's real
kernel.

**Fixed** in `patches/apply_overlays.py`: the call in `apply_main` is commented out with the reason
inline, and `patch_einsum_sm12x_recipe`'s docstring now says DO NOT apply this on proto-v0.2.0 or
later. `patches/assert_image.py` asserted the opposite (`"cap.major == 12" in recipe_src`); it now
asserts the override is **absent** and upstream's recipe is intact, which is what makes the fix stick.

Verified twice: first by bind-mounting a corrected `o_proj.py` over the image's copy on both nodes
(`layout.hpp` went from one occurrence per boot to zero, and the engine got past the forward to
`_check_enough_kv_cache_memory`), then in the rebuilt image, where `compute_fp8_einsum_recipe` no longer
contains `cap.major == 12`.

**One forced capacity difference.** At `GPU_MEMORY_UTILIZATION=0.8389` proto2 reports 9.17 GiB of KV
cache memory and refuses `MAX_MODEL_LEN=65536`, which needs 9.48 GiB (estimated max model length
24204) -- the same 9.48 GiB this file already recorded for the `stockops` arm. Our arms therefore run
`0.86`; the reference recipe runs `0.82` with `max_model_len: 262144`. Only the allocation changed, and
at c6 with 512-token prompts the KV pool is not the constraint.

**Drop the page cache before every arm, or the worker dies before health.** The first real-image run at
`0.86` failed with `buffer_size (1059061760 B) exceeds device memory budget (925720576 B)` from
InstantTensor, which sizes its I/O buffer from free device memory -- the same trap this file already
recorded for the phase-1 boot, where the budget read 762578944 B. After
`sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` on both nodes (118 GiB free), the same image
on the same arms served. `harness/run-arm.sh` now does this itself, before it launches either node, so
a run cannot be silently poisoned by a warm page cache; treat a missing drop as a failed arm.

## Runbook: the moment the proto2 build lands

Verified on spark1: `~/goal/` exists (engine logs land there), `harness/run-arm.sh`,
`harness/probe-moe.sh`, `~/drive-median.sh`, `~/goal/ref-base0731.yaml`,
`~/goal/launch-refbase.sh` and `scripts/spark-launch.sh` are all present.

Readiness checked 2026-09-17 while the build ran. **spark2 is idle and ready**: 2.1 TB free on `/`,
117 GiB available, zero containers, zero `sparkrun_*` leftovers, no llama.cpp or vLLM process, load
0.07. spark1 is compiling, nothing is serving anywhere. No protocol measurement on the new base exists
yet, and none was attempted this session: the build holds load average ~16, and round 8 measured that
condition inflating GPU timings by 13 % on unchanged configuration.

**Early path, if the build runs long: this arm does not need proto2.** Several images are
byte-identical on both nodes (verified by ID), so the `b12x` against `flashinfer_b12x` A/B can run on
`vllm-spark-0731:main-029-proto-ccompile` (`0eea65193697`) with no new image and no 29 GB copy. Export
`IMAGE=vllm-spark-0731:main-029-proto-ccompile` to override the pin, run
`bash harness/run-arm.sh fb12x "MOE_BACKEND=flashinfer_b12x"`, then the same with `MOE_BACKEND=b12x`
and tag it `b12xbase`, and compare. Same image, same day, one variable. It is not the headline
comparison against the reference — that still wants the new base and a same-day reference run — but it
answers the MoE-backend question before proto2 exists.

1. **Verify the image.** `docker image inspect vllm-spark-0731:main-029-proto2`. The build log must
   show `vllm ... f37c550bf635`. Nothing else in this runbook is worth doing on a wrong commit.
2. **Copy to spark2, with the tag given explicitly.** `bash scripts/02-copy-main.sh
   vllm-spark-0731:main-029-proto2`. **Do not call it with no argument**: it sources
   `configs/pin.main.env`, whose `IMAGE` is the v0.28-era `vllm-spark-0731:main-b12x`, so the bare
   form copies the wrong image and the worker will fail to find proto2.
3. **Clear leftovers on both nodes** before any arm: `docker rm -f vllm-ds4-0731` and any
   `sparkrun_*` container. `scripts/07-stop.sh` does not remove the sparkrun ones, and the port guard
   in `05-serve.sh` refuses to start while port 8000 is held.
4. **Run our arm with the tool that already does the sequence:** `bash harness/run-arm.sh proto2
   "<env overrides>"`. It clears `vllm-ds4-0731` and the `/dev/shm` NCCL leftovers on both nodes,
   queues the worker 85 s ahead of the head, polls health for up to 450 s, saves
   `~/goal/<tag>-engine.log`, and meters. `05-serve.sh` targets `IMAGE=main-029-proto2` from the
   pin and runs `patches/assert_stack.py` at startup, so the stack assertion is automatic. A head
   started alone blocks forever and looks like a broken image.
5. **Run the reference arm the same day**, because the completion criterion is same-rig, same-day:
   `bash ~/goal/launch-refbase.sh` (recipe `~/goal/ref-base0731.yaml`), then
   `~/drive-median.sh refproto 3 512 1 3 5 6`. Every pass in both arms must clear `gate_france` and
   `gate_9x8`, and our acceptance and tokens-per-step must not be lower.
6. **Then the first real arm:** `bash harness/run-arm.sh fb12x "MOE_BACKEND=flashinfer_b12x"`, no
   rebuild, with both gates read explicitly for the swiglu-clamp reason recorded above.

Nothing in the repo has been measured on `proto-v0.2.0` yet. Every number in `docs/EXPERIMENTS.md` is
proto-v0.1.0-era, so the re-baseline is the first thing that makes any later comparison meaningful.

## The first arm to run after the proto2 rebuild: `MOE_BACKEND=flashinfer_b12x`

Found 2026-09-17 by reading the comparator's recipe instead of assuming both arms were configured
alike. `harness/ref-base0731.yaml` — the authority on what the reference arm runs — passes
`--moe-backend flashinfer_b12x`; our pin defaults `MOE_BACKEND` to `b12x`. Those are different
implementations, not aliases:

- `flashinfer_b12x` dispatches to `FlashInferB12xExperts`
  (`fused_moe/experts/flashinfer_b12x_moe.py`). Its docstring: "Uses `b12x_fused_moe` from FlashInfer
  PR #3080 which fuses token dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
  **single kernel call**. Input quantization (BF16->FP4) is performed inside the kernel so BF16 hidden
  states are passed directly." It supports NVFP4 only.
- `b12x` dispatches to `B12X_MXFP4`, the b12x package's own MoE.

**This is not a missing-package problem.** Both images expose `b12x_fused_moe` and
`has_flashinfer_b12x_moe() is True`; ours is the newer wheel (0.7.0 against 0.6.15) and a superset,
also carrying `B12xNvfp4Config/Runner` and `B12xW4A16Config/Runner`.
`patches/assert_stack.py` already allows it (`ALLOWED_MOE = ("b12x", "flashinfer_b12x")`) and
`configs/pin.golden.env` already sets it.

**And it has never been measured on our arms.** The MoE sweep in `docs/EXPERIMENTS.md` covered `b12x`,
`humming`, `flashinfer_trtllm` (died at worker init) and `flashinfer_cutlass` (never healthy) — not
this one. Since ffn/MoE is the region the profiler prices at 65 % of a target layer and attention is
already ruled out as the reference's source of speed, this is the highest-value one-variable arm
available: set `MOE_BACKEND=flashinfer_b12x`, rebuild nothing, run the protocol.

Carry one caveat: `FlashInferB12xExperts` asserts NVFP4 expert weights, and
`docs/knowledge/04-quantization-kv.md` records NVFP4 *weight* attempts as a dead end on this model —
though `02-model.md` says the checkpoint's experts ship as fp4 and the MXFP4 oracle maps
`flashinfer_b12x` to `B12X_MXFP4`, so the MXFP4 route may reach the same kernel. It either runs and is
measurable, or it is rejected at load. Both are results, and the negative is worth recording either way.

Evidence: `outputs/driver/one-off/moe-backend-comparator.log`.

Two further facts sharpen it, both from the reference's oracle source.

**The comparator is not running a default.** `oracle/nvfp4.py` says, of its selection order:
"FLASHINFER_B12X is intentionally excluded from auto-selection until the upstream CUTLASS SM121 MMA op
guard is resolved; use `moe_backend="flashinfer_b12x"` to opt in explicitly." So the reference arm asks
for a kernel upstream refuses to pick on its own — someone measured it on this hardware and preferred
it.

**And the clamp set explains why we never got there.** `select_nvfp4_moe_backend` narrows the candidate
list to `NVFP4_BACKENDS_WITH_CLAMP = {FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, MARLIN}` whenever
`config.swiglu_limit is not None`. DeepSeek-V4-Flash sets `swiglu_limit 10.0`, so the auto-selection we
were implicitly fighting is restricted to exactly the three backends this rig rejects: our own sweep
recorded `flashinfer_trtllm` dying at worker init ("does not support the deployment configuration since
kernel does not support current device cuda") and `flashinfer_cutlass` never becoming healthy.
`FLASHINFER_B12X` is **not** in that clamp set, which is both why the reference had to opt in and a
reason to read both gates on our arm rather than assume clamp handling matches: the explicit opt-in may
bypass a swiglu clamp the auto-selected backends would have applied.

The arm command, with `MOE_BACKEND` as a host-side variable (it is read at `05-serve.sh:63` and `:334`
to build the argv, so unlike the `VLLM_*` profiling variables it needs no `SERVE_EXTRA_ENV`):

```
bash harness/run-arm.sh fb12x "MOE_BACKEND=flashinfer_b12x"
```

`harness/run-arm.sh <tag> "<env assignments>"` already does the rest: it clears `vllm-ds4-0731` on both nodes, deletes the `/dev/shm` `psm_*`/`nccl-*`/`sem.mp-*` leftovers, queues the worker 85 s ahead of the head, polls health for up to 450 s, writes the engine log to `~/goal/<tag>-engine.log`, and ends by running `~/drive-median.sh <tag> 3 512 1 3 5 6`. The standard flags (k=7, capture 48, util 0.8389, `--async-scheduling`) are baked in, so the second argument is the only thing that differs between arms.

## A memory-bandwidth floor that the quoted step times cannot satisfy (and the basis error behind it)

Established 2026-09-17 from the checkpoint's own geometry plus a measured bandwidth, and it needs to be
settled before more attribution is built on the old numbers.

- The checkpoint (`config.json`, snapshot `7872f01b1d1f`) has 256 routed experts, 6 per token,
  intermediate 2048, hidden 4096, **43 layers**, fp4 experts. Both MoE probes used exactly these, so
  their geometry is confirmed and the earlier caveat about it is retracted.
- A 1 GiB device copy on this GB10 gives **216.7 GB/s** effective (spec ~273; a copy reaches ~80 %).
- Expert weights are 25,165,824 params each, so **3.22 GB per layer** and **138.5 GB across 43 layers**.
- At the measured bandwidth: 14.87 ms per layer, **639.2 ms per full pass on one rank, 319.6 ms at
  TP=2**.

**So a full 43-layer forward pass cannot be faster than ~320 ms across two ranks**, because it must
stream 138.5 GB of expert weights and the total checkpoint is 155-167 GB. The quoted c6 step time of
38.65 ms is therefore not a full-model pass, and the profiler's 1.84 ms per layer for ffn/MoE cannot
be reconciled with one either. Every per-layer budget in this file and in
`docs/knowledge/05-performance.md` derives from those figures, so **re-derive the step-time basis
before using them again**.

Useful side effect: the MoE layer is bandwidth-bound and both good kernels sit at the floor (ours
7.03 ms against 7.43 ms of traffic at 128 experts; FlashInfer 0.6.15 7.67 ms), so our MoE is not
wasting bandwidth and FlashInfer 0.7.0's 71.78 ms is ~9.7x off the floor.

## Live lead: our FlashInfer wheel runs `b12x_fused_moe` 8-12x slower than the reference's

Recorded 2026-09-17. The comparator's recipe opts into `--moe-backend flashinfer_b12x`. Both wheels
expose the same kernel, `flashinfer.fused_moe.cute_dsl.b12x_moe`, and on identical inputs it is
**8.5x to 12.1x slower in our 0.7.0 than in the reference's 0.6.15**, at 64, 128 and 256 local
experts, with **bit-identical outputs** (`mean_abs` agreeing to four decimals). Same computation,
different speed. Sharding was tested and refuted first: the ratio is flat across local expert counts.

Caveats that travel with it: no absolute cost from these probes is trustworthy, because even the fast
wheel spends 7.67 ms per layer at 128 local experts and 43 such layers would exceed the whole 38.65 ms
step, so the probe's expert geometry is probably not the served one (the profiler's own figure is
1.84 ms per layer for ffn/MoE at c1). The version comparison is unaffected: both wheels got identical
inputs and returned identical outputs. Full tables in `docs/EXPERIMENTS.md`.

Next on this lead: fix the probe geometry against the checkpoint's real expert shape, then decide
between pinning the 0.6.15 kernel and pinning the wheel. **Ask the owner before any upstream contact**
— goal item 4 requires it.

## Goal items: what is closed and what is open

Recorded 2026-09-17 against the goal contract. Each closure carries its measurement.

- **Item 1, real NVFP4 KV quantization — CLOSED, and it was never a real item.** The premise was that
  our `nvfp4_ds_mla` is an envelope alias while the reference has a real NVFP4 writer. The reference's
  own source says otherwise. `harness/probe_nvfp4_kv_width.py`, run read-only against
  `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, finds the KV-width decisions for the two dtypes and both
  return **584 B/token**: `kv_cache_interface.py:381-386` returns `storage_block_size * 584` for
  `fp8_ds_mla` and `nvfp4_ds_mla` alike, and `sparse_swa.py:151-154` returns 584 for the same pair.
  The reference's comment states the composition, "448 NoPE + 128 RoPE + 8 fp8 scale", and 448 + 128 +
  8 = 584 is the same arithmetic our `_DSV4_TOKEN_BYTES = 584` uses. **The number that closes it: 584
  B/token on both engines.** The 7,650 B/token in the goal text is a whole-model footprint counting
  the indexer and SWA caches, not a narrower per-layer dtype. This confirms the 2026-08-26 correction
  already in `docs/knowledge/04-quantization-kv.md`, now verified against the live image.
- **Item 4, another third-party kernel generation — CLOSED for the two surfaces that matter, with a
  negative result on each.** WO: `b12x_ref` wins only at or below 8 tokens by at most 0.050 ms and
  loses from 16 up; a port is worth at most 0.13 % of a c6 step. MLA: ours is **2.7x faster**
  (0.224 ms against 0.614 ms) on the same contract. Nothing to port; the older generation is a
  regression in attention.
- **Item 3, bottlenecks in our own code or configuration — one live lead.** Our MoE generation has no
  `static` backend and runs a flat `max_active_clusters` of 188 across the whole c6 band; a cap sweep on
  a synthetic single-GPU shape separates 175/141/96 from 188/149/130 by a reproducible 6 %, but it ran
  under build contention and the shape is not the served one. Not a result yet.
- **Item 2, NVFP4 per layer family — open, and pointed at the right place.** With attention ruled out
  and item 1 closed, the remaining gap has to be ffn/MoE, which the profiler prices at 65 % of a target
  layer.

## Bugs and follow-ups found but not fixed

- **`DeepGemmFp8BlockScaledMMKernel` produces garbage in this image.** Automatic linear selection
  lands on it and both gates fail (`'carecarecare'`, accept 0.0 %, 1.002 tokens/step). `b12x` linear
  is therefore the only working option on this pin, not a preference. It is also not a performance
  opportunity: the broken arm ran 104 ms/step against our healthy 114.
- **`B12X_MOE_FORCE_A8`, `VLLM_USE_B12X_MHC`, `VLLM_USE_B12X_MOE` and `B12X_MLA_SM120_UNIFIED` are
  dead configuration**: nothing in either image reads them. The live overlay knobs are
  `VLLM_USE_B12X_WO_PROJECTION` and `VLLM_USE_B12X_SPARSE_INDEXER`.
- `pin.main-029.env` now reads `DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-B12X_MLA_SPARSE}"`
  so the draft's backend is A/B-able without editing the pin.

## Constraints that still apply

**Use current components only, with patches on top.** The base is `v0.30.0rc1`, the `b12x` is the
current `1.2.6`, and any gap is closed by an overlay in this repo -- never by swapping in an older
library or image to recover a single kernel. That rule retired the `proto-b12x015` direction (see
*The lead* above): `b12x 0.15.3` is not drop-in, and even if it were, a library regression is not a
lever this repo should pull. The lever is the fast paths the current library already ships and our
stack does not use, of which `b12x/comm/pcie` is the strongest lead.

Change one thing at a time and re-measure with the identical protocol. Do not change the workload,
prompt, seed or token count, and do not change the reference image or its configuration. Do not
delete images. The scope is this repo: `patches/`, `docker/`, `scripts/`, `configs/`, `docs/`. Do not
push to any upstream repo and do not open or comment on any PR.

Do not re-litigate: torch.compile (re-verified 2026-09-17 from the reference's own log: it says
`CompilationMode.VLLM_COMPILE` but then `torch.compile is turned on, but the model ... does not
support it`, so it executes uncompiled and so do we), the breakable-cudagraph
path (measured), `CUDAGRAPH_MODE=FULL` (measured), the spec token count (measured), attention in any
form (measured twice), the MoE vendor menu (measured), and the copy-family lead from before the
re-baseline, whose instrument was reading eager-launch lines, not graph nodes.
