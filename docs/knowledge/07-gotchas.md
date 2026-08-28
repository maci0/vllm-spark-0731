[← Index](00-index.md) · [Glossary](glossary.md)

# Gotchas & Constraints

> **Scope:** The measured “do not” list, operational rules, and failure-recovery procedures.

The hard "do not" list accumulated across the project. Each item was measured and caused regressions.

---

## Never Do These (Measured Worse or Broken)

| # | Action | Result |
|---|--------|--------|
| 1 | **Gather packed-at-store indexer K** (interleaved FlashInfer gather) | Numerically wrong: DSpark accept 38–70% vs ~73%, hitches |
| 2 | **Call b12x MXFP8 `wo_proj.run()`** | France loops (infinite generation) |
| 3 | **Graph `_sample_sequential`** (DSpark full-step graph) | Accept 66.7% → 57.4% (shared `lm_head`) |
| 4 | **Add CUDA graph size 6** (DSpark 1+5 padded to 8) | 1-way 23.98, 8-way 71.52, KV 92–94k |
| 5 | **Feed the 1-row scheduled scorer into 8-row decode** | 8-way 16.29 tok/s |
| 6 | **Multi-row scheduled paged scorer** (`q_rows` 2–8) | 1-way 25.36 vs unscheduled |
| 7 | **`preinitialize_invalid_logits=False`** | Text drift, 1-way 15–19 tok/s |
| 8 | **Pack-every-decode / padded 48×1024 pack** | 1-way below 17.81 |
| 9 | **Packed sidecar at insert** (dual packed+interleaved) | ~3.7 GiB, spark2 OOM |
| 10 | **`plan_paged_schedule` inside CUDA graphs** | Frozen warmup seqlens, accept 47% |
| 11 | **Extra capture sizes** `[1,2,3,...,8,...]` | KV 97k→92k |
| 12 | **Expand 1024-wide page64 tables ×4** | Garbage page ids, 8-way collapse |
| 13 | **Force `LINEAR_BACKEND=triton`** | Unpatched crashes with `KeyError: 'float8_e8m0fnu'`; with PR #47988 boots but is slower than b12x (144 vs 172 tok/s @ c32) |
| 14 | **Pass GLM `scale_format=2` / 432/368 writer into DSV4 584 B** | Wrong writer, silent corruption |
| 15 | **util 0.85 on spark2** (even with swap enabled) | earlyoom SIGTERM at MemAvailable<8% |
| 16 | **FULL cudagraph on the overlay rc2 path** | `的超` / -ln(96) / cudaGraphLaunch failure |
| 17 | **Overlay main Python onto v0.27.1 `.so`** | ABI mismatch, crashes |
| 18 | **Drop ReLU in MLA indexer** (use weighted-Q) | Wrong kernel selected, silent quality loss |
| 19 | **`systemctl mask swap.img.swap`** (or leave it masked) | No swap at boot: earlyoom logs `swap total: 0 MiB` even though fstab has the entry |

---

## Operational Rules

1. **Never chain `07-stop.sh` and `05-serve.sh` in one SSH session** — Stop, confirm containers gone, then serve. Separate SSH per node (`ControlPath=none`).

2. **Start worker (spark2) first, then head (spark1)** — Reverse order causes NCCL timeout.

3. **Validate/bench from spark1 (`127.0.0.1:8000`), not the laptop** — Network latency skews measurements.

4. **`docker commit` must restore `ENTRYPOINT ["vllm","serve"]` and `CMD []`** — Overlays change entrypoint.

5. **After a `b12x-sparse` reapply, pass `--vllm-dir /opt/vllm/vllm`** — A duplicated `B12X_MLA_SPARSE` enum makes `import vllm` raise `TypeError`.

6. **Do not raise spark2 to util 0.85** — earlyoom SIGTERMs at MemAvailable<8% (~10 GiB), swap or no swap.

7. **Do not serve without DSpark k=5** — This recipe never serves without that spec (locked in checkpoint).

8. **`MAX_NUM_SEQS=32` is the ceiling** — 48 hangs at boot.

9. **Swappiness lives in `/etc/sysctl.d/99-dgx-spark-swap.conf`** — `vm.swappiness=10` on both nodes (was 100; disk-swap stalls during decode). zswap stays on (`zstd/zsmalloc`, `max_pool_percent=5`).

10. **BLOCK_SIZE=256 must be multiple of 128** — C128 storage = block_size/128. SWA pages hardcoded 64.

---

## Config-Specific Gotchas

### Overlay rc2 (`pin.nvfp4.env`, `pin.env`)

- **FULL CUDA graphs crash** — Use `CUDAGRAPH_MODE=PIECEWISE`
- **FlashInfer DSV4 needs TOPK 192** — Stock 0.6.16.post3 doesn't have it; overlay adds it
- **`nvfp4_ds_mla` dtype guard** — vLLM 0.28 `validate_nvfp4_kv_cache_with_mla` uses `startswith("nvfp4")`; overlay narrows to exact `"nvfp4"`

### Matched Main (`pin.main.env`)

- **No blanket DeepGEMM kill** — Keep `is_deep_gemm_supported()` guards, don't blanket-return False
- **Cutlass DSL 4.7.0** — Not 4.6.2 from vLLM cuda.txt; metadata rewrite needed
- **b12x from master** — Not the pinned version in vLLM
- **InstantTensor for cold start** — Primary loader; fastsafetensors GDS fallback

---

## Debugging Checklist

| Symptom | Check / Command |
|---------|-----------------|
| France prints `的超` or garbage | KV/attention mismatch, run `python3 patches/assert_stack.py --kv nvfp4_ds_mla --attn B12X_MLA_SPARSE --moe b12x` |
| 1-way < 17.81 tok/s | `max_num_seqs` too low, or graph capture issue |
| 8-way collapses at high concurrency | `max_num_seqs` too high, or KV pool exhausted |
| DSpark acceptance < 40% | Packed gather bug, or wrong attention backend |
| Container OOM on spark2 | util > 0.8 or swappiness>10 stalls, check `earlyoom` + `swapon --show` |
| `import vllm` TypeError | Duplicate `B12X_MLA_SPARSE` enum, reapply with `--vllm-dir /opt/vllm/vllm` |
| `KeyError: 'float8_e8m0fnu'` | Triton backend without PR #47988 selected, use b12x |

---

## Operations & Failure Recovery (external lab corpus)

Failure modes from the Korean lab corpus's operations chapter, distilled with
provenance. Four failure layers — hardware (temp/power/fans/NIC/cables),
OS/driver, communication (NCCL/RDMA/UCX/socket fallback), model/runtime
(quant/KV/parser/CUDA graph) — one layer's symptom can look like another
layer's cause; change one layer at a time.


### Low clock / low power

- `GPU-Util=96%`, `P-state=P0`, no throttle reason ≠ healthy — check SM clock
  and power draw under load too. **[CLAIMED, Blackwellboy X]**: SM clock
  ~799 MHz / ~19.5 W / decode ~44 tok/s; after a full power disconnect:
  ~2.3–2.5 GHz / ~92 W / 73.9 tok/s. Other reported pins: 721 MHz and 550 MHz
  recovered after full power-off (unplug power brick AND AC, wait ~5 min).
- Distinguish an intentional `nvidia-smi -lgc` cap from a fault low-clock;
  a cap is an experiment — record the starting value and the `-rgc` reset
  command, and judge decode (memory-bound) vs prefill (compute-bound)
  separately.
- Capture under load: `nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,power.draw,utilization.gpu --format=csv -l 1`.

### OOM

- Narrow in order: (1) stop/isolate other inference processes; (2) lower
  `max_model_len` and `max_num_seqs`; (3) check prefix/cache, vision, tool-loop
  context growth; (4) check CUDA graph/workspace and container memory cap;
  (5) change quant/KV dtype one at a time; (6) verify memory reclamation in a
  long soak. Restart with short context + single stream to separate weight
  problems from KV problems, then raise context and concurrency one at a time.
- `max_model_len=384000` configured ≠ every request stable to that length.
- **Gotcha**: whole-OS crash when system memory is full (forum) — don't just
  raise `GPU_MEMORY_UTILIZATION`. The DeepSeek one-Spark recipe's
  `GPU_MEMORY_UTILIZATION=0.94` + EarlyOOM disabled is an aggressively
  memory-occupying experiment condition; check free memory + recovery plan
  before copying it to other models.

### NCCL / RDMA

- Distinguish: link up ≠ IP connectivity ≠ RDMA device use ≠ NCCL communicator
  built ≠ all-reduce passed ≠ long-generation stable. **[CLAIMED]**: a case
  where link up, channel establishment and weight load all succeeded, then the
  first all-reduce deadlocked both vLLM AND TensorRT-LLM — "model loaded" ≠
  "distributed inference works". NCCL 2.28.9 → long-generation hangs; 2.30.4
  fixed it (4× DeepSeek report) — pin NCCL versions.
- Socket fallback: expected `NET/IB`, actual `NET/Socket` changes
  speed/latency/CPU; verify `/dev/infiniband` is passed into containers.
  **[CLAIMED]**: RDMA passthrough turned 9.8 → 25.1 tok/s (~2.5×) in a SGLang
  GLM case on 4× Spark.
- `UCX_MEM_MMAP_HOOK_MODE=none` / `UCX_RCACHE_MAX_UNRELEASED=1024` are
  recipe-specific leak mitigations, not an official fix — record before/after
  soak and the rollback method.

### Firmware / ConnectX-7

- ConnectX-7 may negotiate 200G yet deliver low payload (12–13 Gbps cases
  recovered to ~109–111 Gbps after a full power drain — field workaround, not
  official). **[CLAIMED, forum]**: `mlnx-fw-updater` touched a CX-7 during
  apt/dpkg → stuck in pre-init ("STATIC CONFIG NOT DONE", error 110) → RMA.
- Rules: no unsolicited firmware updaters; save current firmware/driver/device
  state first; check official release notes + recovery/RMA path; validate on
  one node before a cluster; never mix firmware changes and model benchmarks
  in same-day results.

### First 5 minutes after a failure

Do NOT change config first — preserve state to reproduce. Collect (timestamps
synchronized across nodes in a cluster):

```bash
date -Is
nvidia-smi
free -h
df -h
ps -ef | grep -E 'vllm|sglang|llama|spark' | grep -v grep || true
docker ps
ss -ltnp
dmesg -T | tail -200
docker logs --tail 500 <container>
```

Save logs BEFORE any forced power-off — a reboot can erase pre-failure
memory/temperature/transport info. Also save the server log, request JSON and
the last successful request. Symptom mapping: low tok/s + low clock →
power/thermal/clock; GPU memory short → model/KV/batch budget; NCCL timeout →
network/interface/collective; JSON/tool failure → parser/template/agent
contract. Changing multiple variables at once recovers but leaves no cause.

### Restart discipline

Safe recovery order: (1) block new requests, stop health checks + client
retries; (2) save logs, `nvidia-smi`, `free -h`, temps; (3) graceful
container/server shutdown; (4) reproduce on single node, small model, short
context; (5) change ONE variable among transport/driver/firmware/quant;
(6) smoke test → short soak → long soak; (7) record cause and workaround
separately. Power-off is NOT step one (it destroys diagnostic data) — except
the suspected low-clock case where a full power-off is the documented
recovery step. After a force-kill/power-off-level freeze, check filesystem,
journal, NIC and driver; don't immediately re-run the same workload.

**Incident report fields** (mandatory): incident time, host, model/revision,
image digest, clock cap, SM clock, P-state, temperature, GPU/wall power,
context, concurrency, prefill, decode, TTFT, network link, NCCL result, error
log, action, recovery result, root-cause status (`confirmed`/`suspected`/
`unresolved`). Without these, "it got slow" is unreproducible.

---


## External-recipe gotchas (from linked references, 2026-08-26)

- **`--gpu-memory-utilization` can kill the host [RECIPE]**: >~0.85 on small models → OS reboot / swap death spiral (unified memory starves the OS); 0.60 is the safe ceiling for 8B-class [shunsuke-nashiki], 0.94 max for the DS4 EXL3 single-Spark [MiaAI], 0.85 → OS starve [natolambert]. Disable swap + systemd memory jail + SSH `OOMScoreAdjust=-1000` as defense-in-depth.
- **16 seats deadlock the anemll stack [RECIPE — AlexLJC]**: EngineCore "No available shared memory broadcast block found in 60 seconds", both TP ranks alive — standing hypothesis: unapplied draft-KV batch-row patch.
- **Fabric reality [RECIPE — bytebunkerlabs, anakronox]**: QSFP negotiates 200 GbE but ConnectX-7 sits on PCIe Gen5 ×4 → **real cross-node ~100 Gbps**; a ConnectX-7 can come up at 10 Gb/s (watch negotiated rate); two independent nodes often beat one tightly-sharded pair.
- **`util` 0.80 → 0.835 on the anemll stack [RECIPE — AlexLJC]**: +21.67% KV pool for −1.2% decode.
- **TTFT measured off the first SSE byte is wrong [RECIPE — AlexLJC]**: vLLM emits the role delta before prefill completes — use first content token or non-streaming `max_tokens=1`.
- **DSpark k must be ≤ `n` and a multiple of block size [RECIPE — MiaAI/hazyumps]**: `n` must be ≥ `dspark_block_size` or the engine refuses to start.
- **HF downloads throttle unauthenticated [RECIPE — shunsuke-nashiki]**: hundreds of KB/s + per-IP temporary blocks — use a token / mirror / `HF_HUB_OFFLINE=1` once cached.
- **vLLM spin-wait is a TP=2 heat hazard [COMMUNITY — nacyot repro]**: `shm_broadcast.py` `busy_loop_s=1` → 85 °C in 90 s / 90 °C in ~2.3 min under load (fanless, uncapped; trip 104.8 °C); heat source is the perf-core cluster (TS1P ~89.6 °C avg), not the GPU. Patch to `busy_loop_s=0.002` caps at ~85 °C for −2–6% single-stream (no difference at 4 concurrent; +7.3% on 30-min saturated load).
- **GB10 has no watt limit and no fan telemetry [RECIPE — nacyot]**: `nvidia-smi -pl` N/A, no powercap/devfreq/nvpmodel; `nvidia-smi fan.speed` empty — use ACPI zone temps. Clock cap is the standard lever (`-lgc 300,2000` snaps ~1989 MHz; persist via systemd oneshot). **PD low-clock bug**: clock sticks at 400–950 MHz / ~10 W with all throttle flags Not Active — reboot and `-rgc` don't clear; full power drain + 1 min; prevent with PD FW update (LVFS `com.asus.gx10dgx.usbpd.firmware` 0x1→0x516).
- **cgroup limits cannot cap vLLM on GB10 [RECIPE — jschmied App G]**: GPU/unified-memory allocations are invisible to cgroups — a memory-capped container can still power-cycle the box. Docker `--memory` only helps SGLang because its transient allocs are CPU-side (next bullet).
- **SGLang accounting misses 25–40 GB transient allocs [RECIPE — hasso5703]**: flashinfer fp8 autotuner + graph capture spike past `--mem-fraction-static`; fraction >0.50 or native (non-Docker) SGLang can hard-freeze the box — Docker `--memory 100g` + fraction 0.50 is safe-by-construction; `--sleep-on-idle` saves 10–12 W.
- **`--load-format fastsafetensors` spikes unified memory [RECIPE — sparkyard]**: 86 GiB peak vs 54 GiB on a 34.9 GiB model — avoid on UMA; llama.cpp needs `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + `--load-mode none` (mmap page-fault thrash / 1.37–1.44× read amplification on unified memory); ~126.5 GB used crashes the box (ceiling ≈117.8 GiB).
- **earlyoom must be off on multi-node KV-pin recipes [COMMUNITY — MiaAI GLM-5.2 / nacyot]**: at 11–12 GiB KV pins the box sits near 1 GiB free and earlyoom SIGTERMs Ray workers mid-graph-capture — looks like GPU OOM; the nacyot single-Spark runbook stops earlyoom explicitly.
- **Cross-node TP=2 CUDA-graph hang on official images [RECIPE — pulsar]**: workaround `--enforce-eager`; stock images can't serve DSV4 at all (prefill livelock, upstream vllm#49026) — need PR-#41834-era builds. CUTLASS FP4 MoE is silently wrong on sm_121 → `--moe-backend marlin` (or b12x); `VLLM_MARLIN_USE_ATOMIC_ADD=1` required on SM121.
- **SSE chunk packing under-counts spec decode [RECIPE — palmfuture / pulsar / jschmied]**: vLLM packs accepted spec tokens into ONE SSE chunk — delta-counting shows 14.7 vs 60.1 tok/s and can make MTP look like a slowdown (11.4 → 7.6 "tok/s"); count `usage.completion_tokens`, never deltas (pulsar under-metered DSpark 3.46× and retracted).
- **0731 DSpark pitfalls [RECIPE — palmfuture]**: Patch 4 required — vLLM's DSpark draft loader silently drops 12 shared-expert tensors (`gate_up_proj` w1/w3 in layers 43–45; acceptance 25.7% → 60.2%, decode 32.7 → 55.4 tok/s); the model card's **k=7 crashes with `dspark_block_size=5`** (drafter emits exactly 5 tokens/pass); `--max-cudagraph-capture-size` must equal `--max-num-seqs`; `VLLM_USE_B12X_MOE=1` essential (else DEEPGEMM_MXFP4 fallback halves decode).
- **Reasoning/thinking traps [RECIPE — MiaAI / nacyot / jschmied]**: `DEFAULT_THINKING=max` produced ~50,000 reasoning chars (~12.5k tokens) on a moderate prompt, and unbounded reasoning ran 2 h 45 m in one case — size `max_tokens` in the tens of thousands, set `thinking_token_budget`; client stop strings can fire inside `<think>`; `reasoning_effort=xhigh` (vendor default on some builds) burns ~26k thinking tokens with zero content — ship `medium`.
- **Build/compile fan-out OOMs the box [RECIPE — jschmied App B / Fulton wheels]**: `BUILD_JOBS=16` + resident model → global OOM, 3-h hang, manual power cycle (use `BUILD_JOBS=4` with ~110 GiB free); `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1` or JIT fan-out OOMs; flash-attn needs `MAX_JOBS` 4–8; `nvidia-smi` memory fields return `[N/A]` on GB10 — use `free -h`.
- **RDMA/NM/ufw traps after cable moves [RECIPE — nacyot]**: 13.3 Gb/s stuck state on a 200G link (all counters normal) fixed ONLY by full power drain (→108 Gb/s, both rails 196) — reboot doesn't clear it; ufw allows ICMP by default (ping OK / TCP dead after a cable move); NetworkManager profile pins the port MAC (`permanent MAC address doesn't match`) — disable NM management of the CX-7 iface.
- **Training on GB10 [COMMUNITY — arXiv:2608.07226]**: kernel compile/autotuning can exceed the default NCCL watchdog (~17-min timeout on a step-zero eval bug — guard `step % eval_every == 0 and step > 0`); rank 1 stays silent by convention (check via nvidia-smi); use `NCCL_TIMEOUT=3600` + `TORCH_NCCL_BLOCKING_WAIT=1`.
- **llama.cpp DSpark/MTP specifics [COMMUNITY — 67ailab / classmethod]**: `spec-type draft-mtp` is the config key (not the `--mtp` download flag); Unsloth "110–135 GB for 3-bit" is a GB/GiB mixup (121 GiB ≈ 130 GB); `cache_prompt` on by default turns 411 → 41 tok/s prefill on repeat prompts; DSpark via antirez + support GGUF measured **−22% decode** — "adding speculation is not guaranteed faster".

## Related Docs

- [03-kernels-attention.md](03-kernels-attention.md) — Packed gather bug details
- [05-performance.md](05-performance.md) — Benchmark impact of each gotcha
- [06-deployment.md](06-deployment.md) — Correct serve/stop order
- [08-upstream.md](08-upstream.md) — Upstream fixes for some gotchas
- [10-operations-agents.md](10-operations-agents.md) — Failure-runbook summary and agent endpoint contract

### Raw evidence (field notes)

- [`../field-notes/dgx-spark/TROUBLESHOOTING.md`](../field-notes/dgx-spark/TROUBLESHOOTING.md) — symptom → cause → fix table for every failure hit
- [`../field-notes/dgx-spark/BUG_REPORT_b12x_2node_deadlock.md`](../field-notes/dgx-spark/BUG_REPORT_b12x_2node_deadlock.md) — the b12x 2-node deadlock write-up

---

**[← Prev](06-deployment.md) · [Glossary](glossary.md) · [Next](08-upstream.md) →**
