[← Index](00-index.md) · [Glossary](glossary.md)

# Gotchas & Constraints

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

## Related Docs

- [03-kernels-attention.md](03-kernels-attention.md) — Packed gather bug details
- [05-performance.md](05-performance.md) — Benchmark impact of each gotcha
- [06-deployment.md](06-deployment.md) — Correct serve/stop order
- [08-upstream.md](08-upstream.md) — Upstream fixes for some gotchas

### Raw evidence (field notes)

- [`../field-notes/dgx-spark/TROUBLESHOOTING.md`](../field-notes/dgx-spark/TROUBLESHOOTING.md) — symptom → cause → fix table for every failure hit
- [`../field-notes/dgx-spark/BUG_REPORT_b12x_2node_deadlock.md`](../field-notes/dgx-spark/BUG_REPORT_b12x_2node_deadlock.md) — the b12x 2-node deadlock write-up

---

**[← Prev](06-deployment.md) · [Glossary](glossary.md) · [Next](08-upstream.md) →**
