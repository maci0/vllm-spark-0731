# Submitting DeepSeek-V4-Flash-0731 to Spark Arena (spark-arena.com)

Community DGX Spark LLM leaderboard. Submission is via the **sparkrun** CLI, which authenticates,
launches its own container from a recipe, runs the standardized `spark-arena-v1` sweep, and uploads.
It is NOT a manual "paste your numbers" form.

## Recipe (authored + validated)

`serving/deepseek-v4-flash-0731-2xspark.yaml` — official FP8 model + our full B12X + 512K + CUDA-graph
config (perf-identical to the abliterated apetersson we ran; abliteration doesn't affect speed).
Uses the official `deepseek-ai/DeepSeek-V4-Flash-0731` for leaderboard comparability.

Staged on **spark1** at `/home/maci/deepseek-v4-flash-0731-2xspark.yaml`. sparkrun 0.3.3 installed
there via `uvx`. Dry-run passes (parses, VRAM estimate KV 43GB / 1.2x ctx at 512K) and stops only at
the login gate.

## Submit — two commands, ON SPARK1 (only the login is not automatable)

```bash
# 1. OAuth login (headless: prints a URL + code to enter in any browser)
uvx sparkrun@latest arena login

# 2. benchmark + submit
uvx sparkrun@latest arena benchmark /home/maci/deepseek-v4-flash-0731-2xspark.yaml \
  --hosts 10.0.1.1,10.0.1.2 --tp 2
```
Use `sparkrun@latest` (plain `uvx sparkrun` may resolve the stale 0.1.13 that lacks `arena`).

## Caveats

1. **Step 2 takes down the running serve** — sparkrun launches its own container to benchmark. Run
   it only when done using the live 512K endpoint.
2. It runs the standardized `spark-arena-v1` profile (concurrency sweep, TTFT, tok/s) under your
   account; the posted number is sparkrun's measurement of this config.
3. `--hosts 10.0.1.1,10.0.1.2` = RoCE IPs (run from spark1, which has passwordless SSH to both). From
   another host use the mgmt IPs `192.168.0.211,192.168.0.212` with your SSH key reachable to sparkrun.

## Our own measured reference (512K, this config)

| Concurrency | Aggregate tok/s |
|---|---|
| 1 | 27.8 |
| 8 | 114 |
| 24 | 181 |
| 48 | **337** |

512K context, FP8, CUDA graphs, no-spec, TP2 over RoCE, util 0.85, seqs 48. Stable (big-ctx + c48
verified). Submission should land near this.
