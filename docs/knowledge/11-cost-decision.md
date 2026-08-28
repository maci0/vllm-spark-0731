[← Index](00-index.md) · [Glossary](glossary.md)

# Cost, TCO & Vendor Decision

> **Scope:** Buying and running Sparks — CAPEX/OPEX/TCO, node count, GPT-5.6 Sol vs local, GB10 vendor choice.

Purchase, power and operating economics for DGX Spark (distilled from an
external lab corpus, archived unlinked). All prices are dated
market snapshots, not fixed quotes; power figures are official-doc ratings or
measured examples, not a normal value. Provenance tags: **[QUOTED]** = official
vendor/API docs; **[CLAIMED]** = community/recipe; **[MEASURED]** = real
hardware.

---

## CAPEX / OPEX / TCO

- **CAPEX** = one-time purchase: Spark unit (1 TB vs 4 TB SKU, Founders vs
  OEM), cabling (QSFP cables / optical modules / breakout — 2-node direct vs
  4-node switch cable counts differ), switch (4+ nodes; verify port count AND
  real RDMA compatibility), power (UPS, strips, spare adapters — check whether
  the 240 W adapter is included), storage (model cache, backup SSD), management
  gear (admin Ethernet, router, console), tax/shipping.
- **OPEX** = electricity for gear + switch; extra fans/AC for cooling;
  storage/backup disk replacement; software maintenance + support contracts;
  time investigating/rebooting/recovering failures; time re-downloading +
  verifying models.
- **TCO** = fix a horizon (1 yr or 3 yr), sum CAPEX + OPEX over the same
  period: `TCO(Y) = CAPEX + Y×(power + maintenance + cooling) − residual`.
  Resale/residual is a separate line, never silently subtracted from the
  purchase price. Never divide cost by tok/s alone — what matters is
  throughput passing model fit, context, concurrency and quality gates.
- **CAPEX by node count** (formulas; `D` = unit, `Q` = one QSFP cable,
  `S` = switch + accessories, `U` = UPS/power/storage/management bundle):
  1 → `D + U`; 2 → `2D + Q + U` (200GbE direct, no switch — the DeepSeek
  TP=2/long-context baseline); 3 → `3D + 3Q + U` (QSFP ring, no switch in the
  official path); 4 → `4D + 4Q + S + U` (first tier where switch + RDMA cost
  becomes a default item); 6–8 → ports/cables/power/cooling become a project;
  2×2 pool → `4D + 2Q + U` when each pair is direct-connected and pairs share
  no compute collective, else `4D + 4Q + S + U`.
- **Power numbers [QUOTED]**: adapter 240 W; GB10 SoC TDP 140 W; rest of system
  ~100 W; regulatory **max 233.2 W AC, idle 38.0 W, off-mode 4.1 W**. Example
  annual cost at 200 KRW/kWh (`W/1000 × 8760 × rate` — NOT a fixed tariff):
  idle 38 W ≈ 333 kWh/yr (~₩67k) per unit; max 233.2 W ≈ 2,043 kWh/yr
  (~₩409k) per unit → 2 units ≈ ₩817k, 4 ≈ ₩1.634M, 8 ≈ ₩3.269M at max.
  A measured 90 W average ≈ 788 kWh/yr (~₩158k). Excludes switch/Mac/fans/
  monitor/UPS losses; 4 Sparks alone max ~932.8 W, 8 ~1,865.6 W — check circuit
  + UPS with switch/cooling added.
- **Wall-power protocol (minimum)**: 30-min idle average post-boot → 10-min
  warm-up c1 decode → c4/c8 aggregate with per-node power+temp → long prefill +
  1–2 h soak → time-weighted average. Use a true-RMS meter between wall and
  unit; `nvidia-smi` device power ≠ total AC.


## Node Purchase Decision

- Node count by question: 1 → single user, dev assist, simple agent; 2 →
  large context/model fit (TP=2) or two endpoints (DP); 3–4 → topology +
  concurrency scaling; 6–8 → team service + cluster ops (switch/power/cooling/
  fault isolation become the problem before model size).
- Don't buy multiple units before defining the target workload on one. The
  purchase conclusion is "on which workload, at what point does it break
  even", not "which count is fastest".
- Add a node ONLY if: the current model actually failed from memory shortage;
  TP/PP/DP use of the new node is decided; the addition grows model capacity
  vs concurrency vs single-request speed (decided); the real topology + NCCL
  transport is verified; ≥2 h soak with memory/power/temp/error/restart results
  exists. Non-24/7 workloads can be scheduled to cut idle power.
- **Value summary**: 1 node = lowest entry cost to validate model + runtime;
  2 = biggest value jump (DeepSeek TP=2, long context); 3 = money goes to
  DP/PP/2+1 service split, not TP=3; 4+ = switch/power/cooling/NCCL ops matter
  as much as hardware price; 8 = a small cluster, not a faster personal PC.
  Decision criterion: not "can it run the biggest model" but "total cost to
  reliably handle the validated workload".
- **Mac roles**: control host/router (UI, SSH, API gateway, auth, logs);
  separate MLX worker (doc preprocessing, small fast models); DS4 decode-assist
  experiment (MCDMA research — not counted for CAPEX savings or memory pooling
  until a public implementation + independent tok/s verification exists).
  Buying a Mac does NOT auto-pool memory with a Spark.
- **Cloud comparison**: `effective work = successful requests × quality pass rate × effective output per request`;
  local monthly = (CAPEX − residual)/
  months + power + cooling + maintenance; cloud monthly = input + output +
  storage/network/reserved-instance. Local is economical when equipment is
  used long + request volume is sufficient, data must not leave the premises,
  or local latency/always-on endpoint matters; not economical at low volume,
  frequent latest-model swaps, or when power/cooling/incident response is
  unacceptable.


## GPT-5.6 "Sol" vs Local

**Naming rule**: "Sol Max" is NOT a separate API model — it is `gpt-5.6-sol`
with `reasoning.effort=max`; mixing model name and reasoning setting misreads
comparisons.

- **[QUOTED] (OpenAI model doc, as of 2026-08-22)**: model id `gpt-5.6-sol`,
  alias `gpt-5.6`; reasoning effort `none/low/medium/high/xhigh/max`; context
  **1,050,000 tokens**; max output **128,000 tokens**; pricing input $5/1M,
  cached $0.50/1M, output $30/1M; prompts >272K tokens → whole request billed
  at 2× input, 1.5× output; function calling, structured outputs, image input.
  ⚠ A later research snapshot (2026-08-23) quotes $4/$0.40/$20 per 1M — verify
  against the live OpenAI model doc before using either figure.
- **Comparison framing**: Sol is a hosted API (cost = request tokens/features/
  account policy; latency includes network + service state; no weight/runtime
  control); local Spark is user-managed (checkpoint + quant + runtime combo;
  KV headroom is recipe-dependent; cost = purchase + power + cooling +
  management time; direct control of weights/runtime/logs/data boundaries).
  "Which is smarter" cannot be settled by one tok/s number.
- **DeepSeek side**: official card reports Terminal-Bench 2.1 **82.7** (Code
  Agent eval, `reasoning_effort=max`, temp 1.0, top_p 0.95, minimal DeepSeek
  Harness). **[MEASURED]** this repo's local C1 (MiaAI recipe, `thinking=false`)
  is a *serving gate*, not a quality-equivalence eval: semantic/JSON/code
  PASS, code-decode median 41.358 tok/s, cold prefill 985.377 tok/s (**failed**
  the 1,000 gate). "Card reported 82.7" ≠ "our Spark reproduced the official
  82.7".
- **[QUOTED] (StateM paper, arXiv:2608.15089)**: harness changes results, not
  just weights — DeepSeek V4 Flash standard 82.7 → **88.1** with the same
  runtime/runbook, **89.1** on the common 88-task core; GPT-5.6 Sol xhigh
  reference 84.9, Sol max 88.8. Evidence that "DeepSeek with a good harness can
  score near Sol on coding-agent tasks", not equivalence on everything.
- **Fixed conclusion**: "DeepSeek V4 Flash 0731 is a very strong local
  coding-agent candidate on DGX Spark. Scores near GPT-5.6 Sol max were
  reported in research applying a common agent runtime and runbook, but this
  does not mean local EXL3 serving's speed, quality and cost are equivalent to
  the Sol API." "Similar speed to Sol" is unverifiable from public data —
  comparing raw output tok/s compares different serving layers.
- **Fair-comparison design**: same Terminal-Bench subset/fixed task set; same
  agent runtime, tool schema, runbook, timeout; official-harness and
  local-serving conditions as separate rows; Sol = `gpt-5.6-sol` +
  `reasoning.effort=max` explicit; record success rate, failure reasons,
  input/output/reasoning tokens, wall time, tool calls; local includes
  power/download/ops time, API includes input/output/cache/long-prompt
  multiplier; separate single-stream decode from aggregate throughput.
  Never mix official hosted-model scores with local C1 results in the same row.


## Vendor Decision (GB10 OEMs)

- **Use-based selection** (from the certified-systems list in
  [01-hardware.md](01-hardware.md)): NVIDIA recipe reproduction → NVIDIA DGX
  Spark; continuing existing ASUS experiments → ASUS Ascent GX10 (most
  DeepSeek field material); thermal headroom priority → Acer as a candidate;
  enterprise support/field replacement → Dell/Lenovo/HP; model-management
  UI/RAG → GIGABYTE; edge/security appliance → MSI; memory capacity/MLX →
  Apple Mac Studio (separate architecture — not a Spark cluster node).
- **Price comparison must include**: unit + SSD SKU + official adapter + QSFP
  cable + switch (4+ units) + support contract + power/cooling. Never use a
  dateless "lowest price" as a baseline (region/currency/stock/support period
  vary). Price-table columns: checked_at, seller, product, memory/storage,
  currency, tax_included, shipping_included, warranty, price.
- **Snapshot (2026-08-21, for reference only, not a recommendation)**:
  **[QUOTED]** Founders Edition MSRP raised $3,999 → **$4,699** (memory
  supply); marketplace 4 TB at $4,699. Korean market 4 TB Founders/OEM units
  ~8.99M–13.5M KRW; Mac Studio from 4.29M KRW (raising memory/storage changes
  the price — don't compare start prices only). One domestic listing for a
  MikroTik CRS812 200G/400G switch ~2.65M KRW — port config for 4 Sparks + real
  NCCL/RoCE compatibility must be verified separately.
- **Power gotchas**: the 240 W adapter rating ≠ real wall power; in clock-cap
  comparisons record GPU rail and wall separately; don't extrapolate a single
  benchmark's 40 tok/s to monthly cost — include actual prompt length, output
  length, wait time, concurrency.
- **Final sequence** (recommended): 1) one node, official recipe, pass
  `/v1/models` + short chat + single tool call; 2) same prompt/task set across
  Qwen/DeepSeek/MiniMax; 3) record prefill, TTFT, context, tool loop, failure
  rate — not just c1 decode; 4) decide 2 nodes for capacity vs concurrency;
  5) add switch/RDMA/MCDMA only after a minimal working endpoint; 6) update
  quality/cost conclusions only with a same-harness Sol comparison.


---

## Related Docs

- [01-hardware.md](01-hardware.md) — GB10 vendor landscape, thermal/power numbers, node setup
- [05-performance.md](05-performance.md) — Measured vs claimed numbers behind the cost framing
- [06-deployment.md](06-deployment.md) — The one-Spark EXL3 and two-Spark profiles this chapter prices
- [10-operations-agents.md](10-operations-agents.md) — Ops burden that OPEX must cover


---

**[← Prev](10-operations-agents.md) · [Glossary](glossary.md) · [Next](12-debug-standin.md) →**
