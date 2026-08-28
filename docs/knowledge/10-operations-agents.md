[← Index](00-index.md) · [Glossary](glossary.md)

# Local Coding-Agent Operations & Failure Recovery

> **Scope:** Running local coding agents against the endpoint — contract, brain/worker split, tool validation, failure runbook.

Operational knowledge for wiring local DGX Spark endpoints to coding
agents (distilled from an external lab corpus, archived unlinked). Provenance tags: **[MEASURED]** = recorded on
real hardware; **[CLAIMED]** = recipe/forum-reported; **[QUOTED]** = official
vendor docs. This chapter is about *operating* the stack — the benchmark
discipline behind the numbers lives in [05-performance.md](05-performance.md)
and the failure table in [07-gotchas.md](07-gotchas.md).

---

## Endpoint Contract

Agents depend on the endpoint contract more than the model name. Pin in client
code/config, before hooking anything up:

- `base_url · model · timeout · max_output_tokens · retry policy` and
  `tool schema · allowed tools · log redaction · health check`.
- Even among "OpenAI-compatible" endpoints, streaming, `usage`, `thinking`,
  structured output, image input and tool-call details differ per provider.
  Before connecting, run in order: `/v1/models` → single chat → single tool
  call → final response that received a tool result.
- Router registry fields per endpoint: endpoint; model id *as actually
  returned by the server*; context limit (actual recipe profile, not the
  launcher ceiling); reasoning on/off default; tool parser (per model/engine);
  vision support + input format; timeout (separate prefill vs tool-loop
  values); fallback (worker or safe-answer path). **Gotcha**: don't swap model
  names while keeping the old parser/context/thinking defaults.
- **Model swap procedure**: send the same prompt to both endpoints; compare
  HTTP status, JSON schema and finish reason BEFORE comparing answer quality.
- **Retry policy**: never retry unconditionally. Timeout → limited retry
  allowed; wrong tool arguments and permission errors → return to the agent
  immediately. Add request IDs + idempotency so auto-retry never re-executes
  the same tool when the server is down.
- Separate model-endpoint health from tool-execution health: an alive endpoint
  does not mean no external-API timeout, file-permission or sandbox errors.


## Brain / Worker Role Separation

A role split is an operational design (based on public recipes + own
experiments), NOT a benchmark of equal intelligence.

- **Roles**: brain/supervisor (plan, long-context summary, worker selection,
  recovery judgment); coding worker (code search/edit/test execution — judge
  patch correctness, test pass, tool arguments); UI/design worker (CSS/layout/
  component iteration); media worker (image/audio/video pipelines); router
  (request classification/fallback/rate-limit → CPU service or small local
  model). Models hold **no permissions** — the tool runner and sandbox do; a
  smarter model does not loosen the approval boundary, timeout or file scope.
- **Recommended split**: DeepSeek V4 Flash 0731 as supervisor; Qwen3.8-27B as
  fast code/UI worker. **Gotcha**: assigning the largest model to every role
  raises memory, latency and cost together.
- **DS4 brain + Qwen3.8 worker needs 4 Sparks total** (2× Spark A: DeepSeek
  TP=2, `:8888`; 2× Spark B: Qwen3.8 TP=2, separate endpoint). Two independent
  TP=2 services, not a TP=4 model. With only 2 Sparks, pick one model; running
  both as independent TP=2 services on the same two nodes is experimental.
- Router judges by task purpose + input form, not model name alone:
  planning/tool/long-context → DeepSeek; component drafts/style variants/screen
  layout → Qwen3.8. Pass worker output (code, diff, decisions) to the
  supervisor as structured results — don't duplicate the whole conversation.
- **Thinking on/off per role**: simple worker/classification → off, short
  output; code edit → off for fast patch, send test results to supervisor;
  planning/recovery → supervisor selectively on; long docs → cap reasoning AND
  context budget; tool loop → separate thinking tokens from tool arguments in
  logs. **Never put thinking-on tok/s in the same column as thinking-off.**
- Agent placement by Spark count: 1 = Qwen worker OR DeepSeek supervisor
  (memory headroom first); 2 = DeepSeek TP=2 supervisor or worker/supervisor
  DP; 3 = DeepSeek TP=2 + 1 worker, or DP=3 (2+1 role split is easier to
  operate); 4 = 2×2 pool; 8 = role pools + router + observability — cluster
  ops outweigh model choice.
- **Failure & fallback chain** (verbatim): tool validation fail → 1×
  arguments repair → re-ask a smaller worker → user approval request → safe
  abort + save logs.


## Tool-Parser Validation

- **Four-step tool-call validation**: (1) schema pass — tool name/required
  args/types identical on server & client; (2) model output — correct tool
  name + JSON arguments; (3) actual execution — runner checks allowlist, path,
  command args, timeout; (4) result return — the model's final answer
  accurately reflects the executed result. A server having a tool-parser flag
  does NOT imply the model's tool calls succeed.
- **≥10 repetitions per model per scenario**: single tool exact name;
  required + optional args; deliberately wrong type/missing args; inject tool
  result → check follow-up; return tool error → check retry/fallback; choose
  among several tools.
- Metrics: `valid_tool_name_rate`, `valid_arguments_rate`,
  `schema_validation_rate`, `unknown_tool_rate`, `tool_error_recovery_rate`,
  `max_loop_depth`, `timeout_rate`.
- **[MEASURED] gotcha**: vLLM needs the model-appropriate `--tool-call-parser`
  + auto tool choice; a function request on a Qwen3.8 smoke server *without* a
  parser returned **HTTP 400** — a config-state limitation, not a
  model-capability limit. On DeepSeek: one mock `lookup_weather` loop passing
  validates the message contract only — not real external tools, error
  recovery, multi-tool parallelism or agent success rates.
- **[CLAIMED] agent-loop gotchas**: Codex showed an empty MCP list because
  `model.json` had `supports_search_tool=true` (set false → works); DeepSeek
  API tool-loop issue outputs one sentence then `stop` without a tool call;
  Hermes Agent infinite reasoning loop on 0731. Model + parser + client
  capability flags must all align — "does the model support MCP" alone is
  insufficient.


## Failure-Recovery Runbook Summary

Full details and the symptom→diagnosis table live in [07-gotchas.md](07-gotchas.md)
and `book/09-*`; this is the compressed runbook.

1. **First 5 minutes** — do NOT change config first; collect `date -Is`,
   `nvidia-smi`, `free -h`, `df -h`, `ps -ef | grep -E 'vllm|sglang|llama|spark'`,
   `docker ps`, `ss -ltnp`, `dmesg -T | tail -200`, `docker logs --tail 500 <container>`
   (synchronized across nodes; save server log, request JSON, last
   successful request). Save logs BEFORE any forced power-off.
2. **Symptom mapping** — low tok/s + low clock → power/thermal/clock; GPU
   memory short → model/KV/batch budget; NCCL timeout → network/interface/
   collective; JSON/tool failure → parser/template/agent contract.
3. **Safe recovery order** — block new requests + stop health checks/retries →
   save logs/`nvidia-smi`/temps → graceful shutdown → reproduce on one node,
   small model, short context → change ONE variable (transport/driver/firmware/
   quant) → smoke → short soak → long soak → record cause and workaround
   separately. Power-off is last resort, except the low-clock case where a
   full power-off is the documented recovery step.
4. **Restart discipline** — after a freeze, check filesystem/journal/NIC/
   driver; don't immediately re-run the same workload; start worker
   (spark2) before head (spark1); never chain stop and serve in one SSH
   session.
5. **Incident record** — time, host, model/revision, image digest, clock/SM
   clock/P-state, temperature, GPU/wall power, context/concurrency, prefill/
   decode/TTFT, network/NCCL, error log, action, recovery result,
   root-cause status (`confirmed`/`suspected`/`unresolved`).

**Permissions (default posture even for dev agents)**: restrict the working
dir to an allowlist; default-deny file deletion, network transfer, credential
access, unlimited shell/sudo, arbitrary package install; dry-run mode showing
command + target files; timeout + max tool count; request ID + idempotency;
never log API keys, bearer tokens, personal file content or env vars. Design
redaction + retention before secrets can mix into model context and logs.


## Agent Completion Criteria (all must hold)

- Fixed task reproduced; tool name + arguments pass schema; no unauthorized
  file/command/network access; execution results accurately reflected in the
  final answer; recovery after failure/timeout/restart without duplicate work;
  prompt, model revision, tool log and judgment re-verifiable.
- **Known unknowns** (as of writing): DeepSeek one-Spark EXL3 real success
  rate as a multi-step coding supervisor; optimal routing between Qwen3.8
  worker and DS4 supervisor; per-framework tool schema/sandbox differences;
  long-agent-memory + prefix-cache stability.

---

## Related Docs

- [01-hardware.md](01-hardware.md) — Node setup, ports, thermal/power
- [05-performance.md](05-performance.md) — Benchmark levels and measured numbers behind agent claims
- [06-deployment.md](06-deployment.md) — Engine selection, EXL3 recipe, endpoint/tool-parser notes
- [07-gotchas.md](07-gotchas.md) — Failure modes and the "do not" list
- [11-cost-decision.md](11-cost-decision.md) — Node-purchase and Sol-vs-local economics


---

**[← Prev](09-golden-deepgemm.md) · [Glossary](glossary.md) · [Next](11-cost-decision.md) →**
