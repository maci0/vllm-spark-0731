# 05-3. 결과 기록과 비교표

이 페이지는 [05-1. 벤치마크를 제대로 설계하기](05-1-benchmark-design.md)의 상세 내용입니다.

좋은 결과표는 숫자보다 조건을 먼저 보여준다.

## 필수 필드

```text
date
hardware
model_revision
quant
runtime_commit
context
kv_dtype
speculative
concurrency
prompt_tokens
output_tokens
prefill_tok_s
decode_tok_s
latency_p50
latency_p95
quality_gate
status
```

`status`에는 `loaded`, `serves`, `benchmarked`, `tool-tested`, `agent-tested` 중 실제로 통과한 상태만 적는다.

## 비교 원칙

- prefill과 decode를 한 숫자로 합치지 않는다.
- single-stream과 aggregate throughput을 분리한다.
- clock cap, 전력 모드, 온도를 숨기지 않는다.
- 공식 수치와 local 측정을 별도 행으로 둔다.

실패한 결과도 저장한다. 실패 조건이 남아 있어야 다음 recipe가 무엇을 바꿨는지 알 수 있다.
