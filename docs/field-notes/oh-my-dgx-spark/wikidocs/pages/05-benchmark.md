# 05. 성능과 품질을 제대로 측정하기

벤치마크의 목적은 가장 큰 숫자를 얻는 것이 아니라 **어떤 조건에서 어떤 작업이 가능한지 설명하는 것**입니다. 공개 수치와 직접 측정값을 한 표에 넣을 때는 출처와 조건을 분리합니다.

## 측정값의 이름부터 나눈다

| 이름 | 의미 |
|---|---|
| prefill | 입력 prompt를 처리하는 속도 |
| decode | 출력 token을 생성하는 속도 |
| TTFT | 요청부터 첫 token까지의 시간 |
| single-stream | 한 요청만 실행한 속도 |
| aggregate throughput | 여러 요청을 합친 처리량 |
| latency | 한 요청이 끝나는 시간과 분포 |

prefill 1,000 tok/s와 decode 40 tok/s는 서로 대체할 수 없습니다. context와 출력 길이가 달라지면 end-to-end 체감도 달라집니다.

## 이 저장소의 검증 단계

```text
L0 loaded → L1 serves → L2 generates → L3 decode/prefill
→ L4 concurrency/long-context → L5 tool loop → L6 soak
```

한 단계에 실패했다고 해서 모든 단계가 실패한 것은 아닙니다. 반대로 L0을 통과했다고 L5까지 통과한 것도 아닙니다.

## 결과에 반드시 들어갈 필드

```json
{
  "hardware": "1x DGX Spark",
  "model_revision": "commit or digest",
  "quant": "format and bpw/dtype",
  "runtime": "image digest or commit",
  "context": 0,
  "kv_dtype": "...",
  "speculative": "disabled or exact config",
  "concurrency": 1,
  "workload": "prompt description and output target",
  "measurement": "warmup, repetitions, statistic"
}
```

## 직접 실행한 DeepSeek 기준선

2026-08-22에 이 저장소에서 기록한 단일 Spark 결과는 다음과 같습니다.

| 항목 | 결과 | 판정 |
|---|---:|---|
| server | TP=1, `max_model_len=384000` | container healthy |
| C1 decode minimum | 37.553 tok/s | serving 기준선 |
| C1 decode median | 41.358 tok/s | serving 기준선 |
| C1 decode mean | 40.915 tok/s | serving 기준선 |
| cold prefill | 985.377 tok/s | 1,000 gate 실패 |
| tool call | mock tool 1회 왕복 | 실제 외부 도구 아님 |

이 결과는 GPT-5.6 Sol과 같은 task set을 실행한 결과가 아니며, DeepSeek 공식 Harness 전체를 재현한 결과도 아닙니다. raw 결과는 [`docs/results-deepseek-c1-2026-08-22.json`](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/results-deepseek-c1-2026-08-22.json)과 tool-loop JSON에 보존되어 있습니다.

## token count를 조심한다

`/tokenize` 결과와 실제 chat usage가 다를 수 있습니다. 이 환경에서는 `thinking=false` chat 요청의 token 처리 차이로 79-token drift가 확인되었습니다. 목표 token 수를 판정할 때는 tokenizer count와 server usage를 별도 필드로 저장합니다.

## 품질 하니스

속도와 품질은 같은 표에서 별도의 열로 다뤄야 합니다. 최소 하니스에는 다음 항목이 포함되어야 합니다.

- 고정된 code edit task
- JSON schema task
- 긴 문맥 회수 task
- tool call arguments task
- tool 결과를 받은 뒤의 final answer
- timeout·malformed JSON·server restart 처리

DeepSeek 공식 모델 카드의 Terminal Bench 2.1 점수는 모델 카드와 harness가 제시한 평가 결과입니다. 우리 C1 tok/s를 그 점수와 직접 비교하지 않습니다.

참고: [NVIDIA Performance Benchmarking Guide](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md), [Sol 비교 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/sol-max-comparison-research-2026-08.md).

## 더 자세히 읽기

- 05-1. 벤치마크 설계 상세: Level 0부터 soak까지의 전체 하니스와 결과 예시를 다룹니다.
- 05-2. 벤치마크 레벨: 최소 실행 순서를 빠르게 확인합니다.
- 05-3. 결과 기록과 비교표: 비교에 필요한 필드와 판정 규칙을 제공합니다.
