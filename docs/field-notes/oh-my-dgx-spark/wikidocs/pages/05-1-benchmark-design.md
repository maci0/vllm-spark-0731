# 05-1. 벤치마크 설계 상세

상태: 리서치 기반 초안

DGX Spark 벤치마크의 첫 규칙은 “tok/s 하나로 순위를 만들지 않는다”는 것이다. 단일 사용자 decode, 긴 prompt prefill, 여러 요청의 aggregate throughput, tool-call 성공률, 장시간 안정성은 서로 다른 질문에 답한다.

## 3분 이해 (ELI5)

벤치마크는 같은 운동장에서 같은 규칙으로 경주하는 일이다.

```text
같은 모델·prompt·context·runtime
              ↓
prefill / decode / 품질 / 안정성 측정
```

입력 길이와 요청 수가 다르면 다른 경주다.

## 7.1 먼저 결과의 종류를 이름 붙인다

| 이름 | 측정 질문 |
|---|---|
| TTFT | 첫 token이 얼마나 빨리 나오는가 |
| prefill | 입력 context를 얼마나 빨리 처리하는가 |
| decode | 생성 중 token을 얼마나 빠르게 만드는가 |
| ITL/TPOT | token 사이 지연이 일정한가 |
| aggregate | 여러 요청이 합쳐서 얼마나 처리되는가 |
| capacity | 지정 context·동시성에서 메모리에 올라가는가 |
| quality | 답·코드·JSON·tool이 맞는가 |
| stability | soak 동안 error·OOM·hang이 없는가 |

`prefill 1024 tok/s`와 `decode 47 tok/s`는 대체 관계가 아니다. DeepSeek one-Spark의 370K 시험처럼 긴 context에서는 초기 prefill과 전체 요청의 실효 prefill이 달라질 수 있다.

## 7.2 benchmark schema

각 실행 결과는 다음 스키마로 저장한다.

```text
run_id:
date:
hardware:
node_count:
topology:
os:
kernel:
driver:
cuda:
nccl:
container_image:
container_digest:
runtime:
runtime_commit:
model_repo:
model_revision:
tokenizer_revision:
total_parameters:
active_parameters:
weight_quant:
weight_size_gib:
kv_dtype:
max_model_len:
max_num_seqs:
gpu_memory_utilization:
tp:
pp:
dp:
attention_backend:
speculative_method:
draft_model:
draft_tokens:
acceptance_rate:
prompt_tokens:
output_tokens:
thinking:
tool_loop:
vision:
warmups:
trials:
ttft_p50_ms:
ttft_p95_ms:
prefill_tok_s:
decode_tok_s:
itl_p50_ms:
aggregate_tok_s:
memory_peak_gib:
temperature_peak_c:
power_idle_w:
power_peak_w:
clock_cap_mhz:
clock_control:
power_measurement_domain: gpu_rail / wall_ac / unknown
power_wall_w:
errors:
timeouts:
quality_pass:
tool_valid_rate:
notes:
```

공개 레시피에서 값을 확인하지 못한 칸은 추정하지 말고 `unknown`으로 둔다.

## 7.3 단계별 테스트

### Level 0 — load

이 단계의 목적은 weight와 runtime이 메모리에 올라오는지 확인하는 데 있다.

```bash
curl -sS http://127.0.0.1:PORT/v1/models
nvidia-smi
free -h
```

이 단계가 PASS여도 생성이나 tool calling까지 지원한다는 뜻은 아니다.

### Level 1 — generate

한국어 단문, 코드, JSON, 멀티턴 marker, thinking on/off를 각각 보낸다. 응답 본문을 눈으로만 확인하지 말고 JSON parser·Python syntax check·marker exact match로 판정한다.

현재 저장소의 [qwen38_smoke.py](https://github.com/recrack/oh-my-dgx-spark/blob/main/tests/qwen38_smoke.py)는 이 단계에 해당하는 기능 테스트와 4-way 요청을 포함한다.

### Level 2 — single-stream performance

입력과 출력을 고정한 뒤 c1에서 warmup과 측정 trial을 분리해서 실행한다.

권장 시작 프로토콜:

1. 서버 시작 직후가 아닌 warmup 3회
2. 동일한 prompt·output budget으로 측정 5회 이상
3. 첫 요청과 steady-state를 별도 표시
4. TTFT·prefill·decode·ITL을 별도 기록
5. generation이 짧으면 `short output`으로 표시

이 프로토콜은 이 책에서 제안하는 재현 절차다. 공개 글의 측정 횟수를 이 절차에 맞춰 소급해 바꾸지는 않는다.

### Level 3 — concurrency

c1, c2, c4, c8을 별도 행으로 측정한다.

| 항목 | 기록할 것 |
|---|---|
| concurrency | 동시에 보낸 요청 수 |
| request rate | open-loop인지 closed-loop인지 |
| prompt | 동일 prefix인지 독립 prompt인지 |
| cache | prefix cache hit/miss |
| output | 요청별 token 수와 총 token 수 |
| result | p50/p95 TTFT, decode, aggregate, error |

`c8 aggregate 180 tok/s`를 단일 사용자 속도 180 tok/s로 바꾸어 쓰지 않는다. 반대로 c1 47 tok/s만으로 8개 agent가 같은 속도로 동작한다고 기록하지도 않는다.

### Level 4 — long context

context 길이를 `8K → 32K → 128K → 256K → recipe ceiling`으로 나누어 시험한다. 각 단계에서 다음 항목을 기록한다.

- tokenizer 기준 실제 prompt token 수
- needle 위치: beginning/middle/end
- exact recall 여부
- TTFT와 effective prefill
- decode와 ITL
- memory peak와 free memory
- timeout·preemption·server health

한 개의 needle을 370K에서 회수한 결과는 강한 stress test다. 그러나 자연어 장문 요약이나 추론 품질을 종합적으로 인증하는 결과는 아니다. 여러 위치와 distractor를 포함한 별도 품질 테스트가 필요하다.

### Level 5 — tool·agent

tool benchmark는 raw generation benchmark와 분리해서 실행한다.

```text
tool schema → model tool call → JSON arguments parser
           → mock tool 실행 → tool result 주입
           → 다음 call 또는 최종 답변
```

최소 기록 항목:

- valid tool call rate
- valid arguments rate
- unknown tool rate
- tool error 후 recovery rate
- 최대 loop 횟수
- context 증가량
- timeout·중단·잘못된 파일/명령 요청

NVIDIA 포럼의 Tool Eval Bench와 Toolery는 각각 다른 scenario 수와 assertion 체계를 사용한다. 점수는 도구 세트·seed·trial·parser를 함께 적을 때만 비교한다.

### Level 6 — soak

장시간 안정성은 긴 요청 한 번만으로 판단하지 않는다.

```text
짧은 요청 반복
  + 긴 prompt 주기적 삽입
  + c1/c4 혼합
  + tool error와 재시도
  + memory/temperature/health polling
```

기록할 것:

- 시작·종료 시간
- 총 요청 수와 성공률
- timeout·restart·preemption
- memory peak와 memory 회수 여부
- temperature/power
- endpoint가 마지막 요청에도 응답했는가

## 7.4 품질 하니스

책의 기본 prompt set은 모델의 능력을 새로 인증하기 위한 것이 아니다. 양자화나 엔진을 바꿨을 때 기능이 깨졌는지 확인하기 위한 것이다.

| 범주 | 예시 판정 |
|---|---|
| 한국어 | 요구된 언어·길이·핵심 사실 |
| 코드 | syntax compile, 지정 API 사용 |
| JSON | `json.loads`와 schema |
| 멀티턴 | marker exact match |
| reasoning | thinking on/off 필드와 최종 답 분리 |
| vision | 이미지에 실제로 보이는 내용만 포함 |
| tool | 이름·필수 인자·타입·재시도 |
| 장문 | 다중 needle 위치와 distractor |

temperature 0이라고 해서 출력이 완전히 결정적인 것은 아니다. backend·speculation·parallel scheduling이 달라지면 출력도 달라질 수 있으므로 seed와 반복 수를 기록한다.

## 7.5 공정한 엔진·모델 비교표

| 비교 | 고정해야 하는 것 | 바꿔도 되는 것 |
|---|---|---|
| BF16 vs NVFP4 | model revision·prompt·output·context·concurrency | weight quant |
| vLLM vs SGLang | model·quant·KV·prompt·hardware | engine/image/commit |
| spec off vs on | target model·prompt·output | draft와 spec flag |
| 1대 vs 2대 | model·workload·quality harness | node count/topology/TP |
| c1 vs c4 | model·prompt corpus·output budget | concurrency |

DeepSeek one-Spark EXL3와 two-Spark FP8 TP=2는 구매 판단에는 유용한 대비다. 그러나 순수 엔진 benchmark에서 직접 비교할 한 쌍은 아니다. 표 제목에는 “구성 비교”라고 쓰고 “속도 우승자”라고 쓰지 않는다.

### 7.5a clock cap A/B 실험

GB10에서 의도적으로 SM clock을 제한하면 decode와 prefill의 변화가 다를 수 있다. 공개 `gb10-clock-cap` reference와 2× Spark c4 보고는 decode가 상대적으로 유지되는 동안 cold prefill과 compute-bound 작업이 더 느려질 수 있음을 보여준다. 다음 조건을 고정한다.

1. stock을 sweep 시작과 끝에 각각 측정해 열·전력 drift를 확인한다.
2. 2대 이상이면 모든 노드에 같은 cap을 적용하고 host별 실제 clock을 기록한다.
3. `c1`과 `c4`, decode와 cold prefill을 별도 행으로 남긴다.
4. `nvidia-smi`/`nvtop`의 GPU rail 값과 벽면 AC meter 값을 별도 필드에 저장한다.
5. `2200MHz` 결과를 모든 모델·실내 온도·concurrency에 대한 최적값으로 일반화하지 않는다.

적용·rollback 명령 자체도 결과에 포함한다. `sudo nvidia-smi -lgc 0,2200`과 `sudo nvidia-smi -rgc`는 실험용 예시이며, 장비의 지원 범위와 권한을 먼저 확인한다.

## 7.6 결과 보고 예시

```text
model: DeepSeek V4 Flash 0731
hardware: 1x DGX Spark GB10/SM121/128 GiB
engine: SparkInfer
quant: EXL3 3.0 bpw / REAP-K216
kv: native NVFP4 / nvfp4_ds_mla
spec: DSpark K5 + K64
context: max_model_len=384000
concurrency: c1
thinking: off
workload: structured decode
decode: 44–47 tok/s (community recipe report)
prefill: initial ~1024 tok/s; long-context effective ~625 tok/s report
kv_pool: ~439,622 tokens (cold-boot recipe report)
needle: 370,104 exact recall
quality_status: not full-FP8/full-expert certification
reproduction_status: pending on local machine
```

이 형식은 숫자보다 측정 조건을 먼저 보여준다. 우리 장비에서 실제로 재현하기 전에는 `community recipe report`와 `reproduction_status: pending`을 지우지 않는다.

## 7.7 실패도 결과다

benchmark에서 다음을 숨기지 않는다.

- 모델은 올라갔지만 첫 생성에서 반복 문자가 나옴
- context ceiling에서 OOM
- tool parser가 400 반환
- c4에서 KV pool 부족
- RDMA가 아닌 socket fallback
- 긴 generation 중 NCCL hang
- 온도 상승 후 shutdown
- warmup 후에만 속도가 나옴

실패 원인을 확정하지 못했다면 `issue` 또는 `unresolved`로 남긴다. 임시 workaround를 공식 해법처럼 기록하지 않는다.

## 이 장의 검증 체크리스트

- [ ] load/generate/serve/benchmark/tool/agent/long-context 상태를 구분했다.
- [ ] TTFT·prefill·decode·aggregate를 별도 측정했다.
- [ ] c1/c4/c8을 따로 기록했다.
- [ ] prompt와 output token 수를 tokenizer/usage 기준으로 확인했다.
- [ ] long context는 capacity와 quality를 분리했다.
- [ ] tool benchmark에 parser·schema·seed·trial을 기록했다.
- [ ] warmup·trial·p50/p95를 기록했다.
- [ ] memory·temperature·power·error·restart를 결과에 포함했다.
- [ ] 공개 recipe 숫자와 로컬 재현 결과를 구분했다.

## 아직 모르는 것

- 통일된 prompt corpus에서 모든 Spark 모델을 비교한 결과
- one-Spark DeepSeek EXL3와 two-Spark FP8의 동일 quality harness 결과
- 공개 tool benchmark 점수와 실제 장기 agent 성공률의 상관관계
- context가 커질 때 모델별 perplexity·retrieval·tool recovery 변화
