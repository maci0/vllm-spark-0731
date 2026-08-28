# 04-1. vLLM·SGLang·llama.cpp·SparkInfer 선택

상태: 리서치 기반 초안

DGX Spark에서 엔진은 모델 이름에 덧붙이는 부가 옵션이 아니다. 같은 checkpoint라도 엔진·커널·KV backend·speculative decoder·컨테이너가 달라지면 속도, context capacity, tool calling, 안정성이 모두 달라진다.

## 3분 이해 (ELI5)

vLLM·SGLang·llama.cpp는 같은 모델을 움직이는 서로 다른 엔진이다.

```text
같은 모델 + 엔진 A → 기능·속도·안정성 A
같은 모델 + 엔진 B → 기능·속도·안정성 B
```

엔진 이름만 바꾸는 일이 아니라, 모델·커널·KV·parser 조합을 바꾸는 일이다.

## 5.1 먼저 목적을 고른다

| 우선 목적 | 첫 엔진 후보 | 이유 |
|---|---|---|
| OpenAI-compatible API와 기능 확인 | vLLM | 모델 지원표·서버 기능·tool parser 기준선이 넓음 |
| Qwen3.8 단일/듀얼의 speculative decode | SGLang | DFlash2/DSpark와 prefix/radix cache 레시피가 공개됨 |
| GGUF를 빠르게 시험 | llama.cpp | 단일 노드 설치와 파일 교체가 단순함 |
| DeepSeek V4 Flash 0731 단일 Spark | SparkInfer + EXL3 | 전용 커널·낮은 bit weight·DSpark 조합을 사용함 |
| 여러 노드의 대형 모델 | vLLM 또는 SGLang recipe | TP/PP/RDMA를 모델별로 고정해야 함 |

이 표는 품질 순위를 정한 표가 아니다. 첫 실행을 어느 엔진에서 시작할지 결정하기 위한 기준표다.

## 5.2 엔진별 성격

| 엔진 | 강점 | 약점·위험 | 책에서 맡는 역할 |
|---|---|---|---|
| vLLM | 넓은 모델·API 생태계, continuous batching, 공식 Spark playbook, TP 경로 | GB10/SM121은 버전·fork·kernel 의존성이 있고 모델별 parser가 다름 | 기능 baseline과 다중 노드 기준선 |
| SGLang | prefix/radix cache, structured output, DFlash2/DSpark, agent workload | Docker memory cap·RDMA passthrough·sm_121 커널 조건을 함께 맞춰야 함 | Qwen3.8 속도·동시성 최적화 |
| llama.cpp | GGUF 선택 폭, 단일 사용자 실험이 쉽고 설치 경로가 명확함 | 복잡한 multi-node·특수 MoE·tool parser는 별도 확인 필요 | GGUF/양자화 비교 기준선 |
| SparkInfer/EXL3 | DeepSeek 단일 Spark에 맞춘 낮은 bit와 전용 backend | 파생 quant·전용 런타임·quality/parser 검증 부담 | DeepSeek 장문 단일 스트림 실험 |

포럼의 Qwen3.8 비교는 같은 GB10에서 llama.cpp 약 27 tok/s, vLLM NVFP4+MTP 약 24.5 tok/s, SGLang+NVFP4+DSpark 약 34–38 tok/s를 보고했다. 이 숫자는 엔진의 보편적인 순위가 아니라 특정 model revision·컨테이너·프롬프트·speculative 설정에서 측정한 결과다. 결과를 인용할 때는 [포럼 원문과 조건](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)을 함께 기록한다.

## 5.3 선택 순서

다음 질문에 `예`라고 답할 수 있는 경로를 선택한다.

1. 모델이 해당 엔진의 현재 지원표에 있는가?
2. GB10/SM121용 image·commit·kernel 조건이 공개되어 있는가?
3. endpoint와 tokenizer가 먼저 smoke test를 통과하는가?
4. 필요한 context에서 KV가 남는가?
5. tool parser와 structured output이 실제 요청 형식과 맞는가?
6. 장시간 실행에서 memory와 temperature가 유지되는가?

하나라도 답하지 못하면 다음 최적화 단계로 넘어가지 말고 `unknown`으로 기록한다.

### 실전 분기

```text
기능/API/모델 적재부터 확인해야 하는가?
  └─ 예 → vLLM BF16 또는 공식 지원 quant baseline

Qwen3.8의 code·agent throughput이 우선인가?
  └─ 예 → SGLang + DFlash2/DSpark recipe

GGUF 파일과 단일 사용자 비교가 우선인가?
  └─ 예 → llama.cpp

DeepSeek V4 Flash 0731을 한 대에서 긴 세션으로 시험하는가?
  └─ 예 → SparkInfer + EXL3 recipe, 단 full-FP8과 분리 기록

2대 이상에서 모델을 분산하는가?
  └─ 예 → 먼저 NCCL/RDMA health, 그 다음 모델별 vLLM/SGLang recipe
```

## 5.4 엔진을 비교할 때 고정할 값

엔진 이름만 바꾸는 비교는 다음 항목을 고정하지 않으면 의미가 약하다.

| 그룹 | 고정할 값 |
|---|---|
| hardware | Spark 수, GB10/SM121, unified memory, QSFP/switch |
| software | DGX OS, driver, CUDA, container image/digest, engine commit |
| model | 원본 repo, revision, tokenizer revision, model config |
| quant | BF16/FP8/NVFP4/AWQ/EXL3/GGUF와 checkpoint 파일 |
| KV | dtype, cache format, max context, memory fraction |
| speculation | MTP/DFlash/DSpark/EAGLE, draft model, draft token 수, acceptance |
| serving | TP/PP/DP, batch token limit, prefix cache, parser |
| workload | prompt tokens, output tokens, thinking, tool loop, image 여부 |

이 중 하나라도 다르면 결과 제목에 `조건 상이`라고 표시한다. 특히 vLLM BF16 baseline과 DeepSeek EXL3 3.0 bpw + DSpark 결과는 같은 모델 순위표에 바로 넣지 않는다.

## 5.5 기능 기준선과 속도 기준선을 분리한다

### 기능 기준선

기능 기준선의 목적은 다음 항목이 정상적으로 동작하는지 확인하는 데 있다.

- `/v1/models`가 예상 model id를 반환하는가
- 한국어와 코드가 비어 있지 않은가
- JSON이 실제 parser로 읽히는가
- 멀티턴 marker를 회수하는가
- thinking on/off가 요청대로 동작하는가
- tool parser가 valid `tool_calls`와 arguments를 내는가
- 이미지·긴 context가 endpoint에서 오류 없이 처리되는가

현재 저장소의 [Qwen3.8 smoke test](https://github.com/recrack/oh-my-dgx-spark/blob/main/tests/qwen38_smoke.py)는 기능 기준선으로 사용할 수 있다. 이 테스트가 PASS를 반환해도 해당 모델의 전반적인 품질이나 benchmark score가 검증되었다는 뜻은 아니다.

### 속도 기준선

속도 기준선은 긴 출력과 고정된 prompt를 사용하고, 다음 지표를 따로 측정한다.

- TTFT: 요청 시작부터 첫 출력 token까지
- prefill throughput: 입력 처리 속도
- decode throughput: 생성 중 token 속도
- aggregate throughput: 여러 요청이 처리한 총 token 속도
- ITL/TPOT: 생성 token 간 지연
- p50/p95: 반복 실행의 분포

짧은 출력 요청 하나의 end-to-end 시간은 이 지표들을 대신할 수 없다.

## 5.6 Tool calling의 엔진 차이

서버가 OpenAI-compatible이라고 해서 tool calling이 자동으로 지원되는 것은 아니다. 모델별 parser, chat template, reasoning mode, auto tool choice 설정이 모두 맞아야 한다.

현재 로컬 Qwen3.8 smoke test의 tool 요청은 parser를 설정하지 않아 vLLM이 400을 반환했다. 이는 모델 능력이 부족해서가 아니라 serving configuration이 갖춰지지 않았기 때문이다. 서버를 다시 시작할 때는 모델에 맞는 parser를 별도 포트에서 검증한다.

```text
tool 호출 결과를 평가할 때 기록할 것:
  parser:
  auto_tool_choice:
  tool schema:
  reasoning on/off:
  valid JSON arguments:
  unknown tool rate:
  recovery after tool error:
```

## 5.7 단일 Spark 모델별 시작점

| 모델/목적 | 시작점 | 다음 단계 |
|---|---|---|
| Qwen3.8-27B 기능 확인 | BF16 vLLM | NVFP4 SGLang/DFlash2와 동일 prompt 비교 |
| Qwen3.8-27B 빠른 코드 agent | SGLang + DFlash2/DSpark | c1/c4/c8, tool-eval, 장시간 soak |
| Qwen3.6-35B-A3B | NVIDIA 공식 vLLM recipe | parser·MTP·262K context 검증 |
| Qwen3.5-122B | 공개 INT4/AutoRound recipe | quality·KV·MTP·재시작 검증 |
| DeepSeek V4 Flash 0731 | MiaAI-Lab one-Spark EXL3/SparkInfer | full-FP8 2대 TP=2와 quality/long-context 비교 |
| GPT-OSS-120B | 공식 llama.cpp 또는 지원 recipe | tool/일반 추론과 Qwen 계열 비교 |

DeepSeek one-Spark recipe는 `MAX_NUM_SEQS=1`, 384K 설정, 약 440K KV pool, 구조화 decode 44–47 tok/s를 보고한다. 다만 EXL3 3.0 bpw·REAP-K216·DSpark 조건에서 얻은 결과다. [원문](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)을 그대로 재현하거나 변형할 때는 이 조건을 모델 이름 옆에 함께 적는다.

## 이 장의 검증 체크리스트

- [ ] 목적에 맞는 엔진 후보를 하나 골랐다.
- [ ] 엔진·컨테이너·model revision·tokenizer revision을 기록했다.
- [ ] 기능 baseline과 속도 baseline을 분리했다.
- [ ] prefill·decode·aggregate를 서로 다른 열에 기록한다.
- [ ] speculative decoder와 draft token 수를 기록한다.
- [ ] tool parser와 request format을 별도로 검증한다.
- [ ] c1과 c4/c8 결과를 같은 숫자로 합치지 않았다.
- [ ] SparkInfer/EXL3 결과를 full-FP8/full-expert 결과로 표현하지 않았다.

## 아직 모르는 것

- 동일한 prompt set에서 vLLM·SGLang·llama.cpp의 품질 차이
- GB10 driver·kernel 업데이트별로 엔진 성능이 얼마나 변하는가
- Qwen3.8과 DeepSeek의 tool loop 성공률을 같은 harness로 비교한 결과
- SparkInfer의 장시간 c1 안정성과 c4/c8의 실사용 headroom
