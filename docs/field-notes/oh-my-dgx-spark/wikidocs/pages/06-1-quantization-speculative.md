# 06-1. 양자화·KV cache·speculative decoding

상태: 리서치 기반 초안

DGX Spark에서 모델을 고를 때 `4-bit`, `FP8`, `NVFP4`, `EXL3`, `MTP`라는 용어만 보고 속도를 예측해서는 안 된다. weight의 정밀도, KV cache의 정밀도, backend kernel, context, draft acceptance는 서로 다른 축에서 결과에 영향을 준다.

## 3분 이해 (ELI5)

양자화는 큰 가방에 물건을 더 효율적으로 넣는 방법이다.

```text
weight 양자화 → 모델 책의 저장 공간을 줄임
KV dtype      → 대화 메모의 공간을 줄임
speculative   → 작은 초안 모델이 큰 모델의 일을 미리 제안
```

가방이 가벼워져도 내용이 같은지와 손잡이가 튼튼한지는 별도로 확인한다.

## 6.1 양자화는 무엇을 줄이는가

먼저 weight와 KV를 분리해서 생각한다.

```text
weight quantization → 모델 파라미터 저장·로드에 필요한 메모리와 일부 연산 비용
KV cache dtype      → 이미 처리한 context를 보관하는 메모리와 attention 비용
workspace/backend   → 실제 엔진이 사용하는 추가 메모리와 kernel 경로
```

weight를 4-bit로 저장했다고 해서 KV가 자동으로 4-bit가 되는 것은 아니다. 모델 표에는 항상 `weight quant`와 `KV dtype`를 따로 기록한다.

## 6.2 주요 포맷의 역할

| 포맷 | 대략적인 성격 | 장점 | 주의점 |
|---|---|---|---|
| BF16 | 높은 정밀도의 기준선 | 품질·기능 디버깅이 쉽고 변환 오차가 적음 | 큰 모델은 weight만으로 Spark 메모리를 압박 |
| FP8 | 8-bit floating-point 계열 | 지원 backend에서 품질과 메모리의 균형 | scale·kernel·KV 설정을 함께 고정 |
| NVFP4 | NVIDIA Blackwell 계열 4-bit 경로 | weight/KV를 크게 줄일 수 있고 전용 kernel 활용 가능 | 지원 모델·engine·driver에 따라 결과가 크게 다름 |
| INT4/AWQ/AutoRound | 4-bit weight 계열 | 공개 checkpoint와 도구가 다양함 | quant calibration, Marlin/kernel, 품질을 모델별 검증 |
| GGUF Q4/Q5 | llama.cpp 계열 파일 포맷 | 파일 교체와 단일 사용자 실험이 단순 | 다른 engine의 NVFP4·EXL3와 같은 품질 축이 아님 |
| EXL3 3.0 bpw | ExLlama 계열의 매우 낮은 bit 경로 | DeepSeek one-Spark에서 weight를 크게 줄임 | bpw 숫자만으로 품질·메모리·호환성을 판단하지 않음 |

`bpw`는 파일 전체의 평균 bit-per-weight에 가까운 표현이다. 모든 tensor가 정확히 같은 bit로 저장된다는 뜻은 아니다. non-routed tensor와 중요한 layer는 더 높은 정밀도로 저장될 수 있다.

## 6.3 unified memory 예산

모델을 적재할 수 있는지 빠르게 판단할 때 다음 식을 쓴다.

```text
필요 메모리
≈ weight 파일을 풀어놓은 실제 크기
 + KV cache(context × concurrency × layer/head 구조 × KV dtype)
 + CUDA graph/workspace
 + tokenizer·prefill·vision buffer
 + runtime·OS·통신 버퍼
```

이 식은 정확한 allocator 계산식이 아니라 필요한 항목을 빠뜨리지 않기 위한 체크리스트다. 특히 MoE는 total parameter와 active parameter가 다르므로, “활성 파라미터가 작다”는 설명만으로 weight까지 작다고 판단하지 않는다.

긴 context에서 우선 확인할 순서는 다음과 같다.

1. 서버가 실제로 선언한 `max_model_len`
2. cold boot에서 확보된 KV pool
3. `MAX_NUM_SEQS` 또는 동시 요청 수
4. KV dtype와 backend
5. CUDA graph/workspace 이후 남는 memory
6. 실제 prompt tokenizer token 수

모델 README의 1M/384K는 capacity ceiling일 수 있다. 이 값만으로 전체 context에서 유지되는 decode 속도나 retrieval 품질을 보장할 수는 없다.

## 6.4 DGX Spark에서 자주 생기는 양자화 착각

### “NVFP4면 무조건 FP8보다 빠르다”

그렇지 않다. 초기 GB10/vLLM 실측에서는 backend와 동시성에 따라 AWQ 4-bit가 NVFP4보다 빠른 사례도 보고됐다. NVFP4의 장점은 모델·kernel·runtime이 모두 맞을 때 나타난다.

### “weight가 올라갔으니 quant가 성공했다”

반복 문자, 빈 출력, 잘못된 config type, vision tower 누락, tokenizer mismatch가 나중에 나타날 수 있다. `loads` 이후에는 생성·JSON·멀티턴·vision·tool을 각각 확인한다.

### “Q4_K_M과 EXL3 3.0 bpw는 같은 3~4 bit다”

파일 포맷, calibration, 중요 tensor의 precision, backend kernel, KV 방식이 모두 다르다. 품질이 “비슷하다”고 기록하려면 동일 prompt set이나 별도의 평가 결과가 필요하다.

## 6.5 speculative decoding의 원리

speculative decoding은 작은 draft 모델이나 draft head가 여러 token을 제안하고, target 모델이 이를 한 번에 검증하도록 돕는 방식이다.

```text
target가 한 token씩 생성
        ↓
draft가 여러 token 제안
        ↓
target이 제안을 검증하고 맞는 부분을 수용
        ↓
수용률이 높으면 decode 단계 수 감소
```

대표 이름은 다음과 같다.

| 방식 | 책에서 기록할 항목 |
|---|---|
| MTP | model-integrated draft head, speculative token 수, acceptance |
| DFlash | draft model·token 수·verification backend |
| DSpark | Spark용 draft/검증 경로, K5/K64 등 recipe 설정 |
| EAGLE | draft model·accept length·모델별 patch |

실제 성능은 다음 관계에 더 가깝다.

```text
실효 decode 속도
= target 계산 비용
  + draft 계산 비용
  + 통신·검증 비용
  + workload별 acceptance rate
```

그러므로 draft token 수를 늘린다고 항상 빨라지는 것은 아니다. acceptance가 낮거나 검증 비용이 큰 workload에서는 오히려 느려질 수 있다.

## 6.6 workload별 speculative 효과

| workload | 예상 위험 | 측정 방법 |
|---|---|---|
| 반복적인 코드·구조화 출력 | draft가 잘 맞을 가능성 | code, JSON, tool arguments를 별도 측정 |
| 자유로운 prose | acceptance가 낮아질 수 있음 | speculation on/off를 같은 prompt로 비교 |
| thinking | 긴 reasoning과 형식 변화 | thinking on/off, reasoning token 포함 여부 기록 |
| 긴 context | prefill과 KV가 병목 | decode와 prefill을 분리하고 memory peak 기록 |
| tool loop | 한 번의 답보다 실패 복구가 중요 | valid call·argument·tool error recovery 평가 |

DeepSeek 단일 Spark recipe는 DSpark K5/K64 draft와 c1 deep-context를 사용한다. README의 구조화 decode 44–47 tok/s는 이 조건에서 얻은 결과다. 이를 creative prose·c4/c8·full-expert 품질로 그대로 확대해석하지 않는다.

## 6.7 비교 프로필을 고정하는 표

| 프로필 | weight | KV | speculation | context/concurrency | 의미 |
|---|---|---|---|---|---|
| A | BF16 | BF16/default | off | 32K/c1 | 기능·품질 baseline |
| B | FP8/NVFP4 | FP8/NVFP4 | off | 같은 조건 | weight/KV 포맷 비교 |
| C | 같은 weight | 같은 KV | MTP/DFlash/DSpark on | 같은 조건 | speculation 순수 효과 |
| D | 같은 optimized recipe | recipe KV | on | 256K/1M·c1/c4 | 실사용 capacity와 trade-off |

A→B→C→D 순서로 바꾸면 각 변화의 원인을 추적하기 쉽다. quant·KV·draft·context를 한 번에 모두 바꾸면 빠른 숫자는 얻을 수 있어도 원인은 확인하기 어렵다.

## 6.8 DeepSeek one-Spark와 two-Spark를 기록하는 방식

### One-Spark EXL3 profile

- 1× GB10/SM121, TP=1
- EXL3 3.0 bpw, REAP-K216
- SparkInfer + DSpark K5/K64
- `MAX_NUM_SEQS=1`, `MAX_MODEL_LEN=384000`
- native NVFP4 KV, 약 439K pool 보고
- 구조화 decode 44–47 tok/s 보고

이는 “한 대에서 긴 단일 세션을 시험할 수 있다”는 의미다. 공식 full-FP8/full-expert checkpoint와 같은 품질 profile로 해석하지 않는다. [원문 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)

### Two-Spark TP=2 profile

2대 DeepSeek 경로는 원본에 가까운 FP8/DSpark checkpoint, TP=2, 256K 또는 1M context를 목표로 한다. 이 구성은 모델 품질·KV·동시성의 headroom을 넓히는 대신 QSFP/RoCE·NCCL·UCX·두 노드 버전 일치를 요구한다. one-Spark EXL3와 속도를 직접 비교하기 전에 모델·quant·context·concurrency가 다른 점을 표에 표시한다.

## 6.9 품질 확인 최소 세트

양자화나 speculation을 바꿀 때는 다음 테스트를 반복한다.

- 한국어 사실 질문 10개
- Python 함수 10개와 syntax check
- JSON schema 10개와 parser validation
- 멀티턴 marker 5개
- tool call schema 10개와 arguments validation
- context beginning/middle/end marker
- thinking on/off 각 5개
- 이미지가 있는 모델은 동일 이미지 세트

평균 점수 하나만 남기지 말고 깨진 출력·반복·timeout·invalid tool call을 별도로 기록한다.

## 이 장의 검증 체크리스트

- [ ] weight quant와 KV dtype을 별도로 기록했다.
- [ ] BF16 또는 FP8 기능 baseline을 먼저 남겼다.
- [ ] quant checkpoint의 model config·tokenizer·revision을 고정했다.
- [ ] speculative token 수와 acceptance rate를 기록했다.
- [ ] c1 decode와 c4/c8 aggregate를 분리했다.
- [ ] prefill과 decode를 분리했다.
- [ ] 1M/384K capacity를 품질 인증으로 과대해석하지 않았다.
- [ ] EXL3·GGUF·NVFP4를 bit 수만으로 동일시하지 않았다.

## 아직 모르는 것

- 동일 prompt set에서 Q4_K_M·EXL3·NVFP4의 품질 delta
- DeepSeek REAP-K216과 full-expert의 실제 tool/코드 차이
- DSpark K5와 K64의 acceptance·전력·온도 trade-off
- 370K needle 통과 이후 자연어 장문 추론 품질
