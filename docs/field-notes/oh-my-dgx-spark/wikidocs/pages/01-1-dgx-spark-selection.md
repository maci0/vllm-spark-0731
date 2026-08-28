# 01-1. DGX Spark 선택 기준 상세

상태: 초안

이 장은 “DGX Spark가 좋은가?”를 묻는 장이 아니다. **내 작업과 예산에서 이 장비를 선택할 이유가 있는가?**를 먼저 판단하는 장이다.

> DGX Spark는 “작은 데이터센터 GPU”라기보다, 큰 unified memory와 CUDA 생태계를 작은 상자에 담아 로컬 모델·에이전트·실험을 장시간 운영하는 장비다.

## 3분 이해 (ELI5)

DGX Spark는 큰 책상 하나를 가진 작은 작업실과 비슷하다. 모델, 대화 기록, 운영체제와 도구가 같은 책상 위에서 자원을 나눠 쓴다. 책상이 크다는 이유만으로 모든 작업이 빠른 것은 아니다.

```text
작업 요청 → 모델 weight·KV cache·workspace → 답변
                     ↘ 온도·전력·네트워크
```

책상은 넓지만 무한하지 않으므로, 모델이 올라간다는 사실과 오래 안정적으로 일한다는 사실을 따로 확인해야 한다.

## 1.1 먼저 답할 질문

| 질문 | 확인할 관점 | 다음 장 |
|---|---|---|
| 모델을 크게 올리는 것이 목적입니까? | capacity와 실제 decode 속도를 분리 | 02장 |
| 빠른 단일 사용자 응답이 목적입니까? | memory bandwidth·quant·speculative decoding | 03-4 → 05-1 |
| 여러 사람이 동시에 사용할 예정입니까? | aggregate throughput·동시성·노드 수 | 07-1 → 07-4 |
| CUDA 모델 개발과 fine-tuning이 목적입니까? | aarch64·CUDA·컨테이너·작업 시간 | 03-1 → 10-1 |
| 개인 데이터와 endpoint를 로컬에 두려는 것입니까? | 운영·보안·백업·복구 책임 | 08-1 → 09-1 |

커뮤니티에서도 “128GB에 모델이 들어가는가”와 “실제로 빠른가”를 별개의 질문으로 다룬다. 한 대에서 여러 모델을 동시에 적재하면 unified memory를 두고 경쟁할 수 있다는 사용담도 있으므로, 처음부터 여러 모델 상주를 기본 설계로 두지 않는다. 이는 커뮤니티 경험에 기반한 운영 주의사항이며, 모델·runtime별 직접 검증이 필요하다. [DGX Spark setup 질문](https://www.reddit.com/r/LocalLLM/comments/1uc4amk/dgx_spark_setup/)

## 1.2 먼저 기대치를 정한다

DGX Spark를 선택하는 이유는 사용 목적에 따라 달라진다.

| 목적 | 중요한 것 | 첫 후보 |
|---|---|---|
| 빠른 코딩·일반 챗 | 단일 스트림 decode, 낮은 TTFT, tool parser | Qwen3.6/Qwen3.8 계열 |
| 큰 supervisor 모델 | 모델 품질, KV 여유, 긴 context | Qwen3.5-122B, DeepSeek V4 Flash 2대 TP=2 또는 최신 1대 EXL3 recipe |
| 여러 에이전트 동시 실행 | aggregate throughput, prefix cache, 안정성 | 작은 모델 여러 개 또는 2대 이상 |
| 멀티모달·긴 문서 | vision/audio/video workspace와 KV | 전용 multimodal recipe |
| 연구·튜닝 | CUDA, kernel, NCCL, quant 실험성 | Spark + 고정된 재현 환경 |

다음 요구가 더 중요하다면 DGX Spark를 첫 선택으로 삼지 않는 편이 낫다.

- 최고 단일 사용자 속도가 필요하다.
- 게임·범용 데스크톱·대형 저장장치가 주목적이다.
- 드라이버·컨테이너·모델 커밋을 고정할 수 없다.
- 냉각·전원·네트워크 장애를 직접 진단할 수 없다.

## 1.3 한 대, 두 대, 네 대의 차이

포럼 자료와 공개 레시피를 함께 보면, 노드 수가 늘어날 때 장비의 역할도 달라진다.

| 노드 | 핵심 변화 | 추천 질문 |
|---:|---|---|
| 1 | 빠른 모델과 안정적인 단일 서버를 선택 | “이 모델이 내 목적에 충분한가?” |
| 2 | 모델 크기·context·동시성이 함께 늘어남 | “TP=2인가, 두 서버로 나눌 것인가?” |
| 3 | 모델별 sharding 제약과 topology가 전면에 등장 | “PP/DP가 TP보다 낫지 않은가?” |
| 4 | switch/RDMA 기반 TP=4가 현실적인 확장점 | “단일 대형 모델인가, 독립 서비스인가?” |
| 8 | 개인용 장비보다 클러스터 운영 문제가 커짐 | “냉각·전력·장애 격리를 설계했는가?” |

[NVIDIA Developer Forum DGX Spark/GB10 카테고리](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)에는 단일 모델 레시피부터 8× GB10 클러스터까지 다양한 자료가 있다. 다만 자료마다 모델·엔진·컨텍스트·동시성 조건이 다르다. 이 책은 노드 수를 단순한 성능 순위가 아니라 사용 목적이 바뀌는 단계로 설명한다.

## 1.4 “지원”이라는 말의 네 가지 의미

모델 카드나 포럼에서 `지원한다`는 표현을 보더라도, 다음 상태를 나누어 확인해야 한다.

| 상태 | 확인한 것 |
|---|---|
| `loads` | weight와 런타임이 메모리에 올라옴 |
| `generates` | 기본 텍스트 prompt에 정상 답변 |
| `serves` | OpenAI-compatible endpoint가 반복 요청에 응답 |
| `benchmarked` | 입력·출력·동시성·버전이 고정된 측정 |
| `tool-tested` | parser와 tool arguments가 정상 |
| `agent-tested` | 다단계 tool loop·실패 복구·context 압박까지 확인 |

예를 들어 1M context를 할당할 수 있다고 해서 1M 문맥에서 retrieval 품질과 decode 속도까지 검증된 것은 아니다. 이 차이를 구분하는 것이 이 책을 읽는 가장 중요한 기준이다.

## 1.5 모델을 고르는 순서

모델 이름부터 정하지 말고 다음 순서로 선택 범위를 좁힌다.

1. 필요한 modality를 정한다: text, code, image, audio, video.
2. worker와 supervisor를 나눈다.
3. 최대 context와 동시 요청을 정한다.
4. quality와 latency 중 우선순위를 정한다.
5. 모델이 들어갈 memory budget을 계산한다.
6. 해당 모델의 런타임·quant·speculative recipe를 찾는다.
7. 마지막에 포럼 속도 수치와 우리 장비 결과를 비교한다.

### 예시: 코딩 에이전트 한 대

목표가 “빠른 코드 수정과 구조화된 tool call”이라면 가장 큰 모델부터 선택하지 않는다. 먼저 Qwen3.6/Qwen3.8급 모델을 단일 Spark에서 `thinking off`, prefix caching, tool parser와 함께 검증한다. 긴 reasoning을 담당할 supervisor가 필요해진 뒤에 2대 DeepSeek 또는 큰 MoE를 추가로 검토한다.

### 예시: 긴 문서 supervisor 두 대

목표가 “긴 문서를 읽고 여러 번 도구를 호출하는 supervisor”라면 2대의 가치는 단일 요청 속도가 두 배가 되는 데 있지 않다. 256K/1M context, KV cache, 여러 agent의 동시성, 재시작 후 cache 복구를 함께 확인해야 한다. 다만 최신 [DeepSeek V4 Flash 0731 one-Spark EXL3 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)는 TP=1에서 `MAX_MODEL_LEN=384000`, 약 440K KV pool, 370K needle stress를 보고한다. 그러므로 “무조건 2대가 필요하다”고 결론 내리지 말고, 목적·품질 기준·동시성을 나누어 판단한다. [DeepSeek V4 Flash 2× recipe](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309)와 [Qwen3.8 2× 측정](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350)은 서로 다른 목적을 다루는 자료다. 단일 EXL3 수치는 full-FP8/full-expert와 동일한 품질이나 기능을 뜻하지 않으며, 단일 스트림·fresh boot 조건의 공개 recipe 결과로 취급한다. thinking off는 needle stress test의 조건으로만 기록한다. 자세한 판정은 [DeepSeek V4 Flash 0731 성능 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/deepseek-v4-flash-0731-performance-research-2026-08.md)를 따른다.

## 1.6 이 책에서 말하는 “실전”

실전 레시피는 설치 명령 하나로 완성되지 않는다. 다음 질문에 답할 수 있어야 한다.

- 서버가 재시작 후 다시 올라오는가?
- memory pressure가 생길 때 깨끗하게 실패하는가?
- tool call이 실제 arguments로 파싱되는가?
- prefix cache hit/miss가 구분되는가?
- 냉각과 전원 상태가 장시간 유지되는가?
- 다른 사람이 같은 commit과 입력으로 재현할 수 있는가?

그래서 각 장의 끝에 `검증 체크리스트`와 `아직 모르는 것`을 둔다. 빠른 숫자 하나보다 이 두 목록이 책을 다시 사용할 수 있게 만든다.

## 이 장의 검증 체크리스트

- [ ] 내 목적을 worker/supervisor/multimodal 중 하나로 분류했다.
- [ ] 단일 스트림과 동시성 중 무엇을 최적화할지 정했다.
- [ ] `loads`와 `agent-tested`를 구분했다.
- [ ] 1대/2대/4대가 해결하는 문제가 서로 다르다는 것을 확인했다.
- [ ] 선택한 모델의 원본·quant·runtime·context 조건을 기록했다.
