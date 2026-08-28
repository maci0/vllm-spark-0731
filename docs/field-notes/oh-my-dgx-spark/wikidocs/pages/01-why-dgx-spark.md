# 01. DGX Spark가 내 작업에 맞는가?

DGX Spark는 작은 케이스에 들어간 일반 데스크톱 GPU가 아닙니다. CPU와 GPU가 128GB unified system memory를 공유하고, Blackwell GPU와 ConnectX-7 네트워크를 한 장치에 묶은 Arm 기반 AI 시스템입니다. 먼저 이 구조가 유리한 작업과 불리한 작업을 구분해야 합니다.

## 공식 하드웨어 사실

NVIDIA 하드웨어 문서가 명시하는 기준은 다음과 같습니다.

| 항목 | 공식 표기 |
|---|---|
| 메모리 | 128GB LPDDR5x unified system memory |
| 메모리 대역폭 | 273GB/s |
| CPU | 20-core Arm processor |
| 네트워크 | 10GbE, Wi‑Fi 7, ConnectX-7, 2× QSFP |
| 전원 | 제공된 240W adapter 사용 |
| NVIDIA의 제품 포지셔닝 | 단일 최대 200B, 듀얼 구성 405B 모델 지원 대상 |

여기서 “200B/405B 지원”은 NVIDIA가 제시한 모델 규모의 범위입니다. 특정 checkpoint가 특정 context와 runtime에서 안정적으로 생성된다는 보증이나 benchmark 결과를 의미하지는 않습니다.

출처: [NVIDIA Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [System Overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html).

## 어떤 작업에 맞는가

| 작업 | 적합성 | 이유 |
|---|---|---|
| 로컬 코딩 assistant | 높음 | CUDA·OpenAI-compatible server·긴 context를 한 시스템에서 운영할 수 있습니다. |
| 긴 문서·코드 supervisor | 높음 | unified memory와 큰 context를 실험하기 좋습니다. 단, KV cache가 여유를 사용합니다. |
| 여러 사용자의 고동시성 API | 조건부 | 메모리와 전력보다 batching·runtime·network가 병목이 될 수 있습니다. |
| 대규모 학습 | 목적에 따라 다름 | fine-tuning 레시피는 있지만 데이터·시간·분산 구성을 별도로 검증해야 합니다. |
| 게임·일반 그래픽 워크스테이션 | 우선순위 낮음 | 구매 이유가 CUDA 로컬 AI가 아니라면 다른 GPU·Mac·워크스테이션이 더 합리적일 수 있습니다. |

## 한 대와 여러 대는 목적이 다르다

- **1대**: 설치와 운영이 가장 단순합니다. Qwen 계열 coding worker, 양자화된 DeepSeek, 공식 vLLM 기준 구성을 비교하기 좋습니다.
- **2대**: TP=2로 한 모델을 나누거나, 두 endpoint를 독립적으로 운영할 수 있습니다. 200GbE QSFP 직결과 NCCL 검증이 필요합니다.
- **3대**: `TP=3`이 자동으로 좋은 선택은 아닙니다. PP 또는 DP, 3-way ring의 운영 복잡도를 함께 평가해야 합니다.
- **4대**: 스위치 기반 TP=4·DP=4 구성을 검토하기 좋습니다. 다만 네트워크와 장애 범위가 개인용 장비의 수준을 넘어섭니다.
- **6~8대**: 모델을 더 크게 올리는 문제보다 클러스터 운영, 전력, 냉각, 모니터링 문제가 먼저 됩니다.

## Sol max와 비교할 때

GPT-5.6 Sol은 hosted API 모델이고, DGX Spark는 직접 운영하는 local system입니다. 이 책에서는 `reasoning.effort=max` 구성을 Sol max라고 부르며, 이를 **비교 기준과 작성 품질 기준**으로 사용합니다. 그러나 로컬 모델의 tok/s를 Sol의 지능 점수로 바꾸어 쓰지는 않습니다.

OpenAI 문서 기준 `gpt-5.6-sol`은 `reasoning.effort=max`, 1,050,000-token context, 128,000-token maximum output, function calling과 structured outputs를 지원합니다. 이 조건은 local server의 `max_model_len`과 비슷해 보이지만, 비용·네트워크·호스팅·tool 실행 경계가 다릅니다.

출처: [GPT-5.6 Sol 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.6-sol).

## 구매 전 결정표

| 질문 | “예”라면 | “아니오”라면 |
|---|---|---|
| 모델을 직접 내려받고 endpoint를 운영해야 하는가? | Spark의 운영 가치를 검토합니다. | hosted API나 일반 GPU가 더 단순할 수 있습니다. |
| 128GB 공유 메모리가 필요한가? | UMA와 모델별 memory budget을 계산합니다. | 작은 GPU·CPU·Mac과 비용을 비교합니다. |
| CUDA·NCCL·컨테이너를 직접 관리할 수 있는가? | 실전 레시피를 따라갈 수 있습니다. | 관리형 환경을 우선 고려합니다. |
| 1대의 결과가 느려도 원인 분석을 감수할 수 있는가? | Spark를 개발 장비로 볼 수 있습니다. | 가격과 운영 시간을 TCO에 넣습니다. |

## 이 장의 결론

DGX Spark를 고르는 이유는 “가장 빠른 모델”이 아니라 **큰 모델을 내 환경에서 통제하며 반복 실험할 수 있다는 점**입니다. 구매 판단은 모델 이름이 아니라 workload, context, 동시성, 운영 시간, 전력과 지원 비용을 함께 놓고 합니다.

## 더 자세히 읽기

- 01-1. DGX Spark 선택 기준 상세: 지원·실행·실전의 차이와 노드 수별 사용 예를 다룹니다.
- 01-2. DGX Spark·GB10 벤더 비교: 같은 GB10을 사용한 시스템과 다른 대안을 구분합니다.
- 01-3. 공식 GB10 시스템별 사양: 제조사별 공통점과 차이를 확인합니다.
- 01-4. 냉각·성능·X 자료 읽기: 통제된 측정과 커뮤니티 보고를 구분합니다.
- 01-5. 구매·클러스터·재현: 구매 뒤 재현할 항목과 Mac·다른 장비의 경계를 정리합니다.
