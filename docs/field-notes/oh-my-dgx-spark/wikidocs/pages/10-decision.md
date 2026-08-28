# 10. 비용·Sol 비교·최종 선택

DGX Spark와 GPT-5.6 Sol은 같은 종류의 제품이 아닙니다. Sol은 hosted model API이고, Spark는 모델·운영체제·전원·네트워크를 직접 관리하는 local system입니다. 따라서 “어느 쪽이 더 똑똑한가”를 tok/s 하나로 결론 내릴 수 없습니다.

![Sol과 로컬 Spark의 선택 기준](../assets/archify-sol-local.svg)

## 비교할 수 있는 사실과 비교할 수 없는 사실

OpenAI 모델 문서에 따르면 `gpt-5.6-sol`은 `reasoning.effort=max`, 1,050,000-token context, 128,000-token maximum output, function calling과 structured outputs를 지원합니다. 이 값은 API 모델의 공개 사양이며, 로컬 모델의 실제 한도와는 별도로 읽어야 합니다.

DGX Spark에서는 모델 weight, quantization, runtime, KV cache, context와 tool runner를 직접 선택합니다. 128GB unified memory와 273GB/s 대역폭은 하드웨어 사실이지만, 특정 모델의 품질·context·tool 성공률을 자동으로 결정하지는 않습니다.

| 비교 항목 | Sol max | DGX Spark 로컬 모델 |
|---|---|---|
| 실행 위치 | hosted API | 사용자가 관리하는 장치 |
| 모델 ID | `gpt-5.6-sol` | checkpoint·quant·runtime 조합 |
| context/output | 모델 문서의 공개 한도 | 실제 KV headroom과 레시피에 따라 달라짐 |
| 도구 실행 | API의 function/structured output 계약 | endpoint·parser·로컬 runner를 직접 검증 |
| 지연 | 네트워크와 서비스 상태 포함 | local queue·prefill·decode·네트워크 포함 |
| 비용 | 요청 token·기능·계정 정책에 따라 변동 | 장비 구입·전력·냉각·관리 시간 |
| 통제 | weight·runtime 변경 불가 | weight·runtime·로그·데이터 경계 직접 통제 |

출처: [GPT-5.6 Sol 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [DGX Spark 하드웨어 문서](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

## DeepSeek가 Sol max와 비견되는가

현재 문서의 결론은 **“특정 coding agent harness에서 강한 로컬 대안 후보”**입니다. “Sol max와 동급”이라고 확정하지 않습니다.

근거는 서로 다른 층위에 있습니다.

- DeepSeek 모델 카드는 Terminal Bench와 tool 계열의 제작자 측 점수를 제시합니다.
- StateM 연구는 runtime·runbook·task set을 조정한 비교에서 DeepSeek 계열과 Sol reference의 점수를 보고합니다.
- 단일 Spark에서 직접 기록한 C1은 decode median 41.358 tok/s와 cold prefill 985.377 tok/s이며, 이 책의 1,000 tok/s prefill gate는 실패했습니다.
- 단일 Spark EXL3 레시피의 44~47 tok/s와 370K needle은 해당 양자화·runtime·single-stream 조건의 보고값입니다.

따라서 같은 task set, 같은 harness, 같은 성공 판정으로 비용과 wall time까지 함께 측정하기 전에는 모델의 전반적인 지능이나 Sol max 대체 가능성을 주장할 수 없습니다. 자세한 근거는 [Sol max 비교 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/sol-max-comparison-research-2026-08.md)와 [DeepSeek 성능 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/deepseek-v4-flash-0731-performance-research-2026-08.md)에 정리했습니다.

## 노드 수별 선택

| 상황 | 먼저 선택할 구성 | 선택 이유 |
|---|---|---|
| 처음 로컬 코딩을 시작함 | 1대, 공식 Qwen3.6 기준 구성 | setup·rollback·지원 범위가 명확합니다. |
| 빠른 코드·UI worker가 필요함 | 1대 Qwen3.8 레시피 | 단일 worker의 latency를 먼저 측정할 수 있습니다. |
| 긴 문맥 supervisor가 필요함 | 1대 DeepSeek EXL3 또는 2대 TP=2 | 레시피의 context·KV·tool 조건을 확인합니다. |
| 두 모델을 동시에 운영함 | 총 4대의 2×2 pool | DeepSeek TP=2와 Qwen TP=2를 독립 endpoint로 분리합니다. |
| 한 모델의 capacity를 크게 늘림 | 4대 TP=4 또는 모델별 공식 경로 | switch와 collective 검증을 전제로 합니다. |
| 서비스 동시성이 우선임 | DP로 여러 endpoint를 복제 | 한 요청의 모델 크기보다 queue와 장애 격리를 개선합니다. |
| 관리 시간을 줄이고 싶음 | Sol max 또는 다른 hosted API | 장비·driver·runtime을 직접 운영하지 않아도 됩니다. |

## TCO를 계산한다

최소한 다음 항목을 같은 기간을 기준으로 계산합니다.

```text
local TCO = 장비 가격
          + 배송·케이블·switch·스토리지
          + 전력(kWh × 요금)
          + 냉각·랙·교체 비용
          + 운영 시간의 비용

hosted TCO = 입력 token 비용
           + 출력 token 비용
           + tool·storage·gateway 비용
           + 네트워크·계정 운영 비용
```

DGX Spark의 240W adapter 정격과 실제 벽면 전력은 다릅니다. clock cap 비교에서는 GPU rail power와 wall power를 별도로 기록합니다. 단일 benchmark의 40 tok/s를 월간 비용으로 곧바로 환산하지 말고, 실제 prompt 길이·출력 길이·대기 시간·동시성을 계산에 넣어야 합니다.

## 이 책이 권하는 최종 순서

1. 한 대에서 공식 레시피로 `/v1/models`, 짧은 chat, 단일 tool call을 통과시킵니다.
2. 동일한 prompt와 task set으로 Qwen·DeepSeek·MiniMax 후보를 비교합니다.
3. C1 decode만이 아니라 prefill, TTFT, context, tool loop와 실패율을 기록합니다.
4. 두 대가 필요한 이유가 capacity인지 concurrency인지 구분합니다.
5. switch·RDMA·MCDMA는 최소 working endpoint가 생긴 뒤에 추가합니다.
6. 동일 harness로 Sol max와 비교할 수 있을 때만 품질·비용 결론을 업데이트합니다.

## 한 문장 결론

DGX Spark는 Sol max의 복제품이 아니라, **데이터와 실행 경계를 직접 통제하면서 모델을 실험하고 운영하는 로컬 플랫폼**입니다. DeepSeek V4 Flash 0731은 현재 강한 코드·에이전트 후보이지만, 이 책의 직접 측정과 공개 레시피만으로 Sol max와 동급이라고 단정할 수 없습니다.

## 더 자세히 읽기

- 10-1. 비용·전력·구성 의사결정
- 10-2. CAPEX·OPEX·TCO
- 10-3. 노드 수별 구매 판단
- 10-4. GPT-5.6 Sol과 로컬 모델 비교
- 10-5. 비교 조건 고정하기
- 10-6. 로컬 모델과 Sol의 역할
