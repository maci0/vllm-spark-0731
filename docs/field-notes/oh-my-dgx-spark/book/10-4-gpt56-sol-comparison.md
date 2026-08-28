# 10-4. GPT-5.6 Sol과 로컬 모델 비교

상태: 공식 자료·로컬 실측을 반영한 초안

기준일: **2026-08-22**

이 장에서 말하는 “Sol Max”는 별도의 API 모델 ID가 아니다. 공식 API 모델은 `gpt-5.6-sol`이고, `reasoning.effort=max`를 선택한 구성을 편의상 **Sol max**라고 부른다. 모델명과 reasoning 설정을 섞으면 비교 결과를 잘못 읽게 된다.

## 3분 이해 (ELI5)

Sol max와 local Spark를 비교하는 일은 “누가 더 똑똑한가” 한 줄로 끝나지 않는다.

```text
같은 문제·도구·시간 제한
          ↓
품질 + latency + 비용 + 데이터 경계
          ↓
어떤 작업을 어디에 맡길지 결정
```

hosted API와 local server는 비용과 운영 조건이 다르므로 점수만 옮겨 적지 않는다.

![Sol max와 DGX Spark local server의 역할을 나누는 Archify 다이어그램](../assets/archify-sol-local.svg)

## 14.1 공식 API 기준

[OpenAI GPT-5.6 Sol 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.6-sol)에 따르면 다음과 같다.

| 항목 | 공식 기준 |
|---|---|
| 모델 ID | `gpt-5.6-sol` |
| 별칭 | `gpt-5.6` |
| reasoning effort | `none`, `low`, `medium`, `high`, `xhigh`, `max` |
| context window | 1,050,000 tokens |
| max output | 128,000 tokens |
| 입력 가격 | 1M tokens당 5달러 |
| cached input | 1M tokens당 0.50달러 |
| 출력 가격 | 1M tokens당 30달러 |
| 긴 입력 가격 | 272K 초과 prompt는 전체 요청에 input 2배, output 1.5배 |
| 기능 | function calling, structured outputs, image input, Responses API 도구 |

이 값은 hosted API의 조건이다. DGX Spark의 전력·다운로드·관리 시간과 같은 로컬 비용 항목이 아니며, API의 실제 지연은 provider, queue, reasoning effort, prompt 길이와 tool loop에 따라 달라진다.

## 14.2 DeepSeek 공식 점수와 로컬 C1을 나눈다

[DeepSeek V4 Flash 0731 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)는 Terminal-Bench 2.1 82.7을 포함한 agent/code 표를 공개한다. 이 표의 Code Agent 평가에는 공개 전인 `minimal DeepSeek Harness`, `reasoning_effort=max`, `temperature=1.0`, `top_p=0.95`가 사용됐다고 적혀 있다.

우리의 [one-Spark recipe C1 하니스](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark/blob/main/image-patch/acceptance_c1.py)는 다른 목적의 serving gate다. semantic, JSON schema, 다섯 언어의 code decode, 252,047-token cold prefill을 확인한다. 이 하니스의 기본 요청은 `thinking=false`이고, 모델 품질을 Terminal-Bench 점수로 환산하는 평가가 아니다.

따라서 다음 두 문장은 서로 바꿔 쓸 수 없다.

- “DeepSeek 공식 모델 카드가 82.7을 보고했다.”
- “우리 Spark가 공식 82.7 품질을 재현했다.”

## 14.3 DGX Spark 로컬 실측

2026-08-22 현재 실행 중인 단일 Spark recipe에서 같은 C1 명령을 다시 실행했다.

```bash
python3 image-patch/acceptance_c1.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-0731
```

| gate | 2026-08-22 결과 |
|---|---|
| semantic | 통과, `17 × 19 = 323` |
| JSON schema | 통과, Python·323·true |
| code C1 decode | 통과, 최저 37.553, 중앙값 41.358, 평균 40.915 tok/s |
| cold prefill 응답 | 통과, `PREFILL OK.`, cached tokens 0 |
| cold prefill token count | 실패, 목표 252,047 / 실제 251,968 |
| cold prefill speed | 실패, 985.377 tok/s / 기준 1,000 |
| 전체 C1 | 실패, token accounting과 prefill speed gate 때문 |

실험 시점의 local clone은 recipe commit `d1dc9e7`이었고, 현재 upstream `main`의 `5ba18b7`과 비교했을 때 차이는 README의 context 라벨 한 줄뿐이었다. C1 하니스와 serving script의 파일 hash는 동일하므로, 실행 코드 기준으로는 현재 recipe와 같은 경로다. 그래도 문서 재현 시에는 upstream commit을 함께 기록한다.

짧은 tool-call 실험에서는 `lookup_weather` 함수와 `{"city":"서울"}` arguments가 생성됐다. 이어서 첫 tool call을 로컬 mock 결과로 되돌려 보내는 한 번의 multi-turn loop를 실행했고, 두 번째 응답에서 추가 tool call 없이 최종 문장을 받았다. [raw loop 결과](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/results-deepseek-tool-loop-2026-08-22.json)를 보존했다. 이 결과는 message contract와 정상적인 한 단계 왕복을 검증하지만, 실제 날씨 API 실행·오류 복구·복잡한 여러 단계 agent 작업까지 검증했다는 뜻은 아니다.

같은 검사는 다음 명령으로 재현할 수 있다.

```bash
python3 tests/tool_loop_smoke.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731 \
  --strict
```

하니스의 `/tokenize`와 실제 `chat` 요청이 `thinking=false`를 다르게 적용하는 79-token 차이는 별도로 기록했다. 이를 고치면 token-count gate의 해석은 더 정확해지지만, 985.377 tok/s라는 속도 gate 결과를 자동으로 통과시키지는 않는다.

## 14.4 StateM 결과를 읽는 법

[StateM 논문](https://arxiv.org/abs/2608.15089)은 모델 weight만 바꾼 비교가 아니다. durable state, runbook, checked transition과 같은 agent runtime을 사용한다. 논문 초록의 비교는 다음과 같다.

| 구성 | 보고된 Terminal-Bench 결과 |
|---|---:|
| DeepSeek V4 Flash standard | 82.7% |
| 같은 runtime·runbook을 적용한 DeepSeek | 88.1% |
| DeepSeek 공통 88-task core | 89.1% |
| GPT-5.6 Sol xhigh reference | 84.9% |
| GPT-5.6 Sol max 보고값 | 88.8% |

이 결과가 보여주는 것은 **agent harness가 결과를 크게 바꿀 수 있다**는 점이다. “DeepSeek 모델 자체가 Sol max와 동급” 또는 “DGX Spark EXL3가 Sol API와 같은 품질”을 증명하는 결과는 아니다. 특히 로컬 C1의 `thinking=false` serving gate와 StateM의 reasoning·runbook 조건은 다르다.

## 14.5 공정한 비교 설계

Sol max와 DeepSeek를 직접 비교하려면 아래를 고정한다.

1. 같은 Terminal-Bench subset 또는 같은 고정 task set
2. 같은 agent runtime, tool schema, runbook과 timeout
3. DeepSeek는 공식 harness 조건과 local serving 조건을 별도 행으로 기록
4. Sol은 `gpt-5.6-sol`과 `reasoning_effort=max`를 명시
5. 성공률, 실패 이유, input/output/reasoning tokens, wall time, tool calls를 저장
6. local은 전력·다운로드·운영 시간을 포함하고, API는 input/output·cache·긴 prompt multiplier를 포함
7. 단일 stream decode와 aggregate throughput을 분리

현재 Codex 세션에서 DeepSeek 결과를 읽고 질적으로 검토할 수는 있지만, 이 대화의 Codex backend를 Sol max benchmark로 간주하지 않는다. 실제 Sol 비교 수치를 만들려면 모델 ID와 reasoning 설정이 기록되는 별도 API 실행 또는 재현 가능한 공개 결과가 필요하다.

## 결론

DGX Spark의 DeepSeek V4 Flash 0731은 강력한 로컬 코드·에이전트 후보다. C1 code decode, tool parser, 로컬 mock을 이용한 한 번의 정상 tool loop는 현재 장비에서 확인됐고, 단일 Spark의 긴 context recipe도 실행된다. 그러나 prefill gate는 아직 실패했으며, 실제 외부 도구 오류 복구와 공식 DeepSeek 점수·Sol max 비교는 별도 검증이 필요하다.

책의 구매·운영 결론은 다음 표현으로 고정한다.

> DeepSeek V4 Flash 0731은 DGX Spark에서 실행할 수 있는 매우 강한 로컬 코드 에이전트 후보다. GPT-5.6 Sol max와 가까운 점수는 공통 agent runtime과 runbook을 적용한 연구에서 보고됐지만, local EXL3 serving의 속도·품질·비용이 Sol API와 동급이라는 뜻은 아니다.

## 참고 자료

- [GPT-5.6 Sol 공식 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [OpenAI 모델 비교표](https://developers.openai.com/api/docs/models/compare)
- [DeepSeek V4 Flash 0731 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [StateM 논문](https://arxiv.org/abs/2608.15089)
- [DeepSeek 단일 Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
- [이 저장소의 DeepSeek 실험 기록](../docs/deepseek-v4-flash-0731-performance-research-2026-08.md)
