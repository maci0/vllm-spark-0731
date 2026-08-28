# GPT-5.6 Sol max 비교 리서치

조사·검증일: **2026-08-23**

## 먼저 정리할 이름

공식 API 문서의 모델 ID는 `gpt-5.6-sol`입니다. `reasoning.effort`에 `max`를 주는 구성을 이 책에서는 `Sol max`라고 부릅니다. “GPT-5.6-Sol Max”를 별도의 모델 ID처럼 기록하지 않습니다.

## 공식 OpenAI 자료 확인

[GPT-5.6 Sol 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.6-sol)에서 확인한 값은 다음과 같습니다.

| 항목 | 값 |
|---|---|
| model ID | `gpt-5.6-sol` |
| alias | `gpt-5.6` |
| reasoning effort | none, low, medium, high, xhigh, max |
| context window | 1,050,000 tokens |
| max output | 128,000 tokens |
| input | $4.00 / 1M tokens |
| cached input | $0.40 / 1M tokens |
| output | $20.00 / 1M tokens |
| 긴 prompt | 272K 초과 시 전체 요청의 input 2배, output 1.5배 |
| 지원 | streaming, function calling, structured outputs, image input |

모델 문서에는 Chat Completions와 Responses endpoint가 모두 표시되고, Responses API의 hosted tools도 지원 목록에 포함되어 있습니다. 오디오와 비디오는 지원되지 않는 것으로 표시됩니다. 이 정보는 API 조건이며 DGX Spark local endpoint의 기능 목록과 섞지 않습니다.

## DeepSeek 공식 평가와 비교 가능 범위

[DeepSeek 공식 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)는 Terminal-Bench 2.1 82.7과 여러 code/agent 점수를 제시합니다. 카드의 주석은 공개 benchmark의 Code Agent task에 `minimal DeepSeek Harness`를 사용하고 `reasoning_effort=max`, `temperature=1.0`, `top_p=0.95`를 적용했다고 설명합니다. 이 harness는 “to be released”라고 적혀 있으므로, 우리 C1과 동일하다고 가정하지 않습니다.

[StateM 논문](https://arxiv.org/abs/2608.15089)은 runtime과 runbook을 바꾼 비교를 보고합니다. standard DeepSeek 82.7을 같은 구조의 runtime·runbook으로 88.1까지 올렸고, 공통 88-task core에서는 89.1을 보고했습니다. 논문은 GPT-5.6 Sol xhigh reference 84.9와 Sol max 88.8을 함께 언급합니다. 이는 모델 weight만 비교한 결과가 아니라 harness scaling 결과입니다.

## 우리 Spark에서 확인한 것

2026-08-22 실행 환경은 다음과 같습니다.

- image: `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`
- model: `deepseek-v4-flash-0731`
- server: `max_model_len=384000`, TP=1, 레시피 기본값
- container: `healthy`, restart count 0
- C1 request: `thinking=false`

실측:

| 항목 | 결과 |
|---|---:|
| semantic | pass |
| JSON schema | pass |
| code C1 decode minimum | 37.553 tok/s |
| code C1 decode median | 41.358 tok/s |
| code C1 decode mean | 40.915 tok/s |
| cold prefill response | `PREFILL OK.` |
| cold prefill actual prompt | 251,968 tokens |
| cold prefill | 985.377 tok/s |
| C1 prefill gate | fail, threshold 1,000 tok/s |
| tool call | `lookup_weather({"city":"서울"})` 생성 |

추가로 첫 tool call을 로컬 mock 결과로 되돌려 보내는 한 번의 multi-turn loop도 실행했습니다. 함수 arguments 파싱, `tool` 메시지 왕복, 최종 자연어 응답까지는 통과했지만 실제 외부 도구나 오류 복구를 실행한 것은 아닙니다. [raw 결과](results-deepseek-tool-loop-2026-08-22.json)와 [재현 스크립트 결과](results-deepseek-tool-loop-script-2026-08-22.json)를 별도로 보존합니다.

`/tokenize` prompt 생성과 실제 chat usage 사이에는 `thinking=false` 처리 차이로 79-token drift가 있습니다. token count gate와 speed gate를 분리해서 기록합니다.

## 검증하지 못한 것

- 공식 DeepSeek Harness 전체 재현
- GPT-5.6 Sol API의 동일 task set 직접 실행
- StateM runtime/runbook을 DGX Spark local endpoint에 그대로 적용한 결과
- GPT-5.6 Sol과 raw tok/s 비교

이 네 항목은 문서에 미검증으로 남깁니다. 현재 편집 검토의 결과를 Sol max의 API 실행 결과나 재현 결과로 기록하지 않습니다.

## 책에 넣을 최종 판정

DeepSeek V4 Flash 0731은 단일 DGX Spark에서 실행할 수 있는 강한 로컬 코드·에이전트 후보입니다. 공개 자료와 StateM 결과를 보면 특정 agent harness에서는 Sol max와 가까운 점수가 가능하지만, 현재 Spark의 EXL3 C1 결과는 serving acceptance이며 prefill gate도 아직 실패했습니다. 따라서 “Sol max와 동급” 대신 “강한 로컬 대안이며, 동일 harness 비교가 필요한 후보”라고 기록합니다.

## 출처

- [GPT-5.6 Sol 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [OpenAI 모델 비교표](https://developers.openai.com/api/docs/models/compare)
- [DeepSeek V4 Flash 0731 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [StateM arXiv 논문](https://arxiv.org/abs/2608.15089)
- [DeepSeek one-Spark 레시피](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
