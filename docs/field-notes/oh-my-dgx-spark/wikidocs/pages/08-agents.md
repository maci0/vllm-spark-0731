# 08. 코딩 에이전트 운영

로컬 모델을 에이전트에 연결하는 일은 `/v1/chat/completions`가 열리는 것에서 끝나지 않습니다. 에이전트는 모델을 호출하고 도구 입력을 검증해야 합니다. 또한 결과를 다시 모델에 전달하고, 실패했을 때 작업을 중단하거나 복구해야 합니다.

![에이전트 모델 역할 분리](../assets/archify-model-roles.svg)

## 먼저 역할을 나눈다

한 모델에 모든 역할을 맡기기보다 작업의 위험도와 반복 패턴에 따라 역할을 나누면, 운영 조건을 설명하기 쉽습니다.

| 역할 | 하는 일 | 확인할 평가 |
|---|---|---|
| supervisor | 계획, 긴 문맥 요약, worker 선택, 복구 판단 | task success, 계획의 일관성, 실패 복구 |
| coding worker | 코드 검색·수정·테스트 실행 | patch 정확성, 테스트 통과, tool arguments |
| UI·design worker | CSS·레이아웃·컴포넌트 반복 | 시각적 요구 충족, 변경 범위, 회귀 여부 |
| media worker | 이미지·음성·영상 파이프라인 | frame/audio throughput, 품질, pipeline 오류 |

DeepSeek를 supervisor로, Qwen3.8-27B를 UI·design worker로 배치하는 구성은 역할 분리의 한 예입니다. 이 배치는 공개 레시피와 직접 실험을 바탕으로 한 **운영 설계**이지, 두 모델의 전반적인 지능이 같다는 benchmark 결과가 아닙니다.

## endpoint 계약을 고정한다

에이전트 클라이언트에는 다음 값을 코드나 설정에 명시합니다.

```text
base_url
model
timeout
max_output_tokens
tool schema
allowed tools
retry policy
log redaction
```

OpenAI-compatible endpoint라도 제공자마다 streaming, `usage`, `thinking`, structured output, image input과 tool call의 세부 동작이 다를 수 있습니다. 연결하기 전에 `/v1/models`, 단일 chat, 단일 tool call, tool 결과를 받은 최종 응답을 순서대로 실행합니다.

## tool call은 네 단계로 검사한다

1. **schema 전달**: 서버와 client가 tool 이름·필수 인자·자료형을 같은 방식으로 전달하는지 확인합니다.
2. **모델 출력**: 모델이 올바른 tool name과 JSON arguments를 생성하는지 확인합니다.
3. **실제 실행**: runner가 allowlist, path, 명령 인자, timeout을 검사한 뒤 실행합니다.
4. **결과 반환**: 실행 결과를 `tool` message로 되돌리고, 모델이 최종 응답에서 실행 결과를 정확히 반영하는지 확인합니다.

서버에 tool parser flag가 있다는 사실만으로 모델의 tool call 성공을 의미하지는 않습니다. parser가 출력 문자열을 구조화했는지와 실제 도구가 안전하게 실행됐는지는 서로 다른 검사입니다.

## 권한 경계가 먼저다

개발용 에이전트라도 처음부터 전체 파일 시스템과 shell에 대한 권한을 열어 두지 않습니다.

- 작업 디렉터리를 allowlist로 제한합니다.
- 파일 삭제·네트워크 전송·credential 접근은 기본 거부합니다.
- 실행 전 command와 대상 파일을 표시하는 dry-run 모드를 둡니다.
- 긴 작업은 timeout과 최대 tool 횟수를 둡니다.
- 같은 요청이 재시도되어도 중복 실행되지 않도록 request ID와 idempotency를 기록합니다.
- 로그에는 API key, bearer token, 개인 파일 내용과 환경 변수를 남기지 않습니다.

OpenClaw, Hermes Agent, NemoClaw는 각각 로컬 endpoint와 도구·채널·sandbox를 연결하는 소프트웨어 또는 공식 playbook의 사례입니다. 이 책에서는 해당 프로젝트가 설치되었다고 주장하지 않습니다. 독자는 레시피의 commit, 권한 정책, gateway 노출 범위를 확인한 뒤 별도 작업 디렉터리에서 시험해야 합니다.

참고: [NVIDIA Hermes Agent playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/hermes-agent), [OpenClaw playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/openclaw), [NemoClaw quickstart](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/get-started/quickstart).

## 모델을 바꿀 때 보는 순서

모델의 역할을 바꿀 때는 속도만 비교하지 않습니다.

| 순서 | 질문 |
|---:|---|
| 1 | 같은 endpoint 계약과 chat template을 유지했습니까? |
| 2 | 같은 code·JSON·long-context·tool task를 실행했습니까? |
| 3 | tool arguments 오류와 잘못된 파일 변경을 분리해 기록했습니까? |
| 4 | timeout, malformed JSON, server restart 뒤 복구했습니까? |
| 5 | 평균 tok/s가 아니라 task 성공률과 총 wall time을 비교했습니까? |

DeepSeek의 단일 Spark C1 결과는 serving 기준선이지 agent 성공률이 아닙니다. NVIDIA의 공식 Qwen3.6 agent-ready 레시피에서도 큰 tool surface에서 malformed tool-call이 발생했다는 보고가 있습니다. 따라서 모델 이름이나 playbook의 별칭보다 실제 tool 목록으로 검사해야 합니다. Qwen3.8의 runtime이 Qwen3.5 architecture를 표시하는 문제도 모델 이름만 보지 말고 config·implementation·runtime commit을 함께 확인해야 합니다.

참고: [NVIDIA playbook Issue #89](https://github.com/NVIDIA/dgx-spark-playbooks/issues/89).

## 에이전트 완료 기준

에이전트는 “답변을 생성했다”는 사실만으로 완료 처리하지 않습니다. 다음 조건을 모두 만족해야 합니다.

- [ ] 고정된 task를 재현했습니다.
- [ ] tool name과 arguments가 schema를 통과했습니다.
- [ ] 허용되지 않은 파일·명령·네트워크에 접근하지 않았습니다.
- [ ] 실행 결과를 최종 답변에 정확히 반영했습니다.
- [ ] 실패·timeout·재시작 뒤 중복 작업 없이 복구했습니다.
- [ ] prompt, model revision, tool log와 판정을 다시 확인할 수 있습니다.

## 더 자세히 읽기

- 08-1. 로컬 에이전트 운영 상세: Hermes·OpenClaw·NemoClaw, 권한, thinking과 fallback을 다룹니다.
- 08-2. endpoint 계약: 에이전트가 기대하는 API 계약을 점검합니다.
- 08-3. brain·worker·권한 분리: DeepSeek와 Qwen을 역할별로 배치하는 기준입니다.
