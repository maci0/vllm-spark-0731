# 08-2. endpoint 계약

이 페이지는 [08-1. 로컬 에이전트 운영](08-1-local-agent-operations.md)의 상세 내용입니다.

에이전트는 모델 내부가 아니라 endpoint 계약을 사용한다. 모델을 바꿔도 다음 동작이 유지되어야 한다.

## 계약 항목

- `/v1/models`에서 식별 가능한 model name
- `/v1/chat/completions`의 messages와 streaming
- tool schema와 arguments JSON
- timeout, context 초과, server error 형식
- thinking content와 최종 content의 분리

모델을 교체할 때 먼저 동일 prompt를 두 endpoint에 보낸다. 답변 품질 비교보다 먼저 HTTP status, JSON schema, finish reason을 비교한다.

## 실패 처리

재시도는 무조건 하지 않는다. timeout은 제한적으로 재시도할 수 있지만, 잘못된 tool arguments와 permission error는 즉시 agent에 반환해야 한다.
