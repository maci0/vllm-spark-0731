# 04-3. OpenAI-compatible endpoint와 tool parser

이 페이지는 [04-1. vLLM·SGLang·llama.cpp·SparkInfer 선택](04-1-engine-selection.md)의 상세 내용입니다.

에이전트는 모델 이름보다 endpoint 계약에 의존한다. `/v1/models`, `/v1/chat/completions`, tool schema, error body를 먼저 고정한다.

## 기본 확인

```bash
curl -fsS http://127.0.0.1:PORT/v1/models
curl http://127.0.0.1:PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"한 문장으로 답해줘"}],"max_tokens":64}'
```

Qwen3.8 vLLM 경로에서는 `--enable-auto-tool-choice`, `--tool-call-parser qwen3_xml`을 빠뜨리지 않는다. parser 옵션을 켰다고 tool call 품질이 검증된 것은 아니므로 arguments JSON과 함수 이름을 별도로 검사한다.

## 운영 기준

- endpoint가 죽었을 때 agent가 재시도할 횟수
- timeout과 context 초과 응답
- thinking 필드와 최종 답변의 분리
- tool call 뒤 결과를 다음 turn에 넣는 방식

이 네 가지를 문서화해야 모델 교체가 에이전트 전체 장애로 번지지 않는다.
