# 04. 서버와 OpenAI-compatible endpoint 만들기

모델을 고르는 일과 서버를 운영하는 일은 별개입니다. 같은 checkpoint라도 vLLM, SGLang, llama.cpp, SparkInfer 중 어떤 경로를 선택하는지에 따라 지원 기능과 측정값이 달라집니다.

## 엔진을 고르는 기준

| 엔진 | 우선 확인할 것 | 적합한 시작점 |
|---|---|---|
| vLLM | 공식 Spark image, model support, TP·tool parser | 표준 OpenAI-compatible serving |
| SGLang | ARM64/SM121 build, structured output, speculative path | 지원되는 Qwen·custom 레시피 |
| llama.cpp | GGUF와 CUDA build, context·offload | 작은 모델·GGUF·단순 endpoint |
| SparkInfer 계열 | 레시피가 요구하는 fork와 patch | 특정 DeepSeek EXL3 경로 |

최신이라는 이유만으로 엔진을 바꾸지 않습니다. 먼저 같은 모델·prompt·context에서 기능 기준선과 속도 기준선을 각각 측정해야 합니다.

## OpenAI-compatible이라는 말의 범위

`/v1/chat/completions`를 받는다고 해서 OpenAI API의 모든 기능을 동일하게 지원한다는 뜻은 아닙니다. 다음 항목을 별도로 확인합니다.

- model name과 endpoint path
- streaming과 usage 필드
- `temperature`, `max_tokens`, `thinking` 처리
- structured output과 JSON schema
- tool choice와 arguments 형식
- image input·vision adapter의 유무

최소 확인 예시는 다음과 같습니다.

```bash
curl -sS http://127.0.0.1:8000/v1/models

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"health check"}],"max_tokens":32}'
```

## Tool parser는 세 가지를 함께 봐야 한다

1. 서버가 요청에서 tool schema를 받는가?
2. 모델이 정해진 형식의 arguments를 출력하는가?
3. 클라이언트가 결과를 `tool` message로 되돌려 multi-turn을 이어 가는가?

`--enable-auto-tool-choice`와 parser flag가 있어도 모델이 올바른 JSON을 만든다는 보장은 없습니다. 단일 tool call, 잘못된 arguments, tool 결과를 받은 뒤의 최종 응답을 각각 기록해야 합니다.

## Qwen3.8 아키텍처 오류를 피하는 법

Qwen3.8-27B는 모델 이름과 runtime이 인식한 architecture가 일치하는지 확인해야 합니다. 서버가 Qwen3.5 계열 architecture를 인식한다고 해서 모델이 Qwen3.5라는 뜻은 아닙니다. Transformers config, runtime model implementation, container commit을 함께 확인합니다.

모델 ID·config·runtime이 어긋나면 “서버는 기동했지만 잘못된 backend를 사용하는 상태”가 될 수 있습니다. 이를 단순한 속도 문제로 기록하지 않습니다.

## DeepSeek 전용 레시피를 표준 endpoint와 구분한다

DeepSeek V4 Flash 0731의 단일 Spark EXL3/SparkInfer 레시피는 표준 vLLM 기준 구성과 다른 quant·draft·KV 경로를 사용합니다. endpoint가 OpenAI 형식이어도 내부 실행 경로가 같아지는 것은 아닙니다.

단일 Spark 레시피: [MiaAI-Lab DeepSeek V4 Flash](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark).

## 운영 계약

endpoint를 다른 에이전트에 연결하기 전에 아래를 문서로 고정합니다.

```text
base_url · model · timeout · max_output_tokens · retry policy
tool schema · allowed tools · log redaction · health check
```

서버가 내려가면 자동 retry가 같은 tool을 반복 실행하지 않도록 request ID와 idempotency 정책을 둡니다.

## 더 자세히 읽기

- [04-1. vLLM·SGLang·llama.cpp·SparkInfer 선택](04-1-engine-selection.md): 엔진별 성격과 기능·속도 기준선을 다룹니다.
- [04-2. 엔진 선택 체크리스트](04-2-engine-checklist.md): 목적부터 고르는 짧은 결정 순서입니다.
- [04-3. OpenAI-compatible endpoint와 tool parser](04-3-openai-endpoint-tool-parser.md): 서버 기동과 tool call 검증을 분리합니다.
