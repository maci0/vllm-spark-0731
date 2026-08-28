# 04-2. 엔진 선택 체크리스트

이 페이지는 [04-1. vLLM·SGLang·llama.cpp·SparkInfer 선택](04-1-engine-selection.md)의 상세 내용입니다.

엔진은 모델 이름보다 작업 목적을 먼저 보고 고른다.

| 목적 | 먼저 볼 엔진 | 확인할 것 |
|---|---|---|
| OpenAI-compatible API | vLLM·SGLang | parser, batching, endpoint 계약 |
| Qwen3.8 recipe | vLLM·SGLang | architecture 인식, chat template |
| DeepSeek EXL3 | SparkInfer 계열 | quant, KV, draft model |
| 단순 local generation | llama.cpp | GGUF, context, Metal/CUDA 경로 |

## 결정 순서

1. 모델 weight와 quant가 지원되는가.
2. 필요한 context가 메모리에 들어가는가.
3. tool call과 reasoning parser가 필요한가.
4. 단일 스트림인지 동시성인지.
5. 속도보다 재현성과 운영성이 중요한가.

“가장 빠른 엔진”을 찾기 전에 같은 모델과 같은 workload를 실행할 수 있는 엔진을 찾는다.
