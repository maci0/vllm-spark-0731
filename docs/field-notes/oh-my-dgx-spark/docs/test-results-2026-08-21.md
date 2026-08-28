# Test results — 2026-08-21

Machine: single NVIDIA DGX Spark / GB10 / 128 GiB unified memory.

## Parser-enabled rerun

2026-08-21에 로컬 캐시된 `OBLITERATUS/Qwen3.8-27B-OBLITERATED`를 vLLM 0.26.0으로 다시 기동했다. 서버는 `qwen3_xml` tool parser, `qwen3` reasoning parser, auto tool choice를 켠 상태였다. 모델은 BF16 51.1GiB를 사용했고, `/health`는 HTTP 200을 반환했다.

Qwen3.8 이름의 모델이 로그에서 `Qwen3_5ForConditionalGeneration`으로 표시되는 것은 정상이다. Qwen3.8은 Qwen3.5 아키텍처를 기반으로 하며, 공식 config와 이번 로컬 snapshot의 `model_type`은 `qwen3_5`다.

## Passed

- Model discovery on both endpoints
- Korean/basic text generation
- Python code generation
- JSON-constrained output and JSON parsing
- Multi-turn identifier recall (`ALPHA-7429`)
- Thinking off and thinking on
- Image description from a local JPEG
- 10,442-token prompt marker recall
- 32,035-token prompt marker recall under the 32K server setting (28.0 s end-to-end in the reproducible script)
- Four concurrent requests

## Tool call

`tests/tool_call_smoke.py --strict` 통과:

- `get_weather` 1회 호출
- `location=Seoul, South Korea` 유효한 JSON arguments
- 일반 content 없음
- `finish_reason=stop`

## Repeat benchmark

이 숫자는 순수 decode가 아니라 `completion_tok_s_e2e`다. prefill, decode, HTTP, 스케줄링을 모두 포함한다.

| 구성 | 조건 | 결과 |
|---|---|---:|
| c1 | warmup 2, trial 5, output 128 | TTFT p50 463.736ms / e2e p50 4.567 tok/s |
| c4 | warmup 1, trial 3, output 64 | TTFT p50 508.904ms / aggregate e2e 17.697 tok/s |

전체 메타데이터는 [실측 결과 JSON](results-qwen38-vllm-parser-2026-08-21.json)에 보존했다.

## Earlier smoke run

아래 기능 결과는 parser-enabled 재실행에서도 유지됐다.

## Earlier limitation

초기 smoke run에서는 vLLM 서버를 tool parser 없이 시작했기 때문에 다음 응답을 받았다.

```text
tool_choice=function "get_weather" requires --tool-call-parser to be set
```

이는 serving configuration 문제였으며 모델 기능 부재의 결론이 아니었다. 이번 parser-enabled 재실행에서 명시적 tool call이 통과해 이 제한을 해소했다.

## Performance caution

초기 짧은 BF16 요청은 약 4~5 completion tokens/s로 보였고, 이번 c1/c4 결과도 같은 범위였다. 다만 출력이 짧고 speculative decoding이 꺼져 있으므로 순수 decode benchmark로 기록하지 않는다. 동일 prompt set과 격리된 프로세스로 모델 간 비교를 해야 한다.
