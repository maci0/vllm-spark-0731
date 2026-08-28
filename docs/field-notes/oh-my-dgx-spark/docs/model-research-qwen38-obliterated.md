# Qwen3.8-27B-OBLITERATED 리서치 노트

기준일: 2026-08-21

## 1. 모델 정체

`OBLITERATUS/Qwen3.8-27B-OBLITERATED`는 `Qwen/Qwen3.8-27B`를 기반으로 refusal/safety 동작을 줄이도록 가중치를 수정한 파생 모델이다. Hugging Face에는 full BF16 safetensors(18 shards, 약 51.75 GiB), GGUF, MLX 변형이 함께 올라와 있다. DGX Spark에서는 현재 full BF16 safetensors를 vLLM으로 로드했다.

중요한 점은 다음과 같다.

- 원본과 같은 27B급 모델이지만, 원본의 평가 점수를 그대로 사용할 수 없다.
- 모델 카드의 “zero refusal”과 품질 수치는 제작자 측 평가다.
- 카드 자체가 MMLU 저하를 보고한다: 원본 87.4% → 파생 81.4%.
- 권장 추론 설정은 temperature 0, repetition penalty 1.15, thinking off이며, system prompt는 비워두는 쪽이다.

### Qwen3.8인데 왜 Qwen3.5로 표시되는가

이것은 잘못 로드된 것이 아니다. Qwen3.8 공식 카드는 Qwen3.5 아키텍처를 기반으로 한다고 설명하고, 공식 `config.json`도 `Qwen3_5ForConditionalGeneration`과 `qwen3_5_text`를 사용한다. 이번에 로컬 캐시한 파생 모델도 같은 메타데이터를 가진다. 따라서 vLLM 로그의 `Resolved architecture: Qwen3_5ForConditionalGeneration`은 모델 이름과 내부 구현 계보가 다르기 때문에 나오는 정상적인 표시다.

## 2. 원본 모델에서 기대할 수 있는 범위

원본 Qwen3.8 카드에는 다음 기능이 공식적으로 설명되어 있다.

- 일반 텍스트·코딩·연구·장기 에이전트 작업
- 이미지 및 비디오 이해
- thinking on/off와 reasoning effort 조절
- native 262,144 context
- YaRN을 사용한 1M context 확장
- SGLang, vLLM, Transformers 등 주요 실행기 호환

이 목록은 원본 모델의 기능 설명이다. 파생 모델에서 같은 기능이 동작하는지는 반드시 별도로 검증해야 한다.

## 3. GB10 실제 smoke test

테스트 서버:

```text
vLLM 0.26.0
PyTorch 2.11.0+cu130
127.0.0.1:8083
max_model_len=32768
gpu_memory_utilization=0.50
```

| 테스트 | 결과 | 관찰 |
|---|---|---|
| `/v1/models` | PASS | served model 정상 노출 |
| 한국어 산수 | PASS | `4` |
| Python 코드 | PASS | 순서 보존 중복 제거 함수 생성 |
| JSON 제약 출력 | PASS | JSON 파싱 성공 |
| 멀티턴 회수 | PASS | `ALPHA-7429` 정확히 회수 |
| thinking off | PASS | `323` |
| thinking on | PASS | reasoning 필드가 별도로 노출됨 |
| 이미지 입력 | PASS | JPEG 한 장 설명 성공 |
| 10,442 prompt tokens | PASS | needle 회수 |
| 32,035 prompt tokens | PASS | needle 회수; 32K 서버 한계 안 |
| 4 concurrent requests | PASS | 4개 모두 완료 |
| function call | PASS | `qwen3_xml` parser에서 `get_weather`와 유효한 JSON arguments 확인 |

긴 문맥 테스트는 다음 marker를 본문 끝에 넣고 회수했다.

```text
CONTEXT-NEEDLE-314159
```

이 결과는 “이 장비와 이 설정에서 기본 동작한다”는 뜻이지, 정확도 벤치마크나 1M 문맥 품질 인증을 의미하지 않는다.

## 4. parser-enabled 재실행 결과

2026-08-21에 단일 Spark에서 `vLLM 0.26.0`, BF16, `max_model_len=32768`, `qwen3_xml`, auto tool choice를 켜고 다시 측정했다. `/health`는 200을 반환했고, 모델은 51.1GiB를 사용해 올라왔다. 이 설정에서 KV cache는 72,089 tokens로 프로파일링됐다.

| 측정 | 결과 |
|---|---:|
| 명시적 tool call | PASS, 1/1 |
| tool arguments | 유효한 JSON, `location=Seoul, South Korea` |
| c1 TTFT p50 / p95 | 463.736 / 465.110ms |
| c1 end-to-end completion p50 | 4.567 tok/s |
| c4 TTFT p50 / p95 | 508.904 / 510.999ms |
| c4 aggregate end-to-end completion | 17.697 tok/s |

세부 provenance는 [실측 결과 JSON](results-qwen38-vllm-parser-2026-08-21.json)에 보존했다.

## 5. 속도 해석

이번 c1·c4 결과의 `completion_tok_s_e2e`는 prefill과 decode, 서버 스케줄링, HTTP 스트리밍 시간을 모두 포함한다. 따라서 순수 decode tok/s로 쓰지 않는다. c1 단일 요청은 약 4.567 tok/s, c4 aggregate는 약 17.697 tok/s였으며, 서로 다른 workload 지표다.

이 결과는 다음 이유로 아직 모델 간 정식 순위표가 아니다.

- 출력이 27 tokens로 짧아 prefill/HTTP/스케줄러 시간이 크게 섞였다.
- speculative decoding을 켜지 않았다.
- 원본 NVFP4 + SGLang DSpark/MTP 결과와 직접 비교할 조건이 아니다.

성능을 비교하려면 `Qwen3.8-27B-NVFP4`와 SGLang DSpark/MTP/DFlash2 레시피를 별도로 측정해야 한다. 모델 품질을 비교하려면 같은 prompt set, output budget, thinking 설정을 사용해야 한다.

## 6. 실제로 아직 모르는 것

- 이 파생 모델의 262K native context 품질
- YaRN 512K/1M에서의 retrieval/position 품질
- 비디오 입력 품질
- 여러 tool을 포함한 function calling 정확도와 장기 agent loop 성공률
- Claude Code/OpenCode/Pi 같은 코딩 에이전트의 장기 성공률
- SWE-bench, LiveCodeBench, MMLU 재현 결과
- 원본 Qwen3.8 대비 한국어·코딩·비전 능력의 세부 delta

이 항목들은 다음 실험에서 각각 따로 기록해야 한다.

## Sources

- [Qwen3.8-27B 커뮤니티 제작물·활용 사례 리서치](qwen38-community-builds-2026-08.md)
- [Qwen3.8-27B official model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [OBLITERATUS/Qwen3.8-27B-OBLITERATED model card](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)
- [NVIDIA DGX Spark vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
