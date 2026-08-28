# oh-my-dgx-spark-lab

**Personal AI machine laboratory for NVIDIA DGX Spark.**

이 공개 리포는 DGX Spark(GB10)를 하나의 AI 머신으로 활용하면서 수행하는 모델 서빙, 로컬 에이전트, 코딩, 벤치마크, 장문 문맥, 시스템 튜닝, 원샷 애플리케이션 실험을 기록합니다. 모델 카드나 다른 사용자의 benchmark 숫자를 그대로 재사용하지 않고, 공식 자료·재현 가능한 레시피·우리 장비의 직접 실측을 구분합니다.

## Lab scope

```text
Machine       DGX Spark / GB10 / 128 GiB unified memory
Serving       vLLM, SGLang, llama.cpp, Docker, NVFP4, speculative decoding
Models        DeepSeek V4 Flash, Qwen, Nemotron and other Spark-compatible models
Agents        Hermes Agent, local coding agents, tool calling and structured output
Experiments   Thinking, KV cache, throughput, ABLATE, one-shot builds and smoke tests
Knowledge     Research notes, reproducible recipes, benchmarks and WikiDocs chapters
```

## Current DeepSeek V4 Flash experiment

A single-DGX-Spark DeepSeek V4 Flash serving experiment is running through the [MiaAI-Lab one-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark). The verified serving profile is:

```text
Model              deepseek-v4-flash-0731
Endpoint           http://127.0.0.1:8888/v1
Runtime            Docker + NVIDIA vLLM/Sparkinfer
Tensor parallel    1
Context            384,000 tokens
Max sequences      1
KV cache           NVFP4 / B12X_MLA_SPARSE
Speculative        DSpark, K=5
Thinking           enabled by default
Reasoning effort   max by default
```

The optional ABLATE research path is kept separate from normal serving:

```text
ABLATE              1 for the current refusal-behavior experiment
Lambda              3.5
Layers              10-42
```

ABLATE is recorded as a refusal-direction research feature, not as a general intelligence, speed, or coding-quality switch. Its effect on this exact EXL3 deployment requires separate controlled evaluation.

## DS4 one-shot experiment

The first complete one-shot implementation task used one user prompt with the local DS4 agent:

> Build a complete local web dashboard for the currently running DGX Spark DeepSeek V4 Flash server. Monitor health, served model, KV cache, and token throughput. Provide a selectable thinking/reasoning effort option and send test requests.

The resulting dashboard monitors the local server, parses Prometheus metrics, exposes thinking controls, sends test requests, and displays reasoning, content, latency, finish reason, and token usage.

Private implementation repository:

- [recrack/ds4-monitor-dashboard](https://github.com/recrack/ds4-monitor-dashboard)

Verified result:

```text
45 unit tests passed
12 live smoke tests passed
Dashboard: http://127.0.0.1:8899
```

## Research and evidence policy

- Prefer live API responses, container inspection, logs, tests, and Git history over model self-reports.
- Treat generated commands, performance numbers, and undocumented capabilities as unverified until executed.
- Record failed tests and environmental blockers instead of replacing them with plausible results.
- Keep private server addresses, credentials, tokens, personal keys, and sensitive logs out of this public repository.
- Use the related [private WikiDocs repository](https://github.com/recrack/oh-my-dgx-spark-wikidocs) for private publishing and deployment material when required.

현재 초점은 [`OBLITERATUS/Qwen3.8-27B-OBLITERATED`](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)와 DeepSeek V4 Flash 실험입니다. 이 모델들은 서로 다른 실행 경로와 검증 범위를 가지므로 결과를 서로 일반화하지 않습니다.

## 결론부터: 어느 정도 가능한가

현재 GB10에서 확인한 범위에서는 **개발 보조와 일반 대화에 충분히 사용할 수 있는 27B급 모델**입니다.

노드 수별 모델 선택과 1대, 2대, 3대, 4대 클러스터 구성은 [DGX Spark 모델·클러스터 리서치](docs/dgx-spark-cluster-model-research-2026-08.md)에 별도로 정리했습니다. 이 문서에서는 공식 레시피, 재현 가능한 GitHub 실험, 한국 커뮤니티 경험담, 우리 장비에서 직접 실행한 테스트를 구분합니다.

책 집필을 위해 수집한 서적, 공식 문서, 플레이북, 모델 카드, 벤치마크, GitHub, 커뮤니티 자료는 [로컬 AI 실전 레시피: DGX Spark 편 참고문헌](docs/dgx-spark-book-references-2026-08.md)에 분류해 두었습니다. 자료를 공식 원문, 재현 레시피, 실측, 커뮤니티 사례로 나누고, 책에 사용할 때 적용할 검증 기준도 함께 기록합니다.

NVIDIA Developer Forum에서 확인한 DGX Spark/GB10 관련 흐름은 [포럼 리서치](docs/dgx-spark-nvidia-forum-research-2026-08.md)에 정리했습니다. Qwen, DeepSeek, GLM, MiniMax, Nemotron, 1·2·3·4·8대 클러스터, vLLM, SGLang, llama.cpp, RoCE/NCCL, 발열·OOM·저클럭·의도적인 clock cap·펌웨어 장애, Hermes/OpenClaw/NemoClaw와 벤치마크 자료를 원문 등급과 함께 구분합니다.

DGX Spark와 Mac의 표준 혼합 구성, Apple MLX/JACCL과 Spark RoCE의 차이, 2·3·4대에서의 스위치 조건, 아직 검증되지 않은 MCDMA USB-C 프로토타입은 [Mac·RDMA·스위치 리서치](docs/dgx-spark-mac-rdma-switch-research-2026-08.md)에 정리했습니다.

WikiDocs 집필 구조와 장별 상태는 [로컬 AI 실전 레시피: DGX Spark 편 책 안내](book/README.md)와 [원고 목차](book/TOC.md)에서 관리합니다. 00~10장과 부록 A·B는 독자의 상위 읽기 경로이며, 기존 상세 원고는 각 장 아래의 서브챕터로 보존합니다. [WikiDocs 배포 bundle](wikidocs/)에는 `README.md`, `TOC.md`, `pages/`, `assets/` 구조를 생성하고, 날짜별 리서치 보고서는 부록 B 아래에 동적으로 추가합니다.

WikiDocs는 [`book/`](book/)을 원고 원본으로 사용합니다. `main`에 push하면 Actions가 bundle을 만들고 [`oh-my-dgx-spark-wikidocs`](https://github.com/recrack/oh-my-dgx-spark-wikidocs)에 배포한 뒤, WikiDocs webhook이 책을 동기화합니다. 최초 한 번의 secret 설정과 자세한 운영 규칙은 [WikiDocs 배포 계획](docs/wikidocs-deployment-2026-08.md)에 기록했습니다.

리서치 자료는 먼저 [자동화 운영 문서](docs/research-automation.md)와 [승격 규칙](docs/research-promotion.md)에 따라 기록·검증하고, 확인된 내용만 책 본문으로 옮깁니다. 전체 운영 흐름은 [research-promotion-flow.svg](docs/research-promotion-flow.svg)에서 확인할 수 있습니다.

| 영역 | 판단 | 근거 |
|---|---|---|
| 한국어·일반 텍스트 | 사용 가능 | 실제 한국어 질의 통과 |
| Python 코드 | 사용 가능 | 중복 제거 함수 생성 통과 |
| JSON/구조화 출력 | 사용 가능 | vLLM JSON 제약 응답이 유효 JSON으로 통과 |
| Thinking 추론 | 사용 가능 | thinking off/on 모두 통과; on에서는 reasoning 필드 확인 |
| 멀티턴 대화 | 사용 가능 | 식별자 회수 테스트 통과 |
| 이미지 이해 | 사용 가능 | 단일 JPEG smoke test 통과; 정량적 비전 평가 아님 |
| 긴 문맥 | 현재 설정에서 사용 가능 | 10,442 및 32,035 prompt tokens 통과; 서버는 32K로 실행 중 |
| 도구 호출 | 사용 가능 | `qwen3_xml` parser와 auto tool choice를 켠 단일 Spark에서 명시적 함수 호출 통과 |
| 비디오·1M 문맥·장시간 에이전트 | 미검증 | 공식 카드의 기능 설명은 있으나 이 파생 모델/장비에서 재현하지 않음 |

따라서 현재 바로 추천할 수 있는 용도는 코드 작성, 문서 요약, 분석, JSON 변환, 이미지 설명, 로컬 챗입니다. SWE-bench 같은 정량적 코딩 에이전트 성능과 복잡한 MCP·도구 루프는 별도로 평가해야 합니다.

## 현재 테스트 환경

- Hardware: NVIDIA DGX Spark / GB10 / 128 GiB unified memory
- Runtime: vLLM 0.26.0, PyTorch 2.11.0+cu130
- Model weights: BF16, checkpoint 약 51.75 GiB
- Current test server: `http://127.0.0.1:8083/v1`
- Served model name: `qwen3.8-27b-obliterated`
- Server limit: `max-model-len=32768`, `gpu-memory-utilization=0.50`, `max-num-seqs=4`
- Tool/reasoning: `--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3`, auto tool choice enabled
- Existing server: `http://127.0.0.1:8082/v1` (`QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`)와 동시 실행 가능

## 실행

```bash
scripts/run-qwen38-vllm.sh
```

간단한 요청:

```bash
curl http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-obliterated",
    "messages": [{"role": "user", "content": "한국어로 한 문장만 답하세요. 2+2는?"}],
    "max_tokens": 64,
    "temperature": 0,
    "repetition_penalty": 1.15,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

이 래퍼는 vLLM venv의 `ninja`, ARM64 Python 개발 헤더, Qwen3.8 내부의 XML chat template에 맞는 `qwen3_xml` parser를 함께 설정합니다. 직접 실행할 때도 `--enable-auto-tool-choice`와 `--tool-call-parser qwen3_xml`을 빠뜨리지 않습니다.

## 재현 테스트

외부 의존성 없는 안전한 smoke test입니다. 기본 테스트는 일반 텍스트, 코드, JSON, 멀티턴 회수, thinking, 8K 문맥, 4-way concurrency를 확인합니다.

```bash
python3 tests/qwen38_smoke.py \
  --base-url http://127.0.0.1:8083/v1 \
  --model qwen3.8-27b-obliterated
```

이미지 테스트와 32K 문맥 테스트는 명시적으로 켭니다.

```bash
python3 tests/qwen38_smoke.py \
  --base-url http://127.0.0.1:8083/v1 \
  --model qwen3.8-27b-obliterated \
  --image /path/to/image.jpg \
  --long-context-tokens 32000
```

도구 호출은 일반 생성이 통과했다는 사실만으로 판단하지 않습니다. parser를 켠 서버에서 지정한 함수와 JSON arguments가 실제 응답에 들어오는지 별도로 확인합니다. 이번 실행은 [실측 결과 JSON](docs/results-qwen38-vllm-parser-2026-08-21.json)에 남겼습니다.

```bash
python3 tests/tool_call_smoke.py \
  --base-url http://127.0.0.1:8083/v1 \
  --model qwen3.8-27b-obliterated \
  --strict
```

고정 prompt 반복 측정은 streaming TTFT와 end-to-end completion throughput을 JSON으로 출력합니다. `completion_tok_s_e2e`에는 prefill과 decode가 함께 들어가므로 순수 decode tok/s로 쓰지 않습니다.

```bash
python3 tests/repeat_benchmark.py \
  --base-url http://127.0.0.1:8083/v1 \
  --model qwen3.8-27b-obliterated \
  --warmups 2 \
  --trials 5 \
  --concurrency 1 \
  > results-qwen38-c1.json
```

서버가 내려가 있거나 tool parser 없이 실행 중이면 위 테스트는 실패합니다. 그 결과를 모델의 기능 부재로 해석하지 말고, endpoint·parser·model revision·실행 시각을 함께 기록합니다.

## 모델 카드와 실제 결과의 차이

원본 [Qwen3.8-27B 공식 카드](https://huggingface.co/Qwen/Qwen3.8-27B)는 코딩, 연구, 에이전트 작업, 이미지·비디오 이해, thinking 제어, native 262K 문맥, YaRN 기반 1M 확장을 설명합니다. 이는 원본 모델의 기능 설명이며, 파생 모델에서 모두 동일하게 재현되었다는 뜻은 아닙니다.

[`OBLITERATUS/Qwen3.8-27B-OBLITERATED`](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) 카드는 안전 거부 동작을 줄이는 것을 목표로 한 파생 모델이라고 설명하며, 자체 평가에서 원본 대비 MMLU가 87.4%에서 81.4%로 낮아졌다고 보고합니다. 0% refusal 같은 수치는 모델 제작자의 주장으로 기록하고, 이 리포에서는 위험한 프롬프트를 재현하지 않습니다.

즉, 이 모델은 **원본보다 모든 면에서 좋아진 모델이 아니라, 응답 제한을 줄이는 대신 일부 일반 능력과 안전 동작을 바꾼 연구·로컬 사용용 모델**로 보는 편이 정확합니다.

## DGX Spark에서 비교할 모델

이 리포에서는 특정 모델 하나의 결과를 모든 모델에 일반화하지 않습니다.

- 이 리포에서 직접 확인: `OBLITERATUS/Qwen3.8-27B-OBLITERATED` BF16
- 원본 Qwen3.8 SGLang 레시피: [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)
- 원본 Qwen3.8 양자화 경로: NVFP4 / FP8 / BF16
- 단일 Spark DeepSeek V4 Flash 0731 최신 EXL3 레시피: [MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
- DeepSeek V4 Flash 0731 성능 판정·단일/2대 측정·GPT-5.6-Sol 비교: [성능 리서치](docs/deepseek-v4-flash-0731-performance-research-2026-08.md)
- GPT-5.6 Sol(`gpt-5.6-sol`, `reasoning_effort=max`) 비교: [Sol max 리서치](docs/sol-max-comparison-research-2026-08.md)와 [WikiDocs 10장](book/10-decision.md)
- DeepSeek V4 Flash 0731 커뮤니티 제작물: [원문 리서치](docs/deepseek-v4-flash-0731-community-builds-2026-08.md)와 [WikiDocs 06장](book/06-model-recipes.md)
- Qwen3.8-27B 커뮤니티 제작물과 활용 사례: [원문 리서치](docs/qwen38-community-builds-2026-08.md)와 [WikiDocs 06장](book/06-model-recipes.md)
- DGX Spark 독자 질문과 초반 장 재설계: [질문 리서치](docs/dgx-spark-reader-questions-research-2026-08.md)와 [WikiDocs 00~04장](book/README.md)
- 2026-08-21·22 단일 Spark 실제 실행 기록: 위 성능 리서치의 `3.1`·`3.2`와 `acceptance_c1.py 정식 실행` 절에 레시피 gate, tool call, 생성 속도, 하니스의 token accounting 차이를 기록했습니다. `3.3`절에는 X, GitHub, NVIDIA Developer Forum, Reddit, 국내 커뮤니티, 공식 카드와 현장 실측을 등급별로 구분했습니다.
- NVIDIA 공식 vLLM Spark 지원표: [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#model-support-matrix)
- 다음 후보: `nvidia/Qwen3.6-35B-A3B-NVFP4`, Qwen3-VL 계열, Nemotron·Phi 계열

`fits in 128 GB`, `vLLM 지원`, `SGLang에서 speculative decoding까지 검증`은 서로 다른 상태입니다. 앞으로 모델을 추가할 때는 이 세 상태를 따로 기록합니다.

## 다음에 할 일

1. 비공개 `recrack/oh-my-dgx-spark-wikidocs`에 `wikidocs/` bundle을 push하고 책 화면에서 목차·링크를 확인
2. DeepSeek 공식 Harness와 동일 조건의 품질 평가가 공개되면 local C1과 별도 비교
3. 원본 Qwen3.8 NVFP4 + SGLang DSpark/MTP와 BF16 OBLITERATED 품질·속도 비교
4. 262K/YaRN 문맥 검증
5. Python/JSON/vision/agent 시나리오를 고정 프롬프트로 반복 평가
