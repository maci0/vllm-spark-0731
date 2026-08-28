# DeepSeek V4 Flash 0731 커뮤니티 제작물·응용 사례 리서치

조사일: **2026-08-22**

이 문서는 DeepSeek V4 Flash 0731을 사람들이 실제로 무엇에 연결했고, 어떤 주변 시스템과 실행 레시피를 만들었는지 정리한다. 단순히 모델이 메모리에 올라왔다는 사례와, OpenAI-compatible endpoint·tool loop·vision·동시 실행까지 확인한 사례를 구분한다.

## 먼저 결론

현재 커뮤니티가 만든 결과물은 크게 다섯 종류다.

1. 단일·듀얼 DGX Spark에서 V4 Flash를 서비스하는 실행 레시피
2. OpenClaw, Hermes, OpenCode, Cursor 같은 코딩 에이전트에 연결하는 endpoint·모델 설정
3. 텍스트 전용 DeepSeek에 작은 VLM 또는 별도 vision encoder를 붙이는 비전 계층
4. DeepSeek와 MiniMax H3·ComfyUI를 같은 Spark 클러스터에서 함께 돌리는 멀티모델 구성
5. 긴 문맥·needle·tool call·동시성·speculative decoding을 검증하는 benchmark 하니스

가장 완성도가 높은 것은 **텍스트용 OpenAI-compatible 서버**와 **2대 Spark의 1M 컨텍스트 서비스**다. 비전은 두 가지 접근이 모두 실험 중이고, 장시간 에이전트 loop는 parser보다 상위의 대화 형식·reasoning·framework 호환성 문제가 아직 남아 있다.

이 문서의 링크와 숫자는 2026-08-22에 확인한 원문을 기준으로 한다. GitHub README와 모델·runtime은 빠르게 바뀌므로, 출간할 때는 commit·image digest·모델 revision을 다시 고정한다.

## 1. 모델과 자료를 섞어 읽지 않기

공개 글에는 다음 네 가지가 자주 함께 등장한다.

| 구분 | 의미 | 이 문서에서의 처리 |
|---|---|---|
| 공식 `deepseek-ai/DeepSeek-V4-Flash-0731` | 0731 공식 checkpoint | 모델 카드·공식 revision과 함께 기록 |
| preview `DeepSeek-V4-Flash-DSpark` | 0731 이전 또는 별도 DSpark checkpoint | 0731 결과와 별도 행으로 유지 |
| full FP8/FP4 serving | 2대 이상에서 원본에 가까운 실행 | TP·RoCE·KV 설정을 함께 기록 |
| EXL3/REAP/GGUF/MLX 파생판 | 단일 장비를 위한 양자화·pruning·runtime 변형 | 원본 품질과 동일하다고 쓰지 않음 |

따라서 `47 tok/s`, `83 tok/s`, `211 tok/s`라는 숫자는 서로 다른 모델 파일과 workload의 결과일 수 있다. 비교할 때는 최소한 `checkpoint · quant · engine · speculative method · context · concurrency · prompt class · throughput definition`을 함께 확인한다.

공식 모델 카드는 0731을 preview를 대체하는 release로 설명하고, reasoning effort와 agent benchmark를 제공한다. 공식 checkpoint 자체는 text-only 경로이며, 비전 입력은 별도 계층이 필요하다. [DeepSeek V4 Flash 0731 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

## 2. 단일 Spark에서 만든 것

### 2.1 EXL3·SparkInfer·DSpark 서버

[MiaAI-Lab의 one-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)는 128GiB GB10 한 대에 실행할 수 있도록 만든 대표적인 커뮤니티 레시피다. 3.0 bpw EXL3/Trellis, REAP-pruned expert checkpoint, SparkInfer 계열 kernel, DSpark draft를 조합해 TP=1로 서비스한다.

README에 기록된 결과는 다음과 같다.

- `MAX_MODEL_LEN=384000`
- `MAX_NUM_SEQS=1`
- 한 부팅에서 약 439,622토큰 KV pool
- structured single-stream decode 약 44~47 tok/s
- 370,104토큰 needle 입력에서 exact recall
- 요청 초반 prefill 약 1,024 tok/s, 깊은 구간에서는 더 낮아짐

이 결과물의 핵심은 “284B/304B급 모델을 128GB 단일 장치에서 서비스 가능한 형태로 줄였다”는 데 있다. 반대로 full-expert FP8과 같은 모델 품질·기능이라고 볼 수는 없다. EXL3 3.0 bpw가 Q4~Q5 GGUF 체감에 가깝다는 문장은 recipe 작성자의 경험적 mapping이며, 동일 평가셋으로 독립 검증한 등가표가 아니다.

우리 저장소의 현재 단일 Spark 실행도 이 계열이다. 직접 확인한 것은 기본 API, C1 code gate, JSON·tool parser, 한 번의 mock tool loop다. 우리가 아직 만들지 않은 것은 370K needle 독립 재현, 실제 외부 API 오류 복구, 장시간 coding-agent 성공률이다. [DeepSeek 성능 리서치](deepseek-v4-flash-0731-performance-research-2026-08.md)

### 2.2 다른 단일 Spark 엔진으로 독립 재현

[emiluzelac의 one-Spark 재현](https://github.com/emiluzelac/deepseek-v4-flash-0731-on-one-dgx-spark)은 `Entrpi/ds4` CUDA serving engine으로 별도의 실험을 수행했다.

- prefill 약 1,000 input tok/s를 127K context에서도 관찰
- 단일 요청 생성 약 23~37 output tok/s
- 12개 동시 요청 aggregate 약 59.7 tok/s
- 127,532토큰에서 verification code 회수
- OpenAI-compatible chat completion과 forced tool call 성공

이 저장소가 특별히 강조하는 것은 prefill, 단일 사용자 decode, aggregate concurrency를 분리해야 한다는 점이다. 같은 59.7이라는 숫자라도 한 사용자가 받는 속도와 12명이 합산한 속도는 전혀 다르다.

### 2.3 단일 Spark 양자화·runtime 변형

커뮤니티에는 EXL3 외에도 IQ3M, EXL3 K2, GGUF, MLX 계열이 생겼다. 이 결과들은 단일 Spark·Mac·RTX 계열에서 “어떤 품질과 컨텍스트를 남기고 어떤 속도를 얻을지”를 고르는 실험이다.

- [0xSero one-Spark SparkInfer](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer): 단일 Spark용 pinned Docker·SparkInfer 경로
- [Y-Computer IQ3M recipe](https://github.com/Y-Computer/recipes/tree/main/benchmarks/published/2026-08-08-deepseek-v4-flash-0731-y-dspark-iq3m-dgx-spark): IQ3M과 speculative path 비교
- [tpurtell EXL3 K2 benchmark](https://github.com/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark): 단일 요청·동시성·prefill을 분리한 K2 측정
- [llama.cpp local experiment](https://dev.classmethod.jp/en/articles/dgx-spark-deepseek-v4-flash-0731-llama-cpp/): 일반 runner와 전용 engine을 비교한 단일 장비 실험

이 표본들은 “단일 Spark에서 공식 FP8을 그대로 돌렸다”는 증거가 아니다. 모델 artifact와 runtime이 달라질 때 속도와 KV headroom이 얼마나 달라지는지 보여주는 자료다.

## 3. 듀얼 Spark에서 만든 것

### 3.1 1M 컨텍스트 운영형 서버

[m9e의 2× DGX Spark recipe](https://github.com/m9e/deepseek-v4-flash-0731-2x-dgx-spark)는 공식 0731 revision을 두 GB10 노드에 TP=2/PP=1로 배치하는 production-shaped 구성을 목표로 한다.

- 2대 GB10, TP=2, RoCEv2
- native `nvfp4_ds_mla` KV cache와 FlashInfer B12X MoE
- DSpark probabilistic drafting
- 1M model length, 16 request slots, prefix caching
- sampled/max coding C1 중앙값 52.03 tok/s
- C6 aggregate 132.18 tok/s, C16 aggregate 211.38 tok/s
- cold 131K prefill 1,928.93 tok/s

이 결과는 단일 사용자 속도만이 아니라 여러 agent가 같은 supervisor를 공유할 때의 aggregate를 측정했다는 점이 중요하다. 두 200G RoCE rail을 사용하면 C1·C6·C16과 prefill이 각각 개선되지만, direct link와 switch fabric의 설정·운영 난이도도 함께 증가한다.

### 3.2 1M capacity·speculative decoding 측정

[MiaAI-Lab의 2× Spark 문서](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md)는 256, 2K, 8K, 32K, 131K prompt와 concurrency 1·2·4·6을 sweep한다.

대표적으로 다음 값이 기록돼 있다.

| Prompt | Concurrency | Decode | Aggregate |
|---:|---:|---:|---:|
| 256 | 1 | 75.4 tok/s | 69.1 tok/s |
| 256 | 6 | 36.9 tok/s/request | 191.2 tok/s |
| 2,048 | 1 | 68.8 tok/s | 62.0 tok/s |
| 8,192 | 1 | 73.9 tok/s | 43.7 tok/s |
| 32,768 | 1 | 64.0 tok/s | 16.6 tok/s |

별도 900K acceptance request는 약 874.8 prefill tok/s와 sentinel 응답을 기록했다. 이것은 “1M 설정이 실제로 큰 입력을 받을 수 있다”는 근거이지, 1M 모든 요청이 빠르거나 장문 추론 품질이 검증됐다는 뜻은 아니다.

### 3.3 speculative decoding을 직접 비교한 결과

[Weschera의 qualified summary](https://github.com/Weschera/DeepSeek-V4-Flash-0731-DSpark-2x-DGX-Spark/blob/main/results/qualified-summary.md)는 40K fixture에서 draft를 끈 경우 27.103 tok/s, DSpark K7 greedy에서 83.808 tok/s를 기록했고, draft acceptance는 84.57%였다.

같은 문서는 SparkBench v6.5에서 TrueScore 87.8, Pass@1 90.8%를 보고하지만, 이 결과는 1M 기본 profile이 아니라 40K·K7 profile을 사용했다. 따라서 83.8 tok/s를 1M·max-thinking·긴 에이전트 요청의 보장값으로 인용하지 않는다.

### 3.4 서로 다른 Spark 본체의 비교

[Reddit의 ASUS 대 NVIDIA 듀얼 Spark 비교](https://www.reddit.com/r/LocalLLM/comments/1vq5xjl/benchmark_deepseekv4flash_on_2x_dgx_sparks/)는 같은 0731·TP=2 stack을 두 하드웨어 구성에서 비교했다.

- 515K cold retrieval: 두 구성 모두 3/3 exact recall
- 30분 이상 stress: 오류·restart 없이 완료
- ASUS: 515K prefill 1,450.81 tok/s
- NVIDIA reference: 515K prefill 1,113.93 tok/s
- 6-worker sustained stream: ASUS 105.21, NVIDIA 95.50 tok/s

이 자료는 모델보다 host·firmware·메모리 headroom·열 상태가 결과에 영향을 줄 수 있다는 사례다. 다만 Reddit 측정이므로 raw environment와 commit을 확보하기 전에는 독립 재현 완료로 표시하지 않는다.

## 4. 사람들이 만든 에이전트 시스템

### 4.1 OpenAI-compatible supervisor endpoint

여러 프로젝트는 DeepSeek를 직접 UI에 넣기보다 OpenAI-compatible endpoint로 감싼다. 그러면 같은 모델을 다음 클라이언트에 연결할 수 있다.

- OpenClaw
- Hermes Agent
- OpenCode
- Cursor
- Continue
- Build Z Code
- Grok 계열 endpoint client

[tonyd2wild의 vision shim 문서](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)는 이 harness들을 한 URL 뒤에 연결하는 예시를 제공한다. 단, endpoint가 연결된 것과 장시간 에이전트 작업이 성공한 것은 별도다.

### 4.2 MCP·Codex 연결

[DeepSeek agent 저장소의 Codex MCP 이슈](https://github.com/deepseek-ai/awesome-deepseek-agent/issues/341)는 0731을 Codex에 연결했을 때 MCP 목록이 비어 보인 사례를 기록한다. 원인은 모델 자체가 아니라 `model.json`의 `supports_search_tool=true` 설정이었고, `false`로 바꾸자 MCP가 동작했다.

이 사례는 로컬 모델 연결에서 모델·parser·클라이언트 capability flag가 함께 맞아야 한다는 점을 보여준다. “MCP를 지원하는 모델인가”만 확인하면 부족하고, client가 어떤 tool schema와 capability를 endpoint에 보낼지 확인해야 한다.

### 4.3 아직 남은 agent loop 문제

[DeepSeek API tool-call loop 이슈](https://github.com/deepseek-ai/DeepSeek-V3/issues/1554)는 reasoning effort와 관계없이 한 문장만 출력하고 tool call 없이 `stop`으로 끝나는 현상을 보고한다. [Hermes Agent 이슈](https://github.com/NousResearch/hermes-agent/issues/78807)에도 0731에서 infinite reasoning loop가 발생하는 사례가 있다.

이 보고는 모든 배포에서 발생한다는 뜻은 아니다. API endpoint, chat encoding, reasoning content 보존, assistant/tool turn의 role boundary, agent retry 정책이 함께 작용할 수 있다. 우리 로컬에서는 한 번의 mock tool loop가 통과했지만, 이는 이 문제를 반박하는 장시간 agent benchmark가 아니다.

## 5. 사람들이 만든 비전 시스템

공식 0731 checkpoint는 text-only이므로 커뮤니티는 두 방향을 만들었다.

### 5.1 Caption shim

[tonyd2wild의 0731 vision shim](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)은 다음 구조다.

```text
harness
  → :8899 OpenAI-compatible shim
      → 이미지 요청만 :8081 작은 VLM으로 전달
      → caption을 :8888 DeepSeek에 전달
      → 결과를 원래 client로 반환
```

README는 0.8B VLM과 약 4.6초의 사진 설명 요청을 보고한다. 텍스트 요청은 기존 DeepSeek endpoint로 통과시키므로 기존 DS4를 재배포하지 않아도 된다. OpenClaw나 Cursor가 “화면을 보고 답하는” 형태를 빠르게 얻을 수 있다는 것이 장점이다.

단점도 명확하다. caption은 이미지 patch를 DeepSeek가 직접 보는 것이 아니므로 작은 글씨, 표, 정확한 좌표, 객체 간 공간 관계가 손실될 수 있다. OCR 중심 작업이면 detail mode나 더 큰 VLM을 사용해야 한다.

### 5.2 Vision encoder·adapter

[FlyCockpit의 2× Spark vision playbook](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks)은 DeepSeek 0731 backbone에 별도 vision encoder와 adapter를 붙여 OpenAI-compatible image endpoint를 제공한다.

- 2× DGX Spark TP 구성
- 약 167GB FP8 backbone과 별도 vision asset
- 화면·UI 설명은 강점으로 보고
- synthetic GUI 좌표 grounding은 동작하지만 실제 웹페이지로의 전이는 제한적
- cold start 약 6분

이 경로는 caption shim보다 native multimodal에 가깝지만, 별도 encoder·adapter의 revision과 호환성을 함께 고정해야 한다. “공식 0731이 vision을 지원한다”는 근거로 사용하지 않는다.

## 6. DeepSeek와 다른 생성 작업을 함께 돌린 것

[DS4 × H3 Video Gen Factory](https://github.com/tonyd2wild/ds4-h3-video-gen-factory)는 2대 Spark에서 DeepSeek V4 Flash 1M context와 MiniMax H3 영상 생성 두 인스턴스를 동시 실행한다.

- DeepSeek: TP=2, 1,048,576 context, 약 1.47M KV pool
- 영상: 노드별 MiniMax H3/ComfyUI, 15초 480p audio video
- 시작 순서: DeepSeek가 먼저 메모리를 확보한 뒤 H3를 시작
- DeepSeek C6 aggregate: idle 285.95 tok/s
- H3 한 개 동시: 130.77 tok/s
- H3 두 개 동시: 100.79 tok/s

이 결과물은 “LLM 서버가 모든 메모리를 독점해야 한다”는 생각을 뒤집는 co-tenancy 실험이다. 하지만 unified memory에서 실행 순서가 중요하고, 영상 모델을 먼저 띄우면 DeepSeek가 시작하지 못할 수 있다. 따라서 일반적인 1대 Spark 동시 실행의 근거로 일반화하지 않는다.

## 7. benchmark·운영 도구로 만든 것

커뮤니티는 모델을 띄우는 것보다 측정 도구도 많이 만들었다.

- 1M context capacity probe
- 515K·900K needle retrieval
- cold prefill과 warm decode 분리
- DSpark draft acceptance 계산
- c1/c4/c6/c16 aggregate throughput
- ASUS와 NVIDIA reference Spark 비교
- 30분 이상 concurrent soak
- 200G RoCE direct link와 dual-rail 비교
- CUDA graph mode와 B12X MoE backend 비교
- quantization별 KV pool·OS headroom 비교

이런 하니스가 중요한 이유는 speculative decoding에서 같은 장비도 prompt에 따라 속도가 크게 달라지기 때문이다. 코드·JSON·반복 패턴은 draft acceptance가 높고, 자연어·긴 reasoning은 낮아질 수 있다. 따라서 숫자와 함께 prompt class와 acceptance rate를 기록해야 한다.

## 8. 현재 우리 환경과의 차이

현재 우리 단일 Spark에서 실제로 떠 있는 것은 다음이다.

```text
DeepSeek V4 Flash 0731
  └─ one-Spark EXL3/SparkInfer/DSpark text endpoint :8888
```

직접 확인한 상태는 `serves`, `benchmarked`, `tool-tested`의 일부다.

| 항목 | 우리 상태 |
|---|---|
| 기본 API·모델 인식 | 직접 통과 |
| C1 semantic/JSON/code | 직접 통과 |
| tool parser | 직접 통과 |
| 한 번의 mock multi-turn tool loop | 직접 통과 |
| 370K needle | 아직 미실행 |
| native vision | 미구성 |
| caption vision shim | 미구성 |
| MiniMax H3 co-tenancy | 미실행 |
| 장시간 OpenClaw/Hermes agent | 미실행 |
| GPT-5.6-Sol 직접 비교 | 미실행 |

현재 `GPU_MEMORY_UTILIZATION=0.94`인 단일 Spark deep-context 서버에 작은 VLM이나 다른 27B 모델을 바로 추가하면 안 된다. vision shim과 Qwen worker는 별도 메모리 profile 또는 두 번째 Spark에서 구성해야 한다.

## 9. 우리 책에서의 권장 구성

### 한 대

```text
DeepSeek EXL3
  → text coding, long context, supervisor, tool parser
```

비전은 현재 서버를 줄이거나 별도 CPU/작은 VLM sidecar를 두는 실험으로 기록한다.

### 두 대

```text
Spark A + Spark B
  → DeepSeek FP8/DSpark TP=2
  → 1M context, 여러 agent, 선택적 vision sidecar
```

2대에서 DeepSeek와 Qwen을 각각 TP=1로 독립 실행할 수도 있지만, 그 경우 듀얼 Spark TP=2의 1M supervisor를 포기하는 trade-off가 생긴다. 무엇이 더 좋은지는 모델 품질보다 실제 workload와 동시성으로 결정한다.

### 네 대

```text
2×Spark: DeepSeek supervisor TP=2
2×Spark: Qwen worker 또는 별도 multimodal worker TP=2
```

이것은 하나의 통합 benchmark가 아니라 역할 분리 architecture다. DeepSeek는 계획·긴 context·도구·검토를 담당하고, Qwen은 UI·디자인·짧은 반복 작업을 맡기는 식이다.

## 10. 재현 큐

다음 실험을 해야 “사람들이 만든 것”을 우리 장비의 결과로 승격할 수 있다.

1. 단일 Spark에서 370K needle을 원문 조건으로 재현한다.
2. 2대 Spark에서 256K와 1M profile을 같은 prompt로 비교한다.
3. `thinking off`, `low`, `high`, `max`를 분리해 tool loop를 반복한다.
4. OpenClaw와 Hermes에서 20회 이상 tool call·error recovery를 측정한다.
5. caption shim을 붙이되 현재 0.94 memory profile과 충돌하지 않도록 별도 headroom을 확보한다.
6. 2대 Spark에서 DS4와 H3 co-tenancy를 직접 측정하지 않고는 단일 Spark 구성에 일반화하지 않는다.
7. 2×2 DeepSeek/Qwen 역할 분리는 네 대에서 독립 endpoint·router·권한 경계를 포함해 측정한다.

## 출처

- [DeepSeek V4 Flash 0731 공식 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [MiaAI-Lab one-Spark EXL3 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
- [0xSero SparkInfer runtime](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer)
- [emiluzelac one-Spark independent reproduction](https://github.com/emiluzelac/deepseek-v4-flash-0731-on-one-dgx-spark)
- [MiaAI-Lab two-Spark 0731 results](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md)
- [m9e two-Spark production-shaped recipe](https://github.com/m9e/deepseek-v4-flash-0731-2x-dgx-spark)
- [Weschera qualified summary](https://github.com/Weschera/DeepSeek-V4-Flash-0731-DSpark-2x-DGX-Spark/blob/main/results/qualified-summary.md)
- [tonyd2wild vision shim](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
- [FlyCockpit vision encoder](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks)
- [DS4 × H3 Video Gen Factory](https://github.com/tonyd2wild/ds4-h3-video-gen-factory)
- [Codex MCP integration issue](https://github.com/deepseek-ai/awesome-deepseek-agent/issues/341)
- [DeepSeek API tool-loop issue](https://github.com/deepseek-ai/DeepSeek-V3/issues/1554)
- [Hermes Agent 0731 issue](https://github.com/NousResearch/hermes-agent/issues/78807)
- [Reddit ASUS/NVIDIA dual-Spark comparison](https://www.reddit.com/r/LocalLLM/comments/1vq5xjl/benchmark_deepseekv4flash_on_2x_dgx_sparks/)
- [DGX Spark 원본 참고문헌 인덱스](dgx-spark-book-references-2026-08.md)
