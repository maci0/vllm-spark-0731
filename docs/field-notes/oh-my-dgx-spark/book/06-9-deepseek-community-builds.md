# 06-9. DeepSeek로 사람들이 만든 것

상태: 리서치 기반 초안

기준일: **2026-08-22**

DeepSeek V4 Flash 0731을 검색하면 속도 숫자가 먼저 보인다. 하지만 커뮤니티가 실제로 만든 것은 단순한 서버보다 훨씬 넓다. 단일 Spark용 양자화 레시피, 듀얼 Spark 1M 컨텍스트 서버, 코딩 에이전트 연결, 비전 shim, 영상 생성과의 동시 실행까지 주변 시스템이 빠르게 만들어지고 있다.

이 장에서는 “모델이 실행된다”와 “사용할 수 있는 제품 형태가 됐다”를 구분한다. 공개 수치는 모델 파일, runtime, speculative decoding, context, concurrency와 prompt가 다르므로 단일 순위표로 합치지 않는다.

> 이 장은 커뮤니티가 만든 결과물과 재현 후보를 모아 두는 사례 장이다. 현재 장비에서 무엇부터 실행할지는 [06-4 모델 선택](06-4-model-selection.md)과 [06-5 DeepSeek 실행 프로필](06-5-deepseek-v4-flash.md)에서 결정한다. 아래 수치는 출처의 주장 또는 커뮤니티 실측이며, 이 저장소의 직접 실측은 별도로 표시한다.

## 3분 이해 (ELI5)

이 장은 완성품 전시장이 아니라 실험 노트 모음이다.

```text
커뮤니티 주장 → 조건 확인 → 재현 후보 → 본문 승격
```

각 숫자는 같은 경기의 결과가 아니라 서로 다른 실험 기록이다.

## 15.1 사람들이 만든 결과물 한눈에 보기

| 결과물 | 무엇을 해결하나 | 대표 구성 |
|---|---|---|
| 단일 Spark DeepSeek 서버 | 128GB 장비에서 큰 MoE를 서비스 | EXL3·REAP·SparkInfer·DSpark |
| 듀얼 Spark supervisor | 원본에 가까운 모델과 긴 context | FP8, TP=2, RoCE, NVFP4 KV |
| 코딩 에이전트 endpoint | 모델을 여러 클라이언트에서 공유 | OpenAI API, OpenClaw, Hermes, OpenCode |
| vision shim | 텍스트 모델에 이미지 입력 추가 | 작은 VLM captioner + proxy |
| vision encoder 경로 | 이미지·화면을 별도 encoder로 처리 | 2×Spark + vision adapter |
| 영상 동시 실행 | LLM과 ComfyUI를 함께 운영 | DeepSeek + MiniMax H3 |
| benchmark 하니스 | 긴 문맥·속도·동시성 검증 | needle, soak, acceptance, draft rate |

## 15.2 단일 Spark: 큰 모델을 서비스 형태로 줄이기

[MiaAI-Lab의 one-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)는 단일 GB10에서 DeepSeek를 실행하는 대표적인 결과물이다. 3.0 bpw EXL3/Trellis와 REAP-pruned checkpoint를 SparkInfer·DSpark와 결합해 TP=1 서버로 만든다.

README가 보고한 프로필은 `MAX_MODEL_LEN=384000`, 단일 sequence, 약 440K KV pool, structured decode 44~47 tok/s다. 370,104토큰 needle 입력에서 정확히 marker를 회수한 기록도 있다.

이 레시피가 보여주는 것은 “128GB 안에서 실행 가능한 serving artifact가 있다”는 사실이다. full FP8·full expert 원본과 같은 품질이라고 증명한 것은 아니다. EXL3 3.0 bpw가 Q4~Q5 GGUF처럼 느껴진다는 설명도 커뮤니티 경험값이므로, 책에서는 품질 동등성으로 단정하지 않는다.

다른 엔진의 독립 실험도 있다. [emiluzelac의 one-Spark 재현](https://github.com/emiluzelac/deepseek-v4-flash-0731-on-one-dgx-spark)은 단일 요청 23~37 tok/s, 12개 동시 요청 aggregate 약 59.7 tok/s, 127K 토큰 회수, forced tool call 성공을 기록했다. 이 사례는 prefill, 한 사용자의 decode, 여러 사용자의 합산 처리량을 구분해야 한다는 점을 잘 보여준다.

## 15.3 듀얼 Spark: supervisor를 제품처럼 운영하기

공식 0731에 가까운 경로를 쓰려는 사람들은 대체로 2대 Spark를 TP=2로 묶는다. [m9e의 recipe](https://github.com/m9e/deepseek-v4-flash-0731-2x-dgx-spark)는 2대 GB10, RoCEv2, native NVFP4 MLA KV, DSpark, 1M model length와 16 request slots를 묶은 운영형 구성이다.

보고된 값은 다음과 같다.

- sampled/max coding C1: 중앙값 52.03 tok/s
- C6 aggregate: 132.18 tok/s
- C16 aggregate: 211.38 tok/s
- cold 131K prefill: 1,928.93 tok/s

이 숫자는 한 명의 응답 속도가 아니라 여러 요청의 합산 속도를 포함한다. 실제 supervisor를 여러 agent가 공유할 때 중요한 지표가 단일 decode보다 aggregate와 error rate일 수 있다는 뜻이다.

[MiaAI-Lab의 2대 측정](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md)은 256토큰 prompt 단일 요청에서 75.4 tok/s, 동시 6개에서 aggregate 191.2 tok/s를 기록했다. 별도 900K acceptance request는 약 875 prefill tok/s로 sentinel을 반환했다.

하지만 1M 설정이 곧 1M 품질을 뜻하지는 않는다. 긴 입력을 받아 sentinel을 돌려주는 capacity test와 문서 요약·코드 수정·needle 위치 다양화 같은 품질 평가를 분리해야 한다.

### 3.3a 원본 FP8·듀얼 레일 현장 실측

[ASUS GX10 2대 원본 FP8 실측](https://nacyot.github.io/artifacts/deepseek-v4-flash-2x-dgx-spark/)은 one-Spark EXL3와 다른 경로다. 공식 `deepseek-ai/DeepSeek-V4-Flash-0731` revision `9e165c30`(166.9GB)을 vLLM `mp`·TP=2·DSpark k=5·`nvfp4_ds_mla` KV cache로 구성하고, ConnectX-7 RoCE v2 dual rail을 사용했다.

작성자는 1,048,576을 서버 설정으로 두고 512K까지 측정했으며, context 256토큰 단일 decode 약 56.1에서 512K 약 45.9 tok/s로 낮아지는 결과를 보고했다. 동시 12개·256토큰에서는 aggregate 206.9 tok/s를 기록했다. 이 수치는 단일 사용자 속도와 서버 합산 처리량을 분리해서 읽어야 한다.

clock sweep에서도 2000MHz cap은 단일 decode 54.4 tok/s·16K prefill 1,794 tok/s, cap 해제는 55.0 tok/s·1,953 tok/s로 보고됐다. 반면 전력은 양 노드 합 기준 약 38.6W에서 78.4W로 증가한다. 이 결과는 원본 FP8·2대·작성자 이미지 조건이며, one-Spark EXL3·단일 `llama-server`·다른 2대 recipe와 합산하지 않는다. 1M 설정은 capacity 근거이지 1M 장문 품질 인증이 아니다.

## 15.4 에이전트: DeepSeek를 brain으로 쓰기

커뮤니티는 DeepSeek를 UI에 직접 박기보다 OpenAI-compatible endpoint로 제공하고, 클라이언트가 모델을 바꾸어 끼우게 만들었다. [vision shim 프로젝트](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)는 OpenClaw, Hermes, Cursor, Continue, Build Z Code 같은 클라이언트에 연결하는 예시를 제공한다.

역할을 나누면 다음과 같다.

```text
router
  ├─ 계획·긴 문서·도구·복구 → DeepSeek supervisor
  ├─ UI·CSS·짧은 반복 작업 → Qwen worker
  └─ 이미지·화면 → 별도 VLM 또는 vision encoder
```

Codex MCP 연결에서는 모델보다 capability 설정이 문제인 사례도 있었다. [관련 이슈](https://github.com/deepseek-ai/awesome-deepseek-agent/issues/341)는 `supports_search_tool=true` 때문에 MCP 목록이 비어 보였고, `false`로 바꾸자 동작했다고 기록한다.

그렇다고 agent-ready를 자동으로 보장할 수는 없다. [DeepSeek API tool loop 이슈](https://github.com/deepseek-ai/DeepSeek-V3/issues/1554)에는 tool call 없이 한 문장만 출력하고 멈추는 현상이, [Hermes Agent 이슈](https://github.com/NousResearch/hermes-agent/issues/78807)에는 무한 reasoning loop가 보고돼 있다. parser 한 번 통과한 것과 장시간 에이전트 성공률은 별도의 시험이다.

## 15.5 텍스트 모델에 눈을 붙인 방법

공식 0731은 text-only 모델이다. 사람들이 만든 vision 결과물은 두 갈래로 나뉜다.

### Caption shim

[tonyd2wild의 shim](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)은 `:8899`에서 이미지를 받고, 작은 0.8B VLM으로 설명문을 만든 뒤, 그 문장을 기존 `:8888` DeepSeek에 전달한다.

```text
OpenAI client → :8899 shim → :8081 VLM caption → :8888 DeepSeek
```

사진 설명 요청을 약 4.6초에 처리한 사례가 있다. 기존 DS4를 다시 배포하지 않고도 스크린샷·사진·카메라 프레임을 받는다는 점이 장점이다.

대신 caption에서 정보가 손실된다. 작은 글자, 정확한 좌표, 표와 복잡한 공간 관계를 다루는 작업에는 native vision 또는 별도 OCR 파이프라인이 더 적합하다.

### Vision encoder·adapter

[FlyCockpit의 2× Spark vision playbook](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks)은 별도 vision encoder와 adapter를 DeepSeek backbone에 붙인다. 화면과 UI 설명을 주요 대상으로 삼고, synthetic GUI 좌표 grounding은 확인했지만 실제 웹페이지 전이는 제한적이라고 기록한다.

이 방식은 caption shim보다 깊지만, backbone·encoder·adapter revision을 모두 맞춰야 하고 cold start와 디스크·메모리 비용이 커진다. 공식 0731 모델 자체가 native vision을 제공한다고 해석하면 안 된다.

## 15.6 LLM과 영상 생성을 동시에 만든 사례

[DS4 × H3 Video Gen Factory](https://github.com/tonyd2wild/ds4-h3-video-gen-factory)는 DeepSeek를 돌리는 2대 Spark에서 MiniMax H3·ComfyUI 영상 생성 두 개를 함께 실행했다.

- DeepSeek: 1M context, TP=2, 약 1.47M KV pool
- H3: 노드당 독립 인스턴스, 15초 480p 오디오 영상
- DS4 C6 aggregate: 영상 없이 285.95 tok/s
- H3 한 개와 동시: 130.77 tok/s
- H3 두 개와 동시: 100.79 tok/s

여기서 가장 중요한 것은 숫자보다 시작 순서다. DeepSeek를 먼저 띄워 메모리를 확보한 뒤 H3를 시작해야 한다. H3를 먼저 띄우면 영상 모델이 메모리를 잡아 DeepSeek가 기동하지 못할 수 있다.

이것은 1대 Spark에서 DeepSeek와 Qwen3.8 BF16을 동시에 띄울 수 있다는 근거가 아니다. 2대 Spark의 unified memory headroom과 모델별 eviction 동작에 의존한 별도 co-tenancy 실험이다.

## 15.7 사람들이 만든 benchmark 도구

커뮤니티가 만든 하니스는 다음을 실제로 측정한다.

- 515K·900K·1M 입력 capacity
- needle retrieval
- cold prefill과 warm decode
- DSpark draft acceptance
- c1/c4/c6/c16 동시성
- 30분 이상 soak와 restart 여부
- ASUS와 NVIDIA reference Spark의 차이
- direct QSFP/RoCE와 dual-rail 차이

speculative decoding은 prompt에 따라 속도가 달라진다. 코드와 JSON처럼 예측 가능한 출력은 acceptance가 높고, 자연어·긴 reasoning은 낮아질 수 있다. 따라서 `tok/s`만 복사하지 말고 prompt class, accepted tokens, TTFT, per-request와 aggregate를 함께 기록해야 한다.

## 15.8 우리 장비에서 이미 확인한 것과 아직 안 한 것

현재 단일 Spark에는 다음 형태의 DeepSeek만 떠 있다.

```text
DeepSeek V4 Flash 0731
  └─ EXL3/SparkInfer/DSpark text endpoint :8888
```

| 항목 | 상태 |
|---|---|
| 기본 API·모델 인식 | 직접 통과 |
| C1 semantic·JSON·code | 직접 통과 |
| tool parser | 직접 통과 |
| mock multi-turn tool loop | 직접 통과 |
| 370K needle | 아직 안 함 |
| native vision | 안 함 |
| caption shim | 안 함 |
| MiniMax H3 동시 실행 | 안 함 |
| 장시간 OpenClaw/Hermes | 안 함 |
| GPT-5.6-Sol 직접 비교 | 안 함 |

현재 서버는 `gpu-memory-utilization=0.94`인 deep-context profile이라 작은 VLM이나 Qwen3.8 BF16을 바로 추가하면 안 된다. vision·worker를 붙이려면 메모리 profile을 다시 만들거나 별도 Spark를 사용해야 한다.

## 15.9 구성 선택

| 장비 | 가장 현실적인 결과물 |
|---:|---|
| 1대 | DeepSeek EXL3 텍스트 supervisor 또는 Qwen worker 중 하나 |
| 2대 | DeepSeek TP=2, 1M supervisor와 여러 agent |
| 3대 | DeepSeek TP=2 + 별도 worker 1대, 또는 DP=3 |
| 4대 | DeepSeek TP=2 + Qwen TP=2 역할 분리, 각자 독립 endpoint |

DeepSeek와 Qwen을 각각 TP=2로 띄우는 2×2 구성은 네 대가 필요하다. 두 대에 두 모델을 억지로 함께 올리는 구성은 모델 크기·KV·통신까지 다시 측정해야 하므로 공식 배치가 아니라 실험으로 남긴다.

## 15.10 다음에 우리가 재현할 것

1. one-Spark 370K needle을 원문 조건으로 재현한다.
2. 256K와 1M profile에서 같은 prompt set을 비교한다.
3. `thinking off/low/high/max`별 tool loop를 반복한다.
4. OpenClaw·Hermes에서 tool call과 error recovery를 20회 이상 측정한다.
5. 현재 0.94 profile과 충돌하지 않는 caption shim headroom을 계산한다.
6. 2대 Spark에서 DeepSeek + H3 co-tenancy를 재현한다.
7. 2×2 DeepSeek/Qwen router에 권한 경계와 fallback을 붙인다.

## 참고 문서

- [DeepSeek V4 Flash 0731 성능 리서치](../docs/deepseek-v4-flash-0731-performance-research-2026-08.md)
- [DeepSeek 커뮤니티 제작물·응용 사례 원문 리서치](../docs/deepseek-v4-flash-0731-community-builds-2026-08.md)
- [로컬 에이전트 운영](08-1-local-agent-operations.md)
- [두 대 연결하기](07-1-two-spark-cluster.md)
- [부록: 모델·레시피·명령어 색인](appendix-a-1-model-recipe-command-index.md)
