# 06-5. DeepSeek V4 Flash 0731

전체 선택 기준은 [DGX Spark에서 돌릴 모델 선택](06-4-model-selection.md)에서 확인할 수 있습니다.

기준일: **2026-08-22**

## 한 줄 결론

DeepSeek V4 Flash 0731은 단일 Spark의 긴 문맥 supervisor와 두 대 Spark의 FP8·TP=2 서버를 비교할 수 있는 강한 후보다. 다만 단일 EXL3 recipe의 빠른 decode, 두 대 FP8 recipe의 context·prefill, GPT-5.6 Sol max와의 품질 비교는 서로 다른 주장이다.

## 실행 프로필

| 항목 | 현재 원고의 판정 |
|---|---|
| 단일 Spark | EXL3·SparkInfer·DSpark community recipe, 이 저장소는 C1 일부와 한 단계 tool loop를 직접 확인 |
| 두 대 Spark | FP8·TP=2·DSpark·NVFP4 KV community recipe 및 현장 측정 사례 |
| 주요 작업 | 긴 context, 코드 생성, supervisor·agent |
| 직접 검증 상태 | semantic·JSON·code decode·mock tool loop는 통과했지만 prefill gate는 통과하지 못함 |
| 수치 해석 | recipe 주장·커뮤니티 실측·로컬 실측을 서로 다른 행으로 기록 |

이 페이지의 결론은 [06-4 모델 선택](06-4-model-selection.md)의 선택 허브를 보완한다. 설치 순서는 [03-4 첫 모델 증명](03-4-single-spark-first-model.md), 비교 조건은 [10-4 Sol과 로컬 모델 비교](10-4-gpt56-sol-comparison.md)를 따른다.

## 3분 이해 (ELI5) — 모델 카드

DeepSeek는 긴 문서를 읽고 계획을 세우는 supervisor 후보다.

```text
긴 context + 큰 모델 → 깊은 작업 가능성
하지만 weight·KV·recipe 조건 → 먼저 검증
```

47 tok/s라는 한 숫자만으로 품질이나 agent 동등성을 말할 수 없다.

DeepSeek V4 Flash 0731은 DGX Spark에서 가장 흥미로운 모델이지만, 동시에 가장 쉽게 숫자를 잘못 옮길 수 있는 모델이다. 공식 text checkpoint, one-Spark EXL3 artifact, 2대 FP8 TP recipe, vision shim을 서로 다른 결과로 기록해야 한다.

## 공식 모델 카드에서 확인한 것

[공식 Hugging Face model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)는 다음을 명시한다.

- 0731은 preview를 대체하는 공식 release다.
- 같은 구조의 DSpark speculative-decoding 모듈을 포함한다고 설명한다.
- MIT 라이선스를 표시한다.
- Jinja chat template 대신 `encoding` 폴더의 메시지 인코더·출력 파서를 사용한다.
- `reasoning_effort`는 `low`, `high`, `max` 세 수준이다.
- vLLM과 SGLang 실행 예시, DSpark 옵션을 제공한다.

공식 카드의 vLLM·SGLang 예시는 4×GB300 등 다른 하드웨어를 포함한다. 그 명령을 1대 Spark나 2대 Spark에 그대로 복사하면 안 된다.

## Spark 실행 경로

### 단일 Spark: EXL3·SparkInfer·DSpark

[MiaAI-Lab의 one-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)는 EXL3 양자화, NVFP4 KV cache, SparkInfer와 DSpark speculative decoding을 조합한다. README에 단일 스트림 약 47 tok/s, 보수적인 384K context, 약 440K KV profile, 약 370K needle 통과가 보고돼 있다.

이 수치는 recipe 작성자의 조건이다. 우리 책에서는 다음처럼 기록한다.

```text
community recipe claim:
  EXL3 + SparkInfer + DSpark
  ~47 tok/s single stream
  384K conservative context
  ~370K needle

not yet implied:
  all prompts at 47 tok/s
  native vision
  long-running tool agent success
  GPT-5.6 Sol equivalence
```

단일 Spark에서 DeepSeek를 supervisor로 쓰고 Qwen3.8을 worker로 함께 띄우는 것은 이 recipe의 기본 조건이 아니다. 현재 profile의 KV와 workspace 여유를 먼저 계산하고, 별도 Spark endpoint로 분리하는 편이 안전하다.

### 2대: 원본 FP8·TP=2·DSpark

[2× DGX Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)는 공식 `deepseek-ai/DeepSeek-V4-Flash-0731` FP8 checkpoint를 대상으로 TP=2, vLLM/SGLang 계열 실행, NVFP4 MLA KV cache와 DSpark를 다룬다. [Nacyot의 현장 측정](https://nacyot.github.io/artifacts/deepseek-v4-flash-2x-dgx-spark/)은 1M을 설정하고 512K까지 측정했으며, prefill·decode·c1·aggregate·clock sweep을 분리했다.

기록할 때는 one-Spark EXL3와 다음 값을 같은 표에 합치지 않는다.

| 구분 | 단일 recipe | 2대 recipe |
|---|---|---|
| weight | EXL3 | 공식 FP8 |
| engine | SparkInfer/DSpark 경로 | vLLM/SGLang·DSpark 계열 |
| topology | 단일 | TP=2, CX-7/RoCE |
| context | 384K 보수 profile 주장 | 1M 설정, 512K 측정 사례 |
| 의미 | 개인 supervisor 후보 | 긴 context·대형 weight·다중 노드 후보 |

## DeepSeek를 선택할 때의 장점과 제한

| 장점 | 제한 |
|---|---|
| 긴 context와 agentic coding을 목표로 한 recipe가 빠르게 나옴 | weight·KV·workspace가 커서 1대 profile이 빡빡함 |
| DSpark가 target checkpoint에 붙어 speculative decoding을 구성 | reasoning effort·prompt class에 따라 acceptance와 속도가 달라짐 |
| OpenAI-compatible endpoint와 tool parser 경로가 있음 | 공식 model card의 일반 실행 예시와 Spark recipe를 혼동하기 쉬움 |
| 2대 TP=2에서 모델 크기와 context를 늘릴 수 있음 | CX-7가 socket fallback되면 통신 병목이 커짐 |

## 비전은 별도 판정

공식 0731 checkpoint를 text endpoint로 돌린 것과 vision 모델은 같은 말이 아니다. [DeepSeek vision shim](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)은 작은 VLM이 이미지를 caption으로 바꾼 뒤 기존 DeepSeek endpoint에 전달하는 구조다.

```text
OpenAI image request
  → caption/VLM shim
  → text prompt
  → DeepSeek V4 Flash endpoint
```

따라서 작은 글씨 OCR, 정확한 좌표, 표·공간 관계는 caption 손실을 별도로 평가해야 한다. native vision을 지원한다고 책에 쓰려면 이미지·문서·화면·도구 입력을 별도 하니스로 통과시킨다.

## 에이전트 판정

DeepSeek를 “brain”으로 부를 수 있으려면 최소한 다음을 확인한다.

- `reasoning_effort=low/high/max`별 응답 종료
- tool name과 JSON arguments 파싱
- tool 오류 후 재시도·복구
- 파일 수정 후 테스트 실행
- 20회 이상 반복하는 장시간 loop
- context가 커질 때 KV 부족·hang·restart 여부

공식 benchmark 점수와 우리 endpoint의 tool loop 성공률은 별도 필드다. 현재 이 장의 기본 상태는 recipe와 커뮤니티 실측을 `B/C`, 로컬 하니스 결과를 별도로 기록하는 것이다.

## 실행 전 체크리스트

- [ ] 모델 revision과 `encoding` 파서를 고정했다.
- [ ] EXL3와 FP8 결과를 분리했다.
- [ ] 1대·2대 topology와 `NET/IB` 경로를 확인했다.
- [ ] NVFP4 KV cache와 context headroom을 기록했다.
- [ ] c1 decode와 c4/c6 aggregate를 따로 측정했다.
- [ ] needle 통과를 장문 품질 전체의 증거로 과장하지 않았다.
- [ ] native vision과 caption shim을 구분했다.

## 참고

- [DeepSeek V4 Flash 0731 성능 리서치](../docs/deepseek-v4-flash-0731-performance-research-2026-08.md)
- [DeepSeek 커뮤니티 제작물](../docs/deepseek-v4-flash-0731-community-builds-2026-08.md)
- [DeepSeek V4 Flash 2× Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- [One-Spark EXL3 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
