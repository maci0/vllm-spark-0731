# 06-12. Qwen3.8로 사람들이 만든 것

상태: 리서치 기반 초안

기준일: **2026-08-22**

Qwen3.8-27B는 모델 카드만 읽으면 코딩과 에이전트 작업을 잘하는 27B 모델처럼 보인다. DGX Spark에 올려 보면 첫 인상은 조금 다르다. dense 모델이라 기본 경로의 decode는 메모리 대역폭에 묶이고, MTP·DSpark·DFlash2·NVFP4·GDN 최적화를 붙여야 커뮤니티가 말하는 속도에 가까워진다.

이 장에서는 사람들이 실제로 만든 serving recipe와 에이전트 활용 사례를 정리한다. 원본 모델, 최적화된 serving artifact, 우리가 직접 테스트한 파생 모델은 서로 다른 결과로 기록한다.

> 이 장은 Qwen3.8-27B의 커뮤니티 recipe와 활용 사례를 보관하는 사례 장이다. 모델 선택과 첫 실행 순서는 06-4 모델 선택과 06-6 Qwen3.8 실행 프로필을 우선한다. 원본 Qwen3.8, 최적화된 artifact, OBLITERATED 파생 모델의 결과를 같은 품질 수치로 읽지 않는다.

## 3분 이해 (ELI5)

이 장의 숫자는 같은 자동차의 기록이 아니라, 서로 다른 엔진과 도로에서 나온 주행 기록이다.

```text
모델·revision → engine·kernel → prompt·동시성 → 결과
```

원본 모델과 파생 artifact를 먼저 구분해야 속도와 품질을 올바르게 비교할 수 있다.

## 16.1 한눈에 보기

| 사람들이 만든 것 | 해결하려는 문제 | 대표 사례 |
|---|---|---|
| 단일 Spark serving | 27B dense 모델을 128GB 안에서 빠르게 서비스 | SGLang NVFP4, FP8, 4-bit |
| speculative decoding | 메모리 대역폭 때문에 낮은 decode 속도 개선 | MTP, DSpark, DFlash2 |
| OpenAI·Anthropic endpoint | 코딩 에이전트와 모델을 분리 | Claude Code, Qwen Code, OpenCode |
| 장문 context 서버 | 262K 또는 1M 입력 처리 | SGLang YaRN, FP8 KV |
| 다중 Spark pool | TP=2 또는 worker와 supervisor 분리 | 듀얼 Spark, 2×2 구성 |
| 장시간 코딩 작업 | 실제 저장소와 시뮬레이션을 에이전트로 구현 | FloodLayer, Prime-Agent |
| Mac과의 역할 분리 | MLX와 CUDA를 각각의 endpoint로 활용 | mlx-dspark, Spark worker |

## 16.2 먼저 모델 이름과 내부 architecture를 맞춰 읽기

[Qwen 공식 저장소](https://github.com/QwenLM/Qwen3.8)는 Qwen3.8을 Qwen3.5 아키텍처 기반의 모델로 설명한다. 그래서 로그에 다음과 같이 표시돼도 잘못된 모델을 불러온 것이 아니다.

```text
model name: Qwen3.8-27B
resolved architecture: Qwen3_5ForConditionalGeneration
```

실제로 확인할 항목은 model id, `config.json`, weight shard, chat template, runtime이다. architecture 문자열만 보고 Qwen3.5 모델이 실행됐다고 판단하면 안 된다.

또 하나 구분할 대상이 있다. 우리가 직접 테스트한 모델은 원본 `Qwen/Qwen3.8-27B`가 아니라 `OBLITERATUS/Qwen3.8-27B-OBLITERATED`다. 이 파생 모델은 refusal과 safety 동작을 수정했으므로 원본 모델 카드의 품질 수치를 그대로 적용할 수 없다.

## 16.3 단일 Spark에서 사람들이 만든 serving

### MiaAI-Lab SGLang recipe

[MiaAI-Lab 레포](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)는 NVFP4 W4A4를 기본으로 하고 BF16과 FP8 경로도 제공한다. EAGLE/MTP, DSpark, DFlash2를 바꿔 실행할 수 있고, native 262K context와 YaRN 기반 1M profile을 별도로 다룬다.

이 recipe에는 다음 최적화가 들어 있다.

- FP8 KV cache
- 동시 요청 수에 맞춘 GDN state pool
- GB10의 빠른 Cortex-X5 코어 pinning
- `qwen3` reasoning parser
- `qwen3_coder` tool parser
- chunked prefill과 decode graph 설정

README에는 CPU pinning만으로도 환경에 따라 약 2~7% decode가 좋아졌다는 측정이 있다. 이 사례가 보여주는 것은 양자화 하나보다 model, runtime, context, KV, CPU affinity를 함께 맞추는 일이 중요하다는 점이다.

### hasso5703 one-command 서비스

[dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)는 설치와 systemd 자동 시작까지 묶은 서비스형 recipe다. reference box의 README 수치는 다음과 같다.

| 조건 | 보고값 |
|---|---:|
| greedy 단일 스트림 중앙값 | 약 50 tok/s |
| code workload | 41~47 tok/s |
| reasoning workload | 52~57 tok/s |
| 자유 문장 | 17~23 tok/s |
| 8개 동시 aggregate | 135~148 tok/s |
| 32개 동시 aggregate | 약 258 tok/s |

이 서버는 OpenAI와 Anthropic 호환 API를 30000번 포트에서 제공하고 Claude Code 연결을 지원한다고 설명한다. 첫 부팅은 CUDA graph capture와 kernel compilation 때문에 약 7~9분이 걸리고, 설치에 약 85GB의 디스크 여유가 필요하다.

이 숫자는 reference box의 측정값이다. 다른 Spark에서 같은 숫자가 나오는 것은 아니다. boot-to-boot variance, driver, image, kernel cache, power state를 함께 고정해야 한다.

### 0xBakeer의 FP8 recipe

[FP8 단일 Spark 레포](https://github.com/0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark)는 weight를 바꾸지 않고 decode 전략을 비교했다.

| 구성 | 단일 fresh generation | edit-heavy | c8 aggregate |
|---|---:|---:|---:|
| stock | 7.88 | 7.88 | 기록 없음 |
| MTP k=3 | 17.70 | 21.3 | 기록 없음 |
| DSpark k=7 | 20.05 | 46.8 | 208.7 |
| DSpark k=14 | 18.77 | 58.5 | 119.7 |

같은 레포의 DFlash2 표본은 fresh generation 31.72, edit-heavy 49.20 tok/s다. 레포는 품질 평가는 수행하지 않았다고 명시한다. `k=14`가 단일 요청에는 유리해도 `c8` aggregate에는 `k=7`이 유리할 수 있다. 한 가지 draft depth를 모든 작업의 정답으로 볼 수 없다.

### 0xBakeer의 4-bit recipe

[4-bit 단일 Spark 레포](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark)는 단일 스트림 약 75 tok/s와 8-way aggregate 약 246 tok/s를 보고한다. 전자는 DSpark `k=14`, 후자는 `k=7` 조건이다.

이 레포도 속도만 측정했다. 품질 평가는 없다. vLLM에서는 4-bit checkpoint의 LM head 조건 때문에 DFlash2를 사용할 수 없었고, llama.cpp에서는 Q4_K_M과 DFlash2를 조합해 generative 37.58, edit-heavy 60.89 tok/s를 측정했다.

### Reddit FP8과 DFlash2 사용기

[Reddit 사용기](https://www.reddit.com/r/LocalLLM/comments/1vtbwtb/dgx_spark_qwen_38_27b_fp8_at_32toks_generation/)는 warm-up된 coding workload에서 speculation 없음 약 14 tok/s, DFlash2 사용 후 안정적으로 약 32 tok/s를 기록했다. 최고 속도는 약 40 tok/s였다고 수정했다.

글에 적힌 조건은 `max-model-len 240000`, `gpu-memory-utilization 0.88`, DFlash2 draft 7 tokens, `qwen3_coder` parser, `qwen3` reasoning parser다. 단일 workload의 초기 사용기이므로 일반적인 Qwen3.8 성능표로 사용하지 않는다.

## 16.4 dense 모델의 기본 속도가 낮은 이유

[67AI Lab 측정](https://67ailab.com/posts/qwen38-27b-dgx-spark-mtp-speedup/)은 Q4_K_M GGUF 기준 기본 약 10.9 tok/s, MTP 사용 후 약 28.9 tok/s를 기록했다. 같은 장치의 Qwen3.5-35B-A3B는 약 65.1 tok/s였다.

이 차이는 Qwen3.8의 품질이 낮아서가 아니다. Qwen3.8은 dense 27B라서 매 토큰에서 많은 weight를 읽는다. Qwen3.5-35B-A3B는 MoE라서 active parameter가 작다. DGX Spark의 unified memory bandwidth에서는 parameter 총량보다 토큰당 실제로 읽는 byte 수가 속도를 크게 좌우한다.

그러므로 Qwen3.8에서는 다음 조합을 함께 기록해야 한다.

```text
weight quantization + KV dtype + speculative method + draft depth
context length + concurrency + prompt class + output budget
```

## 16.5 듀얼 Spark와 여러 노드

[NVIDIA Developer Forum 듀얼 Spark 글](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350)과 [SGLang DFlash2 비교 글](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732)은 Qwen3.8-27B를 두 대에 분산한 사례다.

우리 리서치 표에는 다음 커뮤니티 수치가 기록돼 있다.

| workload | 1대 | 2대 |
|---|---:|---:|
| code | 52~61 tok/s | 약 87 tok/s |
| prose | 약 26 tok/s | 약 41 tok/s |

이 값은 동일한 공식 benchmark에서 나온 하나의 표가 아니다. 포럼 게시글의 runtime, prompt, context, draft 설정이 같은지 다시 확인해야 한다. TP=2에서는 all-reduce와 activation 통신이 추가되므로 모델이 두 배 빨라진다고 예상하면 안 된다.

장비 수에 따른 기본 선택은 다음과 같다.

| Spark 수 | Qwen3.8 역할 | 구성 |
|---:|---|---|
| 1대 | coding worker | NVFP4 또는 FP8 SGLang recipe |
| 2대 | 큰 context 또는 한 endpoint | TP=2 direct QSFP/RoCE |
| 2대 | agent 여러 개 | DP=2 또는 endpoint 두 개 |
| 3대 | supervisor와 worker | Qwen TP=2 + worker 1대 |
| 4대 | DeepSeek와 Qwen 역할 분리 | DS4 TP=2 + Qwen TP=2 |
| 8대 | 여러 서비스 pool | switch fabric과 pool 분리 |

Qwen과 DeepSeek를 각각 TP=2로 운영하는 2×2 구성은 네 대가 필요하다. 두 대에 두 모델을 동시에 올리는 구성은 unified memory와 KV headroom을 다시 측정해야 하므로 아직 공식 배치로 쓰지 않는다.

## 16.6 사람들이 Qwen3.8로 만든 것

### Qwen Code와 Claude Code

[Qwen 공식 저장소](https://github.com/QwenLM/Qwen3.8)는 Qwen Code를 Qwen 모델에 맞춘 오픈소스 터미널 에이전트로 소개한다. Qoder도 Qwen3.8을 직접 지원하는 agentic coding platform으로 안내된다.

커뮤니티 recipe는 OpenAI-compatible와 Anthropic-compatible API를 제공해 Claude Code, OpenCode, Pi, Hermes, Aider 같은 클라이언트를 연결한다. 이때 모델 이름보다 tool schema와 parser가 더 중요할 수 있다. endpoint가 열렸다는 것과 파일 수정, 테스트 실행, 오류 복구가 장시간 안정적으로 된다는 것은 별개의 결과다.

### FloodLayer 3D sandbox

[RTX 5090 사용 사례](https://www.reddit.com/r/Qwen_AI/comments/1vsrq6v/qwen_38_27b_built_this_locally_on_my_rtx_5090/)에서 사용자는 GPT-5.6 Sol로 FloodLayer 계획 문서를 만든 뒤 Qwen3.8-27B와 DeepSeek Harness로 3D AEC sandbox를 구현했다고 설명한다. 물이 바닥 경사를 따라 흐르고 배수구와 문턱을 고려하는 작은 시뮬레이션이다.

해당 설정은 131K context, Q8 KV, full GPU offload, MTP draft max 2였다. MTP 없이 약 54 tok/s, context와 draft acceptance에 따라 MTP 사용 시 약 70~100 tok/s를 보고했다. 댓글에는 Prime-Agent에서 약 48시간 실행해 문제를 해결했다는 사용담도 있다.

이 사례는 공식 평가가 아니다. 그러나 Qwen3.8을 짧은 채팅이 아니라 파일과 테스트를 다루는 장기 코딩 worker로 사용한 사례라는 점에서 의미가 있다.

### 이미지와 화면

공식 생태계는 Qwen3.5 계열의 text와 vision 경로를 llama.cpp와 MLX에서 지원한다고 안내한다. Qwen3.8-27B 모델 카드도 image-text-to-text 경로를 제공한다.

우리 OBLITERATED BF16 파생 모델은 JPEG 한 장을 넣는 smoke test를 통과했다. 하지만 native vision 품질, OCR, 작은 글씨, 화면 좌표 grounding, 영상 입력은 아직 측정하지 않았다. 텍스트 서버가 올라왔다는 사실만으로 이미지 처리가 준비됐다고 쓰지 않는다.

### 문서 요약

[대만 PTT 사용기](https://www.ptt.cc/bbs/AI_Art/M.1787288665.A.216.html)는 같은 긴 문서를 llama.cpp와 SGLang DFlash2로 요약한 뒤, SGLang 쪽 결과가 약 33% 짧았다고 적었다. 이 수치가 DFlash2의 일반 품질을 증명하지는 않는다. 다만 속도가 빨라졌을 때 항목 누락과 사실 보존을 따로 확인해야 한다는 실전 경고다.

### Apple Silicon

[mlx-dspark 사례](https://www.reddit.com/r/LocalLLaMA/comments/1vokrcy/qwen3827b_is_now_up_to_3_faster_on_apple_silicon/)는 M4 Pro에서 8-bit target을 약 8.3에서 20.3 tok/s로, 4-bit target을 약 25.3 tok/s로 측정했다. MLX endpoint는 OpenAI-compatible와 Anthropic Messages API를 제공해 Claude Code 연결도 가능하다고 설명한다.

이것은 Spark와 Mac의 USB 메모리 풀링 결과가 아니다. Mac은 MLX worker, Spark는 CUDA worker로 두고 router가 작업을 나누는 참고 사례다.

## 16.7 우리가 직접 확인한 결과

직접 테스트한 모델은 `OBLITERATUS/Qwen3.8-27B-OBLITERATED` BF16이다.

| 테스트 | 결과 |
|---|---|
| health와 `/v1/models` | 통과 |
| 한국어 산수, Python, JSON | 통과 |
| 멀티턴 marker 회수 | 통과 |
| thinking off와 on | 통과 |
| JPEG 입력 | 통과 |
| 10K와 32K prompt needle | 통과 |
| 4개 동시 요청 | 통과 |
| function call | 통과 |

parser benchmark는 vLLM 0.26.0, BF16, 32K context, `qwen3_xml`에서 수행했다.

| 측정 | 결과 |
|---|---:|
| 명시적 tool call | 1/1 통과 |
| c1 TTFT p50 | 463.736ms |
| c1 end-to-end completion | 4.567 tok/s |
| c4 aggregate end-to-end | 17.697 tok/s |

이 값은 짧은 출력의 end-to-end 결과라서 순수 decode tok/s가 아니다. speculation도 끄고 측정했다. 따라서 원본 Qwen3.8 NVFP4 SGLang recipe의 30~50 tok/s와 직접 비교하면 안 된다.

공식 원본 recipe는 보통 다음 parser 조합을 사용한다.

```text
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--reasoning-parser qwen3
```

우리는 파생 모델의 XML chat template에 맞춰 `qwen3_xml`을 사용했다. parser 이름은 모델 능력의 장식이 아니라 runtime 조건이다. model card, chat template, parser를 함께 고정해야 한다.

## 16.8 Qwen3.8과 DeepSeek 역할 분리

두 모델의 커뮤니티 속도 숫자만 비교하면 안 된다. 현재 문서에서 권장하는 역할은 다음과 같다.

```text
DeepSeek V4 Flash 0731: 긴 context, 계획, tool 복구, supervisor
Qwen3.8-27B: 코드 수정, UI/CSS, 반복 구현, 짧은 tool 작업
별도 VLM: 이미지와 화면
```

이 구성은 Qwen3.8이 DeepSeek보다 모든 면에서 낫다는 뜻이 아니다. Qwen3.8을 가벼운 coding worker로 분리하면 DeepSeek supervisor의 context와 tool history를 불필요하게 늘리지 않을 수 있다는 운영 설계다. 두 endpoint가 각각 TP=2라면 Spark 네 대가 필요하다.

## 16.9 아직 하지 않은 테스트

- 원본 `Qwen/Qwen3.8-27B`와 OBLITERATED 파생 모델의 동일 조건 품질 비교
- BF16, FP8, NVFP4의 같은 prompt set 비교
- 32K, 131K, 262K, 1M retrieval과 coding quality
- MTP, DSpark, DFlash2의 draft acceptance 곡선
- c1, c4, c8, c16 per-request와 aggregate
- Qwen Code, Claude Code, OpenCode, Pi, Hermes의 반복 tool loop
- 듀얼 Spark TP=2의 direct QSFP와 switch 비교
- DeepSeek supervisor와 Qwen worker의 실제 router 성공률
- native vision, OCR, 화면 좌표, 비디오 입력 품질

공개 recipe의 숫자를 우리 장비의 최종 성능으로 승격하려면 위 조건을 같은 prompt와 output budget으로 반복해야 한다.

## 16.10 다음 재현 순서

1. 원본 FP8과 NVFP4를 scratch port에서 각각 load한다.
2. health, 일반 응답, JSON, tool call을 순서대로 확인한다.
3. speculation 없이 c1 baseline을 측정한다.
4. MTP, DSpark, DFlash2를 draft depth별로 비교한다.
5. 8K, 32K, 131K, 262K에서 TTFT, decode, acceptance, retrieval을 기록한다.
6. c1, c4, c8, c16에서 per-request와 aggregate를 따로 저장한다.
7. sandbox 안에서 Qwen Code 또는 Claude Code tool loop를 반복한다.
8. 원본, 최적화 recipe, OBLITERATED 파생 모델을 별도 표로 유지한다.
9. 그 뒤 DeepSeek supervisor와 Qwen worker를 2×2 구성으로 연결한다.

## 참고 문서

- [Qwen3.8 커뮤니티 제작물·활용 사례 원문 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/qwen38-community-builds-2026-08.md)
- [Qwen3.8-27B-OBLITERATED 직접 테스트](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/model-research-qwen38-obliterated.md)
- DeepSeek V4 Flash 0731로 사람들이 만든 것
- 로컬 에이전트 운영
- 두 대 연결하기
- 부록: 모델·레시피·명령어 색인
