# Qwen3.8-27B 커뮤니티 제작물과 활용 사례 리서치

조사일: **2026-08-22**

이 문서는 Qwen3.8-27B를 사람들이 DGX Spark, 다른 GPU, Apple Silicon에서 어떻게 실행하고 실제 작업에 연결했는지 정리한다. 모델 카드가 설명하는 기능과, 커뮤니티가 만든 serving recipe의 주장, 우리 저장소에서 직접 확인한 결과를 서로 다른 근거로 기록한다.

## 먼저 결론

Qwen3.8-27B는 DGX Spark에서 “그냥 올리면 빠른 모델”이 아니다. dense 27B 모델이라서 양자화와 speculative decoding을 쓰지 않은 기본 경로에서는 메모리 대역폭의 영향을 크게 받는다. 반면 MTP, DSpark, DFlash2, SGLang의 GDN 최적화와 CPU pinning을 조합하면 단일 스트림 속도와 동시 처리량이 크게 달라진다는 사례가 빠르게 쌓이고 있다.

커뮤니티가 만든 것은 크게 다음과 같다.

1. 단일 Spark용 NVFP4, FP8, 4-bit serving recipe
2. MTP, DSpark, DFlash2를 이용한 speculative decoding 경로
3. OpenAI-compatible 또는 Anthropic-compatible endpoint와 Claude Code, Qwen Code, OpenCode, Pi, Hermes 연결
4. 2대 Spark의 TP=2, 장문 컨텍스트, 동시 요청 serving
5. 코딩 에이전트로 3D 시뮬레이션과 저장소 작업을 수행한 장시간 사용 사례
6. Apple Silicon의 MLX와 DGX Spark의 CUDA를 각각 다른 역할로 사용하는 구성

이 자료에서 가장 조심해야 할 부분은 속도 숫자다. 같은 Qwen3.8-27B라도 FP8, NVFP4, Q4 GGUF, MTP, DSpark, DFlash2, 출력 유형, draft depth, context length에 따라 결과가 달라진다. `75 tok/s` 단일 스트림과 `246 tok/s` 8-way aggregate는 서로 대체할 수 있는 숫자가 아니다.

## 1. 모델 정체를 먼저 고정한다

### 1.1 원본 Qwen3.8-27B

[Qwen 공식 저장소](https://github.com/QwenLM/Qwen3.8)는 Qwen3.8을 Qwen3.5 아키텍처 기반의 최신 공개 모델로 설명한다. 코딩, 전문 업무, 연구, 장기 에이전트 작업, 환경 피드백을 반영하는 계획 수립을 주요 개선 방향으로 제시한다. `reasoning_effort`로 추론 깊이를 조절하고 `preserve_thinking`으로 대화 이력의 reasoning context를 보존하는 기능도 공식 설명에 포함되어 있다.

Qwen3.8-27B는 dense 모델이다. 커뮤니티의 하드웨어 분석에서는 48개의 Gated DeltaNet 계층과 16개의 full-attention 계층으로 이루어진 hybrid attention 구조가 설명된다. 이 구조는 모든 계층이 같은 방식으로 KV cache를 사용하지 않는다는 뜻이므로, 전통적인 dense Transformer와 메모리 계산을 그대로 비교하면 안 된다.

공식 생태계에는 다음 경로가 포함된다.

- Qwen Studio, Qoder, QwenWork, Qwen Cloud
- 터미널 에이전트인 Qwen Code
- Transformers, llama.cpp, MLX, Unsloth
- SGLang, vLLM, TokenSpeed
- OpenAI-compatible serving과 `qwen3_coder` tool parser

공식 SGLang 예시는 `--context-length 262144`, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`를 사용한다. 공식 vLLM 예시도 같은 262K context와 `qwen3_coder` parser를 사용한다. 이 명령은 원본 모델을 위한 기준점이며, 파생 checkpoint나 다른 chat template에 그대로 적용된다고 보장하지 않는다.

### 1.2 왜 로그에 Qwen3.5 아키텍처가 표시되는가

Qwen3.8이라는 모델 이름과 내부 architecture 이름은 다를 수 있다. 공식 계보가 Qwen3.5 기반이므로 `Qwen3_5ForConditionalGeneration`, `qwen3_5_text`, `qwen35` 같은 값이 config나 runtime 로그에 나타날 수 있다.

따라서 다음 로그만으로 잘못된 모델이 로드됐다고 판단하지 않는다.

```text
model name: Qwen3.8-27B
resolved architecture: Qwen3_5ForConditionalGeneration
```

확인해야 할 것은 `config.json`, model id, weight shard, chat template, runtime이 서로 맞는지다. architecture 이름만 보고 Qwen3.5 모델을 잘못 띄웠다고 결론 내리면 안 된다.

### 1.3 우리 저장소에서 직접 돌린 모델

우리 테스트 대상은 원본 `Qwen/Qwen3.8-27B`가 아니라 다음 파생 모델이다.

```text
OBLITERATUS/Qwen3.8-27B-OBLITERATED
```

이 checkpoint는 refusal과 safety 동작을 줄이도록 원본 가중치를 수정한 파생 모델이다. 모델 카드가 보고한 원본 MMLU 87.4%와 파생 모델 81.4%는 원본과 파생 모델의 차이를 보여주는 제작자 측 수치다. 이 결과를 일반 Qwen3.8-27B의 성능으로 쓰지 않는다.

우리 로컬 결과는 `vLLM 0.26.0`, BF16, `max_model_len=32768`, `gpu_memory_utilization=0.50`, 단일 Spark에서 얻었다. 이 환경은 SGLang NVFP4 recipe도 아니고 DFlash2 또는 DSpark를 켠 환경도 아니다.

## 2. 사람들이 만든 serving recipe

### 2.1 MiaAI-Lab의 SGLang recipe

[MiaAI-Lab의 Qwen3.8-27B SGLang 레포](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)는 DGX Spark에서 세 가지 speculative 경로를 바꿔 사용할 수 있도록 만든 실행 레시피다.

- 기본 checkpoint는 NVFP4 W4A4이고 BF16과 FP8 경로도 제공한다.
- native context는 262K이며, YaRN으로 검증한 1M profile을 별도로 제공한다.
- FP8 KV cache로 KV 메모리를 줄인다.
- GDN state pool을 동시 요청 수에 맞춰 계산한다.
- GB10의 빠른 Cortex-X5 코어에 tokenizer와 scheduler를 pinning해 약 2~7%의 decode 개선을 측정했다.
- `qwen3_coder` tool parser와 `qwen3` reasoning parser를 사용한다.
- EAGLE/MTP, DSpark, DFlash2를 같은 계열의 서버에서 비교한다.

이 레포의 중요한 점은 단순히 “NVFP4가 빠르다”고 주장하는 데 있지 않다. GDN state pool, CPU core 선택, speculative draft depth, context profile을 함께 고정하고 on-device 측정 결과를 남긴다는 데 있다. 같은 모델이라도 이런 항목을 바꾸면 결과가 달라진다.

### 2.2 hasso5703의 one-command 서비스

[dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)는 SGLang, NVFP4, DFlash2, deterministic kernel 설정을 하나의 설치·부팅 서비스로 묶었다. README에 기록된 reference box 결과는 다음과 같다.

| 측정 조건 | 보고값 | 읽는 방법 |
|---|---:|---|
| greedy 단일 스트림 중앙값 | 약 50 tok/s | code 41~47, reasoning 52~57, math 최고 60 |
| 자유 문장 단일 스트림 | 17~23 tok/s | 출력 유형에 따라 크게 낮아짐 |
| 8개 동시 스트림 aggregate | 135~148 tok/s | 한 요청이 148 tok/s를 받는다는 뜻이 아님 |
| 32개 동시 스트림 aggregate | 약 258 tok/s | batch와 메모리 사용 조건이 다름 |

이 recipe는 OpenAI와 Anthropic 호환 API를 30000번 포트에서 제공하고, Claude Code를 바로 연결할 수 있다고 설명한다. systemd service로 부팅 때 자동 실행하고, 설치에 약 85GB의 디스크 여유가 필요하며, 첫 부팅에는 CUDA graph capture와 kernel compilation 때문에 7~9분이 걸린다고 기록한다.

이 수치는 레포 제작자가 reference box에서 측정한 값이다. 다른 Spark에서 같은 값이 보장되는 것은 아니다. 특히 boot-to-boot variance, driver, image, kernel cache, power state를 함께 기록해야 한다.

### 2.3 0xBakeer의 FP8 단일 Spark 측정

[Qwen3.8-27B FP8 단일 Spark 레포](https://github.com/0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark)는 weight를 바꾸지 않고 decode 전략을 바꿔 속도를 측정했다. 레포가 제시한 표본은 다음과 같다.

| 구성 | fresh generation | edit-heavy | c8 aggregate |
|---|---:|---:|---:|
| stock, speculation 없음 | 7.88 | 7.88 | 기록 없음 |
| MTP k=3 | 17.70 | 21.3 | 기록 없음 |
| MTP k=8 | 18.64 | 32.2 | 기록 없음 |
| DSpark k=7 | 20.05 | 46.8 | 208.7 |
| DSpark k=14 | 18.77 | 58.5 | 119.7 |

같은 레포의 DFlash2 측정은 fresh generation 31.72, edit-heavy 49.20 tok/s를 기록한다. 다만 레포는 품질 평가를 수행하지 않았고, 일부 행은 서로 다른 장치 설정에서 측정했으며, edit-heavy workload가 낙관적일 수 있다고 명시한다.

이 결과에서 실무적으로 중요한 부분은 draft depth가 하나의 최적값으로 수렴하지 않는다는 점이다. `k=14`가 단일 요청에서 유리해도 `c8` aggregate에서는 `k=7`이 더 나을 수 있다. latency 최적화와 여러 사용자 처리량 최적화는 서로 다른 설정을 요구한다.

### 2.4 0xBakeer의 4-bit 단일 Spark 측정

[4-bit 단일 Spark 레포](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark)는 FP8과 두 종류의 4-bit checkpoint를 같은 harness에서 비교했다.

- 단일 스트림은 DSpark `k=14`에서 약 75 tok/s를 보고한다.
- 8-way aggregate는 DSpark `k=7`에서 약 246 tok/s를 보고한다.
- 이 두 값은 서로 다른 draft depth와 workload다.
- 레포 자체의 품질 평가는 없으며 속도만 측정했다.
- vLLM에서는 4-bit checkpoint의 unquantized LM head 조건 때문에 DFlash2를 사용할 수 없었다.
- llama.cpp에서는 Q4_K_M과 DFlash2 조합으로 generative 37.58, edit-heavy 60.89 tok/s를 기록했다.

4-bit가 언제나 FP8보다 낫다는 결론은 이 표에서 나오지 않는다. 레포의 c1에서는 NVFP4가 FP8보다 약 27% 빠르지만, c16에서는 차이가 약 0.2%로 줄었다. 낮은 동시성의 interactive latency와 높은 동시성의 fleet throughput은 quantization 선택 기준이 다르다.

### 2.5 Reddit의 FP8과 DFlash2 사용기

[DGX Spark에서 FP8 Qwen3.8-27B를 사용한 Reddit 글](https://www.reddit.com/r/LocalLLM/comments/1vtbwtb/dgx_spark_qwen_38_27b_fp8_at_32toks_generation/)은 warm-up된 coding benchmark에서 speculation 없음 약 14 tok/s, DFlash2 사용 후 안정적으로 약 32 tok/s를 기록했다고 설명한다. 최고 속도는 약 40 tok/s였지만 안정적인 값은 32 tok/s라고 글쓴이가 수정했다.

글에 적힌 serving 조건은 `max-model-len 240000`, `gpu-memory-utilization 0.88`, DFlash2 draft 7 tokens, `qwen3_coder` tool parser, `qwen3` reasoning parser다. 이것은 단일 workload의 초기 실험이며, 모든 코딩 요청의 속도나 품질을 보장하는 benchmark가 아니다.

이 사례는 vLLM을 직접 빌드하거나 GB10용 Docker image를 사용하는 경로도 보여준다. Qwen3.8의 tool calling을 serving 레벨에서 쓰려면 모델 이름보다 parser와 runtime fork가 더 중요할 수 있다.

### 2.6 llama.cpp와 MTP

[67AI Lab의 DGX Spark 측정](https://67ailab.com/posts/qwen38-27b-dgx-spark-mtp-speedup/)은 Q4_K_M GGUF에서 기본 generation 약 10.9 tok/s, MTP를 켠 뒤 약 28.9 tok/s를 보고한다. 같은 장치에서 Qwen3.5-35B-A3B는 약 65.1 tok/s였고, 글은 dense 모델과 MoE 모델이 매 토큰마다 읽는 parameter 수가 다르기 때문이라고 분석한다.

이 글은 오래된 llama.cpp build가 Qwen3.8의 MTP block을 잘못 해석해 로드에 실패할 수 있다는 문제도 기록한다. `qwen35` architecture를 지원하는 것과 Qwen3.8의 실제 layer layout과 MTP tensor를 지원하는 것은 같은 일이 아니다. 새 GGUF는 기존 서버를 건드리기 전에 별도 port에서 load smoke test를 해야 한다.

## 3. 여러 Spark에서 만든 구성

### 3.1 듀얼 Spark TP=2

[NVIDIA Developer Forum의 듀얼 Spark 글](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350)과 [SGLang, DFlash2 비교 글](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732)은 Qwen3.8-27B를 두 대의 Spark에 분산한 사례를 모은다.

우리 리서치 표에는 커뮤니티 측정으로 다음 변화가 기록돼 있다.

| workload | 1대 | 2대 | 해석 |
|---|---:|---:|---|
| code | 52~61 tok/s | 약 87 tok/s | 단일 스트림 측정으로 기록된 커뮤니티 값 |
| prose | 약 26 tok/s | 약 41 tok/s | 출력 유형에 따라 scaling 폭이 다름 |

이 값은 하나의 통합된 공식 benchmark가 아니다. 게시글, runtime, prompt와 context가 모두 같은지 확인한 뒤 재현해야 한다. 두 대를 연결하면 모델이 커지는 것과 통신 비용이 줄어드는 일이 동시에 일어나지 않는다. TP=2에서는 all-reduce와 activation 통신이 추가되고, switch나 direct QSFP 설정에 따라 결과가 달라진다.

### 3.2 두 대를 살 때 선택할 것

Qwen3.8을 두 대에서 운영할 때는 다음 중 하나를 먼저 정해야 한다.

| 목적 | 배치 | 장점 | 주의점 |
|---|---|---|---|
| 한 요청의 context와 모델 headroom | TP=2 | 한 endpoint에서 큰 context 사용 | 통신과 장애 범위가 커짐 |
| 여러 agent의 독립 작업 | DP=2 또는 endpoint 2개 | 요청 격리와 장애 격리가 쉬움 | 단일 요청 속도는 합산되지 않음 |
| DeepSeek supervisor와 Qwen worker | Spark별 역할 분리 | 역할에 맞는 모델을 선택 가능 | 2대만으로 두 모델을 모두 TP=2로 운영할 수 없음 |

두 Spark를 직접 QSFP로 연결하는 경우에는 두 노드 TP를 먼저 검증할 수 있다. 세 개 이상의 노드와 여러 model pool을 묶으면 공통 switch와 포트 설계가 필요하다. Qwen3.8의 듀얼 숫자를 DeepSeek의 듀얼 숫자와 섞어 쓰지 않는다.

### 3.3 네 대와 그 이상

우리 구성에서 가장 이해하기 쉬운 4-Spark 예시는 다음과 같다.

```text
Spark 1 + Spark 2: DeepSeek V4 Flash 0731, TP=2, supervisor
Spark 3 + Spark 4: Qwen3.8-27B, TP=2, coding/UI worker
```

이 구성은 두 개의 독립 서비스다. 하나의 4대 Qwen3.8 benchmark가 아니며, 두 모델을 하나의 메모리 풀로 합치는 구성도 아니다. 4대 이상을 switch에 연결하는 경우에는 switch의 port 수와 fabric topology, NCCL transport, MTU, RoCE 설정을 별도로 확인해야 한다.

3대에서는 Qwen TP=2와 worker 한 대를 나누는 2+1 구성이 현실적이다. 8대에서는 하나의 큰 TP보다 supervisor, worker, vision, benchmark를 여러 pool로 분리하는 편이 운영 측면에서 더 검증하기 쉽다.

## 4. 실제 활용 사례

### 4.1 Qwen Code와 개발 도구

[Qwen 공식 문서](https://github.com/QwenLM/Qwen3.8)는 Qwen Code를 Qwen 모델에 맞춘 오픈소스 터미널 에이전트로 소개한다. 대규모 저장소 이해, 반복 작업 자동화, 코드 변경을 주요 사용 사례로 제시한다. Qoder도 Qwen3.8을 직접 지원하는 agentic coding platform으로 안내된다.

로컬 endpoint에서는 Qwen Code만 가능한 것이 아니다. OpenAI-compatible API를 이해하는 Claude Code, OpenCode, Pi, Hermes, Aider, Kilo Code 계열을 연결할 수 있다. 다만 endpoint가 연결된 것과 장기 코딩 작업이 안정적으로 끝나는 것은 다르다. tool schema, reasoning field, assistant/tool turn, auto-compaction, 파일 권한을 함께 확인해야 한다.

### 4.2 Claude Code를 Qwen worker로 사용

[hasso5703 recipe](https://github.com/hasso5703/dgx-spark-qwen38)는 OpenAI와 Anthropic API 양쪽을 제공하고 Claude Code를 바로 사용할 수 있다고 설명한다. 이 구성은 Qwen3.8을 브라우저 채팅 모델이 아니라 터미널에서 파일을 읽고, 수정하고, 테스트하는 worker로 쓰는 사례다.

우리의 권장 역할 분리는 다음과 같다.

```text
DeepSeek V4 Flash 0731: 계획 수립, 긴 문서, 복구, supervisor
Qwen3.8-27B: 코드 수정, UI/CSS, 반복 구현, 짧은 tool 작업
별도 VLM: 이미지와 화면 입력
```

이 역할 분리는 커뮤니티가 하나의 통합 benchmark로 증명한 사실이 아니라, 현재 리서치와 장비 구성을 바탕으로 한 운영 설계다. DeepSeek와 Qwen을 각각 TP=2로 실행하려면 Spark 네 대가 필요하다.

### 4.3 장시간 코딩 에이전트와 3D 작업

[Reddit의 RTX 5090 사례](https://www.reddit.com/r/Qwen_AI/comments/1vsrq6v/qwen_38_27b_built_this_locally_on_my_rtx_5090/)에서 사용자는 GPT-5.6 Sol로 FloodLayer의 계획 문서를 만든 뒤, Qwen3.8-27B와 DeepSeek Harness로 3D AEC sandbox를 구현했다고 설명한다. FloodLayer는 실제 바닥 경사를 따라 물이 흐르고 배수구와 문턱을 고려하는 작은 시뮬레이션이다.

해당 설정은 131K context, Q8 KV cache, full GPU offload, MTP draft max 2였다. MTP 없음에서는 약 54 tok/s, context와 acceptance에 따라 MTP 사용 시 약 70~100 tok/s를 보고한다. 글쓴이는 auto-compaction으로 장시간 작업을 이어 갈 수 있었다고 말한다.

같은 글의 댓글에는 Qwen3.8-27B를 Prime-Agent에서 약 48시간 실행했고 문제를 해결했다고 적은 사용담도 있다. 이것은 장기 안정성의 공식 검증이 아니라 개인 사용 경험이다. 그래도 단순한 한 번의 코드 생성이 아니라 실제 agent harness, 파일 작업, 긴 세션에 연결한 사례라는 점에서 기록할 가치가 있다.

### 4.4 화면과 이미지 이해

공식 Qwen 생태계는 Qwen3.5 계열의 text와 vision 경로를 llama.cpp와 MLX에서 지원한다고 안내한다. Qwen3.8-27B 모델 카드도 image-text-to-text 모델 경로를 제공한다. 그러나 runtime마다 multimodal projector, image processor, context 계산 방식이 다르므로 텍스트 서버가 올라왔다고 이미지 입력이 자동으로 되는 것은 아니다.

우리의 OBLITERATED BF16 파생 모델은 JPEG 한 장을 넣는 smoke test를 통과했다. 이 결과는 파생 모델에서 한 번의 이미지 요청이 처리됐다는 뜻이다. native vision 품질, OCR, 작은 객체 인식, 화면 좌표 grounding을 검증한 결과는 아니다.

### 4.5 문서 요약과 speculative decoding의 품질 trade-off

[대만 PTT의 Qwen3.8 사용기](https://www.ptt.cc/bbs/AI_Art/M.1787288665.A.216.html)는 같은 긴 문서를 llama.cpp와 SGLang DFlash2로 요약한 경험을 공유한다. 글쓴이는 llama.cpp 쪽이 내용을 더 완전하게 남겼고, SGLang DFlash2 쪽은 약 33% 짧은 요약을 만들었다고 적었다.

이것은 DFlash2가 항상 내용을 잃는다는 뜻이 아니다. prompt, sampling, draft acceptance, 종료 조건, 출력 예산이 다른 실험일 수 있다. 다만 속도가 4~5배 빨라졌다는 이유만으로 문서 추출 품질이 같다고 가정하면 안 된다는 좋은 경고다. 긴 문서 작업에서는 속도와 함께 항목 recall, 사실 보존율, 누락 수를 기록해야 한다.

### 4.6 Apple Silicon과 MLX

[mlx-dspark Qwen3.8 사례](https://www.reddit.com/r/LocalLLaMA/comments/1vokrcy/qwen3827b_is_now_up_to_3_faster_on_apple_silicon/)는 Apple Silicon에서 Qwen3.8-27B를 MLX와 speculative decoding으로 실행한 결과를 공유한다. M4 Pro에서 8-bit target은 평균 8.3에서 20.3 tok/s로, 4-bit target은 약 25.3 tok/s로 측정됐다. 해당 프로젝트는 OpenAI-compatible와 Anthropic Messages API를 제공해 Claude Code 연결도 가능하다고 설명한다.

이 사례는 DGX Spark와 Mac을 USB로 직접 묶었다는 뜻이 아니다. Apple Silicon 쪽은 MLX endpoint, Spark 쪽은 CUDA endpoint로 분리하고 router가 작업을 나누는 참고 사례로 보는 편이 정확하다.

## 5. 속도 숫자를 해석하는 방법

### 5.1 dense 27B라서 기본 속도가 낮을 수 있다

Qwen3.8-27B의 parameter 수만 보면 DGX Spark에서 아주 빠를 것처럼 느껴질 수 있다. 하지만 dense 모델은 매 토큰에서 전체 weight를 읽어야 한다. 67AI Lab의 Q4_K_M 측정은 기본 10.9 tok/s를 기록했고, MTP를 켠 뒤 28.9 tok/s가 됐다. 같은 장치의 Qwen3.5-35B-A3B는 active parameter가 약 3B인 MoE라서 65.1 tok/s를 기록했다.

이 비교는 Qwen3.8의 품질이 낮다는 뜻이 아니다. DGX Spark의 unified memory bandwidth가 dense decode에 미치는 영향을 설명한다. Qwen3.8에서는 양자화, KV dtype, MTP 또는 DFlash2 선택이 성능에 큰 영향을 줄 수 있다.

### 5.2 speculative decoding은 workload 의존적이다

draft가 잘 맞는 출력에서는 여러 토큰을 한 번에 검증해 속도가 빨라진다. 코드, JSON, 반복 형식은 acceptance가 높을 수 있고, 자연어와 긴 reasoning은 acceptance가 낮아질 수 있다. context가 길어질수록 draft 효과가 줄어드는 사례도 있다.

따라서 다음 숫자를 함께 남겨야 한다.

- 단일 요청 decode
- prefill과 TTFT
- c1, c4, c8 aggregate
- draft depth와 acceptance
- context length
- thinking 설정
- prompt class
- 출력 token 수와 종료 원인

`tok/s` 하나만 기록하면 Qwen3.8 recipe 간 비교가 거의 불가능하다.

### 5.3 Qwen3.8과 DeepSeek의 역할은 다르다

Qwen3.8-27B는 단일 Spark에서 돌릴 수 있는 dense coding worker로 보는 것이 자연스럽다. DeepSeek V4 Flash 0731은 별도 양자화와 2대 TP recipe를 사용해 긴 context와 supervisor 역할을 노리는 모델이다.

따라서 다음처럼 비교해야 한다.

| 비교 대상 | Qwen3.8-27B | DeepSeek V4 Flash 0731 |
|---|---|---|
| 한 대 기본 경로 | BF16은 가능하지만 메모리와 대역폭 부담이 큼 | EXL3/SparkInfer/DSpark 경로가 중심 |
| 최적화 방향 | NVFP4, FP8, MTP, DSpark, DFlash2 | EXL3, NVFP4 KV, DSpark, TP=2 |
| 적합한 역할 | coding worker, UI/CSS, 반복 tool | supervisor, 긴 context, 계획과 복구 |
| 직접 비교 상태 | 같은 prompt set의 정식 비교 없음 | 같은 prompt set의 정식 비교 없음 |

Qwen 커뮤니티의 50 tok/s와 DeepSeek 커뮤니티의 47 tok/s를 보고 두 모델의 품질이나 전체 agent 성능이 같다고 결론 내릴 수 없다. speed, quality, agent success, context stability를 따로 평가해야 한다.

## 6. 우리 저장소에서 직접 확인한 것

### 6.1 기능 smoke test

직접 테스트한 모델은 `OBLITERATUS/Qwen3.8-27B-OBLITERATED` BF16이다.

| 테스트 | 결과 |
|---|---|
| `/v1/models`와 health | 통과 |
| 한국어 산수 | 통과 |
| Python 코드 생성 | 통과 |
| JSON 제약 출력 | 통과 |
| 멀티턴 marker 회수 | 통과 |
| thinking off와 on | 통과 |
| JPEG 이미지 입력 | 통과 |
| 10K, 32K prompt needle | 통과 |
| 4개 동시 요청 | 통과 |
| function call | 통과 |

### 6.2 parser와 속도 측정

2026-08-21 parser benchmark는 vLLM 0.26.0, BF16, `max_model_len=32768`, `qwen3_xml` parser에서 수행했다.

| 측정 | 값 |
|---|---:|
| 명시적 tool call | 1/1 통과 |
| c1 TTFT p50 | 463.736ms |
| c1 TTFT p95 | 465.110ms |
| c1 end-to-end completion | 4.567 tok/s |
| c4 TTFT p50 | 508.904ms |
| c4 aggregate end-to-end completion | 17.697 tok/s |

이 결과는 짧은 출력의 HTTP end-to-end 측정이며 순수 decode benchmark가 아니다. speculative decoding도 켜지 않았다. 원본 Qwen3.8 NVFP4 SGLang recipe의 30~50 tok/s와 직접 비교할 수 없는 이유다.

### 6.3 parser 이름도 조건의 일부다

공식 원본 recipe는 보통 다음 조합을 사용한다.

```text
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--reasoning-parser qwen3
```

우리의 파생 모델 smoke test는 이 모델의 XML chat template에 맞춰 `qwen3_xml` parser를 사용했다. 따라서 `qwen3_coder`를 넣었다고 원본 recipe와 같은 tool call 조건이 되는 것도 아니고, `qwen3_xml`을 사용했다고 모든 Qwen3.8 checkpoint에 적용되는 것도 아니다. model card, chat template, parser 구현을 함께 확인해야 한다.

## 7. 장비 수별 추천 구성

| Spark 수 | Qwen3.8 구성 | 용도 | 상태 |
|---:|---|---|---|
| 1대 | NVFP4 또는 FP8 SGLang + MTP/DSpark/DFlash2 | coding worker, 짧은 tool, OpenAI endpoint | 커뮤니티 recipe 재현 대상 |
| 1대 | BF16 vLLM | 모델 기능과 parser smoke test | 우리 환경에서 직접 확인 |
| 2대 | TP=2, direct QSFP/RoCE | 더 큰 context와 한 endpoint | 포럼 사례, 독립 재현 필요 |
| 2대 | DP=2, 독립 endpoint | 여러 agent 요청 격리 | 운영 설계 후보 |
| 3대 | Qwen TP=2 + worker 1대 | supervisor와 worker 분리 | 통합 recipe 없음 |
| 4대 | DeepSeek TP=2 + Qwen TP=2 | DS4 supervisor와 Qwen worker | 2×2 운영 설계 |
| 8대 | 여러 Qwen, DeepSeek, vision pool | 서비스와 benchmark 분리 | switch와 fabric 검증 필요 |

한 대에서 DeepSeek deep-context profile과 Qwen3.8 BF16을 동시에 띄우는 구성은 현재 확인하지 않았다. 2×2 역할 분리는 네 대에서 각각 독립 TP pool을 운영하는 구성이며, 하나의 모델을 네 대에 합치는 Qwen benchmark가 아니다.

## 8. 아직 검증하지 않은 것

- 원본 `Qwen/Qwen3.8-27B`의 동일 조건 기능·품질 평가
- 원본 NVFP4, FP8, BF16을 같은 prompt set으로 비교
- 32K, 131K, 262K, 1M에서 retrieval과 coding quality 비교
- SGLang MTP, DSpark, DFlash2의 draft acceptance 곡선
- c1, c4, c8, c16에서 단일 요청과 aggregate 분리 측정
- Claude Code, Qwen Code, OpenCode, Pi, Hermes의 20회 이상 tool loop
- 2대 TP=2에서 direct QSFP와 switch fabric 차이
- Qwen3.8 worker와 DeepSeek supervisor의 실제 router 성공률
- native vision, OCR, 화면 좌표 grounding, 비디오 입력 품질

현재 공개된 속도 숫자를 우리 장비의 최종 성능으로 승격하지 않는 이유가 여기에 있다. 먼저 원본 모델과 recipe revision을 고정하고, 같은 prompt와 출력 예산으로 반복해야 한다.

## 9. 다음 재현 순서

1. 원본 Qwen3.8-27B FP8과 NVFP4를 scratch port에서 각각 load한다.
2. health, `/v1/models`, 일반 응답, JSON, tool call을 순서대로 확인한다.
3. `qwen3_coder`와 `qwen3` parser 조합을 공식 recipe와 동일하게 기록한다.
4. speculation 없이 c1 baseline을 측정한다.
5. MTP, DSpark, DFlash2를 draft depth별로 측정한다.
6. 8K, 32K, 131K, 262K context에서 TTFT, decode, acceptance, retrieval을 기록한다.
7. c1, c4, c8, c16에서 per-request와 aggregate를 각각 저장한다.
8. Qwen Code와 Claude Code 중 하나를 sandbox 안에서 연결해 tool loop를 반복한다.
9. 원본 Qwen3.8과 파생 OBLITERATED 모델의 결과를 별도 표로 유지한다.
10. 그 뒤에 DeepSeek supervisor와 Qwen worker를 2×2 구성으로 연결한다.

## 참고 링크

### 공식

- [Qwen3.8 공식 저장소](https://github.com/QwenLM/Qwen3.8)
- [Qwen3.8-27B 모델 카드](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8-27B-FP8 모델 카드](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [Qwen Code](https://github.com/QwenLM/qwen-code)

### DGX Spark 레시피

- [MiaAI-Lab Qwen3.8 SGLang DGX Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)
- [hasso5703 dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)
- [0xBakeer Qwen3.8-27B FP8 single Spark](https://github.com/0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark)
- [0xBakeer Qwen3.8-27B 4-bit single Spark](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark)
- [NVIDIA Forum single Spark SGLang recipe](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)
- [NVIDIA Forum dual Spark](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350)
- [NVIDIA Forum Qwen3.8 SGLang DFlash2](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732)
- [Reddit FP8와 DFlash2 DGX Spark 사용기](https://www.reddit.com/r/LocalLLM/comments/1vtbwtb/dgx_spark_qwen_38_27b_fp8_at_32toks_generation/)

### 실제 활용과 비교

- [Qwen3.8 5090 DeepSeek Harness FloodLayer 사례](https://www.reddit.com/r/Qwen_AI/comments/1vsrq6v/qwen_38_27b_built_this_locally_on_my_rtx_5090/)
- [Apple Silicon MLX-DSpark 사례](https://www.reddit.com/r/LocalLLaMA/comments/1vokrcy/qwen3827b_is_now_up_to_3_faster_on_apple_silicon/)
- [Qwen3.8과 MTP의 DGX Spark 측정](https://67ailab.com/posts/qwen38-27b-dgx-spark-mtp-speedup/)
- [Qwen3.8 DFlash2 문서 요약 사용기](https://www.ptt.cc/bbs/AI_Art/M.1787288665.A.216.html)
- [우리의 Qwen3.8-27B-OBLITERATED 직접 테스트](model-research-qwen38-obliterated.md)
- [DeepSeek V4 Flash 0731 커뮤니티 제작물](deepseek-v4-flash-0731-community-builds-2026-08.md)
