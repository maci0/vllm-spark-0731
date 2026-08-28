# DeepSeek V4 Flash 0731 성능 리서치

조사일: **2026-08-22**

> 이 문서는 DeepSeek V4 Flash 0731을 DGX Spark에서 실제로 쓸 수 있는지, 공개된 속도와 긴 문맥 주장이 무엇을 의미하는지, GPT-5.6-Sol과 어디까지 비교할 수 있는지를 분리해 정리한다.

사람들이 이 모델로 실제로 만든 에이전트 endpoint, vision shim, 듀얼 Spark 서비스, 영상 co-tenancy 결과는 [DeepSeek V4 Flash 0731 커뮤니티 제작물·응용 사례 리서치](deepseek-v4-flash-0731-community-builds-2026-08.md)에서 별도로 정리한다.

## 먼저 결론

**좋은 모델이고, DGX Spark에서 특히 값어치가 큰 모델인 것은 맞다. 다만 게시물의 숫자를 그대로 “항상 47 tok/s, 384K에서 안정적인 추론, GPT-5.6-Sol과 같은 성능”으로 읽으면 안 된다.**

판정은 다음과 같다.

- **단일 Spark 실행**: 가능하다. 다만 공식 FP8 원본을 그대로 올린 것이 아니라, 3.0 bpw EXL3와 REAP-K216 pruning, SparkInfer, DSpark, NVFP4 KV를 조합한 특수 경로다.
- **단일 Spark 속도**: 44~47 tok/s는 공개 recipe에서 실제 보고된 수치다. 그러나 구조화된 단일 스트림, 특정 context, 특정 warm-up과 speculative decoding 조건의 결과다. 다른 단일 Spark 재현에서는 대략 23~37 tok/s가 보고됐고, 같은 모델도 양자화와 엔진에 따라 17~35 tok/s까지 내려간다.
- **긴 문맥**: 384K는 launcher의 상한이고, 370,104토큰 입력에서 needle을 회수한 것은 사실이다. 하지만 needle 하나를 찾은 결과는 모델의 장문 추론·요약·코딩 품질을 증명하지 않는다.
- **1,024 tok/s prefill**: 요청 초반에 관찰된 속도다. 370K 전체 입력을 넣은 실효 prefill은 약 625 tok/s였고, 전체 요청은 약 10분 걸렸다.
- **저장소 C1 하니스**: semantic과 JSON schema, 5개 언어 code decode gate는 통과했다. cold prefill은 실제 251,968-token 요청에서 975.44 tok/s를 기록해 1,000 tok/s gate를 통과하지 못했다. 다만 하니스의 `/tokenize`와 실제 chat 요청 사이에 79-token 계산 차이도 함께 확인됐다.
- **모델 품질**: 공식 카드에서 Terminal-Bench 2.1 82.7을 기록해 강한 코드 에이전트임을 보여준다. 하지만 공개된 공식 표에는 GPT-5.6-Sol과의 직접 비교가 없다.
- **GPT-5.6-Sol과의 관계**: 특정 코딩 에이전트 벤치마크에서는 가까운 점수가 나올 수 있지만, 현재 근거로 동급이라고 쓰기는 어렵다. raw tok/s 기준으로는 GPT-5.6-Sol과 같다는 증거가 없다.

## 1. 모델이 정확히 무엇인가

DeepSeek의 Hugging Face 모델 카드는 0731을 preview를 대체하는 공식 release로 설명한다. DSpark speculative decoding 모듈이 붙은 구조이며, 현재 카드의 모델 크기는 **304B parameters**로 표시된다. 모델 카드는 MIT License와 text generation 경로를 제공하고, native vision processor나 vision tower를 제공한다고 말하지 않는다.

연결된 DeepSeek V4 기술 보고서는 preview 계열을 **284B total parameters, 13B activated parameters, 1M context**로 설명한다. 커뮤니티 글에서 자주 보이는 284B라는 숫자와 0731 카드의 304B는 그대로 섞어 쓰지 않는다. preview와 0731 checkpoint, 그리고 양자화된 serving artifact의 숫자가 다를 수 있기 때문이다.

0731은 일반적인 Jinja chat template만으로 끝나는 모델도 아니다. 공식 카드는 encoding 폴더의 Python encoder와 parser를 사용하고, reasoning_effort에 low, high, max를 제공한다. 따라서 weight가 올라왔다는 사실만으로 multi-turn reasoning과 tool call이 정상이라고 결론 내리지 않는다.

## 2. 단일 Spark 게시물의 숫자 검증

기준 자료는 [MiaAI-Lab의 one-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)다. 이 저장소는 DeepSeek 공식 저장소가 아니라 DGX Spark용 커뮤니티 실행 recipe다.

| 게시물의 주장 | 원문에서 실제 확인되는 것 | 판정 |
|---|---|---|
| 두 번째 장치 없이 실행 | GB10 한 대, TP=1, 128 GiB unified memory에서 3.0 bpw EXL3 artifact를 실행 | **사실**. 단, 이 quant/runtime 조합에 한정 |
| EXL3 3.0 bpw | REAP-K216 checkpoint, 256개 expert 중 216개를 유지하고 생존한 weight를 Trellis 방식으로 저장. weight 약 99.5 GiB | **사실**. full-expert FP8과 다름 |
| 384K context | MAX_MODEL_LEN 384000, MAX_NUM_SEQS 1, GPU memory utilization 0.94 설정 | **설정 상한**. 모든 부팅·동시성에서 보장되는 값 아님 |
| 약 440K KV cache | 한 cold boot에서 KV pool 439,622토큰 관찰 | **부팅 의존 측정값**. 고정 사양 아님 |
| 47 tok/s | 최신 README가 `start.sh`, 384K context의 structured decode를 44~47 tok/s로 기록함 | **조건부 재현값**. 일반 평균 아님 |
| 1,024 tok/s prefill | 370K 시험에서 요청 초반 약 1,024 tok/s 관찰 | **초반 peak**. 전체 입력 평균 아님 |
| 370K needle 통과 | 370,104토큰 random filler, 앞부분의 secret phrase를 끝에서 정확히 회수. thinking off, temperature 0 | **용량·회귀 테스트 통과**. 종합 지능 벤치마크 아님 |
| Q4_K_M 또는 Q5 GGUF와 비슷한 품질 | recipe 작성자가 제시한 EXL3와 GGUF의 경험적 mapping | **검증 대기**. 동일 task/perplexity 비교가 아님 |

### 2.1 47 tok/s를 어떻게 읽어야 하나

이 값은 “모델이 짧은 답을 언제나 47 tok/s로 생성한다”는 뜻이 아니다. 최신 recipe의 표기는 structured decode, `start.sh`, 384K context 설정이라는 한 조합이다. thinking effort, prompt 형태, 출력 길이, speculative acceptance, warm state에 따라 결과가 달라진다.

또한 같은 저장소는 c1 deep-context profile에서 MAX_NUM_SEQS를 2 이상으로 올리면 hybrid KV split 때문에 전체 pool이 줄어든다고 설명한다. 즉 **긴 단일 세션과 다중 사용자 처리량을 동시에 최대화하는 설정은 아니다.**

### 2.2 긴 문맥 숫자의 실제 비용

370,104토큰 시험의 recipe 기록은 다음과 같다.

| 항목 | 값 |
|---|---:|
| 모델 길이 설정 | 384,000 |
| 실제 prompt | 370,104토큰 |
| KV pool | 439,622토큰 |
| needle 위치 | 약 20번째 토큰 |
| needle 결과 | exact recall |
| 초기 prefill | 약 1,024 tok/s |
| 300K 이후 prefill | 약 350~614 tok/s |
| 전체 시험의 실효 prefill | 약 625 tok/s |
| end-to-end | 약 594초 |

따라서 “384K를 1,024 tok/s로 읽는다”가 아니라 **입력 초반은 빠르지만 깊은 context에서는 속도가 내려가며, 370K 요청 하나가 약 10분 걸린다**고 기록하는 편이 정확하다.

### 2.3 EXL3 품질 주의점

이 경로는 단순한 uniform 3-bit round-to-nearest가 아니다. REAP가 일부 expert를 제거하고, EXL3/Trellis가 남은 tensor에 비균일 bit allocation을 적용한다. 그래서 3.0 bpw라는 숫자만 보고 Q3 GGUF와 같다고 보면 안 된다.

recipe에는 3.0 bpw가 경험적으로 IQ4 또는 Q4~Q5에 가까울 수 있다는 mapping이 있지만, 이는 작성자의 community consensus와 체감 평가다. Q4_K_M/Q5와 동급이라는 문장을 책에 넣으려면 같은 prompt set, 같은 reasoning effort, 같은 output budget으로 독립 평가를 추가해야 한다.

## 3. 다른 공개 측정으로 교차 확인

아래는 서로 다른 maintainer와 runtime의 공개 측정이다. 같은 열의 숫자처럼 보이지만 checkpoint revision, quantization, speculative method, context, warm-up, concurrency가 모두 다르다. 따라서 순위표가 아니라 **범위와 조건의 증거**로 사용한다.

| 자료 | 구성·조건 | 보고된 결과 | 해석 |
|---|---|---|---|
| [one-Spark EXL3 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) | 1× Spark, EXL3 3.0 bpw, REAP-K216, SparkInfer/DSpark | structured decode 44~47 tok/s, 370K needle exact recall | 최신 deep-context 단일 스트림 경로 |
| [emiluzelac one-Spark 재현](https://github.com/emiluzelac/deepseek-v4-flash-0731-on-one-dgx-spark) | 1× Spark, Entrpi/ds4 HTTP serving | interactive 23~37 tok/s, 12 concurrent aggregate 약 59.7 tok/s, 127.5K needle, tool call 성공 | prefill·단일 응답·aggregate를 분리해 검증 |
| [Y-Computer recipe](https://github.com/Y-Computer/recipes/tree/main/benchmarks/published/2026-08-08-deepseek-v4-flash-0731-y-dspark-iq3m-dgx-spark) | 1× Spark, IQ3M target, 32K context | target 16.93 tok/s, sidecar speculative path 28.29 tok/s | quant과 drafter가 속도를 크게 바꿈 |
| [tpurtell EXL3 K2](https://github.com/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark) | 1× Spark, all experts를 포함한 EXL3 K2, matched sweep | c1 24.7, c4 aggregate 31.7, no-history prefill 1,317 tok/s | 47 tok/s가 보편적인 단일 Spark baseline은 아님 |
| [Weschera 2-Spark](https://github.com/Weschera/DeepSeek-V4-Flash-0731-DSpark-2x-DGX-Spark) | 2× Spark TP=2, 40K prompt, 512 output, 3회 median | speculation off 27.103, DSpark K7 83.808 tok/s | draft decoding 효과를 통제된 fixture로 확인 |
| [MiaAI-Lab 2-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md) | 2× Spark TP=2, FP8 weights, 1,048,576 profile | 2K c1 68.8, 8K c1 73.9, 32K c1 64.0, 128K c1 65.2 tok/s; 900K acceptance 약 875 prefill tok/s | 1M capacity와 short/c1 decode를 별도 확인 |
| [tonyd2wild 2-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) | 2× Spark, 1M profile, NVFP4 KV | c1 61.0, c2 aggregate 91.7, c4 151.1, c6 197.3 tok/s; 100K prefill 2,639 tok/s | 동시성·warm 상태가 결과를 바꿈 |
| [Reddit 2-Spark 비교](https://www.reddit.com/r/LocalLLM/comments/1vq5xjl/benchmark_deepseekv4flash_on_2x_dgx_sparks/) | 2× Spark TP=2, ConnectX-7, 515K prompt, 3 needles | 5 concurrent 75.64, 10 concurrent 102.65 tok/s; needle 3/3; cold prefill 1,113 tok/s | 사용자 측정. 1M 설정·515K retrieval의 보조 근거 |

독립 측정들을 종합하면 다음 범위가 현실적이다.

- **1대 단일 응답**: recipe와 workload에 따라 대략 20~47 tok/s. 44~47은 가능한 상단값이지 모든 요청의 기대값이 아니다.
- **1대 aggregate**: 짧은 요청을 여러 개 넣으면 총합 50~60 tok/s 근처가 보고되지만, 각 요청이 받는 속도는 5 tok/s 안팎까지 내려간다.
- **2대 단일 응답**: 40K~8K short fixture에서는 60~95 tok/s가 보고되지만, 32K 이상과 긴 reasoning에서는 더 낮아질 수 있다.
- **2대 aggregate**: concurrency를 올리면 100~340 tok/s까지 보고되지만, 이는 여러 응답의 합계다. 단일 사용자가 받는 속도와 혼동하지 않는다.

### 3.1 2026-08-21 단일 Spark 로컬 실행

2026-08-21에 이 리포의 DGX Spark 한 대에서 [MiaAI-Lab one-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)를 실제로 기동했다. 이 기록은 recipe README의 커뮤니티 수치를 대체하지 않고, 현재 장비와 현재 컨테이너에서 확인한 별도의 실측으로 관리한다.

#### 시작부터 API 준비까지

시작 시각은 두 가지로 나누어 기록한다. `compose.yml`이 생성된 시각은 실행 명령을 확인하기 위한 시작 표식이고, 컨테이너 생성 시각은 이미지 pull이 끝난 뒤 모델 서비스가 실제로 시작된 시각이다.

| 단계 | 한국 표준시 | 비고 |
|---|---:|---|
| `compose.yml` 생성 | 22:14:18 | 시작 명령을 추적하기 위한 기록 시각 |
| 컨테이너 생성 | 22:23:41 | 약 9분 23초의 이미지 pull 이후 |
| 모델 다운로드 시작 | 22:23:54 | Hugging Face 캐시 사용 시작 |
| 컨테이너 재시작 | 23:12:21 | `exl3-layer-030-tp4-rank3.safetensors` 다운로드가 멈춰 재시작. 기존 캐시는 보존 |
| API 준비 완료 | 23:24:44 | `/health` 요청이 HTTP 200을 반환한 시각 |

따라서 이 환경에서 기록한 소요 시간은 다음과 같다.

- 시작 표식부터 API 준비까지: 약 **1시간 10분 25초**
- 컨테이너 생성부터 API 준비까지: 약 **1시간 1분 3초**
- 첫 실행에는 이미지 pull, 약 106GB의 모델 캐시 다운로드, TP1 coalesce, CUDA graph 준비가 모두 포함된다.
- 실행 후 `/dev/nvme0n1p2`의 여유 공간은 약 91GB였다. `start.sh`의 `LOCAL_MIN_FREE_GIB=130`은 현재 상태에서 경고만 출력하고 중단하지 않으므로, 새 모델을 추가로 받거나 캐시를 다시 만들기 전에는 디스크를 먼저 확보해야 한다.

실행에 사용한 주요 식별자는 다음과 같다.

| 항목 | 값 |
|---|---|
| Hugging Face 저장소 | `0xSero/deepseek-v4-flash-0731-spark` |
| 테스트 recipe commit | `d1dc9e7` |
| 감사 시점 원격 `main` | `5ba18b7`. `start.sh`와 `image-patch/`는 테스트 commit과 동일하고 README의 context 표기만 갱신됨 |
| 모델 revision | `22f28d32b9b29b4352eaa380ff8c2c170b2847ab` |
| 모델 이름 | `deepseek-v4-flash-0731` |
| 실행 방식 | 저장소의 `./start.sh` 기본 프로필. 별도 환경 변수 override 없음 |
| 컨테이너 이미지 | `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4` |
| `MAX_MODEL_LEN` | `384000` |
| `MAX_NUM_SEQS` / batched tokens | `1` / `8224` |
| KV record | `stock432`, native NVFP4. `VLLM_DSV4_PADDED_NVFP4=0`, `KV_FP8_ROPE=0` |
| DSpark | `MODE=dspark`, `DSPARK_TOKENS=5`, K5 draft |
| vLLM kernel path | `load-format=instanttensor`, TP1, `kv-cache-dtype=nvfp4_ds_mla`, attention/moe/linear backend `B12X_MLA_SPARSE`/`b12x`/`b12x` |
| serving features | chunked prefill, prefix caching, async scheduling, FlashInfer autotune |
| parser | `deepseek_v4` tool parser and reasoning parser, auto tool choice enabled |
| GPU memory utilization | `0.94` |
| CUDA graph capture | 최대 `24`, capture sizes `6,12,24` |
| prefill scheduler | `--long-prefill-token-threshold 1024` |
| CPU KV offload | `KV_OFFLOAD_GB=0`, 비활성화 |
| checksum 검증 | `VERIFY_MODEL_CHECKSUMS=1` |
| 서버 기본 thinking | `true`, effort `max`. 벤치마크 요청에서는 `thinking=false`로 override |
| GPU KV cache | `469175` tokens |
| 모델 로딩 메모리 | `95.39 GiB` |
| 모델 캐시 전체 | 약 `106.35 GB` |
| TP1 manifest | `117005` tensors, logical tensor bytes `106.084 GB`; 48 files, on-disk `106.097 GB` |

이 실행은 저장소가 제공하는 pinned image와 `image-patch/`의 read-only bind mount를 함께 사용했다. 이 패치에는 tool-calling boot fix, SparkInfer NVFP4 prefill 수정, DSpark draft KV writer 수정, patched entrypoint가 포함된다. 따라서 이번 결과는 원본 이미지의 무수정 vLLM 결과가 아니라, 해당 저장소가 제공하는 one-Spark recipe 전체의 결과다.

레시피는 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`으로 CUDA graph 메모리의 보수적 예약을 줄인다. 실제 부팅 로그에도 graph 메모리를 KV 산정에 포함하지 않을 수 있으므로 OOM 위험을 확인하라는 경고가 남았다. 이번 부팅은 `469175` tokens의 KV pool을 만들고 healthy 상태가 되었지만, 이 옵션 때문에 KV pool과 부팅 성공 여부를 장비의 고정 사양으로 기록하지 않는다.

모델 로딩 자체는 로그에서 약 47.16초로 기록되었지만, 첫 실행의 전체 시간에는 다운로드와 weight coalesce, CUDA graph capture가 더 큰 비중을 차지했다. 컨테이너 재시작 뒤에는 이미 받은 파일을 캐시에서 재사용했으며, 최종 상태는 `healthy`, 재시작 횟수 `0`이었다.

#### API와 tool call 확인

`/v1/models`에서 모델 이름과 `max_model_len=384000`을 확인했다. 다음 smoke 요청은 `DS4_TEST_OK`를 정확히 반환했다.

| 테스트 | 조건 | 결과 |
|---|---|---|
| 모델 검색 | `/v1/models` | HTTP 200, `deepseek-v4-flash-0731`, `384000` context |
| 기본 생성 | temperature 0, thinking off, 최대 64 tokens | `DS4_TEST_OK`를 정확히 반환 |
| tool call | `lookup_weather`, 강제 tool choice, thinking off | HTTP 200, 함수 이름과 `{"city":"서울"}` arguments 생성 |

tool call은 parser 설정이 실제 함수 이름과 JSON arguments를 응답에 넣는지 확인한 결과다. 실제 도구 실행, 오류 복구, 여러 차례의 tool loop까지 검증한 결과는 아니다. 첫 tool 요청에서는 초기 JIT 비용으로 지연이 길었고, 재요청은 약 8.36초에 응답했다.

#### 단일 스트림 생성 속도

다음 조건으로 비스트리밍 요청 세 번을 순차적으로 보냈다.

- 동일한 이메일 검증 함수 작성 프롬프트에 실행 번호만 변경
- `thinking=false`, temperature 0, `max_completion_tokens=256`
- 단일 스트림, 동시 요청 없음
- 측정값은 요청 전체 wall time을 completion tokens로 나눈 값

| 실행 | 전체 시간 | completion tokens | end-to-end 속도 | 종료 이유 |
|---:|---:|---:|---:|---|
| 1 | 8.574초 | 256 | 29.86 tok/s | `length` |
| 2 | 7.894초 | 256 | 32.43 tok/s | `length` |
| 3 | 8.065초 | 256 | 31.74 tok/s | `length` |
| 평균 | 8.178초 | 256 | **31.34 tok/s** | 세 번 모두 출력 상한 도달 |

이번 측정의 31.34 tok/s는 요청 전체 시간을 포함한 end-to-end 수치다. 순수 decode kernel 속도나 recipe README의 structured decode 수치와 같은 지표가 아니다. 같은 조건에서 공개된 44~47 tok/s를 재현하지 못했으므로, 이 장비와 이 설정의 현재 기록은 **약 31 tok/s**로 남긴다. 출력 길이, prompt 길이, warm-up, speculative acceptance, 컨테이너 버전이 달라 직접적인 우열 비교는 하지 않는다.

#### 저장소 제공 `acceptance_c1.py` 정식 실행

앞의 세 번 측정은 API 동작을 확인하기 위한 보조 측정이었다. recipe 자체의 gate를 확인하기 위해 저장소의 [C1 acceptance harness](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark/blob/main/image-patch/acceptance_c1.py)를 수정하지 않고 기본 인자로 실행했다.

```bash
python3 image-patch/acceptance_c1.py \
  --base-url http://127.0.0.1:8888 \
  --model deepseek-v4-flash-0731
```

이 명령은 semantic gate, strict JSON schema, Python·Rust·TypeScript·CUDA C++·Go의 512-token code decode, 252,047-token cold prefill을 순서대로 검사한다.

| gate | 결과 | 실측 |
|---|---|---|
| semantic | 통과 | `17 × 19 = 323` |
| JSON schema | 통과 | `language=Python`, `answer=323`, `valid=true` |
| code C1 decode | 통과 | 최저 35.734, 중앙값 38.636, 평균 39.329 tok/s. 기준 35 tok/s |
| cold prefill 응답 | 통과 | `PREFILL OK.` 반환, `cached_tokens=0` |
| cold prefill token count | 실패 | 목표 252,047, 실제 chat usage 251,968 tokens |
| cold prefill 속도 | 실패 | 975.441 tok/s. 기준 1,000 tok/s |
| 전체 프로세스 | 실패 | prefill token count와 속도 gate에서 실패 |

이 결과는 “DeepSeek가 GPT-5.6 Sol의 `reasoning_effort=max`보다 낮다”는 비교 결과가 아니다. C1은 이 로컬 serving recipe의 구조화 출력과 code decode, prefill 속도를 정한 기준으로 확인하는 하니스다. 현재 결과는 code decode 기준은 통과했지만, prefill 기준은 통과하지 못했다는 뜻으로만 해석한다. GPT-5.6 Sol과 동일 prompt, 동일 agent harness, 동일 비용·wall time 조건으로 실행한 직접 비교는 아직 없다.

##### C1 하니스의 token accounting 차이

하니스 소스에서 `build_exact_prefill_prompt()`는 `/tokenize` 요청을 보낼 때 `messages`와 `add_special_tokens=true`만 사용한다. 반면 실제 `stream_chat()` 요청에는 `chat_template_kwargs: {"thinking": false}`가 추가된다. 같은 짧은 진단 prompt에서도 `/tokenize`는 92 tokens, 실제 chat usage는 13 tokens로 계산되어 **79 tokens 차이**가 재현됐다.

따라서 252,047-token cold prefill 결과에는 두 가지를 분리해서 기록해야 한다.

1. 하니스가 만든 prompt는 실제 chat 요청에서 251,968 tokens로 계산되어 token-count gate를 실패했다.
2. 2026-08-21 실행에서 실제로 처리된 251,968 tokens의 prefill 속도는 975.441 tok/s였다. 2026-08-22 같은 recipe 재실행은 985.377 tok/s로 개선됐지만, 두 번 모두 하니스의 1,000 tok/s 기준보다 낮았다.

하니스의 token accounting을 먼저 같은 `thinking=false` 조건으로 맞춘 뒤에야 252,047-token gate를 완전히 재검증할 수 있다. 현재 결과만으로 recipe의 252K prefill 수치를 통과했다고 기록하지 않는다.

#### 이번 실행에서 확인하지 않은 항목

- 370K needle의 exact recall
- 384K에 가까운 실제 장문 요청의 품질과 end-to-end 시간
- GPT-5.6-Sol과 동일 prompt 및 동일 agent harness를 사용한 직접 비교
- 장시간 multi-turn tool loop와 동시성별 aggregate throughput

이번 실행으로 확인한 것은 단일 Spark에서 해당 커뮤니티 serving 경로가 실제로 기동되고, 384K 설정을 가진 OpenAI-compatible API와 tool parser, 그리고 로컬 mock을 이용한 한 번의 정상 tool loop를 제공한다는 점이다. 370K needle 통과와 44~47 tok/s는 여전히 recipe 작성자의 별도 결과로 분리해 인용한다.

### 3.2 2026-08-22 C1 재실행

서버를 healthy 상태로 유지한 채 같은 저장소의 `acceptance_c1.py`를 다시 실행했다. 컨테이너 image digest는 `sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`, 재시작 횟수는 0회, `/v1/models`의 `max_model_len`은 384,000이었다. 전체 raw JSON은 [C1 실측 결과](results-deepseek-c1-2026-08-22.json)로 보존한다.

| gate | 2026-08-22 결과 |
|---|---|
| semantic | 통과, `17 × 19 = 323` |
| JSON schema | 통과, `language=Python`, `answer=323`, `valid=true` |
| code C1 decode | 통과, 최저 37.553, 중앙값 41.358, 평균 40.915 tok/s |
| cold prefill 응답 | 통과, `PREFILL OK.`, `cached_tokens=0` |
| cold prefill token count | 실패, 목표 252,047 / 실제 251,968 tokens |
| cold prefill 속도 | 실패, 985.377 tok/s / 기준 1,000 tok/s |
| 전체 프로세스 | 실패, token count와 prefill speed gate |

같은 서버에서 별도 tool-call 요청도 실행했다. `lookup_weather` 함수와 `{"city":"서울"}` arguments가 생성됐으며, raw 응답은 [tool-call 실측 결과](results-deepseek-tool-call-2026-08-22.json)로 보존한다. `finish_reason=length`였으므로 함수 arguments 생성은 통과했지만, 충분한 completion budget에서 자연스럽게 종료한 agent loop까지 확인한 것은 아니다.

#### Multi-turn mock tool loop

두 번째 요청에서는 첫 응답의 `tool_calls`를 파싱한 뒤 실제 네트워크 도구 대신 로컬 mock 결과(`맑음`, `24°C`)를 `tool` 메시지로 되돌려 보냈다. 모델은 두 번째 응답에서 추가 tool call 없이 최종 문장을 반환했다. 첫 직접 실행은 2.524초 + 1.513초, 전체 4.038초였고, 재현 스크립트 실행은 전체 4.740초로 모든 검사를 통과했다. 이 결과는 OpenAI-compatible message contract와 한 번의 정상 tool loop를 검증하지만, 외부 도구의 실제 실행·오류 복구·다중 tool 병렬성·에이전트 작업 성공률을 의미하지 않는다. 원본은 [multi-turn tool loop 실측](results-deepseek-tool-loop-2026-08-22.json)과 [재현 스크립트 raw 결과](results-deepseek-tool-loop-script-2026-08-22.json)로 보존한다.

재현 명령:

```bash
python3 tests/tool_loop_smoke.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731 \
  --strict
```

### 3.3 하니스 종류와 출처별 증거

이번 결과를 해석할 때는 “DeepSeek 하니스를 썼는가”와 “README의 성능 측정을 그대로 재현했는가”를 나눠야 한다. 이번 실행에는 저장소가 제공한 [`image-patch/acceptance_c1.py`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark/blob/main/image-patch/acceptance_c1.py)를 기본 인자로 사용했다. 따라서 DeepSeek recipe의 C1 acceptance는 실행한 것이 맞다.

공식 모델 카드가 말하는 `minimal DeepSeek Harness`는 Code Agent 품질 벤치마크용 별도 harness다. 공식 표의 `reasoning_effort=max`, temperature, top-p 조건을 평가하기 위한 것이며, 이번 Spark serving C1 실행에는 사용하지 않았다. 이 차이는 GPT-5.6-Sol과의 품질 비교에는 중요하지만, C1 prefill gate가 실패한 직접 원인은 아니다.

다만 C1은 모델 품질 종합평가나 README의 needle benchmark가 아니다. C1은 semantic 출력, strict JSON, 다섯 언어의 code decode, 252,047-token cold prefill을 통과시키는 serving gate다. README의 370,104-token needle과 44~47 tok/s는 별도의 측정 조건으로 기록되어 있고, 저장소에는 그 실험을 그대로 재실행하는 독립 스크립트가 포함되어 있지 않다.

또한 C1의 prefill 실패는 두 층으로 봐야 한다.

1. `build_exact_prefill_prompt()`의 `/tokenize` 요청과 실제 `stream_chat()` 요청이 서로 다른 chat template 조건을 사용해 79-token 차이가 생겼다. 이 부분은 하니스의 token accounting 문제다.
2. 실제 chat usage로 계산된 251,968 tokens의 first-token 기준 prefill도 2026-08-21에는 975.441 tok/s, 2026-08-22에는 985.377 tok/s로 C1의 1,000 tok/s gate를 넘지 못했다. 따라서 “하니스가 달라서 전부 실패했다”고 결론 내릴 수는 없지만, C1 gate를 README 수치와 같은 benchmark로 부를 수도 없다.

#### 2026-08-22 upstream recipe revision 확인

실험에 사용한 local recipe clone은 `d1dc9e70d277746e4e369cc68f54d5c67a6afae8`였고, 확인 시점의 upstream `main`은 `5ba18b71b5c08cb7e7a5cb783577fb9832d0ff67`였다. 두 revision의 차이는 README의 측정 라벨을 `330k context`에서 `384k context`로 고친 한 줄뿐이었다. 실행에 직접 관여하는 `image-patch/acceptance_c1.py`와 `image-patch/serve-ds4-flash.sh`의 blob hash는 양쪽이 각각 같았다.

따라서 2026-08-22 C1 결과는 현재 upstream recipe의 실행 코드와 동등한 revision에서 얻은 결과로 볼 수 있지만, upstream README가 갱신된 뒤의 문서 표기와 실행 revision은 구분해 기록한다. README의 44~47 tok/s는 여전히 작성자 측 측정값이며, 우리 C1의 985.377 tok/s prefill 결과로 대체하지 않는다.

#### 출처별 상태

아래 등급은 숫자의 우열이 아니라 증거의 성격을 표시한다. A는 공식 모델·논문 자료, B는 실행 가능한 recipe, C는 조건이 적힌 커뮤니티 benchmark, D는 X 게시물이나 운영 경험이다. 같은 모델명이라도 checkpoint, quantization, engine, speculative decoding, prompt 길이와 측정 지표가 다르면 숫자를 합치지 않는다.

| 등급 | 출처 | 보고되거나 확인된 내용 | 현재 해석 |
|---|---|---|---|
| A | [DeepSeek 공식 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)·[기술 보고서](https://arxiv.org/abs/2606.19348) | Code·agent benchmark, 모델 구성과 공개 weight | 모델의 공식 능력 참고값. 단일 Spark EXL3 실측은 아님 |
| B | [MiaAI-Lab 단일 Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) | EXL3 3.0 bpw, SparkInfer, DSpark K5, 384K 설정, README의 44~47 tok/s와 370,104-token needle | 실행 가능한 원문 recipe. 이번 C1 결과와 동일한 측정은 아님 |
| B | [동일 recipe의 C1 하니스](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark/blob/main/image-patch/acceptance_c1.py) | 35 tok/s code gate, 1,000 tok/s prefill gate, 252,047-token cold prefill | 이번에 실제 실행. code는 통과, prefill은 accounting과 속도 모두 별도 기록 |
| C | [emiluzelac 단일 Spark 재현](https://github.com/emiluzelac/deepseek-v4-flash-0731-on-one-dgx-spark) | 다른 runtime·quant 조건에서 23~37 tok/s와 tool call 사례 | 독립 재현의 범위를 보여주는 자료. MiaAI README와 직접 순위 비교 금지 |
| C | [Y-Computer benchmark](https://github.com/Y-Computer/recipes/tree/main/benchmarks/published/2026-08-08-deepseek-v4-flash-0731-y-dspark-iq3m-dgx-spark) | IQ3M와 DSpark speculative 조건의 단일 Spark 결과 | quant와 draft 조건이 달라 별도 행으로 유지 |
| C | [tpurtell EXL3 K2 측정](https://github.com/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark) | c1·c4와 matched workload, prefill을 분리한 측정 | peak decode와 실사용 workload의 차이를 보여주는 자료 |
| C | [NVIDIA Developer Forum 단일 0731](https://forums.developer.nvidia.com/t/1x-spark-deepseek-v4-flash-0731-1-000-tok-s-prefill-59-tok-s-multi-agent-serving/378855)·[EXL3/SparkInfer](https://forums.developer.nvidia.com/t/c1-1058pp-s-52-tg-s-on-1x-dgx-spark-on-deepseek-v4-flash-0731-full-256-experts/379863) | single Spark prefill, c1, matched c4 등 조건별 보고 | 포럼 작성자의 환경·fork·quant 결과. 공식 baseline 아님 |
| C | [NVIDIA Developer Forum 4대 TP=4](https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-4x-dgx-spark-via-vllm-jasl-fork-tp-4-rdma-mtp-49-54-tok-s-single-stream-full-recipe-the-traps/373808) | RDMA·NCCL·vLLM fork 조건에서 single과 aggregate 결과 | 4대 확장성 자료. 1대 결과와 섞지 않음 |
| C | [Reddit 2대 Spark 비교](https://www.reddit.com/r/LocalLLM/comments/1vq5xjl/benchmark_deepseekv4flash_on_2x_dgx_sparks/) | 2대의 prompt, needle, c5/c10 throughput 보고 | 사용자 benchmark. exact commit과 전체 로그가 없으면 재현 완료로 표시하지 않음 |
| B/C | [2대 Spark 공식 FP8·DSpark 문서](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md)·[1M recipe](https://github.com/Weschera/DeepSeek-V4-Flash-0731-DSpark-2x-DGX-Spark) | TP=2, 200K~1M context, MTP·DSpark 구성 | 2대 모델의 메모리·통신 조건을 비교하는 자료. 단일 EXL3와 다른 경로 |
| D | [Ash Hart MCDMA X 게시물](https://x.com/ashxhart/status/2089749434087227672?s=20) | Mac Studio와 Spark 사이 USB-C direct-memory prototype, 링크 대역폭·지연 보고 | DeepSeek tok/s benchmark가 아님. 독립 검증 전에는 prototype report로만 기록 |
| D | [Blackwellboy 저클록 X 사례](https://x.com/Blackwellboy/status/2090611479653622261?s=20) | P0·96% utilization인데 799MHz·19.5W였고 전원 완전 차단 후 회복 | 모델 순위가 아니라 DGX Spark 운영 장애 진단 자료 |
| D/K | [아카라이브 알파카 원문](https://arca.live/b/alpaca/180567610?p=1)·[서버포럼 AI 자료](https://svrforum.com/ai/3170124)·[Wikidocs 참고문헌](dgx-spark-book-references-2026-08.md#10-wikidocs한국어-자료) | 국내 사용자의 모델 선택, 구성, 속도와 실패 경험 | 접근제어와 측정 조건의 차이가 있어 발견·교차검증용. 공식 benchmark로 사용하지 않음 |
| A/C | [StateM 논문](https://arxiv.org/abs/2608.15089)·[GPT-5.6-Sol 공식 페이지](https://developers.openai.com/api/docs/models/gpt-5.6-sol) | Sol과 DeepSeek의 agent harness·runbook 결과 | harness가 달라진 비교다. C1 tok/s를 Sol 속도나 품질 점수로 환산하지 않음 |
| 현장 실측 | [2026-08-21 이번 실행 기록](#31-2026-08-21-단일-spark-로컬-실행) | recipe 기동, API, tool parser, C1, 보조 decode 수치 | 이 장비에서 재현한 사실. README의 원문 주장을 대체하지 않음 |

전체 X·GitHub·NVIDIA Forum·Reddit·한국 커뮤니티 출처와 링크 상태는 [DGX Spark 참고문헌 인덱스](dgx-spark-book-references-2026-08.md), [NVIDIA 포럼 리서치](dgx-spark-nvidia-forum-research-2026-08.md), [클러스터 모델 리서치](dgx-spark-cluster-model-research-2026-08.md)에 모아 둔다. 이 문서에서는 DeepSeek 0731의 수치와 이번 실행에 직접 필요한 출처만 다시 묶었다.

## 4. 모델 자체의 품질은 어느 정도인가

DeepSeek 공식 0731 카드가 공개한 agent/code 표는 다음과 같다. 표의 점수는 DeepSeek가 제시한 비교 결과이고, 모든 행의 harness·reasoning budget·세부 재현 조건이 동일한지는 별도로 확인해야 한다.

| 평가 | DeepSeek V4 Flash 0731 | V4 Pro Preview | GLM-5.2 | Opus 4.8 |
|---|---:|---:|---:|---:|
| Terminal-Bench 2.1 | 82.7 | 72.1 | 81.0 | 85.0 |
| NL2Repo | 54.2 | 38.5 | 48.9 | 69.7 |
| Cybergym | 76.7 | 52.7 | - | 83.1 |
| DeepSWE | 54.4 | 12.8 | 46.2 | 58.0 |
| Toolathlon-Verified | 70.3 | 55.9 | 59.9 | 76.2 |
| Agents Last Exam | 25.2 | 16.5 | 23.8 | 25.7 |
| AutomationBench Public | 25.1 | 12.8 | 12.9 | 27.2 |
| DSBench-FullStack | 68.7 | 41.8 | 61.8 | 71.6 |
| DSBench-Hard | 59.6 | 31.1 | 54.5 | 71.7 |

공식 카드의 중요한 조건은 Code Agent 공개 벤치마크에 minimal DeepSeek Harness, reasoning_effort max, temperature 1.0, top_p 0.95를 사용했다는 점이다. 또한 DSBench 두 행은 internal test set이다.

이 표만 보면 Flash 0731은 V4 Pro Preview보다 강하고, Opus 4.8과는 일부 코딩·에이전트 지표에서 가까우나 모든 지표에서 동급은 아니다. 무엇보다 이 표는 단일 Spark EXL3 artifact의 품질 점수가 아니다. full official checkpoint와 REAP-K216 3.0 bpw 경로를 같은 모델 점수로 기록하지 않는다.

## 5. GPT-5.6-Sol과 비교

### 5.1 확인된 차이

| 항목 | DeepSeek V4 Flash 0731 | GPT-5.6-Sol |
|---|---|---|
| 제공 형태 | 공개 weight, MIT License, 로컬 serving 가능 | OpenAI hosted API/Codex 모델 |
| 모델 카드 context | V4 계열은 1M context를 지향. one-Spark recipe는 384K, 2-Spark recipe는 1,048,576 설정 | API context 1,050,000, max output 128,000 |
| 모델 실행 | SparkInfer, vLLM/SGLang, DSpark 등 recipe와 runtime 의존 | OpenAI endpoint, reasoning effort none~max와 hosted tools 제공 |
| native vision | 0731 checkpoint는 text-only. 별도 vision shim은 caption pipeline | API 모델은 image input과 여러 hosted tools 지원 |
| 공개 코드 점수 | 공식 카드 Terminal-Bench 82.7 | OpenAI는 Terminal-Bench 2.1에서 state of the art라고 발표 |
| 공개 비교 수치 | StateM 논문에서 standard DS4 82.7을 사용 | StateM의 reference는 Sol xhigh 84.9, Sol max 88.8 |
| raw decode 속도 | one-Spark 공개 recipe 44~47 tok/s, 다른 재현은 23~37 tok/s | standard API에 고정 tok/s 없음. OpenAI는 선택 고객의 Cerebras Ultrafast에서 최대 750 output tok/s를 발표 |
| API 비용 | 장비·전력·운영 비용 | input 1M당 5달러, output 1M당 30달러. 272K 초과 입력은 별도 multiplier |

OpenAI 공식 문서는 GPT-5.6-Sol을 frontier model로 설명하고, Terminal-Bench 2.1에서 state of the art라고 발표하지만 launch 글의 본문에는 점수표 전체가 들어 있지 않다. 별도의 [StateM arXiv 논문](https://arxiv.org/abs/2608.15089)은 Sol xhigh reference 84.9, Sol max 88.8을 기록하고, 같은 실행 runtime과 runbook을 적용한 DeepSeek V4 Flash를 82.7에서 88.1, 공통 88-task core에서 89.1까지 올렸다고 보고한다.

이 비교는 모델만 바꾼 순수 대결이 아니다. StateM은 agent runtime과 runbook을 함께 바꾸었고, DeepSeek 쪽에는 adaptation 비용도 들어갔다. 따라서 이 자료가 말해주는 것은 **DeepSeek V4 Flash가 좋은 harness를 붙이면 Sol과 가까운 코딩-agent 점수를 낼 수 있다**는 정도다. “모든 작업에서 GPT-5.6-Sol과 동급”의 증거는 아니다.

### 5.2 속도 주장의 결론

“GPT-5.6-Sol과 비슷한 속도”라는 문장은 현재 공개 자료로 검증되지 않는다.

- DeepSeek의 44~47 tok/s는 로컬 decode 출력 속도다.
- GPT-5.6-Sol은 provider, queue, reasoning effort, tool call, prompt 길이에 따라 API latency가 달라지는 hosted 모델이다.
- OpenAI가 발표한 750 tok/s는 Cerebras Ultrafast의 선택 고객용 상한이며, 일반 API의 고정 baseline이 아니다.
- 따라서 raw output tok/s를 비교하면 오히려 서로 다른 serving 계층을 비교하게 된다.

그 표현은 “사용자가 체감하기에 빠르다”는 홍보성 문장으로는 이해할 수 있지만, **성능 동률이나 속도 동률로 문서에 기록하면 안 된다.**

### 5.3 품질 주장의 결론

현재 근거에 맞는 표현은 다음과 같다.

> DeepSeek V4 Flash 0731은 로컬에서 돌릴 수 있는 코드·에이전트 모델 중 매우 강한 편이며, Terminal-Bench 계열에서는 GPT-5.6-Sol과 가까운 구성이 보고됐다. 그러나 모델 자체가 GPT-5.6-Sol과 전반적으로 동급이라고 확인된 것은 아니고, broad reasoning·멀티모달·hosted tool 안정성·긴 작업의 회복력까지 포함하면 별도 비교가 필요하다.

## 6. DGX Spark 한 대와 두 대 중 무엇을 선택할까

| 목적 | 권장 구성 | 기대치 |
|---|---|---|
| 혼자 쓰는 로컬 코딩 agent, 비용과 privacy 우선 | 1× Spark EXL3/SparkInfer | 20~40 tok/s를 기본 기대, 잘 맞는 structured workload에서 44~47 가능 |
| 긴 한 세션, 300K 이상 문맥 실험 | 1× Spark deep-context recipe | 384K 설정과 370K retrieval 가능. prefill과 답변까지 오래 걸리는 것을 감수 |
| 공식 checkpoint에 가까운 운영, 1M ceiling | 2× Spark TP=2 FP8/DSpark | 1M profile과 60~75 tok/s급 c1 보고. 긴 context에서는 TTFT와 per-stream 하락 |
| 여러 agent 동시 처리 | 2× Spark TP=2, 256K~1M profile | aggregate는 늘지만 요청당 tok/s는 내려감. c2/c4/c6를 별도로 측정 |
| DeepSeek brain + Qwen worker | 2×2로 총 4대 | DeepSeek TP=2와 Qwen TP=2를 독립 pool로 운영. 두 모델의 memory를 합치는 구성 아님 |

한 대를 사서 바로 안정적인 API 서비스로 쓰려는 경우에는 최신 EXL3 recipe를 실험 프로필로 두고, 모델 자체 품질과 tool loop는 별도의 테스트로 확인한다. “47 tok/s와 370K needle을 통과했으니 GPT-5.6-Sol 대체품”이라는 식의 구매 결론은 아직 이르다.

## 7. 이 저장소에서 다음에 측정할 것

같은 하니스로 다음 조건을 고정해야 GPT-5.6-Sol과 의미 있는 비교가 된다.

1. 모델 revision, quantization, engine image, vLLM/SGLang commit, driver와 CUDA를 기록한다.
2. speculation off와 DSpark on을 분리하고, draft acceptance를 함께 저장한다.
3. prompt 2K, 32K, 128K, 370K와 output 128, 512를 각각 측정한다.
4. c1, c2, c4, c6에서 TTFT, prefill, per-stream decode, aggregate decode를 나눠 기록한다.
5. thinking off, low, high, max를 별도 행으로 기록한다. reasoning token이 속도와 context를 바꾼다.
6. forced tool call과 실제 multi-turn tool loop를 분리한다. 함수 이름과 JSON arguments만 맞은 결과는 agent recovery 검증이 아니다.
7. needle은 한 번이 아니라 앞·중간·뒤에 여러 개를 두고, retrieval과 요약·변환·코딩을 분리한다.
8. GPT-5.6-Sol은 같은 Terminal-Bench subset과 같은 agent harness로 호출하고, score·실패 이유·출력 token·비용·wall time을 함께 저장한다.

이 측정이 끝나기 전까지 책의 최종 문구는 **“DGX Spark에서 매우 강한 로컬 코드 에이전트 후보”**로 유지한다. **“GPT-5.6-Sol과 동급” 또는 “항상 47 tok/s”는 사용하지 않는다.**

## 참고 자료

- [DeepSeek V4 Flash 0731 공식 Hugging Face 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [DeepSeek V4 기술 보고서](https://arxiv.org/abs/2606.19348)
- [단일 Spark EXL3/SparkInfer recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
- [2대 Spark DSpark 측정 문서](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md)
- [2대 Spark 40K controlled result](https://github.com/Weschera/DeepSeek-V4-Flash-0731-DSpark-2x-DGX-Spark/blob/main/results/qualified-summary.md)
- [독립 단일 Spark 재현](https://github.com/emiluzelac/deepseek-v4-flash-0731-on-one-dgx-spark)
- [Y-Computer 단일 Spark benchmark](https://github.com/Y-Computer/recipes/tree/main/benchmarks/published/2026-08-08-deepseek-v4-flash-0731-y-dspark-iq3m-dgx-spark)
- [EXL3 K2 matched Spark sweep](https://github.com/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark)
- [2대 Spark 1M NVFP4/KV 측정](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
- [2대 Spark Reddit head-to-head](https://www.reddit.com/r/LocalLLM/comments/1vq5xjl/benchmark_deepseekv4flash_on_2x_dgx_sparks/)
- [GPT-5.6-Sol 공식 API 모델 페이지](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [OpenAI GPT-5.6-Sol preview](https://openai.com/index/previewing-gpt-5-6-sol/)
- [StateM Terminal-Bench 비교 논문](https://arxiv.org/abs/2608.15089)
