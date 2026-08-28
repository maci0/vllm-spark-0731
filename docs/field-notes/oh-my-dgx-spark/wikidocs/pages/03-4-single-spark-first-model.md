# 03-4. 단일 Spark 첫 모델 상세

상태: 초안. BF16 smoke test는 로컬 검증을 마쳤고, NVFP4/SGLang 경로는 재현을 기다리고 있다.

이 장은 첫날부터 최고 속도를 내기 위한 장이 아니다. “모델이 된다”는 말을 단계별로 증명한다. 먼저 서버가 정상적으로 올라오고, 답변하며, 측정할 수 있는 상태를 만든 뒤 최적화한다.

## 3분 이해 (ELI5)

첫 모델 실험은 집에 새 엔진을 달고 바로 고속도로에 나가는 일이 아니다.

```text
BF16 기준선: 시동·브레이크 확인
recipe 최적화: 속도 향상
benchmark: 같은 도로에서 기록
```

먼저 응답·JSON·기본 기능을 확인하고, 그다음 속도를 최적화한다.

## 4.1 “된다”의 사다리

```text
loaded       → weight와 runtime이 메모리에 올라감
generates    → 기본 prompt에 정상 답변
serves       → endpoint가 반복 요청에 응답
benchmarked  → 고정 조건으로 속도를 측정
tool-tested  → parser와 arguments가 정상
agent-tested → 여러 단계 작업과 실패 복구를 통과
```

한 단계의 성공을 다음 단계의 성공으로 옮겨 적지 않는다. 특히 모델이 로드됐다는 사실만으로 빠른 서버나 코딩 에이전트가 된 것은 아니다.

## 4.2 시작 전에 선택할 두 경로

| 경로 | 목적 | 현재 상태 |
|---|---|---|
| BF16 vLLM baseline | 기능·API·smoke test | 이 저장소에서 Qwen3.8 파생 모델로 검증 |
| Qwen3.8 NVFP4 + SGLang/DFlash2 | 속도·동시성 | 포럼/GitHub recipe를 기준으로 후속 재현 |
| DeepSeek V4 Flash 0731 EXL3 + SparkInfer/DSpark | 단일 Spark 장문·supervisor 실험 | 공개 recipe의 384K 설정·약 440K KV pool·44–47 tok/s 구조화 decode를 조건부로 재현 |

BF16 baseline은 최종 속도 비교값이 아니다. 대신 서버·tokenizer·thinking·JSON·vision·long context에서 발생하는 실패를 분리해서 찾기 쉽다. 최적화된 NVFP4 recipe는 기능 baseline을 통과한 뒤에 적용한다.

## 4.3 Preflight

```bash
hostnamectl
uname -a
nvidia-smi
free -h
df -h
```

다음 항목을 기록한다.

- DGX OS/Ubuntu 버전
- kernel과 NVIDIA driver
- CUDA/PyTorch/vLLM 버전
- 모델 파일 크기와 revision
- 현재 실행 중인 다른 inference process
- idle memory와 temperature

같은 Spark에서 기존 모델 서버가 실행 중이면 새 모델의 측정값에 영향을 준다. 첫 baseline을 측정할 때는 다른 GPU workload를 중지한다. 중지하지 못했다면 해당 workload를 측정 조건에 포함했다고 기록한다.

## 4.4 저장소에서 검증된 BF16 baseline

현재 저장소에서 직접 확인한 서버는 `OBLITERATUS/Qwen3.8-27B-OBLITERATED` BF16 파생 모델이다. 원본 Qwen3.8과는 다른 모델이므로, 이 명령으로 확인한 기능을 원본 모델의 품질로 일반화하지 않는다.

```bash
vllm serve OBLITERATUS/Qwen3.8-27B-OBLITERATED \
  --host 127.0.0.1 \
  --port 8083 \
  --served-model-name qwen3.8-27b-obliterated \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.50 \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --generation-config vllm
```

이 설정은 메모리 여유를 크게 둔 smoke-test용이다. `gpu-memory-utilization=0.50`을 최적 속도 설정으로 제시하는 것은 아니다.

서버가 올라오면 먼저 모델 endpoint를 확인한다.

```bash
curl http://127.0.0.1:8083/v1/models
```

간단한 한국어 요청:

```bash
curl http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-obliterated",
    "messages": [{"role": "user", "content": "한국어로 짧게 답하세요. 2 더하기 2는?"}],
    "max_tokens": 64,
    "temperature": 0,
    "repetition_penalty": 1.15,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## 4.5 기능 smoke test

저장소의 dependency-free 테스트는 models endpoint, 한국어 산수, Python, JSON, 멀티턴 회수, thinking, long context, 4-way concurrency를 확인하도록 구성되어 있다.

```bash
python3 tests/qwen38_smoke.py \
  --base-url http://127.0.0.1:8083/v1 \
  --model qwen3.8-27b-obliterated
```

이미지와 32K 문맥은 명시적으로 켠다.

```bash
python3 tests/qwen38_smoke.py \
  --base-url http://127.0.0.1:8083/v1 \
  --model qwen3.8-27b-obliterated \
  --image /path/to/image.jpg \
  --long-context-tokens 32000
```

응답이 왔다는 사실만으로 성공을 판정하지 않는다. 각 테스트가 정한 기준을 확인한다.

| 테스트 | 통과 기준 |
|---|---|
| models | served model id가 예상값과 같음 |
| 한국어/코드 | 내용이 비어 있지 않고 형식이 맞음 |
| JSON | 응답을 실제 JSON parser로 읽음 |
| 멀티턴 | marker를 정확히 회수 |
| thinking | on/off가 의도대로 동작 |
| long context | marker 회수와 memory peak 기록 |
| concurrency | 4개 모두 성공, error rate 기록 |
| image | 이미지에서 보이는 것만 설명 |

## 4.6 Tool calling은 별도 서버로 검증한다

기능 테스트 서버에 나중에 tool parser를 붙여 결과를 섞지 않는다. tool calling을 테스트할 때는 별도 포트에서 서버를 다시 시작한다.

```bash
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

파서는 모델과 chat template에 맞아야 한다. `tool_calls: null`이 반환되면 모델의 tool 능력이 낮다고 결론 내리기 전에 parser·request format·reasoning mode·served model name을 먼저 확인한다.

## 4.7 단일 Spark DeepSeek V4 Flash 최신 경로

[MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)는 GB10/SM121 한 대에서 DeepSeek V4 Flash 0731을 EXL3 3.0 bpw와 SparkInfer로 실행하는 최신 공개 recipe다. 이 recipe는 0xSero의 REAP-K216 가중치, 즉 256개 expert 중 216개를 유지한 가중치와 DSpark K5/K64 draft, native NVFP4 KV 경로를 조합한다. 그러므로 원본 full-FP8·full-expert 실행과 같은 모델 프로필로 기록하지 않는다.

README가 제시하는 기본 프로필은 `MAX_NUM_SEQS=1`, `MAX_MODEL_LEN=384000`, `GPU_MEMORY_UTILIZATION=0.94`다. 공개 측정에는 structured decode 44–47 tok/s, 약 439,622토큰 KV pool, 320,037/370,104토큰 needle exact recall이 포함된다. thinking을 끈 것은 needle stress test의 조건으로 명시되어 있으며, 44–47 tok/s 측정과 같은 조건이라고 단정하지 않는다. “1024 tok/s prefill”은 초기 구간에서 얻은 수치다. 370K 시험에서는 300K 이후 속도가 약 350–614 tok/s로 내려가고 실효 prefill은 약 625 tok/s로 보고되므로, 두 수치를 하나의 상시 성능으로 기록하지 않는다. 자세한 판정은 [DeepSeek V4 Flash 0731 성능 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/deepseek-v4-flash-0731-performance-research-2026-08.md)를 따른다.

이 경로는 긴 단일 세션의 가능성을 보여주는 유용한 실험 자료다. 그러나 다음 항목까지 증명한 것은 아니다.

- full-expert 또는 원본 FP8과 동등한 품질
- c4/c8 다중 스트림 성능
- 일반 장문 문서의 전반적인 retrieval/추론 품질
- 장시간 agent loop와 tool-call recovery

### 재현 시작점

```bash
git clone https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark.git
cd DeepSeek-v4-Flash-One-DGX-Spark
git log -1 --oneline
free -h
df -h .
./start.sh compose-gen
./start.sh
curl -sS http://127.0.0.1:8888/health
curl -sS http://127.0.0.1:8888/v1/models
```

첫 부팅에는 약 107GB weight 다운로드, TP4→TP1 coalesce, K64 draft 빌드와 CUDA graph capture가 필요할 수 있다. 저장소가 생성한 compose를 기준으로 사용하고, `start.sh`가 만든 설정은 임의로 수정하지 않는다. README의 `GPU_MEMORY_UTILIZATION=0.94`와 EarlyOOM 비활성화는 이 recipe가 메모리를 적극적으로 점유하도록 만든 실험 조건이다. 이를 다른 모델이나 일반 시스템의 보편적인 안전 권장값으로 복사하지 말고, free memory·OOM·재부팅 가능성을 먼저 확인한다.

`MAX_NUM_SEQS=4`로 동시성을 높일 수 있지만 KV pool은 줄어든다. 이 recipe의 중심은 c1/deep-context 단일 스트림이다. 따라서 44–47 tok/s와 370K needle 결과를 다중 사용자 서비스 성능으로 확대해 해석하지 않는다.

## 4.8 Qwen3.8 NVFP4/SGLang으로 넘어가는 순서

[Qwen3.8 단일 Spark SGLang 레시피](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)는 vLLM·llama.cpp·SGLang을 같은 장비에서 비교하고, [단일/듀얼 SGLang+DFlash2 글](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732)은 1대/2대와 tool-eval 결과를 함께 제시한다.

다음 조건을 충족하기 전에는 이 수치를 BF16 baseline과 직접 비교하지 않는다.

- container image/digest와 SGLang commit을 고정
- model revision과 tokenizer revision을 고정
- speculative draft model과 token 수 기록
- `max_model_len`, KV dtype, memory fraction 기록
- native/container 실행 차이 기록
- c1과 c4/c8을 따로 측정
- 출력 token을 SSE chunk가 아니라 usage/tokenizer로 계산

## 4.9 첫날 결과 기록 템플릿

```text
date:
hardware:
os:
kernel:
driver:
cuda:
runtime:
container/image:
model/revision:
quant:
max_model_len:
kv_cache_dtype:
gpu_memory_utilization:
speculative_decoding:
workload:
concurrency:
TTFT:
prefill_tok_s:
decode_tok_s:
aggregate_tok_s:
memory_peak:
temperature_peak:
error_rate:
loads:
generates:
serves:
benchmarked:
tool_tested:
notes:
```

## 이 장의 검증 체크리스트

- [ ] 다른 inference process를 확인했다.
- [ ] 모델 endpoint와 served model name을 확인했다.
- [ ] 기본 생성·JSON·멀티턴·thinking을 통과시켰다.
- [ ] long context는 marker 회수와 memory를 함께 기록했다.
- [ ] tool calling은 parser를 설정한 별도 서버에서 테스트했다.
- [ ] 최적화 recipe로 넘어가기 전 baseline 결과를 보존했다.
- [ ] DeepSeek 단일 노드 recipe에서 weight·image·KV 조건을 확인했다.
- [ ] 370K needle recall을 일반적인 장문 품질 인증으로 과대해석하지 않았다.
