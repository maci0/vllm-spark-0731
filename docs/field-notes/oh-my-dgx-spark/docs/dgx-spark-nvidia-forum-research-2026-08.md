# NVIDIA Developer Forum DGX Spark/GB10 전체 리서치

기준일: 2026-08-21 (KST)

대상: [NVIDIA Developer Forum — DGX Spark / GB10](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)

목적: 포럼에 흩어진 DGX Spark 실사용 경험을 모델, 엔진, 클러스터, 네트워크, 안정성, 에이전트 관점으로 재분류하고 책의 레시피와 재현 실험으로 이어질 자료를 남긴다.

## 1. 조사 범위와 읽는 법

Discourse 카테고리 JSON의 최신순·조회수순 목록과 키워드 검색을 사용했다. 2026-08-21 스냅샷에서 카테고리 목록은 페이지당 30개, 마지막 페이지 25개로 약 2,305개 토픽이었다. 모든 토픽을 같은 깊이로 읽는 대신 전체 목록을 인벤토리화한 뒤 조회수, 답변 수, 최근 활동, 재현 가능성, 책과의 관련성을 기준으로 대표 글을 선별해 본문과 댓글을 검토했다.

검색·분류 키워드:

- 모델: `Qwen`, `Qwen3.5`, `Qwen3.6`, `Qwen3.8`, `DeepSeek`, `Gemma`, `GLM`, `MiniMax`, `Nemotron`, `MiMo`, `LongCat`
- 런타임: `vLLM`, `SGLang`, `llama.cpp`, `SparkInfer`, `TensorRT-LLM`, `Ollama`
- 분산: `NCCL`, `RoCE`, `RDMA`, `ConnectX`, `QSFP`, `TP`, `PP`, `cluster`, `3 node`, `4 node`, `8x`
- 운영: `thermal`, `power`, `shutdown`, `reboot`, `OOM`, `firmware`, `fan`, `memory`
- 에이전트: `Hermes`, `OpenClaw`, `NemoClaw`, `tool calling`, `benchmark`

포럼 수치는 다음 등급으로 기록한다.

| 등급 | 의미 |
|---|---|
| `official` | NVIDIA 직원·공식 FAQ·공식 플레이북 또는 공식 답변 |
| `recipe` | 실행 명령, 이미지/커밋, 하드웨어 조건이 공개된 커뮤니티 레시피 |
| `measurement` | 입력·출력·동시성·엔진·장비가 명시된 실측 |
| `anecdote` | 실제 사용 경험이지만 조건 또는 원시 로그가 부족한 사례 |
| `issue` | 장애 증상과 해결 시도가 기록된 글. 원인이 확정됐다는 뜻은 아님 |
| `opinion` | 모델 선택·구매·운영에 대한 의견 |

`measurement` 등급이라도 NVIDIA 인증 벤치마크를 뜻하지는 않는다. 서로 다른 모델, 양자화, 컨텍스트, 엔진, draft model, 출력 길이의 수치를 하나의 표에 섞지 않는다.

## 2. 먼저 얻은 결론

### 2.1 DGX Spark의 병목은 “GPU가 빠른가” 하나가 아니다

포럼에서 반복해서 나타나는 패턴은 GB10의 128 GiB unified memory, LPDDR5X 메모리 대역폭, KV cache, CUDA workspace, 통신 오버헤드가 하나의 운영 문제로 연결된다는 점이다.

- dense 모델의 단일 스트림 decode는 메모리 대역폭과 per-token 오버헤드에 강하게 제한된다.
- MoE는 active parameter가 작아 같은 총 파라미터 대비 Spark에 더 잘 맞는 경우가 많다.
- NVFP4라는 이름만으로 빠르다고 결론 내리면 안 된다. 초기 vLLM/GB10 경로에서는 AWQ 4-bit가 NVFP4보다 빠른 실측도 있었다.
- MTP·DFlash·DSpark는 모델의 품질을 자동으로 높이는 기능이 아니라, draft acceptance와 검증 커널이 맞을 때 decode를 줄이는 기능이다.
- `1M context 지원`은 KV를 배치할 수 있다는 뜻일 수 있으며, 1M 전체에서 높은 생성 속도나 장기 문맥 품질이 보장된다는 뜻이 아니다.

### 2.2 1대는 “작은 모델만”이 아니라 목적별로 갈린다

1대에서 현실적으로 선택할 수 있는 경로는 27–35B급의 빠른 일상 모델, 최적화된 100B급 MoE, 특정 엔진에 맞춘 DeepSeek 계열이다. 포럼에서 확인되는 단일 Spark 수치는 대략 20–60 tok/s 범위에 넓게 분포하지만, 이를 같은 모델의 공식 순위로 볼 수는 없다.

- Qwen3.8-27B NVFP4: 엔진과 speculative 경로에 따라 약 24–38 tok/s, 특정 SGLang/DFlash 코드 평가에서는 34–38 tok/s가 보고됐다.
- DeepSeek-V4-Flash-0731: 이제 단일 Spark TP=1용 [MiaAI-Lab 레시피](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)가 공개됐다. README 기준 EXL3 3.0 bpw, DSpark K5, NVFP4 KV, `MAX_MODEL_LEN=384000`, 구조화 decode 44–47 tok/s, KV pool 약 439,622 token, 370,104-token needle recall이 보고됐다. 이는 `MAX_NUM_SEQS=1`의 보수적 단일 스트림 profile이며, 공식 full-FP8 경로와 동일하지 않다.
- Qwen3.5-122B-A10B: 한 커뮤니티 레시피는 28.3 → 38.4 tok/s, 후속 업데이트는 51 tok/s를 주장했다. 양자화의 정확성과 vLLM 버전을 함께 고정해야 한다.
- Qwen3-Coder-Next-FP8: 단일 Spark 약 43 tok/s와 FlashInfer 사용 시 더 큰 KV 여유가 보고됐다.

### 2.3 2대의 가치는 “2배 속도”보다 모델·컨텍스트·동시성이다

직접 QSFP/RoCE로 연결한 TP=2는 모델과 runtime이 잘 맞을 때 1.4–1.9배 수준의 결과가 반복해서 보고되지만, 2배 성능이 보장되지는 않는다. 반면 128 GiB를 넘는 모델을 품질 저하가 적은 양자화로 실행하거나 같은 모델을 여러 에이전트가 동시에 사용하는 것이 2대의 더 큰 가치가 될 수 있다.

- Qwen3.8 SGLang+DFlash2: 코드 52–61 → 87 tok/s, prose 26 → 41 tok/s라는 단일 스트림 비교가 있다.
- 다른 Qwen3.8 vLLM+MTP 글에서는 단일 세션 약 22.6 tok/s, 4세션 aggregate 75.0 tok/s, 8세션 aggregate 116.1 tok/s가 보고됐다.
- DeepSeek V4 Flash는 공식 FP8 TP=2, DSpark 256K, 커뮤니티 1M 레시피가 각각 존재한다.
- MiniMax M2.7, Qwen3.5-397B 같은 큰 모델은 2대에 “올라가는가”보다 KV와 동시성까지 남는지가 핵심이다.

### 2.4 3대부터는 모델보다 sharding과 토폴로지가 먼저다

3대 TP는 attention/KV head 수가 노드 수로 나누어지지 않는 모델에서 바로 제약에 부딪힌다. MiMo V2.5와 MiniMax M3 글은 virtual-head padding, MoE zero-fill, MTP 설정 수정 같은 모델별 패치를 사용했다. 이는 일반적인 “TP=3 플래그”만으로 실행하는 레시피가 아니다.

4대는 switch, RoCE, RDMA 설정이 안정화되면 GLM, DeepSeek, Qwen3.5급 모델을 다룰 수 있다. 다만 socket fallback으로 실행하면 속도가 크게 떨어질 수 있다. 8대는 100G breakout 스위치로도 가능하다는 사례가 있지만, 비용, 냉각, 운영 복잡도는 별도의 제품 영역에 가깝다.

### 2.5 하드웨어 안정성은 책에서 별도 장으로 다뤄야 한다

포럼에는 sustained inference 중 hard shutdown이 발생하거나, 95°C 근처에서 thermal shutdown이 일어나거나, unified memory 고갈 후 OS가 멈추거나, ConnectX-7 링크가 저하되거나, 펌웨어 업데이트 후 NIC가 brick된 사례가 반복해서 올라온다. 특정 글의 workaround를 일반 해법으로 그대로 복사하지 않는다.

- 모델이 로드됐다는 사실만으로 안정성을 판정하지 않는다. 장시간 KV 증가·동시성·prefill·tool loop를 포함한 soak test가 필요하다.
- 12–13 Gbps로 보이는 CX-7 직결 링크가 전원 완전 방전 후 약 109–111 Gbps로 회복됐다는 사례가 있다.
- SGLang Docker에서 `/dev/infiniband`를 넘기지 않아 NCCL이 socket으로 떨어진 사례가 있고, RDMA를 켜자 9.8 → 25.1 tok/s로 오른 보고가 있다.
- `mlnx-fw-updater`가 사용자 의도 없이 ConnectX-7 펌웨어를 건드려 복구/RMA로 이어졌다는 사례가 있으므로 firmware 경로는 레시피의 위험 구간으로 표시한다.

## 3. 노드 수별 모델·용도 지도

아래 수치는 포럼 작성자가 제시한 조건을 그대로 옮긴 대표 실측 또는 주장이다. 동일 모델의 절대 성능표로 사용하지 않는다.

| 구성 | 포럼에서 실제 다뤄진 모델·경로 | 관찰된 수치/상태 | 적합한 목적 | 주의점 |
|---|---|---|---|---|
| 1× | Qwen3.8-27B NVFP4 + SGLang/DFlash | 34–38 tok/s 레시피, vLLM은 약 24.5 tok/s 보고 | 코딩, 일반 챗, Claude Code/Hermes | 이미지/컨테이너 transient allocation과 context에 따라 급변 |
| 1× | DeepSeek-V4-Flash-0731 EXL3/SparkInfer | 44–47 tok/s structured, 약 439K KV pool, 370K needle recall | 단일 supervisor, 긴 문맥 | EXL3 3.0 bpw·REAP K216·DSpark K5, c1 최적화 |
| 1× | Qwen3.5-122B-A10B INT4/NVFP4 | 28.3 → 38.4, 후속 51 tok/s 주장 | 큰 단일 supervisor | quant checkpoint별로 garbage output·config mismatch 사례 |
| 1× | Qwen3-Coder-Next-FP8 | 약 43 tok/s, FlashInfer에서 약 170K KV 보고 | 코딩 에이전트 | prefix caching을 끄는 model-card flag를 피해야 함 |
| 2× | Qwen3.8-27B NVFP4 + SGLang/DFlash2 | code 52–61 → 87, prose 26 → 41 | 빠른 supervisor + 여러 세션 | TP 통신으로 TTFT와 thinking workload의 scaling이 제한됨 |
| 2× | DeepSeek-V4-Flash 공식 FP8/DSpark | 200K–262K 레시피, 1M 레시피 존재 | 긴 문맥, agent/tool workload | vLLM commit·NCCL·RoCE·KV 설정을 함께 pin |
| 2× | MiniMax-M2.7 NVFP4 | 약 24.3 tok/s decode, 196K context | 코딩·Hermes | 모델 라이선스가 상업적 사용에 적합한지 별도 확인 |
| 2× | Qwen3.5-397B-A17B | 초기에는 적합한 quant와 KV 여유가 부족하다는 판단 | 대형 모델 탐색 | 2대에 weights만 맞고 KV가 남지 않을 수 있음 |
| 3× | MiMo V2.5 Omni NVFP4 | 1M context, effective 35.1 tok/s, tool eval 97.3 주장 | 멀티모달·긴 문맥 | virtual-head padding 등 모델 전용 patch |
| 3× | MiniMax-M3 NVFP4 | TP=3, single-stream 약 6 tok/s 사례 | 모델 적재·실험 | CX-7 대신 1GbE로 통신한 실험이라 RDMA 수정 필요 |
| 4× | DeepSeek-V4-Flash FP8 | single 49.4, n=8 aggregate 180 tok/s 주장 | 긴 문맥·동시성 | NCCL 2.30.4와 200G RoCE, fork/patch 필요 |
| 4× | GLM-4.7 FP8 + SGLang/EAGLE | 20–27 tok/s, 약 202K context | 대형 MoE agent | GB10 shared-memory tuning과 RDMA가 필수 |
| 4× | Qwen3.5-397B INT4 AutoRound | single 36–37, c4 aggregate 80–94/peak 121 | 큰 daily/supervisor 모델 | Marlin TP=4 patch, 32K KV 조건 |
| 4× | GLM-5.2, DeepSeek 계열 | 256K 또는 1M 레시피 | 멀티모달·장문 agent | 일부는 prune/quant된 비공식 모델 |
| 8× | Nemotron 3 Ultra NVFP4 | 100G와 200G 비교, TP=8 가능 | 연구·대형 모델 | 스위치·냉각·예산·운영 난도가 급격히 증가 |

### 구성 선택을 한 문장으로 줄이면

- 1대: 가장 빠르게 쓰는 모델과 엔진을 고르는 단계.
- 2대: 긴 문맥·큰 supervisor·동시성의 체감이 가장 큰 단계.
- 3대: 모델별 sharding patch를 감수하는 연구 단계.
- 4대: 스위치/RDMA를 제대로 운영해 큰 모델과 aggregate throughput을 얻는 단계.
- 8대: 개인용 레시피를 넘어 클러스터·전력·냉각·장애 대응을 설계하는 단계.

## 4. 모델별 포럼 리서치

### 4.1 Qwen3.8-27B

#### 단일 Spark: 엔진 차이가 모델 차이만큼 크다

[Qwen3.8 단일 Spark one-command 레시피](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)는 같은 GB10에서 llama.cpp 약 27 tok/s, vLLM NVFP4+MTP 약 24.5 tok/s, SGLang+NVFP4+DSpark 약 34–38 tok/s를 비교했다. 작성자는 SGLang을 native로 실행했을 때 unified memory의 transient allocation을 과소계산해 freeze가 발생했고, 이후 memory-capped container로 옮겼다고 기록했다. 이 글의 핵심은 숫자 자체보다 “엔진, 메모리 회계, speculative decode 조건을 함께 고정해야 한다”는 점이다.

[Qwen3.8 FP8 대 NVFP4 동시성 비교](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258)는 vLLM 0.27.1, 16 concurrent, 동일한 입력과 출력 조건에서 NVFP4가 FP8보다 29–34% 높은 aggregate output throughput을 기록했다고 보고했다. 이는 단일 스트림 비교가 아니며 해당 작성자의 조건과 checkpoint에 한정된 결과다.

[단일 Spark vLLM+MTP 측정](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-a-single-dgx-spark-up-to-1m-context-vllm-mtp-measurements/380244)은 Unsloth checkpoint가 한때 tokenizer/checkpoint 문제로 prompt를 2,048 token에서 잘라낼 수 있다는 사실을 발견했다. 이후 fixed upload가 올라왔다는 업데이트가 있으므로 책의 레시피에서는 model revision을 pin하고 긴 prompt가 실제로 전달되는지 확인해야 한다.

#### 2대: TP=2의 실제 scaling

[Qwen3.8-27B 단일/듀얼 SGLang+DFlash2 레시피](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732)는 2026-08-19 실측을 다음처럼 제시한다.

| workload | 1× Spark | 2× Spark TP=2 |
|---|---:|---:|
| code generation | 52–61 tok/s | 87 tok/s |
| prose | 26 tok/s | 41 tok/s |
| thinking chat | 34–49 tok/s | 49 tok/s |
| code, thinking off | 52 tok/s | 약 80 tok/s |
| 짧은 prompt TTFT | 약 0.16s | 비슷함 |
| 반복 16K prefix TTFT | 0.44s | 0.74s |
| context | 262,144 | 262,144 |

동일 글은 HumanEval 159/164, math 10/10, tool-eval-bench 92.3 ± 0.6이라는 품질 측정도 제시한다. 모두 작성자가 고정한 SGLang image, DFlash2 draft, 2× GB10, RoCE 환경에서 얻은 커뮤니티 결과이며 공식 certification은 아니다. YaRN 262K 초과, `--mem-fraction-static 0.95`, 잘못된 MTU/HCA 이름으로 실패한 사례까지 함께 기록했다는 점이 재현성 측면에서 중요하다.

[Qwen3.8 듀얼 vLLM+MTP 실측](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350)은 같은 `SPEC_TOKENS=5`에서 단일 약 24 → 듀얼 약 37 tok/s를 보고했고, 동시성 안정화를 위해 `SPEC_TOKENS=2`로 낮춘 뒤 1세션 22.6, 4세션 aggregate 75.0, 8세션 aggregate 116.1 tok/s를 기록했다. “단일 요청 latency”와 “여러 에이전트 aggregate throughput”이 서로 다른 최적점을 가진다는 좋은 사례다.

#### 단일/듀얼 Qwen3.8을 책에 반영하는 방법

- `vLLM + MTP`, `SGLang + DFlash2`, `llama.cpp + MTP/DSpark`를 같은 prompt set으로 재측정한다.
- code/prose/thinking을 나누고, single-stream과 c4/c8 aggregate를 분리한다.
- NVFP4와 FP8은 품질·KV·prefill·decode를 함께 기록한다.
- “1M”은 capacity, 실제 retrieval 정확도, TTFT, sustained decode를 별도 열로 둔다.

### 4.2 DeepSeek-V4-Flash-0731

[단일 Spark ds4 튜닝 글](https://forums.developer.nvidia.com/t/1x-spark-tuned-dspark-for-deepseek-v4-flash-35-tok-s-800-prefill-and-fast-multi-agent-serving/376884)은 custom CUDA serving fork에서 9개 workload의 plain 평균을 20.1에서 DSpark 27.7 tok/s로, structured workload의 최대값을 34.5 tok/s로 보고했다. draft가 맞지 않는 creative prose에서는 speculation을 자동으로 끄는 guard를 사용한다고 설명한다. “lossless”라는 표현은 해당 구현의 설계와 검증에 관한 주장으로 기록하고, 품질 재현은 별도로 진행해야 한다.

[현재 단일 Spark ds4-on-spark 소개](https://forums.developer.nvidia.com/t/1x-spark-deepseek-v4-flash-0731-1-000-tok-s-prefill-59-tok-s-multi-agent-serving/378855)는 DeepSeek-V4-Flash-0731, continuous batching, prefix cache, disk KV persistence, OpenAI-compatible serving을 한 노드에서 제공하는 방향을 설명한다. bare headless server에서 1.2M context를 시험했다는 댓글도 있으나, 이는 일반 GUI/agent 동시 실행 조건의 보장이 아니다.

[단일 Spark EXL3/SparkInfer 레시피](https://forums.developer.nvidia.com/t/c1-1058pp-s-52-tg-s-on-1x-dgx-spark-on-deepseek-v4-flash-0731-full-256-experts/379863)는 256 routed experts를 유지한 약 2 bpw checkpoint에서 code 시험 c1 52.47 tok/s, matched 2,048 prompt/128 output에서는 약 24.7 tok/s, c4 aggregate 약 31.7 tok/s를 구분한다. speculative acceptance가 workload에 따라 63.81%에서 28.95%까지 달라졌다는 기록은 숫자를 그대로 일반화하면 안 되는 이유다.

[최신 단일 Spark DeepSeek V4 Flash 0731 레시피](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)는 기존의 “2대 전용”이라는 인식을 바꾸는 중요한 업데이트다. 공개 README 기준으로 다음 항목을 기록한다.

- 1× DGX Spark, TP=1, GB10/SM121, 128 GiB unified memory
- `0xSero/deepseek-v4-flash-0731-spark` EXL3 3.0 bpw, REAP-K216 checkpoint, SparkInfer/b12x 계열
- DSpark K5 + K64 draft, `nvfp4_ds_mla` compressed KV
- 기본 `MAX_NUM_SEQS=1`, `MAX_MODEL_LEN=384000`, `GPU_MEMORY_UTILIZATION=0.94`
- 구조화 decode 44–47 tok/s
- cold boot에서 약 439,622-token KV pool 관찰
- 320,037 및 370,104 token prompt에서 exact needle recall, preemption 0
- prefill은 요청 초반 약 1,024 tok/s에서 300K 이후 약 350–614 tok/s로 감소하며, 370K 테스트의 effective prefill은 약 625 tok/s

이 결과는 “384K context와 370K recall을 한 Spark에서 실제로 시험했다”는 강한 community recipe 근거다. 다만 EXL3는 full FP8이 아니며, REAP가 256개 expert 중 216개를 유지하는 파생 checkpoint이고, 1-sequence 깊은 문맥에 맞춘 구성이다. README에는 `MAX_NUM_SEQS=4`로 변경하면 KV pool이 줄어드는 사례도 있으므로 c1 결과를 다중 에이전트 성능으로 기록하지 않는다. 첫 부팅에는 약 107GB weight 다운로드, coalesce, CUDA graph capture가 필요하며 장비의 free RAM, EarlyOOM 설정, 냉각 상태를 확인해야 한다.

[듀얼 Spark 1M recipe](https://forums.developer.nvidia.com/t/deepseek-v4-flash-aiden-recipe-from-reddit-1m-token-session-operational-cuda-12-1-tailored-for-dgx-spark-gb10/372268)는 2× GB10, TP=2, 1M context, 30–45 tok/s decode, tool-calling 89/100을 보고한다. 같은 포럼에는 [공식 FP8 2× recipe](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309)와 [DSpark 2× instructions](https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220)도 있어, “공식 FP8”, “custom quant”, “DSpark”를 비교할 수 있다.

[4대 DeepSeek TP=4 recipe](https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-4x-dgx-spark-via-vllm-jasl-fork-tp-4-rdma-mtp-49-54-tok-s-single-stream-full-recipe-the-traps/373808)는 4× GB10에서 single 49.4, reasoning probe peak 54.4, c8 aggregate 180 tok/s를 보고한다. 핵심 재현 조건은 vLLM fork, `libnccl2=2.30.4`, 200G RoCE, FP8 KV, 384K context, sm_121 fallback patch다. 특히 NCCL 2.28.9에서 long generation이 멈추고 2.30.4에서 해결됐다는 보고는 버전 pin의 근거다.

#### DS4 brain, Qwen3.8 UI worker, 그리고 DS4 vision shim

최근의 역할 분리 제안은 DeepSeek V4 Flash 0731을 supervisor 또는 agent brain으로 유지하고 Qwen3.8-27B를 UI·디자인 작업에 넘기는 방식이다. 두 모델을 각각 2×Spark에서 실행한다면 네 대가 필요하며, `DeepSeek TP=2`와 `Qwen3.8 TP=2`라는 독립 endpoint 두 개로 이해해야 한다. 이는 단일 TP=4 모델의 벤치마크가 아니다. [Qwen3.8 듀얼 vLLM+MTP](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350), [Qwen3.8 SGLang+DFlash2](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732)

[DeepSeek V4 Flash vision shim 저장소](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)는 기존 2× Spark DS4 서버 앞에 `:8899` OpenAI-compatible shim과 `:8081`의 소형 VLM을 추가한다. 이미지 요청은 caption으로 변환한 뒤 `:8888` DS4로 전달하고, 텍스트 요청은 그대로 통과시킨다. 저장소 README의 기본 “eyes”는 `Qwen3.5-0.8B-MLX-8bit`이므로 이 자료를 Qwen3.8-27B native vision의 근거로 사용하지 않는다. 또한 caption pipeline이므로 OCR과 정밀한 공간 추론에서는 정보 손실이 생길 수 있다.

이 조합은 모델을 교체하지 않고 OpenAI-compatible base URL을 라우팅하는 운영 패턴으로는 유용하다. 다만 현재 포럼 자료에는 DS4 2×와 Qwen3.8 2×를 동시에 운용한 동일 하니스의 end-to-end 수치가 없다. 따라서 책에서는 역할과 endpoint 구조, 각 모델의 독립 벤치마크를 분리해 기록한다.

### 4.3 Qwen3.5/Qwen3.6

[Qwen3.5-122B 단일 Spark 최적화](https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-on-single-spark-up-to-51-tok-s-v2-1-patches-quick-start-benchmark/365639)는 baseline 28.3에서 hybrid INT4+FP8 shared dense layer, MTP-1 등을 적용해 38.4 tok/s를 제시하고, 후속 업데이트에서 51 tok/s를 제목에 반영했다. MTP acceptance와 vLLM patch가 결과의 일부이므로 양자화 파일만 바꿔 같은 숫자를 기대하면 안 된다.

[Qwen3.5-122B NVFP4 quantization 글](https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-nvfp4-quantized-for-dgx-spark-234gb-75gb-runs-on-128gb/361819)은 234GB BF16을 75.6GB로 줄여 단일 Spark에 맞추려 했지만, 초기 checkpoint가 반복 `!`, config type mismatch, vision 제거 문제를 일으켰고 작성자가 calibration 문제를 인정했다. 이 글의 최종 가치는 특정 quant를 추천하는 데 있지 않고, “로드 성공”과 “정상 생성·비전·품질”을 분리해야 한다는 데 있다.

[Qwen3.5-397B 4대 실측](https://forums.developer.nvidia.com/t/qwen3-5-397b-a17b-int4-autoround-4-x-db10-node-updated-results-37-94-tok-s/362368)은 4× Ascent, TP=4, 32K context, FP8 KV에서 single 36–37 tok/s, c4 aggregate 80–94 tok/s, peak 121 tok/s를 제시한다. Marlin TP=4에서 linear-attention block을 ReplicatedLinear로 처리하는 patch가 필요했다.

[Qwen3.6-35B-A3B NVFP4 플랫폼 비교](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810)는 같은 vLLM 설정을 Jetson Thor, DGX Spark, Blackwell 6000 Pro에 적용했다. 작성자 실측에서 DGX Spark의 output throughput은 prompt-heavy 171.64, decode-heavy 268.21, balanced 249.47 tok/s였으며, 동시성·8K 입력을 포함한 서버 aggregate 값이다. single-stream decode와 혼동하지 않는다.

[Qwen3.6-35B-A3B와 FP8 발표 스레드](https://forums.developer.nvidia.com/t/qwen-qwen3-6-35b-a3b-and-fp8-has-landed/366822)는 agentic coding, reasoning context 보존, vLLM/FlashInfer/Autoround 지원을 논의한다. 모델 발표 글은 품질 기대를 보여주지만, 실제 Spark 추천은 371810 및 별도 recipe의 재현 결과와 함께 판단한다.

### 4.4 MiniMax, GLM, MiMo, Nemotron, LongCat, Gemma

- [MiniMax-M2.7 NVFP4 듀얼 노드](https://forums.developer.nvidia.com/t/minimax-m2-7-nfvp4-recipe-benchmarks/366324): 2× ASUS Ascent GX10, 약 196K context, pp2048 약 2,074 tok/s, tg128 약 24.3 tok/s. 포럼 댓글은 모델 라이선스가 비상업적 조건인지 확인해야 한다고 지적한다.
- [GLM-4.7 FP8 4대 SGLang](https://forums.developer.nvidia.com/t/running-glm-4-7-fp8-355b-moe-on-4x-dgx-spark-with-sglang-eagle-speculative-decoding/359256): shared-memory kernel tuning 후 20–27 tok/s, 약 202,752 context. GB10용 `lmsysorg/sglang:spark` 이미지와 EAGLE을 사용했다.
- [GLM-4.7 RDMA 수정](https://forums.developer.nvidia.com/t/glm-4-7-fp8-on-4x-dgx-spark-via-sglang-2-5x-speedup-8-2-25-tok-s-just-by-enabling-rdma/373675): SGLang Docker가 `/dev/infiniband`를 받지 않아 socket으로 동작하던 상태에서 RDMA를 켜고 9.8 → 25.1 tok/s로 개선했다. `NCCL_DEBUG=INFO`의 `via NET/IB`를 성공 조건으로 삼는다.
- [GLM-5.2 4대 1M vision recipe](https://forums.developer.nvidia.com/t/recipe-glm-5-2-on-4x-dgx-spark-at-1m-context-with-sparkrun-vision/380448): Hermes와 함께 사용할 수 있는 `sparkrun` 레시피, text quant와 vision tower, Adaptive MTP를 묶었다. 게시 시점에는 재현 benchmark보다 recipe 공유 성격이 강하다.
- [MiMo V2.5 Omni 3대](https://forums.developer.nvidia.com/t/mimo-v2-5-omni-on-3x-dgx-spark-tp-3-mtp-1m-context-39-tok-s/373948): 3× RoCE, 1M context, 4 modalities, effective 35.1 tok/s, tool-call 평가 97.3을 보고한다. virtual-head padding과 MTP draft 수정이 포함된 실험이다.
- [Nemotron 3 Super 120B 4대 MTP](https://forums.developer.nvidia.com/t/nemotron-3-super-120b-a12b-nvfp4-mtp-on-4x-dgx-spark-via-sglang-tp-4-roce-mtp-actually-pays-off-1-70x-single-stream-accept-len-2-7/373625): SGLang dev build에서 accept_len 약 2.7, single-stream 1.70×를 보고한다. 1대에 weights는 올라가도 Mamba state pool과 CUDA graph까지 포함하면 concurrency가 부족할 수 있다고 설명한다.
- [LongCat-Next 단일 Spark](https://forums.developer.nvidia.com/t/longcat-next/372494): 텍스트·이미지·오디오·비디오·음성 생성이 한 Spark에 올라간다는 multimodal 사례다. 다른 사용자는 vision encoder의 allocator fragmentation 때문에 장시간 입력에서 freeze가 났고 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 완화했다고 보고했다.
- [Gemma 4 vLLM 지원 논의](https://forums.developer.nvidia.com/t/gemma-4-models-which-vllm-version-any-prs-spotted/365490): Transformers 버전만 올리는 것으로 끝나지 않고 vLLM model implementation/fallback가 필요했던 초기 상태를 보여준다. 최신 버전에서 다시 확인해야 한다.

## 5. 런타임별 특징과 선택 기준

| 런타임/경로 | 포럼에서 강하게 보이는 용도 | 장점 | 대표 위험 |
|---|---|---|---|
| vLLM | 공식 playbook, OpenAI endpoint, TP=2/4, 긴 문맥 | 생태계·API·서버 기능이 넓음 | GB10 fork/nightly·Ray·NCCL·모델 PR pin 필요 |
| SGLang | Qwen3.8 DFlash2, GLM, 동시성·agent workload | prefix/radix cache와 speculative 경로가 강함 | Docker RDMA passthrough, sm_121 kernel tuning, memory cap |
| llama.cpp | 단일 스트림·GGUF·간단한 실험 | per-step overhead가 작고 설치가 단순함 | multi-node 분할과 큰 모델의 성능·KV가 제한될 수 있음 |
| SparkInfer/EXL3 | DeepSeek 단일 Spark | 낮은 bit 수와 전용 커널로 높은 단일-node 속도 | 공격적 quant 품질 검증과 parser 확인 필요 |
| ds4-on-spark | DeepSeek-V4-Flash 단일 serving | CUDA 전용 최적화, batching, KV persistence, OpenAI API | upstream과 다른 fork; 버전·모델 revision을 pin해야 함 |
| TensorRT-LLM | NVIDIA 계열 최적화·비교 대상 | production kernel/graph 경로 | dual Spark all-reduce deadlock 이슈가 보고됨 |
| Ollama/LM Studio | 로컬 관리·빠른 모델 교체 | UX와 관리가 쉬움 | multi-node TP와 고동시성 agent에는 주 경로로 덜 적합하다는 의견 |

포럼에서 같은 모델의 결과가 크게 갈리는 가장 흔한 이유는 runtime의 차이를 모델 자체의 속성으로 착각하기 때문이다. 책에서는 모델 표에 반드시 `engine`, `image/commit`, `attention backend`, `quant`, `KV dtype`, `spec decode`, `context`, `concurrency`를 함께 적는다.

## 6. 클러스터·네트워크 리서치

### 6.1 2대 직결

[NVIDIA 공식 FAQ](https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344)와 [성능 FAQ](https://forums.developer.nvidia.com/t/dgx-spark-performance-faq/359456)는 초기 설정과 공식 benchmark guide로 이어지는 기준점이다. community recipe에서는 다음 항목을 반복해서 확인한다.

1. 두 노드의 ConnectX-7 포트와 케이블을 확인한다.
2. `enp1...` 계열의 실제 Up 인터페이스를 골라 정적 IP를 둔다. 이름이 비슷한 `enP2...`를 섞지 않는다.
3. `ibdev2netdev`, `ibv_devices`, `ethtool`, `iperf3`, `ib_write_bw`로 관리망이 아닌 CX-7 경로를 확인한다.
4. NCCL 로그에서 `NET/IB`가 사용되는지 확인한다.
5. 링크가 200G로 negotiate돼도 payload가 12–13 Gbps일 수 있다. 포럼의 여러 사례에서는 두 장비의 완전 전원 차단·전원 케이블 분리 후 약 109–111 Gbps가 나왔다.

[직결 링크가 13 Gbps로 제한된 사례](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461)는 PCIe power/SlotPowerLimit 의심과 업데이트 후 회복 사례를 함께 담고 있다. [200G negotiate 후 12 Gbps가 된 사례](https://forums.developer.nvidia.com/t/dgx-spark-200gbe-direct-qsfp-link-negotiates-200g-but-payload-is-12-gbps/373538)는 전원을 완전히 방전한 뒤 회복되었다고 기록한다. 이는 공식적인 보편 해법이 아니라 현장 workaround다.

### 6.2 3대와 4대

[4대 switchless 연결 아이디어](https://forums.developer.nvidia.com/t/4-node-dgx-spark-cluster-without-a-switch/368726)는 ConnectX 포트와 breakout/optics를 조합해 full mesh를 만들려는 실험이다. 설계 제안과 실제 검증을 구분한다.

- 3대 direct ring은 가능성을 보이지만, TP=3은 head/KV 분할 규칙을 만족해야 한다.
- 4대 TP는 일반적으로 switch fabric과 dedicated RoCE가 더 현실적이다.
- [GLM-4.7 글](https://forums.developer.nvidia.com/t/glm-4-7-fp8-on-4x-dgx-spark-via-sglang-2-5x-speedup-8-2-25-tok-s-just-by-enabling-rdma/373675)처럼 socket fallback은 모델에 따라 치명적일 수 있다.
- [SGLang multi-node traps](https://forums.developer.nvidia.com/t/sglang-multi-node-on-dgx-spark-three-traps-that-wasted-a-day-each-and-how-to-spot-them/373677)는 `TORCH_DISTRIBUTED_DEBUG=DETAIL`이 SGLang sidecar의 rank-local broadcast를 global collective mismatch처럼 보이게 할 수 있다고 설명한다. 실제 hang 진단과 production 설정을 분리한다.

### 6.3 NCCL deadlock과 버전 문제

[dual Spark NCCL all-reduce deadlock](https://forums.developer.nvidia.com/t/nccl-all-reduce-deadlock-on-dual-dgx-spark-after-successful-channel-establishment-affects-both-vllm-and-trt-llm/366127)은 vLLM과 TensorRT-LLM 모두 첫 all-reduce에서 멈추는 사례다. 이 글은 link up·channels established·weights loaded가 모두 성공해도 inference-ready가 아닐 수 있음을 보여준다.

[NCCL socket transport/parallelism 문제](https://forums.developer.nvidia.com/t/nccl-socket-transport-fails-with-pipeline-parallelism-mesh-pp-on-dgx-spark/356280), [NCCL bandwidth가 낮은 사례](https://forums.developer.nvidia.com/t/nccl-test-bandwidth-is-only-3gb-s-between-2-dgx-spark-using-qsfp-cable/366373), [NCCL 단일 케이블 100Gbps 제한 사례](https://forums.developer.nvidia.com/t/nccl-single-cable-test-caps-at-100gbps/362403)도 함께 읽는다. 해결책은 모델 레시피를 바꾸기 전에 topology·HCA·MTU·NCCL version·device passthrough를 고정하는 것이다.

### 6.4 Mac 혼합 구성과 MCDMA

표준 경로에서 Mac은 관리 host, router, RAG, 별도 MLX endpoint로 사용하고 Spark의 CUDA TP rank에는 넣지 않는다. Apple MLX의 Thunderbolt 5 RDMA/JACCL과 Spark의 CX-7 RoCE/NCCL은 transport와 backend가 다르다.

최근 Ash Hart의 [MCDMA 게시물](https://x.com/ashxhart/status/2089749434087227672?s=20)은 USB-C로 Mac의 Metal memory와 Spark CUDA memory를 직접 교환하는 커뮤니티 프로토타입을 제시한다. 작성자는 단일 링크 939MB/s, 두 Spark 동시 Mac→Spark 1.80GB/s, Spark→Mac 1.25GB/s, 왕복 24μs를 보고했다. Spark 1과 Spark 2를 CX-7으로 연결해 prompt processing을 맡기고, 각 Spark와 Mac Studio 사이의 두 USB-C 링크로 decode를 처리하는 구조다.

이 자료는 흥미로운 실험 근거지만 조사 시점에는 공개 source repository와 독립적으로 재현한 tokens/s 결과를 확인하지 못했다. 따라서 MCDMA를 NVIDIA 공식 RDMA, GPUDirect RDMA, NCCL 또는 MLX/JACCL의 지원 경로로 기록하지 않는다. 939MB/s는 CX-7 200GbE보다 낮으므로 메모리 pool의 크기보다 실제 activation과 KV 이동량, pipeline 동기화 비용을 먼저 측정해야 한다. 관련 배경과 검증 계획은 [Mac·RDMA·switch 리서치](dgx-spark-mac-rdma-switch-research-2026-08.md)에 별도로 기록한다.

## 7. 발열·전원·메모리·펌웨어

### 7.1 unified memory와 OOM

[메모리 full에서 시스템 전체가 crash한 글](https://forums.developer.nvidia.com/t/system-crashes-when-memory-is-full/352339)은 vLLM/GRPO 작업 중 process가 종료되지 않고 SSH·HDMI까지 사라지는 증상을 기록한다. 댓글의 Docker `--memory` 제한은 한 사용자의 임시 완화책이다.

[stability / OOM / overheating 종합 글](https://forums.developer.nvidia.com/t/dgx-spark-stability-out-of-ram-overheating/368536)은 모델 weights, KV, CUDA workspace가 OS와 같은 128 GiB pool을 함께 사용한다는 운영 문제를 보여준다. `gpu_memory_utilization`을 낮추는 것이 일반적인 방향이지만 정확한 값은 모델, context, batch에 따라 계산해야 한다.

실전 원칙:

- 큰 모델을 두 개 동시에 preload하지 않는다.
- weights가 올라온 직후가 아니라 긴 prompt·반복 prefix·동시 요청 이후의 free memory를 기록한다.
- `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, KV dtype를 함께 조정한다.
- 메모리가 꽉 찬 상태를 정상 운영 상태로 취급하지 않는다. Ray object store와 memory monitor가 false positive 또는 실제 OOM을 만들 수 있다.

### 7.2 thermal shutdown과 냉각

[온도 때문에 자동 shutdown한 글](https://forums.developer.nvidia.com/t/dgxspark-temperature-too-high-automatic-shutdown/363370)은 약 10분 동안 ComfyUI 영상 생성을 실행한 뒤 종료된 사례를 기록하며, NVIDIA 답변은 특정 장비에서 재현되면 RMA를 권고하는 방향이었다. [최근 abrupt shutdown 글](https://forums.developer.nvidia.com/t/spark-abruptly-shuts-down/377478)에는 sustained inference, Qwen3.5-122B, Gemma 4 등에서 경고 없이 종료된 사용자 경험과 여러 workaround가 섞여 있다. 후자의 댓글은 원인과 해결을 확정한 NVIDIA 진단이 아니므로 그대로 따라 하지 않는다.

[dual Spark ducted cooling cage](https://forums.developer.nvidia.com/t/dual-spark-ducted-cooling-cage/365302)는 Noctua 120mm 팬과 3D-printed duct를 사용해 idle GPU 온도를 40°C대까지 낮췄다는 DIY 사례다. 유용한 운영 아이디어이지만 warranty, 소음, 실내 온도, load 온도까지 재현한 공식 설계는 아니다.

책의 thermal test에는 실내 온도, idle/peak GPU·CPU/ACPI, fan mode, input/output workload, power draw, 종료 여부를 함께 기록한다. 온도 하나만으로 안정성을 설명하지 않는다.

### 7.3 저클럭·저전력 상태

[Blackwellboy의 X 보고](https://x.com/Blackwellboy/status/2090611479653622261?s=20)는 GPU utilization 약 96%, P0, throttle reason 없음만 보면 정상처럼 보이지만 SM clock 약 799MHz, power 약 19.5W인 상태에서 Ornith 1.5 35B decode가 약 44 tok/s에 머문 사례를 기록한다. 전원을 완전히 차단한 뒤에는 약 2.3~2.5GHz, 92W, 73.9 tok/s로 회복되었다.

[NVIDIA 포럼의 721MHz 사례](https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-clock-pinned-at-721-mhz-under-full-load-no-throttling-not-liftable-via-nvidia-smi/376039)와 [전원 완전 차단 workaround](https://forums.developer.nvidia.com/t/gpu-clock-bug-looks-like-5-min-wait-is-enough/376239)도 비슷한 증상을 설명한다. 이 사례들은 전원 공급 또는 USB PD 상태를 원인으로 추정하지만, 책에서는 확정된 root cause가 아니라 community workaround로 기록한다.

부하 중에는 utilization과 P-state뿐 아니라 `clocks.sm`, `power.draw`, BF16 compute 또는 실제 model decode를 함께 기록한다. 서버를 정상 종료한 뒤 power brick과 AC를 분리하고 몇 분 기다렸다가 다시 부팅하는 방법이 여러 사례에서 사용되었지만, 같은 현상이 반복되면 driver, EC, USB PD 버전과 직전 OOM·hard crash 정보를 보존하고 공식 지원 절차를 따른다.

### 7.3a 의도적인 clock cap과 전력·발열 trade-off

장애 복구와 별개로, [GB10 clock cap harness](https://github.com/agjs/gb10-clock-cap)는 SM clock을 의도적으로 낮춰 전력·온도와 throughput의 교환을 측정한다. 작성자의 2× GB10 reference에서는 `2200MHz` cap으로 peak 90→78°C, 노드당 GPU rail 63.1→40.1W, decode 73.3→72.5 tok/s를 보고했고, 2000·1800MHz sweep에서는 cold prefill 손실이 각각 +8.1%·+13.0%였다. 이 자료는 decode가 memory-bound이고 prefill이 compute-bound일 때 두 지표가 다르게 움직일 수 있다는 실험 근거다.

별도의 [2× Spark c4 X 보고](https://x.com/ivanfioravanti/status/2088730630875930639?s=20)는 `2455/2300/2200/2000/1800MHz`에서 `nvtop` 전력 약 `47/34/32/27/23`과 온도 약 `72/67/66/63/61°C`를 제시한다. 게시자는 벽면 전력이 더 높다고 명시했으므로 `nvtop`/GPU rail과 AC wall-meter를 분리해 기록한다. `sudo nvidia-smi -lgc 0,2200`과 `sudo nvidia-smi -rgc`는 보고된 실험 명령이며, 모든 모델·동시성의 보편 최적값이나 공식 thermal workaround로 쓰지 않는다.

### 7.4 ConnectX-7 펌웨어

[ConnectX-7이 자동 펌웨어 업데이트 후 brick된 사례](https://forums.developer.nvidia.com/t/connectx-7-bricked-stuck-in-pre-init-static-config-not-done-after-unsolicited-mlnx-fw-updater-firmware-flash-asus-gx10-error-110/373900)는 ASUS GX10에서 `mlnx-fw-updater`가 apt/dpkg 작업 중 동작했고 두 CX-7 인터페이스가 unusable이 됐다고 보고한다. 포럼 작성자는 firmware pinning과 자동 updater 비활성화를 권고했지만, 이것은 해당 시스템의 보호 조치이지 모든 장비에서 실행할 명령으로 복사할 결론은 아니다. 이 글은 “업데이트 전 firmware backup·복구 경로·RMA 연락처를 준비하라”는 책의 운영 체크리스트 근거로 사용한다.

## 8. Hermes/OpenClaw/NemoClaw와 에이전트 운용

포럼은 raw chat보다 agent loop를 중요한 사용처로 본다. [2대 Spark에서 Hermes/OpenClaw 모델을 묻는 글](https://forums.developer.nvidia.com/t/now-running-2x-dgx-spark-stacked-over-qsfp56-looking-for-model-recs-for-agentic-workloads-hermes-openclaw/368649)은 약 24개 agent, supervisor/worker, vision, long-context, tool use를 하나의 환경에서 운영하는 방안을 검토한다. 댓글에서는 MiniMax 2.7과 Qwen3.5-397B를 큰 모델 후보로, SGLang을 smaller concurrent model 경로로 언급한다. 모두 사용자 의견으로 분류한다.

[2-node agent 모델 추천 토론](https://forums.developer.nvidia.com/t/best-2026-model-for-agentic-work-on-a-2-node-spark-cluster/369799)은 coding/email/writing/감독자 역할에 따라 모델이 달라진다고 정리한다. “최고 모델 하나” 대신 다음 구조가 현실적이다.

- worker agent: thinking off, 작은 Qwen/Nemotron/Coder 계열, 낮은 latency와 안정적인 tool parser
- supervisor: 2대 이상에서 DeepSeek/MiniMax/Qwen3.5 등 큰 모델, thinking on을 선택적으로 사용
- vision/long-context agent: 별도 endpoint 또는 3–4대 모델, KV 압박과 이미지 encoder workspace를 분리
- 안전 경계: OpenClaw/NemoClaw가 파일·shell·네트워크 권한을 어디까지 갖는지 모델 성능과 분리해 검증

NVIDIA의 공식 방향은 [Hermes Agent local model playbook](https://forums.developer.nvidia.com/t/new-playbook-run-hermes-agent-with-local-models/369747)과 [NemoClaw/OpenClaw local execution 안내](https://forums.developer.nvidia.com/t/build-a-secure-always-on-local-ai-agent-with-openclaw-and-nvidia-nemoclaw/366929)에서 확인한다. 공식 글은 시작점일 뿐이며 실제 모델의 tool parser, 권한, 실패 복구는 우리 장비에서 별도로 테스트해야 한다.

## 9. 벤치마크 체계

### 9.1 포럼이 이미 만든 도구

- [Spark Arena](https://forums.developer.nvidia.com/t/introducing-the-spark-arena/360319): CLI, runtime flag, quant, topology, memory 조건을 함께 저장하려는 커뮤니티 leaderboard.
- [Tool Eval Bench](https://forums.developer.nvidia.com/t/introducing-tool-eval-bench-cli/366903): OpenAI-compatible endpoint에 대해 63개 시나리오, 14개 category, mock tool handler, seed/trials를 제공한다.
- [Toolery](https://forums.developer.nvidia.com/t/toolery-0-1-0-a-deterministic-tool-calling-benchmark-for-local-llms/371794): 143개 deterministic tool-calling scenario, four difficulty tiers, model-as-judge 없는 assertion 기반 평가를 지향한다.
- [NVIDIA Performance FAQ](https://forums.developer.nvidia.com/t/dgx-spark-performance-faq/359456): 공식 성능 블로그와 [benchmarking guide](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md)로 연결되는 기준점이다.

### 9.2 책에서 고정할 benchmark schema

| 그룹 | 반드시 기록할 값 |
|---|---|
| hardware | Spark/GX10/PGX, GB10, unified memory, node 수, switch/cable, NIC |
| software | DGX OS, kernel, driver, CUDA, NCCL, container digest, runtime commit |
| model | 원본 repo/revision, total/active parameter, quant, model size, tokenizer revision |
| serving | TP/PP/EP/DP, attention backend, MoE backend, KV dtype, context, memory fraction |
| speculation | MTP/DFlash/DSpark/EAGLE, draft model, tokens, acceptance rate |
| workload | prompt tokens, output tokens, thinking on/off, tool loop, vision/audio/video 여부 |
| load | concurrency, request rate, prefix cache hit/miss, batch token limits |
| result | TTFT, prefill tok/s, decode tok/s, ITL/TPOT, aggregate tok/s, p50/p95, error rate |
| quality | exact match/code tests, tool-call validity, arguments, recovery, safety, long-context retrieval |
| operations | load time, first-token time, idle/peak power, temperature, memory peak, crash/restart |

특히 다음 상태를 서로 분리해 기록한다.

1. `loads`: weight가 메모리에 올라옴.
2. `generates`: 기본 prompt에 정상 답변.
3. `serves`: endpoint가 여러 요청에 지속 응답.
4. `benchmarked`: 조건과 반복 수가 고정된 속도 측정.
5. `tool-tested`: parser와 arguments가 실제로 맞음.
6. `agent-tested`: 다단계 tool loop, 오류 복구, context pressure까지 통과.
7. `long-context-tested`: 지정 context에서 성능뿐 아니라 retrieval/품질과 안정성을 확인.

### 9.3 현재 포럼 숫자를 읽는 요령

- `prefill tok/s`와 `decode tok/s`를 섞지 않는다.
- `c8 aggregate 180 tok/s`를 single user 180 tok/s로 쓰지 않는다.
- `peak`는 평균이 아니다.
- SSE chunk 수를 token 수로 세지 말고 OpenAI `usage` 또는 tokenizer 기준으로 센다.
- speculative decode의 높은 속도는 acceptance rate와 workload를 같이 보여준다.
- tool benchmark 점수와 raw generation 속도는 서로 다른 축이다.

## 10. 대표 토픽 인덱스

### 공식·기준 문서

- [DGX Spark / GB10 FAQ](https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344) — 초기 설정, appliance mode, 네트워크 질문
- [DGX Spark Performance FAQ](https://forums.developer.nvidia.com/t/dgx-spark-performance-faq/359456) — 공식 benchmark blog/guide 링크
- [DGX Spark release updates](https://forums.developer.nvidia.com/t/dgx-spark-release-updates/341703) — 초기 release/update 토론
- [Run Hermes Agent with Local Models](https://forums.developer.nvidia.com/t/new-playbook-run-hermes-agent-with-local-models/369747) — NVIDIA 공식 playbook 안내
- [NemoClaw/OpenClaw secure local agent](https://forums.developer.nvidia.com/t/build-a-secure-always-on-local-ai-agent-with-openclaw-and-nvidia-nemoclaw/366929) — 로컬 실행·보안 포지셔닝

### Qwen·단일/듀얼

- [Qwen3.8-27B dual Sparks](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350) — vLLM+MTP, dual concurrency
- [Qwen3.8 single/dual SGLang+DFlash2](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732) — 재현 recipe, TP=2, tool eval
- [Qwen3.8 single vLLM+MTP](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-a-single-dgx-spark-up-to-1m-context-vllm-mtp-measurements/380244) — tokenizer/long-context 주의
- [Qwen3.8 one-command SGLang](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257) — 엔진 비교와 container memory cap
- [Qwen3.8 FP8 vs NVFP4](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258) — 동시성 조건 quant 비교
- [Qwen3.8 MixedInt4 AutoRound](https://forums.developer.nvidia.com/t/qwen3-8-27b-mixedint4-autoround-optimized-for-a-single-dgx-spark/380248) — 단일 Spark용 mixed quant
- [Qwen3.5-122B single Spark](https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-on-single-spark-up-to-51-tok-s-v2-1-patches-quick-start-benchmark/365639) — MTP/patch optimization
- [Qwen3.5-122B NVFP4 quant](https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-nvfp4-quantized-for-dgx-spark-234gb-75gb-runs-on-128gb/361819) — quant 실패와 correction
- [Qwen3.5-397B duo discussion](https://forums.developer.nvidia.com/t/qwen3-5-397b-a17b-dgx-spark-duo/360780) — 2대 memory feasibility
- [Qwen3.5-397B four-node results](https://forums.developer.nvidia.com/t/qwen3-5-397b-a17b-int4-autoround-4-x-db10-node-updated-results-37-94-tok-s/362368) — TP=4 속도/동시성
- [Qwen3-Coder-Next](https://forums.developer.nvidia.com/t/how-to-run-qwen3-coder-next-on-spark/359571) — FP8, prefix cache, coding workload
- [Qwen3.6-35B benchmark](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810) — 플랫폼·동시성 비교
- [FP4/NVFP4 support status](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069) — 초기 NVFP4 vs AWQ 실측과 vLLM 상태

### DeepSeek·대형 모델

- [DeepSeek single ds4-on-spark](https://forums.developer.nvidia.com/t/1x-spark-tuned-dspark-for-deepseek-v4-flash-35-tok-s-800-prefill-and-fast-multi-agent-serving/376884) — DSpark, serving, benchmark
- [DeepSeek single Spark EXL3 384K](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) — TP=1, 44–47 tok/s, 약 440K KV pool, 370K needle stress test
- [DeepSeek single 0731](https://forums.developer.nvidia.com/t/1x-spark-deepseek-v4-flash-0731-1-000-tok-s-prefill-59-tok-s-multi-agent-serving/378855) — single 0731, multi-agent
- [DeepSeek EXL3/SparkInfer](https://forums.developer.nvidia.com/t/c1-1058pp-s-52-tg-s-on-1x-dgx-spark-on-deepseek-v4-flash-0731-full-256-experts/379863) — full experts, matched vs peak
- [DeepSeek dual 1M](https://forums.developer.nvidia.com/t/deepseek-v4-flash-aiden-recipe-from-reddit-1m-token-session-operational-cuda-12-1-tailored-for-dgx-spark-gb10/372268) — 1M context/tool-calling
- [DeepSeek dual official FP8](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309) — TP=2, 200K, vLLM fork
- [DeepSeek DSpark instructions](https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220) — YAML/launcher
- [DeepSeek vision shim on 2× Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark) — caption-based OpenAI-compatible image input
- [DeepSeek 4-node TP=4](https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-4x-dgx-spark-via-vllm-jasl-fork-tp-4-rdma-mtp-49-54-tok-s-single-stream-full-recipe-the-traps/373808) — NCCL 2.30.4, RDMA, MTP
- [GLM-4.7 4-node SGLang](https://forums.developer.nvidia.com/t/running-glm-4-7-fp8-355b-moe-on-4x-dgx-spark-with-sglang-eagle-speculative-decoding/359256) — EAGLE, shared-memory tuning
- [MiniMax M2.7 dual](https://forums.developer.nvidia.com/t/minimax-m2-7-nfvp4-recipe-benchmarks/366324) — 196K context, tool parser
- [Nemotron 3 Super](https://forums.developer.nvidia.com/t/nvidia-nemotron-3-super-120b-a12b-nvfp4/363175) — single-node load/first inference issues
- [Nemotron 3 Super 4-node MTP](https://forums.developer.nvidia.com/t/nemotron-3-super-120b-a12b-nvfp4-mtp-on-4x-dgx-spark-via-sglang-tp-4-roce-mtp-actually-pays-off-1-70x-single-stream-accept-len-2-7/373625) — Mamba state/MTP
- [MiMo V2.5 Omni 3-node](https://forums.developer.nvidia.com/t/mimo-v2-5-omni-on-3x-dgx-spark-tp-3-mtp-1m-context-39-tok-s/373948) — multimodal TP=3
- [GLM-5.2 4-node 1M](https://forums.developer.nvidia.com/t/recipe-glm-5-2-on-4x-dgx-spark-at-1m-context-with-sparkrun-vision/380448) — Hermes/vision/Adaptive MTP
- [LongCat-Next](https://forums.developer.nvidia.com/t/longcat-next/372494) — single-node any-to-any multimodal

### 네트워크·클러스터·운영

- [Going from 1 → 2 Sparks](https://forums.developer.nvidia.com/t/going-from-1-2-sparks/373831) — 1.7–1.9×와 2→4 scaling 의견
- [4-node switchless cluster](https://forums.developer.nvidia.com/t/4-node-dgx-spark-cluster-without-a-switch/368726) — full mesh 설계/실험
- [8-node cluster build](https://forums.developer.nvidia.com/t/8x-dgx-spark-cluster-build-report-crs812-400dd-4x100g-breakouts-nemotron-3-ultra-at-tp-8/373146) — 100G breakout, TP=8
- [SGLang RDMA 2.56×](https://forums.developer.nvidia.com/t/glm-4-7-fp8-on-4x-dgx-spark-via-sglang-2-5x-speedup-8-2-25-tok-s-just-by-enabling-rdma/373675) — `/dev/infiniband` passthrough
- [SGLang multi-node traps](https://forums.developer.nvidia.com/t/sglang-multi-node-on-dgx-spark-three-traps-that-wasted-a-day-each-and-how-to-spot-them/373677) — debug flag/노드별 설정
- [NCCL all-reduce deadlock](https://forums.developer.nvidia.com/t/nccl-all-reduce-deadlock-on-dual-dgx-spark-after-successful-channel-establishment-affects-both-vllm-and-trt-llm/366127) — vLLM/TRT-LLM 공통 hang
- [CX-7 link 13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461) — negotiate vs payload
- [CX-7 link 12 Gbps/full power drain](https://forums.developer.nvidia.com/t/dgx-spark-200gbe-direct-qsfp-link-negotiates-200g-but-payload-is-12-gbps/373538) — 현장 recovery
- [CX-7 firmware brick](https://forums.developer.nvidia.com/t/connectx-7-bricked-stuck-in-pre-init-static-config-not-done-after-unsolicited-mlnx-fw-updater-firmware-flash-asus-gx10-error-110/373900) — firmware/RMA risk
- [System memory full crash](https://forums.developer.nvidia.com/t/system-crashes-when-memory-is-full/352339) — unified memory failure mode
- [Thermal automatic shutdown](https://forums.developer.nvidia.com/t/dgxspark-temperature-too-high-automatic-shutdown/363370) — RMA/reproduction discussion
- [Dual Spark cooling cage](https://forums.developer.nvidia.com/t/dual-spark-ducted-cooling-cage/365302) — DIY airflow
- [MCDMA Mac↔Spark prototype](https://x.com/ashxhart/status/2089749434087227672?s=20) — USB-C direct-memory claim, independent verification pending
- [DGX Spark low-clock recovery](https://x.com/Blackwellboy/status/2090611479653622261?s=20) — P0·96% utilization인데 799MHz/19.5W였던 사례
- [GB10 clock cap harness](https://github.com/agjs/gb10-clock-cap) — clock sweep, thermal soak, GPU rail·decode·prefill trade-off
- [2× Spark c4 clock cap report](https://x.com/ivanfioravanti/status/2088730630875930639?s=20) — `nvtop` power/temperature report; wall power와 분리

## 11. 책 집필용 재현 우선순위

### P0 — 우리 장비에서 먼저 고정

- [ ] Qwen3.8 NVFP4: vLLM+MTP, SGLang+DFlash2, llama.cpp를 동일 prompt로 1대 비교
- [ ] Qwen3.8 2대: single stream과 c4/c8 aggregate를 각각 측정
- [ ] DeepSeek-V4-Flash-0731: 1대 ds4/SparkInfer와 2대 vLLM/DSpark를 256K에서 비교
- [ ] 2대 direct QSFP: `iperf3`, `ib_write_bw`, `nccl-tests`, 실제 TP inference를 한 표에 기록
- [ ] tool-eval-bench 3 trials: thinking off/on, clean context/heavy context, tool failure recovery 분리

### P1 — 클러스터 확장

- [ ] 3대 direct ring과 4대 switch의 setup time·NCCL warmup·recovery 비교
- [ ] DeepSeek 4대에서 NCCL 2.28.9/2.30.4 차이 재현
- [ ] 4대 Qwen3.5-397B 또는 GLM 계열에서 100G/200G, socket/RDMA 비교
- [ ] 3대 MiMo/MiniMax 모델별 TP divisibility와 patch provenance 보존
- [ ] 8대는 모델 실행보다 switch, cable, power, thermal BOM과 운영비부터 기록

### P2 — 운영·책 검증

- [ ] 2–8시간 soak test: KV 증가·prefix cache·동시성·tool loop·재시작
- [ ] idle/serving/peak 전력, ACPI/GPU/CPU 온도, fan mode, memory peak 수집
- [ ] GUI/headless, HDMI disconnect, suspend, reboot, firmware update를 별도 체크
- [ ] Hermes/OpenClaw/NemoClaw 권한 경계와 데이터 보존 정책을 모델 benchmark와 분리
- [ ] 모든 결과에 model revision, container digest, runtime commit, seed, raw JSON 첨부

## 12. 리서치 판단

포럼 전체에서 재현 가치가 가장 높은 자료는 “최고 속도”라는 제목의 글이 아니라 다음 네 가지를 함께 제공하는 글이다.

1. 실행 가능한 명령과 고정된 image/commit
2. 모델·양자화·KV·context·concurrency가 적힌 표
3. 정상 출력·tool call·품질 검증
4. 실패한 설정과 복구 방법

반대로 조회수가 높더라도 모델 발표, 구매 논쟁, 단일 스크린샷, `peak` 수치만 있는 글은 책의 결론이 아니라 후보 목록에 둔다. 이 기준으로 보면 현재 가장 강한 실전 축은 `Qwen3.8 single/dual`, `DeepSeek V4 Flash single/dual/quad`, `Qwen3.5-122B/397B`, `GLM-4.7 4-node`, `MiMo 3-node`, `CX-7/NCCL/thermal` 장애 리서치다.

이 문서는 포럼의 살아 있는 스냅샷이다. 글, 댓글, 모델, 이미지는 빠르게 바뀔 수 있으므로 출간 전에는 원문, 최신 모델 카드, 라이선스, 실제 장비 재현 결과를 다시 확인한다.
