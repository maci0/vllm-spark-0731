# DGX Spark 모델·클러스터 리서치

조사일: **2026-08-21**

이 문서의 대상은 GB10과 128 GiB unified memory를 사용하는 NVIDIA DGX Spark다. 아래의 `tok/s`는 엔진, 프롬프트, 컨텍스트 길이, thinking 설정, speculative decoding 조건이 서로 다른 공개 측정값이므로 절대적인 순위로 볼 수 없다. 성능 수치는 반드시 원문 레시피와 측정 조건을 함께 확인한다.

## 결론

- **1대**: 공식 지원과 운영 안정성을 우선하면 `nvidia/Qwen3.6-35B-A3B-NVFP4`가 기본 후보다. 코딩 속도는 Qwen3.8 27B가, 모델 품질과 장문 작업은 특수 SparkInfer DeepSeek V4 Flash 0731 또는 Qwen3.5-122B가 우선 검토 대상이다. 최신 단일 Spark 경로인 [MiaAI-Lab의 DeepSeek V4 Flash EXL3 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)도 별도 프로필로 추가한다.
- **2대**: 원본에 가까운 DeepSeek V4 Flash 0731을 TP=2로 실행하고 256K~1M 컨텍스트를 확보하는 구성이 가장 설득력 있다. 1M은 단일 스트림과 낮은 동시성에, 256K는 더 높은 aggregate throughput에 적합하다.
- **2×2 총 4대**: DeepSeek TP=2 supervisor와 Qwen3.8-27B TP=2 UI·디자인 worker를 독립 pool로 운영할 수 있다. 이 구성은 TP=4 단일 모델이 아니므로 두 pool 사이의 라우팅과 장애 격리를 별도로 설계해야 한다.
- **3대**: `TP=3`을 기본값으로 두지 않는다. 현재 공개 레시피 기준으로는 `PP=3` 대형 모델 또는 `DP=3` 독립 서비스가 현실적이다. 단일 요청 속도만으로 3대가 2대보다 빨라진다고 가정해서도 안 된다.
- **4대**: 단일 대형 모델을 확장하려면 3대보다 유리하다. `TP=4`가 가능한 Qwen3.5-397B 레시피가 공개되어 있고, 4대부터는 스위치 기반 클러스터 구성이 자연스럽다.
- **구매·구성 판단**: Spark를 한 대 추가할 때마다 모델 크기만 늘어나는 것은 아니다. 2대는 DeepSeek와 장문 컨텍스트, 3대는 서비스 분리와 실험 여유, 4대는 TP 기반 대형 단일 모델이라는 서로 다른 성격을 갖는다.

## 노드 수별 권장 구조

| 구성 | 권장 병렬화 | 네트워크 | 가장 현실적인 용도 | 공개 검증 수준 |
|---|---|---|---|---|
| 1대 | TP=1 | 불필요 | 27~122B급 단일 모델, 개발 보조, 멀티모달 | 공식·커뮤니티 모두 많음 |
| 2대 | TP=2 | 200GbE QSFP/RoCE 직접 연결 | DeepSeek 분산, 1M 컨텍스트, 70B 이상 | 높음 |
| 3대 | PP=3 또는 DP=3 | 3-way ring 또는 스위치 | 397B급 용량 확보, 서비스 3개 분리 | 네트워크/레시피는 검증, 성능은 제한적 |
| 4대 | TP=4 또는 DP=4 | 200GbE QSFP 스위치 권장 | 대형 단일 모델, MiniMax/Qwen 397B | 가장 명확한 다중 노드 경로 |
| 6~8대 | TP/PP/DP 조합 | 스위치 필수 | 랙형 실험·대규모 서비스 | 일반화된 런처는 있으나 모델별 검증 필요 |

NVIDIA는 2대용 직접 QSFP 연결, 3대용 ring 연결, 4대 이상용 QSFP 스위치 연결 플레이북을 각각 제공한다. [2대 연결](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks), [3대 ring](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks), [스위치를 통한 다중 Spark](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch)을 기본 네트워크 참고자료로 사용한다.

## 1대 DGX Spark

### 추천 순서

| 우선순위 | 모델/경로 | 엔진 | 공개 측정·특징 | 판단 |
|---:|---|---|---|---|
| 1 | `nvidia/Qwen3.6-35B-A3B-NVFP4` | vLLM | NVIDIA 공식 TP=1, FP8 KV, 262K, MTP=3, tool calling 레시피 | 가장 안전한 공식 기본값 |
| 2 | Qwen3.8-27B NVFP4 | SGLang + DFlash2/DSpark | 한 커뮤니티 구성에서 greedy median 50 tok/s, 코딩 32~40, reasoning 41~57, 8-stream aggregate 135~148 | 코딩·일상 사용의 속도 우선 |
| 3 | DeepSeek V4 Flash 0731 Spark 변형 | SparkInfer + EXL3/Trellis | 기존 경로는 262K·코드 decode median 38.12 tok/s; 최신 경로는 384K 설정·44–47 tok/s 구조화 decode·약 440K KV pool·370K needle stress | 품질·장문 우선. 공식 원본 FP8과 구분 필요 |
| 4 | Qwen3.5-122B-A10B INT4 AutoRound | 특수 vLLM/Marlin/MTP | 단일 Spark, 256K, 교차 프롬프트 평균 51.6 tok/s, LongCode 54.9 | 큰 모델 품질과 속도의 균형 |
| 5 | `openai/gpt-oss-120b` MXFP4 | llama.cpp | NVIDIA 공식 측정에서 ISL/OSL 2048/128, BS1 기준 55.37 tok/s | tool/일반 추론 기준선 |
| 6 | Nemotron-3-Nano-Omni-30B-A3B | vLLM | NVIDIA 공식 vLLM 지원표에 BF16/FP8/NVFP4와 멀티모달 경로가 있음 | 공식 멀티모달 후보 |

세부 근거는 다음과 같다.

- NVIDIA 공식 Qwen3.6 레시피는 TP=1, FP8 KV, 262,144 토큰, MTP 3, reasoning/tool parser를 사용한다. [공식 vLLM Playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#run-agent-ready-qwen36-35b-model-with-vllm)
- Qwen3.8의 공개 측정은 [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)과 [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)에 정리되어 있다. 숫자는 DFlash2, 프롬프트 종류, 동시성에 따라 크게 달라진다.
- 단일 Spark DeepSeek의 기존 경로는 [0xSero/deepseek-v4-flash-0731-spark-sparkinfer](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer)이고, 최신 경로는 [MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)다. 두 경로 모두 원본 full FP8 체크포인트를 그대로 실행하는 공식 경로로 기록하지 않는다. 최신 recipe는 0xSero의 EXL3 3.0 bpw/REAP-K216 가중치, SparkInfer, DSpark K5/K64 draft, native NVFP4 KV 경로를 조합한다. README 기준 `MAX_MODEL_LEN=384000`, `MAX_NUM_SEQS=1`, `GPU_MEMORY_UTILIZATION=0.94`에서 구조화 decode 44–47 tok/s, 약 439,622토큰의 KV pool, 370,104토큰 needle 회수를 보고한다. 초기 prefill은 약 1024 tok/s지만 300K 이후에는 약 350–614 tok/s로 낮아지며, 370K 시험의 실효 prefill은 약 625 tok/s다. 이 수치는 단일 스트림·fresh boot 조건의 recipe 보고값이며, thinking off는 needle stress test 조건으로만 명시되어 있다. full-expert 품질, 다중 스트림, 일반 장문 품질을 보증하는 종합 벤치마크가 아니다. REAP가 256개 expert 중 216개를 유지하고 c4에서는 KV pool이 줄어들 수 있다는 점, 첫 부팅에 약 107GB 다운로드, coalesce, CUDA graph capture가 필요하다는 운영 조건도 함께 기록한다. 자세한 범위와 GPT-5.6-Sol 비교는 [DeepSeek V4 Flash 0731 성능 리서치](deepseek-v4-flash-0731-performance-research-2026-08.md)를 따른다.
- Qwen3.5-122B 수치는 [albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)의 10 sub-run 측정이다. TurboQuant KV는 약 4배 KV 용량을 늘리지만 51~52에서 약 39 tok/s로 내려가는 속도-메모리 교환이다.
- NVIDIA의 모델 지원 범위는 [공식 vLLM 모델 지원표](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#model-support-matrix)에서 확인한다. 공식 지원은 “실행 경로가 있다”는 뜻이지 모든 모델의 동일한 성능을 보장한다는 뜻은 아니다.

### 1대에서의 권장 선택

- **바로 쓰는 코딩 에이전트**: Qwen3.6 공식 레시피부터 시작하고, 속도·통합을 더 밀고 싶으면 Qwen3.8 SGLang/DFlash2를 비교한다.
- **품질·긴 문서·DeepSeek 실험**: 단일 Spark용 DeepSeek EXL3/SparkInfer 최신 recipe를 별도 프로필로 둔다. “DeepSeek 원본 FP8이 1대에서 공식적으로 돌아간다”라고 기록하면 안 된다. 370K needle 통과는 컨텍스트 회수 스트레스 결과이지, 원본 모델과 동등한 품질 인증이나 모든 장문 작업의 정확도 증명이 아니다.
- **큰 모델을 1대에서**: Qwen3.5-122B는 가능하지만 특수 양자화와 SM121용 빌드가 필요하므로 운영 기본값으로는 Qwen3.6/Qwen3.8보다 복잡하다.

## 2대 DGX Spark

### 핵심 구성: DeepSeek TP=2

2대는 현재 가장 명확한 업그레이드 지점이다. 두 Spark를 200GbE QSFP/RoCE로 연결한 뒤 DeepSeek V4 Flash를 TP=2로 나누어 실행한다.

| 프로필 | 컨텍스트 | 공개 측정 | 용도 |
|---|---:|---:|---|
| 1M profile | 1,048,576 | 약 37 tok/s single-stream, 약 100 tok/s aggregate | 긴 문서·저동시성 |
| 256K profile | 262,144 | 약 40 tok/s single-stream, 약 150 tok/s aggregate | 코딩·일반 서비스 |

이 수치는 [NVIDIA Developer Forum의 재현 레시피](https://forums.developer.nvidia.com/t/guide-deepseek-v4-flash-on-2x-dgx-spark-gb10-reproducible-vllm-serving-recipe-up-to-1m-token-context/374742)와 [MiaAI-Lab의 최신 2x DSpark 레포](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)에 기반한다. 최신 레포는 공식 0731 체크포인트, TP=2, DSpark, `nvfp4_ds_mla` KV, 1M ceiling, 256K 고동시성 프로필을 제공한다.

1M은 모델이 1M 토큰을 여러 요청에서 동시에 처리한다는 뜻이 아니다. 공개 레시피의 KV pool과 `max_num_seqs`를 함께 확인해야 하며, 실제 운영에서는 “한두 개의 긴 요청”과 “여러 개의 짧은 요청” 중 하나를 선택하게 된다.

### 2대에서 가능한 다른 모델

- NVIDIA 공식 2-Spark vLLM 예제는 Llama 3.3 70B를 TP=2로 실행한다. 405B AWQ도 시험 명령은 있지만 `max-model-len=64`와 `max-num-seqs=1` 수준이며 운영용 headroom 부족을 명시한다. [공식 2-Spark vLLM 경로](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#run-on-two-sparks)
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)는 Qwen3.5-397B INT4-AutoRound를 2대에서 vision/full-context로 실행하는 실험 경로를 제공한다. 다만 실험적 레시피이므로 속도·장시간 안정성은 직접 검증해야 한다.
- 1대에 이미 들어가는 Qwen3.8, Qwen3.6, Qwen3.5-122B는 TP로 묶기보다 각 Spark에 한 인스턴스씩 띄우는 DP/서비스 분리가 동시성 측면에서 더 자연스럽다.

### 2대 운영 주의점

2대 DeepSeek에 장시간 부하를 걸면 unified memory와 RDMA transport가 함께 문제를 일으킬 수 있다. 공개 재현 보고서는 UCX 메모리 등록 캐시와 관련된 환경 변수(`UCX_MEM_MMAP_HOOK_MODE=none`, `UCX_RCACHE_MAX_UNRELEASED=1024`)를 누수 완화책으로 제시한다. 이 설정은 모든 환경의 공식 해결책이 아니라 해당 레시피의 안정화 조치이므로 적용 전후의 soak test를 남긴다.

### 2×2 구성: DeepSeek brain과 Qwen3.8 UI·디자인 worker

DeepSeek V4 Flash 0731을 agent brain으로 유지하고 Qwen3.8-27B를 UI·디자인 worker로 넘기는 구성을 실제 장비 수로 표현하면 다음과 같다.

| Spark pool | 모델 | 병렬화 | 역할 |
|---|---|---|---|
| A, 2대 | DeepSeek V4 Flash 0731 | TP=2 | supervisor, 긴 문맥, tool loop, 복구 판단 |
| B, 2대 | Qwen3.8-27B | TP=2 | UI·레이아웃·CSS·디자인 반복 |

이는 총 네 대의 Spark로 두 개의 독립 endpoint를 운영하는 구성이다. `TP=4`로 하나의 모델을 실행하는 방식이 아니며, 두 pool의 unified memory를 자동으로 합치지도 않는다. 각 pair를 직접 QSFP로 연결하면 pair별로 switch 없이 운영할 수 있다. 네 대를 하나의 shared fabric으로 묶거나 공통 switch를 사용하려면 200GbE QSFP switch와 NCCL 경로를 별도로 검증한다.

DS4 이미지 입력은 [DeepSeek vision shim recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)로 기존 `:8888` 서버 앞에 `:8899` shim을 추가하는 방식으로 구성할 수 있다. 이 저장소의 기본 VLM은 `Qwen3.5-0.8B-MLX-8bit`이며 caption을 DS4에 전달한다. 따라서 이 자료를 Qwen3.8-27B의 native vision 지원 근거로 사용하지 않고, DS4를 재배포하지 않는 OpenAI-compatible 입력 계층의 사례로만 기록한다.

보유 장비가 두 대뿐이라면 DS4 TP=2와 Qwen3.8 TP=2를 각각 독립 서비스로 실행할 수 있다고 기록하지 않는다. 같은 두 노드에 두 모델을 함께 올리는 방식은 메모리, KV cache, 통신 경합을 새로 측정해야 한다. 따라서 책의 기본 권장은 두 모델 중 하나를 선택하는 것이다.

## 3대 DGX Spark

3대는 숫자만 보면 2대보다 좋아 보이지만 병렬화 방식 때문에 가장 주의해서 다뤄야 하는 구성이다.

### 왜 TP=3을 기본으로 하면 안 되는가

현재 공개 커뮤니티 레시피는 일반적으로 사용하는 모델에서 `tensor-parallel-size=3`을 지원하지 않는다고 명시한다. 따라서 세 대를 하나의 모델에 모두 사용할 때는 다음 두 가지 경로를 먼저 검토한다.

1. **PP=3**: 레이어를 세 노드에 나눈다. 2대에 안 들어가는 모델을 수용하는 데 유용하지만, 파이프라인 통신과 bubble 때문에 single-stream decode가 자동으로 빨라지지 않는다.
2. **DP=3**: 한 대에 들어가는 모델을 세 Spark에 각각 띄운다. 모델 용량은 늘지 않지만 동시 사용자 수와 장애 격리에는 가장 유리하다.

[eugr의 3-node 문서](https://github.com/eugr/spark-vllm-docker/blob/main/README.md#support-for-3-node-mesh-setups)는 TP=3 대신 PP=3 또는 DP=3을 권장한다. [네트워킹 문서](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md)는 3-node mesh를 PP/DP 중심으로 보고, TP는 2·4·8 같은 power-of-two 노드 수가 일반적으로 더 적합하다고 설명한다.

### 3대의 실제 경로

| 경로 | 레시피 | 판단 |
|---|---|---|
| 대형 단일 모델 | `Intel/Qwen3.5-397B-A17B-int4-AutoRound`, PP=3, 262K | [공개 YAML](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/3x-spark-cluster/qwen3.5-397b-int4-autoround.yaml)이 있음. 용량 우선, 속도 측정은 부족 |
| 세 서비스 분리 | Qwen3.8/Qwen3.6/Qwen3.5-122B를 각 노드에 1개씩, DP=3 | 동시성·역할 분리에 가장 현실적. 이는 구성 권장사항이며 동일 하니스의 공개 종합 벤치는 아님 |
| 2+1 혼합 | 2대 DeepSeek TP=2 + 1대 Qwen3.8 또는 Qwen3.5-122B | 긴 문서/고품질 모델과 빠른 코딩 모델을 동시에 운영하는 실전형 제안 |

### 3대 네트워크

NVIDIA 공식 ring 플레이북은 Spark 3대와 QSFP 케이블 3개를 요구한다. Node1–Node2, Node2–Node3, Node3–Node1을 ring으로 연결한다. 커뮤니티 vLLM 문서는 QSFP mesh와 별도로 모든 노드를 공통 Ethernet에 연결해 NCCL OOB 통신을 확보해야 한다고 안내한다. [NVIDIA 3-Spark ring 플레이북](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks), [NVIDIA forum의 3-node mesh 설명](https://forums.developer.nvidia.com/t/three-node-spark-clusters-without-a-switch-are-now-supported-in-spark-vllm-docker-and-sparkrun/365296)

**결론적으로 3대는 “가장 큰 단일 모델을 가장 빠르게 돌리는 단계”가 아니다.** 이미 3대를 갖고 있다면 PP=3 대형 모델 또는 DP/2+1 서비스 구성이 합리적이다. 단일 모델의 속도 확장이 목적이라면 4대를 기다리는 편이 낫다.

## 4대 및 3대 이상

### 4대: TP=4가 되는 첫 번째 깔끔한 확장점

4대는 200GbE QSFP 스위치에 연결하고 TP=4를 사용하는 구성이 가장 이해하기 쉽다. NVIDIA 공식 스위치 플레이북도 4대를 기준으로 설명하며 같은 구조를 더 많은 Spark로 확장할 수 있다고 안내한다.

커뮤니티 `spark-vllm-docker`의 Qwen3.5-397B INT4-AutoRound TP=4 레시피는 다음 측정값을 공개한다.

- single-user: 약 37 tok/s
- 4 concurrent aggregate: 약 103 tok/s
- 기본 `max_model_len`: 32K
- GB10에서 driver 580.x 권장; 해당 레시피는 590.x의 CUDAGraph deadlock 이슈를 명시한다.

레시피: [Qwen3.5-397B TP=4](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/4x-spark-cluster/qwen3.5-397b-int4-autoround.yaml)

NVIDIA 공식 다중 Spark 예제는 4대에서 `MiniMaxAI/MiniMax-M2.5`를 TP=4, 129K 컨텍스트로 실행하는 흐름을 보여준다. [공식 multi-Spark vLLM 경로](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#run-on-multiple-sparks-through-a-switch)

### 6~8대

스위치 기반으로 노드를 추가할 수 있지만 모델별로 TP, PP, DP 조합을 다시 검증해야 한다. [NVIDIA의 multi-Spark 네트워크 플레이북](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch)은 4대 기준과 확장 가능한 구조를 제공한다. [mark-ramsey-ri/vllm-dgx-spark](https://github.com/mark-ramsey-ri/vllm-dgx-spark)는 1-to-N 런처와 41개 모델 preset을 제공하지만 공개 README 기준으로 1·2대는 end-to-end 검증되었고 n>2 경로는 코드 검토 단계로 기록되어 있다. 따라서 이 문서에서는 6~8대 성능을 숫자로 추정하지 않는다.

## 모델과 노드 수 매트릭스

| 모델/계열 | 1대 | 2대 | 3대 | 4대 이상 | 주된 이유 |
|---|---|---|---|---|---|
| Qwen3.6-35B NVFP4 | 공식 TP1 | DP/동시성 | DP=3 | DP=4 | 이미 1대에 충분히 들어감 |
| Qwen3.8-27B | SGLang + DFlash2 추천 | DP 또는 2개 서비스 | DP=3 | DP=4 | 속도·코딩·서비스 동시성 |
| DeepSeek V4 Flash 0731 | SparkInfer 변형 | **TP=2 + DSpark, 256K/1M** | PP=3 비추천, DP/2+1 권장 | 별도 TP/PP 레시피 필요 | 2대에서 원본 계열의 장문 경로가 가장 명확 |
| Qwen3.5-122B INT4 | 단일 Spark 가능 | DP/더 높은 동시성 | DP/서비스 분리 | DP | 1대 특수 레시피가 이미 강함 |
| Qwen3.5-397B INT4 | 불가 | PP 실험 가능 | PP=3 레시피 | **TP=4 측정 레시피** | 4대가 단일 모델 확장의 기준 |
| GPT-OSS-120B MXFP4 | 공식 단일 Spark | DP | DP | DP/서비스 | 모델이 1대에 들어감 |
| Nemotron-3-Super-120B NVFP4 | 공식/커뮤니티 단일 경로 | 단일 모델 또는 DP | DP/PP 검증 필요 | PP/TP 검증 필요 | 공식 지원은 1~2대 중심 |
| MiniMax M2.5 | 불명확 | 불명확 | PP/TP 검증 필요 | 공식 TP=4 예제 | 4대 이상 대형 모델 경로 |

`DP`는 같은 모델을 노드마다 복제해 동시성을 늘리는 방식이고, `TP`는 한 모델의 가중치를 여러 노드에 나누는 방식이다. `PP`는 레이어를 노드에 나누므로 메모리 용량을 늘리는 데 유리하지만 파이프라인 지연이 발생한다.

## 목적별 최종 추천

| 목적 | 추천 구성 |
|---|---|
| 1대만 구매/보유 | Qwen3.6 공식 레시피 → Qwen3.8 SGLang → Qwen3.5-122B/DeepSeek SparkInfer 순서로 비교 |
| DeepSeek를 제대로 쓰기 | 2대 + 직접 200GbE 연결 + DSpark TP=2 |
| 1M 컨텍스트 | 2대 DeepSeek 1M profile. 동시성보다 긴 단일 요청에 맞춤 |
| 빠른 코딩 에이전트 여러 개 | 1대 Qwen3.8 인스턴스 또는 3대 DP=3 |
| 큰 단일 모델 | 3대 PP=3은 실험, 4대 TP=4를 운영 후보 |
| 멀티모델 운영 | 3대에서 DeepSeek 2대 + Qwen3.8/Qwen3.5 1대의 2+1 배치 |
| 멀티모달 | 1대 Nemotron-3-Nano-Omni 또는 공식 Qwen/Phi VLM 경로 |

## 공개 자료의 신뢰도 구분

1. **공식 실행 근거**: NVIDIA Playbooks, NVIDIA Developer Forum의 재현 레시피. 네트워크·실행 가능성 판단에 우선한다.
2. **재현 가능한 커뮤니티 근거**: 커밋/컨테이너/측정 방법을 고정한 GitHub 레포. 성능 비교에 유용하지만 동일 하니스가 아니면 숫자를 직접 비교하지 않는다.
3. **경험담**: Reddit, X, Serverforum/아카라이브. 실제 사용감과 실패 사례를 찾는 데 유용하지만 벤치마크 근거로는 낮은 가중치를 둔다.

한국 커뮤니티에서는 Serverforum에 Qwen3.8 Q4_K_XL, 128K, q8 KV, MTP 설정으로 약 37 tok/s를 안정적으로 기록한 사용기와 코딩 에이전트 경험담이 올라와 있다. Qwen3.8 v3 IQ3_S의 perplexity 개선 보고도 있다. [Qwen3.8 사용기](https://svrforum.com/ai/3170124), [Qwen3.8 v3 PPL 비교](https://svrforum.com/ai/3174599). 유용한 현장 자료이지만 프롬프트, 빌드, 출력 길이가 고정된 공인 벤치는 아니다. 아카라이브는 이번 조사 시점에 검색과 페이지 접근이 안정적이지 않아 근거로 사용하지 않았다.

품질 비교의 보조 자료로는 [Weschera/spark-bench](https://github.com/Weschera/spark-bench)를 사용한다. 현재 one-Spark v6.7.1 challenge cohort에서는 DeepSeek V4 Flash 0731 spec-off가 TrueScore 92.0, DSpark가 86.7로 기록되어 있지만, 이는 같은 20-scenario/10-domain/TP1 계약 안의 결과이고 methodology 버전 간 숫자를 섞으면 안 된다.

## 이 리포에서 직접 확인한 것

우리 장비에서는 `OBLITERATUS/Qwen3.8-27B-OBLITERATED` BF16 vLLM 서버를 32K 설정으로 실행하고 한국어, 코드, JSON, thinking, vision, 멀티턴, 32K, 4-way smoke test를 통과시켰다. 당시 속도는 GPU를 공유했고 출력이 짧았으며 speculative decoding을 사용하지 않은 조건에서 대략 4~5 tok/s였다. 따라서 위의 공개 최적화 수치와 비교하지 않는다. [직접 테스트 기록](test-results-2026-08-21.md)

## 다음 검증 순서

1. 현재 Spark에서 Qwen3.6 공식 TP1과 Qwen3.8 DFlash2를 같은 프롬프트로 측정한다.
2. 단일 Spark DeepSeek EXL3 recipe는 먼저 이미지·디스크·메모리 요구량, 모델 revision, coalesce/graph capture 시간을 별도 확인한다. README의 `GPU_MEMORY_UTILIZATION=0.94`와 EarlyOOM 비활성화는 이 실험의 메모리 운영 조건으로만 기록하고 일반 권장값으로 확장하지 않는다.
3. 두 번째 Spark가 연결되면 NCCL bandwidth/health를 먼저 통과시킨 뒤 DeepSeek 256K profile을 돌린다.
4. 1M profile은 256K와 분리해 UCX leak, UMA free memory, 장시간 soak를 기록한다.
5. 세 번째 Spark를 추가할 경우 TP=3을 시도하기 전에 DP=3과 PP=3을 각각 비교한다.
6. 네 번째 Spark부터 Qwen3.5-397B TP=4를 동일한 prompt/출력 길이/동시성 하니스로 검증한다.
