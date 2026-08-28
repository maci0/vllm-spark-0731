# A-1. 모델·레시피·명령어 색인

상태: 초안

기준일: **2026-08-22**

이 부록은 본문을 다시 설명하는 장이 아니다. 모델을 고르고 실행할 때 필요한 링크, 버전, 명령어, 용어, 근거 등급을 한곳에 모아 둔 빠른 색인이다. 링크가 있는 항목은 먼저 원문을 확인하고, 이 책의 수치와 실험 결과는 해당 날짜와 조건을 함께 읽는다.

부록은 책의 색인이다. 본문을 다시 읽는 대신 모델·버전·명령어·출처를 찾아가는 지도처럼 사용한다.

## 13.1 이 색인을 사용하는 순서

1. 목적에 맞는 모델과 노드 수를 [모델·노드 수 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-cluster-model-research-2026-08.md)에서 고른다.
2. 실행 전에 첫 부팅에서 실패하지 않는 방법에서 inventory와 메모리 headroom을 기록한다.
3. 단일 노드는 첫 모델을 올리고 “된다”를 증명하는 방법, 다중 노드는 두 대 연결과 다중 Spark 확장을 따른다.
4. 결과는 벤치마크 설계의 schema에 맞춰 저장한다.
5. 장애가 나면 장애 대응과 아래 복구 명령 색인을 먼저 확인한다.

모델 카드에 `supports`, README에 `runs`, 사용기에 `fast`가 적혀 있어도 이 책의 상태는 자동으로 바뀌지 않는다. `loads`, `generates`, `serves`, `benchmarked`, `tool-tested`, `agent-tested`를 각각 확인한다.

## 13.2 버전 기준표

공식 버전은 [DGX Spark Release Notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)를 기준으로 확인하고, 실제 실험에는 사용한 image와 commit을 함께 기록한다.

| 구분 | 버전·값 | 의미 |
|---|---|---|
| 공식 DGX OS 기준 | DGX OS 7.5.0 | Founders Edition release notes 기준 |
| 공식 NVIDIA driver 기준 | 580.159.03 | 파트너 OEM은 업데이트 시점이 다를 수 있음 |
| 공식 CUDA Toolkit 기준 | 13.0.2 | host toolkit 기준 |
| 공식 kernel 기준 | 6.17 | OS 이미지와 함께 확인 |
| 로컬 smoke test | vLLM 0.26.0 | 8083 Qwen3.8 BF16 서버 |
| 로컬 smoke test | PyTorch 2.11.0+cu130 | 위 서버에서 확인한 환경 |
| 로컬 모델 | `OBLITERATUS/Qwen3.8-27B-OBLITERATED` | full BF16 safetensors |
| 로컬 서버 | `http://127.0.0.1:8083/v1` | served model `qwen3.8-27b-obliterated` |

공식 host 버전과 로컬 test 버전이 다를 수 있다. 공식 버전을 표에 적었다고 해서 기존 smoke test가 새 driver나 새 runtime에서 재현되었다는 뜻은 아니다.

## 13.3 모델 색인

| 모델·경로 | 1대 | 다중 노드 | 주 용도 | 근거·상태 |
|---|---|---|---|---|
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | TP=1 | DP 우선 | 공식 agent-ready, tool calling | [NVIDIA vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#run-agent-ready-qwen36-35b-model-with-vllm), official recipe |
| Qwen3.8-27B NVFP4 | SGLang + DFlash2/DSpark | TP=2 또는 DP | 코딩, 빠른 worker, 여러 세션 | [Qwen3.8 SGLang 레포](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark), community recipe |
| `OBLITERATUS/Qwen3.8-27B-OBLITERATED` BF16 | vLLM TP=1 | DP/서비스 분리 | 한국어, 코드, JSON, thinking smoke test | [직접 테스트 노트](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/model-research-qwen38-obliterated.md), locally reproduced 기능 테스트 |
| Qwen3.8-27B community builds | 1대 NVFP4/FP8/4-bit | TP=2, DP, 2×2 pool | serving, speculative decoding, coding agent | 06-12, [원문 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/qwen38-community-builds-2026-08.md) |
| DeepSeek V4 Flash 0731 EXL3 | SparkInfer, TP=1 | 독립 supervisor | 긴 context, c1 agent brain 실험 | [one-Spark EXL3 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark), community recipe |
| DeepSeek V4 Flash 0731 FP8/DSpark | TP=2 | 2대 direct | 256K/1M, 긴 문서와 supervisor | [2대 recipe](https://forums.developer.nvidia.com/t/guide-deepseek-v4-flash-on-2x-dgx-spark-gb10-reproducible-vllm-serving-recipe-up-to-1m-token-context/374742), community recipe |
| DeepSeek V4 Flash 0731 community builds | 1대 EXL3 또는 2대 TP=2 | vision shim·agent·H3 co-tenancy | 사람들이 만든 실제 응용·benchmark | 06-9 커뮤니티 제작물, [원문 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/deepseek-v4-flash-0731-community-builds-2026-08.md) |
| Qwen3.5-122B-A10B INT4 | 특수 vLLM/Marlin/MTP | DP 또는 추가 TP 검증 | 큰 단일 supervisor | [단일 Spark 레포](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4), recipe/measurement |
| Qwen3.5-397B INT4 AutoRound | 불가 | PP=3 실험, TP=4 측정 | 대형 단일 모델 | [4× 레시피](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/4x-spark-cluster/qwen3.5-397b-int4-autoround.yaml), community recipe |
| `openai/gpt-oss-120b` MXFP4 | llama.cpp | DP/서비스 분리 | 일반 추론, tool 기준선 | [NVIDIA 모델 지원표](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md#model-support-matrix), official support path |
| DS4 vision shim | 기존 DS4 앞에 `:8899` | 2×Spark DS4에 추가 | caption 기반 이미지 입력 | [vision shim 레포](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark), caption pipeline |

Qwen3.8-27B를 UI·디자인 worker로 두고 DeepSeek를 supervisor로 두는 구성이 필요하면 로컬 에이전트 운영과 2×2 구성을 함께 본다. 두 모델을 각각 TP=2로 실행하는 경우 전체 장비는 네 대다.

Qwen3.8의 단일 Spark recipe와 실제 활용 사례는 06-12: Qwen3.8-27B로 사람들이 만든 것과 [상세 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/qwen38-community-builds-2026-08.md)에서 원본, 최적화 recipe, OBLITERATED 파생 모델을 나눠 기록한다.

DGX Spark·GB10의 Founders Edition과 Acer·ASUS·Dell·GIGABYTE·HP·Lenovo·MSI 비교는 01-2와 [벤더 비교 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-vendor-comparison-2026-08.md)에서, DeepSeek·Qwen·MiniMax의 모델 선택은 06-4와 [모델 선택 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-model-selection-research-2026-08.md)에서 관리한다.

## 13.4 노드 수와 topology 색인

| 노드 수 | 기본 topology | 스위치 | 먼저 확인할 문서 |
|---:|---|---|---|
| 1 | TP=1 | 불필요 | 첫 모델을 올리고 “된다”를 증명하는 방법 |
| 2 | direct QSFP/RoCE | 불필요 | [NVIDIA Connect Two Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks), 2대 연결 |
| 3 | QSFP ring 또는 DP=3 | 공식 ring은 불필요 | [NVIDIA Connect Three Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks), 3대 확장 |
| 4 | QSFP switch, TP=4 또는 DP=4 | 권장·공식 경로 | [NVIDIA Multi-Spark Switch](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch) |
| 6~8 | switch fabric, TP/PP/DP 조합 | 사실상 필요 | [다중 Spark 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-nvidia-forum-research-2026-08.md) |
| 2×2 | pair별 direct 또는 공통 switch | 구성에 따라 다름 | 2×2 역할 분리 |

link가 올라왔다는 사실과 RDMA가 NCCL에 사용된다는 사실은 다르다. 최소한 `NET/IB`, all-reduce, 작은 TP 요청, 긴 soak까지 확인한 뒤 다중 노드 benchmark로 기록한다.

## 13.5 명령어 색인

명령어는 복사하기 전에 현재 interface 이름, port, model id, context, runtime 버전을 바꾼다. 아래 표는 명령어의 위치를 찾기 위한 색인이고, 실행 순서를 대신하지 않는다.

| 목적 | 대표 명령 | 기준 위치 |
|---|---|---|
| 장치·driver 확인 | `nvidia-smi` | 첫 부팅, 장애 대응 |
| 메모리·디스크 확인 | `free -h`, `df -h` | 첫 부팅 |
| 노드 inventory | `hostnamectl`, `uname -a`, `ip -br addr` | 두 대 연결 |
| RDMA device 확인 | `ibdev2netdev`, `rdma link`, `ibstat` | 두 대 연결, [Mac·RDMA 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-mac-rdma-switch-research-2026-08.md) |
| 물리 링크 확인 | `ethtool <interface>` | 두 대 연결, [스위치 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-mac-rdma-switch-research-2026-08.md) |
| endpoint health | `curl -sS http://127.0.0.1:PORT/v1/models` | 벤치마크 설계 |
| 단일 BF16 server | `vllm serve ...` | 첫 모델을 올리고 “된다”를 증명하는 방법, [루트 README](https://github.com/recrack/oh-my-dgx-spark/blob/main/README.md) |
| Qwen3.8 canonical server | `scripts/run-qwen38-vllm.sh` | [루트 README](https://github.com/recrack/oh-my-dgx-spark/blob/main/README.md), [실측 JSON](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/results-qwen38-vllm-parser-2026-08-21.json) |
| 기능 smoke test | `python3 tests/qwen38_smoke.py ...` | [루트 README](https://github.com/recrack/oh-my-dgx-spark/blob/main/README.md) |
| tool call smoke test | `python3 tests/tool_call_smoke.py --strict ...` | [루트 README](https://github.com/recrack/oh-my-dgx-spark/blob/main/README.md), 로컬 에이전트 |
| 반복 측정 | `python3 tests/repeat_benchmark.py --trials 5 ...` | 벤치마크 설계, [루트 README](https://github.com/recrack/oh-my-dgx-spark/blob/main/README.md) |
| tool parser | `--enable-auto-tool-choice`, `--tool-call-parser ...` | 엔진 선택, 로컬 에이전트 |
| 저클럭 관찰 | `nvidia-smi --query-gpu=... -l 1` | 장애 대응 |
| NCCL 경로 | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=...` | 2대 연결, [포럼 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-nvidia-forum-research-2026-08.md) |
| UCX recipe workaround | `UCX_MEM_MMAP_HOOK_MODE=none` | 2대 연결 |

단일 Spark의 canonical server 명령은 [루트 README의 실행 섹션](https://github.com/recrack/oh-my-dgx-spark/blob/main/README.md#실행)에 두고, 책의 다른 장에서는 같은 명령을 복사해 서로 다른 버전으로 만들지 않는다.

## 13.6 상태와 근거 등급

### 실행 상태

| 상태 | 기록할 때의 의미 |
|---|---|
| `loads` | weight와 runtime이 메모리에 올라왔음 |
| `generates` | 고정한 기본 prompt에 정상 텍스트가 나왔음 |
| `serves` | OpenAI-compatible endpoint가 반복 요청에 응답했음 |
| `benchmarked` | prompt, output, context, concurrency, 반복 수가 고정된 측정이 있음 |
| `tool-tested` | parser, schema, tool name, arguments, error recovery를 확인했음 |
| `agent-tested` | 여러 단계의 tool loop와 실패 복구를 확인했음 |
| `long-context-tested` | 지정 context에서 retrieval, 품질, 안정성을 함께 확인했음 |

모델이 `loads`에 머물렀다면 `usable`, `agent-ready`, `supports 1M`이라고 쓰지 않는다. 한 단계의 PASS가 다음 단계의 PASS를 대신하지 않는다.

### 자료 근거 등급

| 등급 | 출처 | 책에서의 사용 |
|---|---|---|
| A | NVIDIA·Apple·runtime 공식 문서 | 사양, 공식 지원, 설치 전제 |
| B | 실행 명령과 설정이 공개된 recipe | 재현 후보, 장비에서 재검증 |
| C | 조건과 측정 방법이 공개된 benchmark | 숫자 비교, 조건을 함께 인용 |
| D | Reddit, X, 포럼, 국내 커뮤니티 경험담 | 실패 사례와 실험 주제 탐색 |
| E | 책·웹북 | 목차, 독자 수준, 포지셔닝 비교 |
| F | 모델 카드·논문 | 모델 정체, 라이선스, 기능, 자체 평가 |

전체 참고문헌은 [책 집필용 참고문헌](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-book-references-2026-08.md), 포럼 자료는 [NVIDIA 포럼 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-nvidia-forum-research-2026-08.md)에서 관리한다.

## 13.7 결과 파일과 provenance

benchmark 결과 하나만 저장하지 말고, 같은 이름의 metadata와 raw response를 함께 보존한다.

```text
results/
  2026-08-21-qwen38-bf16-c1.json
  2026-08-21-qwen38-bf16-c1.raw.jsonl
  2026-08-21-qwen38-bf16-c1.env.txt
  2026-08-21-qwen38-bf16-c1.log
```

각 결과에는 다음 필드를 넣는다.

```text
measured_at:
hardware:
node_count:
topology:
os_version:
kernel:
driver:
cuda:
nccl:
container_image:
container_digest:
runtime_commit:
model_repo:
model_revision:
quant:
kv_dtype:
context:
concurrency:
prompt_tokens:
output_tokens:
thinking:
speculation:
transport:
wall_power_w:
temperature_peak_c:
result:
quality_result:
error_rate:
```

GitHub URL만 저장하면 시간이 지나 recipe가 바뀔 수 있다. 가능하면 commit SHA, container digest, 모델 revision, 실행 명령, raw JSON, 접근일을 함께 남긴다.

## 13.8 출간 전 업데이트 체크리스트

- [ ] DGX OS와 driver 버전을 공식 Release Notes와 다시 대조했다.
- [ ] 각 모델 링크가 원본 checkpoint와 같은 revision을 가리키는지 확인했다.
- [ ] 모델 카드의 주장과 locally reproduced 결과를 분리했다.
- [ ] 가격·환율·전력 단가에는 조회일과 계산 가정을 적었다.
- [ ] 2대 direct, 3대 ring, 4대 switch topology를 문서와 실제 장비에서 구분했다.
- [ ] `NET/IB`와 socket fallback 결과를 같은 benchmark 행에 넣지 않았다.
- [ ] single-stream, prefill, decode, aggregate, agent success rate를 각각 기록했다.
- [ ] tool parser와 agent framework 결과를 별도 상태로 표시했다.
- [ ] MCDMA를 공식 memory pooling이나 비용 절감 근거로 사용하지 않았다.
- [ ] 본문 링크와 명령어 색인을 다시 실행해 깨진 경로를 확인했다.

## 이 부록의 결론

본문은 판단을 설명하고, 이 부록은 다시 실행할 위치를 알려준다. 숫자를 갱신할 때는 문장을 먼저 고치기보다 원문, revision, 조건, 측정 파일을 함께 갱신한다. 그래야 새로운 모델과 runtime이 나와도 책의 결론과 실험 기록을 분리해 유지할 수 있다.
