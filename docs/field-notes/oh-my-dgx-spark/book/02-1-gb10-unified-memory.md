# 02-1. GB10·unified memory 상세

상태: 초안

이 장은 “128GB니까 128GB 모델이 그대로 들어가는가?”, “왜 큰 모델은 올라가도 느린가?”, “왜 `nvidia-smi`와 runtime의 메모리 숫자가 다른가?”라는 질문에 답한다. 목표는 명령어를 외우는 것이 아니라, 모델이 올라온 뒤 요청을 처리하다가 중단되는 이유를 해석하는 것이다.

## 3분 이해 (ELI5)

unified memory는 모델 전용 서랍이 아니라 여러 사람이 함께 쓰는 큰 책상이다.

```text
큰 책상 128 GiB
├─ 모델 weight: 책
├─ KV cache: 펼쳐 둔 메모
├─ CUDA workspace: 작업 도구
└─ OS·runtime·에이전트: 주변 물건
```

책을 책상에 올릴 수 있어도 펼쳐 놓을 자리가 없으면 긴 작업은 실패한다.

![unified memory를 하나의 메모리 예산으로 읽는 Archify 다이어그램](../assets/archify-unified-memory.svg)

## 2.1 이 장에서 먼저 답할 질문

| 질문 | 먼저 볼 개념 | 잘못 읽기 쉬운 표현 |
|---|---|---|
| 128GB가 모두 모델 weight에 쓰이는가? | weight·KV cache·workspace·OS headroom | “128GB 모델 지원” |
| 큰 모델인데 왜 느린가? | active parameter보다 memory bandwidth와 backend | “파라미터가 크면 무조건 더 똑똑하고 느리다” |
| 긴 context를 늘리면 무엇이 커지는가? | KV cache·workspace·prefill 시간 | “max context를 설정하면 모두 usable하다” |
| 두 대를 연결하면 무엇이 늘어나는가? | TP·PP·DP·통신 경로 | “메모리 2배 = 속도 2배” |
| 메모리 숫자가 왜 도구마다 다른가? | UMA와 swap·프로세스별 보고 | “`nvidia-smi` 숫자가 0이면 GPU를 못 쓴다” |

NVIDIA 공식 문서도 DGX Spark의 UMA에서는 전용 framebuffer가 없기 때문에 일반 GPU와 메모리 표시가 다를 수 있고, `cudaMemGetInfo`만으로 실제 할당 가능 메모리를 판단하지 말라고 설명한다. [DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

## 2.2 하나의 128 GiB 풀로 생각한다

DGX Spark의 GB10 환경에서는 모델 weight만 GPU 메모리에 넣고 계산을 끝내면 안 된다. 다음 자원이 하나의 시스템 memory budget을 함께 사용한다.

```text
unified memory budget
  = model weights
  + KV cache
  + CUDA workspace / graph
  + tokenizer / runtime / framework
  + OS / desktop / agent sidecars
  + safety headroom
```

`weight size < 128 GiB`는 시작 조건일 뿐이다. context와 concurrency를 늘리면 KV cache가 커지고, FlashInfer·CUDA graph·vision encoder가 순간적으로 추가 workspace를 요구할 수 있다.

## 2.3 모델 크기보다 active parameter와 KV가 중요할 때

MoE 모델은 total parameter가 커도 각 token을 처리할 때 일부 expert만 활성화할 수 있다. 이 때문에 Spark 포럼에서는 Qwen3.5, DeepSeek, GLM, MiniMax, Nemotron 같은 MoE 모델이 반복해서 다뤄진다.

하지만 다음을 혼동하지 않는다.

- `total parameters`: checkpoint와 weight memory에 영향을 줌
- `active parameters`: token당 계산량에 영향을 줌
- `KV heads / head dimension`: context와 KV memory에 영향을 줌
- `vision/audio encoder`: 멀티모달 요청 순간의 추가 workspace
- `draft model`: speculative decoding의 추가 memory와 acceptance

특히 모델 weight만 메모리에 들어가고 KV를 위한 공간이 남지 않는다면, 이를 “실행 가능”이라고 부르기 어렵다. 정확히는 “로드 데모가 가능하다”에 가깝다.

## 2.4 메모리 압박을 측정하는 값

모델을 시작할 때 다음 값을 기록한다.

| 시점 | 기록할 값 |
|---|---|
| boot | OS, driver, kernel, idle memory, idle temperature |
| after load | weight load 완료 후 free memory, GPU/runtime allocation |
| first request | first token 전 memory peak와 TTFT |
| long context | context 길이, KV 사용량, free memory |
| concurrency | c1/c2/c4/c8별 peak memory와 error |
| soak | 2–8시간 후 memory fragmentation, temperature, restart 여부 |

포럼의 [memory full crash 사례](https://forums.developer.nvidia.com/t/system-crashes-when-memory-is-full/352339)와 [stability/OOM/overheating 사례](https://forums.developer.nvidia.com/t/dgx-spark-stability-out-of-ram-overheating/368536)는 메모리가 부족할 때 요청 하나만 실패하는 것이 아니라 SSH·HDMI·OS까지 영향을 받을 수 있음을 보여준다. 그러므로 safety headroom은 성능을 포기한 공간이 아니라 운영에 필요한 자원으로 취급한다.

## 2.5 TP·PP·DP를 구분한다

| 방식 | 무엇을 나누나 | 장점 | 단점 | Spark에서의 기본 판단 |
|---|---|---|---|---|
| TP | 한 layer의 tensor/weight | 한 모델의 latency와 용량을 함께 확장 | 매 layer 통신, head divisibility 필요 | 2대·4대에서 먼저 검토 |
| PP | layer 묶음 | TP divisibility가 안 되는 큰 모델도 수용 | pipeline bubble·latency | 3대 실험 경로 |
| DP | 모델 복제 | 동시성·장애 격리 | 모델 용량은 늘지 않음 | 작은 모델 여러 agent에 유리 |
| EP | expert를 노드에 배치 | MoE routing 활용 | routing/all-to-all 복잡 | 모델별 recipe가 있을 때만 |

### TP가 자동으로 2배가 아닌 이유

두 노드가 각각 절반의 계산을 맡더라도 layer 사이마다 all-reduce가 필요하다. 통신이 느리거나 NCCL이 socket으로 떨어지면 계산량을 나눈 만큼 통신량이 늘어 single-stream 속도가 거의 개선되지 않을 수 있다. 반대로 여러 요청을 동시에 처리하면 aggregate throughput이 좋아질 수 있다.

[Qwen3.8 SGLang+DFlash2 듀얼 실측](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732)은 code에서 1대 52–61 → 2대 87 tok/s, prose에서 26 → 41 tok/s를 보고한다. 이 결과는 “2배”라는 단순한 규칙보다 workload별 통신·메모리·speculative 효율을 함께 봐야 한다는 점을 보여준다.

## 2.6 노드 수와 네트워크

| 노드 수 | 일반적인 연결 | 먼저 확인할 것 |
|---:|---|---|
| 1 | network 불필요 | local memory와 thermal |
| 2 | 200GbE QSFP/RoCE direct | CX-7 interface, `NET/IB`, NCCL health |
| 3 | ring 또는 switch | PP/DP/TP divisibility, OOB network |
| 4+ | dedicated switch/RoCE | RDMA passthrough, MTU, HCA mapping |

link가 200G로 negotiate됐다는 사실과 payload가 200G로 전송된다는 사실은 다르다. [12–13 Gbps로 제한됐다가 power drain 후 회복된 사례](https://forums.developer.nvidia.com/t/dgx-spark-200gbe-direct-qsfp-link-negotiates-200g-but-payload-is-12-gbps/373538)는 실제 inference를 시작하기 전에 `iperf3`, `ib_write_bw`, `nccl-tests`를 통과해야 하는 이유를 보여준다.

4대 SGLang 사례에서는 Docker에 `/dev/infiniband`를 넘기지 않아 socket transport를 사용했고, RDMA를 활성화하자 9.8 → 25.1 tok/s로 개선됐다고 보고했다. [RDMA 수정 사례](https://forums.developer.nvidia.com/t/glm-4-7-fp8-on-4x-dgx-spark-via-sglang-2-5x-speedup-8-2-25-tok-s-just-by-enabling-rdma/373675)는 `NCCL_DEBUG=INFO`에서 `via NET/IB`를 확인한다.

## 2.7 장비를 받자마자 기록할 정보

```bash
hostnamectl
uname -a
nvidia-smi
free -h
df -h
ip -br addr
ibv_devices
ibdev2netdev
```

다중 노드를 구성할 때는 모든 노드의 결과를 같은 파일 형식으로 보관한다.

```text
node, hostname, OS, kernel, driver, CUDA, NCCL, memory_total,
cx7_interface, roce_device, link_speed, MTU, switch_or_direct
```

이 정보가 없으면 나중에 “같은 모델인데 왜 속도가 다른가?”라는 질문에 답하기 어렵다.

## 2.8 안전한 기본값

다음 값은 모든 모델에 적용되는 정답이 아니라 실험을 시작하기 위한 기준이다.

- `max_model_len`은 필요한 값으로 시작하고 1M부터 잡지 않는다.
- `max_num_seqs`와 `max_num_batched_tokens`를 낮게 잡고 단계적으로 올린다.
- unified memory에 OS와 sidecar를 위한 여유를 남긴다.
- Docker/SGLang의 memory cap을 모델의 실제 workspace까지 고려해 설정한다.
- power·thermal workaround를 포럼 댓글에서 그대로 복사하지 않고, 공식 진단/RMA 경로를 먼저 확인한다.
- firmware update 전에는 현재 버전과 recovery/RMA 절차를 기록한다.

## 2.9 NVIDIA DGX Spark와 GB10 OEM

NVIDIA Founders Edition만 GB10을 사용하는 것은 아니다. Acer·ASUS·Dell·GIGABYTE·HP·Lenovo·MSI도 NVIDIA가 인증한 GB10 시스템을 제공한다. 공통 메모리와 ConnectX-7이 있어도 냉각·SSD·전원·펌웨어·지원이 다르므로, 벤더를 바꾸면 같은 recipe를 그대로 복사하지 않고 inventory와 c1/c4 benchmark를 다시 기록한다. 자세한 비교는 [01-2. DGX Spark·GB10 벤더 비교](01-2-gb10-vendor-comparison.md)에서 다룬다.

## 이 장의 검증 체크리스트

- [ ] weight, KV, workspace, OS headroom을 분리해 계산했다.
- [ ] TP/PP/DP 중 현재 목적에 맞는 방식을 선택했다.
- [ ] CX-7 링크가 실제 RDMA/NCCL 경로인지 확인했다.
- [ ] single-stream과 aggregate throughput을 구분했다.
- [ ] 각 노드의 OS·driver·CUDA·NCCL·NIC 정보를 보관했다.
