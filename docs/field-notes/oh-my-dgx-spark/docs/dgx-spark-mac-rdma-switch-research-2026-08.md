# DGX Spark·Mac 혼합 구성과 다중 Spark 스위치 리서치

조사일: **2026-08-21**

## 결론

DGX Spark와 Mac을 함께 사용할 수는 있다. 다만 “한 모델을 Spark와 Mac에 나누어 올리는 RDMA 클러스터”와 “각 장비에 모델을 따로 올린 뒤 API로 연결하는 혼합 시스템”은 전혀 다른 구성이다.

| 구성 | 가능성 | 판단 |
|---|---|---|
| Spark ↔ Mac, 일반 Ethernet/TCP·HTTP | 가능 | Mac을 UI·라우터·RAG·저장소·별도 모델 서버로 사용 |
| Mac을 Spark TP/PP/NCCL rank로 직접 참여 | 공식 검증 경로를 찾지 못함 | 하나의 CUDA 모델 병렬 그룹으로 계획하지 않음 |
| Mac ↔ Mac, Thunderbolt 5 RDMA/JACCL/MLX | 가능 | macOS 26.2 이상과 TB5, Apple MLX/JACCL 조합 |
| Mac에서 Linux CUDA Spark 작업을 SSH로 실행 | 가능 | Mac은 control/launcher, CUDA rank는 Linux Spark |
| Spark CX-7 RoCE ↔ Mac Thunderbolt RDMA를 같은 RDMA fabric으로 결합 | 미검증 | RoCE와 JACCL은 transport·backend·device model이 다름 |
| MCDMA 커뮤니티 프로토타입으로 Spark↔Mac USB-C 직접 메모리 전송 | 실험 단계 | Ash Hart가 RDMA semantics와 측정값을 보고했지만 공개 구현·독립 재현은 아직 확인하지 못함 |

## 1. Spark와 Mac의 네트워크 역할

DGX Spark의 다중 노드 경로는 CX-7 QSFP 포트, 200GbE RoCE, NCCL을 중심으로 구성된다. NVIDIA의 두 대 플레이북은 직접 QSFP 연결을 사용하며, 성능 가이드에서는 QSFP 연결의 raw RDMA bandwidth와 NCCL collective bandwidth를 별도로 측정한다. [Connect Two Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks), [NVIDIA performance benchmarking guide](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md)

Mac Studio의 기본 Ethernet은 10GbE RJ-45다. 2025년형 Mac Studio는 Thunderbolt 5 포트를 제공하지만, Apple이 설명하는 RDMA는 Thunderbolt 5 링크 위의 macOS 기능이다. [Apple Mac Studio 기술 사양](https://www.apple.com/uk/mac-studio/specs/)

따라서 혼합 구성에서는 네트워크를 두 영역으로 나누어 생각한다.

```text
관리망: Mac·Spark·스위치가 SSH/API/파일 공유로 연결
        일반 Ethernet 또는 Wi-Fi도 가능

계산망: Spark ↔ Spark만 CX-7 QSFP/RoCE/NCCL
        TP/PP 모델 병렬 통신 전용
```

Mac을 계산망에 연결했다고 해서 Mac이 Spark의 CUDA rank가 되는 것은 아니다.

## 1.1 MCDMA 커뮤니티 프로토타입은 별도 경로로 기록한다

Ash Hart는 Apple Silicon Mac과 DGX Spark를 USB-C로 연결하고, Metal 측 unified memory와 Spark의 CUDA 메모리 사이에서 registered memory, rkey, one-sided READ/WRITE, two-sided SEND/RECV를 사용했다고 보고했다. 게시물의 설명처럼 양쪽에서 같은 동작을 수행할 수 있다면 기존의 “Mac은 단순 control host”라는 구분을 넓힐 가능성이 있다. 다만 이번 조사에서 확인한 근거는 작성자의 [X 게시물](https://x.com/ashxhart/status/2089749434087227672?s=20)과 그 [재게시본](https://site.twstalker.com/ashxhart/status/2089749434087227672)이다.

작성자가 설명한 2×Spark 구성은 다음과 같다.

```text
Spark 1 ── CX-7 ── Spark 2
   │                    │
   └──── USB-C ──┐  ┌── USB-C ────┘
                 Mac Studio

Spark 1·2: prompt processing
Mac Studio: decode
```

게시물에 제시된 전송 측정값은 단일 링크 939MB/s, Mac에서 두 Spark로 동시 전송 1.80GB/s, 두 Spark에서 Mac으로 동시 전송 1.25GB/s, 왕복 지연 24μs, 소형 메시지 처리량 41k msg/s다. 작성자는 1×Spark와 1×Mac 조합도 동작한다고 설명했으며, 두 USB-C 링크를 사용하는 구성은 자신의 2×Spark 실험 방식이라고 구분했다.

이 결과는 매우 흥미롭지만, 현재 책에서는 다음과 같이 범위를 제한해 기록한다.

- MCDMA는 표준 RoCE·NCCL·MLX/JACCL 연결이 아니라 커뮤니티가 제시한 별도 USB-C 전송 프로토타입이다.
- NVIDIA 공식 문서는 Spark에 4개 USB Type-C 포트가 있다는 사실을 설명하지만, MCDMA 메모리 등록이나 CUDA↔Metal one-sided operation을 지원한다고 문서화하지 않는다. [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- 939MB/s는 약 7.5Gbps이므로 Spark의 200GbE CX-7 링크보다 훨씬 낮다. 따라서 이론적인 512GB 메모리 풀의 크기보다 실제 pipeline에서 activation, KV, weight를 얼마나 자주 이동하는지가 더 중요하다.
- 작성자가 말한 405B 및 Mac 메모리를 포함한 512GB 메모리 풀은 이론적인 확장 주장이다. 하나의 모델 weight와 KV가 실제로 Spark와 Mac에 함께 배치되어 정상 생성된 결과라고 기록하지 않는다.
- 게시물은 이번 주에 소스와 완전한 tokens/s benchmark를 공개하겠다고 예고했다. 공개 구현, byte-integrity 시험, 장시간 USB-C 안정성, prefill/decode 실측을 확인한 뒤에야 신뢰 등급을 올린다.

따라서 MCDMA는 “Spark와 Mac을 API로 묶는 일반적인 혼합 구성”과 “하나의 추론 pipeline 안에서 메모리를 직접 교환하는 실험 구성”의 중간에 있는 경로다. 표준 운영 경로는 계속해서 Spark의 CX-7/NCCL과 Mac의 별도 endpoint를 분리하는 방식으로 두고, MCDMA는 재현 가능한 실험 프로필로 관리한다.

### MCDMA를 검증할 때 필요한 최소 항목

1. 공개된 소스의 commit, 빌드 방법, 지원 macOS·DGX OS·kernel·driver를 고정한다.
2. 한 링크에서 고정 크기 buffer를 양방향으로 보내고, 전체 byte checksum과 memory corruption 여부를 확인한다.
3. 작은 메시지 latency와 1MB·64MB·1GB 이상의 큰 block throughput을 분리해 측정한다.
4. 한 Spark·한 Mac, 두 Spark·한 Mac의 동시 전송을 각각 실행하고, USB-C link reset과 오류 복구를 기록한다.
5. 실제 모델에서는 prefill, activation 교환, decode, KV 이동량을 분리해 비교한다. 단순 memcpy 속도를 tokens/s로 바꾸어 쓰지 않는다.
6. 표준 API 분할 구성과 같은 prompt·output·context에서 TTFT, decode tok/s, 전력, 온도, 실패율을 비교한다.

## 2. Mac의 RDMA는 무엇인가

Apple은 macOS 26.2부터 Thunderbolt 5 기반 RDMA를 지원한다고 설명한다. Apple MLX 문서는 JACCL을 이 RDMA transport 위에서 동작하는 collective backend로 소개하며, 여러 Mac의 분산 추론과 학습에 사용한다. [Apple WWDC26 MLX 분산 통신](https://developer.apple.com/videos/play/wwdc2026/233/), [MLX distributed documentation](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/distributed.rst)

이 경로의 조건은 DGX Spark RoCE와 다르다.

| 항목 | DGX Spark 클러스터 | Mac MLX 클러스터 |
|---|---|---|
| 장치 | GB10/SM121 + CX-7 | Apple Silicon GPU |
| 물리 링크 | QSFP, 200GbE | Thunderbolt 5 |
| RDMA/collective | RoCEv2·NCCL | Thunderbolt RDMA·JACCL |
| 주 프레임워크 | CUDA·vLLM·SGLang·TensorRT-LLM | MLX·MLX LM |
| topology | 2대 direct, 3대 ring, 4대 이상 switch 경로 | JACCL full mesh, ring backend는 TCP |
| Mac을 Spark rank로 사용 | 공식 근거 없음 | Mac끼리 MLX rank로 사용 |

Apple MLX 문서는 JACCL에 fully connected mesh가 필요하다고 명시한다. 네 대라면 모든 Mac 쌍을 Thunderbolt 케이블로 연결해야 하며, ring이 필요할 때는 JACCL이 아니라 TCP 기반 ring backend를 선택한다. Apple은 메시지 크기와 작업에 따라 mesh와 ring을 활용하는 예시도 보여주지만, JACCL 자체의 topology 요구사항과 ring backend는 구분해서 읽어야 한다. [MLX JACCL 문서](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/distributed.rst#defining-a-mesh)

Mac에서 RDMA 장치를 확인하는 MLX 문서의 절차에는 Recovery에서 `rdma_ctl enable`을 실행하고 재부팅한 뒤 `ibv_devices`를 확인하는 과정이 포함된다. 이 명령은 Mac의 Thunderbolt RDMA를 활성화하는 절차이지, Spark의 CX-7 RoCE를 설정하는 명령이 아니다.

## 3. Mac을 Spark와 함께 쓰는 현실적인 방법

### A. Mac은 control host

Mac에서 SSH로 Spark 노드에 접속하고 모델은 Spark에서 실행한다. MLX 문서도 Mac에서 Linux CUDA 노드로 작업을 launch할 때 NCCL backend를 선택하는 예시를 제공한다. 이때 host 목록은 Linux CUDA 노드로 구성되며 Mac이 CUDA rank가 되는 것은 아니다. [MLX의 CUDA/NCCL launch 예시](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/launching_distributed.rst#nccl-specifics)

적합한 역할:

- VS Code·터미널·웹 UI
- 실험 노트와 benchmark 결과 저장
- SSH·작업 시작·로그 수집
- 요청 라우팅과 인증
- RAG 문서 전처리·검색
- Mac의 MLX 모델과 Spark API endpoint를 묶는 gateway

### B. Mac은 별도 모델 서버

Mac에는 MLX/llama.cpp 모델을 별도로 실행하고 Spark에는 vLLM, SGLang, SparkInfer 모델을 실행한다. router가 요청 유형에 따라 endpoint를 선택한다.

```text
client
  → local router
      ├─ Mac MLX/llama.cpp: 빠른 worker·문서 전처리
      └─ Spark vLLM/SGLang/SparkInfer: CUDA·큰 supervisor·tool loop
```

이 구성은 API 수준에서 분산하는 방식이다. 두 장비의 unified memory를 합치지 않으며, 한 요청의 weight와 KV를 Spark와 Mac 사이에 나누지도 않는다.

### C. Mac 여러 대는 별도 Apple 클러스터

Mac만으로 MLX 클러스터를 만들 때는 TB5와 JACCL을 사용하고 Spark 클러스터에는 CX-7 QSFP와 NCCL을 사용한다. 두 클러스터는 관리망과 API gateway로 연결한다.

```text
Mac TB5/JACCL/MLX cluster ── API/router ── Spark QSFP/RoCE/NCCL cluster
```

두 클러스터를 하나의 TP 그룹으로 합친다고 기록하지 않는다. 품질, 지연, 장애 격리를 관리하기 쉬운 구조는 역할별 endpoint를 분리하는 방식이다.

## 4. 스위치가 필요한 시점

NVIDIA 현재 playbook의 기본 경로는 다음과 같다.

| Spark 수 | 공식/권장 연결 | 스위치 |
|---:|---|---|
| 1 | 없음 | 불필요 |
| 2 | QSFP direct 200GbE | 불필요. 한 개 QSFP baseline 가능 |
| 3 | QSFP 3개 direct ring | 불필요. 각 노드를 이웃 두 곳에 연결 |
| 4 | QSFP switch, 노드당 케이블 1개 | 공식 multi-Spark 경로에서 필요 |
| 6~8 | QSFP switch와 확장 port | 사실상 권장. direct mesh는 케이블·포트·운영 부담이 큼 |

NVIDIA의 3대 플레이북은 200GbE QSFP 3개를 사용하는 ring을 설명한다. 네 대 플레이북은 최소 4개의 QSFP56-DD port를 가진 200Gbps QSFP switch와 노드당 케이블 하나를 요구하며, 같은 switch에 더 많은 Spark를 연결할 수 있다고 안내한다. [3대 ring](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks), [4대 이상 switch](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch)

“여러 대면 무조건 스위치가 필요하다”는 설명은 정확하지 않다. 3대까지는 NVIDIA가 직접 ring을 지원한다. 반대로 4대부터는 공식 문서와 운영 측면에서 switch 구성이 가장 깔끔하다. full mesh를 직접 만들면 케이블 수가 `N × (N - 1) / 2`로 늘어나며, 4대는 6개, 8대는 28개가 필요하다. 이 계산은 topology가 요구하는 물리적 부담을 보여줄 뿐, 특정 runtime이 full mesh를 지원한다는 보장은 아니다.

## 5. 스위치 구성에서 확인할 것

스위치는 단순한 Ethernet hub가 아니다. NVIDIA 4대 playbook은 다음 조건을 요구한다.

- QSFP56-DD 또는 호환 200Gbps port
- 모든 Spark port가 같은 Layer-2 bridge/domain에 속함
- 각 port의 link speed가 200Gbps로 협상됨
- 필요하면 switch에서 auto-negotiation을 끄고 200G를 수동 설정
- 관리망과 CX-7 workload망을 분리
- Spark 쪽의 CX-7 interface·IP·MTU·SSH 설정
- NCCL sanity test와 실제 `NET/IB` 경로 확인

스위치 포트가 100Gbps로 협상되어도 link 자체는 올라올 수 있다. `ethtool`에서 200000Mb/s인지 확인하고 NCCL 로그에 socket fallback이 없는지도 확인한다. NVIDIA playbook은 switch port를 bridge에 넣어 단일 Layer-2 domain으로 만들고, 관리와 인터넷 트래픽은 Ethernet/Wi-Fi로 분리하는 방식을 권장한다. [NVIDIA multi-Spark switch README](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch)

## 6. 권장 조합

| 목표 | 권장 구성 |
|---|---|
| Spark 2대 + Mac 1대 | 표준 경로는 Spark 2대 direct QSFP/RoCE, Mac은 관리망·router. MCDMA는 별도 실험 프로필 |
| Spark 3대 + Mac | Spark 3대 QSFP ring, Mac은 control/router |
| Spark 4대 이상 + Mac | Spark는 QSFP switch, Mac은 management/API망 |
| Mac 여러 대 + Spark 여러 대 | Mac은 JACCL/MLX 별도 cluster, Spark는 NCCL 별도 cluster |
| 하나의 대형 CUDA 모델을 Spark+Mac에 분할 | 현재 공식 검증 없음. 계획에서 제외 |

## 7. 재현 체크리스트

### Mac

```bash
sw_vers
system_profiler SPThunderboltDataType
networksetup -listallhardwareports
```

Mac MLX RDMA cluster를 별도로 시험한다면 `ibv_devices`와 `mlx.distributed_config`를 사용한다. Spark 혼합 endpoint만 필요하면 Mac의 Ethernet IP와 SSH/API connectivity를 측정한다.

### Spark

```bash
ibdev2netdev
sudo ethtool <cx7-interface> | grep Speed
ip -br addr
nvidia-smi
```

다중 Spark benchmark를 실행하기 전에는 raw RDMA perftest, NCCL all-gather/all-reduce, 작은 TP 요청, 긴 context 순서로 검증한다. Mac과의 `iperf3` 또는 HTTP latency는 계산망 RDMA 결과와 별도의 표에 기록한다.

## 8. 조사 결론의 신뢰도

- Apple RDMA over Thunderbolt 5와 MLX/JACCL: Apple 공식 문서와 MLX 공식 문서에 근거한 사실
- Spark 2대 direct, 3대 ring, 4대 switch: NVIDIA 공식 playbook에 근거한 사실
- Spark+Mac 하나의 NCCL TP/PP 그룹: 공식 상호운용 recipe를 찾지 못한 미검증 가정
- MCDMA USB-C direct memory path: 작성자 측정값은 확인했지만 공개 구현과 독립 재현이 없는 커뮤니티 프로토타입
- Mac을 API/router/control host로 쓰는 구성: SSH·HTTP·일반 네트워크를 이용한 실전 권장안

마지막 항목은 구현 가능한 아키텍처 제안일 뿐, 특정 vendor가 인증한 성능 보장은 아니다.
