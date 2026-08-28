# 07. 여러 Spark와 네트워크

Spark를 여러 대 연결해도 메모리가 자동으로 하나의 큰 풀로 합쳐지는 것은 아닙니다. 먼저 **한 모델을 나눌지(TP·PP), 같은 모델을 복제할지(DP), 모델별 endpoint를 분리할지**를 결정한 뒤 네트워크를 설계해야 합니다.

![여러 Spark의 연결 형태](../assets/archify-multi-spark.svg)

## 노드 수별 기본 경로

NVIDIA의 공식 플레이북은 두 대의 직접 QSFP 연결, 세 대의 링 연결, 네 대 이상을 위한 스위치 연결을 각각 설명합니다. 이 구조는 네트워크 배선의 출발점이지, 모든 모델과 모든 runtime에서 같은 성능을 보장하는 표가 아닙니다.

| 노드 수 | 먼저 검토할 구성 | 네트워크 형태 | 주의할 점 |
|---:|---|---|---|
| 1대 | TP=1 또는 독립 endpoint | CX-7 연결 불필요 | 모델·KV cache·동시성을 위한 메모리 여유를 확인합니다. |
| 2대 | TP=2 또는 두 endpoint | 200GbE QSFP 직접 연결 | NCCL/RoCE와 OOB 관리망을 모두 확인합니다. |
| 3대 | PP=3, DP=3 또는 2+1 | 3-way QSFP 링 | `TP=3`을 자동 기본값으로 두지 않습니다. |
| 4대 | TP=4, DP=4 또는 2×2 pool | QSFP 스위치 | 스위치, MTU, firmware, port speed을 함께 고정합니다. |
| 6~8대 | TP·PP·DP 조합 | 스위치 fabric | 모델별 launcher와 장애 범위를 다시 검증합니다. |

참고: [Connect Two Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks), [Connect Three Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks), [Multiple Sparks Through a Switch](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch).

## TP·PP·DP를 구분한다

- **Tensor parallelism(TP)**은 한 레이어의 계산과 weight를 여러 노드에 나눕니다. 단일 모델의 메모리 요구량을 낮출 수 있지만, decode 중 collective 통신이 반복됩니다.
- **Pipeline parallelism(PP)**은 모델의 레이어 구간을 노드에 나눕니다. 큰 모델을 수용하는 데 유리하지만, stage 사이의 pipeline bubble과 요청 스케줄링을 측정해야 합니다.
- **Data parallelism(DP)**은 같은 모델을 여러 노드에 복제하고 요청을 분산합니다. 모델 하나의 최대 크기를 늘리지는 않지만, 여러 요청을 처리하고 한 노드의 장애를 격리하기 쉽습니다.

따라서 “Spark 두 대면 메모리 256GB”라고 쓰지 않습니다. 정확한 표현은 “두 노드에 모델을 분할하거나 두 개의 독립 인스턴스를 운영할 수 있다”입니다. 실제로 어떤 메모리가 공유되는지는 runtime과 병렬화 설정에 달려 있습니다.

## 두 대: DeepSeek TP=2와 독립 endpoint

두 대는 DeepSeek V4 Flash 0731을 TP=2로 실행하는 경로가 가장 뚜렷한 확장 지점입니다. 공개 레시피는 256K 또는 1M profile, DSpark, NVFP4 KV와 같은 별도 조건을 사용합니다. 이 수치는 단일 Spark EXL3 레시피와 섞어 쓰지 않습니다.

두 대를 다음처럼 사용할 수도 있습니다.

| 구성 | 장점 | 비용 |
|---|---|---|
| DeepSeek TP=2 | 한 모델의 context·capacity를 늘릴 수 있습니다. | 통신과 NCCL 장애가 요청 경로에 들어옵니다. |
| Qwen endpoint 2개 | 요청을 나누고 장애를 격리하기 쉽습니다. | 한 요청에서 모델 메모리를 합치지 않습니다. |
| DeepSeek 1대 + Qwen 1대 | supervisor와 worker를 분리합니다. | 두 모델의 품질·역할 라우팅을 따로 평가해야 합니다. |

### 두 대 연결 전 확인

1. 관리망에서 각 노드의 hostname, SSH, 시간과 software revision을 맞춥니다.
2. QSFP 포트와 케이블을 공식 playbook의 포트 번호에 맞춰 연결합니다.
3. NCCL이 사용할 인터페이스와 OOB 통신 경로를 고정합니다.
4. 작은 all-reduce 또는 공식 성능 테스트를 먼저 실행합니다.
5. 그 다음 짧은 TP 요청, 긴 context, soak 순서로 올라갑니다.

QSFP 링크가 올라왔다는 사실만으로 NCCL이 올바른 경로를 사용한다고 판정하지 않습니다. NCCL 로그에서 socket fallback, 잘못된 interface, 낮은 link speed를 확인하고, raw RDMA bandwidth와 NCCL collective bandwidth를 별도 결과로 기록해야 합니다. [NVIDIA 성능 측정 가이드](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md)를 기준으로 삼습니다.

## 세 대: ring은 가능하지만 목적을 먼저 정한다

NVIDIA는 세 대를 3-way 링으로 연결하는 절차를 제공합니다. 그렇다고 세 대가 항상 단일 모델의 `TP=3`으로 동작하거나 두 대보다 빠른 것은 아닙니다.

현실적인 선택은 다음과 같습니다.

- `PP=3`: 두 대에 들어가지 않는 모델을 수용하는 용도입니다. single-stream decode와 pipeline bubble을 측정해야 합니다.
- `DP=3`: 같은 worker를 세 대에 복제해 동시성을 높이는 용도입니다.
- `2+1`: 두 대를 DeepSeek TP=2로 쓰고, 남은 한 대에 Qwen worker를 별도로 올리는 용도입니다.

세 대에서 모델을 하나로 묶기 전에 해당 runtime이 3-way tensor parallelism을 실제로 지원하는지, checkpoint의 shard 수가 맞는지, collective가 ring topology를 사용하는지 확인합니다.

## 네 대 이상: switch를 구성 요소로 본다

네 대부터는 각 노드를 서로 직접 연결하는 대신 QSFP 스위치를 사용하는 편이 관리하기 쉽습니다. 스위치가 필요하다는 말은 “아무 200GbE 스위치나 연결하면 된다”는 뜻이 아닙니다.

구매·설치 전에 다음을 확인합니다.

- Spark당 필요한 QSFP 포트 수와 스위치의 실제 port 수
- 200Gbps 협상 여부와 케이블·transceiver 호환성
- 스위치 firmware와 Spark의 driver·DGX OS 조합
- 관리망과 계산망의 분리 방식
- MTU, bridge, IP 주소, NCCL interface mapping
- 전원·소음·랙 공간·고장 시 우회 경로

`ethtool`에서 링크가 올라온 것만으로 실제 payload가 200Gbps라고 기록하지 않습니다. link speed, raw transport, NCCL collective를 차례로 측정해야 합니다. 일부 환경에서는 링크가 100Gbps로 협상되어도 연결 자체는 정상처럼 보일 수 있습니다.

## 2×2 pool은 TP=4가 아니다

DeepSeek를 supervisor로 유지하고 Qwen3.8-27B를 UI·디자인 worker로 분리하려면 Spark가 총 네 대 필요합니다.

| pool | 모델 | 병렬화 | 역할 |
|---|---|---|---|
| A | DeepSeek V4 Flash 0731 | 2대 TP=2 | 긴 문맥과 supervisor |
| B | Qwen3.8-27B | 2대 TP=2 또는 독립 worker | 코드·UI·JSON 반복 작업 |

이것은 TP=4 단일 모델도 아니고, 두 pool의 unified memory가 합쳐지는 구성도 아닙니다. router가 두 endpoint를 선택하는 **서비스 구성**입니다. 따라서 전체 latency에는 router, 두 endpoint의 queue, tool 결과 전달 시간이 포함됩니다.

## Mac과 Spark를 섞을 때

Mac의 Metal unified memory와 Spark의 CUDA memory를 USB-C로 직접 연결하는 MCDMA는 커뮤니티가 제시한 실험 프로토타입입니다. 작성자는 단일 링크 약 939MB/s, 왕복 지연 약 24µs 등의 값을 보고했습니다. 그러나 조사 기준으로 NVIDIA의 표준 RoCE·NCCL 경로이거나 독립적으로 재현된 일반 레시피로 확인되지는 않았습니다.

따라서 운영 기본값은 다음처럼 분리합니다.

```text
Spark ↔ Spark : CX-7 / QSFP / RoCE / NCCL 계산망
Mac ↔ Spark   : Ethernet 또는 API gateway 관리·서비스망
Mac ↔ Mac     : Thunderbolt 5 / MLX / JACCL 별도 경로
```

MCDMA를 시험하려면 byte integrity, 장시간 전송, link reset, 한 Spark·한 Mac과 두 Spark·한 Mac의 동시 전송, 실제 prefill/decode end-to-end 결과를 별도로 남겨야 합니다. “USB-C로 RDMA가 된다”는 설명만으로 Spark와 Mac을 하나의 TP fabric으로 기록하지 않습니다.

근거: [DGX Spark와 Mac·RDMA·switch 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-mac-rdma-switch-research-2026-08.md), [MCDMA 원 게시물](https://x.com/ashxhart/status/2089749434087227672?s=20).

## 이 장의 완료 기준

- [ ] TP·PP·DP 중 어떤 목적의 병렬화인지 썼습니다.
- [ ] 직접 연결·ring·switch 중 물리 topology를 기록했습니다.
- [ ] 링크 speed와 NCCL collective를 따로 측정했습니다.
- [ ] 2×2 pool을 TP=4와 혼동하지 않았습니다.
- [ ] MCDMA를 표준 RDMA 운영 경로가 아닌 실험 항목으로 표시했습니다.

## 더 자세히 읽기

- 07-1. 두 대 연결하기
- 07-2. 두 대 토폴로지와 사전 점검
- 07-3. DeepSeek TP=2 실행
- 07-4. 세 대·네 대·여덟 대
- 07-5. 세 대와 네 대
- 07-6. 여섯·여덟 대와 확장 벤치마크
