# 07-1. 두 대 연결하기

상태: 리서치 기반 초안

두 번째 DGX Spark를 추가한다고 GPU 메모리 128 GiB만 늘어나는 것은 아니다. 두 노드 사이의 통신, 동기화, KV 분할이 새로운 병목이 될 수 있다. 따라서 모델을 실행하기 전에 링크와 NCCL을 별도로 검증한다.

## 3분 이해 (ELI5)

두 대 Spark는 사람 둘이 한 일을 나누는 것과 같다.

```text
요청 → Spark A ↔ 고속 링크 ↔ Spark B → 결과
```

두 사람이 빨라지는 대신 서로 자주 대화해야 하므로 링크와 분담 방식이 성능을 결정한다.

![여러 Spark가 링크와 스위치로 일을 나누는 Archify 다이어그램](../assets/archify-multi-spark.svg)

## 8.1 두 대가 해결하는 문제

| 목표 | 두 대의 이점 | 대가 |
|---|---|---|
| 더 큰 모델 | TP/PP로 weight를 나눌 수 있음 | 모든 token 단계에서 통신 필요 |
| 긴 context | KV pool과 headroom 증가 | context가 길수록 메모리·통신 pressure 증가 |
| 여러 agent | DP로 독립 endpoint를 두거나 TP 모델을 공유 | 서비스 라우팅·장애 격리 필요 |
| 단일 요청 속도 | 맞는 모델·backend에서는 scaling 가능 | 2배가 보장되지 않음 |

Qwen3.8 SGLang+DFlash2 공개 측정에서는 code workload가 1대 52–61 tok/s에서 2대 87 tok/s로, prose workload가 26 tok/s에서 41 tok/s로 증가했다. 다른 vLLM+MTP 측정은 single 22.6 tok/s, c4 aggregate 75.0 tok/s, c8 aggregate 116.1 tok/s를 기록했다. workload와 engine이 다르므로 이 결과를 하나의 순위표로 합치지 않는다. 상세한 원본·recipe·파생 모델 구분은 [06-12: Qwen3.8-27B로 사람들이 만든 것](06-12-qwen38-community-builds.md)과 [Qwen3.8 커뮤니티 리서치](../docs/qwen38-community-builds-2026-08.md)에서 확인한다. [Qwen3.8 듀얼 측정](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732), [vLLM+MTP 측정](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350)

## 8.2 토폴로지 선택

NVIDIA 플레이북은 두 Spark를 직접 QSFP로 연결하는 방식과 스위치를 거쳐 연결하는 방식을 구분한다.

| 구성 | 장점 | 주의점 |
|---|---|---|
| 직접 QSFP/RoCE | 장비가 적고 경로가 짧음 | 케이블·인터페이스 이름·MTU·RDMA 설정을 직접 맞춤 |
| QSFP 스위치 | 3대 이상으로 확장하기 쉬움 | 스위치·펌웨어·포트·전력·비용이 추가됨 |
| 일반 Ethernet만 사용 | 초기 관리·SSH가 쉬움 | tensor parallel 통신이 socket으로 떨어질 수 있음 |

구성할 때는 [NVIDIA connect-two-sparks 플레이북](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks)과 [multi-Spark switch 플레이북](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch)을 함께 참고한다.

## 8.3 연결 전 양쪽 노드의 조건

두 노드에서는 다음 조건을 동일하게 맞춘다.

- DGX OS·kernel·NVIDIA driver·CUDA·NCCL 계열
- container image와 digest
- vLLM/SGLang commit 또는 패키지 버전
- model·tokenizer revision
- 시간·호스트명·노드 rank
- MTU·HCA·interface 이름
- 방화벽·필요 포트·SSH 접근

각 노드에서 다음을 저장한다.

```bash
hostnamectl
uname -a
nvidia-smi
ip -br addr
ip -br link
ethtool <qsfp-interface> 2>/dev/null || true
rdma link 2>/dev/null || true
ibstat 2>/dev/null || true
```

`Link detected: yes`는 물리 링크가 올라왔다는 뜻일 뿐이다. RDMA가 실제 통신에 사용되는지는 NCCL 로그와 별도의 bandwidth test로 확인한다.

## 8.4 검증 순서

두 노드 실험은 다음 상태를 차례로 통과해야 한다.

```text
물리 link up
  → IP/MTU/interface 확인
  → RDMA device 확인
  → NCCL communicator 생성
  → all-reduce/bandwidth 통과
  → 작은 모델 TP 요청
  → 긴 context·동시성·soak
```

앞 단계가 실패하면 다음 단계의 모델 수치를 기록하지 않는다. NCCL이 socket fallback으로 실행된 상태에서 “2대 성능”이라고 기록하면 실제 topology를 숨기게 된다.

NCCL 로그에는 통신 경로가 `NET/IB`인지, 의도하지 않은 `NET/Socket`인지 기록한다. SGLang Docker에서 `/dev/infiniband`를 전달하지 않아 socket 경로로 내려갔다가 RDMA passthrough를 설정한 뒤 성능이 개선된 사례도 있다. [RDMA 개선 사례](https://forums.developer.nvidia.com/t/glm-4-7-fp8-on-4x-dgx-spark-via-sglang-2-5x-speedup-8-2-25-tok-s-just-by-enabling-rdma/373675)

## 8.5 DeepSeek TP=2 프로필

DeepSeek V4 Flash 0731을 2대에서 실제로 서비스한 커뮤니티 레시피와, 1M context·agent aggregate·vision·동시 영상 생성 사례는 [06-9: DeepSeek V4 Flash 0731로 사람들이 만든 것](06-9-deepseek-community-builds.md)과 [커뮤니티 제작물 원문 리서치](../docs/deepseek-v4-flash-0731-community-builds-2026-08.md)를 함께 참고한다. 이 절의 네트워크·기동 순서는 특정 recipe의 조건이고, 모든 0731 checkpoint에 자동으로 적용되는 공통 명령은 아니다.

두 대에서 DeepSeek V4 Flash 0731을 실행하는 방식은 현재 가장 명확한 TP=2 사용 사례 중 하나다.

| 프로필 | context | 목적 | 기록할 조건 |
|---|---:|---|---|
| 256K | 262,144 | 코딩·일반 서비스·상대적으로 높은 동시성 | TP=2, DSpark, KV dtype, c1/c4 |
| 1M | 1,048,576 | 긴 단일 문서·낮은 동시성 | KV pool, preemption, UCX, long-context quality |

공개 recipe에는 single-stream 약 37–40 tok/s와 aggregate 약 100–150 tok/s 범위의 결과가 포함되어 있다. 다만 모델 revision, fork, MTP, prompt, 동시성이 서로 다르다. `max_model_len`을 1M으로 선언할 수 있다는 것과 1M context에서 높은 decode 성능과 retrieval 품질을 보장한다는 것은 다른 문제다. [NVIDIA Developer Forum TP=2 recipe](https://forums.developer.nvidia.com/t/guide-deepseek-v4-flash-on-2x-dgx-spark-gb10-reproducible-vllm-serving-recipe-up-to-1m-token-context/374742)

## 8.5a 2×Spark pool 두 개를 함께 운영하는 구성

DeepSeek를 에이전트의 supervisor로 고정하고 Qwen3.8-27B를 UI·디자인 worker로 사용하려면 “둘 다 2×Spark에서 실행한다”는 표현을 정확히 해석해야 한다. 각 모델에 두 대씩 배정하는 구성이라면 전체 장비는 네 대다.

```text
Pool A: Spark 1 ── CX-7/RoCE ── Spark 2
        DeepSeek V4 Flash 0731, TP=2, supervisor
        DS4 native :8888
        optional vision shim :8899 → eyes :8081 → :8888

Pool B: Spark 3 ── CX-7/RoCE ── Spark 4
        Qwen3.8-27B, TP=2, UI·디자인 worker
        separate OpenAI-compatible endpoint

router → model role routing → OpenAI-compatible endpoint
```

이 구성은 `TP=4`인 하나의 모델이 아니다. 메모리와 장애 범위를 각각 갖는 독립적인 `TP=2` 서비스 두 개를 운영하는 방식이다. 각 pair를 직접 연결한다면 두 pair 사이에 계산용 스위치는 필요하지 않다. 네 대를 하나의 클러스터 fabric으로 묶거나 두 pair 사이의 라우팅과 관리망을 통합하려면 NVIDIA의 200GbE QSFP switch 구성을 기준으로 네트워크를 다시 설계한다.

역할은 다음과 같이 나누어 시작한다.

| 요청 성격 | 기본 endpoint | 이유 |
|---|---|---|
| 계획 수립, 긴 문맥, tool loop, 복구 판단 | DS4 `:8888` | supervisor 역할과 DeepSeek 장문 profile을 유지 |
| 화면 구성, CSS·컴포넌트, 시각적 대안, 빠른 수정 | Qwen3.8-27B | UI·디자인 worker로 별도 latency와 동시성을 측정 |
| DS4에 이미지 입력 | DS4 vision shim `:8899` | 이미지 caption을 만든 뒤 기존 DS4로 전달 |
| OCR, 작은 객체, 정밀한 공간 추론 | 별도 native VLM endpoint | DS4 shim의 caption 손실을 피함 |

연결된 [DS4 vision 저장소](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)는 DS4를 재배포하지 않고 `:8899` shim과 `:8081`의 소형 VLM을 추가한다. 기본 “eyes” 모델은 `Qwen3.5-0.8B-MLX-8bit`이다. 따라서 이 저장소를 Qwen3.8-27B의 native vision 지원 근거로 사용하지 않는다. 이 이미지 경로는 pixel을 DS4에 직접 전달하지 않고, caption을 만든 뒤 그 결과를 텍스트로 전달한다.

전체 Spark가 두 대뿐이라면 DS4 TP=2와 Qwen3.8-27B TP=2를 독립 서비스로 동시에 운영할 수 있다고 가정해서는 안 된다. 두 모델을 같은 두 노드에 나누어 올리는 혼합·공유 메모리 구성에는 별도의 runtime 지원과 메모리·통신 측정이 필요하다. 이 책에서는 기본 구성을 `2×Spark DS4` 또는 `2×Spark Qwen3.8` 중 하나를 선택하는 방식으로 두고, `2×2` 역할 분리는 네 대 프로필로 기록한다.

## 8.6 TP와 DP를 먼저 결정한다

| 방식 | 모델 메모리 | 요청 처리 | 적합한 경우 |
|---|---|---|---|
| TP=2 | 하나의 모델을 두 노드에 분할 | 한 요청을 공동 처리 | 한 노드에 안 들어가는 모델·긴 context |
| DP=2 | 각 노드에 모델 복제 | 요청을 두 endpoint로 분산 | 이미 한 노드에 들어가는 모델·동시성 |
| 2+1의 일부 | 두 노드 TP와 독립 서비스 조합 | 역할별 라우팅 | supervisor와 worker를 분리 |

Qwen3.8, Qwen3.6, Qwen3.5-122B처럼 한 노드에 들어가는 모델은 TP로 묶기보다 DP 또는 서비스 분리를 선택하는 편이 aggregate throughput과 장애 격리에 유리할 수 있다. 먼저 모델을 더 크게 실행하려는지, agent 수를 늘리려는지 결정한다.

## 8.7 긴 부하에서의 UCX·메모리 주의

DeepSeek TP=2에 장시간 부하를 걸면 unified memory와 RDMA memory registration이 함께 메모리 pressure를 만들 수 있다. 공개 재현 자료에서 제시한 UCX 환경 변수는 특정 recipe에서 누수를 완화하기 위한 조치다.

```text
UCX_MEM_MMAP_HOOK_MODE=none
UCX_RCACHE_MAX_UNRELEASED=1024
```

이 값을 모든 시스템에 적용할 수 있는 공식 해결책으로 간주하지 않는다. 적용 전과 후에 다음 항목을 비교한다.

- 30분·2시간 soak의 memory peak
- 요청 성공률과 timeout
- node free memory 회수 여부
- RDMA 경로 유지 여부
- 재시작 후 다시 연결되는가

## 8.8 두 대 첫 실행 절차

1. 두 노드의 inventory와 version을 저장한다.
2. QSFP link·MTU·RDMA를 확인한다.
3. NCCL bandwidth/all-reduce를 작은 설정으로 통과시킨다.
4. 짧은 prompt와 작은 output의 TP 요청을 보낸다.
5. `models`, 한국어, JSON, 멀티턴, tool parser를 확인한다.
6. 256K profile에서 c1/c4를 측정한다.
7. 1M profile은 별도 실행·별도 memory/quality 결과로 기록한다.
8. 장시간 soak 후 노드별 로그와 온도를 보존한다.

## 이 장의 검증 체크리스트

- [ ] 두 노드의 OS·driver·CUDA·NCCL·container를 고정했다.
- [ ] 물리 link와 RDMA를 NCCL 경로와 분리해 확인했다.
- [ ] `NET/IB` 또는 실제 사용 transport를 로그로 확인했다.
- [ ] TP=2와 DP=2 중 목적에 맞는 방식을 선택했다.
- [ ] DeepSeek 256K와 1M을 별도 profile로 기록했다.
- [ ] c1과 aggregate를 분리했다.
- [ ] UCX workaround를 recipe-specific 조치로 표시했다.
- [ ] 긴 부하 후 memory·temperature·error·restart를 확인했다.

## 아직 모르는 것

- 동일한 DeepSeek checkpoint와 prompt에서 1대 EXL3와 2대 FP8의 품질 차이
- 직접 연결과 스위치 연결의 동일 하니스 scaling 차이
- 1M에서 여러 agent가 동시에 사용할 때의 실제 KV headroom
- UCX 설정이 driver·NCCL 버전별로 재현되는 범위
