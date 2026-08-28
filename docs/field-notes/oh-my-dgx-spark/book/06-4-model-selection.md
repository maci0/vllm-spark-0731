# 06-4. DGX Spark에서 돌릴 모델 선택

상태: 리서치 기반 모델 선택 허브

기준일: **2026-08-23**

DGX Spark에서 모델을 고를 때는 파라미터 수나 한 번의 tok/s 숫자만 보지 않는다. 128GB unified memory 안에는 weight와 KV cache뿐 아니라 CUDA workspace, vision encoder, runtime, 운영체제와 endpoint가 함께 들어간다. 따라서 모델 선택은 다음 순서로 판단한다.

```text
작업 정의 → 메모리·노드 수 확인 → 실행 경로 선택 → benchmark → tool/agent 검증
```

이 장은 그 판단을 빠르게 시작하는 허브다. 상세 명령과 사례는 서브챕터와 관련 장으로 보낸다.

## 3분 이해 (ELI5)

모델 선택은 한 명의 천재를 고르는 일이 아니라 팀원을 역할에 맞게 배치하는 일이다.

```text
DeepSeek → 긴 계획·supervisor
Qwen     → 빠른 코드·UI worker
MiniMax  → tool-heavy coding 후보
H3       → 영상·음성 생성
```

노드 수와 검증 상태까지 맞아야 팀이 실제로 일한다.

![작업에 따라 DeepSeek·Qwen·MiniMax-H3 역할을 나누는 Archify 다이어그램](../assets/archify-model-roles.svg)

## 이 장에서 바로 고르기

| 우선순위 | 첫 후보 | 현실적인 시작 구성 | 다음 검증 |
|---|---|---|---|
| 한 대에서 빠른 코드·UI·JSON worker | Qwen3.8-27B | NVFP4 또는 FP8 + SGLang/vLLM speculative 경로 | [03-4](03-4-single-spark-first-model.md) → [05-1](05-1-benchmark-design.md) |
| 긴 문맥과 복잡한 코드·문서 supervisor | DeepSeek V4 Flash 0731 | 한 대 EXL3 recipe 또는 두 대 FP8·TP=2 recipe | [06-5](06-5-deepseek-v4-flash.md) → [06-1](06-1-quantization-speculative.md) |
| agentic coding과 다중 tool 실험 | MiniMax M2.7/M3 | 두 대 이상에서 공개 recipe를 조건 그대로 재현 | [06-7](06-7-minimax-m2-m3.md) → [07-1](07-1-two-spark-cluster.md) |
| 영상·음성 생성 | MiniMax-H3 | 한 대 GB10 + ComfyUI recipe | [06-8](06-8-minimax-h3.md) |
| Sol max와 공정하게 비교 | 작업별 local 후보 | 같은 task·harness·tool 조건을 고정 | [10-4](10-4-gpt56-sol-comparison.md) |

이 표는 최종 순위표가 아니다. “빠른 worker”, “긴 문맥 supervisor”, “영상 생성”은 서로 다른 workload이므로 하나의 tok/s 순위로 합치지 않는다.

## 작업별 선택의 기준

### 단일 Spark

단일 사용자나 개발 보조가 목적이면 먼저 Qwen3.8-27B를 검토한다. 긴 문맥과 복잡한 supervisor 역할이 더 중요하면 DeepSeek V4 Flash 0731의 EXL3 recipe를 검토한다. 두 경우 모두 `loads`에서 멈추지 말고 endpoint, 고정 prompt, tool schema를 순서대로 확인한다.

### 두 대

두 대를 사용한다고 모델 두 개를 동시에 올릴 수 있는 것은 아니다. 한 모델을 TP=2로 묶으면 모델 크기와 context 여유가 늘 수 있지만, CX-7·RoCE·NCCL·KV cache와 통신 비용을 함께 검증해야 한다. DeepSeek의 FP8·TP=2는 이 경로를 시험할 대표 후보이고, Qwen은 TP=2와 두 endpoint를 각각 비교할 수 있다.

### 세 대·네 대·여덟 대

세 대는 모델의 tensor-parallel divisibility와 topology가 선택을 좌우한다. 네 대부터는 `2 × 2`처럼 서로 독립된 endpoint를 구성할 수 있지만, DeepSeek와 Qwen을 각각 TP=2로 실행하면 네 대가 필요하다. 여덟 대는 개인용 첫 단계가 아니라 스위치·RDMA·전력·운영 자동화까지 포함하는 클러스터 문제로 다룬다.

| 노드 수 | 먼저 검토할 구성 | 반드시 확인할 것 |
|---:|---|---|
| 1 | Qwen worker 또는 DeepSeek EXL3 supervisor | weight·KV·workspace·단일 stream decode |
| 2 | DeepSeek TP=2, Qwen TP=2 또는 endpoint 두 개 | direct link/RoCE, NCCL, aggregate와 단일 요청 분리 |
| 3 | TP=3이 필요한 실험 모델 또는 2+1 역할 분리 | head·expert divisibility, PP/TP, 관리망 병목 |
| 4 | Qwen TP=2 + DeepSeek TP=2 또는 단일 대형 TP | 스위치, MTU, 장애 시 한 endpoint 복구 |
| 8 | 팀 서비스·동시성·대형 모델 | fabric, 전력·냉각, observability, 운영 비용 |

## 모델 유형을 먼저 구분한다

이 장의 네 서브챕터는 같은 종류의 모델을 나열한 것이 아니다.

| 서브챕터 | 모델 유형 | 대표 작업 | 속도 단위 |
|---|---|---|---|
| [06-5 DeepSeek](06-5-deepseek-v4-flash.md) | 대형 언어·에이전트 모델 | 긴 문맥, supervisor, 코드 | prefill·decode·aggregate tok/s |
| [06-6 Qwen](06-6-qwen38-27b.md) | dense 언어·멀티모달 모델 | 코드, UI, JSON, worker | 단일·aggregate tok/s와 tool 성공률 |
| [06-7 MiniMax](06-7-minimax-m2-m3.md) | sparse MoE 언어·에이전트 모델 | tool-heavy coding, 장기 작업 | recipe별 decode·prefill |
| [06-8 MiniMax-H3](06-8-minimax-h3.md) | 영상·음성 생성 모델 | ComfyUI 영상과 오디오 | step/clip 시간과 출력 품질 |

MiniMax-H3는 언어 모델의 decode tok/s 표에 넣지 않는다. 영상 한 편의 렌더 시간과 언어 모델의 token 생성 속도는 측정 대상과 품질 기준이 다르다.

## 공통 실행 프로필

모델마다 다음 항목을 같은 순서로 기록한다.

```text
model id / revision / license
nodes / topology / transport
weight quant / KV dtype / context
runtime / image / commit / parser
workload / prompt tokens / output limit / thinking setting
prefill / decode / TTFT / end-to-end / aggregate
tool schema / agent loop / error recovery / wall power
```

검증 상태는 [00장](00-how-to-read.md)의 정의를 따른다.

| 상태 | 이 책에서의 의미 |
|---|---|
| `loads` | weight와 runtime이 메모리에 올라온 상태 |
| `generates` | 고정 prompt에서 정상 출력이 나온 상태 |
| `serves` | endpoint가 health와 반복 요청에 응답한 상태 |
| `benchmarked` | 조건과 측정 방법을 고정한 raw 결과가 있는 상태 |
| `tool-tested` | parser·schema·arguments·오류 복구를 확인한 상태 |
| `agent-tested` | 여러 단계의 실제 tool loop를 통과한 상태 |
| `long-context-tested` | 지정 context에서 회수·품질·안정성을 확인한 상태 |

모델 카드의 “지원”, recipe의 “가능”, 커뮤니티의 “50 tok/s”는 위 상태를 자동으로 보장하지 않는다. 각 서브챕터에서 공식 사양, recipe 주장, 커뮤니티 실측, 이 저장소의 직접 실측을 분리해서 적는다.

## 모델 역할 분리

모델 하나에 모든 작업을 맡길 필요는 없다. 역할 분리는 모델을 동시에 한 장비에 억지로 적재한다는 뜻이 아니라, endpoint와 workload의 경계를 명확히 한다는 뜻이다.

```text
router
  ├─ 긴 문서·계획·복구·복잡한 tool loop → DeepSeek supervisor
  ├─ UI·CSS·짧은 코드·JSON 변환 → Qwen3.8 worker
  ├─ 장기 코딩·다중 tool 작업 → MiniMax 후보
  ├─ 영상·음성 생성 → MiniMax-H3 + ComfyUI
  └─ 이미지 이해·OCR → 실제로 vision encoder가 로드된 별도 endpoint
```

DeepSeek와 Qwen을 각각 TP=2로 실행하는 `2 × 2`는 네 대가 필요하다. 두 대에 두 모델을 함께 올리는 구성은 기본 recipe가 아니라 메모리·KV·통신·장애 복구를 새로 검증해야 하는 실험이다.

## 선택에서 실행으로 내려가기

1. [03-1](03-1-first-boot-safe-environment.md)에서 OS·driver·전력·디스크 상태를 확인한다.
2. [03-4](03-4-single-spark-first-model.md)에서 한 모델의 smoke test를 통과시킨다.
3. [04-1](04-1-engine-selection.md)과 [06-1](06-1-quantization-speculative.md)에서 runtime·quant·speculative 경로를 고정한다.
4. [05-1](05-1-benchmark-design.md)에서 prefill·decode·end-to-end·aggregate를 분리해 측정한다.
5. tool call과 실제 agent loop가 필요하면 [08-1](08-1-local-agent-operations.md)으로 이동한다.
6. 노드를 늘릴 때는 [07-1](07-1-two-spark-cluster.md)과 [07-4](07-4-multi-spark-scaling.md)을 먼저 읽고, [09-1](09-1-operations-failure-recovery.md)에서 복구 절차를 준비한다.

## 서브챕터 안내

- [06-5. DeepSeek V4 Flash 0731](06-5-deepseek-v4-flash.md): 단일 Spark EXL3와 두 대 FP8·TP=2의 차이, C1·tool loop 상태
- [06-6. Qwen3.8-27B](06-6-qwen38-27b.md): 단일 worker, parser·speculative decoding, Qwen3.5 architecture 표시 해석
- [06-7. MiniMax M2.7과 M3](06-7-minimax-m2-m3.md): 2·3·4대 recipe와 라이선스·custom kernel 조건
- [06-8. MiniMax-H3](06-8-minimax-h3.md): 단일 Spark ComfyUI 영상·음성 recipe와 `sm_121` 커널 주의점

상세 사례의 출처와 재현 큐는 [06-9 DeepSeek 사례](06-9-deepseek-community-builds.md), [06-12 Qwen 사례](06-12-qwen38-community-builds.md), [부록 B 리서치 로그](appendix-b-research-log.md)에서 확인한다.

## 결론

단일 Spark에서 시작하는 일반적인 순서는 Qwen3.8 worker 또는 DeepSeek EXL3 supervisor다. 두 대부터는 DeepSeek TP=2와 Qwen TP=2·endpoint 분리를 비교하고, 세 대 이상에서는 topology와 통신을 모델 선택의 일부로 본다. MiniMax는 agentic coding의 실험 후보이며, MiniMax-H3는 별도의 영상·음성 경로다.

최종 선택은 “어떤 모델이 가장 빠른가”가 아니라 **내 작업이 요구하는 품질·context·tool loop·운영 비용을 어느 recipe가 재현 가능한 조건으로 만족하는가**로 결정한다.

## 체크리스트

- [ ] 작업을 worker, supervisor, tool-heavy agent, vision, 영상·음성 중 하나로 분류했다.
- [ ] 노드 수와 TP·PP·DP·endpoint 분리 중 선택 이유를 적었다.
- [ ] weight, KV dtype, context, runtime과 commit을 기록했다.
- [ ] `loads`와 `serves`, `benchmarked`, `tool-tested`, `agent-tested`를 구분했다.
- [ ] 단일 요청 속도와 aggregate throughput을 같은 값처럼 비교하지 않았다.
- [ ] community claim과 직접 실측을 별도 행으로 적었다.
- [ ] 라이선스와 장시간 운영·장애 복구 조건을 확인했다.

## 아직 모르는 것

- [ ] 같은 prompt·runtime·품질 기준으로 Qwen·DeepSeek·MiniMax를 직접 비교한 공개 표는 없다.
- [ ] 2대·3대·4대·8대의 모든 조합에서 context와 tool loop가 선형으로 확장되는지 확인되지 않았다.
- [ ] 단일 Spark에서 두 언어 모델을 동시에 서비스할 때의 안정적인 KV·workspace profile은 recipe별 검증이 필요하다.
- [ ] MiniMax-H3의 렌더 품질과 커널 최적화가 다른 PyTorch·CUDA 버전에서도 같은지 재현이 필요하다.

## 참고

- [DGX Spark 모델 선택 리서치](../docs/dgx-spark-model-selection-research-2026-08.md)
- [01-2. DGX Spark·GB10 벤더 비교](01-2-gb10-vendor-comparison.md)
- [05-1. 벤치마크를 제대로 설계하기](05-1-benchmark-design.md)
- [08-1. 로컬 에이전트 운영](08-1-local-agent-operations.md)
