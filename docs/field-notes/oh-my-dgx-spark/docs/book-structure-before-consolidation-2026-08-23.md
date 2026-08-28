# 통합 전 책 구성 기록

Qwen3.8·DeepSeek부터 멀티 Spark 클러스터와 코딩 에이전트까지

![책 표지](../assets/book-cover-v1.png)

NVIDIA DGX Spark(GB10)에서 로컬 모델을 선택하고 설치한 뒤, 성능을 측정하고 실제 에이전트로 운영하기 위한 한국어 실전서다.

상태: Wikidocs GitHub 연동 배포판 / 00~19장 원고와 실험 기록을 함께 관리

기준일: 2026-08-23

## 책 소개

DGX Spark는 128GB unified memory와 CUDA 생태계를 한 장비에 담아 로컬 AI를 직접 운영하려는 사람을 위한 컴퓨터다. 하지만 모델이 메모리에 올라온다고 바로 쓸 만한 서버가 되는 것은 아니다. 모델의 크기와 양자화, 실행 엔진, 컨텍스트 길이, 동시성, 네트워크와 전력 조건을 함께 결정해야 한다.

이 책은 DGX Spark에서 Qwen3.8·DeepSeek·MiniMax 계열 언어 모델과 MiniMax-H3 영상·음성 생성 경로를 실행하고, NVIDIA Founders Edition과 GB10 OEM을 비교하며, 한 대에서 여러 대로 확장하고 코딩 에이전트와 연결하는 과정을 실전 레시피로 정리한다. vLLM·SGLang·llama.cpp 선택, FP8·NVFP4·EXL3와 KV cache, MTP·DSpark·DFlash2, tool parser, OpenAI-compatible endpoint, RoCE·NCCL·스위치 구성, 발열·OOM·저클럭 장애까지 다룬다.

처음 읽는 사람은 보통 “이 장비가 내 작업에 맞는가?”, “128GB에 어떤 모델이 들어가는가?”, “어디까지 해야 실제로 쓸 수 있는가?”를 먼저 묻는다. 그래서 초반 장은 사양 설명보다 질문과 판단 순서를 먼저 보여주고, 이후 장에서 설치·측정·운영 레시피를 자세히 다룬다.

커뮤니티의 tok/s 숫자를 그대로 옮기지 않는다. 공식 자료, 실행 가능한 GitHub recipe, 포럼과 Reddit의 사용담, DGX Spark에서 직접 확인한 결과를 나누고, `loaded`, `serves`, `benchmarked`, `tool-tested`, `agent-tested`를 서로 다른 상태로 기록한다. 이 책의 목표는 가장 큰 모델을 자랑하는 것이 아니라, 자신의 장비와 작업에 맞는 구성을 고르고 재현 가능한 결과를 남기도록 돕는 것이다.

## 먼저 결정하기

처음부터 모든 장을 읽을 필요는 없다. 먼저 00장의 읽기 지도를 보고, 아래 표에서 자신의 질문을 찾은 뒤 필요한 경로로 이동하면 된다.

| 지금 하려는 일 | 첫 페이지 | 이어서 읽을 장 |
|---|---|---|
| 아직 구매·보유를 결정하지 못한 경우 | [00장 읽기 지도](../book/00-how-to-read.md) → [01장 적합성](../book/01-why-dgx-spark.md) | [02장 메모리와 대역폭](../book/02-gb10-architecture.md) → [12장 비용 판단](../book/10-1-cost-power-decision.md) |
| Spark 한 대에서 처음 코딩을 시작하기 | [00장 읽기 지도](../book/00-how-to-read.md) | [01장 적합성](../book/01-why-dgx-spark.md) → [03장 안전한 기본 환경](../book/03-1-first-boot-safe-environment.md) → [04장 첫 모델](../book/03-4-single-spark-first-model.md) |
| 어떤 모델이 들어가고 실제로 빠른지 알고 싶은 경우 | [01장 적합성](../book/01-why-dgx-spark.md) → [02장 메모리](../book/02-gb10-architecture.md) | [18장 모델 선택](../book/06-4-model-selection.md) → [07장 benchmark](../book/05-1-benchmark-design.md) |
| 긴 문맥의 코드·문서 supervisor 만들기 | [18-1 DeepSeek](../book/06-5-deepseek-v4-flash.md) | [05장 엔진](../book/04-1-engine-selection.md) → [06장 양자화·speculative decoding](../book/06-1-quantization-speculative.md) → [10장 에이전트](../book/08-1-local-agent-operations.md) |
| 빠른 코드·UI·JSON worker 만들기 | [18-2 Qwen3.8](../book/06-6-qwen38-27b.md) | [04장 첫 모델](../book/03-4-single-spark-first-model.md) → [07장 benchmark](../book/05-1-benchmark-design.md) → [10장 에이전트](../book/08-1-local-agent-operations.md) |
| 두 대 이상으로 긴 문맥·동시성을 확장하기 | [01장 적합성](../book/01-why-dgx-spark.md) → [08장 두 대 연결](../book/07-1-two-spark-cluster.md) | [09장 확장](../book/07-4-multi-spark-scaling.md) → [11장 장애 대응](../book/09-1-operations-failure-recovery.md) → [12장 비용 판단](../book/10-1-cost-power-decision.md) |
| 영상·음성 생성 파이프라인을 만들기 | [18-4 MiniMax-H3](../book/06-8-minimax-h3.md) | [03장 안전한 기본 환경](../book/03-1-first-boot-safe-environment.md) → [11장 장애 대응](../book/09-1-operations-failure-recovery.md) |
| 커뮤니티가 실제로 만든 결과를 조사하기 | [15장 DeepSeek 사례](../book/06-9-deepseek-community-builds.md) 또는 [16장 Qwen 사례](../book/06-12-qwen38-community-builds.md) | [14장 비교 조건](../book/10-4-gpt56-sol-comparison.md) → [19장 리서치 로그](../book/appendix-b-research-log.md) |

모델 이름만으로 선택하지 않는다. 먼저 **작업 유형과 노드 수**를 정하고, `loaded → serves → benchmarked → tool-tested → agent-tested` 순서로 검증한다. 18장은 선택을 빠르게 돕는 허브이고, 15·16장은 커뮤니티 결과의 원문과 재현 후보를 모아 두는 사례 장이다.

## 독자가 가장 먼저 묻는 질문

| 질문 | 먼저 읽을 장 | 답을 얻는 기준 |
|---|---|---|
| 이 장비가 내 작업에 맞는가? | [01장](../book/01-why-dgx-spark.md) | 속도·메모리·CUDA·운영비를 workload와 함께 비교 |
| 128GB면 모델이 모두 들어가는가? | [02장](../book/02-gb10-architecture.md) | weight·KV cache·workspace·OS 여유를 함께 계산 |
| 첫날에 무엇을 설치하고 기록해야 하는가? | [03장](../book/03-1-first-boot-safe-environment.md) | 전원·업데이트·Docker·driver·디스크 기준점 확보 |
| 언제 “실행됐다”고 말할 수 있는가? | [04장](../book/03-4-single-spark-first-model.md) | loaded → serves → benchmarked → tool-tested 단계 통과 |
| 인터넷에서 본 tok/s를 믿어도 되는가? | [07장](../book/05-1-benchmark-design.md) | prompt·context·quant·runtime·동시성 조건 확인 |
| 두 대를 사면 무엇이 달라지는가? | [08장](../book/07-1-two-spark-cluster.md) | 모델 크기뿐 아니라 링크·통신·장애 격리까지 검증 |

## 이 책의 약속

이 책은 `어떤 모델이 제일 빠른가`만 설명하지 않는다. 독자가 다음 질문에 스스로 답할 수 있도록 구성한다.

- 내 Spark가 한 대면 어떤 모델부터 시작할까?
- 두 대를 사면 속도, 모델 크기, 컨텍스트, 동시성 중 무엇이 좋아질까?
- 3대와 4대는 왜 네트워크 구성이 달라질까?
- `loaded`, `generates`, `serves`, `benchmarked`, `agent-tested`는 어떻게 다른가?
- 커뮤니티의 tok/s 숫자를 내 장비에서 어떻게 검증할까?
- 발열·OOM·NCCL hang·펌웨어 문제를 어떻게 안전하게 진단할까?

모든 성능 수치는 다음 정보를 함께 기록해야 비교할 수 있다.

`hardware · model revision · quant · runtime/commit · context · KV dtype · speculative decoding · concurrency · workload · measurement method`

## Wikidocs 목차

| 페이지 | 제목 | 상태 | 주된 산출물 |
|---:|---|---|---|
| 00 | 이 책을 읽는 방법 | draft | 용어·증거 등급·재현 원칙 |
| 01 | DGX Spark가 내 작업에 맞는가? | draft | 구매·구성·workload 판단표 |
| 02 | 128GB unified memory와 대역폭을 읽는 법 | draft | 메모리·대역폭·TP/PP/DP 설명 |
| 03 | 첫 부팅에서 실패하지 않는 방법 | draft | 전원·preflight·업데이트·백업 체크리스트 |
| 04 | 첫 모델을 올리고 “된다”를 증명하는 방법 | draft | 단계별 smoke test와 첫 serving 경로 |
| 05 | vLLM·SGLang·llama.cpp 선택 | draft | 엔진 선택표와 실패 패턴 |
| 06 | 양자화와 speculative decoding | draft | FP8/NVFP4/AWQ/EXL3/MTP/DFlash 비교 |
| 07 | 벤치마크를 제대로 설계하기 | draft | 속도·품질·에이전트 평가 하니스 |
| 08 | 두 대 연결하기 | research-backed | QSFP/RoCE/NCCL·DeepSeek TP=2 |
| 09 | 세 대·네 대·여덟 대 | research-backed | PP/TP/DP·switch·확장 한계 |
| 10 | 로컬 에이전트 운영 | research-backed | Hermes/OpenClaw/NemoClaw·권한 경계 |
| 11 | 발열·OOM·NCCL·펌웨어 장애 | research-backed | 저클럭·clock cap·진단 순서와 복구 원칙 |
| 12 | 비용·전력·구성 의사결정 | research-backed | 장비 수별 TCO와 운영비 |
| 13 | 부록: 모델·레시피·명령어 색인 | draft | 고정 버전·링크·용어집 |
| 14 | GPT-5.6 Sol과 로컬 모델 비교 | research-backed | Sol max 용어·공식 조건·공정한 비교표 |
| 15 | DeepSeek V4 Flash 0731로 사람들이 만든 것 | research-backed | 에이전트·비전·영상·듀얼 Spark 제작물과 재현 큐 |
| 16 | Qwen3.8-27B로 사람들이 만든 것 | research-backed | serving recipe·speculative decoding·코딩 에이전트·노드 수별 구성 |
| 17 | DGX Spark·GB10 벤더 비교 | research-backed | NVIDIA Founders Edition과 Acer·ASUS·Dell·GIGABYTE·HP·Lenovo·MSI 비교 |
| 18 | DGX Spark에서 돌릴 모델 선택 | research-backed | DeepSeek·Qwen·MiniMax·MiniMax-H3의 노드 수·recipe·검증 상태 |
| 19 | DGX Spark 리서치 로그 | daily | 수집·검증·승격 대기 기록 |

현재 WikiDocs export는 20개 상위 장과 고정 서브챕터로 구성된다. 19장 아래의 날짜별 리서치 서브챕터는 `docs/research-issue-N_YYYY-MM-DD.md` 원문을 export할 때 `TOC.md`에 동적으로 추가된다. 리서치 원문은 `docs/`에만 보관하며, `book/appendix-b-research-issue-*.md` 파일을 저장소에 중복해서 만들지 않는다. 상위 장은 맥락과 전체 결론을 설명하고, 서브챕터는 독자가 바로 실행할 수 있는 독립 레시피 또는 날짜별 연구 기록으로 분리한다.

## 현재 장별 파일

- [00장 — 이 책을 읽는 방법](../book/00-how-to-read.md)
- [01장 — DGX Spark가 내 작업에 맞는가?](../book/01-why-dgx-spark.md)
- [02장 — 128GB unified memory와 대역폭을 읽는 법](../book/02-gb10-architecture.md)
- [03장 — 첫 부팅에서 실패하지 않는 방법](../book/03-1-first-boot-safe-environment.md)
- [04장 — 첫 모델을 올리고 “된다”를 증명하는 방법](../book/03-4-single-spark-first-model.md)
- [05장 — 엔진 선택](../book/04-1-engine-selection.md)
- [06장 — 양자화·KV cache·speculative decoding](../book/06-1-quantization-speculative.md)
- [07장 — benchmark 설계](../book/05-1-benchmark-design.md)
- [08장 — 두 대 연결하기](../book/07-1-two-spark-cluster.md)
- [09장 — 세 대·네 대·여덟 대](../book/07-4-multi-spark-scaling.md)
- [10장 — 로컬 에이전트 운영](../book/08-1-local-agent-operations.md)
- [11장 — 장애 대응](../book/09-1-operations-failure-recovery.md)
- [12장 — 비용·전력·구성 의사결정](../book/10-1-cost-power-decision.md)
- [13장 — 부록: 모델·레시피·명령어 색인](../book/appendix-a-1-model-recipe-command-index.md)
- [14장 — GPT-5.6 Sol과 로컬 모델 비교](../book/10-4-gpt56-sol-comparison.md)
- [15장 — DeepSeek V4 Flash 0731로 사람들이 만든 것](../book/06-9-deepseek-community-builds.md)
- [16장 — Qwen3.8-27B로 사람들이 만든 것](../book/06-12-qwen38-community-builds.md)
- [17장 — DGX Spark·GB10 벤더 비교](../book/01-2-gb10-vendor-comparison.md)
- [18장 — DGX Spark에서 돌릴 모델 선택](../book/06-4-model-selection.md)
- [19장 — DGX Spark 리서치 로그](../book/appendix-b-research-log.md)

## 기존 리서치와의 연결

- [DGX Spark 독자 질문과 초반 장 재설계 리서치](../docs/dgx-spark-reader-questions-research-2026-08.md)
- [노드 수별 모델·클러스터 리서치](../docs/dgx-spark-cluster-model-research-2026-08.md)
- [NVIDIA Developer Forum 전체 리서치](../docs/dgx-spark-nvidia-forum-research-2026-08.md)
- [책 집필용 참고문헌](../docs/dgx-spark-book-references-2026-08.md)
- [DGX Spark·Mac 혼합 구성과 스위치 리서치](../docs/dgx-spark-mac-rdma-switch-research-2026-08.md)
- [Qwen3.8-27B-OBLITERATED 직접 테스트](../docs/model-research-qwen38-obliterated.md)
- [GPT-5.6 Sol max 비교 리서치](../docs/sol-max-comparison-research-2026-08.md)
- [DeepSeek V4 Flash 0731 커뮤니티 제작물·응용 사례](../docs/deepseek-v4-flash-0731-community-builds-2026-08.md)
- [Qwen3.8-27B 커뮤니티 제작물·활용 사례](../docs/qwen38-community-builds-2026-08.md)
- [DGX Spark·GB10 벤더 비교 리서치](../docs/dgx-spark-vendor-comparison-2026-08.md)
- [DGX Spark 모델 선택 리서치](../docs/dgx-spark-model-selection-research-2026-08.md)
- [DGX Spark 리서치 승격 규칙](../docs/research-promotion.md)
- [Wikidocs 배포 계획과 상태](../docs/wikidocs-deployment-2026-08.md)
- [Wikidocs page ID 회수와 본문 링크 연결](../docs/wikidocs-page-id-recovery-2026-08.md)

## Wikidocs 배포 구조

Wikidocs GitHub 연동 규칙에 맞춘 export는 [`wikidocs/`](../wikidocs/)에 생성한다. 이 bundle은 `README.md`, `TOC.md`, `pages/`, `assets/`를 포함하며, [Wikidocs GitHub 연동 안내](https://wikidocs.net/321336)의 구조를 따른다. 실제 연동 대상은 [`recrack/oh-my-dgx-spark-wikidocs`](https://github.com/recrack/oh-my-dgx-spark-wikidocs)다. `README.md`의 첫 번째 제목과 책 소개는 이 파일에서 읽어 export한다. `TOC.md`의 장 제목에는 `00`~`19` 번호를 붙여 이름순 자동 정렬에서도 순서를 유지한다. 원고를 고친 뒤에는 `main`에 commit·push하면 Actions가 export, 검사, 배포 저장소 push를 자동으로 처리한다.

## 집필 규칙

1. 공식 자료, 재현 가능한 커뮤니티 레시피, 실측, 사용담을 구분한다.
2. 모델이 메모리에 올라온 것과 정상적으로 생성·서빙·도구 호출한 것을 구분한다.
3. 단일 스트림, aggregate throughput, prefill, decode를 한 숫자로 합치지 않는다.
4. 로컬에서 검증하지 않은 명령은 `실험용`으로 표시한다.
5. 모델·런타임·컨테이너가 빠르게 바뀌므로 페이지마다 기준일과 버전을 남긴다.
