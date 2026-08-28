# 로컬 AI 실전 레시피: DGX Spark 편

Qwen·DeepSeek·MiniMax부터 멀티 Spark 클러스터와 코딩 에이전트까지 다룹니다.

![책 표지](../assets/book-cover-v1.png)

NVIDIA DGX Spark(GB10)에서 로컬 모델을 선택하고, 설치하고, 측정하고, 운영하기 위한 한국어 실전서입니다.

기준일: **2026-08-23**

## 책 소개

DGX Spark는 128GB unified memory와 CUDA 생태계를 한 장치에 담아 로컬 AI를 직접 운영하려는 사람을 위한 시스템입니다. 하지만 모델이 메모리에 올라왔다고 해서 곧바로 쓸 만한 서버가 되는 것은 아닙니다. 모델 크기와 양자화 형식, 실행 엔진, 컨텍스트 길이, 동시성, 네트워크와 전력 조건을 함께 결정해야 합니다.

이 책은 “무슨 모델을 설치할까?”에서 시작하지 않습니다. 먼저 Spark가 내 작업에 맞는지 판단하고, 한 대에서 최소 endpoint를 만든 다음, 성능과 tool call을 측정합니다. 그 결과 두 대 이상이 필요할 때만 클러스터로 확장합니다. Qwen·DeepSeek·MiniMax 계열, vLLM·SGLang·llama.cpp·SparkInfer, FP8·NVFP4·EXL3, speculative decoding, RoCE·NCCL·스위치, 발열·OOM·저클럭 장애를 하나의 실행 순서로 묶었습니다.

공식 문서, 실행 가능한 GitHub 레시피, 포럼·Reddit·X의 공개 보고, 이 저장소에서 직접 측정한 결과를 같은 종류의 근거처럼 섞지 않습니다. 각 내용은 **공식 사실**, **공개 보고**, **직접 실험**, **편집자의 판단**으로 구분합니다. DeepSeek V4 Flash 0731의 공개 레시피 수치를 GPT-5.6 Sol의 품질과 같은 의미로 읽지 않습니다. 동일한 task harness가 없을 때는 “강한 후보”라고만 씁니다.

## 이 책의 읽는 순서

상위 읽기 경로를 00~10장과 두 부록으로 재배치했습니다. 기존의 세부 장, 실측, 명령, 커뮤니티 사례와 출처는 삭제하지 않고 해당 상위 장 아래의 서브챕터로 보존했습니다. 독자는 먼저 판단 → 설치 → 측정 → 모델 선택 → 확장 → 운영 → 비용 판단의 흐름을 읽고, 필요한 주제만 상세 페이지로 내려가면 됩니다.

| 목적 | 읽을 경로 |
|---|---|
| 이 책의 기준과 측정 언어 이해 | [00장. 이 책을 읽는 방법](00-how-to-read.md) |
| 구매·보유 여부 결정 | [01장. DGX Spark가 내 작업에 맞는가?](01-why-dgx-spark.md) → [10장. 최종 선택](10-decision.md) |
| 한 대에서 첫 모델 실행 | [02장. 모델·메모리·노드 수](02-gb10-architecture.md) → [03장. 첫 모델](03-first-model.md) → [04장. endpoint](04-serving.md) |
| 성능과 품질 비교 | [05장. benchmark](05-benchmark.md) → [06장. 모델 레시피](06-model-recipes.md) |
| 두 대 이상과 스위치 | [07장. 여러 Spark와 네트워크](07-cluster.md) |
| 코딩 에이전트 연결 | [08장. 코딩 에이전트](08-agents.md) |
| 발열·저클럭·OOM·NCCL 장애 | [09장. 운영과 복구](09-operations.md) |
| 명령과 결과 양식 | [부록 A](appendix-a-commands.md) |
| 매일 수집되는 공개 자료 | [부록 B](appendix-b-research-log.md)와 날짜별 리서치 페이지 |

## 상위 목차를 줄인 이유

모델별 사례, 엔진 선택, 양자화, benchmark, 장애 대응을 모두 별도 상위 장으로 두면 처음 읽는 독자가 시작점을 찾기 어렵습니다. 현재 상위 목차는 11개 장과 부록 2개로 유지하되, 기존 상세 원고는 서브챕터로 모두 공개합니다. 따라서 상위 목차는 짧아졌지만 본문의 정보량을 줄인 것은 아닙니다.

고정 페이지의 순서와 상·하위 구조는 [`book/TOC.md`](TOC.md)가 관리합니다. `book/`의 Markdown 파일이 이 목차에 빠져 있으면 export가 실패하므로, 파일만 남고 WikiDocs에서 사라지는 상태를 허용하지 않습니다.

## 근거를 확인하는 방법

- 공식 사양·설치 전제는 [NVIDIA DGX Spark 문서](https://docs.nvidia.com/dgx/dgx-spark/index.html)와 공식 playbook을 기준으로 합니다.
- 모델의 품질과 지원 범위는 모델 카드와 runtime 문서를 기준으로 합니다.
- GitHub 레시피의 tok/s와 긴 context 결과는 해당 레시피의 조건으로만 기록합니다.
- 직접 실험은 raw JSON, 모델 revision, runtime/image, context, concurrency와 실패 상태를 함께 남깁니다.
- Sol 비교는 동일 task set과 harness가 없으면 품질 동급으로 표현하지 않습니다.

전체 참고문헌은 [DGX Spark 책 참고문헌](../docs/dgx-spark-book-references-2026-08.md), 모델·클러스터 판단 근거는 [모델 선택 리서치](../docs/dgx-spark-model-selection-research-2026-08.md), Sol 비교 근거는 [Sol max 비교 리서치](../docs/sol-max-comparison-research-2026-08.md)에서 확인할 수 있습니다. 기존 원고의 복원 범위와 새 목차의 대응 관계는 [원고 보존·재배치 기록](../docs/book-content-preservation-2026-08-24.md)에 남겼습니다.

## WikiDocs 연동

고정 장과 서브챕터의 원고는 `book/`에서 수정하고, 공개 순서는 `book/TOC.md`에서 관리합니다. 날짜별 리서치 원문은 `docs/research-issue-N_YYYY-MM-DD.md`에서 관리합니다. `main`에 push하면 GitHub Actions가 `README.md`, `TOC.md`, `pages/`, `assets/` bundle을 만들고 `oh-my-dgx-spark-wikidocs` 저장소로 배포합니다. WikiDocs의 상·하위 구조는 export된 `TOC.md`가 관리합니다. 본문 장 사이의 이동 링크는 회수된 숫자형 page ID가 있을 때만 생성합니다.

Archify 다이어그램의 원본은 `docs/diagrams/archify/`에, WikiDocs에 들어가는 정적 SVG는 `assets/`에 둡니다. SVG는 XML로 검증하고 본문에서 읽을 수 있는 인쇄용 글자 크기로 추출합니다.

## 책의 상태 표기

`loaded → serves → generates → benchmarked → tool-tested → agent-tested`는 서로 다른 상태입니다. 모델이 올라왔다는 이유만으로 benchmark나 에이전트 검증을 통과했다고 쓰지 않습니다.

기준일: **2026-08-23**
