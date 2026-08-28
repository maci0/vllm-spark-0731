# 로컬 AI 실전 레시피: DGX Spark 편

Qwen·DeepSeek·MiniMax부터 멀티 Spark 클러스터와 코딩 에이전트까지 다룹니다.

GitHub 연동 배포 저장소: [https://github.com/recrack/oh-my-dgx-spark-wikidocs](https://github.com/recrack/oh-my-dgx-spark-wikidocs)

![책 표지](assets/book-cover-v1.png)

DGX Spark는 128GB unified memory와 CUDA 생태계를 한 장치에 담아 로컬 AI를 직접 운영하려는 사람을 위한 시스템입니다. 하지만 모델이 메모리에 올라왔다고 해서 곧바로 쓸 만한 서버가 되는 것은 아닙니다. 모델 크기와 양자화 형식, 실행 엔진, 컨텍스트 길이, 동시성, 네트워크와 전력 조건을 함께 결정해야 합니다.

이 책은 “무슨 모델을 설치할까?”에서 시작하지 않습니다. 먼저 Spark가 내 작업에 맞는지 판단하고, 한 대에서 최소 endpoint를 만든 다음, 성능과 tool call을 측정합니다. 그 결과 두 대 이상이 필요할 때만 클러스터로 확장합니다. Qwen·DeepSeek·MiniMax 계열, vLLM·SGLang·llama.cpp·SparkInfer, FP8·NVFP4·EXL3, speculative decoding, RoCE·NCCL·스위치, 발열·OOM·저클럭 장애를 하나의 실행 순서로 묶었습니다.

공식 문서, 실행 가능한 GitHub 레시피, 포럼·Reddit·X의 공개 보고, 이 저장소에서 직접 측정한 결과를 같은 종류의 근거처럼 섞지 않습니다. 각 내용은 **공식 사실**, **공개 보고**, **직접 실험**, **편집자의 판단**으로 구분합니다. DeepSeek V4 Flash 0731의 공개 레시피 수치를 GPT-5.6 Sol의 품질과 같은 의미로 읽지 않습니다. 동일한 task harness가 없을 때는 “강한 후보”라고만 씁니다.

현재 원고는 **WikiDocs GitHub 연동 배포판**입니다. 연결된 배포 저장소에 push하면 WikiDocs 책으로 자동 동기화됩니다. 책의 공개 설정과 GitHub 저장소의 공개 설정은 별개입니다. 모델과 runtime이 바뀌면 원문 링크, revision, 실행 조건, raw 결과를 함께 갱신합니다.

## 책의 구성

상위 목차는 00~10장과 부록 A·B로 구성합니다. 상위 장은 판단, 설치, serving, benchmark, 모델 선택, 클러스터, 에이전트, 운영, 비용 판단의 읽기 경로를 안내합니다. 기존 상세 원고와 재현 절차는 해당 장 아래의 서브챕터로 보존하며, 날짜별 공개 리서치는 부록 B 아래에 추가합니다.

- 한 대, 두 대, 세 대, 네 대, 여덟 대 DGX Spark의 역할과 네트워크
- Qwen, DeepSeek, MiniMax 모델의 선택 기준
- vLLM, SGLang, llama.cpp, SparkInfer와 양자화·speculative decoding
- prefill, decode, end-to-end, aggregate throughput의 차이
- tool parser, 로컬 에이전트, 권한 경계와 장애 복구
- GPT-5.6 Sol(`gpt-5.6-sol`, `reasoning_effort=max`)과 로컬 모델을 공정하게 비교하는 방법

## 실험 원칙

성능 숫자에는 hardware, model revision, quant, runtime/image, context, KV dtype, speculative decoding, concurrency, workload와 측정 방법을 함께 적습니다. 공개 레시피의 44~47 tok/s나 370K needle 결과를 이 장비의 모든 요청에 대한 보장값으로 쓰지 않습니다.

## 원본 저장소와 실험 기록

- [원본 GitHub 저장소](https://github.com/recrack/oh-my-dgx-spark)
- [DeepSeek V4 Flash 0731 성능 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/deepseek-v4-flash-0731-performance-research-2026-08.md)
- [Qwen3.8-27B 커뮤니티 제작물·활용 사례](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/qwen38-community-builds-2026-08.md)
- [DGX Spark·GB10 벤더 비교 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-vendor-comparison-2026-08.md)
- [DGX Spark 모델 선택 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-model-selection-research-2026-08.md)
- [GPT-5.6 Sol max 비교 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/sol-max-comparison-research-2026-08.md)
- [책 집필용 참고문헌](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-book-references-2026-08.md)
- [기존 원고 보존·재배치 기록](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/book-content-preservation-2026-08-24.md)
- [WikiDocs 배포 계획과 상태](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/wikidocs-deployment-2026-08.md)
- [WikiDocs page ID 회수와 본문 링크 연결](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/wikidocs-page-id-recovery-2026-08.md)

기준일: **2026-08-23**
