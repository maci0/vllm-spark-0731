# 로컬 AI 실전 레시피: DGX Spark 편 참고문헌

기준일: **2026-08-22**
목적: `로컬 AI 실전 레시피: DGX Spark 편` 집필에 사용할 참고문헌, 레시피, 벤치마크를 분류한 인덱스

이 문서는 링크를 모아 두는 데서 끝내지 않고, 각 자료를 책의 어느 장에 사용할지와 신뢰 수준을 함께 기록한다. DGX Spark 소프트웨어와 모델은 빠르게 바뀌므로 링크의 내용, 버전, 성능 수치는 기준일과 함께 읽어야 한다.

### 링크 점검

기준일에 문서 안의 URL을 리다이렉트를 따라가는 HTTP 점검으로 확인했다. 대부분 `200`으로 응답했다. 아카라이브, 서버포럼, Wikidocs의 일부 URL은 사이트 접근제어로 `403`을 반환했으므로 삭제 링크로 판정하지 않고 **브라우저 수동 확인 필요**로 분류한다. 이번에 추가한 X 게시물은 원문 접근 제한 때문에 게시물 링크와 재게시본을 함께 보존한다.

## 읽는 법

자료의 종류는 다음처럼 표시한다.

- **A — 공식 원문**: NVIDIA, 모델 제작자, 런타임 프로젝트의 공식 문서·모델 카드. 사양·지원 범위·설치 전제의 기준으로 사용한다.
- **B — 재현 레시피**: 명령어, 컨테이너, 설정 파일, 커밋 또는 측정 조건이 공개된 GitHub 자료. 책의 실행 레시피 후보로 사용하되 장비에서 다시 실행한다.
- **C — 실측·벤치마크**: 측정 방법과 결과를 제공하는 자료. 숫자를 옮길 때 모델·양자화·엔진·컨텍스트·동시성·날짜를 함께 기록한다.
- **D — 커뮤니티 사례**: Reddit, NVIDIA Forum, 국내 커뮤니티의 사용기·문제 보고. 새로운 실험 주제를 찾는 데 사용하며 단일 사례를 일반적인 성능으로 쓰지 않는다.
- **E — 서적·웹북**: 책의 구성, 독자 수준, 가격·공개 방식, 이미 다뤄진 범위를 비교하기 위한 자료.
- **F — 모델 카드·논문**: 모델 정체, 라이선스, 기능, 평가 방법의 원문. 모델 카드의 자체 주장은 독립 검증 결과와 분리한다.

### 책 원고에 적용할 검증 규칙

1. 하드웨어·소프트웨어 버전은 **A 자료**를 기준으로 쓴다.
2. 실행 명령은 가능하면 **B 자료를 직접 재현**하고, 우리 장비에서 통과한 커밋·날짜를 남긴다.
3. `tok/s` 하나만으로 모델 순위를 만들지 않는다. 최소한 TTFT, prefill, decode, context, concurrency, 실패 여부를 분리한다.
4. NVIDIA·모델 제작자 측 수치는 `vendor-reported`, 커뮤니티 수치는 `community-reported`, 우리 측 수치는 `locally reproduced`로 표시한다.
5. 책의 코드·표·그림은 원문을 복사하지 않고, 링크와 인용을 남기며 우리 환경에서 다시 만든다.

## 1. 이미 나온 책·웹북

현재 확인한 범위에서는 DGX Spark를 제목과 주제로 직접 다루는 자료가 영어권과 중국어권에 먼저 나와 있다. 한국어 Wikidocs에는 전용 책보다 관련 벤치마크와 뉴스형 글이 중심이다.

### E-01 — AI Research on NVIDIA DGX Spark

- 링크: <https://leanpub.com/ai-research-nvidia-dgx-spark>
- 언어/형식: 영어, Leanpub PDF·EPUB, 유료
- 페이지 상태: 2026-05-30 기준 완성본, 최소 가격 55달러·권장 가격 65달러로 표시
- 범위: 단일 DGX Spark에서 foundations, inference/RAG, training, fine-tuning, agentic systems, observability, deployment
- 강점: 실제 실험과 재현 코드, Field Kit Python 라이브러리 연계
- 책에 활용: 영문 시장의 포지셔닝과 연구형 구성 비교
- 주의: 우리 책의 모델별 1·2·3·4대 비교 자료로 직접 대체할 수는 없음

### E-02 — DGX Spark 玩透指南

- 링크: <https://alingowangxr.github.io/dgx-spark-book/>
- 언어/형식: 중국어 번체, 공개 웹북, CC BY-SA 4.0 표시
- 범위: 개봉·시스템 설정, Ollama, Open WebUI, LM Studio, llama.cpp, vLLM, TensorRT-LLM, SGLang, NIM, 이미지·영상·음성 생성, LoRA/QLoRA, RAG, Agent, CUDA-X/JAX, 다중 노드
- 강점: 초보자용 단계별 흐름과 광범위한 도구 목록
- 책에 활용: 독자 난이도와 전체 목차 설계 비교
- 주의: 모델 버전·실측 숫자는 현재 시점에서 다시 확인

### E-03 — From Box to Cluster: Building a Personal AI Supercomputer

- 링크: <https://mohnishbasha.github.io/dgx-spark-bundle/books/from-box-to-cluster/>
- 언어/형식: 영어, 공개 웹북·PDF
- 게시 정보: First Edition, 2026-07
- 범위: 2× DGX Spark, 하드웨어·첫 부팅, CUDA 업데이트, k3s, GPU Operator, KubeRay, vLLM tensor parallelism, AIBrix, Prometheus/Grafana/DCGM 모니터링
- 강점: ARM64 이미지 문제와 실제 2노드 운영 구성을 구체적으로 다룸
- 책에 활용: 우리 책의 2대 클러스터 장에서 반드시 비교할 기준 자료
- 차별화 여지: 한국어, 1대부터 4대까지, 모델 선택과 일반 사용자용 레시피

### E-04 — NVIDIA DGX Spark User Guide

- 링크: <https://docs.nvidia.com/dgx/dgx-spark/index.html>
- PDF: <https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf>
- 형식: 공식 사용자 매뉴얼. 서적은 아니지만 하드웨어·초기 설정·DGX OS·Docker·NVIDIA Sync·PXE·기업 관리의 1차 기준
- 책에 활용: 첫 부팅, 시스템 업데이트, 메모리 보고, 지원 범위를 이 문서와 대조

## 2. 공식 제품·시스템 문서

### A-01 — NVIDIA 한국 제품 페이지

- <https://www.nvidia.com/ko-kr/products/workstations/dgx-spark/>
- 128GB 통합 메모리, GB10, 제품 포지셔닝, 공식 플레이북 진입점

### A-02 — NVIDIA Developer DGX Spark 시작 페이지

- <https://developer.nvidia.com/topics/ai/dgx-spark>
- 복구 이미지, NVIDIA Sync, AI Workbench, 하드웨어·초기 설정·릴리스 문서 링크 허브

### A-03 — DGX Spark User Guide

- <https://docs.nvidia.com/dgx/dgx-spark/index.html>
- 하드웨어 개요, 첫 부팅, ConnectX-7, DGX Dashboard, Docker, NGC, PXE와 기업 관리

### A-04 — DGX Spark Release Notes

- <https://docs.nvidia.com/dgx/dgx-spark/release-notes.html>
- 현재 DGX OS·드라이버·CUDA 버전, 메모리 압박 처리, 다중 Spark 연결 지원, 알려진 문제
- 책에 활용: 모든 벤치마크 표의 `DGX OS / driver / CUDA / runtime` 열을 만드는 근거

### A-05 — DGX Spark Porting Guide

- HTML: <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html>
- PDF: <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/dgx-spark-porting-guide.pdf>
- ARM64, Ubuntu/DGX OS, unified memory, 273GB/s 메모리 대역폭, ConnectX-7, 애플리케이션 포팅·최적화
- 책에 활용: “왜 x86용 컨테이너가 실패하는가”와 “메모리에 올라간다 ≠ 빠르다” 설명

### A-06 — NVIDIA Sync Cluster Assistant

- <https://docs.nvidia.com/sync/latest/cluster-assistant.html>
- 2대부터 최대 4대 DGX Spark의 연결·클러스터 설정 지원 범위
- 책에 활용: 수동 netplan/NCCL 레시피와 GUI/Sync 경로 비교

### A-07 — DGX Spark Support

- <https://www.nvidia.com/en-us/support/dgx-spark/>
- Quick Start, User Guide, 사양, 복구·지원 진입점

### A-08 — DGX Spark 공식 플레이북 포털

- <https://build.nvidia.com/spark>
- NVIDIA가 유지하는 단계별 워크플로. vLLM, SGLang, llama.cpp, 양자화, 모델, 에이전트, 다중 노드 등을 최신 상태로 확인한다.

### A-09 — DGX Spark Hardware Overview

- <https://docs.nvidia.com/dgx/dgx-spark/hardware.html>
- 128GB unified memory, 4개 USB Type-C, 2개 ConnectX-7 QSFP 포트, 200B 단일 장치와 405B 듀얼 구성이라는 공식 하드웨어 설명
- USB Type-C 포트가 있다는 사실만으로 MCDMA와 같은 직접 메모리 경로가 공식 지원된다는 뜻은 아니다.

### A-10 — DGX Spark 가격 변경 공지

- <https://forums.developer.nvidia.com/t/2-23-2026-price-change-announcement/361713>
- Founders Edition MSRP가 3,999달러에서 4,699달러로 조정된 시점과 사유
- 국내 판매가, OEM 가격, 환율·세금과 분리해 기록

### A-11 — NVIDIA DGX Spark Marketplace

- <https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/dgx-spark/>
- 현재 NVIDIA Marketplace의 4TB DGX Spark 표시 가격과 기본 하드웨어 구성
- 책에서는 조회일의 가격 스냅샷으로만 사용

### A-12 — DGX Spark 전력·규제 정보

- <https://docs.nvidia.com/dgx/dgx-spark/compliance.html>
- 최대 233.2W, idle 38.0W, off mode 4.1W의 공식 전력 표기
- adapter 정격과 실제 벽면 AC 측정값을 구분하는 근거

### A-13 — Apple 한국 Mac 구매 페이지

- <https://www.apple.com/kr/shop/buy-mac>
- Mac Studio 현재 시작 가격과 구성·보상판매 진입점
- Mac을 Spark CUDA rank가 아닌 control host·router·별도 MLX endpoint로 비교할 때 사용

### A-14 — NVIDIA Certified DGX Spark / GB10 Systems 목록

- <https://docs.nvidia.com/certification-programs/latest/nvidia-certified-systems.html>
- 현재 NVIDIA가 인증한 GB10 파트너: Acer, ASUS, Dell Technologies, GIGABYTE, HP, Lenovo, MSI
- 책에 활용: Founders Edition과 OEM partner system을 “같은 GB10 계열”로 묶되, 냉각·firmware·지원이 같다고 가정하지 않는 기준

### A-15 — NVIDIA DGX Spark 공식 제품·하드웨어 사양

- 제품: <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>
- 하드웨어 가이드: <https://docs.nvidia.com/dgx/dgx-spark/hardware.html>
- 128GB LPDDR5x unified memory, 273GB/s, ConnectX-7, 240W adapter, 1/4TB SKU와 200B/405B 포지셔닝

### A-16 — Acer Veriton GN100

- <https://news.acer.com/acer-unveils-the-veriton-gn100-ai-mini-workstation-built-on-the-nvidia-gb10-superchip>
- 128GB, 최대 4TB, GB10, NVIDIA AI stack, PyTorch·Jupyter·Ollama, CX-7, 2대/4대 switch 확장 안내
- 2025년 발표가 USD 3,999/EUR 3,999/AUD 6,499는 현재 국내 가격이 아닌 출시 시점 지역별 스냅샷

### A-17 — ASUS Ascent GX10

- 제품: <https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/>
- FAQ: <https://www.asus.com/us/support/faq/1056142/>
- QuietFlow cooling, 128GB, 273GB/s, 1/2/4TB, CX-7, DGX OS와 switch 확장 관련 공식 설명
- FAQ의 3대/4대 이상 표현은 구성·시점별 답변이 달라 현재 장비와 switch recipe를 별도 검증

### A-18 — Dell Pro Max with GB10

- <https://www.dell.com/en-sg/shop/pcs-desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro>
- 2TB QLC 또는 4TB self-encrypting SSD, 2×200G QSFP CX-7, 280W adapter, Desktop/Network Appliance mode, ProSupport 선택지
- 가격과 SKU는 지역·계약·재고에 따라 변하므로 조회일 없는 구매가로 쓰지 않음

### A-19 — GIGABYTE AI TOP ATOM

- <https://www.gigabyte.com/AI-TOP-PC/GIGABYTE-AI-TOP-ATOM>
- GB10, 128GB unified, 최대 4TB, CX-7, NVIDIA AI stack, AI TOP Utility
- 책에 활용: 모델 다운로드·추론·RAG·머신러닝을 묶은 OEM별 운영 UI 비교

### A-20 — HP ZGX Nano AI Station

- 제품: <https://www.hp.com/us-en/workstations/zgx-nano-ai-station.html>
- QuickSpecs: <https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c09212373>
- DGX OS/Ubuntu 24.04, 128GB·273GB/s, 2/4TB 계열, 2×200GbE QSFP, 240W adapter
- QuickSpecs는 장비의 Windows와 rack mounting을 지원하지 않는다고 명시

### A-21 — Lenovo ThinkStation PGX

- <https://lenovopress.lenovo.com/lp2321-thinkstation-pgx>
- 128GB·273GB/s, 1/4TB SED, 2×CX-7 QSFP, DGX OS, 240W, TPM 2.0·Secure Boot·FW Recovery·NVLink-C2C enclave
- 책에 활용: enterprise workstation·보안·지원 비교

### A-22 — MSI EdgeXpert MS-C931

- <https://ipc.msi.com/product_detail/EdgeXpert-MS-C931>
- 128GB·273GB/s, 1/4TB self-encrypting NVMe, 2×QSFP CX-7, DGX OS, 약 240W
- Docker Compose, 1~2노드 horizontal scaling, block encryption, SSL ingress, private CA 방향을 공식 페이지에 표시

### A-23 — AMD Ryzen AI Halo / Ryzen AI Max+ 395

- <https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html>
- 최대 128GB LPDDR5x, 256GB/s, Radeon 8060S, 120W, Linux·Windows
- GB10·CUDA·DGX OS·CX-7과 다른 대안 플랫폼으로 분류

### A-24 — Apple Mac Studio M3 Ultra

- 발표: <https://www.apple.com/uk/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/>
- 전력: <https://support.apple.com/en-la/102027>
- 최대 512GB unified memory, 800GB/s 초과 메모리 대역폭, M3 Ultra 512GB/16TB 구성 최대 270W 표기
- Metal/MLX 별도 endpoint·control host 후보이며 Spark CUDA TP 노드로 직접 혼합하지 않음

## 3. NVIDIA 공식 플레이북·실행 기준

### B-01 — 전체 플레이북 저장소

- <https://github.com/NVIDIA/dgx-spark-playbooks>
- 공식 레시피의 소스, 스크립트, 컨테이너와 변경 이력

### B-02 — vLLM for Inference

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md>
- DGX Spark용 vLLM 실행, 모델 지원표, Qwen3.6 agent-ready 경로, 2대·스위치 다중 노드 경로

### B-03 — llama.cpp on DGX Spark

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/llama-cpp/README.md>
- CUDA 빌드, GGUF 서버, Qwen3.6 예시, 모델 지원·메모리 전제

### B-04 — SGLang for Inference

- <https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/sglang>
- SGLang, structured output, OpenAI-compatible serving와 Spark용 실행 경로

### B-05 — NVFP4 Quantization

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nvfp4-quantization/README.md>
- Model Optimizer를 이용한 NVFP4 변환·검증과 DGX Spark 성능 튜닝 전제

### B-06 — Connect Two Sparks

- <https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks>
- 200GbE QSFP direct 연결, SSH 키, netplan, 통신 검증, 롤백

### B-07 — Connect Three Sparks in a Ring

- <https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks>
- 3대 ring topology와 연결 검증

### B-08 — Connect Multiple Sparks Through a Switch

- <https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch>
- 스위치를 이용한 4대 이상 확장 경로

### B-09 — NCCL for Multiple Sparks

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nccl/README.md>
- 2·3·4대 NCCL build/test, topology 검증, 분산 학습·통신의 기준 레시피

### B-10 — 공식 성능 벤치마크 가이드

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md>
- TensorRT-LLM, vLLM, SGLang, llama.cpp의 single/dual Spark benchmark 방법
- 책에 활용: 우리 벤치마크 프로토콜을 만들 때 가장 먼저 대조할 자료

### B-11 — Hermes Agent with Local Models

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/hermes-agent/README.md>
- 로컬 vLLM Qwen3.6 모델과 Hermes, Telegram gateway, local terminal backend
- 보안 주의: 에이전트의 명령 실행과 외부 채널을 별도 위험 항목으로 기록

### B-12 — OpenClaw on DGX Spark

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/openclaw/README.md>
- vLLM 로컬 모델과 OpenClaw, Web UI·채널 연결, 공개 노출 금지 등 운영 주의

### B-13 — NemoClaw with Local LLM

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nemoclaw/README.md>
- OpenShell sandbox, local vLLM, OpenClaw, Telegram, 정책·권한 설정
- 별도 공식 quickstart: <https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/get-started/quickstart>

### B-14 — Nemotron Playbook

- <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nemotron/README.md>
- 단일 Spark에서 Nemotron 계열 endpoint를 만드는 공식 진입점

## 4. 공식 성능·발표 자료

### C-01 — How NVIDIA DGX Spark’s Performance Enables Intensive AI Tasks

- <https://developer.nvidia.com/blog/?p=107650>
- fine-tuning, Flux, cuML/cuDF, LLM 등의 NVIDIA 측 측정
- 분류: `vendor-reported`; 다른 하드웨어·런타임과 직접 순위화하지 않는다.

### C-02 — New Software and Model Optimizations Supercharge NVIDIA DGX Spark

- <https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/>
- NVFP4, dual Spark, Qwen-235B, llama.cpp 최적화와 출시 이후 성능 변화
- 책에 활용: 버전 업데이트가 같은 하드웨어의 성능을 바꾸는 사례

### C-03 — Run Local AI Agents with Faster Models and Multi-Node Clustering

- <https://developer.nvidia.com/blog/run-local-ai-agents-with-faster-models-and-multi-node-clustering-on-nvidia-dgx-spark/>
- Qwen3.6, NemoClaw, 로컬 에이전트, 다중 노드 관련 NVIDIA 설명

### C-04 — Scaling Autonomous AI Agents and Workloads

- <https://developer.nvidia.com/blog/?p=114188>
- 에이전트 워크로드, 메모리·커널·roofline 관점

### C-05 — OpenClaw/NemoClaw 보안형 에이전트

- <https://developer.nvidia.com/blog/build-a-secure-always-on-local-ai-agent-with-nvidia-nemoclaw-and-openclaw/>
- 모델 서버부터 Telegram까지의 end-to-end 구성과 보안 경계

### C-06 — Practical LLM Performance on DGX Spark 발표

- <https://www.youtube.com/watch?v=c5-kx2bwoCk>
- 1.5B~14B, vLLM, warm-up, latency·throughput·NVFP4를 다루는 NVIDIA 발표
- 발표 원문과 우리 측 재현 숫자를 분리해 인용

### C-07 — DGX Spark Performance FAQ

- <https://forums.developer.nvidia.com/t/dgx-spark-performance-faq/359456>
- 공식 성능 블로그와 benchmark guide로 연결되는 포럼 공지

## 5. 모델 원문·모델 카드

모델 카드의 “지원한다”, “최대 컨텍스트”, “자체 벤치마크”는 모델 정체와 상한을 설명하는 자료이지, 우리 장비에서의 성공·속도를 보장하지 않는다.

### F-01 — Qwen3.8-27B

- <https://huggingface.co/Qwen/Qwen3.8-27B>
- 원본 모델 카드, multimodal·thinking·context·MTP 관련 기준
- FP8 변형: <https://huggingface.co/Qwen/Qwen3.8-27B-FP8>

### F-02 — Qwen3.6

- <https://github.com/QwenLM/Qwen3.6>
- Qwen3.6-35B-A3B와 Qwen3.5 계열의 공식 저장소·모델 링크

### F-03 — DeepSeek-V4-Flash-0731

- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- 공식 0731 체크포인트, 라이선스, Transformers 사용 경로
- V4 계열 허브: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>

### F-04 — gpt-oss-120b

- <https://developers.openai.com/api/docs/models/gpt-oss-120b>
- OpenAI 공식 모델 설명과 단일 H100 기준의 모델 크기·활성 파라미터 설명
- 모델 카드/기술 자료: <https://arxiv.org/abs/2508.10925>

### F-05 — NVIDIA Nemotron-3.5 Lightning

- <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4>
- DGX Spark 단일 GPU, DSpark/MTP/DFlash, tool parser, 1M context를 명시한 최신 모델 카드
- 책에 활용: agent-ready 단일 Spark 후보

### F-06 — Nemotron-3 Nano Omni

- <https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4>
- 멀티모달·오디오·비디오·문서·툴 호출 후보
- URL이 변경되면 NVIDIA 공식 모델 카드 검색 결과를 우선 확인한다.

### F-07 — Nemotron-3 Super

- <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4>
- NVFP4, 256K/1M 설정, vLLM parser와 Spark 실행 명령

## 6. 단일 Spark 재현 레시피·실험 저장소

### B-15 — Qwen3.8-27B SGLang DGX Spark

- <https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark>
- SGLang, DSpark/DFlash 계열, 단일 Spark의 Qwen3.8 실행·속도 실험

### B-16 — dgx-spark-qwen38

- <https://github.com/hasso5703/dgx-spark-qwen38>
- Qwen3.8 단일 Spark 실험과 설정·측정 결과

### B-16a — Qwen3.8-27B FP8 single Spark

- <https://github.com/0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark>
- FP8 weight를 유지하면서 MTP, DSpark, DFlash2, prefix caching을 비교한 측정 harness
- README 보고값: stock 7.88 tok/s, DSpark k=14 edit-heavy 58.5 tok/s, c8 aggregate 208.7 tok/s, DFlash2 fresh generation 31.72 tok/s
- 속도만 측정했고 품질 평가는 없으며, 단일 스트림과 동시성에서 최적 draft depth가 다르다고 명시한다.

### B-16b — Qwen3.8-27B 4-bit single Spark

- <https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark>
- FP8, MixedInt4-AutoRound, NVFP4, Q4_K_M의 단일 Spark serving 비교
- README 보고값: DSpark k=14 단일 스트림 약 75 tok/s, k=7 c8 aggregate 약 246 tok/s
- 품질 평가는 없고, vLLM 4-bit checkpoint에서는 DFlash2 LM head 조건을 통과하지 못했다고 기록한다.

### B-16c — Qwen3.8-27B DFlash2 FP8 사용기

- <https://www.reddit.com/r/LocalLLM/comments/1vtbwtb/dgx_spark_qwen_38_27b_fp8_at_32toks_generation/>
- DGX Spark FP8 coding workload에서 speculation 없음 약 14 tok/s, DFlash2 안정값 약 32 tok/s라는 개인 측정
- `max-model-len=240000`, `qwen3_coder`, `qwen3` parser 조건을 기록했으나 초기 단일 workload 사용기다.

### B-16d — Qwen3.8-27B 장시간 coding agent 사례

- <https://www.reddit.com/r/Qwen_AI/comments/1vsrq6v/qwen_38_27b_built_this_locally_on_my_rtx_5090/>
- RTX 5090에서 DeepSeek Harness로 FloodLayer 3D AEC sandbox를 구현한 사용담
- 131K context, Q8 KV, MTP draft max 2, 약 54 tok/s에서 MTP 사용 시 약 70~100 tok/s라는 조건부 수치
- 모델 품질·장기 안정성의 공식 평가는 아니며, Prime-Agent 약 48시간 사용 댓글도 개인 경험으로 분리한다.

### B-16e — Qwen3.8-27B의 Apple Silicon MLX 경로

- <https://www.reddit.com/r/LocalLLaMA/comments/1vokrcy/qwen3827b_is_now_up_to_3_faster_on_apple_silicon/>
- M4 Pro에서 mlx-dspark를 사용한 8-bit, 4-bit speculative decoding과 Claude Code endpoint 사례
- Spark와 Mac의 메모리 풀링이 아니라 MLX와 CUDA endpoint를 각각 운영하는 참고 사례다.

### B-17 — Qwen3.5-122B-A10B INT4

- <https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4>
- 단일 Spark에서 큰 MoE를 돌리기 위한 INT4/AutoRound·MTP·KV cache 트레이드오프

### B-18 — DeepSeek V4 Flash single Spark SparkInfer

- <https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer>
- 단일 Spark용 SparkInfer·EXL3/Trellis 경로
- 주의: 공식 full-FP8 원본 실행과 동일한 결과로 쓰지 않는다.

### B-18a — DeepSeek V4 Flash 0731 one-Spark EXL3 recipe

- <https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark>
- GB10/SM121 단일 Spark에서 TP=1, EXL3 3.0 bpw, REAP-K216, SparkInfer와 DSpark K5/K64 draft를 조합한 최신 실행 경로
- README 보고값: `MAX_MODEL_LEN=384000`, `MAX_NUM_SEQS=1`, 구조화 decode 44–47 tok/s, 약 439K KV pool, 370K needle stress test
- 초기 prefill 약 1024 tok/s와 370K 시험의 실효 약 625 tok/s를 구분해서 기록한다. fresh boot과 단일 스트림 조건이며, thinking off는 needle stress test에 명시된 조건이다. 공식 full-FP8·full-expert 품질 벤치가 아니다.

### B-18c — DeepSeek V4 Flash 0731 성능 판정과 GPT-5.6-Sol 비교

- <https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/deepseek-v4-flash-0731-performance-research-2026-08.md>
- <https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/sol-max-comparison-research-2026-08.md>
- 단일 Spark 44–47 tok/s, 1,024 tok/s 초기 prefill, 370K needle, 2대 Spark 측정, 독립 재현과 GPT-5.6-Sol 비교를 분리해 기록한다.
- 공식 API의 모델 ID는 `gpt-5.6-sol`이고 `reasoning_effort=max`를 Sol max로 부른다. 별도 모델 ID처럼 쓰지 않는다.
- 결론: 강한 로컬 코드·에이전트 후보이지만, raw 속도 동률이나 전반적인 GPT-5.6-Sol 동급을 주장하지 않는다.

### B-18b — DeepSeek V4 Flash vision shim on 2× DGX Spark

- <https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark>
- 기존 2× Spark DS4 서버를 재배포하지 않고, 별도 vision process와 OpenAI-compatible shim을 앞에 둔다.
- 기본 흐름은 `harness → :8899 shim → :8081 tiny VLM → :8888 DS4`다. 이미지 요청은 caption으로 바꾸어 DS4에 전달하고, 텍스트 요청은 그대로 통과시킨다.
- README는 기본 `mlx-community/Qwen3.5-0.8B-MLX-8bit` VLM, 이미지 요청의 약 4.6초 end-to-end 실험, 이미지당 약 1~3초 caption overhead를 보고한다.
- 이는 native multimodal DS4가 아니다. caption 손실이 있으므로 OCR·작은 객체·정밀한 공간 관계에는 별도 검증이 필요하다.
- OpenClaw와 raw OpenAI curl/SDK는 검증되었고, Hermes 등 다른 클라이언트는 endpoint 계약상 가능하지만 원문에서 미검증으로 표시한다.

### B-18d — DeepSeek V4 Flash 0731 커뮤니티 제작물·응용 사례

- <https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/deepseek-v4-flash-0731-community-builds-2026-08.md>
- 단일·듀얼 Spark serving recipe, OpenClaw/Hermes/OpenCode/Cursor 연결, Codex MCP 설정, caption shim·vision encoder, DeepSeek+MiniMax H3 co-tenancy와 benchmark 하니스를 한 문서에서 분리한다.
- 자료 등급과 범위: 공식 모델 카드, 실행 가능한 GitHub recipe, Reddit·NVIDIA Forum 실측, GitHub issue·사용담을 `A/B/C/D`로 구분한다.

### B-18e — DeepSeek 0731 에이전트 통합 이슈

- Codex MCP capability 설정 사례: <https://github.com/deepseek-ai/awesome-deepseek-agent/issues/341>
- DeepSeek API tool-call loop 조기 종료 사례: <https://github.com/deepseek-ai/DeepSeek-V3/issues/1554>
- Hermes Agent infinite reasoning loop 사례: <https://github.com/NousResearch/hermes-agent/issues/78807>
- tool parser가 한 번 통과했다는 사실과 장시간 agent loop 성공률을 분리해 기록하는 근거

### B-18f — DeepSeek + MiniMax H3 co-tenancy

- <https://github.com/tonyd2wild/ds4-h3-video-gen-factory>
- 2× DGX Spark에서 DeepSeek 1M context·TP=2와 노드별 MiniMax H3/ComfyUI를 함께 실행하는 결과물
- DeepSeek를 먼저 띄운 뒤 영상 모델을 시작해야 하는 unified-memory 기동 순서와 C1~C6 aggregate 측정 포함

### B-19 — Qwen3.5/3.6 DGX Spark guide

- <https://github.com/adadrag/qwen3.5-dgx-spark>
- Qwen3.5-35B-A3B 설치, benchmark, vision, troubleshooting와 다른 모델 비교

### B-20 — DGX Spark setup guide

- <https://github.com/jschmied/dgx-spark-setup-guide>
- 초기 설정, 성능 튜닝, 모델별 설정·라우팅, Qwen3.6 예시

### B-21 — DGX Spark optimized engine/leaderboard experiment

- <https://github.com/omnia-projetcs/spark-dgx>
- 여러 모델의 설정·컨텍스트·추론 기능·대략적인 속도 표
- 분류: `community-reported`; 측정 조건을 원문에서 확인한 뒤 사용

### B-22 — vLLM Spark Arena

- <https://github.com/Sapid-Labs/vllm-spark-arena>
- GB10/SM121용 vLLM 패치와 동일 출력 검증을 포함한 엔진 실험

### B-23 — vLLM for DGX Spark container

- <https://huggingface.co/nologik/vllm-dgx-spark>
- GB10/SM121과 CUDA graph·Nemotron 경로를 위한 커스텀 컨테이너
- 커스텀 이미지 사용 시 공식 이미지와 별도 프로필로 기록

## 7. 다중 Spark·클러스터·네트워크 레시피

### B-24 — DeepSeek V4 Flash 0731 on 2× DGX Spark

- <https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark>
- TP=2, DSpark, NVFP4 MLA KV, MTP, prefix caching, 256K/1M 프로필
- 문서: <https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/main/docs/DEEPSEEK_V4_FLASH_0731.md>

### B-24a — DS4 supervisor와 Qwen3.8 UI·디자인 worker 분리

- Qwen3.8 듀얼 측정: <https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350>
- Qwen3.8 SGLang/DFlash2 측정: <https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732>
- DS4 vision shim: <https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark>
- `2× Spark DS4`와 `2× Spark Qwen3.8-27B`를 각각 독립 TP=2 pool로 두는 4-Spark 역할 분리안의 근거다.
- 위 세 자료는 하나의 통합 4-Spark 벤치마크가 아니다. DS4의 비전 shim은 Qwen3.8-27B가 아니라 기본 0.8B VLM을 “eyes”로 사용한다는 점을 분리해서 기록한다.

### B-25 — DeepSeek 1M profile on 2× DGX Spark

- <https://github.com/palmfuture/deepseek-v4-flash-0731-dspark-1m-nvfp4-kv-2x-dgx-spark>
- 1M context, NVFP4 KV, 2대 TP 경로

### B-26 — NVIDIA Developer Forum DeepSeek recipe

- <https://forums.developer.nvidia.com/t/guide-deepseek-v4-flash-on-2x-dgx-spark-gb10-reproducible-vllm-serving-recipe-up-to-1m-token-context/374742>
- 공식 포럼의 2대·1M context 재현 논의
- 포럼의 후속 댓글·버전 변경도 함께 기록해야 한다.

### B-27 — spark-vllm-docker

- <https://github.com/eugr/spark-vllm-docker>
- 1·2·3·4대 Spark용 Docker·vLLM 레시피와 mesh/networking
- 네트워크 문서: <https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md>

### B-28 — SGLang DGX Spark

- <https://github.com/mark-ramsey-ri/sglang-dgx-spark>
- single, 2-node, 3+ Spark switched-fabric 구성과 SGLang 실행

### B-29 — vLLM cluster mode setup

- <https://github.com/eelbaz/dgx-spark-vllm-setup/blob/main/CLUSTER.md>
- 다중 노드 vLLM, Ray, 포트, 환경 일치 조건
- 개인 스크립트는 공식 플레이북과 비교 후 사용

### B-30 — Multi-node vLLM/Ray template

- <https://github.com/makiisthenes/dgx-spark-multinode-vllm-ray>
- 2대 vLLM·Ray·200GbE QSFP 구성 템플릿

### B-31 — spark-vllm-compose

- <https://github.com/pfn/spark-vllm-compose>
- DGX Spark 다중 노드 vLLM Docker Compose와 node rank 설정

### B-32 — ArgentAIOS DGX Spark cluster

- <https://github.com/ArgentAIOS/dgx-spark-cluster>
- 2대 direct attach, RDMA/NCCL, 3·4대 확장 시 topology와 운영 문서

### B-33 — 3-node mesh forum thread

- <https://forums.developer.nvidia.com/t/three-node-spark-clusters-without-a-switch-are-now-supported-in-spark-vllm-docker-and-sparkrun/365296>
- 스위치 없는 3대 mesh 지원 관련 커뮤니티·레시피 논의

### B-34 — nixos-dgx-spark NCCL helpers

- <https://github.com/graham33/nixos-dgx-spark/tree/main/playbooks/nccl-two-sparks>
- MTU, ARP, firewall, NCCL 네트워크 설정 자동화 예시

### B-35 — GB10 clock cap measurement harness

- <https://github.com/agjs/gb10-clock-cap>
- OpenAI-compatible endpoint preflight, decode·cold prefill·soak·clock sweep, raw results, verdict와 systemd persistence를 포함한 MIT 라이선스 harness
- 작성자의 2× GB10 reference 결과: `2200MHz` cap에서 peak 90→78°C, 노드당 GPU rail 63.1→40.1W, decode 73.3→72.5 tok/s
- 분류: 실행 가능한 GitHub recipe이므로 `B`; 숫자는 작성자 장비의 benchmark이므로 `C` 성격도 함께 표시. 이 저장소에서 직접 재현한 결과는 아님

### B-36 — DeepSeek V4 Flash 0731 원본 FP8 2× GX10 실측 레시피

- <https://nacyot.github.io/artifacts/deepseek-v4-flash-2x-dgx-spark/>
- 공식 0731 FP8 166.9GB, TP=2, vLLM `mp`, DSpark k=5, NVFP4 MLA KV, RoCE v2 dual rail 구성
- 1M을 설정하고 512K까지 실제 측정했으며, C1·C12 aggregate·prefill·clock sweep을 분리해 기록
- 분류: `B/C`; ASUS GX10 작성자 환경의 recipe·measurement. one-Spark EXL3와 동일 모델/성능표로 합치지 않음

### B-37 — vLLM spin-wait GB10 재현 실험

- <https://nacyot.github.io/artifacts/vllm-spin-wait-gb10-repro/>
- 2× ASUS GX10, DeepSeek FP8 TP=2, vLLM `mp`, `busy_loop_s` 1초 대 2ms 통제 실험
- 멀티스트림 CPU·SoC 온도 감소와 단일 스트림 trade-off를 함께 기록
- 분류: `B/C`; 설치된 vLLM 파일을 직접 수정하는 workaround이므로 버전별 독립 재현 전에는 기본 설정으로 채택하지 않음

### B-38 — DGX Spark 모델 선택 리서치

- <https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-model-selection-research-2026-08.md>
- DeepSeek V4 Flash 0731, Qwen3.8-27B, MiniMax M2.7/M3를 1·2·3·4대 기준으로 구분한 모델 선택표
- `loads`, `generates`, `serves`, `benchmarked`, `tool-tested`, `agent-tested` 상태를 분리

### A-25 — DeepSeek V4 Flash 0731 공식 모델 카드

- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- 공식 release, MIT, encoding parser, reasoning effort low/high/max, vLLM·SGLang·DSpark 실행 경로
- 모델 카드의 하드웨어 예시와 Spark community recipe는 별도로 기록

### A-26 — Qwen3.8-27B 공식 모델 카드

- <https://huggingface.co/Qwen/Qwen3.8-27B>
- 27B dense + vision encoder, native 262K, YaRN 최대 1M, thinking/reasoning control, MTP 학습, Apache 2.0
- `Qwen3_5ForConditionalGeneration` architecture 표기는 Qwen3.5 기반 구조와 관련된 것이며 모델 id와 혼동하지 않음

### A-27 — MiniMax M2.7/M3 공식 모델 카드·API

- M2.7 NVFP4: <https://huggingface.co/nvidia/MiniMax-M2.7-NVFP4>
- M3 DSpark: <https://huggingface.co/nvidia/MiniMax-M3-DSpark>
- API/model guide: <https://minimax-m2.com/docs/api/models>
- M2.7 230B/10B active·204.8K 입력, M3 428B/23B active·1M context·multimodal 설명
- NVIDIA quant card와 MiniMax 원본 license 조건을 상업 사용 전에 함께 확인

## 8. 실측·벤치마크·리더보드

### C-08 — SparkBench

- <https://sparkbench.dev/>
- 실제 DGX Spark에서 모델을 측정하고 재현 레시피·도구·리더보드를 제공하는 프로젝트
- 리더보드: <https://wesche.com/dgx/>
- 책에 활용: 모델 선택 표의 외부 비교 자료. 측정 시점과 엔진 버전을 함께 보존

### C-09 — dgx-spark-benchy

- <https://github.com/abhishek085/dgx_spark_benchy>
- “몇 tok/s인가”뿐 아니라 부하·동시성·공유 사용자·tool agent 가능 여부를 평가하는 single Spark benchmark

### C-10 — dgx-spark-bench

- <https://github.com/Kleybrink/dgx-spark-bench>
- Ollama/모델별 raw benchmark와 결과 보고서
- 결과 예시: <https://github.com/Kleybrink/dgx-spark-bench/blob/main/results/20260504_031837/REPORT.md>

### C-11 — DGX Spark benchmarks dataset

- 원자료: <https://github.com/djangodevreng/dgx-spark-benchmarks>
- 데이터셋: <https://huggingface.co/datasets/Djangodevreng/dgx-spark-benchmarks>
- raw run, 모델 품질 메타데이터, llama-benchy/vLLM bench serve 결과를 구분

### C-12 — spark-evals

- <https://github.com/DanTup/spark-evals>
- 단일 Spark에서 실행 가능한 모델·quant의 기본 eval 결과

### C-13 — dgx-spark-hijinks benchmark report

- <https://github.com/jethac/dgx-spark-hijinks/blob/main/docs/BENCHMARKING_REPORT.md>
- SGLang, Qwen/Gemma, FP8 KV pool, Spark-class GB10의 병목 분석

### C-14 — benchmark methodology reference

- <https://github.com/Weschera/spark-bench>
- SparkBench 계열의 실행·측정 도구와 benchmark 결과 추적

### C-15 — community Qwen3.6 concurrency benchmark

- <https://www.reddit.com/r/LocalLLM/comments/1uiuxmn/qwen3635b_on_a_dgx_spark_2835_aggregate_toks_at/>
- Qwen3.6-35B-A3B의 높은 동시성 aggregate throughput 사례
- 분류: `community-reported`; 단일 사용자 decode와 혼동하지 않는다.

### C-16 — community Qwen3.5 122B benchmark

- <https://www.reddit.com/r/LocalLLaMA/comments/1sko0ft/qwen_35_122b_a10b_running_50toks_on_dgx_spark/>
- Qwen3.5-122B-A10B 단일 Spark 측정과 원본 레시피 연결

### C-17 — community 4× Spark GLM recipe

- <https://www.reddit.com/r/LocalLLaMA/comments/1uidtb8/highquality_glm52_quant_on_4x_dgx_spark_guide/>
- 4대 Spark의 GLM 계열, vLLM patch와 single-user serving 사례

### C-18 — 3 Spark cluster community experiment

- <https://www.reddit.com/r/LocalLLaMA/comments/1q8hqgd/i_clustered_3_dgx_sparks_that_nvidia_said_couldnt_be_clustered_yet/>
- 3대 mesh·NCCL 플러그인 관련 초기 실험
- 책에 활용: 3대 구성을 “공식 기본값”으로 오해하지 않게 하는 역사적 사례

### C-19 — vLLM spin-wait controlled reproduction

- <https://nacyot.github.io/artifacts/vllm-spin-wait-gb10-repro/>
- 2대 ASUS GX10의 30분·동시 4개 부하에서 CPU, TSOC, aggregate와 단독 probe를 함께 측정
- `busy_loop_s=1` 대 `0.002`; head TSOC 평균 89.6→80.5°C, vLLM CPU 합 396~400→211~214% 보고
- 1회 phase 결과와 별도 단일 스트림 손실을 분리해 읽는다.

### C-20 — GB10 clock cap field measurement

- <https://nacyot.github.io/artifacts/gb10-clock-cap/>
- ASUS GX10 단일 노드, 1Hz GPU sensor sampling, DeepSeek UD-Q2_K_XL·MiniMax-H3, 1400MHz~무제한 sweep
- DeepSeek 2000MHz cap은 약 16.6 tok/s·21.9W, cap 해제는 약 17.5 tok/s·42.6W로 보고
- 드라이버 580.173.02와 `llama-server` 조건이므로 2대 FP8 vLLM 또는 595.84 세팅 자료와 섞지 않음

### C-21 — DGX Spark OEM 열·전력 비교

- <https://www.storagereview.com/review/nvidia-dgx-spark-thermal-test-how-oem-cooling-designs-stack-up>
- NVIDIA Founders Edition, Gigabyte, Dell, Acer, ASUS를 같은 환경·GPT-OSS-120B vLLM workload로 비교
- 256/256, 4096/512, 512/4096 workload와 batch 1~128, 30초 cooldown, 1초 metric 수집
- Acer 샘플이 CPU 74.6°C·GPU 68°C로 가장 낮았고 FE·Dell·Gigabyte는 CPU 87~88°C·GPU 80~82°C로 보고됨
- end-to-end tok/s 순위가 아니며 SSD·GPU rail power 조건을 함께 읽어야 함

### D-01 — DGX Spark 저클럭·저전력 상태와 전원 완전 차단 복구

- X 게시물: <https://x.com/Blackwellboy/status/2090611479653622261?s=20>
- NVIDIA 포럼의 반복 사례: <https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-clock-pinned-at-721-mhz-under-full-load-no-throttling-not-liftable-via-nvidia-smi/376039>, <https://forums.developer.nvidia.com/t/gpu-clock-bug-looks-like-5-min-wait-is-enough/376239>
- 커뮤니티 보고값은 약 799MHz·19.5W·44 tok/s에서 완전 전원 차단 뒤 약 2.3~2.5GHz·92W·73.9 tok/s로 회복된 사례다.
- `P0`, 높은 GPU utilization, 비활성 throttle reason만으로 정상 상태라고 판정하지 않는다. SM clock과 power draw를 부하 중 함께 기록한다.
- 원인은 게시물과 포럼에서 전원 공급 또는 USB PD 상태로 추정하지만, 책에서는 `suspected`로 표시한다. 공식 원인 확정으로 쓰지 않는다.

### D-02 — MCDMA: Metal CUDA Direct Memory Access

- X 게시물: <https://x.com/ashxhart/status/2089749434087227672?s=20>
- 재게시본: <https://site.twstalker.com/ashxhart/status/2089749434087227672>
- 작성자는 Apple Silicon Mac과 Spark 사이 USB-C 링크에서 registered memory, rkey, one-sided READ/WRITE, SEND/RECV를 구현했다고 주장한다.
- 보고된 전송값은 단일 링크 939MB/s, Mac에서 두 Spark로 동시 전송 1.80GB/s, 두 Spark에서 Mac으로 동시 전송 1.25GB/s, 왕복 지연 24μs다.
- Spark 1과 Spark 2는 CX-7로 prompt processing을 수행하고, 두 USB-C 링크로 Mac Studio에 decode를 분산하는 구성을 설명한다.
- 조사 시점에 공개 구현 저장소와 독립 재현 토큰 벤치마크는 확인하지 못했다. 따라서 MCDMA는 표준 RoCE·NCCL·MLX/JACCL 경로가 아니라 **검증 대기 중인 커뮤니티 프로토타입**으로 분류한다.

### D-03 — 2× DGX Spark clock cap X 보고

- <https://x.com/ivanfioravanti/status/2088730630875930639?s=20>
- MiaAI-Lab DeepSeek V4 Flash 0731 NVFP4 recipe를 c4로 실행하며 `2455/2300/2200/2000/1800MHz`의 `nvtop` 표시 전력·온도를 비교한 현장 보고
- 보고값은 약 `47/34/32/27/23`과 `72/67/66/63/61°C`; 게시자는 벽면 AC 전력이 더 높다고 명시
- `sudo nvidia-smi -lgc 0,2200` 적용과 `sudo nvidia-smi -rgc` rollback을 제시한다. 분류: `D`; GPU rail과 wall power를 섞지 않고, 독립 재현 전에는 reference report로만 사용

### D-04 — ASUS Ascent GX10 노드 세팅 현장 보고

- <https://nacyot.github.io/artifacts/dgx-spark-node-setup/>
- ASUS Ascent GX10(GB10) 세 대의 서버 모드 전환, 플랫폼 패키지 보호, 드라이버·USB-C PD 펌웨어, GPU/CPU clock cap, vLLM spin, ConnectX-7, Ollama Qwen3.8, DeepSeek EXL3 세팅 기록
- `MemTotal` 약 121.6GiB와 서버 모드 `MemAvailable` 약 119GiB, 200G 링크 협상과 payload 차이, 2000MHz clock cap, 단일 Spark 레시피 수치를 함께 다룸
- 분류: `D/C`; 작성자 장비의 현장 측정과 절차. ASUS GX10 전용 패키지·펌웨어·커널·인터페이스 이름을 NVIDIA DGX Spark 전체의 공통값으로 복사하지 않음
- 책 반영 노트: [ASUS GX10 노드 세팅 자료 검토](dgx-spark-node-setup-research-2026-08.md)

### D-05 — ASUS Ascent GX10 노드 세팅 영문판

- <https://nacyot.github.io/artifacts/dgx-spark-node-setup-en/>
- D-04의 영문판. 독립 장비·벤치마크 자료로 중복 집계하지 않고 영어 원문 링크로 보존

### D-06 — GB10 clock cap 원자료

- <https://nacyot.github.io/artifacts/gb10-clock-cap/>
- ASUS GX10 단일 노드에서 1Hz GPU sensor sampling, DeepSeek UD-Q2_K_XL·MiniMax-H3, 1400MHz~무제한 sweep
- DeepSeek 2000MHz cap은 약 16.6 tok/s·21.9W, cap 해제는 약 17.5 tok/s·42.6W로 보고
- 드라이버 580.173.02와 `llama-server` 조건이므로 2대 FP8 vLLM 또는 595.84 세팅 자료와 섞지 않음
- 분류: `C/D`; 작성자 장비의 측정값이며 GPU rail power와 wall AC power를 구분

### D-07 — Lenovo Project Kubit X 게시물

- <https://x.com/thinkstations/status/2024514312312647851>
- 두 ThinkStation PGX를 개인 AI 허브로 묶는 제품 개념
- 분류: `D`; 활용 방향·제품 홍보 자료이며 처리량 benchmark가 아님

### D-08 — X의 Mac Studio 클러스터 요약

- <https://x.com/i/trending/2001731662288486469>
- Thunderbolt·Exo로 여러 Mac Studio를 묶는 사례 요약
- X의 2차 요약과 Grok 경고가 포함되어 있으므로 원문·재현 자료 확인 전 정량 근거로 사용하지 않음

### D-09 — X local hardware ecosystem mentions

- <https://x.com/0xSero/status/2039742489276395818>
- RTX·MacBook·DGX Spark에서 local model을 사용하는 흐름을 언급
- 분류: `D`; 특정 vendor의 사양·성능 비교가 아님

## 9. 연구 논문·기술 배경

논문은 DGX Spark 레시피의 직접 실행 명령보다, 다중 노드·커널·긴 문맥을 설명하는 배경 자료로 사용한다.

### F-08 — Dual-Node NVIDIA DGX Spark over Tailscale

- <https://arxiv.org/abs/2608.07226>
- 2대 DGX Spark의 원격 관리, 분산 학습, Tailscale, QSFP 연결, 재현 스크립트

### F-09 — Kernel Forge on DGX Spark

- <https://arxiv.org/abs/2607.24762>
- DGX Spark/GB10에서 CUDA kernel 생성·최적화와 Qwen/Gemma 평가

### F-10 — DeepSeek V4 Flash model report

- 모델 카드가 가리키는 공식 논문: <https://arxiv.org/abs/2606.19348>
- DeepSeek V4 계열의 아키텍처·평가·모델 정체 확인용

### F-11 — FlashMemory-DeepSeek-V4

- <https://arxiv.org/abs/2606.09079>
- DeepSeek V4 긴 문맥의 KV 메모리·sparse attention 배경
- 책에 활용: 1M context를 단순히 “메모리에 모델이 들어간다”로 설명하지 않기 위한 배경

### F-12 — Six Times to Spare: LDPC Acceleration on DGX Spark

- <https://arxiv.org/abs/2602.04652>
- DGX Spark의 CPU/GPU 협업과 비-LLM workload 사례
- 책의 부록 “LLM 외에 무엇을 할 수 있나” 후보

## 10. Wikidocs·한국어 자료

현재 Wikidocs에서 확인되는 DGX Spark 자료는 전용 단행본보다는 뉴스·벤치마크·모델 사용기 성격이다.

### K-00 — Wikidocs GitHub 연동 공식 안내

- <https://wikidocs.net/321336>
- 기존 GitHub 저장소 연결, OAuth 승인, `README.md`·`TOC.md`·`pages/`·`assets/` 구조, push webhook과 수동 동기화 절차
- 책에 활용: 본 원고를 `recrack/oh-my-dgx-spark-wikidocs`에 분리하고 private 책으로 연결하는 배포 기준

### K-01 — 로컬에서 14B LLM을 돌리면 어떤 일이 벌어질까

- <https://wikidocs.net/blog/%40jaehong/11406/>
- vLLM, warm-up, 1.5B/14B 실측, NVFP4와 메모리 대역폭 해석

### K-02 — Qwen3.8-27B 실전 사용기

- <https://wikidocs.net/blog/%40openwiki/28757/>
- Qwen3.8, MTP, reasoning effort, DGX Spark와 M5 Max 비교 사례

### K-03 — Qwen3.8-27B 기술 해설

- <https://wikidocs.net/blog/%40jaehong/28556/>
- hybrid attention, GQA, MTP, context 구조 설명

### K-04 — 로컬 AI, 왜 지금인가

- <https://wikidocs.net/blog/%40jaehong/23387/>
- 로컬 AI·에이전트·DGX Spark·Exo Labs 최적화 사례의 맥락

### K-05 — AMD Ryzen AI Halo와 DGX Spark 비교

- <https://wikidocs.net/blog/%40jaehong/22315/>
- DGX Spark와 Strix Halo의 가격·메모리·CUDA 생태계 비교

### K-06 — 아카라이브 알파카 게시판

- 게시판: <https://arca.live/b/alpaca>
- 사용자 제공 원문: <https://arca.live/b/alpaca/180567610?p=1>
- 분류: `D`; 모델·속도·구성 경험을 발견하는 용도. 게시글 원문과 첨부 이미지, 댓글을 수동 보존한 뒤 인용

### K-07 — 서버포럼 AI 게시글

- <https://svrforum.com/ai/3170124>
- <https://svrforum.com/ai/3174599>
- 분류: `D`; 국내 사용자 경험·가격·구성·실패 사례 후보. 본문 접근 시점과 작성자 측정 조건을 함께 기록

### K-08 — NVIDIA 한국 DGX Spark 페이지

- <https://www.nvidia.com/ko-kr/products/workstations/dgx-spark/>
- 한국어 사양·공식 포지셔닝·플레이북 링크

### K-09 — 국내 DGX Spark 가격 스냅샷

- <https://search.danawa.com/mobile/dsearch.php?keyword=dgx+spark>
- 국내 Founders Edition·OEM·해외구매 가격을 조회일 기준으로 비교하는 시장 자료
- 재고·환율·세금·판매처에 따라 변하므로 구매 추천가로 사용하지 않음

### K-10 — 국내 200G/400G 스위치 판매 예시

- <https://www.fibermart.co.kr/goods/view?no=16351>
- 200G·400G QSFP 포트를 가진 스위치의 국내 판매가와 사양을 확인하는 예시
- DGX Spark 4대용 공식 호환성을 확인한 제품 목록이 아니므로 예산 참고로만 사용

## 11. 커뮤니티 자료를 읽을 때의 주의점

커뮤니티 자료는 실제 문제를 빨리 발견하는 데 매우 유용하지만, 다음 항목을 확인하지 않은 숫자는 책의 결론으로 쓰지 않는다.

- 모델 원본인지, prune/abliterated/quantized 파생인지
- 총 파라미터와 active 파라미터를 혼동하지 않았는지
- BF16, FP8, NVFP4, INT4, EXL3, GGUF 중 무엇인지
- vLLM, SGLang, llama.cpp, TensorRT-LLM, SparkInfer 중 어느 엔진인지
- prompt length, output length, context, batch, concurrency
- MTP/DFlash/DSpark 등 speculative decoding 사용 여부
- 단일 요청 decode인지, aggregate throughput인지
- OS·driver·CUDA·container·commit이 고정되어 있는지
- 모델이 실제로 응답한 것인지, 단순히 메모리에 로드된 것인지
- function calling·vision·long context가 실제 테스트된 것인지

특히 `fits in 128 GB`, `runs`, `usable`, `supports 1M context`, `agent-ready`는 서로 다른 주장이다. 책에서는 다음 상태를 별도의 열로 기록한다.

| 상태 | 의미 |
|---|---|
| `loads` | 모델 파일과 런타임이 메모리에 올라옴 |
| `generates` | 기본 텍스트 생성이 성공함 |
| `serves` | OpenAI-compatible endpoint가 지속적으로 응답함 |
| `benchmarked` | 고정 조건의 반복 측정이 있음 |
| `tool-tested` | tool parser와 function call을 실제 확인함 |
| `agent-tested` | 다단계 도구 루프·실패 복구까지 확인함 |
| `long-context-tested` | 지정 context에서 retrieval/품질·안정성을 확인함 |

## 12. 기존 자료와 우리 책의 빈 공간

| 영역 | 기존 자료 상태 | 우리 책의 기회 |
|---|---|---|
| 하드웨어·첫 부팅 | NVIDIA User Guide가 충분함 | 초보자용 체크리스트와 실제 오류만 압축 |
| 단일 노드 도구 | 공식 플레이북·중국어 웹북·영문 책이 있음 | “어떤 목적에 어떤 런타임인가” 선택표 필요 |
| 단일 노드 모델 | Qwen·Nemotron·DeepSeek 레시피가 빠르게 생김 | 같은 프롬프트·조건의 한국어 비교가 부족 |
| 2대 클러스터 | DeepSeek TP=2와 Kubernetes 웹북이 있음 | 1M/256K, 비용·전력·운영 난이도 비교 필요 |
| 3대 클러스터 | 공식 ring/스위치 경로와 커뮤니티 실험이 공존 | 공식·실험·서비스 분리를 명확히 설명할 필요 |
| 4대 클러스터 | Qwen/GLM/Nemotron 사례와 NVIDIA switch playbook이 있음 | 실제 대형 단일 모델과 독립 서비스의 선택 기준 필요 |
| 벤치마크 | SparkBench·NVIDIA guide·여러 GitHub 결과가 있음 | 재현 가능한 한국어 prompt set과 실패율 기록 필요 |
| 에이전트 | Hermes/OpenClaw/NemoClaw 공식 레시피가 있음 | 보안·권한·로컬 데이터 경계까지 포함한 안전한 운영 레시피 필요 |
| 한국어 책 | DGX Spark 전용 Wikidocs 단행본은 확인되지 않음 | 한국어로 “사서 무엇을 돌릴지”에 답하는 첫 책이 될 수 있음 |

따라서 우리 책은 단순 설치 매뉴얼이 아니라 다음 질문에 답하는 방향으로 구성한다.

> **Spark가 몇 대 필요한가? 어떤 모델을 어떤 엔진으로 돌려야 하는가? 실제 속도와 실패 가능성은 어느 정도인가?**

## 13. 후속 조사·재현 큐

### 우선순위 높음

- [ ] 공식 Qwen3.6-35B-A3B-NVFP4 vLLM 레시피를 현재 DGX OS/driver에서 재현
- [ ] Qwen3.8 27B를 BF16, FP8, NVFP4/SGLang으로 동일 prompt set 비교
- [ ] 단일 Spark DeepSeek SparkInfer 경로에서 `loads / generates / serves / benchmarked` 상태 분리
- [ ] 2대 DeepSeek V4 Flash 0731의 256K와 1M profile을 같은 요청으로 비교
- [ ] 2대 direct QSFP에서 NCCL bandwidth와 vLLM/SGLang 통신 오버헤드 측정
- [ ] 3대 ring과 4대 switch의 setup 시간·실패 지점·복구 절차 기록
- [ ] 4대 Qwen3.5-397B/GLM 계열은 모델·vLLM patch·commit을 고정해 재현
- [ ] OpenClaw/Hermes/NemoClaw에서 tool call, 파일 권한, 외부 네트워크 정책을 별도 검증
- [ ] MCDMA 공개 구현이 나오면 코드·USB-C protocol·byte integrity·대용량 전송을 독립 재현
- [ ] 저클럭 장애에서 전원 차단 전후의 SM clock, power, BF16/모델 decode를 같은 하니스로 비교

### 다음 단계

- [ ] 동일 prompt set을 한국어·코딩·JSON·vision·long-context·tool-call로 고정
- [ ] 모든 benchmark에 TTFT, prefill tok/s, decode tok/s, ITL, aggregate tok/s, error rate 기록
- [ ] idle/serving/peak 전력과 시스템 온도·팬·메모리 압박을 함께 기록
- [ ] 모델 로딩 시간과 첫 요청 시간, 캐시 hit/miss를 분리
- [ ] 공개 숫자와 우리 장비 숫자를 `vendor / community / local`로 표기
- [ ] 링크가 삭제·변경될 경우 제목, 저장소, commit, 접근일을 남겨 provenance 보존

## 14. 우리 저장소의 관련 원고·실험

- [DGX Spark 모델·클러스터 리서치](dgx-spark-cluster-model-research-2026-08.md)
- [Qwen3.8-27B-OBLITERATED 리서치 노트](model-research-qwen38-obliterated.md)
- [2026-08-21 smoke test 결과](test-results-2026-08-21.md)
- [DGX Spark·Mac 혼합 구성과 스위치 리서치](dgx-spark-mac-rdma-switch-research-2026-08.md)
- [책 부록 — 명령어와 결과 양식](../book/appendix-a-commands.md)
- [GPT-5.6 Sol max 비교 리서치](sol-max-comparison-research-2026-08.md)
- [Wikidocs 배포 bundle](../wikidocs/)
- [2026-08-22 DeepSeek C1 raw 결과](results-deepseek-c1-2026-08-22.json)
- [GitHub Issue #1 — X 북마크 기반 실사용·성능·비용 자료](https://github.com/recrack/oh-my-dgx-spark/issues/1)

## 15. NVIDIA Developer Forum 전체 리서치

- [DGX Spark/GB10 포럼 리서치](dgx-spark-nvidia-forum-research-2026-08.md)
- [NVIDIA Developer Forum — DGX Spark / GB10](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)
- 범위: 2026-08-21 기준 약 2,305개 토픽 인벤토리와 조회수·활동·키워드 기반 대표 본문 검토
- 모델: Qwen3.5/3.6/3.8, DeepSeek V4 Flash 0731, GLM, MiniMax, MiMo, Nemotron, LongCat, Gemma
- 시스템: 1/2/3/4/8대 구성, TP/PP/EP, direct QSFP/RoCE, switch, NCCL, RDMA, ConnectX-7
- 운영: unified-memory OOM, thermal shutdown, power drain recovery, firmware updater/NIC 장애
- 에이전트·평가: Hermes, OpenClaw, NemoClaw, Tool Eval Bench, Toolery, Spark Arena
- 등급: NVIDIA 공식 자료, 실행 가능한 community recipe, 조건이 명시된 measurement, anecdote, issue, opinion을 구분

이 문서는 책의 참고문헌 인덱스이며 각각의 원문을 대신하지 않는다. 최종 출간 전에는 각 링크의 최신 버전, 라이선스, 모델 카드, 실행 결과를 다시 확인한다.
