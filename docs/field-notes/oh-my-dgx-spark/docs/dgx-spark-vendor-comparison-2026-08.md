# DGX Spark·GB10 벤더 비교 리서치

기준일: **2026-08-22**
목적: NVIDIA DGX Spark Founders Edition과 같은 GB10 플랫폼을 사용하는 파트너 장비, 그리고 GB10과는 다른 메모리 대형 로컬 AI 장비를 구분해 비교한다.

## 먼저 결론

“DGX Spark와 다른 벤더 제품”은 두 종류로 나눠 읽어야 한다.

1. **같은 NVIDIA GB10 플랫폼의 OEM 시스템**: Acer Veriton GN100, ASUS Ascent GX10, Dell Pro Max with GB10, GIGABYTE AI TOP ATOM, HP ZGX Nano, Lenovo ThinkStation PGX, MSI EdgeXpert다. NVIDIA는 이 7개를 DGX Spark/Grace Blackwell GB10 인증 시스템으로 현재 열거한다.
2. **다른 아키텍처의 대안**: AMD Ryzen AI Halo/Strix Halo, Apple Mac Studio 같은 장비다. 메모리 용량은 경쟁할 수 있지만 CUDA·DGX OS·ConnectX-7·NCCL 레시피가 같지 않으므로 GB10 클러스터의 노드로 바로 섞으면 안 된다.

같은 GB10을 쓴다고 해서 모든 장비가 같은 제품은 아니다. 모델 메모리와 메모리 대역폭, CPU/GPU 구조, ConnectX-7, NVIDIA 소프트웨어 스택은 공통 기반에 가깝지만, 저장장치 SKU, 냉각, 전원 어댑터, 펌웨어 패키지, 포트 구현, 지원 계약과 초기 이미지가 달라진다. 따라서 벤더 선택은 “어느 제품이 1 PFLOP인가”보다 **어떤 냉각·지원·스토리지·클러스터 운영 조건을 살 것인가**의 문제다.

현재 확인한 자료만으로는 7개 OEM의 동일 모델·동일 런타임·동일 장시간 LLM 성능 순위를 만들 수 없다. 다섯 장비를 같은 방에서 시험한 독립 열·전력 비교는 있지만, 그 자료도 end-to-end tok/s 순위를 제공하지 않는다. 이 문서에서는 공식 사양, 독립 측정, X·포럼의 사용기를 같은 표에 섞지 않는다.

## 1. 비교 범위와 증거 등급

| 등급 | 자료 | 이 책에서의 사용 |
|---|---|---|
| A | NVIDIA 인증 목록·공식 사용자 가이드·제조사 제품 문서 | 플랫폼·포트·메모리·지원 범위 |
| B | 공개된 설치·클러스터·운영 recipe | 재현 후보. 장비에서 다시 실행 |
| C | 측정 조건과 raw 수치가 공개된 독립 벤치마크 | 조건을 붙인 비교 |
| D | X·Reddit·NVIDIA Forum 사용기 | 실패 사례·실험 주제·현장 가설 |

제조사의 “최대 200B”, “최대 1 PFLOP”, “두 대로 405B”는 제품 포지셔닝과 공식 사양이다. 이것만으로 특정 모델의 실제 생성 속도, 1M context 성공, tool call 성공을 뜻하지 않는다.

## 2. NVIDIA가 인증한 GB10 시스템

NVIDIA의 [NVIDIA-Certified Systems 목록](https://docs.nvidia.com/certification-programs/latest/nvidia-certified-systems.html)은 다음 7개를 `DGX Spark / Grace Blackwell GB10 systems`로 명시한다.

| 제조사 | 제품 | NVIDIA가 인증한 GPU | 이 장비를 고를 때 볼 점 |
|---|---|---|---|
| NVIDIA | DGX Spark | GB10 | Founders Edition 기준 장비, NVIDIA 문서·플레이북의 기준점 |
| Acer | Veriton GN100 AI Mini Workstation GN100-UD11 | GB10 | 4TB 구성, PyTorch·Jupyter·Ollama, 4대 switch 확장 안내 |
| ASUS | Ascent GX10 | GB10 | QuietFlow 냉각, ASUS 지원·현장 recipe, 1/2/4TB SKU |
| Dell Technologies | Pro Max with GB10 | GB10 | 2/4TB SKU, 280W 어댑터, Dell 지원 계약, 데스크톱/네트워크 appliance 운영 |
| GIGABYTE | AI TOP ATOM ATAGB10-9000 계열 | GB10 | AI TOP Utility, 1/4TB 계열 SKU, 로컬 모델·RAG 작업 흐름 |
| HP | ZGX Nano AI Station | GB10 | HP ZGX Toolkit, 2/4TB SED, DGX OS 전용, 기업형 지원·엣지 배치 |
| Lenovo | ThinkStation PGX Workstation | GB10 | ThinkStation 서비스, SED·TPM·Secure Boot·FW Recovery 등 보안 문서 |
| MSI | EdgeXpert MS-C931 | GB10 | self-encrypting SSD, Docker Compose·SSL·private CA를 포함한 엔터프라이즈/엣지 방향 |

인증은 “GB10 기반 시스템으로 기능·성능·확장성·보안을 평가했다”는 의미이지, 서로 다른 OEM 사이의 모든 드라이버·펌웨어·열 설계가 동일하다는 뜻은 아니다.

## 3. 공통으로 기대할 수 있는 기반

NVIDIA [DGX Spark 제품 사양](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)과 [하드웨어 사용자 가이드](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)를 기준으로 보면 기준 플랫폼은 대략 다음과 같다.

| 항목 | 공통 기준 | 주의할 점 |
|---|---|---|
| SoC | NVIDIA Grace Blackwell GB10, 20-core Arm CPU + Blackwell GPU | clock·firmware·냉각은 장비별로 확인 |
| 통합 메모리 | 128GB LPDDR5x, 256-bit, 273GB/s | OS·KV cache·workspace가 함께 사용 |
| AI 표기 성능 | 최대 1 PFLOP FP4, sparsity 조건 | 실제 모델 tok/s·품질과 다른 지표 |
| 연결 | 2× QSFP ConnectX-7, 포트당 최대 200Gb/s 표기 | 케이블·RoCE·NCCL·Docker passthrough를 따로 검증 |
| 관리망 | 10GbE, Wi-Fi 7 등 | hostname·인터페이스 이름은 OEM/이미지에서 확인 |
| OS | NVIDIA DGX OS 계열 | HP·ASUS·Lenovo 등도 DGX OS를 기본/권장으로 표기하지만 이미지 릴리스는 다를 수 있음 |
| 전원 | 기준 장비는 240W 외장 어댑터, GB10 SoC TDP 140W | Dell은 280W 어댑터를 표기하며, 벽면 전력과 GPU rail을 혼동하지 않음 |

`128GB`를 그대로 모델 용량으로 읽으면 안 된다. 모델 weight, KV cache, CUDA graph/workspace, OS, container와 서비스가 하나의 unified memory를 나눈다. 이 규칙은 NVIDIA 제품뿐 아니라 GB10 OEM에도 동일하게 적용한다.

## 4. 제조사별 차이

### 4.1 NVIDIA DGX Spark Founders Edition

NVIDIA 제품은 이 책의 기준 장비다. 공식 문서와 [DGX Spark 플레이북](https://github.com/NVIDIA/dgx-spark-playbooks)이 가장 직접적으로 연결되고, 2대 direct, 3대 ring, switch를 이용한 다중 노드 문서를 같은 브랜드 기준으로 따라가기 쉽다.

- 공식 사양: 128GB unified memory, 273GB/s, 1TB 또는 4TB self-encrypting NVMe SKU, ConnectX-7, 240W 어댑터
- 강점: NVIDIA 릴리스 노트·DGX OS·NCCL·NGC recipe의 기준점
- 주의: Founders Edition도 저클럭·전원·열 문제에서 예외가 아니다. 부하 중 `clocks.sm`, `power.draw`, 온도와 실제 tok/s를 함께 기록한다.

### 4.2 Acer Veriton GN100

[Acer 공식 발표](https://news.acer.com/acer-unveils-the-veriton-gn100-ai-mini-workstation-built-on-the-nvidia-gb10-superchip)는 128GB unified memory, 최대 4TB NVMe, GB10, NVIDIA AI software stack, PyTorch·Jupyter·Ollama를 명시한다. Acer 제품 페이지는 두 대 direct 연결과 200GbE switch를 이용한 최대 네 대 연결 경로도 안내한다.

- 고를 이유: 4TB 저장장치와 일반 개발 도구를 포함한 온프레미스 개발 장비 포지셔닝
- 공식 가격 자료: 2025년 발표 당시 북미 시작가 USD 3,999, EMEA EUR 3,999, 호주 AUD 6,499. 현재 가격이나 국내 실구매가는 아니다.
- 독립 측정: StorageReview의 다섯 장비 열 비교에서 Acer 샘플은 CPU·GPU·NVMe·NIC 온도가 모두 가장 낮게 측정됐다.
- 주의: 해당 열 결과는 특정 Acer 샘플과 특정 SSD·환경의 결과다. 모든 GN100 SKU가 같은 온도나 tok/s를 보장한다는 뜻은 아니다.

### 4.3 ASUS Ascent GX10

[ASUS 공식 제품 페이지](https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/)는 GB10, 128GB unified memory, 1 PFLOP, NVLink-C2C, ConnectX-7, DGX OS, NVIDIA AI stack과 `QuietFlow` 냉각(3개 팬·dual vapor chamber)을 강조한다. ASUS FAQ는 1/2/4TB SSD와 240W peak system power를 설명한다.

- 고를 이유: 이 책에 포함된 ASUS GX10 현장 자료와 실제 DeepSeek/Qwen 다중 Spark recipe가 많다.
- 공식 지원 정보: ASUS FAQ는 QSFP direct 연결과 switch를 이용한 다중 노드를 언급한다. FAQ 안에서 3대와 4대 이상에 대한 답변 표현이 서로 달라, 구매 결정에서는 현재 펌웨어·switch recipe를 별도로 확인한다.
- 우리 자료의 위치: [ASUS GX10 노드 세팅 리서치](dgx-spark-node-setup-research-2026-08.md)는 ASUS 장비의 현장 보고서이지, NVIDIA Founders Edition 전체의 공통 설치 매뉴얼이 아니다.
- 주의: ASUS의 드라이버, USB-C PD 펌웨어, 패키지 보호, 인터페이스 이름을 다른 GB10 장비에 그대로 복사하지 않는다.

### 4.4 Dell Pro Max with GB10

[Dell 공식 제품 페이지](https://www.dell.com/en-sg/shop/pcs-desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro)는 128GB·273GB/s, 2TB QLC 또는 4TB self-encrypting SSD, 2×200G QSFP ConnectX-7, 10GbE, DGX OS를 명시한다. 다른 OEM과 달리 페이지에 280W USB-C 어댑터와 Dell의 1/2/3년 Basic Hardware·ProSupport 계열 선택지가 구체적으로 표시된다.

- 고를 이유: 기업 구매, 현장 교체, 지원 계약, 기존 Dell 자산 관리가 중요한 경우
- 운영 특징: [Dell 초기 설정 문서](https://www.dell.com/support/kbdoc/en-ap/000398800/dell-pro-max-gb10-fcm1253-initial-setup-instructions)는 Desktop Mode와 Network Appliance Mode를 구분한다.
- 주의: Dell 페이지의 SKU와 가격은 지역·계약·스토리지에 따라 변한다. 280W 어댑터를 다른 240W 장비의 성능 차이로 곧바로 해석하지 말고 벽면 AC와 rail power를 따로 잰다.

### 4.5 GIGABYTE AI TOP ATOM

[GIGABYTE 공식 AI TOP ATOM 페이지](https://www.gigabyte.com/AI-TOP-PC/GIGABYTE-AI-TOP-ATOM)는 GB10, 최대 200B 모델, 1 PFLOP FP4, 128GB unified memory, 최대 4TB NVMe, ConnectX-7, NVIDIA AI software stack을 명시한다. GIGABYTE의 차별점은 하드웨어보다 [AI TOP Utility](https://www.gigabyte.com/AI-TOP-PC/GIGABYTE-AI-TOP-ATOM)를 이용한 모델 다운로드·추론·RAG·머신러닝 흐름을 앞세운다는 점이다.

- 고를 이유: 명령줄만이 아니라 로컬 모델 관리와 RAG를 한 제품 흐름으로 묶고 싶은 경우
- SKU: 제품군 페이지에는 ATAGB10-9000 계열이 표시되며 지역별 1TB·4TB 구성이 다를 수 있다.
- 독립 측정: StorageReview에서는 Gigabyte가 비교군 중 GPU 전력은 높았지만 CPU 온도는 기준 장비·Dell과 비슷했고, NIC 온도는 Dell·ASUS보다 낮았다.
- 주의: AI TOP Utility의 편의성과 CUDA recipe의 호환성은 별도다. 책의 vLLM/SGLang benchmark에는 엔진·commit·모델 revision을 고정한다.

### 4.6 HP ZGX Nano AI Station

[HP QuickSpecs](https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c09212373)는 128GB LPDDR5x unified memory, 273GB/s, 2TB/4TB 계열 M.2, 2×200GbE QSFP, 10GbE, 240W USB-C 어댑터를 명시한다. HP는 [ZGX Toolkit](https://www.hp.com/us-en/workstations/zgx-nano-ai-station.html)과 네트워크로 연결한 Windows·Mac·Linux 클라이언트에서의 개발·배치를 강조한다.

- 고를 이유: HP Workstation 지원 체계, ZGX Toolkit, 기업·엣지 배치가 중요한 경우
- 중요한 제한: QuickSpecs는 장비 자체가 Microsoft Windows를 지원하지 않고 NVIDIA DGX OS/Ubuntu 계열을 사용한다고 명시한다. Windows·Mac·Linux는 클라이언트로 이해해야 한다.
- 물리적 제한: HP QuickSpecs는 rack mounting을 지원하거나 인증하지 않는다고 적는다. 소형 서버 랙에 넣을 계획이면 별도 검증이 필요하다.
- 주의: HP의 toolkit이 제공하는 모델·배포 기능과 NVIDIA 공식 플레이북의 지원 상태를 같은 것으로 취급하지 않는다.

### 4.7 Lenovo ThinkStation PGX

[Lenovo ThinkStation PGX 제품 가이드](https://lenovopress.lenovo.com/lp2321-thinkstation-pgx)는 AI 개발 전용 workstation, DGX OS, NVIDIA AI software stack, PyTorch·Jupyter, 128GB·273GB/s, 1TB/4TB self-encrypting NVMe, 2× ConnectX-7 QSFP와 240W 전원을 명시한다.

- 고를 이유: ThinkStation·Premier Support, 보안과 자산 관리, 기존 Lenovo workstation 표준화
- 보안 문서: self-encrypting NVMe, TPM 2.0, NVLink-C2C enclave, NVIDIA FW Recovery, AMI setup password, UEFI Secure Boot가 명시되어 있다.
- 클러스터: 두 PGX를 ConnectX-7로 묶어 최대 405B 모델을 다루는 구성을 공식 가이드에 넣고 있다.
- X 자료: [Lenovo Workstations의 Project Kubit 게시물](https://x.com/thinkstations/status/2024514312312647851)은 두 PGX를 이용한 개인 AI 허브 개념을 보여주지만, 처리량 측정이나 제품 보증 수치가 아니다.

### 4.8 MSI EdgeXpert MS-C931

[MSI IPC 공식 사양](https://ipc.msi.com/product_detail/EdgeXpert-MS-C931)은 GB10, 20-core Arm, 128GB LPDDR5x unified memory, 273GB/s, 1TB/4TB self-encrypting NVMe, 10GbE, 2×QSFP ConnectX-7, Wi-Fi 7, DGX OS, 약 240W USB-C 전원을 명시한다.

- 고를 이유: 일반 개발자용 미니 PC보다 엔터프라이즈·산업·엣지 AI appliance를 목표로 하는 경우
- 소프트웨어 방향: MSI 페이지의 기술 사양은 Docker Compose, DGX OS, 1~2노드 수평 확장, block-level encryption, ingress SSL, private CA 기반 secrets 관리를 언급한다.
- 주의: 이는 MSI의 솔루션/제품 페이지에 적힌 운영 방향이다. 특정 Docker Compose stack이 모든 NVIDIA recipe와 동일한 성능이나 안정성을 제공한다는 뜻은 아니다.

## 5. 실제 성능에서 OEM 차이가 생기는 지점

### 5.1 독립 열·전력 비교

[StorageReview의 2026-01-25 비교](https://www.storagereview.com/review/nvidia-dgx-spark-thermal-test-how-oem-cooling-designs-stack-up)는 NVIDIA Founders Edition, Gigabyte, Dell, Acer, ASUS 다섯 장비를 같은 환경에 놓고 vLLM의 `GPT-OSS-120B`를 사용했다. 256/256 equal, 4096/512 prefill-heavy, 512/4096 decode-heavy 세 workload를 batch 1·2·4·8·16·32·64·128로 시험하고, 단계 사이에 30초 냉각 시간을 넣었다. 최신 NVIDIA Ubuntu 이미지를 사용하고 1초 간격으로 kernel interface와 `nvidia-smi`를 읽었다.

기사에 공개된 핵심 관찰은 다음과 같다.

| 항목 | 관찰 |
|---|---|
| CPU | Acer 샘플의 prefill-heavy 최고점 74.6°C. Founders Edition·Dell·Gigabyte는 약 87~88°C, ASUS는 그 중간 |
| GPU | Acer 68°C, 나머지 네 장비는 약 80~82°C |
| NVMe | Acer 51.8°C, 다른 장비는 약 58~63°C 범위. SSD 모델이 달라 완전한 동등 비교는 아님 |
| ConnectX-7 | Acer 62°C, Founders Edition 75°C. Gigabyte는 Dell·ASUS보다 낮은 NIC 온도 |
| GPU rail power | prefill-heavy peak 약 69.3W(Acer)~76.0W(Gigabyte) |
| tok/s 순위 | 해당 글은 열·전력 비교이며 end-to-end LLM tok/s 순위를 공개하지 않음 |

따라서 “Acer가 무조건 가장 빠르다”가 결론이 아니다. 이 자료가 강하게 보여주는 것은 **동일 GB10 계열이어도 냉각 설계가 지속 부하의 온도 여유를 바꾼다**는 점이다. 장시간 prefill, fine-tuning, 여러 사용자 요청에서는 온도와 clock cap이 실제 속도에 영향을 줄 수 있지만, 그 연결을 모델별 tok/s로 확정하려면 별도의 동일 조건 benchmark가 필요하다.

### 5.2 이 책에 이미 있는 ASUS·NVIDIA 비교의 범위

우리 저장소에는 ASUS GX10을 사용한 DeepSeek V4 Flash 0731 FP8 2대 recipe, clock cap, spin-wait 실험과 NVIDIA reference Spark와의 커뮤니티 비교가 있다. 이것은 벤더 차이를 알아보는 데 유용하지만 다음 조건이 다르다.

- 모델·quant: DeepSeek FP8, EXL3, GGUF를 섞지 않는다.
- 엔진: vLLM, llama.cpp, SparkInfer를 섞지 않는다.
- 노드·토폴로지: single, TP=2, dual rail, switch를 구분한다.
- 전력: `nvidia-smi` GPU rail과 벽면 AC를 구분한다.
- 결과: ASUS 한 장비에서 얻은 현장 측정은 모든 GX10과 Founders Edition의 공통 성능표가 아니다.

## 6. X에서 확인한 자료의 위치

X는 신제품 사용자가 실제 구성과 문제를 빠르게 공유하는 장소지만, 링크가 사라지거나 조건이 생략되기 쉬우므로 공식 사양과 분리한다.

| 자료 | 확인한 내용 | 판정 |
|---|---|---|
| [Lenovo Workstations Project Kubit](https://x.com/thinkstations/status/2024514312312647851) | 두 ThinkStation PGX를 이용한 개인 AI 허브 개념 | D. 제품 방향·활용 사례. benchmark 아님 |
| [Ivan Fioravanti의 clock cap 보고](https://x.com/ivanfioravanti/status/2088730630875930639?s=20) | 2× Spark에서 2455~1800MHz의 온도·전력 변화를 DeepSeek c4로 표시 | D. clock/전력 실험. OEM 순위 아님 |
| [Ash Hart의 MCDMA](https://x.com/ashxhart/status/2089749434087227672?s=20) | Mac Studio와 Spark 사이 USB-C 직접 메모리 접근 주장, 2 Spark+Mac 구성 | D. 공개 재현·토큰 benchmark 전에는 프로토타입 |
| [0xSero의 local SOTA 언급](https://x.com/0xSero/status/2039742489276395818) | RTX·Mac·DGX Spark를 같은 로컬 모델 생태계로 언급 | D. 플랫폼 소개. 사양·성능 근거 아님 |
| [X의 Mac Studio 클러스터 요약](https://x.com/i/trending/2001731662288486469) | Thunderbolt·Exo 기반 Mac 여러 대 구성 소개 | D. X의 2차 요약이며 원문·조건을 다시 확인해야 함 |

X의 “이 제품이 더 빠르다”는 문장은 모델 revision, quant, context, MTP/DSpark, cooling state를 확인하기 전에는 책의 순위표에 넣지 않는다.

## 7. GB10이 아닌 대형 unified-memory 대안

다음 장비는 “DGX Spark clone”이 아니라 다른 생태계의 대안이다.

| 플랫폼 | 공식적으로 확인한 범위 | 장점 | DGX Spark와 섞을 때의 문제 |
|---|---|---|---|
| AMD Ryzen AI Halo / Ryzen AI Max+ 395 | 최대 128GB LPDDR5x, 256GB/s, Radeon 8060S, 120W TDP, Linux·Windows 경로 | x86 호환성, 128GB급 shared memory, ROCm·llama.cpp·vLLM 실험 | GB10·CUDA·DGX OS·ConnectX-7 노드가 아님. 같은 TP/NCCL cluster로 바로 합치지 않음 |
| Apple Mac Studio M3 Ultra | 최대 512GB unified memory, Apple 발표상 800GB/s 초과 메모리 대역폭, 최대 270W 표기(2025 M3 Ultra) | 큰 모델 용량, Metal/MLX 생태계, 조용한 control host·별도 endpoint | CUDA·NCCL·CX-7 direct link가 없음. MCDMA/Thunderbolt는 별도 실험 경로이며 공식 Spark memory pooling으로 쓰지 않음 |

AMD와 Apple의 메모리 수치가 더 크거나 높아 보여도, 같은 모델의 실제 tok/s·tool call·긴 문맥 품질을 의미하지 않는다. 특히 Spark의 TP=2 recipe를 AMD나 Mac에 옮기는 것은 “노드 한 대를 교체”하는 일이 아니라 runtime과 통신 경로를 다시 설계하는 일이다.

## 8. 용도별 선택 가이드

| 목적 | 우선 검토 | 판단 근거 |
|---|---|---|
| 이 책의 NVIDIA recipe를 그대로 재현 | NVIDIA DGX Spark, ASUS GX10 | 공식 플레이북·기존 ASUS 현장 자료·커뮤니티 recipe가 많음 |
| 장시간 열 여유를 우선 | Acer를 후보로 두고 실제 샘플 검증 | 독립 열 비교는 Acer 샘플이 유리했지만 tok/s 순위는 아님 |
| 기업 지원·현장 교체·서비스 계약 | Dell Pro Max, Lenovo PGX, HP ZGX | 지원·보안·관리 기능이 제품 문서에 구체적 |
| 로컬 RAG와 모델 관리 UI | GIGABYTE AI TOP ATOM | AI TOP Utility가 차별점. CLI recipe와 별도 검증 |
| 산업·엣지·보안 appliance | MSI EdgeXpert | Docker Compose·SSL·private CA·수평 확장 방향 |
| 메모리 용량과 Metal/MLX가 우선 | Mac Studio | GB10 클러스터 대신 별도 endpoint 또는 control host로 설계 |
| CUDA 대신 x86/ROCm을 실험 | AMD Ryzen AI Halo | GB10과 별도 benchmark·runtime으로 운영 |

가격은 지역, SSD, 어댑터, 지원 기간, 재고에 따라 크게 달라진다. 같은 128GB라는 이유만으로 가장 싼 SKU를 구매하는 것보다, 실제 필요한 4TB SSD·200GbE 케이블·switch·지원 계약을 포함한 총액을 비교한다.

## 9. 벤더 비교를 직접 재현하는 최소 프로토콜

### 9.1 장비 inventory

```bash
hostnamectl
uname -a
cat /etc/os-release
nvidia-smi
free -h
df -h
ip -br addr
ibv_devices || true
ibdev2netdev || true
docker version
```

다음 값을 함께 저장한다.

```text
vendor, model, sku, serial
os, image, kernel, driver, cuda, nccl
memory_total, storage_model, storage_size
management_interface, cx7_interfaces, link_speed, mtu
power_adapter_rating, ambient_temperature
```

### 9.2 동일 LLM 조건

1. 같은 모델 repository와 revision을 사용한다.
2. 같은 quant·KV dtype·context·engine commit을 고정한다.
3. `thinking`, prompt tokens, output tokens, batch/concurrency를 고정한다.
4. prefill·decode·TTFT·end-to-end·aggregate를 분리한다.
5. `nvidia-smi` rail power와 wall power를 각각 기록한다.
6. cold start, warm start, 30분 이상 soak를 구분한다.
7. 다중 노드는 `NCCL_DEBUG=INFO`에서 `NET/IB`와 socket fallback을 구분한다.

### 9.3 결과 판정

| 상태 | 의미 |
|---|---|
| `loads` | 모델과 runtime이 메모리에 올라감 |
| `serves` | endpoint가 반복 요청에 응답함 |
| `benchmarked` | 고정 조건의 raw 결과가 있음 |
| `long-context-tested` | 지정 context에서 retrieval·품질·안정성을 확인함 |
| `tool-tested` | parser·schema·arguments·오류 복구를 확인함 |
| `agent-tested` | 여러 단계 tool loop를 실제로 통과함 |

`loads` 또는 제조사 사양만으로 “이 벤더가 더 빠르다”, “1M context가 된다”, “GPT-5.6 Sol과 비슷하다”고 쓰지 않는다.

## 10. 아직 비어 있는 검증 항목

- Acer·ASUS·Dell·GIGABYTE·HP·Lenovo·MSI·NVIDIA를 같은 `DeepSeek V4 Flash 0731` recipe로 1대씩 비교한 공개 end-to-end 표
- 7개 OEM의 동일 SSD·동일 어댑터·동일 DGX OS image 열 비교
- 2대 direct와 4대 switch에서 vendor-mixed cluster의 NCCL/RDMA 지원 조합
- 각 OEM의 firmware·USB-C PD·ConnectX-7 업데이트 정책과 rollback 절차
- AMD Ryzen AI Halo·Mac Studio·GB10의 동일 prompt/quality/tool-call 비교

따라서 이 장의 현재 결론은 성능 순위가 아니라 **플랫폼 경계와 검증 방법을 정리한 구매·실험 기준**이다. 실제 장비가 추가되면 이 문서의 표에 `locally reproduced` 행을 추가하고, 커뮤니티 수치와 합치지 않는다.

## 참고 링크

### 공식

- [NVIDIA DGX Spark 제품 사양](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA Certified Systems](https://docs.nvidia.com/certification-programs/latest/nvidia-certified-systems.html)
- [ASUS Ascent GX10](https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/)
- [ASUS GX10 FAQ](https://www.asus.com/us/support/faq/1056142/)
- [Acer Veriton GN100 발표](https://news.acer.com/acer-unveils-the-veriton-gn100-ai-mini-workstation-built-on-the-nvidia-gb10-superchip)
- [Dell Pro Max with GB10](https://www.dell.com/en-sg/shop/pcs-desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro)
- [GIGABYTE AI TOP ATOM](https://www.gigabyte.com/AI-TOP-PC/GIGABYTE-AI-TOP-ATOM)
- [HP ZGX Nano QuickSpecs](https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c09212373)
- [Lenovo ThinkStation PGX Product Guide](https://lenovopress.lenovo.com/lp2321-thinkstation-pgx)
- [MSI EdgeXpert MS-C931](https://ipc.msi.com/product_detail/EdgeXpert-MS-C931)
- [AMD Ryzen AI Halo Developer Platform](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html)
- [Apple Mac Studio M3 Ultra 발표](https://www.apple.com/uk/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)

### 독립·커뮤니티

- [StorageReview: OEM 냉각·전력 비교](https://www.storagereview.com/review/nvidia-dgx-spark-thermal-test-how-oem-cooling-designs-stack-up)
- [NVIDIA Forum: NVIDIA와 Lenovo GB10 혼합 클러스터](https://forums.developer.nvidia.com/t/mixing-different-gb10-systems-in-a-dgx-spark-cluster-nvidia-lenovo/367711)
- [이 저장소의 ASUS GX10 노드 세팅 리서치](dgx-spark-node-setup-research-2026-08.md)
- [이 저장소의 DGX Spark·Mac·RDMA·스위치 리서치](dgx-spark-mac-rdma-switch-research-2026-08.md)
