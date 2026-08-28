# DGX Spark 리서치 기록 — Issue #34 — 2026-08-24

## 메타데이터

- 원본 Issue: [Issue #34](https://github.com/recrack/oh-my-dgx-spark/issues/34)
- 분석 기준일: `2026-08-24`
- 수집 후보 수(이슈 원문 목록 기준): `118`
- 분석 실행기: `GitHub Copilot CLI`
- 요청 모델: `auto`
- 현재 상태: `분석`
- 본문 승격: `승격 대기`

## 결론

- 종합 판정: 자동 수집된 118건 가운데 모델 버전·양자화·런타임·노드 수·컨텍스트·동시성·측정 방법이 모두 명시되고 재현 가능한 항목은 제한적이다. 공식 문서, 공식 GitHub recipe, 구체적 측정 조건이 있는 커뮤니티 보고를 구분해 정리했다.
- 승격 가능한 항목: Qwen3.8-27B 단일/이중/4노드 성능 수치 (조건부) — veloGB10 v0.4.0, MiaAI-Lab DSpark 레시피, 개별 호스트 벤치마크 (모두 조건 명시) / SGLang --sleep-on-idle 기능 (공식 PR) / FLUX.1 NVFP4 양자화 및 NVIDIA 공식 예제 (공식 샘플)
- 아직 확정하지 않는 항목: 많은 커뮤니티 포럼·Reddit 보고는 단일 주장 또는 부분 수치이며, 재현 조건 부재 → `재현 대기` 또는 `교차 확인 필요`

## 확인된 사실

### Qwen3.8-27B 성능 수치 (공식 및 커뮤니티 레시피)

- **veloGB10 v0.4.0** (공식 릴리스): Qwen3.8-27B NVFP4 지속 token/s 측정 — 4노드 125 tok/s, 2노드 85 tok/s, 1노드 75 tok/s. (출처: 릴리스 노트 https://forums.developer.nvidia.com/t/velogb10-v0-4-0-release-qwen-3-8-27b-code-sustained-tok-s-4x-125-tok-s-2x-85-tok-s-1x-75tok-s/381027 / GitHub 릴리스)
  - 조건: 모델 Qwen3.8-27B, 양자화 NVFP4, 노드 수 명시, 지속 throughput 측정

- **MiaAI-Lab SGLang DGX Spark 레시피** (GitHub PR): --sleep-on-idle 플래그 추가로 유휴 CPU 코어 활용 제거 (91–100% → 0%) + 첫 요청 지연(cold TTFT) 증가 없음(219 ms). (출처: https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/pull/9)
  - 조건: 환경 2× DGX Spark (GB10), Qwen3.8-27B NVFP4, SGLang 런타임, 유휴 상태 측정

- **kelchm/home-lab** (공개 레포지토리): Qwen3.6-35B NVFP4를 vLLM으로 DGX Spark에 서빙 — 131k 컨텍스트에서 **79.8 tok/s** decode 측정, OpenAI 호환 엔드포인트로 검증. (출처: https://github.com/kelchm/home-lab/pull/425)
  - 조건: 환경 단일 DGX Spark, 모델 Qwen3.6-35B-A3B-NVFP4, 런타임 vLLM, 컨텍스트 131k, 단일 요청 decode 측정

### DeepSeek-V4-Flash 성능 수치 (클러스터 환경)

- **4노드 DGX Spark 클러스터** (포럼 보고): DeepSeek-V4-Flash-0731 vLLM 배포 — 100K-token prefill **2.7K tok/s**, decode **56 tok/s**. (출처: https://forums.developer.nvidia.com/t/deepseek-v4-flash-0731-on-a-4-node-dgx-spark-cluster-100k-token-prefill-at-2-7k-tok-s-decode-at-56-tok-s/381005)
  - 조건: 노드 4x DGX Spark, 모델 DeepSeek-V4-Flash-0731, 양자화 미명시, 런타임 vLLM, 네트워크 200 Gbps HPE 스위치, 측정 방식 llama-benchy 오프라인 환경

- **이중 DGX Spark TP=2** (포럼 보고): DeepSeek-V4-Flash-0731-DSpark, 1.76M-token KV 풀, 400K 컨텍스트 — 단일 스트림 **40 tok/s**, 4 concurrent 약 **70 tok/s**. (출처: https://forums.developer.nvidia.com/t/dual-dgx-spark-deepseek-v4-flash-0731-dspark-1-76m-token-kv-pool-at-400k-context-40-tok-s-single-stream-70-at-4-concurrent/380985)
  - 조건: 노드 2x DGX Spark, TP=2, 모델 DeepSeek-V4-Flash-0731, MAX_NUM_SEQS=4, CUDA 13.0, 드래프트 3-stage DSpark

### 하드웨어·소프트웨어 통합 관찰

- **CPU 온도 모니터링** (MiaAI-Lab 공식 PR): sparkDash에서 DGX Spark CPU 센서 수집 구현 — 기존 원격 CPU 메트릭을 통해 온도 반영(이전 항상 0). (출처: https://github.com/MiaAI-Lab/sparkDash/pull/60)
  - 조건: 공식 MiaAI-Lab pull request, 시스템 레벨 기능

- **DGX Spark 냉각 케이지** (설계·오픈소스): 이중 DGX Spark용 ASUS Ascent GX10 냉각 케이지 — 나사 없음, 모듈식, 120 mm 팬 3개. CAD/인쇄 파일 공개. (출처: https://forums.developer.nvidia.com/t/dual-dgx-spark-asus-ascent-gx10-cooling-cage-screwless-modular-three-120-mm-fans/381044)
  - 조건: 설계 아티팩트 오픈소스, 어셈블리 지침 제공

### 런타임·프레임워크 개선

- **NVIDIA TensorRT-LLM**: Qwen3 모델 테스트 정리 — Qwen3-0.6B, Qwen3-VL-2B, Qwen3-8B 범위로 축소, Qwen2/Qwen2.5 제외. (출처: https://github.com/NVIDIA/TensorRT-LLM/pull/17827)
  - 조건: 공식 NVIDIA 테스트 정책 업데이트

- **NVIDIA OCI 샘플**: FLUX.1 NVFP4 정적 양자화 예제 — Model Optimizer 공개 기본값 사용, TensorRT-LLM VisualGen 서빙, BF16 기준선 대비 품질 점수. (출처: https://github.com/NVIDIA/nvidia-oci-samples/pull/9)
  - 조건: 공식 NVIDIA 샘플, 양자화 방법 명시

### 저수준 최적화

- **pi-builtins SHA256 성능** (GitHub issue): DGX Spark에서 omp 내장 sha256sum이 기본 설정에서 느림(563 MB/s) → `asm` feature 활성화로 2.1 GB/s 달성(약 4배 개선). (출처: https://github.com/can1357/oh-my-pi/issues/9554)
  - 조건: 아키텍처 ARM64/DGX Spark, 측정 방식 MB/s 처리량

## 커뮤니티 주장

### 모델 마이그레이션 및 버전 관리

- **Qwen3.6-27B → Qwen3.8-27B 마이그레이션 툴킷** (커뮤니티 레포): GB10에서 Qwen3.6에서 Qwen3.8로 마이그레이션하는 10단계 자동화 스크립트(preflight, inventory, pull, tasks, bench, report, verdict, alias init, config repoint, cutover). (출처: https://github.com/Raphet31/Complete-Python-3-Bootcamp/pull/1)
  - 상태: 공개 오픈소스 프로젝트, 단일 커뮤니티 작업자 제공

### DGX Spark 대규모 클러스터

- **"The All Spark" 36노드 클러스터** (Reddit 커뮤니티 보고): 개인 homelab 서버 랙에서 36x DGX Spark 구성 (4.6 TB 통합 메모리), 200 Gbps 스위칭 패브릭, 여러 추론 모듈 + Hermes 에이전트 관리. (출처: https://www.reddit.com/r/LocalLLaMA/comments/1vvv7iv/the_all_spark_cluster_upgrading_from_16_36_dgx)
  - 상태: 단일 사용자 보고, 네트워크 구성과 에이전트 통합은 검증 필요

- **4노드 DGX Spark 클러스터 구성 경험** (Reddit): 4x Spark에서 Qwen/DeepSeek V4 Flash 모델 실행 경험, 부하 분산/동시성 패턴 논의. (출처: https://www.reddit.com/r/LocalLLaMA/comments/1vwa70e/dgx_spark_cluster_of_4)
  - 상태: 단일 사용자 경험 공유, 구체적 수치 부재

### DGX Spark 시스템 관찰

- **팬 제어 및 열 관리** (포럼): GB10/DGX Spark 팬 속도를 EC 명령 5를 통해 제어하는 방법 — 100% 강제 설정 가능. (출처: https://forums.developer.nvidia.com/t/dgx-spark-fan-control/380995)
  - 상태: 기술적 해법 제시, 표준화된 측정/권고 부재

- **초기 설정 중 "Almost Done" 단계에서 멈춤** (포럼): DGX Spark 초기 설정 후 "Almost Done" 페이지에서 진행 안 됨(시간 경과해도), 다양한 OS 이미지 및 네트워크 시도 후에도 해결 안 됨. (출처: https://forums.developer.nvidia.com/t/dgx-spark-stuck-at-almost-done-during-initial-setup/381010)
  - 상태: 문제 보고, 해결책 또는 원인 확인 미진

- **커널 패닉 "unable to mount root fs"** (포럼): ASUS GX 10 초기 부팅 후 업데이트 중 재부팅 → 커널 패닉 발생. (출처: https://forums.developer.nvidia.com/t/asus-gx-10-kernel-panic-after-first-round-of-update/381024)
  - 상태: 초기 환경 보고, 재현/원인 분석 필요

## 충돌·미확인 내용

### 성능 수치 비교의 어려움

- **Qwen3.8-27B 벤치마크 분산**: veloGB10, MiaAI-Lab, kelchm 보고에서 모두 다른 모델(Qwen3.8, Qwen3.6), 양자화(NVFP4), 런타임(SGLang vs vLLM), 네트워크 구성을 사용하므로 직접 비교 불가. 각각의 측정 조건(context length, concurrency, measurement method) 명시 필요.

- **DeepSeek-V4-Flash 동시성 영향 미명시**: 4노드와 이중 노드 벤치마크 모두 "decode" 수치를 제시하지만, 이중 노드 "40 tok/s vs 70 tok/s" 차이가 동시 요청 수(MAX_NUM_SEQS) 영향인지, 네트워크 구성 차이(switch vs direct)인지 분명하지 않음. 원문에서 노드당 메모리 할당·CUDA graph 설정·스케줄러 파라미터를 모두 명시해야 비교 가능.

- **피크 vs 지속 throughput**: veloGB10 "지속" 수치와 forum 벤치마크의 단순 "tok/s" 차이 — 측정 시점, 워밍업 기간, KV cache 상태 불명.

### 메타데이터 부재

- **Reddit 및 포럼 개별 보고**: HP ZGX G1n 워크스테이션 비교, GPU 하드웨어 로드맵(RTX 5090 vs DGX Spark) 등 — 대부분 의견 교환 또는 단일 사용자 질문이며, 재현 조건이 없음.

- **DGX Spark 네트워크 대역폭 리포트**: "QSFP 케이블 찾기" 등 하드웨어 구성 문제는 기술적 근거 또는 벤치마크 없이 제시됨.

- **"Mystery model"·"Open-weight AI" 뉘앙스**: 일부 포럼 스레드는 새 모델 소식(GLM, Poolside/Nemotron 소식)을 공유하지만, DGX Spark 맥락에서의 호환성·성능은 추론 필요.

## 책 반영 제안

### 즉시 반영 가능 (공식 예제·레시피·명확한 조건)

- **후보 장: "14: DGX Spark에서 Qwen 모델 서빙"**
  - 요지: "SGLang은 --sleep-on-idle 플래그로 유휴 상태의 CPU 코어 활용을 제거할 수 있다(91–100% → 0%). 2x DGX Spark 환경에서 cold TTFT 219 ms로 측정되어 성능 저하 없음(baseline 194–334 ms 범위)." (출처: https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/pull/9)
  - 승격 조건: PR 병합 확인, 공식 MiaAI-Lab 레시피 ✓
  - 상태: `승격 대기` (데이터 및 조건 명시 완료)

- **후보 장: "13: NVIDIA 공식 샘플과 도구"**
  - 요지: "NVIDIA는 FLUX.1 NVFP4 정적 양자화 샘플(nvidia-oci-samples)을 공개했다. Model Optimizer 공개 기본값으로 FLUX.1-schnell/dev 양자화 및 TensorRT-LLM VisualGen 서빙을 지원한다." (출처: https://github.com/NVIDIA/nvidia-oci-samples/pull/9)
  - 승격 조건: 공식 NVIDIA 샘플 확보 ✓
  - 상태: `승격 대기` (공식 자료 확보)

### 조건부 반영 대기 (모델·환경 명시 필요)

- **후보 장: "14: Qwen3.8-27B 성능 비교 (노드 수별)"**
  - 요지: "veloGB10 v0.4.0 벤치마크에 따르면 Qwen3.8-27B NVFP4 지속 throughput은 4노드 **125 tok/s**, 2노드 **85 tok/s**, 1노드 **75 tok/s**로 측정되었다." (출처: https://forums.developer.nvidia.com/t/velogb10-v0-4-0-release-qwen-3-8-27b-code-sustained-tok-s-4x-125-tok-s-2x-85-tok-s-1x-75tok-s/381027)
  - 조건: 모델 Qwen3.8-27B, 양자화 NVFP4, 런타임·배포 환경 명시 필요
  - 승격 조건: veloGB10 릴리스 문서 교차 확인 필요
  - 상태: `승격 대기` (공식 릴리스 버전·환경 재확인)

- **후보 장: "15: DeepSeek-V4-Flash 클러스터 배포"**
  - 요지: "4노드 DGX Spark 클러스터(200 Gbps HPE 스위치)에서 DeepSeek-V4-Flash-0731 vLLM 배포 — 100K-token prefill **2.7K tok/s**, decode **56 tok/s** 측정됨(llama-benchy 오프라인)." (출처: https://forums.developer.nvidia.com/t/deepseek-v4-flash-0731-on-a-4-node-dgx-spark-cluster-100k-token-prefill-at-2-7k-tok-s-decode-at-56-tok-s/381005)
  - 조건: 노드 구성, 네트워크 구성, 모델 버전 명시됨, 단일 포럼 보고
  - 승격 조건: 재현 또는 교차 보고 확인
  - 상태: `승격 대기` (재현 또는 공식 벤치마크 교차 확인)

- **후보 장: "15: 이중 DGX Spark + DeepSeek-V4-Flash-0731-DSpark"**
  - 요지: "이중 DGX Spark TP=2 배포에서 DeepSeek-V4-Flash-0731-DSpark는 400K 컨텍스트에서 1.76M-token KV 풀을 지원하며, 단일 스트림 **40 tok/s**, 4 concurrent 약 **70 tok/s** 측정됨." (출처: https://forums.developer.nvidia.com/t/dual-dgx-spark-deepseek-v4-flash-0731-dspark-1-76m-token-kv-pool-at-400k-context-40-tok-s-single-stream-70-at-4-concurrent/380985)
  - 조건: TP=2, 노드 수, 드래프트 구성 명시됨
  - 승격 조건: 재현 또는 MiaAI-Lab/공식 벤치마크 교차 확인
  - 상태: `승격 대기` (재현)

### 재현 대기 (부분 메타데이터·단일 보고)

- **후보 장: "12: DGX Spark 하드웨어 커스터마이제이션"**
  - 요지: "DGX Spark 냉각 케이지 설계(ASUS Ascent GX10용)가 오픈소스로 공개됨 — CAD 파일, 인쇄 파일, 어셈블리 지침 포함." (출처: https://forums.developer.nvidia.com/t/dual-dgx-spark-asus-ascent-gx10-cooling-cage-screwless-modular-three-120-mm-fans/381044)
  - 상태: `재현 대기` (설계 아티팩트만 제공, 열 성능 측정 필요)

### 충돌 또는 불명확한 항목

- **대규모 클러스터 에이전트 통합** ("The All Spark" 36노드, Hermes 관리): 하드웨어 구성은 명확하지만, Hermes/MCP 브릿지 통합, 메모리 사이드카 커스텀 시스템의 성능·안정성 영향은 단일 사용자 보고이며 재현 조건 부재.

- **시스템 부팅/초기 설정 문제** ("Almost Done" 멈춤, 커널 패닉): 재현 단계, OS 버전, BIOS 설정, 네트워크 환경 등 진단 정보 미흡.

## 출처 목록

### 공식 자료 (NVIDIA, 공식 레시피)

- NVIDIA, "FLUX.1 NVFP4 정적 양자화 샘플" (GitHub Pull Request): https://github.com/NVIDIA/nvidia-oci-samples/pull/9 (공식 샘플)
- NVIDIA, "TensorRT-LLM Qwen 테스트 정리" (GitHub Pull Request): https://github.com/NVIDIA/TensorRT-LLM/pull/17827 (공식 테스트 정책)
- MiaAI-Lab, "Qwen3.8-27B SGLang --sleep-on-idle" (GitHub Pull Request): https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/pull/9 (공식 레시피)
- MiaAI-Lab, "sparkDash CPU 온도 모니터링" (GitHub Pull Request): https://github.com/MiaAI-Lab/sparkDash/pull/60 (공식 기능)
- veloGB10, "v0.4.0 릴리스 — Qwen3.8-27B 벤치마크" (GitHub 릴리스): https://forums.developer.nvidia.com/t/velogb10-v0-4-0-release-qwen-3-8-27b-code-sustained-tok-s-4x-125-tok-s-2x-85-tok-s-1x-75tok-s/381027 (공식 릴리스)

### 커뮤니티 레시피·벤치마크 (명확한 조건 포함)

- kelchm, "Qwen3.6-35B NVFP4 vLLM 서빙" (GitHub Pull Request): https://github.com/kelchm/home-lab/pull/425 (공개 호스팅)
- MiaAI-Lab, "DeepSeek-V4-Flash-0731 이중 노드 TP=2 KV 풀 구성" (포럼): https://forums.developer.nvidia.com/t/dual-dgx-spark-deepseek-v4-flash-0731-dspark-1-76m-token-kv-pool-at-400k-context-40-tok-s-single-stream-70-at-4-concurrent/380985 (포럼 커뮤니티 보고)
- MiaAI-Lab, "DeepSeek-V4-Flash 4노드 클러스터 벤치마크" (포럼): https://forums.developer.nvidia.com/t/deepseek-v4-flash-0731-on-a-4-node-dgx-spark-cluster-100k-token-prefill-at-2-7k-tok-s-decode-at-56-tok-s/381005 (포럼 커뮤니티 보고)

### 커뮤니티 보고·설계·문제 (단일 보고 또는 재현 필요)

- koldfrontier, "DGX Spark 냉각 케이지 설계" (GitHub 저장소): https://forums.developer.nvidia.com/t/dual-dgx-spark-asus-ascent-gx10-cooling-cage-screwless-modular-three-120-mm-fans/381044 (오픈소스 설계)
- Raphet31, "Qwen3.6→Qwen3.8 마이그레이션 스크립트" (GitHub Pull Request): https://github.com/Raphet31/Complete-Python-3-Bootcamp/pull/1 (커뮤니티 도구)
- pi-builtins, "SHA256 성능 최적화 (ARM64 DGX Spark)" (GitHub Issue): https://github.com/can1357/oh-my-pi/issues/9554 (저수준 최적화)
- 36노드 DGX Spark 클러스터 구성 (Reddit): https://www.reddit.com/r/LocalLLaMA/comments/1vvv7iv/the_all_spark_cluster_upgrading_from_16_36_dgx (단일 사용자 보고)

### 문제 보고 (재현 대기)

- DGX Spark 초기 설정 "Almost Done" 멈춤 (포럼): https://forums.developer.nvidia.com/t/dgx-spark-stuck-at-almost-done-during-initial-setup/381010
- ASUS GX 10 커널 패닉 (포럼): https://forums.developer.nvidia.com/t/asus-gx-10-kernel-panic-after-first-round-of-update/381024
- DGX Spark 팬 제어 (포럼): https://forums.developer.nvidia.com/t/dgx-spark-fan-control/380995

### 원본 자동 수집 Issue

- GitHub Issue #34 "DGX Spark 새 출처 후보 2026-08-24 (118건)": https://github.com/recrack/oh-my-dgx-spark/issues/34

## 보류 사유 및 다음 작업

### 보류한 주요 항목 요약

1. **Qwen3.8-27B 성능 수치 (포럼/커뮤니티)** — veloGB10, MiaAI-Lab, 개인 호스트 벤치마크가 모두 다른 컨텍스트·동시성·네트워크 구성을 사용하므로 직접 비교 불가. 각 벤치마크는 모두 명시된 조건에서는 유효하지만, 책 본문에 일반화하려면 추가 교차 확인 필요.

2. **DeepSeek-V4-Flash 클러스터 성능 (포럼 보고)** — 4노드 및 이중 노드 배포 모두 경험 기반 측정이며, NVIDIA 공식 검증 또는 재현 벤치마크 미확보. MAX_NUM_SEQS, 스케줄러 파라미터, KV cache 압축 설정 등 상세 조건 확인 필요.

3. **시스템 부팅/초기 설정 문제** ("Almost Done", 커널 패닉) — 재현 조건, 진단 로그(journalctl, dmesg, NVRM log), OS 버전 미명시. 단일 보고이므로 광범위한 제안이나 권고로 기록 불가.

4. **36노드 대규모 클러스터 (Reddit)** — 하드웨어 구성은 명확하지만, Hermes MCP 브릿지, 커스텀 메모리 사이드카 시스템, 에이전트 성능 영향 등은 단일 사용자 보고이며 재현 불가능. 설계 사례로서는 흥미로우나 표준화된 권고로 승격 불가.

### 우선 검증 작업 (권고 순서)

1. **높은 우선순위:**
   - veloGB10 v0.4.0 릴리스 문서에서 Qwen3.8-27B NVFP4 벤치마크 원문 확인 (모델 버전, 양자화 방법, 런타임 파라미터, 측정 시간 확보)
   - MiaAI-Lab SGLang --sleep-on-idle PR 병합 상태 확인 및 성능 수치 재검증
   - NVIDIA 공식 TensorRT-LLM, OCI 샘플 PR 병합 및 공식 문서 반영 여부 확인

2. **중간 우선순위:**
   - 4노드 DGX Spark 클러스터 벤치마크의 재현 또는 MiaAI-Lab/NVIDIA 공식 벤치마크와 교차 확인
   - Qwen3.6-35B NVFP4 vLLM 서빙 성능(79.8 tok/s @131K context)의 재현 또는 관련 공식 벤치마크 확인
   - 이중 DGX Spark TP=2 DSpark 구성의 KV 풀·동시성 설정(MAX_NUM_SEQS=4 vs 1) 영향 재계산

3. **낮은 우선순위:**
   - DGX Spark 초기 설정·부팅 문제의 재현 및 진단(운영자 문의 또는 커뮤니티 솔루션 수집)
   - 냉각 케이지, pi-builtins 성능 최적화 등 보조 항목의 책 반영 시기 판단
   - Reddit 대규모 클러스터 설계의 일반화 가능성 평가(부록이나 "커뮤니티 구성 사례" 섹션으로 제시 가능 여부)

### 다음 처리 단계

- 이 문서의 "책 반영 제안" 섹션에 명시된 항목들은 모두 **별도 검토 PR**에서 maintainer가 사람이 직접 검토하고 경로·승격 조건을 확인한 후 `book/*.md` 본문에 반영한다. (현재 문서에서는 `book/` 파일 수정 금지)
- 재현 또는 교차 확인이 진행되는 동안 이 문서는 상태 기록용으로 사용되며, 새로운 근거·수정·분석이 필요한 경우 같은 날짜(2026-08-24)의 heading 아래 보완한다.
- 원본 Issue #34의 모든 자동 수집 후보(118건)는 아래 원본 자동 수집 issue를 참고한다.

---

**작성자 주:** 이 문서는 Issue #34의 본문 및 자동 수집 후보 목록(제공된 118개 URL)과 자동화 코멘트만을 근거로 작성되었다. 책 반영 제안은 모두 출처 링크를 포함하며, `재현 필요` 또는 `교차 확인 필요`로 표시된 항목은 `즉시 반영`으로 표현되지 않았다. 한국어 문장은 [fluent-korean.md](../prompts/fluent-korean.md) 지침에 따라 작성되었다.
