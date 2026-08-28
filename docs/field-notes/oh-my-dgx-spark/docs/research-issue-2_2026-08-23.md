# DGX Spark 리서치 기록 — Issue #2 — 2026-08-23

## 메타데이터

- 원본 Issue: [Issue #2](https://github.com/recrack/oh-my-dgx-spark/issues/2)
- 분석 기준일: `2026-08-23`
- 수집 후보 수: `40`
- 분석 실행기: `GitHub Copilot CLI`
- 요청 모델: `auto`
- 현재 상태: `분석`
- 본문 승격: `승격 대기`

## 결론

- 종합 판정: 자동 수집된 40건 가운데 근거·교차확인이 충분해 "책 반영 후보(승격 대기)"로 분류할 항목은 소수이다. 대부분은 커뮤니티 보고 또는 메타데이터가 부족해 재현·교차검증이 필요하다.
- 승격 가능한 항목: MAX_NUM_BATCHED_TOKENS 관련 메모리/프로파일링 관찰 (MiaAI-Lab Issue #4) — "교차 확인 후 책 반영(승격 대기)" 권고
- 아직 확정하지 않는 항목: Qwen3.8-27B 관련 벤치마크 수치, GB10 재현 로그, 커뮤니티 단일 보고들은 모두 `재현 대기` 또는 `교차 확인 필요`로 둔다.

## 확인된 사실

- MAX_NUM_BATCHED_TOKENS 프로파일링 결과(관찰): MiaAI-Lab의 보고에서 기본값 `8224`가 프로파일링 산출물로 인해 KV 예약을 크게 키우며, `2048`으로 낮추면 약 3.9 GiB의 KV 공간이 회복되었다고 보고한다. (출처 유형: GitHub issue; 링크: https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark/issues/4)
  - 조건: 보고서에 적힌 환경(GB10 단일 호스트, 통합 메모리 약 121 GiB 등)과 프로파일링 절차에 의존함.

- GB10 TP=2 배포에서 모델 로드 중 NVRM OOM 및 시스템 hard-reset이 관찰되었다는 보고가 있다. 동일한 증상은 MiniMax-H3 저장소와 dgx-spark-playbooks에서 각각 보고되었다. (출처 유형: GitHub issues; 링크: https://github.com/joeynyc/MiniMax-H3-2x-DGX-Spark/issues/2 ; https://github.com/NVIDIA/dgx-spark-playbooks/issues/97)
  - 조건: 보고서들은 모델 로드 시점(8–12분 내)에 NVRM 메모리 할당 실패가 발생했다고 기술함; 재현을 위해 추가 시스템 로그(journalctl, dmesg, NVRM 로그)가 필요하다.

## 커뮤니티 주장

- SGLang Qwen3.8-27B의 DGX Spark 단일-노드 검증과 관련해 "Final Verification In Progress"라는 표기가 있고, 관련 BENCHMARKS.md가 있다는 주장(작업자 제공). (출처 유형: GitHub discussion/issue/PR; 링크: https://github.com/sgl-project/sglang/pull/35825 , https://github.com/hasso5703/dgx-spark-qwen38/blob/main/BENCHMARKS.md)
  - 상태: PR 병합·원문 데이터 접근 여부 확인 필요 → `재현 대기`

- 다수의 커뮤니티 보고(포럼·RSS·레포지토리 이슈)가 DGX Spark/GB10 환경에서의 문제·최적화·호환성(ARM64, NVFP4, DSpark 등)을 주장하지만, 많은 항목은 측정 조건(모델 버전·양자화·런타임·컨텍스트·동시성)이 누락되어 있어 `재현 대기`로 분류한다.

## 충돌·미확인 내용

- 충돌 사례 없음이 명시적으로 보고되지는 않았으나, 성능 수치(토크 수, 지연, 메모리 사용량)는 원문에서 모델 버전·양자화·런타임·노드 수·동시성·측정 방법을 일관되게 제공하지 않은 경우가 많아 직접 비교 불가.
- 일부 커뮤니티 리포트는 단일 호스트 환경에서의 측정을 기반으로 전사(일반화)하려는 경향이 있으므로 `교차 확인 필요`.

## 책 반영 제안

- 후보 장: "11-2: OOM/NCCL 및 메모리 운영"(권장 반영 항목)
  - 요지(문장 수준): "MiaAI-Lab 보고에 따르면 vLLM/DeepSeek 구성에서 MAX_NUM_BATCHED_TOKENS 프로파일링 산출물이 KV 예약을 크게 키울 수 있으며, 상황에 따라 값(예: 8224→2048)을 조정해 약 3.9 GiB의 KV 회복을 관찰했다(출처: MiaAI-Lab Issue #4)." (승격 조건: 원문 데이터와 재현 결과 확보)
  - 반영 상태: `승격 대기` (재현 또는 추가 교차확인 필요)

- 후보 장: "Qwen3.8-27B cookbook 노트"
  - 요지: "SGLang PR #35825는 Qwen3.8-27B의 DGX Spark 단일-노드 검증을 위해 커밋 단위로 재측정을 진행 중이며, 최종 벤치마크는 PR 병합 후에 책에 반영한다." (승격 조건: PR 병합 및 BENCHMARKS.md 접근성 확인)
  - 반영 상태: `승격 대기`

- 주의: `즉시 반영` 항목으로 기록하지 않는다. 모든 숫자와 권고는 모델·양자화·런타임·노드 수·동시성·측정 방법을 명시한 원문 근거가 확보된 경우에만 책 본문 문장으로 옮긴다.

## 출처 목록

- MiaAI-Lab, "MAX_NUM_BATCHED_TOKENS..." (GitHub issue): https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark/issues/4 (공식/레시피형 증거)
- MiniMax-H3, "GB10 NVRM OOM during model loading" (GitHub issue): https://github.com/joeynyc/MiniMax-H3-2x-DGX-Spark/issues/2 (커뮤니티 재현 보고)
- NVIDIA dgx-spark-playbooks, "head hard-resets during 128K-token vLLM prefill" (GitHub issue): https://github.com/NVIDIA/dgx-spark-playbooks/issues/97 (교차 보고)
- SGLang PR (Qwen3.8-27B remeasure): https://github.com/sgl-project/sglang/pull/35825 (커뮤니티/공식 PR)
- 원본 자동 수집 Issue: https://github.com/recrack/oh-my-dgx-spark/issues/2

(위 링크들은 Issue 본문과 자동 수집 후보 목록에서 제공된 URL만 사용해 인용함.)

## 보류 사유 및 다음 작업

- 보류 사유 요약: 대부분 항목이 "커뮤니티 보고" 또는 "부분적 수치만 제공" 상태여서, 모델 버전·양자화·런타임·노드 수·동시성·측정 방법이 명시될 때까지 `재현 대기`로 둔다.

- 우선 검증 작업 (권고 순서):
  1. SGLang PR #35825 병합 상태 확인 및 BENCHMARKS.md 접근 (우선순위: 높)
  2. MAX_NUM_BATCHED_TOKENS=2048로 조정해 동일 환경(단일 GB10)에서 KV 회복 재현 (우선순위: 높)
  3. GB10 NVRM OOM / hard-reset 관련 추가 시스템 로그(journalctl, dmesg, NVRM 로그) 수집 및 교차검증 (우선순위: 중)
  4. 기타 커뮤니티 보고(포럼·RSS) 중 재현 가능한 항목을 선별해 재현표준 템플릿 작성 (우선순위: 중)

- 보류된 주요 주장 예시(일부): Qwen3.8-27B 단일-노드 속도/토큰 수치, Ollama/llama.cpp/DSpark 대규모 버전 비교, ARM64 패키지 지원 여부 등 — 모두 원문에 명시적 측정 조건이 필요함.

---

작성자 주: 이 문서는 Issue #2의 본문과 자동 수집 후보 목록(제공된 URL) 및 Issue의 코멘트(자동화 코멘트 포함)만을 근거로 작성되었다. 책 반영은 본문 승격 규칙에 따라 별도 검토 PR에서 진행한다.
