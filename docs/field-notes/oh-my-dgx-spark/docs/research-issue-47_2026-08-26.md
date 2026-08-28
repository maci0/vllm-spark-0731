# DGX Spark 리서치 기록 — Issue #47 — 2026-08-26

## 메타데이터

- 원본 Issue: [Issue #47](https://github.com/recrack/oh-my-dgx-spark/issues/47)
- 분석 기준일: `2026-08-26`
- 수집 후보 수(이슈 원문 목록 기준): `44`
- 분석 실행기: `GitHub Copilot CLI`
- 요청 모델: `gpt-5.6-terra`
- 현재 상태: `분석`
- 본문 승격: `승격 대기`

## 결론

- 종합 판정: `분석`. 이번 후보 중에는 GB10/SM121에서 DeepSeek V4 계열을 vLLM으로 실행할 때의 FP8·FlashInfer 경로를 고치는 원저장소 vLLM PR이 있다. 그러나 모두 PR 또는 이슈 단계이며, 병합·릴리스·DGX Spark 재현 여부를 원문에서 확인하지 못했으므로 책 본문으로 승격하지 않는다.
- 승격 가능한 항목: 현재는 없다. vLLM의 해당 수정이 병합된 release 또는 고정 commit에서 단일·2대 GB10 기능 기준선을 통과한 뒤에만 운영 주의 사항으로 승격할 수 있다.
- 아직 확정하지 않는 항목: DeepGEMM `8b1392b`의 SM12x pure-FP8 경로가 실제 배포물에서 수치 오류를 일으키는지, 각 PR의 병합 상태와 포함 릴리스, 그리고 후보에 제시된 성능·cosine 수치다.

## 확인된 사실

- **원저장소 vLLM PR, 구현 제안 단계:** [PR #53521](https://github.com/vllm-project/vllm/pull/53521)은 GB10이 compute capability major 12를 보고하는데 기존 분기가 이를 SM100으로 취급한다고 설명한다. PR은 SM12x에서 Hopper용 FP8 einsum recipe를 사용하도록 제안하며, FP8 scale layout 불일치가 `o_proj` 출력 이상으로 이어질 수 있다고 기술한다. 모델 revision, 양자화 checkpoint, vLLM commit, 컨테이너, context, 동시성 및 실제 장비 재현 결과는 이번 접근 가능한 원문에서 확정하지 못했다.
- **원저장소 vLLM PR, 구현 제안 단계:** [PR #53425](https://github.com/vllm-project/vllm/pull/53425)은 SM120 계열 FlashInfer DeepSeek V4 sparse MLA decode kernel이 64-token page를 사용한다고 설명하고, SM12x에서 지원 block size로 64를 반환하도록 제안한다. PR 본문에는 GB10에서 page-64 결과와 PyTorch의 cosine이 `0.99966`이었다는 주장이 있으나, 이는 모델·quant·runtime·context·동시성·측정 방법이 완비된 성능 결과가 아니므로 일반화하지 않는다.
- **원저장소 vLLM PR, 구현 제안 단계:** [PR #53522](https://github.com/vllm-project/vllm/pull/53522)은 DeepGEMM을 import할 수 있다는 사실만으로 paged MQA metadata 경로를 선택하면, 2-state인 DSV4 compress-128 page가 32 또는 64 states를 요구하는 host assertion에 걸릴 수 있다고 설명한다. 제안된 조건은 CUDA, DeepGEMM 지원, 32 또는 64 states다. 이 경로가 현재 release의 GB10에 적용되는지는 확인하지 못했다.
- **원저장소 vLLM PR, 구현 제안 단계:** [PR #34822](https://github.com/vllm-project/vllm/pull/34822)은 GB10을 SM121로 설명하며, 기존 `major == 10` 또는 family 100 판정이 SM12x를 놓친다고 기술한다. 이는 GB10 전용 kernel·feature 분기가 compute capability 판정에 의존한다는 점을 보여 주지만, PR 병합 전에는 지원 보장이 아니다.
- **원저장소 DeepGEMM 이슈 및 vLLM PR, 교차 확인 대기:** [DeepGEMM #417](https://github.com/deepseek-ai/DeepGEMM/issues/417)과 [vLLM #53680](https://github.com/vllm-project/vllm/pull/53680)은 `a6b593d`에서 `8b1392b` 사이에 SM12x pure-FP8 1d1d kernel 파일이 제거되었다고 지목한다. vLLM PR은 GB10에서 FP8 weight를 FP4로 잘못 읽을 수 있는 silent numerical corruption 위험 때문에 DeepGEMM pin을 되돌리는 변경을 제안한다. 두 자료 모두 같은 회귀 주장을 다루지만, 이슈와 이를 소비하는 PR의 관계이므로 독립 재현 근거 두 건으로 세지 않는다.

## 커뮤니티 주장

- 후보의 MiaAI-Lab DeepSeek V4 Flash 단일·2대 Spark 이슈와 NVIDIA Developer Forum·Reddit 글은 커뮤니티 보고다. 이번 분석에서는 원문 조건을 확인하지 못했으므로 모델 revision, quant, runtime, 노드 수, context, 동시성, 측정 방법이 빠진 모든 성능·context·발열 주장을 `재현 대기`로 둔다.
- 특히 PR #53680과 DeepGEMM #417에 언급된 2대 DGX Spark 수치도 작성자가 제시한 측정일 뿐이다. 성능 수치로 책에 옮기지 않는다.

## 충돌·미확인 내용

- 모든 핵심 vLLM 자료는 PR 또는 이슈로 수집됐다. 병합 여부, target branch, 포함된 package/container release와 rollback 가능 여부가 미확인이다.
- PR #53425의 대상은 설명상 SM120 FlashInfer 경로지만 후보와 다른 PR은 GB10 SM121을 다룬다. SM120과 SM121을 같은 지원 상태로 취급하지 않는다.
- PR #53521, #53522, #53425, #53680의 문제 범위는 각각 FP8 einsum, MQA metadata, sparse MLA page block, pure-FP8 GEMM이다. 하나의 수정으로 DeepSeek V4의 모든 SM12x 문제가 해결된다고 결론 내릴 수 없다.
- 이번 수집에는 성능 수치에 필요한 모델 버전·양자화·런타임·노드 수·context·동시성·측정 방법이 완비된, 독립적으로 확인 가능한 결과가 없다.

## 책 반영 제안

- **후보 장:** `book/04-1-engine-selection.md`의 GB10/SM121 vLLM 위험 설명. **제안 요지:** “GB10에서 vLLM과 DeepSeek V4 계열을 사용할 때는 GPU capability 분기와 DeepGEMM·FlashInfer backend의 고정 commit을 기록하고, 업그레이드 뒤 FP8 출력 기능 기준선을 다시 확인한다.” **승격 조건:** 관련 PR의 병합·release 포함을 확인하고, 고정 model revision·quant·runtime·단일/2대 노드·context·동시성에서 정상 출력과 tool/parser smoke test를 재현한다.
- **후보 장:** `book/06-5-deepseek-v4-flash.md`의 2대 FP8 경로. **제안 요지:** “DeepSeek V4 Flash의 FP8 결과는 DeepGEMM과 FlashInfer 버전 의존성을 함께 기록한다.” **승격 조건:** DeepGEMM #417/#53680의 원인과 해결 commit을 독립적으로 교차 확인하고, 원본 FP8 checkpoint에서 기준 출력 비교를 통과한다.
- 성능 수치나 cosine 수치는 `재현 필요`이므로 즉시 반영하지 않는다.

## 출처 목록

- [vLLM PR #53680 — DeepGEMM nv_dev pin rollback](https://github.com/vllm-project/vllm/pull/53680) — 원저장소 PR, GB10/SM12x pure-FP8 회귀 주장.
- [DeepGEMM issue #417 — SM12x pure-FP8 1d1d regression](https://github.com/deepseek-ai/DeepGEMM/issues/417) — 원저장소 이슈, PR #53680의 원인 주장.
- [vLLM PR #53521 — Hopper FP8 einsum recipe on SM12x](https://github.com/vllm-project/vllm/pull/53521) — 원저장소 PR.
- [vLLM PR #53522 — paged MQA metadata guard](https://github.com/vllm-project/vllm/pull/53522) — 원저장소 PR.
- [vLLM PR #53425 — SM12x FlashInfer sparse MLA block size](https://github.com/vllm-project/vllm/pull/53425) — 원저장소 PR.
- [vLLM PR #34822 — Blackwell-class detection for SM121/GB10](https://github.com/vllm-project/vllm/pull/34822) — 원저장소 PR.

## 보류 사유 및 다음 작업

- GitHub 원문 페이지 접근이 제한되어 PR의 병합·리뷰·CI·release 상태를 확인하지 못했다. 수집 시점의 PR 설명만으로는 안정 지원을 판정할 수 없다.
- 다음 작업은 PR별 merge commit과 포함된 vLLM/DeepGEMM release를 확인하는 일이다. 이어서 고정된 FP8 model checkpoint와 tokenizer, image digest, driver/CUDA, Spark 수, context, `max_num_seqs`를 기록한 기능·수치 비교를 수행한다.
- 재현에서는 FP8 선형층의 기준 출력 비교, DeepSeek V4 sparse MLA decode, 2-state paged MQA 경로, c1과 다중 동시성 응답을 분리해 확인한다. 오류가 나면 container log, capability 값, backend 선택 경로를 함께 보관한다.
