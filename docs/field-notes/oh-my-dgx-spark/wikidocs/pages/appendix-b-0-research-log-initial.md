# B-0. 자동 리서치 초기 기록

상태: 날짜별 리서치 색인 / 본문 승격 대기
기준일: **2026-08-22**

이 장은 DGX Spark·GB10·로컬 AI 생태계를 계속 관찰하면서 날짜별 리서치 문서로 이동하는 색인이다. 여기의 내용은 책의 최종 결론이 아니라, 출처를 확인하고 실험한 뒤 본문으로 승격하기 위한 작업 기록이다.

각 날짜의 상세 리서치 문서는 저장소의 `docs/research-issue-N_YYYY-MM-DD.md`에 단 한 번 기록한다. WikiDocs export가 이 원문을 `appendix-b-research-issue-N_YYYY-MM-DD.md`라는 공개 서브챕터로 변환해 부록 B 아래에 배치한다. 따라서 이 파일에는 날짜별 보고서 내용을 다시 붙여 넣지 않는다.

자동 리서치 Workflow도 이 장을 수정하지 않는다. 새 보고서는 `docs/`에만 생성되고, `main` 병합 뒤 WikiDocs 배포 Workflow가 현재 `docs/` 목록을 다시 계산한다. `docs/`의 날짜별 보고서를 삭제하면 다음 배포에서 대응하는 WikiDocs 서브챕터도 `rsync --delete`로 제거된다.

## 3분 이해 (ELI5)

리서치 로그는 실험실 노트다.

```text
후보 → 분석 → 교차 확인 → 재현 → 승격
```

이 장의 기록은 결론 자체가 아니라, 결론이 만들어지는 과정이다.

## 이 로그를 읽는 법

- `후보`: 링크와 주제가 수집됐지만 아직 충분히 확인하지 않은 자료
- `분석`: 출처를 읽고 서로 비교한 상태
- `교차 확인`: 공식 자료·독립 자료·직접 측정 중 둘 이상이 같은 사실을 지지하는 상태
- `재현`: 하드웨어·모델·런타임·조건을 기록한 실험이 통과한 상태
- `승격`: 별도 본문 PR에서 검토하고 책의 안정적인 설명으로 옮긴 상태

각 기록에서 `빠르다`, `지원한다`, `안정적이다` 같은 표현은 출처의 주장인지 직접 확인한 결과인지 함께 표시한다. 검증되지 않은 항목은 본문의 사실처럼 읽히지 않도록 `보류`, `재현 대기`, `커뮤니티 보고`로 남긴다.

## 승격 원칙

1. 원문 URL과 출처 유형을 남긴다.
2. 수치에는 모델 revision, quant, runtime, 노드 수, context, 동시성, 측정 방법을 붙인다.
3. 원인 설명과 관찰된 증상을 구분한다.
4. 공식 recipe 또는 독립된 재현이 없으면 단일 커뮤니티 보고로 표시한다.
5. 충분히 확인된 내용만 별도 본문 PR로 승격한다.
6. 승격 뒤에도 기준일과 버전을 남겨 오래된 결과를 다시 확인할 수 있게 한다.

상세 기준은 [리서치 승격 규칙](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/research-promotion.md)에 정리한다.

## 2026-08-22 — 자동 수집 흐름 첫 기록

### 수집

- 자동 수집 후보: [Issue #2](https://github.com/recrack/oh-my-dgx-spark/issues/2)
- 자동 수집 결과: 40개 후보로 기록됨
- 수집 범위: DGX Spark·GB10 관련 GitHub, NVIDIA Forum, Reddit RSS와 등록된 검색 소스
- 주의: 수집 개수는 관련성이나 사실 확인을 의미하지 않는다.

### 분석 초안

- Copilot 분석 결과: [draft PR #4](https://github.com/recrack/oh-my-dgx-spark/pull/4)
- 상세 분석 문서: [PR #4의 `docs/research-issue-2.md` 제안](https://github.com/recrack/oh-my-dgx-spark/pull/4/files)
- PR #4의 book 변경은 자동 제안이며, 이 로그에서는 아직 승격된 사실로 취급하지 않는다.
- 자동 분석이 제안한 TP=2 hard-reset, `MAX_NUM_BATCHED_TOKENS`, Qwen3.8 recipe 항목은 출처와 재현 조건을 다시 확인해야 한다.

### 다음 검증

- [ ] 공식 recipe와 upstream 변경 상태를 다시 확인한다.
- [ ] 커뮤니티 수치에 모델·quant·runtime·context·동시성 조건을 붙인다.
- [ ] TP=2 장애는 로그와 복구 가능성을 분리해 확인한다.
- [ ] 재현을 통과한 항목만 운영·모델 레시피 등 관련 본문으로 승격한다.

## 날짜별 서브챕터 운영

- 원본: `docs/research-issue-N_YYYY-MM-DD.md`
- 공개 경로: `wikidocs/pages/appendix-b-research-issue-N_YYYY-MM-DD.md` 및 연결된 WikiDocs 책
- 자동 반영: 원본을 `main`에 병합한 뒤 `Publish WikiDocs` Action이 export와 배포를 수행한다.
- 삭제: 원본을 삭제하고 `main`에 병합하면 export 결과와 배포 저장소에서도 대응 페이지가 제거된다.
- 본문 승격: 운영·모델 레시피 등 안정된 본문은 별도의 작은 PR에서 출처와 재현 조건을 다시 확인한다.
