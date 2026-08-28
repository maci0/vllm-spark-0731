# 부록 B. DGX Spark 리서치 로그

이 부록은 DGX Spark와 GB10에 관한 공개 자료가 어떻게 책의 본문으로 승격되는지 설명하는 안내 페이지입니다. 매일 수집되는 원문을 이 파일에 계속 덧붙이지 않습니다. 날짜별 원문은 `docs/research-issue-N_YYYY-MM-DD.md`로 만들고, export가 `appendix-b-research-issue-N_YYYY-MM-DD.md`라는 WikiDocs 서브챕터로 변환합니다.

## 리서치 흐름

```text
검색·RSS·GitHub·사용자 Issue
        ↓
후보 URL 중복 제거
        ↓
출처·조건·수치 분석
        ↓
공식 문서와 교차 확인
        ↓
재현 가능성·직접 실험 여부 판정
        ↓
날짜별 보고서와 Draft PR
        ↓
사람 검토·본문 승격
```

자동 수집은 책에 바로 사실을 추가하지 않습니다. 검색 결과는 후보이고 LLM 분석은 초안입니다. PR 리뷰와 필요한 직접 실험을 통과한 내용만 본문 문장이 됩니다.

## 날짜별 페이지의 규칙

원본 파일명은 다음 형식을 사용합니다.

```text
docs/research-issue-N_YYYY-MM-DD.md
```

여기서 `N`은 GitHub Issue 번호이고, 날짜는 해당 보고서의 실행 날짜입니다. 후보 수를 40개로 고정하지 않습니다. 검색·RSS·GitHub 결과에서 실제로 수집된 링크 수를 그날의 보고서에 기록하며, 다음 실행에서는 후보 수가 늘거나 줄 수 있습니다.

export 결과는 다음처럼 됩니다.

```text
book/appendix-b-research-log.md
  └─ wikidocs/pages/appendix-b-research-issue-N_YYYY-MM-DD.md
```

날짜별 원본을 삭제하면 다음 export에서 해당 `pages/` 파일과 `TOC.md` 항목도 사라집니다. 고정 안내 페이지는 유지되지만, 날짜별 보고서는 현재 `docs/`에 존재하는 파일만 공개됩니다.

## 본문 승격 기준

리서치 보고서의 내용을 본문에 옮길 때는 다음을 확인합니다.

- 공식 사양·모델 카드·runtime 문서의 원문 링크가 있습니다.
- 공개 레시피의 model revision, quant, engine, context, concurrency와 측정 방법이 있습니다.
- tok/s, prefill, decode, TTFT와 end-to-end 결과를 섞지 않았습니다.
- 다른 사람이 보고한 수치와 이 저장소에서 직접 실행한 수치를 분리했습니다.
- “실행 가능”, “benchmark 통과”, “tool-tested”, “agent-tested”를 구분했습니다.
- 확인되지 않은 원인과 Sol 동급 같은 해석을 사실처럼 쓰지 않았습니다.

모델별 본문은 주로 06장. 모델 레시피, 클러스터 자료는 07장. 여러 Spark와 네트워크, 운영 장애는 09장. 발열·장애·복구로 승격합니다. 리서치 원문은 날짜별 서브챕터에 남겨 출처의 변화와 과거 판단을 추적할 수 있게 합니다.

## 현재 리서치의 근거 층위

| 등급 | 자료 | 사용 방법 |
|---|---|---|
| A | NVIDIA·OpenAI·모델 제작자의 공식 문서 | 사양, 지원 범위, API 계약의 기준 |
| B | 명령·container·config가 공개된 GitHub 레시피 | 재현 후보와 설치 경로 |
| C | 측정 조건을 공개한 benchmark·artifact | 조건부 수치와 비교 기준 |
| D | X·Reddit·포럼·Arca·서버포럼 사용담 | 문제 발견과 후속 재현 후보 |

D등급 자료의 숫자는 흥미로운 출발점이지만, 우리 장비의 보장값이 아닙니다. 예를 들어 MCDMA USB-C 전송 보고와 저클럭 회복 사례는 본문에서 커뮤니티 관찰값으로만 표시합니다.

## 사람이 확인할 체크리스트

- [ ] 링크가 실제로 열리고 원문 게시일과 작성자를 확인했습니다.
- [ ] 원문이 주장한 장비·모델·quant·runtime을 적었습니다.
- [ ] 숫자의 측정 단위와 workload를 적었습니다.
- [ ] 공식 사실과 작성자의 해석을 분리했습니다.
- [ ] 책 본문에 넣을 문장과 “재현 필요” 항목을 분리했습니다.
- [ ] 비밀값·개인정보·원문 전체를 저장하지 않았습니다.

자동화 구현과 보고서 형식은 [리서치 자동화 문서](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/research-automation.md), 승격 규칙은 [리서치 승격 규칙](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/research-promotion.md), 참고문헌 목록은 [책 참고문헌](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-book-references-2026-08.md)을 봅니다.

## 초기 기록과 날짜별 원문

B-0. 자동 리서치 초기 기록은 기존 원고에 있던 첫 수집 흐름과 daily 기록 형식을 보존합니다. 이후 자동 수집 보고서는 이 장의 하위 페이지로 날짜순으로 생성됩니다.
