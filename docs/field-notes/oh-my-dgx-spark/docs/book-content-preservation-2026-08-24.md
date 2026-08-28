# 책 원고 보존·재배치 기록

기준일: **2026-08-24**

## 목적

상위 장을 줄이더라도 기존 원고의 실측, 명령, 사례와 출처가 사라져서는 안 됩니다. 이번 재배치는 00~10장을 독자의 기본 읽기 경로로 유지하고, 통합 전에 있던 상세 원고를 해당 장 아래의 WikiDocs 서브챕터로 공개하는 방식으로 수정했습니다.

## 보존한 범위

- 통합 커밋 이전에 삭제되었던 `book/` 상세 원고 49개의 본문을 복원했습니다.
- 통합 과정에서 내용이 바뀐 00·01·02장과 초기 리서치 로그는 제목만 새 구조에 맞춘 별도 상세 페이지로 복원했습니다.
- 통합 전 `book/README.md`는 [통합 전 책 구성 기록](book-structure-before-consolidation-2026-08-23.md)으로 보존했습니다.
- 새 상위 장의 독자 안내, 결정표와 요약은 유지했습니다. 기존 상세 원고를 대체하지 않고 들어가는 입구 역할을 합니다.

검증 기준이 된 통합 직전 커밋은 `74ad9b6`입니다. 복원 시점에는 49개 파일을 이 커밋과 바이트 단위로 비교했습니다. 이후 새 목차에 맞춰 파일명, H1과 장 간 링크 번호를 변경했지만 실험 내용, 명령, 수치와 출처는 삭제하지 않았습니다. 별도 상세 페이지 네 개도 같은 원칙으로 현재 위치에 맞춘 제목과 링크만 조정했습니다.

## 새 읽기 구조

| 상위 장 | 보존·재배치한 상세 내용 |
|---|---|
| 00 | 실행 상태, 증거 등급, 숫자와 명령을 읽는 기준 |
| 01 | 구매·용도 판단, GB10 벤더, 냉각·지원·대안 비교 |
| 02 | unified memory, KV, 양자화, TP·PP·DP의 상세 설명 |
| 03 | 첫 부팅, preflight, smoke test, Qwen·DeepSeek 첫 실행 |
| 04 | vLLM·SGLang·llama.cpp·SparkInfer, endpoint와 parser |
| 05 | 벤치마크 Level 0~6, 결과 schema와 품질 하니스 |
| 06 | 양자화, DeepSeek·Qwen·MiniMax, 커뮤니티 제작물과 활용 사례 |
| 07 | 두 대 직결, TP=2, 세·네·여덟 대와 스위치 확장 |
| 08 | 로컬 에이전트, endpoint 계약, brain·worker·권한 분리 |
| 09 | 발열, 저클럭, OOM, NCCL·RDMA, 펌웨어와 복구 절차 |
| 10 | 비용·전력·TCO, 노드 수 구매 판단, Sol 비교 |
| 부록 A | 명령어, 결과 양식, 모델·레시피 색인 |
| 부록 B | 초기 자동 리서치 기록과 날짜별 생성 보고서 |

## 누락 방지 규칙

고정 페이지의 단일 목차 원본은 [`book/TOC.md`](../book/TOC.md)입니다. export는 `book/`의 모든 Markdown 파일을 검사하며 `README.md`와 `TOC.md`를 제외한 파일이 목차에 없으면 실패합니다. 목차 번호와 파일명 접두사가 다르거나, 원고의 H1이 목차 제목과 다를 때도 실패합니다. 따라서 원고 파일을 복원하거나 새로 추가했는데 WikiDocs 배포에서 조용히 빠지는 상태와 제목·파일 번호가 서로 어긋나는 상태를 모두 막습니다.

날짜별 리서치의 단일 원본은 `docs/research-issue-N_YYYY-MM-DD.md`입니다. export는 이를 `wikidocs/pages/appendix-b-research-issue-N_YYYY-MM-DD.md`로 만들고 부록 B 아래에 배치합니다. 자동 수집 Workflow는 원본만 수정하며, 공개 파일을 `book/`에 중복 생성하지 않습니다.

현재 기준으로 고정 페이지 66개와 날짜별 리서치 페이지 1개, 모두 67페이지를 생성합니다. 상위 장은 11개와 부록 2개이며, 나머지는 한 단계 아래의 서브챕터입니다.

## 검증 명령

```bash
python3 tools/build_wikidocs_export.py
python3 tests/check_wikidocs_export.py
python3 -m py_compile tools/build_wikidocs_export.py tests/check_wikidocs_export.py
git diff --check
```

`check_wikidocs_export.py`는 다음 항목을 검사합니다.

- `book/`에 있으나 `book/TOC.md`에 없는 고정 원고를 거부합니다.
- 목차 번호, 파일명 접두사와 원고 H1이 서로 맞는지 확인합니다.
- 검증 실패가 기존 export bundle을 지우지 않는지 확인합니다.
- 고정 원고와 날짜별 리서치의 추가·삭제가 `pages/`와 `TOC.md`에 함께 반영되는지 확인합니다.
- 생성 페이지의 H1 제목이 새 독자용 목차 제목과 일치하는지 확인합니다.
- WikiDocs에서 깨지는 Markdown 파일명 링크가 남지 않는지 확인합니다.
