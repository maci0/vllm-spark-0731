# WikiDocs page ID 회수와 본문 링크 연결

기준일: **2026-08-22**

## 결론

GitHub 연동 책에는 두 종류의 링크가 있습니다.

| 위치 | 올바른 형식 | 역할 |
|---|---|---|
| `TOC.md` | `pages/파일명.md` | 페이지 생성과 상·하위 계층 정의 |
| 페이지 본문 | `https://wikidocs.net/<숫자형 page_id>` | 공개된 WikiDocs 페이지로 이동 |

따라서 `https://wikidocs.net/03-1-first-boot-safe-environment.md`는 사용하지 않습니다. 파일명은 GitHub 원고의 식별자이고, WikiDocs URL은 페이지가 생성된 뒤 부여되는 숫자형 ID를 사용합니다.

## 왜 공개 설정만으로 해결되지 않는가

WikiDocs 책을 공개하면 독자가 페이지를 읽을 수 있습니다. 그러나 공개 설정은 GitHub 파일명과 WikiDocs 페이지 ID를 연결하는 매핑을 만들지 않습니다. 또한 GitHub 연동 책에서 `TOC.md`의 파일명 링크는 목차를 가져오기 위한 입력값이지, 페이지 본문에 삽입할 URL이 아닙니다.

본문 링크를 연결하려면 다음 순서를 지킨다.

1. GitHub 저장소에 안정된 `README.md`, `TOC.md`, `pages/`, `assets/`를 push합니다.
2. WikiDocs webhook 또는 책 수정 화면의 `지금 동기화`로 페이지를 생성·갱신합니다.
3. 인증된 CLI로 책의 TOC를 조회해 파일명, 제목, 숫자형 page ID를 회수합니다.
4. 파일명과 ID의 매핑을 저장하고 export를 다시 실행합니다.
5. 생성된 본문에 `https://wikidocs.net/<page_id>` 링크가 들어갔는지 검사한 뒤 push합니다.

## ID 조회

WikiDocs CLI 공식 문서에 따라 CLI와 API 토큰을 준비한 뒤, 책 ID를 사용해 TOC를 JSON으로 조회합니다. 토큰은 소스 코드나 채팅에 기록하지 않습니다.

```bash
npm install -g @wikidocs/cli
wikidocs --json books toc "$BOOK_ID" > /tmp/wikidocs-toc.json
```

설치한 CLI의 명령어가 다르면 `wikidocs --help`에서 `books toc` 형식을 확인합니다. TOC 결과의 각 항목에서 `id`, `subject`, `children`을 확인하고, GitHub의 `pages/` 파일명과 제목을 대조합니다. 제목만으로 매칭하지 말고 중복 제목과 순서도 함께 확인합니다.

## 매핑 파일

메인 저장소에 `docs/wikidocs-page-map.json`을 만들고 다음 형식으로 저장합니다. 숫자는 실제 TOC 조회 결과로 바꿉니다.

```json
{
  "book_id": 12345,
  "pages": {
    "00-how-to-read.md": 123456,
    "03-first-model.md": 123457,
    "06-model-recipes.md": 123458
  }
}
```

`pages/파일명.md`를 키로 사용해도 됩니다. export 스크립트가 앞의 `pages/`를 제거하고 처리합니다. `book_id`는 기록용이며, 본문 링크 생성에는 각 페이지의 `pages` 값만 사용합니다.

```bash
python3 tools/build_wikidocs_export.py
python3 tests/check_wikidocs_export.py
git diff -- wikidocs/pages
```

매핑 파일이 없을 때는 아직 ID를 회수하지 않은 초기 상태로 보고 해당 본문 링크를 일반 제목으로 내보냅니다. 매핑 파일을 만들었다면 고정 페이지와 이미 동기화된 날짜별 리서치 페이지의 ID를 모두 채워야 합니다. 아직 WikiDocs에 생성되지 않은 새 날짜별 리서치 페이지는 ID가 없을 수 있으므로 export가 허용합니다. 그러나 고정 페이지의 ID가 빠지거나 어떤 ID든 중복되면 export가 실패합니다. 날짜별 원문이 삭제되어 현재 TOC에서 사라진 오래된 날짜 페이지 ID는 stale mapping으로 허용하며, 새 export에는 사용하지 않습니다.

## 페이지를 추가하거나 이동할 때

`TOC.md`에 새 파일을 추가하면 WikiDocs가 새 페이지 ID를 부여할 수 있습니다. 날짜별 리서치 페이지는 `docs/` 원문이 존재할 때만 export됩니다. 원문을 삭제한 뒤 다시 만들면 제목이 같아도 ID가 달라질 수 있습니다. 그러므로 구조를 바꾼 뒤에는 TOC를 다시 조회하고 매핑 파일을 갱신합니다. 기존 페이지의 상·하위 관계만 바뀌고 페이지 자체가 유지되면 URL ID가 유지되는지 확인합니다.

파일명을 바꾸면 제목이 같아도 기존 ID를 그대로 쓸 수 있다고 가정하지 않습니다. 이번 00~10장 재배치처럼 `17-*`, `18-*`, `19-*` 파일을 새 장 번호와 `appendix-b-*` 이름으로 바꾼 경우에는 먼저 WikiDocs 동기화를 완료한 뒤 전체 TOC를 다시 조회합니다. 이전 파일명으로 만든 매핑은 새 export에 복사하지 않습니다.

## 참고

- [WikiDocs 새 책 만들기: GitHub 연동](https://wikidocs.net/321336)
- [WikiDocs CLI](https://wikidocs.net/390627)
- [WikiDocs 스킬집의 GitHub 연동·page ID 운영 예시](https://wikidocs.net/346917)
