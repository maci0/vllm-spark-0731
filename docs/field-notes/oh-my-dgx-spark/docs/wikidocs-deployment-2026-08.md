# WikiDocs 배포 계획과 상태

기준일: **2026-08-23**

## 목표

`로컬 AI 실전 레시피: DGX Spark 편`을 WikiDocs의 GitHub 연동 책으로 운영합니다. 부제는 `Qwen3.8·DeepSeek부터 멀티 Spark 클러스터와 코딩 에이전트까지`입니다. 원고의 기준 저장소는 [oh-my-dgx-spark](https://github.com/recrack/oh-my-dgx-spark)로 두고, WikiDocs가 읽는 배포용 저장소에는 `README.md`, `TOC.md`, `pages/`, `assets/`만 둡니다.

WikiDocs 책 설명란에는 다음 요약을 사용합니다.

> DGX Spark에서 Qwen3.8과 DeepSeek를 실행하고, 한 대에서 여러 대의 클러스터로 확장하며, 코딩 에이전트와 연결하는 실전 레시피입니다. vLLM·SGLang·llama.cpp, FP8·NVFP4·EXL3, MTP·DSpark·DFlash2, tool parser, RoCE·NCCL·스위치 구성, 발열·OOM·저클럭 장애를 다룹니다. 공식 자료와 커뮤니티 레시피, 직접 측정한 결과를 구분해 어떤 모델을 어떤 조건에서 선택해야 하는지 설명합니다.

WikiDocs 공식 안내에 따르면 GitHub 연동 책에서는 다음 파일의 역할이 나뉩니다. 기존 GitHub 저장소를 연결할 수도 있지만, 책 생성은 WikiDocs 로그인과 GitHub OAuth 권한 승인 뒤에 진행됩니다.

| 파일·폴더 | 역할 |
|---|---|
| `README.md` | 책 제목과 소개 |
| `TOC.md` | 책 목차와 페이지 순서 |
| `pages/` | 장별 Markdown 원고 |
| `assets/` | 이미지와 첨부 자산 |

고정 원고는 `oh-my-dgx-spark/book/`에서 수정하고, 날짜별 리서치 원문은 `oh-my-dgx-spark/docs/research-issue-N_YYYY-MM-DD.md`에서 수정합니다. `WIKIDOCS_DEPLOY_TOKEN` secret을 한 번 등록한 뒤 `main`에 commit·push하면 Actions가 export, 검사, 배포 저장소 push를 자동으로 실행합니다. 배포 저장소는 WikiDocs 책에 연결되어 있으므로 등록된 webhook이 책 동기화까지 처리합니다. 반영이 지연되면 책 수정 화면의 GitHub 탭에서 `지금 동기화`를 실행합니다. 책의 공개 설정과 GitHub 저장소의 공개 설정은 서로 별개입니다.

## 현재 상태

| 항목 | 상태 | 근거 |
|---|---|---|
| 원고 | WikiDocs GitHub 연동 배포판 | 00~10장 + 기존 상세 원고 서브챕터 + 부록 A·B + 날짜별 리서치 서브챕터 |
| GitHub 원고 저장소 | 공개 | [`recrack/oh-my-dgx-spark`](https://github.com/recrack/oh-my-dgx-spark) |
| Sol max 비교 | 조사·검증 반영 | [Sol max 리서치](sol-max-comparison-research-2026-08.md) |
| DGX Spark 실험 | 2026-08-22 C1·tool parser·multi-turn mock loop 기록 | [DeepSeek 성능 리서치](deepseek-v4-flash-0731-performance-research-2026-08.md) |
| export bundle | 생성·검사 완료 | `wikidocs/README.md`, `TOC.md`, `pages/`, `assets/book-cover-v1.png`, 고정 목차와 날짜별 리서치 페이지 |
| export 검사 | 통과 | `python3 tests/check_wikidocs_export.py` |
| GitHub 배포 저장소 | 비공개, WikiDocs 연결 대상 | [`recrack/oh-my-dgx-spark-wikidocs`](https://github.com/recrack/oh-my-dgx-spark-wikidocs), 원격 latest `b8e9170` |
| WikiDocs 책·webhook | 연결 완료, push 자동 동기화 | 새 저장소에 `wikidocs.net` 대상 active `push` webhook 확인; 책 수정 화면에서 `지금 동기화` 가능 |
| 원고 저장소 → 배포 저장소 | GitHub Actions 자동화 추가 | `WIKIDOCS_DEPLOY_TOKEN` secret을 한 번 등록하면 `main` push마다 export·검사·배포 |

## WikiDocs 매뉴얼 점검 결과

[GitHub 연동 책 공식 안내](https://wikidocs.net/321336)의 요구사항과 현재 bundle을 다음처럼 맞췄습니다.

- `README.md`의 첫 번째 `#` 제목은 책 제목으로 두고, 그 아래에 책 설명·요약을 배치했습니다.
- `TOC.md`는 `pages/파일명.md` 링크 형식을 사용하고, 제목에 `00`~`10` 번호를 붙여 읽기 순서를 유지합니다. 날짜별 리서치 페이지는 부록 B 아래에 자동으로 추가합니다.
- 기존 상세 원고는 상위 장 아래에 두 칸 들여쓴 서브챕터로 배치합니다. 현재 snapshot은 고정 페이지 66개와 날짜별 리서치 1개를 생성합니다.
- 루트 구조는 `README.md`, `TOC.md`, `pages/`, `assets/`로 제한한 WikiDocs 전용 bundle입니다.
- 표지는 영문 파일명 `assets/book-cover-v1.png`으로 저장했습니다. PNG는 매뉴얼의 권장 이미지 형식이며, 책 생성 후 `책 수정 > 기본`에서 대표 이미지로 지정합니다.

## siderefresh-guide-ko 링크 구조를 반영한 부분

기존 [siderefresh-guide-ko](https://github.com/recrack/siderefresh-guide-ko) 저장소의 연결 방식을 참고했습니다.

- 루트 `README.md`에서는 비공개 저장소에서도 표시되도록 `assets/book-cover-v1.png` 상대 경로를 사용합니다. 비공개 저장소의 `raw.githubusercontent.com` URL은 인증 토큰 없이는 404가 되므로 표지 링크로 사용하지 않습니다.
- `TOC.md`에는 `pages/파일명.md`를 넣고, 현재 책처럼 서브챕터는 두 칸 들여씁니다.
- `pages/` 안의 이미지가 있을 때는 `../assets/파일명` 상대 경로를 사용합니다.
- GitHub 연동 중인 페이지 본문의 장 간 링크는 WikiDocs가 발급한 숫자형 페이지 ID를 사용합니다. ID를 확인하기 전에는 `pages/파일명.md` 상대 링크를 넣지 않고 제목만 표시합니다.

## 본문 링크와 WikiDocs page ID

`TOC.md`의 `pages/파일명.md` 링크는 GitHub 연동기가 목차와 상·하위 관계를 만들 때 사용하는 입력 형식입니다. 이 경로가 WikiDocs 본문에서 클릭할 수 있는 주소로 바뀌는 것은 아닙니다. 본문에서 다른 WikiDocs 페이지로 이동하려면 동기화가 끝난 뒤 발급된 ID를 사용해 다음처럼 작성합니다.

```markdown
[03장. 첫 부팅에서 실패하지 않는 방법](https://wikidocs.net/123456)
```

따라서 `https://wikidocs.net/03-1-first-boot-safe-environment.md`는 올바른 WikiDocs 링크가 아닙니다. 공개 설정을 바꾸어도 GitHub 파일명에서 페이지 ID를 자동으로 추론할 수는 없습니다. 실제 ID는 책의 TOC 조회 결과에서 회수해야 합니다.

현재 export는 `docs/wikidocs-page-map.json`이 있으면 파일명과 숫자형 ID를 매핑해 절대 URL을 생성합니다. 매핑 파일이 없을 때는 깨진 상대 링크를 배포하지 않고 제목만 남깁니다. 회수 절차와 매핑 형식은 [WikiDocs page ID 회수 절차](wikidocs-page-id-recovery-2026-08.md)에 기록했습니다.

## 재생성 명령

```bash
python3 tools/build_wikidocs_export.py
python3 tests/check_wikidocs_export.py
```

생성 스크립트는 `book/TOC.md`에 등록된 고정 원고와 서브챕터를 `wikidocs/pages/`로 복사하고, `docs/research-issue-N_YYYY-MM-DD.md`를 부록 B 아래의 `wikidocs/pages/appendix-b-research-issue-N_YYYY-MM-DD.md`로 변환합니다. 메인 저장소의 `docs/`와 `tests/` 링크는 GitHub 절대 링크로 바꿉니다. WikiDocs page ID 매핑이 있으면 장 간 링크를 `https://wikidocs.net/<page_id>`로 바꾸고, 없으면 링크 제목만 남깁니다. 전체 장 구조와 이동은 WikiDocs가 해석하는 `TOC.md`의 `pages/파일명.md` 링크로 관리합니다.

날짜별 리서치 원문을 삭제하면 `get_pages()`가 해당 공개 페이지를 더 이상 만들지 않습니다. 이후 `rsync --delete`가 배포 저장소의 이전 `pages/appendix-b-research-issue-...md`와 `TOC.md` 항목을 제거합니다. 고정 장이나 서브챕터를 추가·이동·삭제할 때는 `book/TOC.md`를 함께 수정합니다. 반대로 `book/`에 Markdown 파일이 있는데 목차에 등록되지 않았으면 export가 실패하므로, 원고가 조용히 배포에서 누락되지 않습니다.

## 배포 저장소

연동 이름: `recrack/oh-my-dgx-spark-wikidocs`

이 저장소는 책의 GitHub 연동 전용으로 사용합니다. 메인 개발 저장소의 테스트 코드와 대용량 실험 파일을 독자에게 모두 노출하지 않고, WikiDocs가 요구하는 bundle만 제공합니다. 저장소는 비공개로 생성했고(`private: true` 확인), 현재 WikiDocs 책에 연결되어 있습니다. 이 저장소는 직접 수정하지 않으며, 원고 저장소의 Actions가 생성한 export bundle만 받습니다.

배포 저장소의 루트에는 다음 구조가 있어야 합니다.

```text
README.md
TOC.md
pages/
  00-how-to-read.md
  00-1-reading-principles.md
  01-why-dgx-spark.md
  01-2-gb10-vendor-comparison.md
  ...
  06-model-recipes.md
  06-5-deepseek-v4-flash.md
  06-12-qwen38-community-builds.md
  07-cluster.md
  07-1-two-spark-cluster.md
  08-agents.md
  09-operations.md
  10-decision.md
  appendix-a-commands.md
  appendix-b-research-log.md
  appendix-b-research-issue-N_YYYY-MM-DD.md  # docs/ 원문에서 export 시 생성
assets/
  book-cover-v1.png
```

표지 원본은 메인 저장소의 `assets/book-cover-v1.png`에 보관하고, export 스크립트가 이를 WikiDocs bundle의 `assets/`로 복사합니다. WikiDocs 계정에서 책을 만든 뒤에는 책 수정 화면의 기본 탭에서 이 이미지를 표지로 직접 지정합니다.

## 연동 배포 운영 체크

1. 고정 장과 서브챕터는 `book/`에서 수정하고, 순서는 `book/TOC.md`에서 관리합니다. 날짜별 리서치 원문은 `docs/research-issue-N_YYYY-MM-DD.md`에서 수정합니다.
2. 원고 저장소 `main`에 commit·push합니다.
3. Actions가 export, 검사, 배포 저장소 push를 자동으로 실행합니다.
4. 배포 저장소 push webhook이 연결된 WikiDocs 책을 자동으로 동기화합니다.
5. 반영이 지연되면 책 수정 화면의 GitHub 탭에서 `지금 동기화`를 실행합니다.
6. 책 제목·요약·표지와 `TOC.md`의 00장부터 10장, 부록 A·B와 날짜별 리서치 서브챕터 링크를 확인합니다.
7. `pages/` 안의 상대 링크, 외부 GitHub·Hugging Face·NVIDIA Forum·X·Reddit 링크, 코드 블록과 표가 정상적으로 보이는지 확인합니다.

`.github/workflows/publish-wikidocs.yml`이 `main` push마다 export와 검사를 실행한 뒤, `oh-my-dgx-spark-wikidocs`에 변경된 bundle을 push합니다. 배포 저장소에 push한 뒤 WikiDocs가 책으로 동기화하는 단계는 등록된 webhook으로 자동 처리됩니다. 최초 한 번만 배포 저장소에 `Contents: Read and write` 권한을 부여한 fine-grained token을 원고 저장소의 Actions secret에 `WIKIDOCS_DEPLOY_TOKEN` 이름으로 등록합니다.

배포 저장소의 자동 커밋은 원본 커밋을 추적할 수 있도록 `docs: sync WikiDocs — <원본 커밋 제목>` 형식을 사용합니다. 커밋 본문에는 원본 SHA, 원본 제목, bundle 통계와 실제 변경 파일 목록을 기록합니다. export 결과가 이전과 같으면 target 저장소에 빈 커밋을 만들지 않습니다.

```bash
gh secret set WIKIDOCS_DEPLOY_TOKEN \
  --repo recrack/oh-my-dgx-spark \
  --body '<oh-my-dgx-spark-wikidocs에 Contents: Read and write 권한이 있는 토큰>'
```

토큰 값은 채팅이나 원고에 기록하지 않습니다. secret 등록 뒤에는 `book/` 또는 날짜별 `docs/` 원문을 수정하고 `main`에 push하는 것만으로 배포합니다.

## 출처

- [새 책 만들기(깃허브 연동) 공식 안내](https://wikidocs.net/321336)
- [자동 생성된 GitHub 저장소 구조 예시](https://wikidocs.net/346592)
- [GitHub push 후 WikiDocs에서 확인하기](https://wikidocs.net/346513)

## 계정 화면에서 확인할 항목

WikiDocs 연결과 webhook 등록은 완료 상태입니다. GitHub API에서도 새 배포 저장소에 `wikidocs.net` 대상 active `push` webhook을 확인했습니다. 이 실행 환경에서는 WikiDocs 로그인 화면을 직접 열 수 없으므로, 다음 운영 정보는 책 수정 화면에서 확인합니다.

- WikiDocs 책 ID와 실제 책 URL
- WikiDocs 책의 공개/비공개 설정
- 책 화면에서의 Markdown·표·외부 링크 렌더링

연결 자체는 완료되어 있으므로 위 항목은 배포를 막는 조건이 아니라 운영 확인 항목입니다.
