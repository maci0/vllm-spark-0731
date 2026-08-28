# Archify 다이어그램

책의 핵심 설명 그림은 Archify architecture specification을 원본으로 관리합니다.

- `*.architecture.json`: 배치, 연결, 라벨을 정의하는 원본입니다.
- `rendered/*.html`: Archify가 검증·생성한 독립형 HTML입니다. GitHub에서 구조를 확인하거나 로컬 브라우저로 열 수 있습니다.
- `assets/archify-*.svg`: WikiDocs 본문에 삽입하는 정적 SVG입니다.

정적 SVG를 만들 때 HTML에서 허용되는 valueless `data-*` 속성을 XML 호환 값(`="true"`)으로 정규화합니다. 따라서 GitHub와 WikiDocs의 SVG 이미지 파서에서도 읽을 수 있습니다.

WikiDocs 본문은 GitHub 화면보다 좁게 표시될 수 있으므로 추출 스크립트가 노드 라벨과 세부 라벨의 인쇄용 글자 크기를 별도로 지정합니다. 다이어그램의 구조·좌표·문구는 바꾸지 않고, `assets/`의 정적 SVG에서만 읽기 크기를 보정합니다.

## 다시 생성하기

후보를 수정한 뒤 각 파일을 `showcase` 품질로 검증하고 HTML을 생성합니다.

```bash
for spec in docs/diagrams/archify/*.architecture.json; do
  name=$(basename "$spec" .architecture.json)
  node /home/grid/.agents/skills/archify/bin/archify.mjs validate architecture "$spec" --quality showcase --json
  node /home/grid/.agents/skills/archify/bin/archify.mjs deliver architecture "$spec" "docs/diagrams/archify/rendered/$name.html" --quality showcase --json
  node tools/extract_archify_svg.mjs "docs/diagrams/archify/rendered/$name.html" "assets/archify-$name.svg"
done
python3 tools/build_wikidocs_export.py
```

`visual-check`는 Chrome 또는 Chromium이 설치된 환경에서 추가로 실행합니다. 실행 환경에 브라우저가 없으면 검증을 통과했다고 기록하지 않고 `skipped`로 남깁니다.
