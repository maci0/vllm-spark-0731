# 06-13. Qwen3.8 serving 레시피

이 페이지는 [06-12. Qwen3.8-27B로 사람들이 만든 것](06-12-qwen38-community-builds.md)의 상세 내용입니다.

Qwen3.8 사례를 읽을 때는 원본 모델, 파생 weight, serving recipe를 먼저 구분한다. 같은 이름이 있어도 architecture와 chat template이 다를 수 있다.

## 시작 순서

1. 모델 카드와 config의 architecture를 확인한다.
2. runtime이 해당 architecture를 인식하는지 확인한다.
3. BF16 또는 공식 quant recipe로 짧은 생성부터 한다.
4. tool parser와 reasoning parser를 별도로 켠다.
5. context와 concurrency를 한 단계씩 올린다.

현재 local runner의 핵심은 `qwen3_xml` parser, `--enable-auto-tool-choice`, `max-model-len`과 model path다. 옵션을 복사할 때도 runtime 버전과 commit을 함께 고정한다.
