# 06-10. DeepSeek 단일·듀얼 Spark 제작물

이 페이지는 [06-9. DeepSeek V4 Flash 0731로 사람들이 만든 것](06-9-deepseek-community-builds.md)의 상세 내용입니다.

커뮤니티 사례는 단일 Spark와 듀얼 Spark를 먼저 나눠 읽는다.

- 단일 Spark: EXL3, KV cache, speculative decoding으로 context와 decode를 노린다.
- 듀얼 Spark: TP·PP·DP, supervisor, prefill/decode 역할 분리를 실험한다.

각 사례에서 “모델이 올라갔다”, “endpoint가 응답했다”, “몇 tok/s가 나왔다”를 분리한다. recipe의 원본 commit과 hardware가 다르면 같은 결과로 취급하지 않는다.

## 재현 카드

`nodes · model/quant · runtime · context · concurrency · prefill · decode · quality test`

이 카드가 없는 숫자는 아이디어로만 기록한다.
