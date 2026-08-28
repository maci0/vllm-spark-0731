# 06-3. speculative decoding과 품질 게이트

이 페이지는 06-1. 양자화·KV cache·speculative decoding의 상세 내용입니다.

speculative decoding은 draft가 여러 토큰을 제안하고 target이 검증하는 방식이다. 속도 이득은 workload에 따라 달라진다.

## A/B 조건

같은 prompt set에서 다음 두 프로필을 비교한다.

- target only
- target + draft

context, temperature, max tokens, concurrency, clock, quant, server commit을 고정한다. prefill과 decode를 분리해 기록한다.

## 품질 게이트

- 코드가 문법적으로 유효한가.
- JSON이 깨지지 않는가.
- tool name과 arguments가 유지되는가.
- 긴 답변에서 반복과 누락이 늘지 않는가.

속도가 올라도 품질 게이트를 통과하지 못하면 production profile로 채택하지 않는다. 반대로 속도 이득이 작아도 agent workload의 tail latency가 좋아질 수 있으므로 단일 tok/s만 보고 끄지 않는다.
