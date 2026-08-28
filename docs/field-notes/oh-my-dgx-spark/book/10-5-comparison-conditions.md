# 10-5. 비교 조건 고정하기

이 페이지는 [10-4. GPT-5.6 Sol과 로컬 모델 비교](10-4-gpt56-sol-comparison.md)의 상세 내용입니다.

Sol과 local model을 비교할 때 모델 이름만 맞추면 안 된다. API tier, reasoning effort, context, prompt, output limit, tool 환경을 고정한다.

## 비교표

```text
model / endpoint / date
reasoning setting
system prompt
input tokens / output limit
tool availability
latency p50 / p95
quality rubric
failure and retry policy
```

호스티드 모델의 공식 점수와 DGX Spark의 local C1 결과는 같은 행에 섞지 않는다. 공식 score는 참고 자료이고, 우리 workload의 acceptance test는 별도 결과다.
