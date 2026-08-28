# 05-2. 벤치마크 레벨

이 페이지는 [05-1. 벤치마크를 제대로 설계하기](05-1-benchmark-design.md)의 상세 내용입니다.

벤치마크는 한 번에 크게 돌리지 않는다. 결과의 상태를 단계별로 올린다.

| 레벨 | 질문 | 통과 결과 |
|---|---|---|
| 0 | weight가 올라오는가 | `loaded` |
| 1 | 한 요청에 답하는가 | `generates` |
| 2 | 한 스트림 속도는 얼마인가 | prefill·decode 기록 |
| 3 | 동시 요청에서 유지되는가 | concurrency·queue 기록 |
| 4 | 긴 context에서 안정적인가 | prompt·KV·recall 기록 |
| 5 | tool·agent loop가 끝나는가 | call·error·stop 기록 |
| 6 | 오래 유지되는가 | soak·temperature·전력 기록 |

앞 단계가 실패한 상태에서 뒤 단계의 숫자를 만들지 않는다. 특히 모델이 load됐다는 사실만으로 agent benchmark를 시작하지 않는다.

## 최소 실행 순서

짧은 텍스트 → JSON → 멀티턴 → concurrency → long context → tool call 순서로 진행한다. 각 단계가 끝날 때 서버 log와 요청 JSON을 함께 저장한다.
