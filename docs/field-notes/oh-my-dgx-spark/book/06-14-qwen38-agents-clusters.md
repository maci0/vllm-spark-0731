# 06-14. Qwen3.8 에이전트와 멀티 Spark

이 페이지는 [06-12. Qwen3.8-27B로 사람들이 만든 것](06-12-qwen38-community-builds.md)의 상세 내용입니다.

Qwen3.8은 단일 Spark에서 coding worker, UI·디자인 worker, JSON 변환 endpoint 역할을 맡길 수 있다. DeepSeek를 brain으로 두고 Qwen3.8에 짧은 작업을 넘기는 구성은 모델별 역할을 나누는 한 가지 방법이다.

## 멀티 Spark 배치

- 독립 worker를 여러 노드에 배치하면 DP처럼 운영할 수 있다.
- 하나의 큰 모델을 나누면 TP·PP 통신 비용을 측정해야 한다.
- vision, tool runner, browser는 모델과 별도 프로세스로 격리한다.

에이전트 평가에서는 답변 품질만 보지 않는다. tool arguments, 오류 복구, 종료 조건, 권한 경계를 함께 기록한다.
