# 07-3. DeepSeek TP=2 실행

이 페이지는 [07-1. 두 대 연결하기](07-1-two-spark-cluster.md)의 상세 내용입니다.

DeepSeek TP=2는 “두 대의 메모리를 합치면 된다”가 아니다. 통신 경로, model parallel 설정, context, KV 배치가 함께 맞아야 한다.

## 실행 전 기록

`node0`, `node1`의 hostname과 IP, ConnectX interface, NCCL·UCX 환경 변수, model revision, quant를 고정한다. 두 노드의 clock과 전력 상태도 같이 기록한다.

## 검증 순서

1. 양쪽에서 서로 ping한다.
2. RDMA와 NCCL collective를 확인한다.
3. 짧은 prompt 하나를 생성한다.
4. prompt length를 늘려 prefill과 KV를 확인한다.
5. 두 번째 요청과 오류 복구를 테스트한다.

한 노드만 GPU utilization이 높거나, socket fallback이 발생하면 속도 숫자를 기록하지 않는다. 먼저 통신 경로를 고친다.

## 결과 문장

“2대에서 모델이 실행됨”과 “TP=2가 단일 Spark보다 빨라짐”은 별도 주장이다. 두 결과를 각각 증명한다.
