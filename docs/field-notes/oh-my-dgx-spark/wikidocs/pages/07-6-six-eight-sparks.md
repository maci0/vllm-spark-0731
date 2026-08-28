# 07-6. 여섯·여덟 대와 확장 벤치마크

이 페이지는 07-4. 세 대·네 대·여덟 대의 상세 내용입니다.

여섯·여덟 대부터는 단순한 모델 실행이 아니라 클러스터 운영이 된다. 스위치, IP 계획, 전원, 냉각, 장애가 모두 실험 조건이다.

## 확장할 때 측정할 것

- 노드 추가 전후의 single-stream latency
- aggregate throughput과 요청 수
- 통신 bandwidth와 collective time
- 한 노드 장애 후 복구 시간
- idle·load 전력과 온도

노드를 늘려서 모델만 올라가고 throughput이 줄어들 수 있다. scale-out 결과는 `모델 수용`, `속도`, `동시성`, `운영 안정성` 네 행으로 보고한다.

## 중단 기준

NCCL hang, 반복적인 OOM, thermal throttling, 전원 불안정이 있으면 노드 수를 더 늘리지 않는다. 먼저 작은 topology에서 원인을 좁힌다.
