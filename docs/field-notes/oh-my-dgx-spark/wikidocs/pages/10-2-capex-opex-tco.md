# 10-2. CAPEX·OPEX·TCO

이 페이지는 10-1. 비용·전력·구성 의사결정의 상세 내용입니다.

장비 비교에서 가격 하나만 보면 판단이 흔들린다.

- CAPEX: 장비·스위치·케이블을 처음 사는 비용
- OPEX: 전기·냉각·스토리지·운영 시간
- TCO: 정한 기간 동안 CAPEX와 OPEX를 합친 비용

## 계산에 넣을 것

노드 수, 메모리 용량, 실제 wall power, 사용 시간, 전기 단가, 스위치와 케이블, 장애 대응 시간을 기록한다. `nvtop`의 값은 wall meter와 다를 수 있으므로 전력 비교에서는 측정 위치를 표시한다.

tok/s만으로 비용을 나누지 않는다. 필요한 것은 model fit, context, concurrency, quality gate를 통과한 처리량이다.
