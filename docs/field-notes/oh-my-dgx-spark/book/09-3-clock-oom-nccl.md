# 09-3. 저클럭·OOM·NCCL 대응

이 페이지는 [09-1. 발열·OOM·NCCL·펌웨어 장애](09-1-operations-failure-recovery.md)의 상세 내용입니다.

저클럭은 GPU utilization이 높아도 정상 성능처럼 보이지 않을 수 있다. SM clock, power, temperature를 함께 본다.

```bash
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv
```

clock cap을 실험했다면 시작값과 reset 명령을 결과에 적는다. decode가 memory bound인지 prefill이 compute bound인지 분리해서 판단한다.

`nvidia-smi -lgc`로 의도적으로 건 상한과 고장성 저클럭은 구분한다. 부하 중 utilization과 P-state가 높아도 `clocks.sm`과 `power.draw`가 비정상적으로 낮으면 정상 성능으로 기록하지 않는다. 먼저 같은 workload에서 clock·power·temperature·실제 tok/s를 저장하고, [ASUS GX10 노드 세팅 현장 자료](../docs/dgx-spark-node-setup-research-2026-08.md)의 수치는 해당 장비의 community measurement로만 비교한다.

GPU rail power와 벽면 AC power는 같은 값이 아니다. clock sweep 결과에는 둘의 측정 위치와 context·concurrency·prefill/decode 조건을 함께 남긴다.

OOM은 단순히 weight가 큰 경우만 뜻하지 않는다. context, KV dtype, batch, CUDA graph, 다른 서버의 메모리 점유를 순서대로 확인한다.

NCCL 오류는 socket fallback으로 숨기지 않는다. interface와 RDMA 경로를 먼저 확인하고, 작은 collective를 통과시킨 뒤 model request로 넘어간다.
