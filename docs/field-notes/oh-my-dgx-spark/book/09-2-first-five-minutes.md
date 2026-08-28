# 09-2. 장애 발생 후 첫 5분

이 페이지는 [09-1. 발열·OOM·NCCL·펌웨어 장애](09-1-operations-failure-recovery.md)의 상세 내용입니다.

장애가 나면 설정을 먼저 바꾸지 않는다. 현재 상태를 보존해야 원인을 재현할 수 있다.

## 수집 순서

```bash
date -Is
nvidia-smi
nvidia-smi dmon -s pucm -c 5
free -h
df -h
dmesg --ctime | tail -n 80
```

멀티 Spark라면 각 노드에서 같은 명령을 실행하고 시각을 맞춘다. 서버 log, 요청 JSON, 마지막 성공 요청도 함께 저장한다.

## 증상 분류

- 낮은 tok/s + 낮은 clock: 전원·thermal·clock 상태
- GPU memory 부족: model·KV·batch 예산
- NCCL timeout: network·interface·collective
- JSON/tool 실패: parser·template·agent contract

한 번에 여러 변수를 바꾸면 복구는 되어도 원인은 남지 않는다.
