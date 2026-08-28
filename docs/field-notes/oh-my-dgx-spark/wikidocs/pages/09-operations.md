# 09. 발열·장애·복구

성능이 떨어졌을 때 곧바로 모델이나 양자화를 바꾸지 않습니다. 전원·클럭·온도, 메모리·디스크, runtime, 네트워크를 순서대로 분리해 확인해야 합니다.

## 5분 진단

아래 출력은 장애가 발생한 시각과 함께 저장합니다. secret이 출력될 수 있는 환경 변수나 command line은 포함하지 않습니다.

```bash
date -Is
nvidia-smi
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE
docker ps
free -h
df -h
dmesg --level=err,warn | tail -n 80
```

그다음 서버 로그에서 model load, CUDA error, OOM, NCCL timeout을 찾습니다. `GPU-Util=96%`, `P-state=P0`, throttle reason 없음만으로 정상 상태라고 판정하지 않습니다. 부하 중 SM clock과 power draw도 반드시 함께 확인합니다.

## 저클럭 증상은 별도 사건으로 기록한다

커뮤니티에는 같은 증상이 보고되었습니다. 한 사례에서는 Ornith 1.5 35B를 실행할 때 GPU utilization 약 96%, P0, SM clock 약 799MHz, GPU power 약 19.5W, decode 약 42.7 tok/s가 관찰되었습니다. 전원을 약 10분간 완전히 분리한 뒤에는 SM clock 약 2.3~2.5GHz, 약 92W, decode 약 73.9 tok/s로 회복되었다고 작성자가 보고했습니다.

이것은 특정 사용자의 관찰값입니다. 모든 저속 현상의 원인이 전원 상태라고 확정하지 않습니다. 전원 차단 전후의 로그와 벽면 전력 측정을 함께 보존해야 합니다.

참고: [커뮤니티 사례 원문](https://x.com/Blackwellboy/status/2090611479653622261?s=20), [DGX Spark 포럼 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-nvidia-forum-research-2026-08.md).

## 클럭 cap은 실험 프로필이다

GB10 clock cap과 관련된 공개 harness와 사용 보고에는 다음 명령이 제시되어 있습니다.

```bash
sudo nvidia-smi -lgc 0,2200
sudo nvidia-smi -rgc
```

첫 번째 명령은 최대 graphics clock을 제한하고, 두 번째 명령은 제한을 되돌립니다. 낮은 클럭은 decode가 memory-bound인 작업에서는 속도 손실이 작을 수 있지만, prefill과 BF16처럼 compute-bound인 작업에는 더 큰 영향을 줄 수 있습니다.

따라서 cap을 적용할 때는 다음 항목을 같은 하니스로 비교해야 합니다.

```text
clock · temperature · GPU power · wall power · prefill · decode · TTFT · soak errors
```

`nvtop`의 GPU rail power와 벽면 AC 측정값은 같은 값이 아닙니다. 온도만 낮아졌다는 이유로 효율이 좋아졌다고 결론 내리지 않습니다. [GB10 clock cap harness](https://github.com/agjs/gb10-clock-cap), [field artifact](https://nacyot.github.io/artifacts/gb10-clock-cap/)를 실험 설계 참고자료로 사용합니다.

## OOM과 긴 context

OOM이 나면 먼저 다음 순서로 설정을 줄입니다.

1. `max_model_len`과 실제 prompt 길이를 확인합니다.
2. KV cache dtype, `max_num_seqs`, batch와 speculative 설정을 기록합니다.
3. 모델·runtime의 메모리 사용량과 OS의 `MemAvailable`을 확인합니다.
4. 짧은 context와 한 stream으로 서버를 다시 기동해 weight 문제와 KV 문제를 분리합니다.
5. 안정화된 뒤에만 context와 concurrency를 하나씩 올립니다.

`max_model_len=384000`이 설정되었다는 사실은 모든 요청이 그 길이까지 안정적이라는 뜻이 아닙니다. DeepSeek EXL3 레시피의 370K needle 결과와 실제 code·tool 작업의 장문 품질도 별도로 기록합니다.

## NCCL·RDMA 장애

두 대 이상에서 서버는 뜨지만 요청이 멈추면 다음 항목을 분리해 확인합니다.

- QSFP link speed와 interface 상태
- IP·MTU·bridge·OOB 관리망
- NCCL이 선택한 interface와 socket fallback
- driver·CUDA·container·runtime commit
- collective test의 bandwidth와 timeout

특정 커뮤니티 레시피가 `UCX_MEM_MMAP_HOOK_MODE=none` 같은 환경 변수를 제시하더라도, 이를 모든 Spark의 공식 해결책처럼 복사하지 않습니다. 적용 전후의 soak 결과와 되돌리기 방법을 함께 남깁니다.

## 복구 순서

1. 요청을 멈추고 현재 model·runtime·clock 설정을 기록합니다.
2. `nvidia-smi -q`, container 상태, 서버 로그와 최근 benchmark를 저장합니다.
3. context·concurrency·speculative decoding을 낮춘 최소 설정으로 재현합니다.
4. 한 노드·한 endpoint로 줄여 전원/driver/runtime/network 중 어느 계층에서 문제가 생겼는지 확인합니다.
5. 설정을 되돌릴 수 있는 상태에서만 DGX OS나 driver를 변경합니다.
6. 소프트웨어 원인이 남지 않았고 안전한 종료가 확인될 때만 전원 완전 차단을 고려합니다.

전원 차단은 진단 데이터가 사라질 수 있으므로 첫 단계가 아닙니다. 다만 저클럭 bug가 의심되고 공식 안전 절차를 따를 수 있다면, 전후 수치를 비교하기 위한 복구 단계로 고려할 수 있습니다.

## 장애 보고서에 남길 것

```text
incident time · host · model/revision · image digest
clock cap · SM clock · P-state · temperature · GPU/wall power
context · concurrency · prefill · decode · TTFT
network link · NCCL result · error log · action · recovery result
```

이 항목이 없으면 “느려졌다”는 관찰을 다른 사람이 재현할 수 없습니다.

## 더 자세히 읽기

- 09-1. 발열·OOM·NCCL·펌웨어 장애: 장애를 네 계층으로 나누고 전체 복구 순서를 설명합니다.
- 09-2. 장애 발생 후 첫 5분: 재부팅 전에 수집할 정보를 정리합니다.
- 09-3. 저클럭·OOM·NCCL 대응: 증상별 최소 대응 순서입니다.
