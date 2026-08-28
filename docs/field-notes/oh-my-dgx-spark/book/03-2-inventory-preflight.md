# 03-2. 인벤토리와 preflight

이 페이지는 [03-1. 첫 부팅에서 실패하지 않는 방법](03-1-first-boot-safe-environment.md)의 상세 내용입니다.

첫 실행에서 가장 먼저 할 일은 모델을 띄우는 것이 아니라 기준 상태를 남기는 것이다. 그래야 나중에 속도 저하와 OOM을 모델 문제, runtime 문제, 하드웨어 문제로 나눌 수 있다.

## 최소 수집 명령

```bash
hostnamectl
nvidia-smi
df -h
free -h
ip -br addr
ip route
```

가능하면 명령어 출력에 실행 시각과 장비 이름을 함께 남긴다. 모델 revision, 컨테이너 이미지, vLLM·SGLang commit도 같은 기록에 넣는다.

## 통과 기준

- GPU가 보이고 driver 오류가 없다.
- 모델을 받을 디스크에 여유 공간이 있다.
- 사용할 포트가 비어 있다.
- 기존 서버와 새 서버의 endpoint를 혼동하지 않는다.
- 실험 전 baseline을 저장했다.

이 페이지의 통과는 모델 생성 성공을 뜻하지 않는다. `loaded` 이전의 환경 확인만 완료한 상태다.

## 기록 템플릿

`hardware · driver · runtime · model revision · quant · context · port · git commit`

이 한 줄을 모든 benchmark 결과에 붙인다. 숫자만 남긴 tok/s는 나중에 비교할 수 없다.
