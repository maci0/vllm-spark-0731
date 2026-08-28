# 03-1. 첫 부팅과 안전한 기본 환경

상태: 초안

이 장은 첫 모델보다 먼저 읽는다. 첫날의 목표는 최고 속도가 아니라, 전원·OS·driver·Docker·디스크·네트워크를 확인하고 문제가 생겼을 때 되돌아갈 수 있는 기준점을 만드는 것이다.

## 3분 이해 (ELI5)

첫 부팅은 레이싱이 아니라 출발선 점검이다.

```text
전원·OS → GPU 인식 → runtime → 모델 적재 → 서버 → 에이전트
```

앞 단계가 통과되지 않으면 뒤 단계의 실패를 모델 문제로 단정하지 않는다.

## 3.1 첫날에 먼저 답할 질문

| 질문 | 통과 기준 | 참고 |
|---|---|---|
| 어떤 방식으로 처음 설정할 것인가? | display 로컬 설정 또는 network appliance 절차를 선택 | [NVIDIA First Boot](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html) |
| 전원과 네트워크가 충분한가? | 제공된 240W adapter와 안정적인 인터넷 연결 | 전원·업데이트 실패를 모델 문제로 보지 않음 |
| 업데이트가 끝났는가? | 자동 reboot와 초기 이미지 설치가 완료됨 | 초기 설정 중에는 최대 10분 이상 기다릴 수 있음 |
| 메모리와 디스크 기준점이 있는가? | 모델 다운로드 전 `free -h`, `df -h`, `nvidia-smi` 기록 | UMA 메모리 숫자를 전용 VRAM처럼 읽지 않음 |
| 실행 권한과 비밀값이 준비됐는가? | Docker 권한, Hugging Face token, 저장 위치를 직접 확인 | token을 로그와 저장소에 남기지 않음 |

NVIDIA 공식 문서는 안정적인 인터넷 연결과 제공된 전원 어댑터를 초기 조건으로 제시한다. 이 책에서는 첫 모델 다운로드 전에 그 조건을 확인하고, 그 다음에만 Docker와 runtime 문제를 진단한다. [DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

## 3.2 첫 원칙: 환경을 먼저 고정한다

DGX Spark에서 “모델이 안 된다”는 말은 실제로 다음 상태 중 하나를 뜻할 수 있다.

| 상태 | 실제로 확인한 것 |
|---|---|
| 부팅 | OS와 NVIDIA 드라이버가 정상 시작됨 |
| 장치 확인 | `nvidia-smi`와 메모리 정보가 정상임 |
| 런타임 확인 | Docker·PyTorch·vLLM/SGLang이 GPU를 볼 수 있음 |
| 모델 적재 | weight가 메모리에 올라감 |
| 생성 | 한 번의 요청에 정상 응답함 |
| 서빙 | endpoint가 반복 요청과 동시 요청을 처리함 |
| 벤치마크 | 조건과 반복 수를 고정한 숫자가 있음 |
| 에이전트 | tool loop와 실패 복구까지 통과함 |

이 상태를 모두 묶어 `성공`이라고 기록하면 어느 단계에서 문제가 생겼는지 찾기 어렵다. 첫 부팅 기록에는 각 상태를 별도 열로 남긴다.

## 3.3 하드웨어·소프트웨어 인벤토리

첫 모델을 내려받기 전에 다음 명령의 출력을 파일이나 작업 노트에 저장한다.

```bash
date -Is
hostnamectl
uname -a
nvidia-smi
free -h
df -h
docker version
docker info
```

클러스터를 구성할 계획이면 네트워크도 함께 기록한다.

```bash
ip -br addr
ip -br link
ls -l /dev/infiniband 2>/dev/null || true
ibstat 2>/dev/null || true
```

최소 기록 항목은 다음과 같다.

```text
date:
machine:
dgx_os:
kernel:
driver:
cuda:
container_runtime:
docker_image_and_digest:
model_revision:
free_memory_before_server:
free_disk_before_download:
temperature_idle:
network_and_switch:
```

`latest` 태그만 기록하면 나중에 같은 실험을 재현할 수 없다. 가능한 범위에서 이미지 digest, 모델 commit, tokenizer revision, 엔진 commit을 함께 고정한다.

## 3.4 unified memory를 안전하게 읽는 법

DGX Spark의 128 GiB unified memory는 모델 weight만 사용하는 전용 VRAM이 아니다. 실제 메모리 사용량에는 다음 항목이 동시에 포함된다.

```text
전체 unified memory
  - OS와 백그라운드 프로세스
  - 모델 weight
  - KV cache
  - CUDA graph와 workspace
  - tokenizer·prefill·멀티모달 입력 버퍼
  - 컨테이너와 통신 버퍼
  = 실제로 남는 headroom
```

그러므로 “weight가 128 GiB보다 작다”는 사실만으로 실행 가능하다고 결론 내리지 않는다. 특히 긴 context에서는 weight보다 KV와 workspace가 먼저 한계를 만들 수 있다.

첫 실행 전과 후에 다음을 비교한다.

```bash
free -h
nvidia-smi
ps -eo pid,cmd,%mem,%cpu --sort=-%mem | head -20
```

메모리가 계속 줄거나 시스템 전체가 멈춘다면 더 높은 `gpu-memory-utilization`을 시도하지 말고 먼저 서버를 중지한다. 그 다음 로그·프로세스·컨테이너 제한을 확인한다. unified memory가 완전히 차면 요청 하나가 실패하는 데 그치지 않고 OS가 응답하지 않을 수 있다.

단일 DeepSeek EXL3 최신 recipe의 `GPU_MEMORY_UTILIZATION=0.94`와 EarlyOOM 비활성화는 해당 서버가 메모리를 적극적으로 점유하도록 만든 실험 조건이다. 이를 다른 모델이나 일반 데스크톱 설정의 기본값으로 복사하지 않는다. [DeepSeek one-Spark README](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)의 조건을 적용할 때도 먼저 free memory와 복구 방법을 확인한다.

## 3.5 디스크와 첫 다운로드

모델 파일에 표시된 용량 외에도 다음 공간이 필요할 수 있다.

- 다운로드 중인 임시 파일
- 압축 해제·병합·coalesce 결과
- CUDA graph와 엔진 캐시
- Docker image layer
- tokenizer와 테스트 입력
- 재시작을 위한 원본 파일

DeepSeek one-Spark EXL3 recipe는 README 기준 약 107GB weight 다운로드와 첫 부팅 coalesce·K64 draft 빌드·CUDA graph capture가 필요하다. 이 숫자는 해당 recipe의 요구량이며 모든 모델의 공통 기준은 아니다.

```bash
df -h .
du -sh "$PWD" 2>/dev/null || true
docker system df
```

디스크가 부족한 상태에서 모델 다운로드를 반복하지 않는다. 먼저 어느 디렉터리가 공간을 차지하는지 확인하고, 삭제할 대상을 정확히 확인한 경우에만 Docker cache나 임시 파일을 정리한다.

## 3.6 포트와 기존 서버 확인

같은 Spark에서 기존 Qwen3-VL 서버가 8082, Qwen3.8 smoke-test 서버가 8083을 사용했던 것처럼 포트와 GPU 사용 주체를 명시적으로 나눈다.

```bash
ss -ltnp | grep -E ':808[0-9]|:8888' || true
ps -ef | grep -E 'vllm|sglang|llama|spark' | grep -v grep || true
```

새 서버를 시작하기 전에 다음을 기록한다.

| 항목 | 기록할 내용 |
|---|---|
| endpoint | host와 port |
| served model | API에 노출되는 정확한 model id |
| GPU 사용자 | 다른 inference process·컨테이너 |
| memory cap | 엔진 설정과 Docker 제한 |
| 로그 위치 | stdout, compose log, 파일 log |

공유 GPU에서 측정한 속도는 새 모델만의 결과가 아니다. baseline을 측정할 때는 가능하면 다른 추론 프로세스를 중지한다. 중지하지 못했다면 공유 상태를 결과에 명시한다.

## 3.7 업데이트와 실험 순서

운영 중인 서버에서 드라이버·CUDA·컨테이너·모델을 한꺼번에 업데이트하지 않는다. 다음 순서에 따라 한 번에 한 변수만 바꾼다.

1. 현재 버전과 smoke-test 결과를 저장한다.
2. 모델 revision만 바꾸고 모델 적재·생성·JSON을 확인한다.
3. 엔진 또는 컨테이너를 바꾸고 같은 prompt set을 반복한다.
4. quant 또는 KV dtype을 바꾸고 메모리·품질·속도를 다시 측정한다.
5. speculative decoding을 켜고 acceptance와 실패율을 기록한다.
6. 마지막에 context와 concurrency를 늘린다.

실험 중에는 `apt upgrade`, 드라이버 교체, 펌웨어 플래시를 동시에 진행하지 않는다. ConnectX-7 firmware와 NCCL/RDMA 문제는 모델 문제와 분리해서 확인해야 한다.

## 3.8 첫 부팅 후 최소 smoke test

모델을 올린 직후 장시간 benchmark를 실행하지 말고, 다음 순서로 상태를 확인한다.

```bash
curl -sS http://127.0.0.1:PORT/v1/models
curl -sS http://127.0.0.1:PORT/health || true
```

그 다음에 다음 기능을 작은 출력 길이로 확인한다.

- 한국어 단문
- Python 코드 블록
- JSON parser로 읽을 수 있는 JSON
- 멀티턴 marker 회수
- thinking on/off
- OpenAI-compatible tool parser

현재 저장소의 Qwen3.8 테스트는 [tests/qwen38_smoke.py](https://github.com/recrack/oh-my-dgx-spark/blob/main/tests/qwen38_smoke.py)로 models endpoint, 한국어·코드·JSON·멀티턴·thinking·긴 문맥·4-way 요청을 확인한다. 이 스크립트는 특정 모델의 품질 벤치가 아니라 서버 기능 smoke test다.

## 3.9 문제가 생겼을 때의 순서

| 증상 | 먼저 확인할 것 | 바로 하지 않을 것 |
|---|---|---|
| endpoint가 안 뜸 | 포트·컨테이너 로그·served model | 모델을 다시 여러 번 다운로드 |
| CUDA OOM | 다른 프로세스·KV/context·workspace | 무조건 memory fraction 증가 |
| OS가 느려짐 | unified memory·EarlyOOM·백그라운드 작업 | 강제 전원 차단 반복 |
| 응답이 깨짐 | model revision·tokenizer·quant config·KV dtype | 품질 문제로 단정 |
| tool call 400 | parser·request format·served model id | 모델이 tool을 못한다고 결론 |
| 다중 노드 hang | NCCL log·RDMA 경로·driver·interface | 통신 환경을 바꾼 채 모델 비교 |

복구가 필요하면 먼저 서버를 정상 종료하고 로그와 `nvidia-smi`·`free -h` 결과를 남긴다. 재부팅은 마지막 단계로 두고, 재부팅 전후의 driver·firmware·link 상태를 비교한다.

## 3.10 ASUS GX10 현장 자료를 적용하는 범위

[ASUS Ascent GX10 노드 세팅 자료](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-node-setup-research-2026-08.md)는 서버 모드 전환, 플랫폼 패키지 보호, `MemAvailable` 측정, USB-C PD·ConnectX-7 문제를 실제 장비에서 다룬다. 다만 ASUS GX10의 공장 이미지와 NVIDIA DGX Spark의 이미지가 같다고 가정하지 않는다.

이 자료에서 일반화할 수 있는 것은 명령어 묶음보다 순서다.

1. OS·kernel·driver·firmware·Docker·패키지 상태를 저장한다.
2. 데스크톱이나 snap을 줄이기 전에 NVIDIA/DGX/CUDA 패키지를 보호 대상으로 표시한다.
3. `apt-get -s`로 제거 목록을 확인하고 보호 대상이 나오면 중단한다.
4. 재부팅 뒤 `/proc/cmdline`, `nvidia-smi`, Docker, failed units를 다시 비교한다.
5. 모델의 적재 여부는 `MemTotal`이 아니라 `MemAvailable`, KV, workspace와 함께 판단한다.

드라이버 595.84, USB-C PD 0x516, 특정 NetworkManager 인터페이스 이름, 2000MHz cap은 해당 보고서의 장비 조건이다. 현재 장비에서는 `nvidia-smi`, `fwupdmgr`, `ip -br link`, 공식 릴리스 노트를 먼저 확인한 뒤 별도의 A/B 측정으로 채택한다. `apt autoremove --purge`, firmware flash, 전원 방전은 백업·복구 경로 없이 실행하지 않는다.

## 이 장의 검증 체크리스트

- [ ] OS·kernel·driver·CUDA·Docker·disk·memory를 기록했다.
- [ ] 다른 inference process와 포트 충돌을 확인했다.
- [ ] weight 외에 KV·workspace·graph·OS headroom을 고려했다.
- [ ] 모델 다운로드와 coalesce에 필요한 디스크를 확인했다.
- [ ] 모델 revision·tokenizer·container digest를 기록했다.
- [ ] 한국어·코드·JSON·멀티턴 smoke test를 통과시켰다.
- [ ] `GPU_MEMORY_UTILIZATION=0.94`와 EarlyOOM 비활성화를 일반 권장값으로 복사하지 않았다.
- [ ] 문제가 생겼을 때 로그·메모리·통신 상태를 먼저 보존한다.
- [ ] 제품별 현장 자료의 버전·펌웨어·패키지 값을 공통 기본값으로 복사하지 않았다.

## 아직 모르는 것

- DGX OS와 driver 업데이트별로 각 엔진의 안정성이 어떻게 달라지는가
- GPU memory fraction과 OS headroom의 모델별 최적점
- CUDA graph capture가 모델·context·concurrency별로 차지하는 실제 크기
- 장시간 agent loop에서의 누수와 재시작 복구 시간
