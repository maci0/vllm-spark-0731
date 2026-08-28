# ASUS Ascent GX10 노드 세팅 자료 검토

기준일: **2026-08-22**  
자료: [ASUS Ascent GX10(DGX Spark) 한 대를 추론 서버로 세팅하기](https://nacyot.github.io/artifacts/dgx-spark-node-setup/)

## 자료 성격

이 자료는 ASUS Ascent GX10(GB10) 세 대를 직접 세팅한 작성자의 현장 보고서다. 서버 모드 전환, NVIDIA 플랫폼 패키지 보호, 드라이버·USB-C PD 펌웨어, GPU·CPU 클럭 상한, vLLM 멀티프로세스 스핀, ConnectX-7 링크, Ollama Qwen3.8, DeepSeek V4 Flash EXL3까지 한 문서에 묶었다.

책에서는 이 자료를 `community field report`와 `author-measured`로 분류한다. ASUS GX10의 펌웨어·기본 이미지·패키지 구성이 NVIDIA DGX Spark Founders Edition이나 다른 GB10 OEM과 같다고 가정하지 않는다.

## 책에 반영할 핵심

| 주제 | 원문에서 보고한 내용 | 책에서의 처리 |
|---|---|---|
| 서버 모드 | graphical target에서 `multi-user`로 전환하고 데스크톱 패키지를 줄이면 모델 기동 여유가 늘었다고 보고 | 첫 부팅 체크리스트와 메모리 headroom 원칙에 반영. purge 명령은 장비별 시뮬레이션 없이는 복사하지 않음 |
| 플랫폼 패키지 | `apt autoremove --purge`가 NVIDIA/DGX 플랫폼 패키지까지 제거한 사례 | `apt-mark manual`과 `apt-get -s` 게이트를 안전 절차로 기록. `dgx-spark-ota-update-meta`의 역할은 제품 이미지별 확인 대상으로 표시 |
| 메모리 | 128GiB 제품에서 `MemTotal` 약 121.6GiB, 서버 모드 유휴 `MemAvailable` 약 119GiB | 모델 적재 가능 여부를 제품 표기 용량이 아니라 `MemAvailable`·KV·workspace·OS 여유로 판단 |
| 드라이버·PD | ASUS 장비에서 580.173.02→595.84, USB-C PD 0x1→0x516을 적용했다고 보고 | 특정 장비의 스냅샷. 모든 Spark에 595/0x516을 정답으로 쓰지 않고 현재 장비의 `nvidia-smi`, `fwupdmgr`, 공식 릴리스 노트를 먼저 확인 |
| 저클럭 | 2000MHz cap에서 LLM decode 전력·온도가 낮아지고 속도 손실이 작았다는 측정 | 의도적 cap과 고장성 저클럭을 분리하고, SM clock·power·temperature·실제 tok/s를 함께 기록 |
| vLLM 스핀 | `mp` 경로의 `busy_loop_s`를 2ms로 바꿔 CPU 사용량·SoC 온도가 줄었다고 보고 | 설치 패키지 직접 수정은 임시 실험으로만 기록. 버전별 소스 위치와 단일 스트림 성능 손실을 확인한 뒤 적용 |
| ConnectX-7 | 물리 포트 하나가 Linux 인터페이스 여러 개로 보이며, 두 rail에 별도 IP를 주는 구성을 사용 | 인터페이스 이름·MAC 고정 여부를 먼저 확인하고, bond와 NCCL rail 구성을 혼동하지 않음 |
| 링크 성능 | 링크가 200G로 협상돼도 `ib_write_bw`가 약 13Gbps에 고착된 사례가 전원 완전 차단 뒤 회복 | `link up`과 payload·NCCL 성능을 분리. 전원 방전은 공식 보편 해법이 아닌 현장 workaround로 표시 |
| 단일 레시피 | GPU cap 조건에서 Ollama Qwen3.8 Q4_K_M 11.5~11.8 tok/s, DeepSeek EXL3 단일 스트림 36~37 tok/s 등을 보고 | 모델·양자화·엔진·clock 조건이 고정된 community measurement로만 인용. 우리 실측 결과와 하나의 순위표로 합치지 않음 |

## 1. 서버 모드 전환에서 재사용할 원칙

원문에서 가장 재사용 가치가 높은 부분은 특정 `apt purge` 명령보다 **삭제 전에 보호 대상을 식별하고 시뮬레이션하는 절차**다.

```text
1. OS·kernel·driver·CUDA·Docker·firmware·패키지 목록 저장
2. NVIDIA/DGX/CUDA/Docker 관련 패키지를 보호 대상으로 표시
3. purge와 autoremove를 실제 실행하기 전에 -s로 시뮬레이션
4. 보호 대상이 제거 목록에 나오면 중단
5. 재부팅 뒤 /proc/cmdline, nvidia-smi, Docker, failed units를 다시 대조
```

`nvidia-*`, `dgx-*`, `cuda-*`라는 이름만으로 모든 장비의 보호 목록을 완성할 수는 없다. 제조사 이미지와 DGX OS 릴리스에 따라 메타패키지와 부팅 스니펫이 달라질 수 있기 때문이다. 따라서 책의 기본 레시피는 데스크톱을 지우는 것보다 먼저 서버 모드 전환 여부와 headroom을 측정하도록 한다.

## 2. 메모리 숫자를 읽는 법

원문의 121.6GiB는 고장이나 “128GB가 사라진” 상태를 뜻하지 않는다. unified memory에서는 다음을 따로 기록해야 한다.

```text
MemTotal
  - kernel·firmware·display reserved
  - OS·driver·서비스
  - 모델 weight
  - KV cache·CUDA graph·workspace
  - 입력·sidecar·통신 버퍼
  = 운영 가능한 여유
```

단일 모델이 적재되었다는 사실만으로 긴 context와 동시 요청이 가능한 것은 아니다. 특히 DeepSeek EXL3처럼 메모리를 적극적으로 점유하는 레시피는 기동 전 `MemAvailable`, 모델 적재 후 여유, 첫 요청 peak, 장시간 soak 결과를 각각 남겨야 한다.

## 3. 클럭과 전력 측정의 분리

원문은 2000MHz 부근에서 decode 속도 손실보다 전력·온도 감소가 큰 측정과, 2400MHz 이상에서 compute-bound 영상 작업의 이득이 제한되는 측정을 제시한다. 이 결과를 모든 엔진에 적용할 수는 없다.

책의 측정 행에는 다음을 함께 넣는다.

```text
clock cap command:
actual SM clock:
GPU rail power:
wall AC power:
temperature peak:
prefill tok/s:
decode tok/s:
concurrency / context:
```

`nvidia-smi -lgc`로 의도적으로 설정한 cap과, 부하 중 utilization·P-state는 높지만 SM clock과 power가 비정상적으로 낮은 장애를 같은 현상으로 기록하지 않는다. 후자의 경우 먼저 로그를 보존하고, 커뮤니티 사례의 전원 분리·방전은 임시 복구 방법으로만 취급한다.

## 4. ConnectX-7과 두 rail

원문은 socket-direct 구조에서 하나의 물리 QSFP 포트가 여러 Linux 인터페이스로 노출될 수 있다고 설명한다. 이때 인터페이스 이름만 보고 bond를 만들지 말고 다음을 확인한다.

```bash
ip -br link
cat /sys/class/net/*/phys_switch_id 2>/dev/null
cat /sys/class/net/*/phys_port_name 2>/dev/null
ibdev2netdev
```

두 인터페이스에 별도 서브넷을 줄지는 NVIDIA 플레이북·현재 NCCL recipe와 함께 결정한다. 실제 inference 전에 다음 계층을 분리해 통과시킨다.

```text
link speed 200G
  → IP connectivity
  → RDMA/ib_write_bw
  → NCCL collective
  → model load
  → short generation
  → long generation / soak
```

200G 협상, `ping`, `ib_write_bw`, NCCL이 모두 같은 것을 의미하지 않는다. 한 단계의 workaround로 다음 단계가 검증되었다고 기록하지 않는다.

## 5. 레시피 수치의 provenance

원문에서 보고한 단일 Spark 수치는 다음 조건을 포함한다.

- ASUS Ascent GX10, GB10, 128GB unified memory
- GPU clock cap 약 1989MHz 상태
- Ollama Qwen3.8-27B GGUF Q4_K_M: 약 11.5~11.8 tok/s decode
- DeepSeek V4 Flash: EXL3 3.0bpw, REAP K216, SparkInfer·DSpark 경로
- DeepSeek 단일 스트림: thinking off 기준 약 36~37 tok/s, 긴 context·구조화 출력은 별도 조건
- 330K needle 테스트와 recipe acceptance test는 작성자 환경의 결과

따라서 이 수치는 `community-reported`, 일부는 `author-measured`로 남긴다. Qwen3.8 BF16 vLLM, Qwen3.8 NVFP4 SGLang, DeepSeek FP8 TP=2, 우리 로컬 harness 결과와 직접 비교하지 않는다.

## 6. 관련 문서 비교

네 링크는 서로 다른 실험을 하나의 결과로 합칠 수 있는 자료가 아니다. 공통 장비 계열은 GB10/ASUS GX10이지만 모델 파일, 엔진, 드라이버, 노드 수, clock cap, workload가 다르다.

### 6.1 영문판

[영문판](https://nacyot.github.io/artifacts/dgx-spark-node-setup-en/)은 앞서 검토한 한국어 노드 세팅 글의 번역판이다. 독립적인 장비·벤치마크 근거로 중복 집계하지 않고, 영어 독자를 위한 동일 원문 링크로만 보존한다.

### 6.2 vLLM spin-wait 재현

[vLLM spin-wait 후속 실험](https://nacyot.github.io/artifacts/vllm-spin-wait-gb10-repro/)은 다음 조건에서 `busy_loop_s` 하나만 바꾼 통제 실험이다.

- ASUS GX10 2대, ConnectX-7 200GbE direct
- `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, vLLM `0.25.2.dev0+g752a3a504`
- DeepSeek V4 Flash 0731 원본 FP8, TP=2, `mp`, MTP draft 5
- GPU cap은 두 상 모두 300~2000MHz로 동일
- `busy_loop_s=1` 대 `busy_loop_s=0.002`

30분·동시 4개 부하에서 head 노드 vLLM CPU 합은 약 396~400% 대 211~214%, TSOC 평균은 89.6°C 대 80.5°C, 최고 온도는 95°C 대 85°C로 보고됐다. 처리량은 85.8 대 92.1 tok/s였지만 각 상을 한 번씩 실행한 결과이며 재기동 분산 범위 안이라고 작성자가 명시한다. 부하 중 단독 probe는 18.1 대 17.8 tok/s였고, 별도 측정에서 단일 스트림은 패치 후 2~6% 느려질 수 있다고 기록한다.

따라서 이 자료의 결론은 “2ms 패치가 항상 더 빠르다”가 아니라 **멀티스트림·발열이 문제인 vLLM `mp` 조건에서 시도할 수 있는 trade-off**다. vLLM 설치 파일을 직접 수정하는 명령은 버전·이미지별로 달라질 수 있으므로 책의 기본 recipe로 복사하지 않는다.

### 6.3 원본 FP8 듀얼 Spark

[DeepSeek V4 Flash 0731 2대 세팅·실측](https://nacyot.github.io/artifacts/deepseek-v4-flash-2x-dgx-spark/)은 단일 Spark EXL3가 아니라 공식 `deepseek-ai/DeepSeek-V4-Flash-0731` 원본 FP8을 대상으로 한다.

- 모델 revision `9e165c30`, 166.9GB, TP=2, vLLM `mp`, `--nnodes 2`
- DSpark `k=5`, draft acceptance 83~87% 보고
- `nvfp4_ds_mla` KV cache, 1,048,576 설정, 512K까지 실측
- ConnectX-7 200GbE direct, RoCE v2 dual rail, 196Gb/s 실측 보고
- context sweep에서 단일 decode는 256토큰 약 56.1→512K 약 45.9 tok/s
- prompt 256·동시 12에서 aggregate 206.9 tok/s 보고
- 2000MHz cap에서 단일 decode 54.4 tok/s·16K prefill 1,794 tok/s, cap 해제에서 55.0 tok/s·1,953 tok/s

이 문서의 1M은 서버 설정·capacity 범위이고, 실제 장문 품질을 1M 전체에서 인증했다는 뜻이 아니다. 또한 이 수치는 one-Spark EXL3, 2대 FP8 recipe, 다른 MiaAI·m9e recipe와 섞지 않는다.

### 6.4 GB10 clock-cap 원자료

[GB10 clock cap 측정](https://nacyot.github.io/artifacts/gb10-clock-cap/)은 단일 ASUS GX10에서 1Hz GPU 센서 기록으로 `llama-server`의 DeepSeek UD-Q2_K_XL과 MiniMax-H3를 비교한 자료다. 이 원자료의 DeepSeek 환경은 드라이버 580.173.02와 단일 노드 llama.cpp 계열이며, 앞의 2대 FP8 vLLM 결과와 동일한 benchmark가 아니다.

대표 측정은 다음과 같다.

| workload | cap 해제 | 2000MHz cap | 해석 |
|---|---:|---:|---|
| DeepSeek decode | 17.5 tok/s, 42.6W | 16.6 tok/s, 21.9W | 약 4.9% 속도 감소, 약 49% GPU 전력 감소 |
| H3 Turbo 8 | 97.7s, 82.6W, 80.8°C | 114.4s, 44.0W, 59.5°C | compute-bound 작업은 시간 손실이 큼 |
| H3 20 step | 81.9s, 87.0W, 84.5°C | 89.7s, 51.2W, 68.1°C | workload별 최적 cap이 다름 |

DeepSeek sweep에서는 2000→2400MHz에서 decode가 16.67→17.37 tok/s로 조금만 오르고, 전력은 22.3→34.0W로 증가했다. 2600MHz 이상은 실제 SM clock이 약 2495MHz에 머물러 추가 이득이 거의 없었다. 모든 전력은 GPU sensor 값이며 벽면 AC 전력이 아니다.

## 7. 독립 검증 큐

- [ ] NVIDIA DGX Spark와 ASUS GX10에서 `MemAvailable`·서버 모드 차이를 같은 모델로 비교
- [ ] 현재 DGX OS/driver에서 PD firmware 버전과 저클럭 증상 상관관계 확인
- [ ] 200G direct 두 rail의 `ib_write_bw`·NCCL·실제 TP decode를 같은 시각에 기록
- [ ] vLLM 버전별 `busy_loop_s` 패치 전후 CPU·SoC·single-stream trade-off 재현
- [ ] GPU rail power와 벽면 AC power를 같은 clock sweep에서 동시에 측정
- [ ] Ollama Q4_K_M과 vLLM/SGLang Qwen3.8 결과를 동일 prompt·출력 길이로 비교
- [ ] vLLM spin 재현을 현재 이미지·vLLM 버전에서 `busy_loop_s` 단일 변수로 다시 측정
- [ ] 원본 FP8 2대 recipe의 512K capacity, 1M 설정, C1/C6/C12 aggregate를 같은 harness로 재현
- [ ] clock cap 원자료의 GPU rail 값과 별도 벽면 AC meter 값을 같은 sweep에서 비교

## 참고

- [원문: ASUS Ascent GX10 노드 세팅](https://nacyot.github.io/artifacts/dgx-spark-node-setup/)
- [NVIDIA DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/index.html)
- [NVIDIA DGX Spark Release Notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
- [NVIDIA Connect Two Sparks 플레이북](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks)
- [NVIDIA ConnectX-7 Networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
- [LVFS ASUS GX10 USB-C PD firmware](https://fwupd.org/lvfs/devices/com.asus.gx10dgx.usbpd.firmware)
- [영문판 노드 세팅](https://nacyot.github.io/artifacts/dgx-spark-node-setup-en/)
- [vLLM spin-wait 후속 재현](https://nacyot.github.io/artifacts/vllm-spin-wait-gb10-repro/)
- [DeepSeek V4 Flash 2대 원본 FP8 실측](https://nacyot.github.io/artifacts/deepseek-v4-flash-2x-dgx-spark/)
- [GB10 clock cap 실측](https://nacyot.github.io/artifacts/gb10-clock-cap/)

이 문서는 원문의 명령과 수치를 그대로 실행하라는 매뉴얼이 아니다. 모델·펌웨어·제품별 차이를 확인하고, 위험한 시스템 변경은 공식 복구 경로와 rollback을 먼저 확보한다.
