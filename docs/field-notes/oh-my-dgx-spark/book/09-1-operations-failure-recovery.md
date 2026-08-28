# 09-1. 발열·OOM·NCCL·펌웨어 장애

상태: 리서치 기반 초안

DGX Spark 장애를 모델 품질 문제로만 보면 복구가 늦어진다. 발열, unified memory, 통신, 컨테이너, 펌웨어를 서로 다른 층에서 관찰하고 한 번에 한 층만 변경한다.

## 3분 이해 (ELI5)

장애 대응은 자동차 계기판을 보고 경고등의 층위를 나누는 일이다.

```text
온도·전원 → OS·driver → 통신 → 모델·runtime
```

한 번에 여러 부품을 바꾸지 않고, 먼저 관찰한 뒤 한 층씩 원인을 좁힌다.

## 11.1 장애를 네 층으로 나눈다

```text
하드웨어: 온도·전원·팬·NIC·케이블
OS/driver: kernel·NVIDIA driver·firmware·memory pressure
통신: NCCL·RDMA·UCX·socket fallback
모델/runtime: quant·KV·parser·CUDA graph·scheduler
```

한 층에서 나타난 증상이 다른 층의 원인처럼 보일 수 있다. 예를 들어 긴 생성 중 멈춤은 model bug일 수도 있고, NCCL all-reduce hang이나 memory pressure 때문일 수도 있다.

## 11.2 첫 5분 수집 명령

서버가 아직 응답한다면 먼저 다음 정보를 저장한다.

```bash
date -Is
nvidia-smi
free -h
df -h
ps -ef | grep -E 'vllm|sglang|llama|spark' | grep -v grep || true
docker ps
ss -ltnp
journalctl -k -n 200 --no-pager
dmesg -T | tail -200
```

컨테이너를 쓰면 해당 컨테이너의 마지막 로그도 저장한다.

```bash
docker logs --tail 500 <container>
```

강제 전원 차단 전에 로그를 남길 수 있다면 먼저 저장한다. 재부팅하면 장애 직전의 메모리, 온도, transport 정보가 사라질 수 있다.

## 11.3 증상별 진단표

| 증상 | 1차 확인 | 2차 확인 | 보류할 결론 |
|---|---|---|---|
| 서버가 시작되지 않음 | 로그·port·model id | config·revision·disk | 모델이 지원되지 않음 |
| CUDA OOM | free memory·KV·context | workspace·concurrency·graph | quant 자체가 불량 |
| OS freeze | unified memory·kernel log | EarlyOOM·다른 process·container cap | Spark 하드웨어 불량 |
| generation 반복/깨짐 | tokenizer·config·KV dtype | quant checkpoint·backend | 모델 품질이 낮음 |
| 다중 노드 hang | NCCL debug·rank·transport | RDMA·MTU·NCCL version | TP가 불가능함 |
| 속도가 급락 | thermal·socket fallback | cache·spec acceptance | 모델이 느림 |
| 링크가 낮은 속도 | `ip`·`ethtool`·RDMA | 전원·cable·firmware 상태 | 케이블이 확정적으로 불량 |
| 부팅 후 NIC 이상 | firmware log·device state | 공식 recovery/RMA 절차 | 임의 flash로 해결 |

## 11.4 발열과 전원

지속적인 prefill과 decode는 짧은 smoke test보다 높은 열과 전력 상태를 만든다. 다음 항목을 요청별로 기록한다.

```text
temperature_idle:
temperature_after_warmup:
temperature_peak:
power_idle:
power_peak:
fan_or_cooling_setup:
shutdown_or_throttle:
```

포럼에는 sustained inference 중 95°C 부근에서 thermal shutdown이 발생한 사례와 냉각 duct·cage 사용기가 보고되어 있다. 장비 배치에 따른 차이도 있으므로 한 사용자의 DIY cooling을 모든 Spark에 적용되는 공식 해법으로 보지 않는다. [thermal shutdown 사례](https://forums.developer.nvidia.com/t/dgxspark-temperature-too-high-automatic-shutdown/363370), [dual Spark cooling cage](https://forums.developer.nvidia.com/t/dual-spark-ducted-cooling-cage/365302)

온도가 오르면 먼저 workload를 중지한 뒤 airflow, 주변 온도, 먼지, 장비 간격을 확인한다. thermal protection을 우회하거나 threshold를 임의로 높이지 않는다.

### 11.4a 저클럭·저전력 상태는 정상처럼 보일 수 있다

DGX Spark가 갑자기 느려졌을 때 GPU utilization과 P-state만 확인하면 문제를 놓칠 수 있다. 최근 커뮤니티 보고에서는 GPU utilization 약 96%, P0, 활성 throttle reason 없음, SM clock 약 799MHz, 부하 전력 약 19.5W인 상태에서 Ornith 1.5 35B decode가 약 44 tok/s에 머물렀다. 전원을 완전히 차단한 뒤에는 SM clock 약 2.3~2.5GHz, 부하 전력 약 92W, BF16 compute 36.5→91.6 TFLOP/s, decode 42.7→73.9 tok/s로 회복되었다. 이 수치는 [Blackwellboy의 X 보고](https://x.com/Blackwellboy/status/2090611479653622261?s=20)에 따른 community-reported 값이다.

NVIDIA Developer Forum에도 721MHz 또는 550MHz로 고정된 clock이 전원을 완전히 차단한 뒤 회복되었다는 사례가 반복해서 올라왔다. 일부 사용자는 AC와 power brick을 모두 분리한 뒤 약 5분을 기다리는 방법을 보고했고, 다른 사용자는 전원 공급 장치의 power-control 상태를 원인으로 추정했다. [721MHz 고정 사례](https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-clock-pinned-at-721-mhz-under-full-load-no-throttling-not-liftable-via-nvidia-smi/376039), [5분 전원 차단 사례](https://forums.developer.nvidia.com/t/gpu-clock-bug-looks-like-5-min-wait-is-enough/376239)

부하 중에는 다음 정보를 같은 시점에 수집한다.

```bash
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,power.draw,utilization.gpu --format=csv -l 1
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE
```

정상적인 workload에서 clock이 비정상적으로 낮고 power draw도 낮다면 다음 순서로 복구를 시도한다.

1. 새 요청과 inference process를 중지하고 로그를 저장한다.
2. 정상적인 OS 종료를 수행한다.
3. 전원 brick을 Spark와 AC 콘센트에서 모두 분리한다.
4. 커뮤니티 사례처럼 몇 분 동안 기다린 뒤 다시 연결하고 부팅한다.
5. 동일한 prompt·output·runtime에서 clock, power, compute, tok/s를 재측정한다.

이 절차는 커뮤니티 workaround이며, 공식 root cause가 확인되었거나 영구적인 해결책이라는 뜻은 아니다. 같은 현상이 반복되면 DGX OS, driver, EC, USB PD 버전, 사용한 power adapter, 직전 OOM·hard crash 여부를 함께 기록하고 NVIDIA 지원 절차를 따른다. `P0`와 높은 utilization만으로 healthy 판정을 내리지 않는다는 규칙을 preflight에 추가한다.

### 11.4b 의도적인 clock cap: 전력·발열과 prefill의 교환

[`agjs/gb10-clock-cap`](https://github.com/agjs/gb10-clock-cap)은 OpenAI-compatible endpoint를 발견하고 preflight·decode·cold prefill·soak·clock sweep을 실행한 뒤 `results/summary.json`에 판정을 남기는 공개 harness다. 작성자의 2× GB10 reference system(DeepSeek V4 Flash DSpark, TP=2, 1M context, 8분 sustained load)에서는 `2200MHz` cap이 stock 대비 peak temperature 90→78°C, 노드당 GPU rail power 63.1→40.1W, decode 73.3→72.5 tok/s로 보고됐다. 같은 저장소의 sweep에서는 2000MHz에서 decode 71.4 tok/s·cold prefill +8.1%, 1800MHz에서 decode 70.5 tok/s·cold prefill +13.0%가 기록됐다. 이는 해당 reference system의 결과이지 모든 Spark의 보장값은 아니다.

사용자가 추가한 [2× DGX Spark c4 X 보고](https://x.com/ivanfioravanti/status/2088730630875930639?s=20)는 MiaAI-Lab DeepSeek V4 Flash 0731 NVFP4 recipe에서 `2455/2300/2200/2000/1800MHz`를 비교해 각각 약 `47/34/32/27/23`의 `nvtop` 표시 전력과 `72/67/66/63/61°C`를 제시한다. 게시자도 벽면 전력은 이보다 높다고 명시했다. 따라서 GPU rail·`nvtop` 값과 AC wall-meter 값을 같은 전력 열에 넣지 않는다.

보고된 설정·복구 명령은 다음과 같다. 2대 구성에서는 각 노드에 같은 설정을 적용하고, 적용 전후의 host·clock·power domain을 기록한다.

```bash
sudo nvidia-smi -lgc 0,2200   # cap 적용, 보고된 예시
sudo nvidia-smi -rgc          # cap 해제
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1
```

decode가 memory-bandwidth bound인지 prefill이 compute bound인지에 따라 손실이 달라진다. 그러므로 cap을 적용했을 때는 c1만 보지 말고 c4, decode, cold prefill, peak temperature, GPU rail power, 벽면 AC power를 stock/capped interleaved A/B로 따로 측정한다. 이 자료는 전력·열 최적화의 실험 후보이지, thermal protection을 우회하거나 클럭을 무조건 낮추라는 운영 지침이 아니다.

[GB10 clock cap 원자료](https://nacyot.github.io/artifacts/gb10-clock-cap/)의 단일 ASUS GX10·`llama-server`·DeepSeek UD-Q2_K_XL 측정에서는 2000MHz cap이 16.6 tok/s·21.9W, cap 해제가 17.5 tok/s·42.6W였다. MiniMax-H3에서는 같은 cap이 작업 시간을 더 크게 늘렸다. 이 수치는 드라이버 580.173.02와 단일 노드 조건이므로 2대 FP8 vLLM 결과나 595.84 세팅 글과 같은 표에 합치지 않는다.

### 11.4c vLLM spin-wait와 GB10 SoC 발열

[vLLM spin-wait 후속 재현](https://nacyot.github.io/artifacts/vllm-spin-wait-gb10-repro/)은 2× ASUS GX10에서 DeepSeek V4 Flash 0731 원본 FP8·TP=2·`mp` executor를 사용하고 `busy_loop_s`만 1초에서 2ms로 바꾼 실험이다. GPU cap과 workload는 두 상에서 같게 두었다.

작성자 측정에서 동시 4개·30분 부하의 head 노드는 다음과 같았다.

| 항목 | stock 1초 spin | 2ms patch |
|---|---:|---:|
| vLLM CPU 합 | 396~400% | 211~214% |
| TSOC 평균 / 최고 | 89.6 / 95°C | 80.5 / 85°C |
| 동시 4개 aggregate | 85.8 tok/s | 92.1 tok/s |
| 부하 중 단독 probe | 18.1 tok/s | 17.8 tok/s |

이 결과는 멀티스트림에서 CPU 열을 줄일 수 있다는 근거이지, 모든 vLLM 버전에서 2ms가 더 빠르다는 증거가 아니다. 작성자도 단일 스트림에서 2~6% 손실을 별도 관찰했다. 또한 stock·patched 각 한 상의 결과라 aggregate 차이는 재기동 분산 안에 있다고 기록한다.

따라서 TP=1 단일 Spark에 이 패치를 기본 적용하지 않는다. 다중 노드 `mp` executor에서 CPU spin이 실제로 관찰될 때만 별도 branch/container로 재현하고, 원본 패키지 파일을 수정하기 전 vLLM 버전·이미지 digest·rollback 방법을 기록한다. 온도는 `nvidia-smi`만 보지 말고 성능코어/SoC thermal zone을 함께 수집한다.

## 11.5 unified-memory OOM

OOM은 다음 순서로 원인을 좁히며 줄인다.

1. 다른 inference process를 중지하거나 분리한다.
2. `max_model_len`과 `max_num_seqs`를 낮춘다.
3. prefix/cache·vision·tool loop의 context 증가를 확인한다.
4. CUDA graph/workspace와 container memory cap을 확인한다.
5. quant·KV dtype을 한 번에 하나씩 바꾼다.
6. 장시간 soak에서 memory가 회수되는지 확인한다.

메모리가 가득 찬 상태에서 OS 전체가 crash했다는 사례도 있으므로, `GPU_MEMORY_UTILIZATION`만 높여 문제를 해결하려 하지 않는다. [system memory full 사례](https://forums.developer.nvidia.com/t/system-crashes-when-memory-is-full/352339)

DeepSeek one-Spark recipe의 `GPU_MEMORY_UTILIZATION=0.94`와 EarlyOOM 비활성화는 메모리를 적극적으로 점유하는 조건이다. 일반 모델에 복사하기 전 free memory와 recovery plan을 확인한다.

## 11.6 NCCL·RDMA hang

통신 장애를 진단할 때는 다음 상태를 구분한다.

```text
link up
  ≠ IP 연결
  ≠ RDMA device 사용
  ≠ NCCL communicator 완성
  ≠ all-reduce 통과
  ≠ 긴 generation 안정
```

NCCL 로그를 자세히 남길 때는 recipe에 맞게 debug level과 subsystem을 설정한다. 일반적으로 다음 항목을 수집한다.

```text
NCCL_DEBUG:
NCCL_DEBUG_SUBSYS:
node/rank:
interface/HCA:
transport: IB/RDMA or Socket
NCCL version:
```

포럼에는 link up, channel establishment, weight load까지 성공한 뒤 첫 all-reduce에서 vLLM과 TensorRT-LLM이 함께 멈춘 사례가 있다. “모델이 로드됐다”는 사실을 “분산 추론이 정상이다”라는 의미로 기록하지 않는다. [NCCL all-reduce deadlock 사례](https://forums.developer.nvidia.com/t/nccl-all-reduce-deadlock-on-dual-dgx-spark-after-successful-channel-establishment-affects-both-vllm-and-trt-llm/366127)

## 11.7 socket fallback

RDMA를 의도했는데 socket으로 떨어지면 속도, 지연, CPU 사용량이 달라진다. 컨테이너에서는 `/dev/infiniband`와 필요한 device/capability가 전달되었는지 확인한다.

```text
expected: NCCL transport = NET/IB
actual:   NCCL transport = NET/Socket
```

이 두 결과는 같은 benchmark 행에 기록하지 않는다. SGLang GLM 사례에서는 RDMA passthrough 후 9.8 → 25.1 tok/s로 변한 결과가 보고되었다. [원문](https://forums.developer.nvidia.com/t/glm-4-7-fp8-on-4x-dgx-spark-via-sglang-2-5x-speedup-8-2-25-tok-s-just-by-enabling-rdma/373675)

## 11.8 ConnectX·펌웨어

ConnectX-7 링크가 200G로 negotiate되었더라도 실제 payload가 낮게 나오는 사례가 있다. 일부 글에서는 전원을 완전히 방전한 뒤 링크가 회복되었다고 기록하지만, 이는 현장 workaround일 뿐 일반 복구 절차는 아니다.

펌웨어를 다룰 때는 다음 원칙을 지킨다.

- 사용자가 의도하지 않은 updater를 자동 실행하지 않는다.
- 현재 firmware·driver·device 상태를 먼저 저장한다.
- 공식 release note와 recovery/RMA 절차를 확인한다.
- 한 노드에서 검증한 뒤 전체 클러스터에 적용한다.
- firmware 변경과 모델 benchmark를 같은 날 한 결과로 섞지 않는다.

포럼에는 `mlnx-fw-updater`가 ConnectX-7을 건드린 뒤 pre-init에서 멈추고 RMA로 이어진 사례도 있다. [firmware brick 사례](https://forums.developer.nvidia.com/t/connectx-7-bricked-stuck-in-pre-init-static-config-not-done-after-unsolicited-mlnx-fw-updater-firmware-flash-asus-gx10-error-110/373900)

## 11.9 안전한 복구 순서

```text
1. 신규 요청 차단
2. endpoint health와 client retry 중지
3. 로그·nvidia-smi·free -h·온도 저장
4. 컨테이너/서버 정상 종료
5. 단일 노드·작은 모델·짧은 context로 재현
6. transport·driver·firmware·quant 중 한 변수만 변경
7. smoke test → short soak → long soak
8. 원인과 workaround를 분리해 기록
```

강제 종료나 전원 차단이 필요할 만큼 OS가 멈췄다면 복구 후 파일시스템, journal, NIC, driver 상태를 확인한다. 같은 workload를 즉시 반복해서 실행하지 않는다.

## 11.10 장애 보고서 템플릿

```text
incident_id:
time_start:
time_end:
node(s):
workload:
model/revision:
engine/image/commit:
quant/kv/speculation:
context/concurrency:
temperature_peak:
memory_peak:
network_transport:
last_successful_request:
symptom:
logs:
reproduction_steps:
workaround:
root_cause_status: confirmed / suspected / unresolved
follow_up:
```

## 이 장의 검증 체크리스트

- [ ] 장애 직전 로그·memory·temperature·transport를 보존했다.
- [ ] thermal·OOM·NCCL·firmware를 서로 다른 층으로 분리했다.
- [ ] socket fallback을 RDMA 결과와 섞지 않았다.
- [ ] model load와 all-reduce/generation 안정성을 구분했다.
- [ ] memory full 상태에서 무리하게 utilization을 올리지 않았다.
- [ ] 펌웨어 updater·flash는 공식 절차와 rollback/recovery를 확인했다.
- [ ] 한 번에 한 변수만 바꾸어 재현했다.
- [ ] workaround와 confirmed root cause를 구분했다.

## 아직 모르는 것

- driver·firmware 조합별 thermal/OOM 재현 범위
- 장시간 agent loop와 unified-memory pressure의 상관관계
- 3~8대 클러스터에서 장애 노드 자동 격리·재구성 수준
- Spark OS 업데이트 후 NCCL·RDMA regression을 자동 감지하는 방법
