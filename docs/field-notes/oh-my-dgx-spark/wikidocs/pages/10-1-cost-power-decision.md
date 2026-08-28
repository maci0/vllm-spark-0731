# 10-1. 비용·전력·구성 의사결정

상태: 리서치 기반 초안

DGX Spark를 몇 대 살지 결정할 때 모델 크기와 tok/s만 비교하면 실제 비용을 놓치게 된다. 장비 가격, QSFP 케이블, 스위치, 전원과 냉각, 관리 장비, 운영 시간, 실패를 복구하는 시간까지 함께 계산해야 한다.

이 장의 가격과 전력 기준일은 **2026-08-21**이다. 가격은 판매처와 환율, 세금, 배송, 메모리·스토리지 구성에 따라 바뀌므로 고정된 구매 견적이 아니라 계산 방법과 시점이 표시된 스냅샷으로 읽는다.

## 3분 이해 (ELI5)

Spark 구매는 본체 가격만 보는 일이 아니다.

```text
처음 지출: 장비·케이블·스위치·저장소
계속 지출: 전기·냉각·운영·복구 시간
최종 판단: 필요한 작업을 통과한 비용
```

“가장 빠른 장비”보다 “내 workload를 감당하는 구성”의 비용을 계산한다.

## 12.1 먼저 구분할 세 가지 비용

### CAPEX: 처음 한 번 지출하는 비용

CAPEX에는 다음 항목을 포함한다.

| 항목 | 포함할 내용 | 빠뜨리기 쉬운 점 |
|---|---|---|
| Spark 본체 | DGX Spark 또는 GB10 기반 OEM 장비 | 1TB·4TB, Founders Edition·OEM, 국내 유통가가 다름 |
| 연결 | QSFP 케이블, 광모듈, breakout cable | 2대 direct와 4대 switch의 케이블 수가 다름 |
| 스위치 | 4대 이상 QSFP switch, 관리 포트, 전원 | 스위치 포트 수와 실제 RDMA 호환성을 확인해야 함 |
| 전원 | UPS, 멀티탭, 전원 분배, 여분 adapter | 240W adapter가 장비 가격에 포함되는지 확인 |
| 저장소 | 모델 캐시, 백업 SSD, 공유 스토리지 | 첫 DeepSeek recipe는 대규모 다운로드가 필요할 수 있음 |
| 관리 장비 | 관리용 Ethernet, 라우터, 콘솔, 모니터 | 계산망과 관리망을 분리할 때 추가됨 |
| Mac 선택 구성 | Mac Studio, Thunderbolt 케이블, 별도 저장소 | API worker인지 계산 rank인지 먼저 구분해야 함 |

### OPEX: 계속 발생하는 비용

OPEX에는 전력만 포함하지 않는다.

- 장비와 스위치를 켜 두는 전기료
- 냉각을 위해 추가한 팬과 에어컨의 전력
- 모델 저장소와 백업 디스크의 교체·확장 비용
- 소프트웨어 유지보수와 지원 계약
- 장애를 조사하고 재부팅·복구하는 시간
- 모델을 다시 다운로드하고 검증하는 시간

### TCO: 기간을 정해 합산하는 비용

비교 기간을 1년 또는 3년으로 고정하고, 같은 기간의 CAPEX와 OPEX를 합친다. 중고 판매가나 장비의 잔존 가치를 포함할 때는 별도 항목으로 기록하며, 처음부터 구매가에서 임의로 차감하지 않는다.

```text
CAPEX = 장비 + 연결 + 스위치 + UPS/전원 + 저장소 + 관리 장비 + 세금/배송

연간 전력비 = (평균 AC 전력 W / 1000) × 연간 가동 시간 × 실제 전력 단가

TCO(Y년) = CAPEX + Y × (연간 전력비 + 유지보수비 + 냉각비)
          - 기간 종료 시점의 잔존 가치
```

## 12.2 가격은 단일 숫자로 고정하지 않는다

NVIDIA Developer Forum의 2026년 2월 공지는 DGX Spark Founders Edition MSRP가 메모리 공급 문제로 **3,999달러에서 4,699달러로 조정되었다**고 설명한다. 현재 NVIDIA Marketplace에도 4TB 모델이 4,699달러로 표시되어 있다. [NVIDIA 가격 변경 공지](https://forums.developer.nvidia.com/t/2-23-2026-price-change-announcement/361713), [NVIDIA Marketplace](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/dgx-spark/)

한국에서 실제로 지불하는 금액은 이 MSRP와 다를 수 있다. 2026-08-21에 확인한 [다나와 DGX Spark 검색 결과](https://search.danawa.com/mobile/dsearch.php?keyword=dgx+spark)에는 4TB Founders Edition과 OEM·해외구매 제품이 대략 899만 원부터 1,350만 원대까지 서로 다른 가격으로 표시되어 있었다. 이 값은 재고와 환율에 따라 변하는 판매처 스냅샷이며, 특정 판매처나 OEM을 추천하는 근거가 아니다.

Mac Studio를 함께 고려할 때도 같은 원칙을 적용한다. Apple 한국 온라인 스토어에는 2026-08-21 기준 Mac Studio가 **429만 원부터** 표시되어 있다. 실제 메모리와 저장소를 올리면 가격이 달라지므로 Spark와 단순히 시작 가격만 비교하지 않는다. [Apple 한국 Mac 구매 페이지](https://www.apple.com/kr/shop/buy-mac)

가격을 표에 적을 때는 다음 열을 고정한다.

```text
checked_at:
seller:
product:
memory/storage:
currency:
tax_included:
shipping_included:
warranty:
price:
```

## 12.3 노드 수별 하드웨어 비용 구조

정확한 구매가는 `D`를 실제 견적값으로 넣어 계산한다. 아래 식에서 `Q`는 QSFP 케이블 1개, `S`는 호환 스위치와 부속품, `U`는 UPS·전원·저장소·관리 장비의 묶음이다.

| 구성 | 계산용 CAPEX 구조 | 네트워크 비용의 성격 | 구매 판단 |
|---:|---|---|---|
| 1대 | `D + U` | 계산용 interconnect 불필요 | 단일 모델과 개발 보조를 먼저 검증 |
| 2대 | `2D + Q + U` | 200GbE direct QSFP, 스위치 불필요 | DeepSeek TP=2와 긴 context의 기준점 |
| 3대 | `3D + 3Q + U` | QSFP ring, 공식 경로에서 스위치 불필요 | PP/DP와 모델별 실험을 감수할 때 선택 |
| 4대 | `4D + 4Q + S + U` | QSFP switch와 노드별 케이블 | TP=4 대형 모델 또는 shared fabric |
| 6~8대 | `ND + NQ + S + U` | 포트·케이블·전원·냉각이 별도 프로젝트 | 개인용 장비보다 클러스터 운영에 가까움 |
| 2×2 pool | `4D + 2Q + U` 또는 `4D + 4Q + S + U` | 두 pair를 직접 연결하거나 하나의 switch fabric으로 구성 | DS4 supervisor와 Qwen worker를 분리 |

`2×2 pool`에서 `2Q`가 가능한 경우는 DeepSeek pair와 Qwen pair를 각각 직접 연결할 때다. 두 pair 사이에 계산 collective가 없으므로 공통 계산용 switch가 필수는 아니다. 네 대를 하나의 TP fabric으로 사용하거나 공통 NCCL 경로를 만들면 `4Q + S` 구성을 다시 계산한다.

NVIDIA의 공식 playbook은 2대 direct QSFP, 3대 ring, 4대 이상 switch 경로를 나누어 설명한다. 따라서 3대에 스위치가 없다는 이유만으로 비정상 구성이라고 판단하지 않으며, 4대에서 스위치와 RDMA 비용이 처음으로 기본 항목이 된다고 기록한다. [2대 direct playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks), [3대 ring playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks), [다중 Spark switch playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch)

국내 시장의 스위치 가격은 호환성과 포트 수에 따라 크게 다르다. 예를 들어 한 판매 페이지에는 200G·400G 포트를 갖춘 MikroTik CRS812 계열이 265만 원으로 표시되어 있지만, 4대 Spark에 필요한 포트 구성과 실제 NCCL/RoCE 호환성은 별도로 검증해야 한다. 이 가격은 구매 추천가가 아니라 예산을 잡기 위한 시장 참고값이다. [국내 200G/400G 스위치 판매 예시](https://www.fibermart.co.kr/goods/view?no=16351)

## 12.4 전력 계산: adapter 정격과 실제 소비전력을 나눈다

DGX Spark에는 240W 외부 전원 공급 장치가 포함되지만, 240W는 장비가 항상 소비하는 전력이 아니다. NVIDIA Hardware Overview는 GB10 SoC TDP를 140W로 설명하고, 나머지 시스템 구성 요소에 100W를 배정한다. 규제 문서에는 최대 233.2W, idle 38.0W, off mode 4.1W가 별도로 제시되어 있다. [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [DGX Spark 전력·규제 정보](https://docs.nvidia.com/dgx/dgx-spark/compliance.html)

전력 예산은 다음 세 숫자를 구분해 기록한다.

| 숫자 | 의미 | TCO 계산에 쓰는가 |
|---|---|---|
| 240W | 제공된 외부 adapter의 정격 | 전원·UPS 용량 계산의 상한 참고 |
| 140W | GB10 SoC TDP | 열 설계와 부하 특성 참고 |
| 233.2W | 규제 문서의 최대 AC 전력 | 보수적인 회로·최대치 예산 |
| 실제 평균 AC W | 벽면에서 측정한 시간 평균 | 전기료 계산의 기본값 |

### 12.4.1 계산 예시

전력 단가를 **200원/kWh라고 가정한 예시**다. 실제 요금은 주택·일반용 계약, 누진 구간, 시간대, 부가요금에 따라 달라지므로 이 숫자를 고정된 전기요금으로 사용하지 않는다.

| Spark 수 | idle 38W | 최대 233.2W | 최대치 기준 연간 전력비 예시 |
|---:|---:|---:|---:|
| 1 | 333 kWh / 약 6.7만 원 | 2,043 kWh / 약 40.9만 원 | 약 40.9만 원 |
| 2 | 666 kWh / 약 13.3만 원 | 4,086 kWh / 약 81.7만 원 | 약 81.7만 원 |
| 3 | 999 kWh / 약 20.0만 원 | 6,129 kWh / 약 122.6만 원 | 약 122.6만 원 |
| 4 | 1,332 kWh / 약 26.6만 원 | 8,171 kWh / 약 163.4만 원 | 약 163.4만 원 |
| 8 | 2,663 kWh / 약 53.3만 원 | 16,343 kWh / 약 326.9만 원 | 약 326.9만 원 |

계산식은 `W ÷ 1000 × 8,760시간 × 원/kWh`다. 예를 들어 벽면에서 측정한 평균이 90W라면 Spark 한 대의 연간 사용량은 약 788kWh이고, 200원/kWh 가정에서는 약 15.8만 원이다. 90W는 특정 workload의 측정값이지 모든 inference의 정상 전력값으로 쓰지 않는다.

스위치, Mac, 팬, 모니터, UPS 자체 손실은 위 표에 포함하지 않았다. 4대 이상은 Spark만 최대 932.8W이고, 8대는 최대 1,865.6W이므로 스위치와 냉각 장비를 더한 뒤 사용하는 전원 회로와 UPS를 확인한다. 실제 설치 전에는 현지 전기 규정과 전기기사의 검토를 따른다.

## 12.5 전력은 벽면에서 측정한다

`nvidia-smi`의 GPU 또는 장치 전력은 전체 AC 소비전력과 같지 않다. CPU, unified memory, SSD, ConnectX-7, 팬, adapter 손실, USB-C 출력이 포함되는 범위가 다르다. TCO 표에 넣을 값은 가능하면 true-RMS 전력계를 벽면과 장비 사이에 연결해 기록한다.

최소 측정 프로토콜은 다음과 같다.

1. 부팅 후 30분 동안 아무 요청을 보내지 않고 idle 평균을 기록한다.
2. 같은 모델과 runtime으로 10분 warm-up 후 c1 decode를 측정한다.
3. 같은 조건에서 c4/c8 aggregate를 측정하고 노드별 전력과 온도를 기록한다.
4. 긴 prefill과 1~2시간 soak를 별도로 실행한다.
5. warm-up, 요청 처리, 대기, 재시작 구간의 시간 비율을 사용해 시간 가중 평균을 계산한다.

```text
power_profile:
  wall_meter_model:
  sample_interval_s:
  idle_w:
  warmup_w:
  c1_decode_w:
  c4_or_c8_w:
  prefill_w:
  soak_avg_w:
  soak_peak_w:
  temperature_peak_c:
  node_count:
  switch_w:
  mac_w:
```

낮은 clock과 낮은 power draw가 동시에 나타나면 성능이 낮은데도 건강해 보일 수 있다. 따라서 `P0`, GPU utilization, tok/s 하나만으로 전력 효율을 계산하지 않는다. clock, power, 실제 decode, 온도, 오류율을 같은 하니스에서 수집한다. 저클럭·저전력 장애 기록

## 12.6 구성별 구매 판단

| 목표 | 우선 구성 | 이유 | 구매 전 확인 |
|---|---|---|---|
| 로컬 개발과 단일 모델 | 1대 | interconnect와 switch 없이 시작할 수 있음 | 모델이 128 GiB 안에서 KV headroom을 남기는가 |
| DeepSeek 장문·TP=2 | 2대 direct | 1M/256K profile과 원본 계열 TP=2 경로가 가장 명확함 | QSFP, RDMA, NCCL, 전원·냉각 |
| 세 개의 독립 endpoint | 3대 DP 또는 2+1 | TP=3보다 서비스 분리가 예측 가능함 | 관리망, 장애 격리, 모델별 메모리 |
| 대형 단일 모델 | 4대 switch/TP=4 | Qwen·DeepSeek·GLM 계열의 다중 노드 레시피가 존재함 | switch port, `NET/IB`, NCCL version, 케이블 |
| DS4 brain + Qwen UI worker | 2×2 pool | 두 모델을 독립 `TP=2` 서비스로 운영함 | 전체 4대, pair별 direct 또는 공통 switch |
| 연구용 클러스터 | 6~8대 | 여러 모델과 topology를 실험할 여유가 있음 | 전용 전원, 냉각, 스위치, 복구 절차, 운영 시간 |

다음 질문에 “예”라고 답할 수 있을 때만 노드를 추가한다.

- 현재 모델이 실제로 메모리 부족 때문에 실패했는가?
- 추가 노드가 TP, PP, DP 중 어떤 방식으로 사용될지 정해졌는가?
- 추가된 노드가 늘리는 것은 모델 용량인가, 동시성인가, 단일 요청 속도인가?
- direct, ring, switch 중 실제 topology와 NCCL transport를 검증했는가?
- 두 시간 이상 soak 후 memory, power, temperature, error, restart 결과가 있는가?
- 장비를 24시간 켜 두지 않아도 되는 workload라면 스케줄링으로 유휴 전력을 줄일 수 있는가?

## 12.7 Mac을 추가할 때의 손익

Mac은 Spark의 CUDA TP rank가 아니라 다음 역할로 평가한다.

| Mac 역할 | 얻는 것 | 추가 비용 |
|---|---|---|
| control host/router | UI, SSH, API gateway, 인증, 로그 | Mac 본체와 전력 |
| 별도 MLX worker | 문서 전처리, 빠른 작은 모델, 역할 분리 | 모델 중복 메모리와 endpoint 운영 |
| DS4 decode 보조 실험 | MCDMA 또는 별도 API pipeline 연구 | USB-C 안정성, 검증 시간, 소스 의존성 |
| Spark 메모리 확장 | 공식적으로 검증된 방법이 아님 | 구매 근거로 삼지 않음 |

표준 구성에서는 Spark의 계산망과 Mac의 API endpoint를 분리한다. [Mac·RDMA·스위치 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-mac-rdma-switch-research-2026-08.md)에 정리한 MCDMA는 공개 구현과 독립 tokens/s 검증이 끝나기 전까지 CAPEX 절감이나 메모리 pooling의 근거로 계산하지 않는다.

Mac을 추가한다고 Spark 한 대로 충분했던 모델이 자동으로 두 장비의 메모리를 합쳐 사용하는 것은 아니다. Mac을 사는 이유가 UI·라우팅·문서 전처리·별도 모델인지, 실제 모델 capacity를 늘리려는 것인지 먼저 적는다.

## 12.8 TCO와 클라우드 비용을 비교하는 법

로컬 장비와 클라우드를 비교할 때 raw tok/s나 장비 가격만 비교하지 않는다. 다음 단위를 먼저 정한다.

```text
유효 작업량 = 성공한 요청 수 × 품질 통과율 × 요청당 유효 output

로컬 월 비용 = (CAPEX - 잔존 가치) / 사용 개월 수
             + 월 전력비 + 냉각비 + 유지보수비

클라우드 월 비용 = input token 비용 + output token 비용
                 + 저장소·네트워크·예약 인스턴스 비용
```

클라우드 token 단가는 계약과 모델에 따라 바뀌므로 이 장에는 고정값을 넣지 않는다. 같은 prompt set, 같은 output budget, 같은 품질 통과 기준, 같은 동시성으로 측정한 뒤 월간 요청량을 대입한다.

로컬이 경제적인 경우는 다음과 같다.

- 장비를 장시간 사용하고 요청량이 충분하다.
- 입력 데이터와 tool 결과를 외부로 보내지 않아야 한다.
- 네트워크 왕복보다 로컬 지연과 항상 켜진 endpoint가 중요하다.
- 하나의 모델보다 여러 모델과 실험을 반복해 활용한다.

반대로 요청량이 적거나 최신 모델을 자주 바꾸거나, 전력·냉각·장애 대응을 직접 운영하기 어렵다면 CAPEX가 낮아도 로컬이 경제적이지 않을 수 있다.

## 이 장의 기록 템플릿

```text
decision_date: 2026-08-21
currency: KRW / USD
electricity_tariff_krw_per_kwh:
usage_hours_per_day:
target_model:
target_profile: c1 / c4 / long-context / agent

spark_count:
spark_unit_price:
mac_count:
mac_unit_price:
topology: direct / ring / switch / 2x2-pairs
qsfp_cable_count:
switch_model:
switch_price:
ups_and_power_price:
storage_and_backup_price:
cooling_price:

capex_total:
measured_idle_w:
measured_serving_avg_w:
measured_peak_w:
annual_energy_cost:
annual_maintenance_cost:
tco_1y:
tco_3y:
quality_pass_rate:
successful_requests_per_month:
cloud_comparison_assumption:
decision:
```

## 이 장의 결론

1대는 가장 낮은 진입 비용으로 모델과 runtime을 검증하는 구성이다. 2대는 DeepSeek TP=2와 긴 context에서 비용 대비 체감이 가장 큰 확장점이다. 3대는 TP=3 자체보다 DP, PP, 2+1 서비스 분리에 돈을 쓰는 단계다. 4대부터는 switch, 전원, 냉각, NCCL 운영이 장비 가격만큼 중요해진다. 8대는 더 빠른 개인용 PC가 아니라 작은 클러스터다.

구매 결론은 `가장 큰 모델을 올릴 수 있는가`가 아니라 `검증된 workload를 얼마의 총비용으로 안정적으로 처리하는가`로 정한다.

## 참고 자료

- [NVIDIA DGX Spark Marketplace](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/dgx-spark/)
- [NVIDIA DGX Spark 가격 변경 공지](https://forums.developer.nvidia.com/t/2-23-2026-price-change-announcement/361713)
- [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark 전력·규제 정보](https://docs.nvidia.com/dgx/dgx-spark/compliance.html)
- [Apple 한국 Mac 구매 페이지](https://www.apple.com/kr/shop/buy-mac)
- [다나와 DGX Spark 시장 가격 검색](https://search.danawa.com/mobile/dsearch.php?keyword=dgx+spark)
- [국내 200G/400G 스위치 판매 예시](https://www.fibermart.co.kr/goods/view?no=16351)
- [NVIDIA Connect Two Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks)
- [NVIDIA Connect Three Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks)
- [NVIDIA Multi-Spark Switch](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch)
