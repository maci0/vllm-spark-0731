# 01-2. DGX Spark·GB10 벤더 비교

상태: 공식 사양·독립 측정·커뮤니티 자료를 분리한 리서치 장
기준일: **2026-08-22**

NVIDIA DGX Spark만 있는 것이 아니다. 같은 NVIDIA GB10을 사용하는 Acer, ASUS, Dell, GIGABYTE, HP, Lenovo, MSI의 제품이 함께 시장에 나와 있다. 이 장의 핵심은 “어느 회사 제품이 같은 칩을 쓰는가”와 “실제로 어떤 차이가 생기는가”를 나누는 것이다.

## 3분 이해 (ELI5)

같은 GB10을 쓰는 장비는 같은 엔진을 넣은 자동차와 비슷하다.

```text
같은 GB10
├─ 냉각
├─ 저장장치
├─ 전원·펌웨어
└─ 지원·운영
→ 지속 성능과 복구 경험의 차이
```

모델·runtime·workload를 고정한 뒤 장비 차이를 비교한다.

![같은 GB10이라도 시스템 구성에 따라 운영 결과가 달라지는 Archify 다이어그램](../assets/archify-gb10-vendor.svg)

## 17.1 두 종류의 대안

### 같은 GB10을 쓰는 파트너 시스템

NVIDIA의 [공식 인증 목록](https://docs.nvidia.com/certification-programs/latest/nvidia-certified-systems.html)은 다음을 `DGX Spark / Grace Blackwell GB10 systems`로 열거한다.

| 제조사 | 모델 | 차이를 볼 항목 |
|---|---|---|
| NVIDIA | DGX Spark | Founders Edition·NVIDIA recipe의 기준 |
| Acer | Veriton GN100 | 냉각, 4TB 구성, 4대 switch 안내 |
| ASUS | Ascent GX10 | 냉각, ASUS 현장 recipe·지원 |
| Dell | Pro Max with GB10 | 280W adapter, 지원 계약, appliance mode |
| GIGABYTE | AI TOP ATOM | AI TOP Utility, 모델 관리·RAG |
| HP | ZGX Nano | ZGX Toolkit, 기업 배치, rack 제한 |
| Lenovo | ThinkStation PGX | 보안 기능과 ThinkStation 지원 |
| MSI | EdgeXpert | Docker Compose, SSL/private CA, 엣지 appliance |

이 장비들은 대체로 GB10, 128GB unified memory, 273GB/s, ConnectX-7이라는 공통 기반을 가진다. 그러나 storage SKU, 냉각, 펌웨어, power adapter, 초기 이미지, 지원 정책은 다르다.

### GB10이 아닌 대형 메모리 장비

AMD Ryzen AI Halo와 Apple Mac Studio는 메모리 용량만 보면 경쟁 대안이지만, CUDA·DGX OS·ConnectX-7·NCCL recipe가 같은 장비가 아니다. 01-5. 구매·클러스터·재현에서 이 경계를 따로 다룬다.

## 17.2 같은 사양이 같은 성능을 보장하지 않는 이유

128GB와 1 PFLOP 표기는 공통 플랫폼을 설명하지만, 지속 부하 성능은 다음에 영향을 받는다.

- 냉각 경로와 fan curve
- SSD 모델과 chassis 내부 열 전달
- GB10 clock·전력·펌웨어 상태
- ConnectX-7 온도와 실제 RDMA/NCCL 경로
- DGX OS image·driver·runtime 버전
- 장비 지원·복구·어댑터 조건

[StorageReview의 다섯 장비 비교](https://www.storagereview.com/review/nvidia-dgx-spark-thermal-test-how-oem-cooling-designs-stack-up)는 NVIDIA·Gigabyte·Dell·Acer·ASUS를 같은 환경에서 시험했고, Acer 샘플이 prefill-heavy에서 CPU 74.6°C, GPU 68°C로 가장 낮게 측정됐다고 보고했다. Founders Edition·Dell·Gigabyte는 CPU 약 87~88°C, GPU 약 80~82°C였고 ASUS는 중간이었다. 이 자료는 냉각 차이를 보여주지만, 모든 workload의 tok/s 순위는 아니다.

## 17.3 벤더별 상세와 선택 기준

- 01-3. 공식 GB10 시스템별 사양
- 01-4. 냉각·성능·X 자료 읽기
- 01-5. 구매·클러스터·재현

## 이 장의 결론

NVIDIA Founders Edition은 문서와 recipe의 기준점이다. OEM을 고를 때는 같은 GB10이라는 사실보다 냉각·스토리지·지원·운영 모드를 비교한다. 독립 측정에서는 Acer 샘플의 열 여유가 돋보였지만, 이것을 곧바로 “가장 빠른 모델”로 번역하지 않는다. 실제 구매 후에는 같은 모델·quant·engine·context·concurrency로 다시 측정해야 한다.

자세한 출처와 표는 [DGX Spark·GB10 벤더 비교 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-vendor-comparison-2026-08.md)에 기록한다.
