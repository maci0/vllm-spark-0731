# 01-4. 냉각·성능·X 자료 읽기

전체 선택 기준은 [DGX Spark·GB10 벤더 비교](01-2-gb10-vendor-comparison.md)에서 확인할 수 있습니다.

## StorageReview의 통제된 열 비교

[StorageReview](https://www.storagereview.com/review/nvidia-dgx-spark-thermal-test-how-oem-cooling-designs-stack-up)는 NVIDIA Founders Edition, Gigabyte, Dell, Acer, ASUS 다섯 장비를 같은 방에서 시험했다.

- 모델: `GPT-OSS-120B`
- 엔진: vLLM online serving benchmark
- workload: 256/256 equal, 4096/512 prefill-heavy, 512/4096 decode-heavy
- batch: 1·2·4·8·16·32·64·128
- 단계 사이 30초 cooldown
- metric: kernel interface와 `nvidia-smi`를 1초 간격 수집

| 항목 | 기사에 공개된 관찰 |
|---|---|
| CPU 최고점 | Acer 74.6°C, FE·Dell·Gigabyte 87~88°C, ASUS는 중간 |
| GPU 최고점 | Acer 68°C, 나머지 네 장비 80~82°C |
| NVMe 최고점 | Acer 51.8°C, 다른 장비 약 58~63°C. SSD가 달라 완전 동일 비교 아님 |
| CX-7 최고점 | Acer 62°C, FE 75°C |
| GPU rail peak | Acer 약 69.3W~Gigabyte 약 76.0W |

이 글은 냉각·전력 비교다. end-to-end tok/s 리더보드가 아니며, NVMe 모델과 GPU rail power가 모두 완전히 동일하지도 않다. 따라서 책의 결론은 “Acer가 무조건 빠르다”가 아니라 “OEM 냉각이 지속 부하의 열 여유를 바꾼다”다.

## 우리 저장소의 커뮤니티·현장 자료

현재 저장소에는 ASUS GX10을 중심으로 한 DeepSeek FP8·EXL3·Qwen·clock cap·spin-wait 자료가 있다. 이 수치는 모델·quant·engine·driver가 서로 다르므로 벤더 순위표에 합치지 않는다.

- [ASUS GX10 노드 세팅](../docs/dgx-spark-node-setup-research-2026-08.md)
- [DeepSeek V4 Flash ASUS/NVIDIA 비교](../docs/deepseek-v4-flash-0731-community-builds-2026-08.md)
- [GB10 clock cap GitHub harness](https://github.com/agjs/gb10-clock-cap)
- [Nacyot GB10 clock cap 측정](https://nacyot.github.io/artifacts/gb10-clock-cap/)

## X에서 확인한 자료

| 링크 | 무엇을 보여주는가 | 이 책의 판정 |
|---|---|---|
| [Lenovo Project Kubit](https://x.com/thinkstations/status/2024514312312647851) | 두 PGX를 개인 AI 허브로 묶는 개념 | 제품 활용 방향. benchmark 아님 |
| [Ivan Fioravanti clock cap](https://x.com/ivanfioravanti/status/2088730630875930639?s=20) | DeepSeek c4에서 GPU clock·온도·전력 변화 | clock 실험. OEM 비교 아님 |
| [Ash Hart MCDMA](https://x.com/ashxhart/status/2089749434087227672?s=20) | Mac Studio와 Spark의 USB-C 메모리 전송 주장 | 공개 재현 전 prototype |
| [0xSero local SOTA](https://x.com/0xSero/status/2039742489276395818) | RTX·Mac·DGX Spark에서 local model을 돌리는 흐름 | 플랫폼 언급. 사양·성능 근거 아님 |
| [X Mac cluster 요약](https://x.com/i/trending/2001731662288486469) | Thunderbolt·Exo로 여러 Mac을 묶는 사례 요약 | 2차 요약. 원문과 조건 재확인 필요 |

X에서 본 `tok/s`, `temperature`, `power`는 다음을 확인하기 전까지 D 등급으로 남긴다.

- 모델 revision과 quant
- engine과 speculative decoding
- context·prompt·output·concurrency
- GPU rail인지 wall AC인지
- firmware·clock cap·cooling state
- raw log와 반복 횟수

## 직접 비교할 때의 최소 표

```text
vendor/model, hardware sku, os/image, driver/cuda/nccl
model revision, quant, kv dtype, context
prompt/output, thinking, speculation, concurrency
prefill, decode, ttft, e2e, aggregate
gpu rail power, wall power, cpu/gpu/nic/nvme temp
transport, nccl path, errors, quality/tool result
```

`P0`이거나 utilization이 높다는 사실만으로 정상 성능이라고 판정하지 않는다. 저클럭·저전력 상태는 `clocks.sm`, `power.draw`, 온도와 실제 token rate를 함께 볼 때만 드러난다.
