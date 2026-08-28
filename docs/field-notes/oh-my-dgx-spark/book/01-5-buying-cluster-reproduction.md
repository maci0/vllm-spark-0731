# 01-5. 구매·클러스터·재현

전체 선택 기준은 [DGX Spark·GB10 벤더 비교](01-2-gb10-vendor-comparison.md)에서 확인할 수 있습니다.

## 용도별 선택

| 우선순위 | 후보 | 이유 |
|---|---|---|
| NVIDIA recipe 재현 | NVIDIA DGX Spark | 공식 문서와 플레이북의 기준 |
| 기존 ASUS 실험을 이어감 | ASUS GX10 | 이 저장소의 현장·DeepSeek 자료가 많음 |
| 열 여유를 우선 | Acer를 후보로 검증 | 독립 열 비교의 Acer 샘플이 가장 낮았음 |
| 기업 지원·현장 교체 | Dell·Lenovo·HP | 지원 계약·보안·관리 기능이 구체적 |
| 모델 관리 UI/RAG | GIGABYTE | AI TOP Utility가 차별점 |
| 엣지·보안 appliance | MSI | Docker Compose·SSL·private CA 방향 |
| 메모리 용량·MLX | Apple Mac Studio | GB10 cluster가 아닌 별도 endpoint |
| x86·ROCm 실험 | AMD Ryzen AI Halo | GB10과 다른 runtime·통신 경로 |

가격 비교에는 본체만 넣지 않는다.

```text
본체 + SSD SKU + 공식 adapter + QSFP cable
+ switch(4대 이상) + support contract + 전력·냉각
```

지역·환율·재고·지원 기간에 따라 가격이 변하므로, 조회일이 없는 “최저가”를 책의 기준으로 쓰지 않는다.

## GB10 클러스터

같은 GB10 OEM끼리의 mixed cluster가 이론적으로 가능해 보여도 다음을 확인한다.

1. OS·driver·CUDA·NCCL을 같은 수준으로 맞춘다.
2. 각 장비의 CX-7 인터페이스와 GID·MTU를 기록한다.
3. 2대 direct는 `ib_write_bw`와 `NCCL_DEBUG=INFO`의 `NET/IB`를 확인한다.
4. 3대 ring·4대 switch는 management NIC로 fallback하지 않는지 확인한다.
5. 작은 all-reduce와 실제 모델의 TP 요청을 순서대로 실행한다.
6. 30분 이상 soak에서 온도·clock·power·error를 기록한다.

```bash
hostnamectl
uname -a
cat /etc/os-release
nvidia-smi
free -h
df -h
ip -br addr
ibv_devices || true
ibdev2netdev || true
docker version
```

## GB10이 아닌 대안과의 조합

### AMD Ryzen AI Halo

[AMD 공식 자료](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html)는 최대 128GB LPDDR5x, 256GB/s, Radeon 8060S, 120W TDP, Linux·Windows 경로를 표기한다. 메모리·전력 비교에는 유용하지만 CUDA·DGX OS·CX-7이 아니므로 Spark TP 노드로 바로 섞지 않는다.

### Apple Mac Studio

[Apple의 M3 Ultra 발표](https://www.apple.com/uk/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)는 최대 512GB unified memory와 800GB/s 초과 메모리 대역폭을 설명한다. [Apple 전력 표](https://support.apple.com/en-la/102027)는 2025 M3 Ultra 512GB/16TB 구성의 최대 소비전력을 270W로 표시한다.

Mac은 Metal/MLX worker, router, control host 또는 별도 endpoint로 쓰는 것이 기본이다. MCDMA·Thunderbolt·RDMA 유사 경로는 별도 커뮤니티 실험이며, 공식 DGX Spark memory pooling이나 CUDA/NCCL TP 경로로 문서화하지 않는다.

## 구매 후 재현 체크리스트

- [ ] 모델명과 SKU를 사진·serial과 함께 저장했다.
- [ ] adapter 정격과 벽면 AC 측정기를 기록했다.
- [ ] OS image·driver·CUDA·NCCL·kernel을 저장했다.
- [ ] `nvidia-smi`의 clock·power·temperature를 baseline으로 남겼다.
- [ ] storage model·capacity·firmware를 기록했다.
- [ ] CX-7의 link speed와 실제 `NET/IB` 경로를 확인했다.
- [ ] 동일 모델·quant·engine·context로 c1/c4를 측정했다.
- [ ] `loads`와 `tool-tested`를 별도 상태로 표시했다.
- [ ] 제조사 주장·커뮤니티 숫자·직접 실험을 별도 표로 보존했다.

## 이 장의 결론

GB10 OEM 선택은 “같은 칩이니 아무거나”도 아니고 “브랜드가 성능을 결정한다”도 아니다. 공통 플랫폼 위에 냉각·전원·firmware·storage·support가 다르게 얹힌다. 따라서 먼저 공식 인증 목록에서 후보를 좁히고, 그다음 자신의 모델 recipe를 고정해 열·성능·agent 성공률을 직접 비교한다.
