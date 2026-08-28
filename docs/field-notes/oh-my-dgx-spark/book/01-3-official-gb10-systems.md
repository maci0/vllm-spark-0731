# 01-3. 공식 GB10 시스템별 사양

전체 선택 기준은 [DGX Spark·GB10 벤더 비교](01-2-gb10-vendor-comparison.md)에서 확인할 수 있습니다.

기준일: **2026-08-22**

## 공통 기준

[NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)를 기준으로 보면 GB10 계열의 공통 축은 다음과 같다.

| 항목 | 기준값 |
|---|---|
| SoC | Grace Blackwell GB10, 20-core Arm CPU + Blackwell GPU |
| unified memory | 128GB LPDDR5x, 256-bit, 273GB/s |
| AI 표기 | 최대 1 PFLOP FP4, sparsity 조건 |
| 고속 연결 | 2× QSFP ConnectX-7, 포트당 최대 200Gb/s 표기 |
| 관리 연결 | 10GbE, Wi-Fi 7 등 |
| 운영체제 | NVIDIA DGX OS 계열 |

공통 사양을 모델의 실제 메모리 여유나 tok/s로 해석하지 않는다. weight·KV·workspace·OS가 같은 풀을 사용한다.

## 제조사별 차이

| 제조사·제품 | 공식 문서에서 확인한 내용 | 구매·운영 포인트 |
|---|---|---|
| NVIDIA DGX Spark | 1/4TB NVMe SKU, 240W adapter, DGX OS, CX-7 | NVIDIA 문서·플레이북의 기준 |
| Acer Veriton GN100 | 128GB, 최대 4TB, PyTorch/Jupyter/Ollama, CX-7, 최대 4대 switch 안내 | 저장장치와 냉각 후보 |
| ASUS Ascent GX10 | 1/2/4TB 계열, QuietFlow 3팬·dual vapor chamber, 240W peak | ASUS 현장 recipe와 support |
| Dell Pro Max with GB10 | 2TB QLC 또는 4TB SED, 2×200G QSFP, 280W adapter, Desktop/Network Appliance mode | 기업 지원·어댑터·운영 모드 |
| GIGABYTE AI TOP ATOM | 최대 4TB, CX-7, NVIDIA stack, AI TOP Utility | 모델 다운로드·RAG·관리 UI |
| HP ZGX Nano | 2/4TB 계열 SED, 2×200G QSFP, 240W, DGX OS/Ubuntu | Windows는 장비 OS가 아니며 rack mount 미지원 |
| Lenovo ThinkStation PGX | 1/4TB SED, 2×CX-7 QSFP, 240W, DGX OS, TPM·Secure Boot·FW Recovery | workstation 보안·서비스 |
| MSI EdgeXpert | 1/4TB SED, 2×QSFP CX-7, 240W, DGX OS, Docker Compose·private CA | 엣지 appliance·보안 orchestration |

### 사양 링크

- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [Acer Veriton GN100](https://news.acer.com/acer-unveils-the-veriton-gn100-ai-mini-workstation-built-on-the-nvidia-gb10-superchip)
- [ASUS Ascent GX10](https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/)
- [ASUS GX10 FAQ](https://www.asus.com/us/support/faq/1056142/)
- [Dell Pro Max with GB10](https://www.dell.com/en-sg/shop/pcs-desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro)
- [GIGABYTE AI TOP ATOM](https://www.gigabyte.com/AI-TOP-PC/GIGABYTE-AI-TOP-ATOM)
- [HP ZGX Nano QuickSpecs](https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c09212373)
- [Lenovo ThinkStation PGX](https://lenovopress.lenovo.com/lp2321-thinkstation-pgx)
- [MSI EdgeXpert](https://ipc.msi.com/product_detail/EdgeXpert-MS-C931)

## 반드시 따로 기록할 것

```text
vendor, model, sku, serial
memory_total, storage_model, storage_size
os, kernel, driver, cuda, nccl
cx7_interface, link_speed, mtu, transport
power_adapter, ambient_temperature, firmware
```

제조사 페이지가 4TB를 말해도 실제 구매 SKU가 1TB일 수 있고, 제품 페이지의 `200Gb/s`가 실제 inference payload를 보장하지 않는다. 구매 후 `ethtool`, `ibdev2netdev`, `NCCL_DEBUG=INFO`로 확인한다.
