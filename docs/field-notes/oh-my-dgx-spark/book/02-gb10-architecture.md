# 02. 모델·메모리·노드 수를 이해하는 법

이 장의 목적은 128GB라는 숫자를 모델 파일 크기와 혼동하지 않는 데 있습니다. DGX Spark의 unified memory는 모델 weight만 보관하는 전용 공간이 아니라 CPU·GPU·runtime·운영체제가 함께 사용하는 예산입니다.

![unified memory를 하나의 메모리 예산으로 읽는 Archify 다이어그램](../assets/archify-unified-memory.svg)

## 하나의 메모리 예산으로 계산한다

실행 중인 메모리는 최소한 다음처럼 나뉩니다.

```text
128 GiB
 ├─ model weights
 ├─ KV cache: context와 sequence 수에 따라 증가
 ├─ CUDA workspace·graph·temporary buffer
 └─ OS·Docker·파일 cache·다른 process
```

따라서 `weight가 100GB이므로 28GB가 남는다`는 계산만으로는 충분하지 않습니다. quant format, KV dtype, context, `max_num_seqs`, CUDA graph와 runtime overhead를 함께 확인해야 합니다.

NVIDIA Porting Guide는 UMA에서 CPU와 GPU가 같은 물리 메모리 공간을 공유한다고 설명합니다. 이 구조는 CPU와 GPU 사이의 복사를 줄일 수 있지만, CPU 작업이 메모리 예산을 사용하는 사실까지 없애지는 않습니다.

출처: [NVIDIA DGX Spark Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html).

## 모델 크기와 실행 가능성을 분리한다

| 상태 | 의미 |
|---|---|
| 파일 다운로드 완료 | 디스크에 checkpoint가 저장되어 있습니다. |
| weight load 완료 | runtime이 weight를 메모리에 배치했습니다. |
| 짧은 생성 성공 | 최소 요청을 처리했습니다. |
| 긴 context 성공 | 지정한 context에서 KV와 workspace 여유가 남았습니다. |
| 장시간·tool loop 성공 | 실제 운영 조건을 일부 통과했습니다. |

MoE 모델에서는 total parameter와 active parameter도 구분해야 합니다. active parameter가 작아도 모든 expert weight, KV cache, runtime buffer가 사라지는 것은 아닙니다.

## 양자화와 speculative decoding

양자화는 weight가 차지하는 공간을 줄이는 방법입니다. NVFP4, FP8, INT4, GGUF, EXL3는 같은 “몇 bit”라는 표현만으로 품질과 성능을 비교할 수 없는 서로 다른 형식과 runtime 경로입니다.

speculative decoding은 draft가 후보 token을 만들고 target이 검증해 decode를 가속하는 방식입니다. draft acceptance가 낮거나 workload가 draft와 맞지 않으면 이득이 작거나 사라질 수 있습니다. `MTP`, `DSpark`, `DFlash2`는 같은 단어가 아니며 레시피별 구현과 조건을 기록합니다.

## TP·PP·DP를 먼저 구분한다

- **TP(Tensor Parallelism)**: 한 모델의 tensor 계산을 여러 노드가 나눕니다. 통신이 매 step의 성능에 영향을 줍니다.
- **PP(Pipeline Parallelism)**: 모델 층을 stage로 나눕니다. pipeline bubble과 microbatch 설계가 필요합니다.
- **DP(Data Parallelism)**: 같은 모델 endpoint를 여러 개 띄워 요청을 나눕니다. 모델을 더 크게 만드는 방법은 아닙니다.

노드 수가 늘면 “메모리가 합쳐진다”기보다 병렬화 방식과 통신 경로가 추가됩니다. 두 대를 마련하기 전에 한 모델을 나눌지, 두 endpoint를 독립적으로 운영할지 결정해야 합니다.

## 공식 범위와 개인 실험의 경계

공식 하드웨어 페이지의 200B 단일·405B 듀얼 표기는 출발점입니다. 다음 질문에 답하려면 별도 레시피가 필요합니다.

- 어느 checkpoint인가?
- quant와 KV dtype은 무엇인가?
- context와 `max_num_seqs`는 얼마인가?
- vLLM·SGLang·llama.cpp 중 어느 runtime인가?
- 단일 stream인가, aggregate throughput인가?
- 긴 요청과 tool loop를 통과했는가?

## 이 장의 결론

Spark의 핵심은 128GB라는 숫자 자체가 아니라 **공유 메모리 예산과 273GB/s 대역폭을 workload에 맞게 사용하는 것**입니다. 모델을 고를 때 파일 크기부터 보지 말고, weight·KV·workspace·OS를 합친 실행 예산을 계산해야 합니다.

## 참고

- [NVIDIA Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)
- [DGX Spark 모델·클러스터 리서치](../docs/dgx-spark-cluster-model-research-2026-08.md)

## 더 자세히 읽기

[02-1. GB10·unified memory 상세](02-1-gb10-unified-memory.md)에는 메모리 압박을 측정하는 값, TP·PP·DP의 차이, 장비를 받은 직후 기록할 정보와 안전한 기본값을 보존했습니다.
