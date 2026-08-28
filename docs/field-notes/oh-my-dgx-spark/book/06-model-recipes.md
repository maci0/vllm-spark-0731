# 06. DeepSeek·Qwen·MiniMax 레시피

모델을 고르는 기준은 이름이나 인터넷에 공개된 최고 tok/s가 아닙니다. **작업, 메모리 예산, 실행 경로, 검증 상태**를 함께 봐야 합니다.

## 한눈에 비교하기

| 모델·경로 | 주된 역할 | 1대 | 2대 이상 | 현재 판정 |
|---|---|---|---|---|
| DeepSeek V4 Flash 0731 | 긴 문맥·코딩·supervisor | EXL3/SparkInfer 레시피 | TP=2·DSpark 레시피 | 강한 후보이지만, 원본과 양자화 경로를 구분해야 함 |
| Qwen3.8-27B | 코드·UI·JSON worker | SGLang/DFlash2 등 공개 레시피 | TP보다 독립 worker가 단순할 때가 많음 | 빠른 worker 후보이지만, runtime architecture 확인 필요 |
| Qwen3.6-35B-A3B-NVFP4 | 공식 agent-ready 기준 구성 | NVIDIA vLLM | 공식 multi-Spark 경로 확인 | 첫 기준 구성으로 적합 |
| MiniMax M2.7/M3 | 대형 reasoning 후보 | 레시피별 확인 | 메모리·runtime 조건이 큼 | 공개 보고와 직접 검증을 분리해야 함 |
| MiniMax-H3 | 영상·음성 생성 | `sm_121` custom 레시피 | 별도 pipeline | 일반 text LLM benchmark와 분리해야 함 |

## DeepSeek V4 Flash 0731

DeepSeek 공식 모델 카드는 Terminal Bench 2.1 82.7, Toolathlon-Verified 70.3 등의 평가 결과를 제시합니다. 이는 모델 제작자가 수행한 benchmark이며, DGX Spark에서 얻은 tok/s나 이 책의 C1과 같은 측정이 아닙니다.

단일 Spark의 MiaAI-Lab 레시피는 EXL3 quantization, SparkInfer, DSpark draft, native NVFP4 KV 경로를 조합합니다. README가 보고하는 384K 설정, 구조화 decode 약 44–47 tok/s, 370K needle 결과는 **그 레시피의 single-stream 조건**입니다. 원본 full-FP8 checkpoint의 모든 품질·동시성·장시간 안정성을 보증하지는 않습니다.

출처: [DeepSeek 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), [단일 Spark 레시피](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark), [성능 리서치](../docs/deepseek-v4-flash-0731-performance-research-2026-08.md).

## Qwen3.8-27B

Qwen3.8-27B는 이 책에서 코드·UI·JSON 반복 작업을 맡는 worker 후보로 다룹니다. 커뮤니티 레시피의 SGLang, NVFP4, DFlash2, speculative decoding 조합은 서로 다르므로 “Qwen3.8의 속도”라는 단일 값을 만들지 않습니다.

특히 runtime이 Qwen3.5 architecture를 인식하는 로그가 보이면 모델 이름만 보고 성공으로 판정하지 않습니다. `config.json`, model implementation, runtime commit, tool parser를 함께 기록해야 합니다.

출처: [Qwen3.8 community builds](../docs/qwen38-community-builds-2026-08.md), [MiaAI-Lab 레시피](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark).

## MiniMax와 H3

MiniMax 계열은 text reasoning 모델과 영상·음성 생성 모델을 같은 benchmark 표에 넣지 않습니다. H3 custom kernel 레시피가 특정 `sm_121` 환경에서 동작해도, 일반 vLLM endpoint나 text decode 성능을 의미하지 않습니다.

H3를 평가할 때는 생성 품질, frame/audio throughput, VRAM·unified memory 사용량, pipeline latency를 따로 기록합니다. 현재 문서의 공개 결과는 재현 후보로 남기며, 직접 실행하기 전에는 확정 수치로 기록하지 않습니다.

## 1대·2대·4대 선택

- **1대**: 공식 Qwen3.6 기준 구성 또는 Qwen3.8 worker로 시작합니다. DeepSeek는 특수한 single-Spark 레시피로 별도 비교합니다.
- **2대**: DeepSeek TP=2와 긴 context가 가장 설득력 있는 사용 사례입니다. 독립 endpoint 두 개를 운영하는 DP 구성도 단순합니다.
- **4대**: 단일 TP=4 모델, 또는 DeepSeek TP=2와 Qwen TP=2로 구성한 두 pool을 비교합니다. 두 pool의 메모리는 자동으로 합쳐지지 않습니다.

## 모델 선택 문장

다음처럼 쓰면 사실과 판단을 분리할 수 있습니다.

> 이 레시피는 `1x DGX Spark`, `EXL3`, `SparkInfer`, `max_model_len=384000`, single-stream 조건에서 실행되었다는 보고가 있습니다. 따라서 긴 문맥 supervisor 후보로 분류합니다. 일반적인 “DeepSeek가 Sol max와 동급” 또는 모든 요청에서 47 tok/s라는 문장은 이 자료만으로는 쓸 수 없습니다.

## 더 자세히 읽기

기초와 선택 기준:

- [06-1. 양자화·KV cache·speculative decoding](06-1-quantization-speculative.md)
- [06-2. 양자화와 메모리 예산](06-2-quant-memory-budget.md)
- [06-3. speculative decoding과 품질 게이트](06-3-speculative-quality-gate.md)
- [06-4. DGX Spark에서 돌릴 모델 선택](06-4-model-selection.md)

모델별 상세:

- [06-5. DeepSeek V4 Flash 0731](06-5-deepseek-v4-flash.md)
- [06-6. Qwen3.8-27B](06-6-qwen38-27b.md)
- [06-7. MiniMax M2.7과 M3](06-7-minimax-m2-m3.md)
- [06-8. MiniMax-H3 영상 생성](06-8-minimax-h3.md)

커뮤니티 활용과 재현 후보:

- [06-9. DeepSeek로 사람들이 만든 것](06-9-deepseek-community-builds.md)
- [06-10. DeepSeek 단일·듀얼 Spark 제작물](06-10-deepseek-single-dual.md)
- [06-11. DeepSeek vision shim과 에이전트](06-11-deepseek-vision-agent.md)
- [06-12. Qwen3.8로 사람들이 만든 것](06-12-qwen38-community-builds.md)
- [06-13. Qwen3.8 serving 레시피](06-13-qwen38-serving-recipes.md)
- [06-14. Qwen3.8 에이전트와 멀티 Spark](06-14-qwen38-agents-clusters.md)
