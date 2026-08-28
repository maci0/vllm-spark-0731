# DGX Spark에서 실행할 모델 선택 리서치

기준일: **2026-08-22**
범위: DGX Spark/GB10에서 실제 실행 경로가 확인된 모델과, 아직 특정 다중 노드 recipe가 필요한 모델을 구분한다.

## 결론

DGX Spark에서 “돌아간다”는 말은 최소 네 단계로 나눠야 한다.

| 상태 | 의미 |
|---|---|
| `loads` | weight와 runtime이 unified memory에 올라감 |
| `generates` | 고정 prompt로 정상 텍스트가 나옴 |
| `serves` | OpenAI-compatible endpoint가 반복 요청에 응답함 |
| `benchmarked` | 모델·quant·엔진·context·concurrency·측정 방법이 고정된 숫자가 있음 |
| `tool-tested` | parser와 tool schema, arguments, 오류 복구를 확인함 |
| `agent-tested` | 여러 단계의 실제 tool loop를 통과함 |

현재 책에서 가장 현실적인 역할 분담은 다음과 같다.

- **DeepSeek V4 Flash 0731**: 1대 EXL3 supervisor 후보, 2대 FP8/DSpark 긴 context·agent brain 후보
- **Qwen3.8-27B**: 1대 NVFP4/FP8 speculative serving, coding/UI worker와 여러 동시 세션 후보
- **MiniMax M2.7/M3**: 모델의 agent 지향성은 강하지만, Spark에서는 2대 이상과 모델별 kernel/TP recipe가 필요한 실험 후보
- **MiniMax-H3**: 언어 모델이 아닌 영상·음성 생성 모델로, 단일 Spark ComfyUI·`sm_121` 최적화 recipe가 나온 재현 대기 후보

이것은 모델의 절대적인 품질 순위가 아니다. 같은 모델도 quant, engine, draft model, context, prompt class, transport가 바뀌면 다른 시스템처럼 동작한다.

## 1. 노드 수별 모델 지도

| 모델 | 1대 | 2대 | 3~4대 | 책에서의 기본 역할 | 현재 증거 |
|---|---|---|---|---|---|
| DeepSeek V4 Flash 0731 | EXL3·SparkInfer·DSpark 단일 recipe | FP8·vLLM/SGLang·DSpark TP=2 | 추가 노드 구성은 recipe별 검증 | 긴 문서·supervisor·agent brain | B/C: 공식 model card + community recipe |
| Qwen3.8-27B | NVFP4/FP8/4-bit SGLang·vLLM | TP=2 또는 DP/서비스 분리 | pool·2×2 역할 분리 | coding worker·UI·JSON·동시성 | A/B/C: model card + forum/레포 |
| MiniMax M2.7 | 신뢰할 수 있는 Spark 단일 recipe 미확인 | 2× ASUS GX10 community recipe | TP=3/4는 모델별 patch 검증 필요 | agentic coding·장기 tool use | A/B/C: model card + forum |
| MiniMax M3 | 단일 Spark 경로 미확인 | 2대 GGUF/RPC·REAP 실험과 새 NVFP4 경로 | 3대 TP 또는 4대 TP recipe·custom kernel | multimodal agent·1M 후보 | A/B/C: DSpark card + forum recipes |
| MiniMax-H3 | 단일 Spark ComfyUI·Sol-Attn·FirstBlockCache recipe | 여러 클립 data parallel 우선 검토 | CP/TP는 fabric 측정 후 판단 | 영상·음성 생성 | B/C: community GitHub recipe, 직접 재현 대기 |
| Qwen3.5-122B-A10B | 특수 INT4/NVFP4 recipe | 별도 검증 | DP/추가 TP | 큰 단일 supervisor | B/C: forum recipe |
| Qwen3.5-397B | 일반적으로 무리 | 메모리·KV headroom 주의 | 4대 TP recipe | 대형 daily/supervisor | B/C: 4대 recipe |
| GPT-OSS-120B | llama.cpp 등 공식 playbook 경로 | 서비스 분리 | DP/pool | 일반 추론·기준선 | A/B |

`1대` 열의 “가능”은 모든 context와 동시성을 뜻하지 않는다. 예를 들어 DeepSeek 단일 EXL3가 정상 생성돼도 370K needle, native vision, 장시간 agent loop까지 통과했다는 뜻은 아니다.

## 2. 모델을 고르는 순서

1. 가장 먼저 필요한 작업을 정한다: 일반 채팅, 코딩, tool loop, 장문 검색, vision, 동시 사용자.
2. 모델 원본과 실제 배포 artifact를 구분한다: FP8 원본, NVFP4, EXL3, GGUF, abliterated 파생 모델은 별개다.
3. unified memory에서 weight 뒤에 남는 KV·workspace·OS headroom을 계산한다.
4. Spark 수와 topology를 고른다: single, direct TP=2, ring, switch를 섞지 않는다.
5. engine과 parser를 고정한다: vLLM·SGLang·llama.cpp·SparkInfer의 결과를 한 순위표에 넣지 않는다.
6. `serves` 후에 `benchmarked`, `tool-tested`, `agent-tested`를 따로 통과시킨다.

## 3. 공식 모델 카드와 Spark recipe의 차이

공식 모델 카드는 모델의 구조·기능·라이선스와 일반 실행 예시를 제공한다. Spark recipe는 GB10의 unified memory, SM 12.1 계열 kernel, NVFP4/FP8 경로, ConnectX-7 통신과 같은 하드웨어 조건을 해결해야 한다.

예를 들어 DeepSeek 공식 model card는 vLLM/SGLang 실행 예시와 DSpark 옵션을 제공하지만, 그 예시의 tensor/data parallel 크기와 Spark 커뮤니티 recipe의 1대·2대 설정은 동일하지 않다. Qwen 공식 카드는 native 262K와 YaRN 1M 확장을 설명하지만, 1M을 설정하는 것과 1M에서 품질·안정성을 검증하는 것은 다르다. MiniMax의 NVIDIA NVFP4 card는 B200의 TP=8 예시를 제공하며, 이를 2대 Spark 명령으로 복사할 수 없다.

## 4. 벤치마크 표준

각 모델의 결과는 다음 파일 묶음으로 저장한다.

```text
results/
  <date>-<model>-<hardware>-<profile>.json
  <date>-<model>-<hardware>-<profile>.raw.jsonl
  <date>-<model>-<hardware>-<profile>.env.txt
  <date>-<model>-<hardware>-<profile>.log
```

필수 필드:

```text
hardware, node_count, topology, vendor, model_sku
os, kernel, driver, cuda, nccl, container_digest, runtime_commit
model_repo, model_revision, quant, kv_dtype
context, prompt_tokens, output_tokens, thinking, reasoning_effort
speculation, draft_tokens, concurrency, workload
prefill_tok_s, decode_tok_s, ttft_ms, e2e_tok_s, aggregate_tok_s
transport, wall_power_w, gpu_power_w, temperature_peak_c
quality_result, tool_result, agent_result, error_rate
```

## 5. 참고 원문

### 공식 모델 카드·프로젝트

- [DeepSeek V4 Flash 0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8)
- [MiniMax M2.7 NVIDIA NVFP4 model card](https://huggingface.co/nvidia/MiniMax-M2.7-NVFP4)
- [MiniMax M3 DSpark model card](https://huggingface.co/nvidia/MiniMax-M3-DSpark)
- [MiniMax model/API guide](https://minimax-m2.com/docs/api/models)
- [NVIDIA DGX Spark playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)

### DGX Spark 실행·실측

- [DeepSeek V4 Flash one-Spark EXL3 recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
- [DeepSeek V4 Flash 2× Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- [Qwen3.8 SGLang/DFlash2 recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)
- [Qwen3.8 dual Spark NVIDIA Forum](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350)
- [MiniMax-H3 단일 DGX Spark ComfyUI recipe](https://github.com/drowzeys/keys-SM121-Optimized-MiniMax-H3-Nvidia-Sol-Engine-Kijai-SolAttn_Triton-Single-DGX-Spark)
- [MiniMax M2.7 dual Spark recipe·benchmark](https://forums.developer.nvidia.com/t/minimax-m2-7-nfvp4-recipe-benchmarks/366324)
- [MiniMax M3 2×/4× NVFP4 recipes](https://forums.developer.nvidia.com/t/minimax-m3-nvfp4-and-nvfp4-reap-50-for-4x-2x-dgx-sparks/373177)
- [MiniMax M3 TP=3 on 3× Spark](https://forums.developer.nvidia.com/t/minimax-m3-on-3-sparks-tp-3-is-now-working/373388)
- [MiniMax M3 4× Spark vLLM](https://forums.developer.nvidia.com/t/successfully-serving-minimax-m3-nvfp4-on-4x-dgx-spark-with-vllm/373927)
- [MiniMax M3 1M/native vision 4× Spark](https://forums.developer.nvidia.com/t/minimax-m3-nvfp4-1m-context-31-tok-s-native-vision-4x-dgx-spark-gb10/376979)
- [Qwen3.5-397B 4× Spark recipe](https://forums.developer.nvidia.com/t/qwen3-5-397b-a17b-int4-autoround-4-x-db10-node-updated-results-37-94-tok-s/362368)
- [이 책의 포럼 전체 리서치](dgx-spark-nvidia-forum-research-2026-08.md)
- [DeepSeek 커뮤니티 제작물](deepseek-v4-flash-0731-community-builds-2026-08.md)
- [Qwen3.8 커뮤니티 제작물](qwen38-community-builds-2026-08.md)
