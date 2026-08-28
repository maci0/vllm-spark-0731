# 06-7. MiniMax M2.7과 M3

전체 선택 기준은 DGX Spark에서 돌릴 모델 선택에서 확인할 수 있습니다.

기준일: **2026-08-22**

## 한 줄 결론

MiniMax M2.7과 M3는 DGX Spark에서 agentic coding과 긴 context를 시험할 수 있는 후보지만, 단일 Spark의 첫 모델로 고르기보다 공개된 2·3·4대 recipe를 조건 그대로 재현하는 편이 안전하다. 모델 크기, sparse kernel, TP divisibility, KV cache, 라이선스를 함께 확인해야 한다.

## 실행 프로필

| 항목 | 현재 원고의 판정 |
|---|---|
| 단일 Spark | 기본 선택으로 권하지 않음. weight·KV·workspace 여유와 안정적인 recipe를 먼저 확인해야 함 |
| 두 대 Spark | M2.7과 M3의 community recipe가 있는 주요 실험 구간 |
| 세·네 대 Spark | custom vLLM·GGUF RPC·NVFP4·vision 경로를 recipe별로 검증 |
| 주요 작업 | agentic coding, multi-tool search, 장기 계획·문서 편집 |
| 직접 검증 상태 | 커뮤니티 recipe와 포럼 수치를 정리한 상태이며, 이 저장소의 직접 재현값과 섞지 않음 |

M2.7·M3의 세부 사례는 이 페이지에 남기고, Qwen·DeepSeek와 무엇을 비교할지는 06-4 모델 선택의 목적별 표를 따른다.

## 3분 이해 (ELI5) — 모델 카드

MiniMax M2.7·M3는 필요한 전문가 일부만 불러 쓰는 큰 팀에 가깝다.

```text
큰 전체 모델 → token마다 일부 expert 활성화
여러 Spark   → weight·KV·통신을 함께 맞춤
```

활성 파라미터가 작아 보여도 단일 Spark에서 곧바로 안정적으로 서비스된다는 뜻은 아니다.

MiniMax는 “작은 활성 파라미터로 큰 모델을 돌리는 MoE”와 agentic coding을 결합한 계열이다. DGX Spark에서는 Qwen3.8처럼 한 대에서 바로 시작하는 모델이 아니라, 모델 크기·KV·TP divisibility·커널·통신을 함께 맞춰야 하는 후보로 보는 편이 정확하다.

## M2.7: 공식 카드와 Spark 자료

[NVIDIA의 MiniMax-M2.7-NVFP4 card](https://huggingface.co/nvidia/MiniMax-M2.7-NVFP4)는 다음을 명시한다.

- Transformer 기반 sparse MoE
- 총 230B, active 10B
- 256 local experts 중 token당 8개 활성화
- 입력 context 204,800
- text-only 입력
- coding, agent harness, dynamic tool search, office workflow 용도
- NVIDIA card의 사용 조건에는 연구·개발과 MiniMax의 non-commercial 조건이 함께 붙어 있으므로 상업 사용 전 라이선스를 확인해야 함

카드의 예시 실행은 `SGLang`, `modelopt_fp4`, `minimax-append-think`, `minimax-m2` tool parser, `flashinfer_cutlass`를 사용하고 tensor parallel size 8을 지정한다. 이 명령은 B200 예시이며 2대 Spark recipe가 아니다.

### 2대 Spark에서 확인된 사례

[NVIDIA Developer Forum의 MiniMax M2.7 recipe](https://forums.developer.nvidia.com/t/minimax-m2-7-nfvp4-recipe-benchmarks/366324)는 2× ASUS Ascent GX10, NVFP4, 약 196K context와 `tg128` 약 24.3 tok/s를 보고한다. `pp2048` 약 2,074 tok/s도 함께 제시한다.

이 사례는 “M2.7이 Spark에서 가능하다”는 강한 후보 근거지만, 다음을 의미하지 않는다.

- 모든 M2.7 quant가 2대에서 같은 결과를 낸다.
- 1대 Spark에서 weight와 KV까지 여유 있게 동작한다.
- 24.3 tok/s가 Qwen·DeepSeek와 직접 비교 가능한 값이다.
- 상업 프로젝트에 바로 사용할 수 있다.

## M3: 더 큰 multimodal·long-context 후보

[NVIDIA MiniMax-M3-DSpark card](https://huggingface.co/nvidia/MiniMax-M3-DSpark)는 MiniMax-M3의 DSpark draft head를 설명한다.

- 총 428B, active 23B
- 최대 1,048,576 context
- text·image·video workflow를 위한 DSpark 경로
- coding assistant, multimodal agent, long-context reasoning 용도

그러나 DSpark card의 존재와 DGX Spark에서의 end-to-end 성공은 다르다. 현재 확인되는 Spark 경로는 대부분 custom vLLM/SGLang/llama.cpp, 패치된 sparse-attention·MoE kernel, 또는 아직 실험적인 quant를 포함한다.

### M3의 실제 Spark 사례

자료가 빠르게 갱신되므로 아래 숫자는 모두 `community-reported`다.

| 구성 | 보고된 결과 | 해석 |
|---|---|---|
| 2× Spark, llama.cpp RPC, UD-IQ4_XS GGUF | 약 10.7 tok/s, 65K context, native tool-calling hybrid template | 안정적인 OpenAI-compatible 실험 경로. full checkpoint가 아니라 GGUF |
| 3× Spark, TP=3 custom vLLM | clean reasoning·tool calling. 관리망 1GbE가 병목 | head 수를 virtual padding하는 patch와 OOM 수정 필요 |
| 4× Spark, custom vLLM | c1 약 9~10 tok/s, c2 약 14~18, c5 약 26, prefill 약 5,000→600 tok/s | NVFP4·custom kernel·context depth에 따라 변함 |
| 4× Spark, `nvidia/MiniMax-M3-NVFP4` | 약 31 tok/s, 1M KV profile, native vision·tool calling 주장 | EAGLE3, 4-bit KV, MTU 9000/IB와 mainline bug fix 포함 |
| 2× Spark, [newer W4A16/NVFP4 KV recipe](https://forums.developer.nvidia.com/t/working-recipe-minimax-m3-nvfp4-at-tp-3-on-3x-dgx-spark-no-4th-node-the-oom-fixes/373387?page=2) | 약 36 tok/s 주장 | 최신 recipe이지만 독립 조건·재현성 확인 전 기준선 아님 |

이 표에서 31·36 tok/s를 곧바로 Qwen3.8이나 DeepSeek의 속도와 비교하지 않는다. M3는 sparse attention, EAGLE3, KV dtype, custom runtime이 결과의 일부다.

## MiniMax를 Spark에 배치하는 판단

| 구성 | 판정 | 이유 |
|---:|---|---|
| 1대 | 기본 선택 아님 | M2.7/M3의 weight·KV·workspace 여유와 stable recipe가 부족 |
| 2대 | M2.7 또는 M3 실험 후보 | M2.7 196K recipe, M3 GGUF/RPC·최신 quant 사례 존재 |
| 3대 | M3 TP=3 실험 | virtual head padding, custom kernel, 1GbE fallback 여부 확인 |
| 4대 이상 | M3 서비스 후보 | 1M·native vision recipe가 있지만 custom image·IB·bug fix를 고정해야 함 |

MiniMax를 선택하는 이유는 단순 tok/s보다 agentic coding, multi-tool search, 장기 계획·문서 편집 같은 workload다. 따라서 모델을 비교할 때 `decode tok/s` 외에 다음을 측정한다.

- tool call schema 성공률
- 긴 계획 후 실제 파일 수정 성공률
- tool 오류 후 복구
- 여러 agent 또는 team 호출에서 context 유지
- 128K·196K·1M 등 실제 context profile
- 동시 요청 aggregate와 단일 요청 latency

## parser와 endpoint

MiniMax M2.7의 NVIDIA card는 `minimax-append-think` reasoning parser와 `minimax-m2` tool parser를 예시로 든다. OpenAI-compatible endpoint가 뜨더라도 다음을 확인한다.

```text
thinking content → final content 분리
tool name → arguments JSON
assistant tool call → tool result → next turn
multi-tool / retry / stop condition
```

M3 API 문서는 OpenAI·Anthropic-compatible API와 request-level thinking control을 설명하지만, hosted API 문서의 지원 범위와 local DSpark checkpoint의 지원 범위는 분리해서 읽는다.

## 라이선스와 배포

MiniMax는 모델 버전별로 조건이 다를 수 있다. 특히 NVIDIA의 M2.7 NVFP4 card에는 NVIDIA Software and Model Evaluation license와 MiniMax non-commercial 조건이 함께 표시된다. 책이나 상용 에이전트에 넣기 전에는 원본 MiniMax 카드, NVIDIA quant card, adapter와 runtime의 조건을 모두 확인한다.

## MiniMax를 선택할 때의 결론

MiniMax M2.7은 2대 Spark에서 실제 recipe가 나온 agentic coding 후보다. M3는 1M·multimodal·DSpark가 매력적이지만, Spark에서의 안정된 실행·성능·라이선스를 별도로 검증해야 한다. 따라서 현재 구매·운영 기본값은 Qwen3.8 또는 DeepSeek로 잡고, MiniMax는 다중 Spark와 실험 예산이 있을 때 비교 endpoint로 추가하는 순서가 안전하다.

## 체크리스트

- [ ] M2.7과 M3, 원본과 NVIDIA NVFP4/DSpark artifact를 구분했다.
- [ ] B200 TP=8 예시를 Spark 명령으로 복사하지 않았다.
- [ ] 2대 recipe의 context·quant·engine·GPU 모델을 기록했다.
- [ ] `minimax-m2` parser와 실제 tool loop를 확인했다.
- [ ] M3의 1M·multimodal을 Spark에서 검증된 사실처럼 쓰지 않았다.
- [ ] 상업 사용 전 모델·quant·adapter 라이선스를 확인했다.

## 참고

- [MiniMax M2.7 NVIDIA NVFP4 model card](https://huggingface.co/nvidia/MiniMax-M2.7-NVFP4)
- [MiniMax M3 DSpark model card](https://huggingface.co/nvidia/MiniMax-M3-DSpark)
- [MiniMax 공식 모델/API 문서](https://minimax-m2.com/docs/api/models)
- [MiniMax M2.7 2× Spark 포럼 recipe](https://forums.developer.nvidia.com/t/minimax-m2-7-nfvp4-recipe-benchmarks/366324)
- [MiniMax M3 2×/4× NVFP4 forum thread](https://forums.developer.nvidia.com/t/minimax-m3-nvfp4-and-nvfp4-reap-50-for-4x-2x-dgx-sparks/373177)
- [MiniMax M3 TP=3 on 3× Spark](https://forums.developer.nvidia.com/t/minimax-m3-on-3-sparks-tp-3-is-now-working/373388)
- [MiniMax M3 4× Spark vLLM result](https://forums.developer.nvidia.com/t/successfully-serving-minimax-m3-nvfp4-on-4x-dgx-spark-with-vllm/373927)
- [MiniMax M3 1M context·native vision·4× Spark](https://forums.developer.nvidia.com/t/minimax-m3-nvfp4-1m-context-31-tok-s-native-vision-4x-dgx-spark-gb10/376979)
- [MiniMax M3 2× Spark GGUF/RPC tool-calling](https://forums.developer.nvidia.com/t/minimax3-on-2-nodes-decode-10-7-tok-s-4bits/373421)
- [MiniMax M3 최신 2×/4× serving discussion](https://forums.developer.nvidia.com/t/working-recipe-minimax-m3-nvfp4-at-tp-3-on-3x-dgx-spark-no-4th-node-the-oom-fixes/373387?page=2)
- [MiniMax 모델 리서치](https://github.com/recrack/oh-my-dgx-spark/blob/main/docs/dgx-spark-model-selection-research-2026-08.md)
