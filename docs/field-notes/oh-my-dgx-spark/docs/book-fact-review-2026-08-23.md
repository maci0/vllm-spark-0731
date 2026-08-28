# 책 통합본 사실성 검토 기록

기준일: **2026-08-23**

이 문서는 00~10장의 상위 읽기 경로와 기존 상세 원고를 서브챕터로 재배치한 뒤 수행한 편집 검토 기록입니다. 문장의 범위를 출처보다 넓히지 않고, 공식 사양·공개 보고·직접 실험을 구분하는 것을 목표로 합니다. 이번 2차 검토에서는 `prompts/fluent-korean.md`의 문장 지침도 함께 적용했습니다.

## 검토 범위

| 검토 영역 | 기준 출처 | 본문에 적용한 원칙 |
|---|---|---|
| DGX Spark 하드웨어 | [NVIDIA Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [System Overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html) | 128GB unified memory, 273GB/s, 20-core Arm, ConnectX-7, 240W adapter와 200B/405B 포지셔닝을 공식 사실로만 기록합니다. |
| serving baseline | [NVIDIA vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/vllm) | Qwen3.6 레시피를 실행 기준으로 기록하되, 모든 모델·모든 tool surface의 성공으로 확장하지 않습니다. |
| tool calling | [NVIDIA Issue #89](https://github.com/NVIDIA/dgx-spark-playbooks/issues/89) | 공식 레시피에 malformed tool-call 보고가 있으므로 parser flag와 실제 tool loop를 분리해 검사합니다. |
| 다중 노드 | [Connect Two Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks), [Connect Three Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-three-sparks), [switch playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch) | 두 대 direct, 세 대 ring, 네 대 이상 switch를 topology 출발점으로 쓰고 성능 보장으로 해석하지 않습니다. |
| benchmark | [NVIDIA performance guide](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md) | prefill, decode, TTFT, offline/online, concurrency를 분리합니다. |
| Sol 비교 | [GPT-5.6 Sol model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol), [Sol 비교 리서치](sol-max-comparison-research-2026-08.md) | `gpt-5.6-sol`의 API 사양과 로컬 모델의 직접 측정을 같은 benchmark로 취급하지 않습니다. |
| DeepSeek | [공식 모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), [단일 Spark 레시피](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark), [직접 결과](results-deepseek-c1-2026-08-22.json) | 모델 카드 점수, 레시피 보고값, C1 직접 결과를 각각 별도 층위로 기록합니다. |
| MCDMA·저클럭 | [Mac/RDMA 리서치](dgx-spark-mac-rdma-switch-research-2026-08.md), [clock cap harness](https://github.com/agjs/gb10-clock-cap), [커뮤니티 보고](https://x.com/Blackwellboy/status/2090611479653622261?s=20) | 커뮤니티 관찰값과 실험 프로토타입은 표준 운영 경로나 보편적 원인으로 확정하지 않습니다. |

## 장별 판정

| 장 | 사실성 판정 | 남은 주의점 |
|---|---|---|
| 00 | 통과 | 공개 숫자는 조건을 붙여야 합니다. |
| 01 | 통과 | NVIDIA의 모델 규모 표현을 성공 보장으로 읽지 않도록 제한했습니다. |
| 02 | 통과 | unified memory 용량과 실제 모델·KV headroom을 분리했습니다. |
| 03 | 통과 | 공식 Qwen 레시피를 기준 구성으로만 두고 Issue #89를 함께 표시했습니다. |
| 04 | 통과 | OpenAI-compatible endpoint의 기능 차이와 Qwen3.8 architecture 로그를 분리했습니다. |
| 05 | 통과 | C1 직접 수치와 DeepSeek 모델 카드·레시피 수치를 섞지 않았습니다. |
| 06 | 통과 | DeepSeek·Qwen·MiniMax의 모델·runtime·양자화 경로를 하나의 순위로 만들지 않았습니다. |
| 07 | 통과 | TP·PP·DP와 2×2 pool을 구분하고 MCDMA를 실험으로 제한했습니다. |
| 08 | 통과 | agent-ready 명칭과 실제 tool loop 성공을 구분했습니다. |
| 09 | 통과 | 저클럭 회복 사례와 clock cap을 커뮤니티 보고·실험 프로필로 표시했습니다. |
| 10 | 통과 | Sol max 동급 표현을 쓰지 않고 동일 harness 필요성을 명시했습니다. |

## Sol·Opus 검토에 관한 경계

이 실행 환경에서는 Sol 또는 Opus 모델을 별도로 호출하지 않았습니다. 따라서 이 기록과 원고를 Sol·Opus가 실제로 검토했다고 주장하지 않습니다. 대신 Sol Max 수준의 엄격한 검토 기준을 편집 규칙으로 삼아 다음 순서로 검토했습니다.

1. 각 수치가 링크된 원문에서 실제로 주장되는지 확인했습니다.
2. 공식 사실, 다른 사람의 보고, 이 저장소의 직접 실험, 편집자의 판단을 분리했습니다.
3. 한 조건의 수치를 다른 모델·엔진·노드 수로 일반화하는 문장을 제거했습니다.
4. 재현되지 않은 주장은 `후보`, `공개 보고`, `검증 대기`로 낮췄습니다.
5. 한국어 문장은 `prompts/fluent-korean.md`의 fluent-korean 규칙에 맞춰 주어·조사·어미, 기술 용어와 사실·해석의 경계를 다시 확인했습니다.

나중에 실제 Sol·Opus 호출 결과를 확보하면 이 문서의 “모델 검토” 항목에 별도로 기록합니다. 그 결과도 기존 출처 검증을 대체하지는 않습니다.

## 아직 확정하지 않은 주장

- DeepSeek V4 Flash 0731과 GPT-5.6 Sol의 전반적인 동급 여부
- 동일 task set에서의 local·hosted quality score
- MCDMA의 독립 재현과 Spark↔Mac 단일 TP fabric 지원
- 세 대의 일반적인 `TP=3` 성능 확장
- Qwen3.8 모델 이름과 runtime이 표시하는 architecture가 항상 일치한다는 가정

이 항목들은 본문에서 결론이 아니라 후속 검증 목록으로 남겼습니다.
