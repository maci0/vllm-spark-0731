# DGX Spark 독자 질문과 초반 장 재설계 리서치

상태: 책 구조 설계용 리서치
기준일: **2026-08-23**

이 문서는 초반 장을 어떤 순서로 읽게 할지 정하기 위한 조사 기록입니다. 특정 모델의 성능을 확정하는 문서가 아니라, DGX Spark를 처음 접한 사람이 실제로 던지는 질문을 분류하고 그 질문에 어느 장에서 답할지 정하는 데 목적이 있습니다.

## 조사에서 확인한 사실

NVIDIA 공식 문서는 독자가 처음 만나는 문제를 하드웨어 사양보다 넓게 다룹니다.

- 첫 부팅은 화면을 연결하는 방식과 네트워크 appliance 방식으로 나뉘며, 안정적인 인터넷 연결과 초기 업데이트가 필요합니다.
- DGX Spark는 128GB unified system memory, 273GB/s 메모리 대역폭, 20코어 Arm CPU, ConnectX-7을 사용합니다.
- 공식 하드웨어 페이지는 한 대에서 최대 200B, 듀얼 구성에서 405B 모델을 지원 대상으로 설명합니다. 그러나 이 표현은 특정 모델·양자화·컨텍스트·런타임의 성공을 보장하는 벤치마크 결과가 아닙니다.
- 제공된 240W 전원 어댑터를 사용하지 않으면 성능 저하·부팅 실패·예기치 않은 종료가 생길 수 있습니다.
- UMA 환경에서는 `nvidia-smi`의 전용 GPU 메모리 표시와 일반 GPU의 메모리 표시가 다르며, CUDA가 보고하는 값만으로 실제 할당 가능 메모리를 판단하기 어렵습니다.
- 공식 benchmark playbook은 vLLM·SGLang·llama.cpp·TensorRT-LLM의 offline/online 측정, 이미지 생성과 fine-tuning을 서로 다른 작업으로 나눕니다.

## 반복해서 나타난 독자 질문

커뮤니티 글에서는 다음 질문이 반복됩니다. Reddit과 포럼의 속도·품질 수치는 직접 실측이 아니라 각 글의 주장으로 기록하고, 질문의 존재와 관심사를 확인하는 데만 사용합니다.

| 독자가 먼저 묻는 질문 | 질문이 나타난 맥락 | 책에서 답할 장 |
|---|---|---|
| 이 장비가 내 작업에 맞는가? | 가격, RTX·Mac과의 비교, 메모리 대역폭과 CUDA 생태계 논쟁 | 01장 |
| 128GB에 어떤 모델이 들어가는가? | 큰 모델의 적재 가능 여부, MoE active parameter, 양자화와 context 문제 | 02장·06장 |
| 모델 두 개를 동시에 띄울 수 있는가? | unified memory 경쟁과 load/unload, 여러 endpoint 운영 질문 | 04장·10장 |
| 한 대와 두 대는 무엇이 달라지는가? | 더 큰 모델, 긴 context, TP/RDMA와 비용의 관계 | 01장·07장 |
| 왜 같은 모델인데 8·30·60 tok/s가 모두 나오는가? | 엔진·quant·speculative decoding·prompt·동시성 차이 | 00장·07장 |
| 첫날에 무엇을 설치해야 하는가? | Docker 권한, Hugging Face token, aarch64/CUDA container, SSH와 API endpoint | 03장·04장 |
| 공식 사양의 1 PFLOP이나 200B가 실제 사용을 뜻하는가? | peak compute와 memory-bound decode, 모델 카드의 지원 표현 혼동 | 00장·01장·02장 |
| 느려졌을 때 모델 문제인지 장비 문제인지 어떻게 아는가? | 저클럭·전원·온도·메모리 압박·NCCL 장애 사례 | 09장 |
| 실제로 코딩 에이전트에 쓸 수 있는가? | 단일 stream 숫자와 tool call·context compaction·장시간 운영의 차이 | 04장·08장·10장 |
| 학습·fine-tuning용으로도 적합한가? | LoRA·qLoRA·full fine-tuning의 시간과 메모리 차이 | 01장·부록 |

## 초반 장의 새 읽기 순서

기존 순서는 하드웨어 설명을 먼저 읽게 만들었지만, 독자는 대체로 “내가 살 만한가”, “무엇부터 돌리나”, “숫자를 믿어도 되나”를 먼저 묻습니다. 따라서 초반 장을 다음 흐름으로 고정합니다.

```text
00  질문을 고르고 읽는 경로 선택
 ↓
01  DGX Spark가 내 작업에 맞는지 결정
 ↓
02  128GB·대역폭·UMA를 모델 실행 관점에서 이해
 ↓
03  첫 부팅·전원·업데이트·Docker·기록 기준점 확보
 ↓
04  첫 모델을 loaded → serves → benchmarked로 증명
```

이 순서에서 00장은 사용 설명서가 아니라 지도이고, 01장은 구매·구성 판단 장이며, 02장은 사양표를 읽는 법을 설명하는 장입니다. 03장부터 명령을 실행하고, 04장에서 처음으로 모델을 선택해 서버를 올립니다.

## 초반 장에서 의도적으로 미루는 것

- DeepSeek·Qwen·MiniMax의 세부 비교는 06장과 날짜별 리서치 서브챕터에서 다룹니다.
- vLLM·SGLang의 세부 플래그는 04장으로 보냅니다.
- tok/s 순위와 C1·agent harness는 05장·08장과 날짜별 리서치 서브챕터로 보냅니다.
- TP·RDMA·스위치의 명령어는 07장으로 보냅니다.
- 장애의 전체 runbook은 09장으로 보냅니다.

초반 장에서 모든 모델과 플래그를 소개하면 처음 읽는 사람이 “그래서 지금 무엇을 해야 하는가”를 놓치기 쉽습니다. 초반에는 선택 기준과 실패를 해석하는 기준만 제공합니다.

## 참고한 문서와 커뮤니티 사례

### 공식 문서

- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/): 전체 목차와 지원 범위
- [Initial Setup - First Boot](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html): 첫 부팅, 네트워크 appliance, 초기 업데이트
- [Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html): unified memory, 대역폭, 전원, ConnectX-7, 공식 모델 규모 표현
- [Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html): 전원 어댑터, `nvidia-smi`, UMA 메모리 보고
- [ConnectX-7 Networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html): QSFP 포트와 클러스터 연결 개념
- [NVIDIA DGX Spark User Performance Guide](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md): 엔진별 benchmark와 fine-tuning 경로

### 커뮤니티 질문과 실사용 사례

- [DGX Spark setup](https://www.reddit.com/r/LocalLLM/comments/1uc4amk/dgx_spark_setup/): 여러 모델을 동시에 적재할 수 있는지, 모델을 교체해야 하는지에 대한 질문과 답변
- [Qwen3.8-27B NVFP4 측정](https://www.reddit.com/r/LocalLLM/comments/1vvtquz/dgx_spark_qwen_38_27b_nvfp4_at_60toks_generation/): code/prose, thinking, concurrency, prefill을 분리한 수치 기록
- [Qwen3.8-FP8 lmstack 구성](https://www.reddit.com/r/LocalLLM/comments/1vvdwke/got_qwen3827bfp8_running_on_a_dgx_spark_via/): Docker·Ansible·LiteLLM·권한과 비밀값을 포함한 실제 설치 경험
- [DGX Spark가 지금도 가치가 있는가](https://www.reddit.com/r/LocalLLM/comments/1vivtcm/is_it_worth_getting_the_dgx_spark_now/): 단일 Spark와 듀얼 Spark의 용도 차이에 대한 구매 질문
- [2x DGX Sparks 실사용 Q&A](https://www.reddit.com/r/LLMDevs/comments/1uey7sn/2x_nvidia_dgx_sparks_real_world_usage_qa/): 모델, 비용, 유지보수와 다른 하드웨어를 함께 비교하는 질문
- [16x DGX Sparks 클러스터](https://www.reddit.com/r/LocalLLaMA/comments/1sz0lyk/16x_dgx_sparks_what_should_i_run/): 노드 수가 늘어날 때 스위치·대역폭·운영 문제가 커지는 사례

위 커뮤니티 링크의 수치는 이 책의 직접 측정값이 아닙니다. 이 문서에서는 독자가 무엇을 궁금해하는지와 어떤 조건을 기록해야 하는지를 확인하는 근거로만 사용합니다.
