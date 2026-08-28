# DGX Spark 리서치 기록 — Issue #39 — 2026-08-25

## 메타데이터

- 원본 Issue: [Issue #39](https://github.com/recrack/oh-my-dgx-spark/issues/39)
- 분석 기준일: `2026-08-25`
- 수집 후보 수(이슈 원문 목록 기준): `47`
- 분석 실행기: `GitHub Copilot CLI`
- 요청 모델: `gpt-5.6-terra`
- 현재 상태: `분석`
- 본문 승격: `승격 대기`

## 결론

- 종합 판정: `분석`. 이번 수집에는 Qwen3.8-27B의 조건이 갖춰진 단일 커뮤니티 벤치마크와 일부 구현 이슈가 있다. 그러나 독립 재현, 공식 지원 범위, 장시간 안정성은 확인되지 않아 안정 본문으로 즉시 승격할 항목은 없다.
- 승격 가능한 항목: 없다. Qwen3.8-27B의 양자화·speculative decoding·TP 조합은 재현 실험의 후보로 기록할 수 있다.
- 아직 확정하지 않는 항목: 커뮤니티의 tok/s, DSpark와 DFlash2의 성능·품질·안정성, multi-Spark 확장 효과, vLLM main의 DFlash2 로드 회귀, NemoClaw의 DGX Spark 전용 장애는 모두 `재현 대기` 또는 `교차 확인 필요` 상태이다.

## 확인된 사실

- [NVIDIA Developer Forum의 단일 게시물](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102)은 Qwen3.8-27B를 1~4대 DGX Spark에서 11개 구성으로 측정했다고 명시한다. 작성자가 적은 장비 조건은 GB10, 노드당 128GB UMA, CX-7 100Gbps 연결이고, 측정은 128 input tokens, 128 output tokens, 동시성별 10회이다. 이는 공식 벤치마크가 아니라 공개된 단일 작성자의 측정 조건이다.
- 같은 게시물에서 C1의 최고 결과는 RadixArk NVFP4 모델, DSpark draft, SGLang, TP=4에서 77.3 tok/s와 TTFT 227ms이다. 해당 결과의 모델은 `Qwen3.8-27B`, 양자화는 NVFP4, 런타임은 SGLang, 노드 수는 4, context 상한은 게시된 레시피 기준 262,144 tokens이다. 측정 동시성은 C1이며, 품질·tool call·장시간 안정성 결과는 이 수치만으로 확인되지 않는다.
- 같은 게시물은 vLLM 공식 이미지 경로에서 FP8 모델에 MTP 3 tokens, TP=1을 적용한 C1 결과로 17.1 tok/s와 TTFT 392ms를 제시한다. 모델은 `Qwen/Qwen3.8-27B-FP8`, KV cache dtype은 FP8이며, C1~C4가 작성자의 통과 기준(TTFT 1,000ms 미만, TPS 15 이상)을 만족했다고 기록되어 있다. 이 기준은 작성자가 정한 서비스 기준이지 보편적 합격선이 아니다.
- [vLLM issue #53428](https://github.com/vllm-project/vllm/issues/53428)의 Issue 본문 요약은 vLLM main에서 DFlash2 draft checkpoint가 로드되지 않는 사례를 제시한다. 보고 환경은 RTX PRO 5000 Blackwell이며 DGX Spark 결과가 아니므로, DGX Spark 호환성 결론으로 일반화할 수 없다. 원문 페이지는 이 실행에서 GitHub 접근 제한으로 상세 본문을 다시 검증하지 못했다.
- [NemoClaw issue #9584](https://github.com/NVIDIA/NemoClaw/issues/9584)의 Issue 본문 요약은 DGX Spark에서 interactive managed llama.cpp onboard 뒤 `doctor`와 `destroy`가 persisted authority 불일치 오류로 실패하는 보고를 제시한다. 비대화형 설치 경로에서는 재현되지 않았다고 적혀 있다. 이는 NVIDIA 저장소의 공개 버그 보고이며, 수정·배포 상태는 이번 조사에서 확인하지 못했다.

## 커뮤니티 주장

- `커뮤니티 보고 · 재현 대기`: 위 Qwen3.8-27B 게시물의 77.3 tok/s는 모델 버전, 양자화, 런타임, TP=4, 262K context 상한, C1, 128/128-token·10-round 측정법이 제시되어 있어 비교 후보로는 사용할 수 있다. 단일 게시물이며 image digest, GPU/host software revision, acceptance rate, 품질 게이트와 재현 로그가 없으므로 책의 대표 성능값으로 쓰지 않는다. [원문](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102)
- `커뮤니티 보고 · 재현 대기`: 단일 Spark에서 Qwen3.8-27B NVFP4와 DFlash2/SGLang을 사용해 약 104GB unified memory를 썼다는 주장이 있다. tok/s, 모델 정확한 revision, context, 측정 방법이 없어 메모리 예산이나 성능 근거로 승격하지 않는다. [원문](https://forums.developer.nvidia.com/t/1x-spark-new-workhorse-dflash2-sglang-is-fairly-fast-for-qwen3-8-27b-nvfp4/381142)
- `커뮤니티 보고 · 재현 대기`: llama.cpp에서 DeepSeek V4 Flash 0731의 batch/ubatch 조합을 비교한 Reddit 게시물은 GB10 128GB UMA, 4,143-token 입력 조건을 밝힌다. 다만 모델 파일·양자화·llama.cpp revision·동시성·정확한 측정 절차가 충분하지 않아 제시된 PP/TG 수치를 일반 성능으로 사용하지 않는다. [원문](https://www.reddit.com/r/LocalLLaMA/comments/1vx2i7k/benchmark_llamacpp_batchubatch_impacts_on_pp_and)
- `커뮤니티 보고 · 재현 대기`: unified-memory OOM이 host SSH와 gateway까지 불응 상태로 만들었다는 보고가 있다. 보고는 128GB DGX Spark에서 추가 benchmark 프로세스를 같은 engine container 안에서 시작한 상황을 설명하지만, cgroup·메모리 제한·kernel log·재현 절차의 교차 확인이 없다. [원문](https://github.com/letsinferlabs/letsinfer/issues/110)

## 충돌·미확인 내용

- Qwen3.8-27B 성능은 비교 대상의 양자화, speculative method, engine, TP와 동시성이 동시에 달라진다. 예를 들어 포럼의 77.3 tok/s는 TP=4 SGLang/DSpark/NVFP4 결과이므로, TP=1 vLLM의 FP8 또는 BF16 값과 단순 비교해 어느 요소가 개선을 만들었는지 단정할 수 없다.
- DFlash2 관련 [vLLM issue #53428](https://github.com/vllm-project/vllm/issues/53428)은 main의 로드 실패를 보고하지만, 포럼 게시물은 SGLang DSpark 구성을 사용한다. 두 사실은 서로 직접 충돌하지 않으며, DFlash2라는 이름만으로 runtime·revision·draft format을 같은 것으로 취급해서는 안 된다.
- [vLLM issue #37141](https://github.com/vllm-project/vllm/issues/37141)은 DGX Spark의 NVFP4 개선을 upstream에 제안하는 기능 요청이다. 기능 요청은 지원 또는 성능 개선의 증거가 아니므로 결과가 merge·release되기 전까지 본문 근거로 사용하지 않는다.
- [FreeToken roadmap #79](https://github.com/FlashML-org/FreeToken/issues/79)의 DGX Spark aarch64, sm_121, unified-memory 지원과 multi-GPU 최적화는 계획 항목이다. 구현·테스트·릴리스 증거가 없어 지원 사실로 기록하지 않는다.

## 책 반영 제안

- `book/05-benchmark.md` 및 `book/06-6-qwen38-27b.md` 후보: “Qwen3.8-27B의 공개 수치는 양자화뿐 아니라 speculative method, runtime, TP, 동시성, prompt/output 길이에 좌우되므로 서로 다른 조합의 tok/s를 단일 모델 성능으로 합치지 않는다.” 승격 조건은 포럼 외 독립 재현 1건 이상과 동일 하니스의 C1·동시성 sweep, TTFT, 품질·tool call 결과 공개이다.
- `book/06-model-recipes.md` 후보: “Qwen3.8-27B의 FP8+MTP 및 NVFP4+DSpark는 공개 재현 후보이며, runtime과 draft·target checkpoint의 정확한 revision을 고정해야 한다.” 승격 조건은 DGX Spark에서 image digest, model revision, KV dtype, context, concurrency, 측정법을 포함한 재현이다.
- `book/09-operations.md` 후보: “unified-memory 압박 시험은 engine만이 아니라 host 제어면의 응답성과 복구 절차를 함께 관찰해야 한다.” 승격 조건은 OOM 사건의 kernel log, 메모리 제한, 재현·복구 기록을 갖춘 독립 사례 또는 공식 문서이다.
- `book/08-agents.md` 후보: managed inference onboarding은 interactive와 non-interactive 경로를 분리해 `doctor`·`destroy`까지 점검한다. 승격 조건은 [NemoClaw issue #9584](https://github.com/NVIDIA/NemoClaw/issues/9584)의 수정 상태 확인과 해당 버전에서의 재현 또는 회귀 시험이다.

## 출처 목록

- 커뮤니티 보고: [Comprehensive Qwen3.8-27B Study on DGX Sparks](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102)
- 커뮤니티 보고: [DFlash2 + SGLang on one Spark](https://forums.developer.nvidia.com/t/1x-spark-new-workhorse-dflash2-sglang-is-fairly-fast-for-qwen3-8-27b-nvfp4/381142)
- 커뮤니티 보고: [llama.cpp batch/ubatch benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1vx2i7k/benchmark_llamacpp_batchubatch_impacts_on_pp_and)
- 공개 upstream issue: [vLLM #53428: DFlash2 draft load failure](https://github.com/vllm-project/vllm/issues/53428)
- 공개 NVIDIA 저장소 issue: [NemoClaw #9584: managed llama.cpp authority mismatch](https://github.com/NVIDIA/NemoClaw/issues/9584)
- 공개 issue: [Let's Infer #110: unified-memory OOM liveness](https://github.com/letsinferlabs/letsinfer/issues/110)
- 공개 기능 요청: [vLLM #37141: DGX Spark improvements](https://github.com/vllm-project/vllm/issues/37141)

## 보류 사유 및 다음 작업

- Qwen3.8 결과는 단일 커뮤니티 보고이므로 보류한다. 같은 model revision과 prompt/output 128/128, context 상한 262K, C1~C16, 10 rounds 조건으로 vLLM FP8+MTP와 SGLang NVFP4+DSpark를 각각 재현하고, image digest·GPU/OS revision·acceptance rate·TTFT·tok/s를 남긴다.
- 성능 재현에는 tool call, structured output, 긴 context, cold/warm prefix cache, 최소 1회 soak를 포함한다. 성공한 endpoint만 비교하지 말고 실패·재시작·OOM·NCCL 로그도 기록한다.
- vLLM #53428과 NemoClaw #9584은 원문 discussion, 연결 PR, merge와 release 여부를 확인한다. 영향 버전과 수정 버전이 확인되기 전에는 일반 운영 지침으로 승격하지 않는다.
- OOM 주장은 독립 재현 전까지 사례로만 보관한다. benchmark를 engine container에 추가로 넣는 조건과 별도 host process 조건을 구분하고, `MemAvailable`, cgroup limit, OOM killer 로그, SSH·gateway health와 recovery 시간을 수집한다.
