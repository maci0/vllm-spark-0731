# 06-8. MiniMax-H3 영상 생성

전체 선택 기준은 DGX Spark에서 돌릴 모델 선택에서 확인할 수 있습니다.

기준일: **2026-08-23**

## 한 줄 결론

MiniMax-H3는 언어 모델의 tok/s 경쟁자가 아니라, 단일 DGX Spark에서 ComfyUI로 영상과 음성을 생성하는 별도 recipe다. 공개 수치의 핵심은 token decode가 아니라 step 시간·클립 생성 시간·출력 품질이며, 현재 이 책에서는 community recipe와 재현 대기 상태로 기록한다.

## 실행 프로필

| 항목 | 현재 원고의 판정 |
|---|---|
| 장비 | 단일 DGX Spark·GB10·`sm_121`·128GB unified memory recipe |
| 실행 경로 | ComfyUI·Sol-Attn·FirstBlockCache·batched VAE·Triton kernel 조합 |
| 주요 작업 | 864×480·5초 영상과 native stereo audio 생성 |
| 성능 단위 | step 시간, clip 생성 시간, seed·출력 품질 |
| 직접 검증 상태 | 저장소의 community recipe와 보고 수치를 확인했으며, 이 책의 장비에서 설치·실행한 결과는 아님 |

MiniMax-H3를 DeepSeek·Qwen·MiniMax M2.7/M3의 언어 모델 비교표에 넣지 않는다. 모델 선택의 전체 구조는 06-4, 커널·장애 점검은 09-1과 연결한다.

## 3분 이해 (ELI5) — 모델 카드

H3는 대화용 worker가 아니라 작은 미디어 스튜디오에 가깝다.

```text
텍스트·영상·음성 입력 → Omni-DiT → 영상·음성 출력
```

그래서 언어 모델의 `tok/s` 대신 clip 생성 시간, step 시간과 출력 품질로 확인한다.

MiniMax-H3는 MiniMax M2.7·M3처럼 텍스트 에이전트에 사용하는 언어 모델이 아니다. 이 장에서 다루는 H3는 텍스트·영상·음성을 하나의 시퀀스로 처리하는 **33B audio+video Omni-DiT** 모델이며, ComfyUI에서 영상과 음성을 생성하는 경로다. 따라서 DeepSeek·Qwen·MiniMax M2.7/M3의 `tok/s` 표와 H3의 렌더 시간을 같은 순위표에 섞지 않는다.

## 저장소에서 확인한 범위

핵심 레시피는 [drowzeys/keys-SM121-Optimized-MiniMax-H3-Nvidia-Sol-Engine-Kijai-SolAttn_Triton-Single-DGX-Spark](https://github.com/drowzeys/keys-SM121-Optimized-MiniMax-H3-Nvidia-Sol-Engine-Kijai-SolAttn_Triton-Single-DGX-Spark)다. 저장소 작성자가 공개한 범위는 다음과 같다.

- 단일 NVIDIA DGX Spark·GB10·`sm_121`·128GB unified memory
- aarch64, CUDA 13, ComfyUI 0.30.1
- PyTorch 2.11.0+cu130에서 테스트했다는 설명
- 설치 스크립트가 ComfyUI, 약 41GB의 weight, 최적화 노드를 함께 준비
- 약 45GB의 여유 디스크가 필요하다는 요구사항
- `h3-fullstack.json` workflow로 864×480·5초·native stereo audio 작업을 재현

위 항목은 저장소의 recipe와 요구사항이다. 이 저장소를 이 책의 DGX Spark에서 직접 설치·실행했다는 뜻은 아니다. 현재 상태는 **community recipe / 재현 대기**로 기록한다.

## 보고된 단일 Spark 결과

저장소의 seed-matched·idle GPU 측정은 다음과 같다. 수치는 작성자가 제공한 결과이며, NVIDIA 공식 벤치마크나 이 책의 직접 측정값이 아니다.

| 구성 | 스텝 시간 | 클립 생성 시간 | stock 대비 |
|---|---:|---:|---:|
| Dense stock ComfyUI | 14.40초/step | 312.7초 | 1.00× |
| Sol-Attn만 적용 | 10.88초/step | — | 1.32× |
| FirstBlockCache만 적용 | 9.72초/step | 215.7초 | 1.45× |
| 전체 stack | **8.39초/step** | **202.5초** | **1.54×** |

별도 batched VAE decode는 chunk당 4.87초에서 2.93초로 줄었다고 보고되어 있다. “1.54×”는 5초짜리 영상 하나를 만드는 end-to-end 결과이고, 언어 모델의 decode tok/s가 아니다.

## 무엇을 최적화했는가

이 레시피는 모델 weight만 바꾼 것이 아니라, ComfyUI의 실행 경로에 여러 최적화를 묶는다.

| 구성 | 역할 | 이 레시피에서의 상태 |
|---|---|---|
| Sol-Attn | query block이 필요한 KV block을 선택하도록 attention 계산량을 줄임 | `sm_121`용 Triton 경로를 사용하도록 패치 |
| FirstBlockCache | 첫 블록의 변화가 작을 때 뒤쪽 계산을 재사용 | ComfyUI용 노드로 이식 |
| batched VAE decode | 같은 모양의 VAE tile을 묶어 launch 수를 줄임 | ComfyUI용 노드로 이식 |
| INT8 QK + TMA | Blackwell에서 QK와 메모리 이동을 최적화 | kijai Triton 경로 사용 |
| AdaLN precompute | 반복되는 AdaLN branch 계산을 미리 처리 | pruned checkpoint에 포함됐다고 설명 |

### `sm_121` 커널 결과를 먼저 확인한다

저장소는 `sm_121`에서 upstream Sol-Attn plugin이 `flex_attention` 경로를 선택할 수 있고, 이 경로가 오류를 내지 않은 채 잘못된 결과를 만들 수 있다고 경고한다. 작성자는 `flex_attention`과 SDPA의 cosine similarity가 0.92~0.97까지 내려간 사례를 제시하고, `patches/`에서 Triton 커널을 강제한다.

이 주장은 H3·특정 PyTorch/CUDA 조합의 저장소 보고다. 다른 모델과 다른 runtime에도 자동으로 적용되는 GB10 전체의 결함으로 확대하지 않는다. 설치 뒤에는 최소한 다음을 확인한다.

```text
정상 프레임·정상 음성
checkerboard 또는 반복 noise 여부
seed 고정 A/B 결과
커널 변경 전후 cosine 또는 reference 차이
idle 상태와 background workload 상태
```

H3 block이 입력 tensor를 in-place로 바꾸기 때문에 cache에서 input을 clone하지 않으면 `diff_ratio=0`으로 오판할 수 있다는 경고도 있다. 캐시가 모든 스텝을 건너뛰고 checkerboard를 출력한다면 속도보다 먼저 이 경로를 점검한다.

## 설치 경로와 주의점

저장소가 제시하는 기본 흐름은 다음과 같다.

```bash
git clone https://github.com/drowzeys/keys-SM121-Optimized-MiniMax-H3-Nvidia-Sol-Engine-Kijai-SolAttn_Triton-Single-DGX-Spark.git
cd keys-SM121-*
./install.sh
~/comfy/h3-comfy-launch.sh
```

weight를 이미 갖고 있다면 `./install.sh --skip-weights` 경로가 있다. ComfyUI에서 `~/comfy/h3-fullstack.json`을 열고 실행하거나, 저장소가 제공한 API workflow를 별도로 사용할 수 있다.

이 명령을 운영 Spark에서 그대로 실행하기 전에 다음을 확인한다.

- 설치 스크립트가 고정하는 ComfyUI·PyTorch·CUDA 버전
- weight 다운로드 용량과 디스크 위치
- `patches/`가 적용하는 파일과 원본 commit
- vendor plugin의 라이선스와 NOTICE
- 설치 스크립트가 clone하는 외부 저장소의 revision
- 서비스 포트와 ComfyUI endpoint의 외부 노출 여부

저장소 자체는 Apache-2.0을 사용하지만, 모든 외부 plugin이 같은 라이선스라는 뜻은 아니다. README는 kijai Triton 저장소에 license file이 없어 설치 과정에서 clone할 뿐 재배포하지 않는다고 설명한다. 책이나 제품에 포함할 때는 repo, weight, plugin, patch의 조건을 각각 확인한다.

또한 설치 스크립트는 GB10·`sm_121`에서만 동작하도록 검사한다. RTX 5090·`sm_120`에는 이 저장소의 패치를 그대로 적용하지 말고 upstream plugin 경로를 별도로 검토한다.

## 여러 Spark로 늘릴 때

이 레시피는 단일 클립의 지연시간을 줄이는 데 초점을 둔다. 저장소의 측정에서는 fabric이 약 5.52GB/s이고, NVIDIA의 8×GB200 NVLink 가정은 503GB/s이므로 context parallelism 2개가 단일 tuned node보다 느려졌다고 보고한다.

이 결과에서 다음 결론만 가져온다.

- Spark를 추가하면 단일 영상 latency가 자동으로 줄어들지 않는다.
- 여러 클립을 나누는 data parallel·batch throughput은 별도로 설계할 수 있다.
- Spark의 fabric 속도와 실제 collective 경로를 측정하지 않고 GB200의 scaling 수치를 복사하지 않는다.
- `5.52GB/s`는 이 저장소의 측정 조건이지 모든 CX-7·스위치 구성의 고정값이 아니다.

따라서 H3를 여러 대에 배치할 때는 먼저 한 대의 정상 출력과 재현 가능한 시간을 확보한 다음, `scripts/h3-dispatch.py` 같은 배치 분산 경로를 별도 측정한다. TP·CP를 붙여 한 클립을 나누는 것과 여러 클립을 여러 Spark에 배분하는 것은 서로 다른 실험이다.

## H3 전용 벤치마크 설계

H3는 다음 조건을 한 행으로 묶어 기록한다.

```text
model/checkpoint · weight revision · ComfyUI commit
CUDA/PyTorch/Triton · resolution · frames · audio
steps · seed · Sol-Attn · FirstBlockCache threshold
VAE tile batch · idle/contended · wall power · temperature
seconds/step · VAE time · end-to-end seconds · output quality
```

최소 비교 프로필은 다음 세 가지다.

1. 가속을 끈 dense baseline
2. Sol-Attn·cache·VAE를 하나씩 켠 단계별 결과
3. 전체 stack의 end-to-end 결과

언어 모델의 “72 tok/s” 같은 숫자와 비교하지 않고, 같은 해상도·프레임·steps·seed의 클립을 만들 때 걸린 시간과 결과 품질을 비교한다. 저장소의 “A/B에서 눈에 보이는 품질 손실 없음”은 유용한 smoke test지만, 영상·음성 품질의 독립적인 정량 평가를 대신하지 않는다.

## 모델 레시피 안에서의 역할

MiniMax M2.7·M3는 텍스트 에이전트와 tool calling을 검토하는 06-7에 남긴다. H3는 다음 용도로 분리한다.

| 목적 | 우선 경로 |
|---|---|
| 코드·문서·tool loop | DeepSeek, Qwen, MiniMax M2.7/M3 |
| 영상·음성 생성 | MiniMax-H3 + ComfyUI |
| 여러 영상 배치 처리 | H3 data-parallel dispatch부터 검토 |
| 한 영상의 단일 요청 latency | 단일 Spark 최적화부터 검증 |

현재의 결론은 “DGX Spark에서 MiniMax를 돌릴 수 있다”보다 구체적이다. **단일 Spark에서 MiniMax-H3 영상·음성 생성을 시도할 수 있는 커뮤니티 레시피가 있고, 특정 `sm_121` 커널과 ComfyUI 최적화를 함께 적용해야 한다.** 1.54× 결과는 재현 전에는 저장소 보고값으로만 취급한다.

## 체크리스트

- [ ] H3를 M2.7·M3 언어 모델과 구분했다.
- [ ] GB10·`sm_121`과 CUDA/PyTorch 버전을 고정했다.
- [ ] `flex_attention` 출력 이상 여부를 확인했다.
- [ ] dense·단계별 최적화·full stack을 같은 seed로 비교했다.
- [ ] seconds/step과 end-to-end seconds를 기록했다.
- [ ] 영상·음성 품질을 속도 숫자와 분리해 확인했다.
- [ ] 외부 plugin·weight·patch의 라이선스를 확인했다.
- [ ] 여러 Spark 구성에서 CP/TP와 batch/data parallel을 구분했다.

## 참고

- [MiniMax-H3 단일 DGX Spark 최적화 저장소](https://github.com/drowzeys/keys-SM121-Optimized-MiniMax-H3-Nvidia-Sol-Engine-Kijai-SolAttn_Triton-Single-DGX-Spark)
- [저장소의 RECIPE.md](https://github.com/drowzeys/keys-SM121-Optimized-MiniMax-H3-Nvidia-Sol-Engine-Kijai-SolAttn_Triton-Single-DGX-Spark/blob/main/RECIPE.md)
- [NVIDIA Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine)
- [kijai ComfyUI SolAttn Triton](https://github.com/kijai/ComfyUI-SolAttn_triton)
- MiniMax M2.7과 M3
