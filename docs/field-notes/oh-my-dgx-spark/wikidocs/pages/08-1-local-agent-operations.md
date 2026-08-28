# 08-1. 로컬 에이전트 운영 상세

상태: 리서치 기반 초안

로컬 에이전트는 모델 endpoint 하나만 실행하는 시스템이 아니다. 모델이 빠르게 답하는 것만큼이나 tool을 잘못 호출했을 때 피해를 제한하고, 실패한 뒤 안정적으로 작업을 이어 가는 구조가 중요하다.

## 3분 이해 (ELI5)

에이전트는 똑똑한 모델 하나가 모든 일을 혼자 하는 구조가 아니다.

```text
router → supervisor: 계획·검토
       → worker: 코드·반복 작업
       → tool: 파일·명령·외부 시스템
```

모델이 도구를 호출할 수 있다는 사실과, 도구를 안전하게 운영할 수 있다는 사실은 다르다.

## 10.1 모델 역할을 나눈다

DeepSeek V4 Flash 0731을 실제 에이전트에 연결한 커뮤니티 제작물과 통합 실패 사례는 06-9: DeepSeek V4 Flash 0731로 사람들이 만든 것에 별도로 정리했다. Qwen3.8-27B의 serving recipe와 코딩 에이전트 활용 사례는 06-12: Qwen3.8-27B로 사람들이 만든 것에 정리했다. 이 장에서는 운영 구조와 권한 경계를 다룬다.

| 역할 | 필요한 성격 | DGX Spark 배치 예시 |
|---|---|---|
| worker | 낮은 latency, 코드·분류·간단한 tool | 1대 Qwen3.6/Qwen3.8, 또는 DP endpoint |
| supervisor | 긴 context, 계획·검토·복구 | 1대 DeepSeek EXL3 또는 2대 DeepSeek TP=2 |
| vision/document | 이미지·문서·긴 입력 | 별도 multimodal endpoint 또는 3~4대 profile |
| router | 요청 분류·fallback·rate limit | CPU 서비스 또는 작은 로컬 모델 |

모든 역할에 가장 큰 모델을 배정하면 memory, latency, 비용이 함께 증가한다. 2대 Spark의 2+1 구성처럼 supervisor와 worker를 나누면 운영이 더 단순해질 수 있다.

### DS4 brain과 Qwen3.8 UI·디자인 worker를 분리하는 프로필

현재 조사에서 가장 명확한 역할 분리안은 DeepSeek V4 Flash 0731을 supervisor 또는 agent brain으로 두고 Qwen3.8-27B를 UI·디자인 작업의 worker로 두는 구성이다. 두 모델을 각각 2×Spark에서 실행한다면 전체 장비는 네 대가 되며, 두 개의 독립적인 `TP=2` 서비스로 운영한다.

```text
2×Spark A: DeepSeek V4 Flash 0731, TP=2, :8888
            optional vision shim :8899 → tiny VLM :8081 → :8888

2×Spark B: Qwen3.8-27B, TP=2, separate worker endpoint

router: planning/tool/long-context → DS4
        UI/layout/CSS/design iteration → Qwen3.8
```

라우터는 모델 이름만 보고 요청을 바꾸지 말고 작업 목적과 입력 형태를 함께 판단한다. 예를 들어 긴 계획 수립, 도구 호출, 오류 복구는 DS4로 보내고 컴포넌트 초안, 스타일 변형, 화면 구성은 Qwen3.8로 보낸다. worker가 만든 코드, diff, 결정 사항은 구조화된 결과로 supervisor에 전달해 전체 대화를 매번 복제하지 않는다.

DS4에 이미지 입력이 필요하면 [DeepSeek vision shim recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-Vision-DSpark-1M-NVFP4-KV-2x-DGX-Spark)의 `:8899` endpoint를 사용할 수 있다. 이 recipe는 DS4를 다시 배포하지 않고 별도 shim을 추가하며, 기본 `Qwen3.5-0.8B-MLX-8bit` VLM이 이미지를 caption으로 변환한다. 따라서 이 경로를 native multimodal DS4나 Qwen3.8-27B vision의 근거로 사용해서는 안 된다. OCR과 정밀한 공간 추론은 별도 VLM으로 분리한다.

Spark가 두 대뿐인 경우에는 이 프로필을 그대로 적용하지 않는다. DS4 TP=2와 Qwen3.8 TP=2를 동시에 독립적으로 운영하려면 네 대가 필요하다. 두 모델을 같은 두 대에 함께 올리는 방식은 unified memory, KV cache, TP 통신을 다시 측정해야 하므로 실험 구성으로 남겨 둔다.

## 10.2 OpenAI-compatible endpoint를 계약으로 본다

에이전트와 모델을 직접 결합하기보다 endpoint 계약을 먼저 만든다.

```text
GET  /v1/models
POST /v1/chat/completions
  - model
  - messages
  - tools
  - tool_choice
  - response_format
  - usage
```

라우터가 모델별 차이를 흡수할 수 있도록 다음 정보를 저장한다.

| 필드 | 예시 |
|---|---|
| endpoint | `http://127.0.0.1:8083/v1` |
| model id | 서버가 실제로 반환한 id |
| context limit | 실제 recipe profile |
| reasoning | on/off 기본값 |
| tool parser | 모델·엔진별 parser |
| vision | 지원 여부·입력 형식 |
| timeout | prefill과 tool loop용 별도 값 |
| fallback | worker 또는 안전한 답변 경로 |

모델 이름만 바꾸면서 parser, context, thinking 기본값을 그대로 사용하지 않는다.

## 10.3 Hermes·OpenClaw·NemoClaw를 구분한다

NVIDIA Developer Forum의 [Hermes Agent local model playbook](https://forums.developer.nvidia.com/t/new-playbook-run-hermes-agent-with-local-models/369747)과 [NemoClaw/OpenClaw 로컬 실행 안내](https://forums.developer.nvidia.com/t/build-a-secure-always-on-local-ai-agent-with-openclaw-and-nvidia-nemoclaw/366929)는 로컬 모델을 에이전트에 연결할 때 참고할 수 있는 시작점이다.

책에서는 다음을 분리해 기록한다.

| 층 | 확인할 것 |
|---|---|
| 모델 | tool schema를 이해하고 valid call을 만드는가 |
| parser | tool name·arguments를 구조화하는가 |
| agent framework | loop·memory·retry·approval을 관리하는가 |
| sandbox | 파일·shell·network 권한을 제한하는가 |
| channel | Telegram·Web·CLI 등 외부 입력을 인증하는가 |
| observability | prompt·tool·결과·비밀값을 어떻게 기록하는가 |

에이전트 프레임워크가 동작했다고 해서 모델의 tool quality까지 검증된 것은 아니다. 반대로 tool call이 실패했다고 해서 원인이 모델에만 있다고 단정할 수도 없다.

## 10.4 권한은 모델 성능과 별도 축이다

권한은 최소 권한을 기본값으로 설정한다.

```text
모델 endpoint
  → tool router
      → allowlist 검사
          → sandbox 실행
              → 결과 정제
                  → 모델에 반환
```

다음 권한은 처음부터 열지 않는다.

- 홈 디렉터리 전체 읽기·쓰기
- credential·SSH key·browser profile 접근
- 무제한 shell과 sudo
- 외부 네트워크 전체
- 임의 package 설치
- 사용자의 확인 없는 삭제·전송·게시

tool 결과에 API key, cookie, 개인 문서가 섞이면 모델 context와 로그에 함께 남을 수 있다. redaction 방식과 저장 기간을 먼저 설계한다.

## 10.5 tool parser 검증

모델별로 다음 시나리오를 최소 10회 이상 반복한다.

1. 도구 하나를 제공하고 정확한 이름을 요청한다.
2. 필수 인자와 선택 인자를 함께 검증한다.
3. 잘못된 타입·누락 인자를 일부러 넣는다.
4. tool 결과를 주입하고 후속 답변을 확인한다.
5. tool error를 반환하고 retry·fallback을 확인한다.
6. 여러 도구 중 하나를 선택하게 한다.

측정값:

```text
valid_tool_name_rate:
valid_arguments_rate:
schema_validation_rate:
unknown_tool_rate:
tool_error_recovery_rate:
max_loop_depth:
timeout_rate:
```

vLLM은 모델에 맞는 `--tool-call-parser`와 auto tool choice 설정이 필요하다. parser를 켜지 않은 현재 Qwen3.8 smoke-test 서버에서 function request가 400을 반환한 것은 모델 능력의 한계가 아니라 설정 상태의 제한이었다.

## 10.6 thinking과 agent loop

thinking on/off는 agent의 역할에 따라 다르게 설정한다.

| 상황 | 기본 방향 |
|---|---|
| 단순 worker·분류 | thinking off, 짧은 출력 |
| 코드 수정 | off로 빠른 patch 후 test 결과를 supervisor에 전달 |
| 계획·복구 | supervisor에서 선택적으로 on |
| 긴 문서 | reasoning budget과 context budget을 함께 제한 |
| tool loop | thinking token과 tool arguments를 로그에서 분리 |

thinking을 켜면 출력 속도뿐 아니라 context 사용량과 tool loop 지연도 변한다. 따라서 `thinking on`의 tok/s를 `thinking off`의 값과 같은 열에 기록하지 않는다.

## 10.7 Spark 수별 agent 배치

| 구성 | 권장 역할 배치 | 이유 |
|---|---|---|
| 1대 | Qwen worker 또는 DeepSeek supervisor 중 하나 | memory headroom과 안정성 우선 |
| 2대 | DeepSeek TP=2 supervisor 또는 worker/supervisor DP | 긴 context와 동시성 선택 |
| 3대 | DeepSeek TP=2 + worker 1대, 또는 DP=3 | 2+1 역할 분리가 운영하기 쉬움 |
| 4대 | TP=4 supervisor + 별도 CPU router, 또는 DP 서비스 | 대형 모델과 aggregate를 선택 |
| 8대 | 여러 role pool·router·observability 분리 | 클러스터 운영이 모델보다 중요 |

단일 Spark DeepSeek EXL3는 c1 deep-context profile로 본다. 44–47 tok/s라는 수치가 여러 agent의 합산 속도를 뜻하는 것은 아니다. 여러 agent를 운영하려면 c4/c8과 DP 구성을 별도로 측정한다.

## 10.8 장애와 fallback

agent가 실패하더라도 같은 요청을 무한히 재시도하지 않는다.

```text
tool validation 실패
  → arguments repair 1회
  → 더 작은 worker로 재질문
  → 사용자 승인 요청
  → 안전한 중단·로그 저장
```

모델 endpoint health와 tool 실행 health를 분리해서 확인한다. endpoint가 살아 있어도 외부 API timeout, 파일 권한, sandbox 오류가 발생할 수 있다.

## 10.9 agent benchmark

raw generation과 별도로 다음 agent 시나리오를 만든다.

- 파일 목록 읽기 후 특정 파일만 수정
- 테스트 실행 후 실패 원인 요약
- JSON API 호출 후 결과 필터링
- 잘못된 tool 결과에 대한 재시도
- context가 긴 대화에서 marker 유지
- 권한 없는 경로 접근 거부
- 외부 네트워크가 막힌 상태에서 안전한 fallback

통과 기준은 “최종 문장이 그럴듯한가”가 아니다. 파일 diff, test exit code, tool schema, 권한 위반 여부를 기계적으로 검사한다.

## 이 장의 검증 체크리스트

- [ ] worker·supervisor·vision·router 역할을 분리했다.
- [ ] endpoint·model id·context·parser 계약을 기록했다.
- [ ] tool parser와 agent framework를 별도 평가했다.
- [ ] shell·파일·network·secret 권한을 allowlist로 제한했다.
- [ ] thinking on/off와 tool loop를 별도 측정했다.
- [ ] 실패·timeout·tool error 후 fallback을 확인했다.
- [ ] raw tok/s와 agent 성공률을 같은 점수로 합치지 않았다.
- [ ] log redaction과 데이터 보존 기간을 정했다.

## 아직 모르는 것

- DeepSeek one-Spark EXL3가 실제 multi-step coding supervisor에서 보이는 성공률
- Qwen3.8 worker와 DeepSeek supervisor의 최적 라우팅 정책
- Hermes/OpenClaw/NemoClaw별 tool schema·sandbox 차이
- 장시간 agent memory와 prefix cache가 함께 증가할 때의 안정성
