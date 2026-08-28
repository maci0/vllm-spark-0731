# 00. 이 책을 읽는 방법

이 책은 DGX Spark에서 로컬 AI를 **설치하고, 측정하고, 운영하는 과정**을 설명합니다. 모델 이름을 나열하는 책이 아니라, 어떤 구성을 선택했으며 그 선택이 실제로 맞았는지 확인하는 책입니다.

## 먼저 결론부터 읽는다

처음부터 모든 페이지를 읽지 않아도 됩니다.

| 지금 궁금한 것 | 읽을 경로 |
|---|---|
| Spark를 사도 되는가? | [01장](01-why-dgx-spark.md) → [10장](10-decision.md) |
| 한 대에서 모델을 띄우고 싶은가? | [02장](02-gb10-architecture.md) → [03장](03-first-model.md) → [04장](04-serving.md) |
| 어떤 모델이 맞는가? | [02장](02-gb10-architecture.md) → [06장](06-model-recipes.md) |
| 인터넷의 tok/s를 믿어도 되는가? | [05장](05-benchmark.md) |
| 두 대 이상을 연결하려는가? | [07장](07-cluster.md) |
| 코딩 에이전트를 운영하려는가? | [08장](08-agents.md) |
| 갑자기 느려졌거나 멈췄는가? | [09장](09-operations.md) |

## 이 책의 문장에는 세 가지 층위가 있다

공식 사실, 다른 사람이 공개한 주장, 직접 실행한 결과를 섞으면 책이 그럴듯해 보여도 재현할 수 없습니다. 각 문장은 다음 중 하나로 분류합니다.

| 표시 | 의미 | 예시 |
|---|---|---|
| **공식 사실** | NVIDIA·OpenAI·모델 제작자의 문서에서 확인한 내용 | 128GB unified memory, API model ID |
| **공개 보고** | GitHub·포럼·Reddit 작성자가 자신의 조건에서 보고한 내용 | “47 tok/s가 나왔다” |
| **직접 실험** | 이 저장소에서 실행하고 raw 결과를 보존한 내용 | C1 median 41.358 tok/s |

공개 보고는 출처 링크를 붙여도 우리 장비의 보장값이 되지 않습니다. 직접 실험도 모델 revision, quant, runtime, context, concurrency가 달라지면 별개의 실험입니다.

## “실행된다”를 다섯 단계로 나눈다

```text
loaded → serves → generates → benchmarked → tool-tested / agent-tested
```

- `loaded`: weight가 메모리에 올라갔습니다.
- `serves`: 서버가 endpoint를 열었습니다.
- `generates`: 실제 요청에 답했습니다.
- `benchmarked`: 조건을 고정한 수치를 기록했습니다.
- `tool-tested`: tool call을 검증했습니다. `agent-tested`는 여러 턴과 오류 복구까지 확인한 경우입니다.

모델이 올라왔다는 이유만으로 에이전트에 적합하다고 기록하지 않습니다.

## 숫자를 읽는 최소 규칙

`prefill`, `decode`, `TTFT`, `single-stream`, `aggregate throughput`은 서로 다른 측정값입니다. `prefill 1,000 tok/s`와 `decode 40 tok/s`를 더하거나 평균 내서 “속도”라고 기록하지 않습니다.

모든 비교표에는 다음 조건을 남깁니다.

```text
hardware · model revision · quant · runtime/image · context
KV dtype · speculative decoding · concurrency · workload · measurement method
```

## 명령을 실행할 때

책의 명령은 운영체제, Docker 권한, 이미지 tag, 모델 접근 권한에 따라 달라질 수 있습니다. 복사하기 전에 `--help`와 레시피의 revision을 확인합니다. 토큰·비밀번호·API key는 명령과 로그에 넣지 않습니다.

이 책에서 직접 실행하지 않은 명령은 `실험용` 또는 `재현 후보`라고 표시합니다.

## 기준일과 업데이트

DGX OS, CUDA, vLLM, SGLang, 모델 weight와 가격은 바뀝니다. 원고의 기준일은 **2026-08-23**이며, 최신 값이 필요한 독자는 장의 공식 링크와 레시피 revision을 다시 확인해야 합니다.

자동 리서치는 [부록 B](appendix-b-research-log.md)와 `docs/research-issue-N_YYYY-MM-DD.md`에 기록합니다. 리서치 문서는 후보 수집 기록이며, 본문에 넣을 사실은 별도 검토를 거쳐야 합니다.

## 더 자세히 읽기

실행 상태, 성능 숫자, 증거 등급과 업데이트 규칙의 원문 설명은 [00-1. 결과 상태·증거 등급·업데이트 원칙](00-1-reading-principles.md)에 보존했습니다. 처음 읽을 때는 이 장만 읽고, 수치를 비교하거나 문서를 수정할 때 상세 페이지를 함께 보면 됩니다.

## 이 장의 체크리스트

- [ ] 내가 풀려는 작업을 한 문장으로 썼다.
- [ ] 필요한 노드 수와 허용 가능한 지연을 정했다.
- [ ] `loaded`와 `agent-tested`를 구분했다.
- [ ] 성능 숫자의 측정 조건을 확인했다.
- [ ] 공식 사실과 공개 보고를 분리해서 읽었다.
