# 부록 A. 명령어와 결과 양식

이 부록은 각 장의 설명을 실행할 때 필요한 최소 명령과 기록 항목을 모은 것입니다. 명령은 레시피의 image·commit·model revision에 맞춰 다시 확인해야 합니다. secret은 명령줄과 로그에 넣지 않습니다.

## 환경 스냅샷

```bash
date -Is
uname -a
nvidia-smi
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE
nvcc --version
docker info
docker ps
free -h
df -h
ip -br addr
```

다음 텍스트를 함께 기록합니다.

```text
host · DGX OS · driver · CUDA · Docker/NVIDIA Container Toolkit
image tag/digest · model ID/revision · quant · runtime commit
```

## 서버 smoke test

```bash
curl -sS http://127.0.0.1:8000/v1/models

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL_ID",
    "messages": [{"role": "user", "content": "health check"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

이 결과는 `serves`와 `generates`를 확인하는 용도입니다. benchmark 통과나 tool call 성공으로 기록해서는 안 됩니다.

## 단일 tool call 확인

실제 파일·shell 도구를 연결하기 전에 side effect가 없는 mock tool을 사용해야 합니다.

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "lookup_status",
        "description": "Return a fixed status without changing files.",
        "parameters": {
          "type": "object",
          "properties": {"service": {"type": "string"}},
          "required": ["service"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

판정은 `schema accepted`, `valid arguments`, `mock result returned`, `final answer`를 각각 기록해야 합니다.

## 성능 결과 양식

```json
{
  "date": "YYYY-MM-DDTHH:MM:SS+09:00",
  "hardware": "1x DGX Spark",
  "model_revision": "commit-or-digest",
  "quant": "format and dtype",
  "runtime": "image digest or commit",
  "context": 0,
  "kv_dtype": "...",
  "speculative": "disabled or exact config",
  "concurrency": 1,
  "warmup": 0,
  "repetitions": 0,
  "prompt_tokens_tokenizer": 0,
  "prompt_tokens_server_usage": 0,
  "prefill_tok_s": null,
  "decode_tok_s": null,
  "ttft_ms": null,
  "wall_time_ms": null,
  "tool_loop": "not-run",
  "status": "candidate",
  "notes": ""
}
```

`prefill`, `decode`, `TTFT`, `end-to-end`와 `aggregate throughput`은 서로 다른 필드입니다. `/tokenize`와 chat usage가 다르면 두 값을 덮어쓰지 않고 함께 저장합니다.

## 두 노드 이상

다중 Spark 레시피가 요구하는 interface·환경 변수·container를 먼저 확인한 뒤, 다음과 같은 계층으로 시험합니다.

```text
link negotiated
→ IP/MTU/management reachability
→ raw transport bandwidth
→ NCCL collective
→ short TP request
→ long-context request
→ concurrency and soak
```

`NCCL_SOCKET_IFNAME`, `UCX_*` 같은 변수를 무작정 복사하지 않습니다. 레시피가 요구하는 값과 실행 로그를 남기고, 실패하면 한 항목씩 되돌려야 합니다.

## 장애 보고서 양식

```text
incident time:
host:
model / revision:
image / runtime:
clock cap:
SM clock / P-state / temperature:
GPU rail power / wall power:
context / concurrency:
prefill / decode / TTFT:
network / NCCL:
error summary:
action:
recovery result:
raw result path:
```

## 결과 상태

| 상태 | 의미 |
|---|---|
| `candidate` | 공개 자료나 첫 실행으로 재현 후보를 찾았습니다. |
| `serves` | endpoint가 열리고 모델 목록을 반환했습니다. |
| `benchmarked` | 조건과 반복 수를 고정해 수치를 저장했습니다. |
| `tool-tested` | mock 또는 허용된 도구의 multi-turn을 통과했습니다. |
| `agent-tested` | 실제 task 성공·권한·복구까지 평가했습니다. |
| `blocked` | 실패 원인과 재현 조건을 기록했고 다음 조치가 필요합니다. |

raw 결과는 `docs/` 또는 별도의 실험 저장소에 보존하고, 본문에는 상태·조건·판정과 링크만 옮깁니다.

## 더 자세히 읽기

A-1. 모델·레시피·명령어 색인에는 모델별 시작점, 노드 수별 토폴로지, 재현 명령, 결과 파일과 출처 등급을 보존했습니다.
