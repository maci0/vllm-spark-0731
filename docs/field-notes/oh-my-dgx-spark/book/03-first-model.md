# 03. 첫 부팅부터 첫 모델까지

첫날의 목표는 최고 성능이 아닙니다. **환경을 기록하고, 짧은 요청에 답하고, 그 결과를 다시 재현할 수 있게 만드는 것**이 목표입니다.

## 1. 전원과 환경을 고정한다

NVIDIA는 제공된 240W adapter를 사용하도록 안내합니다. 다른 전원 공급 장치로 부팅되더라도 성능 저하나 예기치 않은 종료가 생길 수 있으므로, 첫 실험부터 같은 adapter를 사용합니다.

첫 부팅 뒤 다음 기준점을 저장합니다.

```bash
nvidia-smi
nvcc --version
docker info
df -h
free -h
ip -br addr
uname -a
```

DGX OS release, NVIDIA driver, CUDA, Docker·NVIDIA Container Toolkit, 모델 revision, 디스크 여유 공간도 함께 기록합니다. 출력에 token이나 비밀값이 들어가지 않는지도 확인합니다.

공식 첫 부팅 절차는 [DGX Spark Initial Setup](https://docs.nvidia.com/dgx/dgx-spark/first-boot.html)을 따릅니다. 이 책의 명령은 공식 문서의 대체물이 아닙니다.

## 2. Docker와 모델 접근 권한을 확인한다

컨테이너를 실행하기 전에 다음 명령으로 GPU 접근을 확인합니다.

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

이미지 tag와 CUDA 버전은 레시피에 맞춰 고정합니다. Hugging Face gated model은 `HF_TOKEN` 또는 로그인 권한이 필요할 수 있습니다. secret을 shell history와 GitHub 로그에 남기지 않습니다.

## 3. 첫 모델은 공식 경로부터 시작한다

NVIDIA의 vLLM playbook은 DGX Spark에서 `nvidia/Qwen3.6-35B-A3B-NVFP4`를 serving과 agent-ready의 기준 예시로 제공합니다. 다만 공식 playbook의 tool-call 경로에서는 도구 목록이 클 때 malformed output이 발생했다는 이슈가 보고되었습니다. 따라서 “agent-ready”라는 이름만으로 production tool loop를 통과했다고 기록하지 않습니다. playbook의 이미지·flag·revision을 확인한 뒤, 짧은 요청과 실제 도구 목록을 별도로 시험합니다.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "12*17만 답해라."}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

이 응답을 받으면 `generates` 단계입니다. 아직 benchmark나 tool calling을 통과한 것은 아닙니다.

공식 레시피: [NVIDIA vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/vllm).
tool-call 주의 사례: [NVIDIA playbook Issue #89](https://github.com/NVIDIA/dgx-spark-playbooks/issues/89).

## 4. 모델 서버를 판정하는 순서

| 단계 | 확인 | 실패하면 |
|---|---|---|
| loaded | model load log와 메모리 사용량 | weight·quant·권한을 확인합니다. |
| serves | `/v1/models`와 health endpoint | port·container·process를 확인합니다. |
| generates | 짧은 text 요청 | tokenizer·chat template·runtime을 확인합니다. |
| benchmarked | 고정 prompt와 반복 측정 | benchmark 조건을 먼저 고정합니다. |
| tool-tested | JSON arguments와 multi-turn 왕복 | parser·server flag·모델 출력을 분리해 봅니다. |

## 5. Qwen과 DeepSeek를 첫날에 동시에 비교하지 않는다

첫날에는 하나의 기준 구성을 통과시킨 뒤 모델을 바꿉니다. Qwen3.8, DeepSeek V4 Flash, MiniMax의 레시피는 엔진과 양자화가 다르므로, 모델 이름만 바꾸는 A/B 비교가 되지 않습니다.

다음 기록을 남기면 비교가 가능합니다.

```text
date · host · image digest · model revision · quant · runtime commit
max_model_len · max_num_seqs · KV dtype · speculative config · prompt hash
```

## 6. 첫날에 하지 않을 것

- 공식 모델 규모 표기를 근거로 무리하게 400B checkpoint부터 받지 않습니다.
- server가 뜨자마자 긴 context와 여러 stream을 동시에 켜지 않습니다.
- tool parser가 있다고 해서 tool call 성공으로 기록하지 않습니다.
- 첫 실험 중 DGX OS·driver·runtime을 함께 업데이트하지 않습니다.

## 첫날 완료 기준

- [ ] 240W adapter와 네트워크를 기록했다.
- [ ] `nvidia-smi`, `docker`, 디스크, OS 정보를 저장했다.
- [ ] 하나의 공식 또는 재현 가능한 레시피로 서버를 띄웠다.
- [ ] `/v1/models`와 짧은 chat 요청을 통과했다.
- [ ] 사용한 image와 model revision을 기록했다.

## 더 자세히 읽기

- [03-1. 첫 부팅과 안전한 기본 환경](03-1-first-boot-safe-environment.md)
- [03-2. 인벤토리와 preflight](03-2-inventory-preflight.md)
- [03-3. 저장 공간·포트·첫 smoke test](03-3-storage-port-smoke-test.md)
- [03-4. 단일 Spark 첫 모델 상세](03-4-single-spark-first-model.md)
- [03-5. Qwen3.8 단일 Spark](03-5-qwen38-single-spark.md)
- [03-6. DeepSeek V4 Flash 단일 Spark](03-6-deepseek-single-spark.md)
