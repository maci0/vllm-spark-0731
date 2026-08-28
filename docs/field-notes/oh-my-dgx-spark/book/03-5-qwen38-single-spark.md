# 03-5. Qwen3.8 단일 Spark

이 페이지는 [03-4. 첫 모델을 올리고 “된다”를 증명하는 방법](03-4-single-spark-first-model.md)의 상세 내용입니다.

단일 Spark에서 Qwen3.8을 시작할 때는 먼저 BF16 baseline과 endpoint를 확인한다. 양자화와 speculative decoding은 그 다음 변수다.

## 시작점

```bash
scripts/run-qwen38-vllm.sh
```

현재 runner는 vLLM, `max-model-len=32768`, `max-num-seqs=4`, Qwen reasoning parser와 `qwen3_xml` tool parser를 함께 설정한다. 로컬 모델 경로가 다르면 `QWEN38_MODEL_PATH`로 명시한다.

```bash
curl http://127.0.0.1:8083/v1/models
python3 tests/qwen38_smoke.py \
  --base-url http://127.0.0.1:8083/v1 \
  --model qwen3.8-27b-obliterated
```

## 기록할 것

모델 파일이 메모리에 올라온 것과 실제 생성·tool call이 된 것을 분리한다. `loaded`, `generates`, `serves`, `tool-tested` 중 어느 상태인지 결과에 적는다.

이 페이지의 숫자는 특정 local snapshot과 runtime의 결과다. 다른 Qwen3.8 파생 모델이나 다른 vLLM image의 보장값으로 일반화하지 않는다.
