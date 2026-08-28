# 03-6. DeepSeek V4 Flash 단일 Spark

이 페이지는 03-4. 첫 모델을 올리고 “된다”를 증명하는 방법의 상세 내용입니다.

DeepSeek V4 Flash 0731을 단일 Spark에서 실행하는 recipe는 모델 이름만으로 재현할 수 없다. EXL3 bpw, REAP expert 구성, SparkInfer·DSpark 버전, KV dtype, context와 concurrency를 함께 고정해야 한다.

## 확인 순서

1. [MiaAI-Lab recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)가 가리키는 현재 commit을 기록한다.
2. weight 파일과 양자화 프로필을 확인한다.
3. server가 실제로 사용하는 model name과 context를 확인한다.
4. 짧은 생성, 긴 context, needle을 각각 별도 결과로 남긴다.

공개 recipe의 `44~47 tok/s`, `384K context`, `370K needle`은 하나의 상시 보장값이 아니다. prefill, decode, long-context stress를 서로 다른 행으로 기록한다.

## 최소 결과 형식

`weight · bpw · KV dtype · context · draft model · single stream · prompt tokens · output tok/s · exact recall`

이 중 하나라도 빠지면 “DeepSeek 속도”가 아니라 “조건이 빠진 사례”로 표시한다.
