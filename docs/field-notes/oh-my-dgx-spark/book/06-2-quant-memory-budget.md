# 06-2. 양자화와 메모리 예산

이 페이지는 [06-1. 양자화·KV cache·speculative decoding](06-1-quantization-speculative.md)의 상세 내용입니다.

양자화는 weight만 줄이는 작업이 아니다. KV cache, runtime workspace, CUDA graph, draft model과 여유 메모리까지 합쳐야 실제 예산이 나온다.

## 계산 순서

1. 모델 weight 크기를 확인한다.
2. context와 KV dtype을 정한다.
3. batch와 동시성을 정한다.
4. runtime workspace와 여유분을 남긴다.
5. 실제 load와 long-context를 따로 확인한다.

NVFP4, FP8, AWQ, GGUF, EXL3는 이름만으로 품질과 속도를 비교할 수 없다. 같은 모델 계열이라도 kernel과 runtime이 다르면 결과가 달라진다.

## 판정 문장

“weight가 올라갔다”는 `loaded`다. “요청을 처리한다”는 `generates` 또는 `serves`다. “목표 context에서 안정적이다”는 별도 long-context 테스트가 통과해야 쓸 수 있다.
