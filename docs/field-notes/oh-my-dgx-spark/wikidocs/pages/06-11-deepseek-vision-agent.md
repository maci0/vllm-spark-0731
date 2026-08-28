# 06-11. DeepSeek vision shim과 에이전트

이 페이지는 06-9. DeepSeek V4 Flash 0731로 사람들이 만든 것의 상세 내용입니다.

텍스트 DeepSeek에 vision을 붙인 사례는 구현을 나눠 읽어야 한다. 기존 DS4 endpoint 앞에 caption shim을 두고 작은 VLM이 이미지를 설명하는 방식은 native multimodal model과 다르다.

## 확인할 것

- pixel이 DS4에 직접 들어가는가.
- 별도 VLM과 caption 단계가 있는가.
- 이미지 latency가 전체 agent loop에 포함되는가.
- OCR·공간 추론을 별도 평가했는가.

vision이 붙었다는 말만으로 이미지 이해 품질을 보장하지 않는다. image input, caption, final answer를 각각 저장하고 실패 위치를 표시한다.
