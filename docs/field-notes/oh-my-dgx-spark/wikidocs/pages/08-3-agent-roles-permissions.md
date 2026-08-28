# 08-3. brain·worker·권한 분리

이 페이지는 08-1. 로컬 에이전트 운영의 상세 내용입니다.

큰 모델 하나로 모든 일을 처리할 필요는 없다. DeepSeek를 brain으로 두고 Qwen3.8을 UI·디자인·구조화 worker로 분리하는 구성처럼 역할을 나눌 수 있다.

## 역할 기준

- brain: 계획, 긴 문맥, 복잡한 판단
- worker: 짧은 변환, UI 초안, JSON, 반복 작업
- vision: 이미지·화면 caption과 OCR
- tool runner: 실제 파일·shell·네트워크 권한

모델은 권한을 갖지 않는다. 권한은 tool runner와 sandbox가 갖는다. 모델이 똑똑해져도 승인 경계, timeout, 파일 범위는 그대로 둔다.

## 운영 프로필

각 역할마다 endpoint, model name, context, timeout, fallback을 문서화한다. 한 endpoint가 느려져도 다른 역할이 모두 멈추지 않도록 분리한다.
