# 03-3. 저장 공간·포트·첫 smoke test

이 페이지는 [03-1. 첫 부팅에서 실패하지 않는 방법](03-1-first-boot-safe-environment.md)의 상세 내용입니다.

첫 smoke test는 작게 시작한다. 긴 context와 tool call을 한 번에 켜면 어느 단계에서 실패했는지 알 수 없다.

## 순서

```bash
df -h /
ss -ltnp
curl -fsS http://127.0.0.1:PORT/v1/models
```

`PORT`는 실제 서버 포트로 바꾼다. `/v1/models`가 응답하면 endpoint가 살아 있다는 뜻이지, 모델 품질이나 tool parser가 검증됐다는 뜻은 아니다.

그다음 짧은 텍스트 한 건을 보낸다. 응답의 `model`, `finish_reason`, usage 필드를 저장한다. 오류가 나면 요청 JSON, 서버 로그, runtime 버전을 함께 보관한다.

## 실패를 나누는 기준

- 포트 연결 실패: 프로세스·주소·방화벽 문제
- 모델 목록 실패: 서버 기동 또는 route 문제
- 생성 실패: weight·template·메모리 문제
- 구조화 출력 실패: parser·schema·chat template 문제

첫 smoke가 통과한 뒤에만 긴 문맥과 동시성 테스트로 넘어간다.
