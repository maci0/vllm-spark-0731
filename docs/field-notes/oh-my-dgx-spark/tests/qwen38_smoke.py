#!/usr/bin/env python3
"""Safe, dependency-free smoke tests for an OpenAI-compatible vLLM endpoint."""

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def request_json(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())


def chat(args, messages, max_tokens=128, extra=None, timeout=180):
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "repetition_penalty": 1.15,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if extra:
        payload.update(extra)
    started = time.perf_counter()
    try:
        response = request_json(args.base_url.rstrip("/") + "/chat/completions", payload, timeout)
        elapsed = time.perf_counter() - started
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        usage = response.get("usage") or {}
        return {
            "ok": True,
            "elapsed_s": round(elapsed, 3),
            "content": content[:800],
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "reasoning_present": bool(message.get("reasoning") or message.get("reasoning_content")),
        }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode()[:1000]}
    except Exception as exc:  # Keep the remaining smoke tests running.
        return {"ok": False, "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8083/v1")
    parser.add_argument("--model", default="qwen3.8-27b-obliterated")
    parser.add_argument("--image", help="Optional local JPEG/PNG for a vision smoke test")
    parser.add_argument("--long-context-tokens", type=int, default=8192)
    args = parser.parse_args()
    results = []

    try:
        models = request_json(args.base_url.rstrip("/") + "/models", timeout=10)
        results.append({"test": "models", "ok": True, "models": [m.get("id") for m in models.get("data", [])]})
    except Exception as exc:
        results.append({"test": "models", "ok": False, "error": repr(exc)})

    cases = [
        ("korean_math", [{"role": "user", "content": "한국어로 짧게 답하세요. 2 더하기 2는 얼마인가요?"}], 64, {}),
        ("python_code", [{"role": "user", "content": "파이썬으로 리스트 중복 제거 함수를 코드 블록 하나로만 작성하세요."}], 160, {}),
        (
            "json",
            [{"role": "user", "content": "JSON 하나만 출력하세요: name Ada, age 36, skills Python과 math."}],
            128,
            {"response_format": {"type": "json_object"}},
        ),
        (
            "conversation_recall",
            [
                {"role": "user", "content": "기억할 식별자는 ALPHA-7429 입니다."},
                {"role": "assistant", "content": "기억했습니다."},
                {"role": "user", "content": "식별자만 출력하세요."},
            ],
            64,
            {},
        ),
        ("thinking_on", [{"role": "user", "content": "17 곱하기 19의 결과를 계산하고 답하세요."}], 192, {"chat_template_kwargs": {"enable_thinking": True}}),
    ]

    for name, messages, max_tokens, extra in cases:
        result = chat(args, messages, max_tokens, extra)
        result["test"] = name
        if name == "json" and result.get("ok"):
            try:
                json.loads(result["content"])
                result["valid_json"] = True
            except json.JSONDecodeError:
                result["valid_json"] = False
        results.append(result)

    line = "Neutral context-window test line. It contains ordinary filler text and no instructions.\n"
    marker = "CONTEXT-NEEDLE-314159"
    repeats = max(1, int(args.long_context_tokens / 16))
    body = line * repeats + "\nThe verification marker is " + marker + ".\nOutput only the marker."
    result = chat(args, [{"role": "user", "content": body}], 32)
    result["test"] = "long_context"
    results.append(result)

    if args.image:
        with open(args.image, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "이미지를 한 문장으로 설명하세요. 보이는 것만 말하세요."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
        ]}]
        result = chat(args, messages, 96)
        result["test"] = "image"
        results.append(result)

    def concurrent(index):
        result = chat(args, [{"role": "user", "content": f"요청 번호 {index}를 확인했다고 한 문장으로 답하세요."}], 48)
        result["request_id"] = index
        return result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(concurrent, index) for index in range(1, 5)]
        concurrent_results = [future.result() for future in as_completed(futures)]
    results.append({
        "test": "concurrency_4",
        "ok": all(item.get("ok") for item in concurrent_results),
        "wall_s": round(time.perf_counter() - started, 3),
        "results": concurrent_results,
    })

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

