#!/usr/bin/env python3
"""Check OpenAI-compatible tool calling without third-party dependencies.

The test intentionally uses an explicit function choice. A successful HTTP
response alone is not enough: the response must contain the requested tool,
and its arguments must be valid JSON with the expected field.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def request_json(url, payload, api_key="", timeout=180):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tool_definition(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Return the weather lookup location. Do not execute a real network request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country or region, for example Seoul, South Korea.",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }


def validate_response(response, expected_name):
    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    calls = message.get("tool_calls") or []
    names = []
    valid_arguments = []
    locations = []

    for call in calls:
        function = call.get("function") or {}
        name = function.get("name")
        names.append(name)
        arguments = function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            parsed = None
        is_valid = isinstance(parsed, dict) and isinstance(parsed.get("location"), str)
        valid_arguments.append(is_valid)
        if is_valid:
            locations.append(parsed["location"])

    valid_tool_call = (
        len(calls) >= 1
        and expected_name in names
        and all(valid_arguments)
    )
    return {
        "valid_tool_call": valid_tool_call,
        "tool_call_count": len(calls),
        "tool_names": names,
        "valid_arguments": valid_arguments,
        "locations": locations,
        "finish_reason": (choices[0].get("finish_reason") if choices else None),
        "content_present": bool(message.get("content")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8083/v1")
    parser.add_argument("--model", default="qwen3.8-27b-obliterated")
    parser.add_argument("--tool-name", default="get_weather")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code unless a valid tool call is observed",
    )
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "도구가 필요하면 지정된 함수를 호출하세요. 실제 네트워크 요청은 하지 않습니다.",
            },
            {
                "role": "user",
                "content": "서울의 현재 날씨를 확인해야 합니다. 반드시 도구를 한 번 호출하고, 일반 문장은 쓰지 마세요.",
            },
        ],
        "tools": [tool_definition(args.tool_name)],
        "tool_choice": {"type": "function", "function": {"name": args.tool_name}},
        "parallel_tool_calls": False,
        "max_tokens": 128,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    result = {
        "test": "explicit_tool_call",
        "base_url": args.base_url,
        "model": args.model,
        "expected_tool": args.tool_name,
    }
    started = time.perf_counter()
    try:
        response = request_json(
            args.base_url.rstrip("/") + "/chat/completions",
            payload,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        result.update(validate_response(response, args.tool_name))
        result["ok"] = True
    except urllib.error.HTTPError as exc:
        result.update({"ok": False, "status": exc.code, "error": exc.read().decode()[:1200]})
    except Exception as exc:  # Keep the diagnostic output machine-readable.
        result.update({"ok": False, "error": repr(exc)})
    result["elapsed_s"] = round(time.perf_counter() - started, 3)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and not result.get("valid_tool_call", False):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
