#!/usr/bin/env python3
"""Check one OpenAI-compatible tool loop with a local mock tool.

The script deliberately does not call a real weather service. It verifies the
message contract in two requests: the model emits a function call, the harness
returns a deterministic ``tool`` message, and the model produces a final
answer without requesting the tool again.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def request_json(url, payload, api_key="", timeout=240):
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
            "description": "Return a deterministic local mock weather result.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def first_tool_call(response, expected_name):
    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None, message, {
            "tool_name_ok": False,
            "arguments_json_ok": False,
        }

    call = calls[0]
    function = call.get("function") or {}
    raw_arguments = function.get("arguments")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except (TypeError, json.JSONDecodeError):
        arguments = None

    parsed = {
        "id": call.get("id"),
        "name": function.get("name"),
        "arguments": arguments,
    }
    checks = {
        "tool_name_ok": parsed["name"] == expected_name,
        "arguments_json_ok": (
            isinstance(arguments, dict) and isinstance(arguments.get("city"), str)
        ),
    }
    return parsed, message, checks


def assistant_message(message, calls):
    """Keep only standard assistant fields needed for the follow-up request."""
    result = {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": calls,
    }
    if message.get("name"):
        result["name"] = message["name"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--tool-name", default="lookup_weather")
    parser.add_argument("--city", default="Seoul")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero unless both calls and the final answer pass",
    )
    args = parser.parse_args()

    tools = [tool_definition(args.tool_name)]
    messages = [
        {
            "role": "system",
            "content": "도구가 필요하면 함수를 호출하세요. 도구 결과를 받은 뒤 한국어 한 문장으로 답하세요.",
        },
        {
            "role": "user",
            "content": f"{args.city}의 날씨를 확인하고 섭씨 온도까지 알려주세요.",
        },
    ]
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    def call(request_messages, tool_choice):
        payload = {
            "model": args.model,
            "messages": request_messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
            "max_tokens": 256,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        response = request_json(endpoint, payload, args.api_key, args.timeout)
        return response, time.perf_counter() - started

    result = {
        "test": "openai_compatible_tool_loop_mock",
        "base_url": args.base_url,
        "model": args.model,
        "tool": args.tool_name,
        "city": args.city,
        "settings": {"thinking": False, "temperature": 0, "max_tokens": 256},
    }
    started = time.perf_counter()
    try:
        first, first_elapsed = call(
            messages,
            {"type": "function", "function": {"name": args.tool_name}},
        )
        choices = first.get("choices") or []
        choice = choices[0] if choices else {}
        calls = (choice.get("message") or {}).get("tool_calls") or []
        parsed, first_message, first_checks = first_tool_call(first, args.tool_name)
        result["first_call"] = {
            "elapsed_s": round(first_elapsed, 3),
            "finish_reason": choice.get("finish_reason"),
            "tool_call": parsed,
            "usage": first.get("usage"),
            "checks": first_checks,
        }

        mock_result = None
        if parsed and first_checks["tool_name_ok"] and first_checks["arguments_json_ok"]:
            mock_result = {
                "city": parsed["arguments"]["city"],
                "condition": "맑음",
                "temperature_c": 24,
            }
        result["mock_tool"] = mock_result

        followup = list(messages)
        followup.append(assistant_message(first_message, calls))
        if mock_result is not None:
            followup.append(
                {
                    "role": "tool",
                    "tool_call_id": parsed["id"],
                    "name": args.tool_name,
                    "content": json.dumps(mock_result, ensure_ascii=False),
                }
            )

        second = None
        second_elapsed = None
        second_message = {}
        if mock_result is not None:
            second, second_elapsed = call(followup, "auto")
            second_choices = second.get("choices") or []
            second_choice = second_choices[0] if second_choices else {}
            second_message = second_choice.get("message") or {}
            final_content = second_message.get("content") or ""
            result["second_call"] = {
                "elapsed_s": round(second_elapsed, 3),
                "finish_reason": second_choice.get("finish_reason"),
                "content": final_content,
                "tool_calls": bool(second_message.get("tool_calls")),
                "usage": second.get("usage"),
            }
        else:
            result["second_call"] = {"executed": False}

        final_content = (second_message.get("content") or "").strip()
        checks = {
            "first_tool_name_ok": first_checks["tool_name_ok"],
            "first_arguments_json_ok": first_checks["arguments_json_ok"],
            "mock_tool_executed": mock_result is not None,
            "second_final_content_ok": bool(final_content),
            "second_loop_finished_without_tool": bool(final_content)
            and not second_message.get("tool_calls"),
        }
        result["checks"] = checks
        result["ok"] = all(checks.values())
    except urllib.error.HTTPError as exc:
        result.update({"ok": False, "error": f"HTTP {exc.code}: {exc.read().decode()[:1200]}"})
    except Exception as exc:  # Keep diagnostics machine-readable.
        result.update({"ok": False, "error": repr(exc)})
    result["total_elapsed_s"] = round(time.perf_counter() - started, 3)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and not result.get("ok", False):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
