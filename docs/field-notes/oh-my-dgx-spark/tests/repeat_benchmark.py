#!/usr/bin/env python3
"""Run a small, fixed-prompt benchmark against an OpenAI-compatible endpoint.

This is a measurement harness, not a tokenizer or a quality benchmark. With
streaming it reports time to first non-empty delta and end-to-end completion
throughput. It never labels wall-clock completion throughput as pure decode
throughput. The server's usage object must provide token counts for token/s
values; otherwise those fields remain null.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def headers(api_key):
    result = {"Content-Type": "application/json"}
    if api_key:
        result["Authorization"] = "Bearer " + api_key
    return result


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def usage_metrics(usage):
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return prompt_tokens, completion_tokens


def stream_request(url, payload, api_key, timeout):
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers(api_key),
    )
    first_token = None
    usage = {}
    text_parts = []
    finish_reason = None
    events = 0

    with urllib.request.urlopen(request, timeout=timeout) as response:
        while True:
            line = response.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded.startswith("data:"):
                continue
            data = decoded[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            events += 1
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if content or reasoning:
                    if first_token is None:
                        first_token = time.perf_counter()
                    text_parts.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")

    elapsed = time.perf_counter() - started
    prompt_tokens, completion_tokens = usage_metrics(usage)
    return {
        "ok": True,
        "elapsed_s": round(elapsed, 6),
        "ttft_s": round(first_token - started, 6) if first_token is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tok_s_e2e": (
            round(completion_tokens / elapsed, 3)
            if completion_tokens is not None and elapsed > 0
            else None
        ),
        "finish_reason": finish_reason,
        "stream_events": events,
        "sample": "".join(text_parts)[:160],
    }


def non_stream_request(url, payload, api_key, timeout):
    started = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers(api_key),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    usage = body.get("usage") or {}
    prompt_tokens, completion_tokens = usage_metrics(usage)
    choices = body.get("choices") or []
    return {
        "ok": True,
        "elapsed_s": round(elapsed, 6),
        "ttft_s": None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tok_s_e2e": (
            round(completion_tokens / elapsed, 3)
            if completion_tokens is not None and elapsed > 0
            else None
        ),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "stream_events": None,
        "sample": ((choices[0].get("message") or {}).get("content") or "")[:160]
        if choices
        else "",
    }


def run_one(args, prompt, index):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    if args.stream:
        payload.update({"stream": True, "stream_options": {"include_usage": True}})
    else:
        payload["stream"] = False

    try:
        if args.stream:
            result = stream_request(
                args.base_url.rstrip("/") + "/chat/completions",
                payload,
                args.api_key,
                args.timeout,
            )
        else:
            result = non_stream_request(
                args.base_url.rstrip("/") + "/chat/completions",
                payload,
                args.api_key,
                args.timeout,
            )
    except urllib.error.HTTPError as exc:
        result = {"ok": False, "status": exc.code, "error": exc.read().decode()[:1200]}
    except Exception as exc:
        result = {"ok": False, "error": repr(exc)}
    result["trial"] = index
    return result


def load_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as prompt_file:
            return prompt_file.read().strip()
    return args.prompt


def run_batch(args, prompt, start_index):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(
            executor.map(
                lambda index: run_one(args, prompt, index),
                range(start_index, start_index + args.concurrency),
            )
        )
    return results, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8083/v1")
    parser.add_argument("--model", default="qwen3.8-27b-obliterated")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    if args.prompt and args.prompt_file:
        parser.error("--prompt와 --prompt-file은 함께 사용할 수 없습니다.")
    if args.max_tokens <= 0 or args.warmups < 0 or args.trials <= 0 or args.concurrency <= 0:
        parser.error("max-tokens, trials, concurrency는 양수이고 warmups는 0 이상이어야 합니다.")
    prompt = load_prompt(args) if args.prompt_file or args.prompt else "한국어로 핵심만 설명하세요: DGX Spark에서 prefill과 decode의 차이는 무엇인가요?"

    warmup_results = []
    for index in range(args.warmups):
        warmup_results.append(run_one(args, prompt, index + 1))

    measured = []
    batch_walls = []
    next_index = args.warmups + 1
    for _ in range(args.trials):
        batch, wall = run_batch(args, prompt, next_index)
        measured.extend(batch)
        batch_walls.append(wall)
        next_index += args.concurrency

    successful = [item for item in measured if item.get("ok")]
    ttft_ms = [item["ttft_s"] * 1000 for item in successful if item.get("ttft_s") is not None]
    e2e_tok_s = [item["completion_tok_s_e2e"] for item in successful if item.get("completion_tok_s_e2e") is not None]
    completion_tokens = [item["completion_tokens"] for item in successful if item.get("completion_tokens") is not None]
    aggregate_tok_s = None
    if completion_tokens and batch_walls and len(completion_tokens) == len(measured):
        aggregate_tok_s = round(sum(completion_tokens) / sum(batch_walls), 3)

    report = {
        "measurement": "streaming_ttft_and_end_to_end_completion",
        "warning": "completion_tok_s_e2e includes prefill and decode; it is not pure decode throughput.",
        "base_url": args.base_url,
        "model": args.model,
        "prompt_chars": len(prompt),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "stream": args.stream,
        "warmups": args.warmups,
        "trials": args.trials,
        "concurrency": args.concurrency,
        "warmup_results": warmup_results,
        "results": measured,
        "summary": {
            "successful": len(successful),
            "failed": len(measured) - len(successful),
            "ttft_p50_ms": round(percentile(ttft_ms, 50), 3) if ttft_ms else None,
            "ttft_p95_ms": round(percentile(ttft_ms, 95), 3) if ttft_ms else None,
            "completion_tok_s_e2e_p50": round(percentile(e2e_tok_s, 50), 3) if e2e_tok_s else None,
            "completion_tok_s_e2e_p95": round(percentile(e2e_tok_s, 95), 3) if e2e_tok_s else None,
            "aggregate_completion_tok_s_e2e": aggregate_tok_s,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(successful) == len(measured) else 1


if __name__ == "__main__":
    sys.exit(main())
