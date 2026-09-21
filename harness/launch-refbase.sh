#!/usr/bin/env bash
# Launch the reference arm (anemll image, base DeepSeek-V4-Flash-0731 checkpoint) for the
# goal comparison against vllm-spark-0731:main-029-1rc0.
#
# The recipe is staged at a path without "sparkrun" in it on purpose: spark-launch.sh runs
# `pkill -9 -f '[s]parkrun'`, which matches any process whose command line contains that
# string, including the caller's, so invoking it with the original
# ~/tonyd2wild/sparkrun/<recipe>.yaml path kills the invoking shell.
set -uo pipefail
cd "$HOME/vllm-spark-0731"
docker rm -f vllm-ds4-0731 >/dev/null 2>&1
ssh spark2 "docker rm -f vllm-ds4-0731 >/dev/null 2>&1"
timeout 150 scripts/spark-launch.sh "$HOME/goal/ref-base0731.yaml" "$HOME/refbase-run.log"
echo "launcher rc=$?"
