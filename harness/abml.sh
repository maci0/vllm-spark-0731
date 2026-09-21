#!/usr/bin/env bash
# Interleaved A/B: the reference's max_model_len (262144) vs our 65536, on our
# default. Order current,match,match,current cancels drift. 3 passes each.
# This closes the last unmeasured serving-config axis that differs from the
# reference (everything else already matches or is validated: k, MoE, WO,
# capture sizes, NCCL, util).
cd "$HOME/vllm-spark-0731"
docker rm -f $(docker ps -aq) >/dev/null 2>&1
ssh spark2 "docker rm -f \$(docker ps -aq) >/dev/null 2>&1"
MAX_MODEL_LEN=65536  PASSES=3 bash harness/run-arm.sh abml-a
MAX_MODEL_LEN=262144 PASSES=3 bash harness/run-arm.sh abml-b
MAX_MODEL_LEN=262144 PASSES=3 bash harness/run-arm.sh abml-c
MAX_MODEL_LEN=65536  PASSES=3 bash harness/run-arm.sh abml-d
echo "== ABML DONE"
