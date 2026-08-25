# Audit Plan: vllm-spark-0731 Architecture, Papers, and Codebase Audit

**Slug:** `vllm-spark-0731-docs`  
**Date:** 2026-08-25  
**Target Architecture & Model:** DeepSeek-V4-Flash-0731 (`deepseek-ai/DeepSeek-V4-Flash-0731`), DeepSeek-V3 / DeepSeek-V4 technical architecture (MLA, DSA Lightning Indexer with ReLU scoring, Multi-Token Prediction / DSpark $k=5$).  
**Target Hardware Platform:** 2× NVIDIA DGX Spark (Grace Blackwell GB10 SoC, SM121 / SM family 120, 128 GiB coherent LPDDR5x UMA per node, ConnectX-7 RoCE v2, TP=2).  
**Target Codebase:** `vllm-spark-0731` (`configs/`, `patches/`, `scripts/`, `docker/`, `docs/`, `tests/`, `README.md`, `HANDOFF.md`).  

---

## 1. Objectives & Scope

Conduct a rigorous audit comparing claimed methods, defaults, metrics, and data handling against actual code and hardware execution reality across:
1. **DeepSeek Architectural Papers & Specifications:** Multi-Head Latent Attention (MLA), DeepSeek Sparse Attention (DSA) non-linear ReLU scoring vs linear weighted-Q contraction, and Multi-Token Prediction (MTP) / DSpark $k=5$ speculative decoding.
2. **NVIDIA GB10 Grace Blackwell Superchip Official Datasheet Specifications:** 48 SMs, 6,144 CUDA cores, 5th generation Tensor Cores (native FP4/FP8/BF16/TF32), 20-core Armv9 CPU (10× Cortex-X925 + 10× Cortex-A725), 128 GB coherent LPDDR5x UMA (256-bit bus, ~273 GB/s peak bandwidth), and ConnectX-7 RoCE v2 (`enp1s0f1np1`, GID index 3).
3. **Blackwell SM12x Kernel Compatibility & Fallbacks:** DeepGEMM `nv_dev` (`8b1392b`) compilation status vs TMA hardware assertions at `csrc/apis/attention.hpp:122`, 2-state MQA indexer gates (PR #53522), `deepgemm-pr-403.diff` layout transformations, CUTLASS FP8 block-scaled GEMM exclusion (PR #53055), and grouped `torch.bmm` output projection in `try_b12x_wo_proj`.
4. **FlashInfer Sparse Dispatch & Upstream PR Accounting:** TOPK=192 dispatch arithmetic ($\lceil 133/64 \rceil \times 64 = 192$, FlashInfer PR #4380), builder contiguity fixes (PR #53574), query shape adaptations (PR #52499), and kernel block size 64 support (PR #53425).
5. **Memory Bounds, Quality Gates & Benchmarks:** 128 GiB UMA boundary constraints (safe util $\le 0.80$, earlyoom <8% MemAvailable cliff, swap unmasking on spark2, `vm.swappiness=10`), greedy France quality gate (32 tokens, first token `' Paris'`, logprob $\in [-0.27, -0.24]$, $n_{\text{tie}}=1$), and concurrency scaling (c1 ~25.8 tok/s, c8 ~95 tok/s, c32 ~172 tok/s).

---

## 2. Audit Execution Strategy

1. **Researcher Subagent:** Gather primary evidence across repository source files, patch diffs, assertion scripts, upstream pull requests, and DeepSeek/NVIDIA literature.
2. **Verifier Subagent:** Verify source URLs, commit SHAs, PR numbers, mathematical equations, and check inline citations for the final audit artifact.
3. **Artifact Delivery:** Write the complete, cited audit report to `outputs/vllm-spark-0731-docs-audit.md` with structured sections covering system architecture, hardware execution, upstream PR matrix, memory limits, gap analysis, reproduction risks, and primary sources.
