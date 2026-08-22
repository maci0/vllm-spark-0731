# DeepSeek-V4-Flash-0731 on 2x DGX Spark (GB10, sm_121a).
#
# Official vLLM v0.28.0rc2 + public b12x==1.2.6 + cherry-picks:
#   - PR #52018 b12x MXFP4 MoE (merged 2026-08-21, after rc2)
#   - PR #50645 mHC TileLang guard (open; SM12x DeepGEMM miss)
#   - B12xWarmupUnit + get_b12x_fused_moe (main, not in rc2)
#   - nvfp4_ds_mla 584-byte DSV4 page (no CUDA writer vendored)
#
# Build on the Spark (linux/arm64). The amd64 official tag will not run on GB10.
#
#   docker build --platform linux/arm64 -t vllm-spark-0731:v0.28.0rc2-b12x .

ARG VLLM_RELEASE=v0.28.0rc2
FROM vllm/vllm-openai:${VLLM_RELEASE}

ARG B12X_VERSION=1.2.6
ARG RECIPE_VERSION=0.1.0

LABEL org.opencontainers.image.title="vllm-spark-0731" \
      org.opencontainers.image.description="v0.28.0rc2 + b12x kernels + fp8_ds_mla/nvfp4_ds_mla + DSpark for DeepSeek-V4-Flash-0731 on GB10" \
      org.opencontainers.image.version="${RECIPE_VERSION}" \
      recipe.checkpoint="deepseek-ai/DeepSeek-V4-Flash-0731" \
      recipe.vllm="${VLLM_RELEASE}" \
      recipe.b12x="${B12X_VERSION}" \
      recipe.moe="b12x" \
      recipe.linear="b12x"

ENV CUTE_DSL_ARCH=sm_121a \
    HF_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731 \
    VLLM_USE_DEEP_GEMM_E8M0=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTORCH_ALLOC_CONF=expandable_segments:True

WORKDIR /opt/spark-0731

# uv into the same interpreter that already has vLLM. Do not pip.
COPY patches /opt/spark-0731/patches
COPY configs /opt/spark-0731/configs
COPY scripts /opt/spark-0731/scripts
COPY VERSION /opt/spark-0731/VERSION

RUN set -eux; \
    if ! command -v uv >/dev/null 2>&1; then \
      curl -LsSf https://astral.sh/uv/install.sh | sh; \
      export PATH="${HOME}/.local/bin:${PATH}"; \
    fi; \
    uv pip install --python "$(command -v python3)" --no-cache "b12x==${B12X_VERSION}"; \
    python3 /opt/spark-0731/patches/apply_overlays.py; \
    python3 /opt/spark-0731/patches/assert_image.py

# Keep the stock vLLM entrypoint (vllm serve ...).
