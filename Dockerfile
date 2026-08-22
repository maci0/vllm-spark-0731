# DeepSeek-V4-Flash-0731 on 2x DGX Spark (GB10, sm_121a).
#
# v0.27.1 base (only arm64 image on Docker Hub) + vLLM v0.28.0rc2 from source.
# The base provides PyTorch 2.13.0+cu130, CUDA 13.0, system libs.
# rc2 source replaces v0.27.1 Python code; C extensions recompile.
#
# Public b12x==1.2.6 provides SM12x MoE + attention kernels.
# Minimal overlays for features merged after the rc2 tag cut.
#
#   docker build --platform linux/arm64 -t vllm-spark-0731:v0.28.0rc2-b12x .

ARG BASE_RELEASE=v0.27.1
FROM vllm/vllm-openai:${BASE_RELEASE}

ARG BASE_RELEASE=v0.27.1
ARG VLLM_RELEASE=v0.28.0rc2
ARG B12X_VERSION=1.2.6
ARG RECIPE_VERSION=0.2.0

LABEL org.opencontainers.image.title="vllm-spark-0731" \
      org.opencontainers.image.description="vLLM ${VLLM_RELEASE} + b12x for DeepSeek-V4-Flash-0731 on GB10" \
      org.opencontainers.image.version="${RECIPE_VERSION}" \
      recipe.checkpoint="deepseek-ai/DeepSeek-V4-Flash-0731" \
      recipe.vllm="${VLLM_RELEASE}" \
      recipe.base="${BASE_RELEASE}" \
      recipe.b12x="${B12X_VERSION}"

ENV CUTE_DSL_ARCH=sm_121a \
    HF_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731 \
    VLLM_USE_DEEP_GEMM_E8M0=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    MAX_JOBS=8

WORKDIR /opt/spark-0731

COPY patches /opt/spark-0731/patches
COPY configs /opt/spark-0731/configs
COPY scripts /opt/spark-0731/scripts
COPY VERSION /opt/spark-0731/VERSION

RUN set -eux; \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git && rm -rf /var/lib/apt/lists/*; \
    if ! command -v uv >/dev/null 2>&1; then \
      curl -LsSf https://astral.sh/uv/install.sh | sh; \
      export PATH="${HOME}/.local/bin:${PATH}"; \
    fi; \
    # Overlay rc2 Python code onto v0.27.1's compiled extensions. \
    # Base image has no CUDA dev headers, so building from source is not viable. \
    # v0.27.1 .so extensions (FlashAttn, MLA, DeepGEMM, etc.) are ABI-stable. \
    SITE="/usr/local/lib/python3.12/dist-packages"; \
    git clone --depth 1 --branch "${VLLM_RELEASE}" \
      https://github.com/vllm-project/vllm.git /tmp/vllm-rc2; \
    # Preserve compiled extensions and metadata \
    find "${SITE}/vllm" -name '*.so' -exec cp --parents -t /tmp/vllm-so {} +; \
    find "${SITE}" -maxdepth 1 -name 'vllm-*.dist-info' -exec cp -r {} /tmp/ \;; \
    # Replace Python code with rc2 \
    rm -rf "${SITE}/vllm"; \
    cp -r /tmp/vllm-rc2/vllm "${SITE}/vllm"; \
    # Restore .so extensions \
    cp -r /tmp/vllm-so/. /; \
    # Update version in dist-info \
    if [ -d /tmp/vllm-*.dist-info ]; then \
      rm -rf "${SITE}"/vllm-*.dist-info; \
      cp -r /tmp/vllm-*.dist-info "${SITE}/"; \
    fi; \
    rm -rf /tmp/vllm-rc2 /tmp/vllm-so /tmp/vllm-*.dist-info; \
    # Clear stale bytecode \
    find "${SITE}/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true; \
    python3 -m compileall -q "${SITE}/vllm" || true; \
    # Install b12x \
    uv pip install --python "$(command -v python3)" --no-cache "b12x==${B12X_VERSION}"; \
    # Apply post-rc2 overlays (only patches not yet in rc2) \
    python3 /opt/spark-0731/patches/apply_overlays.py; \
    python3 /opt/spark-0731/patches/assert_image.py

# Keep the stock vLLM entrypoint (vllm serve ...).
