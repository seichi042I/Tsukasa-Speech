# syntax=docker/dockerfile:1
# ===========================================================================
# Tsukasa Speech Training - Optimized Docker Image
#
# Multi-stage build: devel (compile) → runtime (run)
# Model weights are NOT baked in — downloaded at first run or volume-mounted.
#
# Expected size: ~10GB (vs ~38GB with devel + baked models)
# ===========================================================================

# ============ Stage 1: Builder (compile native extensions) ============
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r /tmp/requirements.txt

# ============ Stage 2: Runtime ============
# devel image required: xlstm sLSTM JIT-compiles CUDA kernels at import time
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Runtime-only system deps (no build-essential, no git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from pre-built wheels (offline, no git needed)
RUN --mount=type=bind,from=builder,source=/wheels,target=/tmp/wheels \
    --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    sed 's|^git+https://.*monotonic_align.*|monotonic-align|' /tmp/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r /tmp/req.txt

WORKDIR /app

# Layer 1: Library code + configs (changes rarely)
COPY tsukasa_speech/ tsukasa_speech/
COPY Utils/ Utils/
COPY Configs/ Configs/
COPY static/ static/
COPY reference_sample_wavs/ reference_sample_wavs/
COPY OOD_LargeScale_.csv .

# Layer 2: Accelerate shims + entrypoints (changes frequently)
COPY train_first.py finetune_accelerate.py ./
COPY entrypoint.sh train.sh ./
RUN chmod +x entrypoint.sh train.sh \
    && ln -s /app/train.sh /usr/local/bin/train \
    && mkdir -p /tmp/wave_cache

EXPOSE 7860 6006
ENTRYPOINT ["./entrypoint.sh"]
