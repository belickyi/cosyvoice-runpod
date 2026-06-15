# CosyVoice3 TTS RunPod Serverless Handler
# Model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/pretrained_models/Fun-CosyVoice3-0.5B
ENV HF_HOME=/app/hf_cache
ENV TORCH_HOME=/app/torch_cache

# Pip settings for slow network (timeout 600s, 5 retries)
ENV PIP_DEFAULT_TIMEOUT=600
ENV PIP_RETRIES=5

# Note: HF_HUB_ENABLE_HF_TRANSFER removed - causes issues with newer huggingface_hub

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (most stable layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clone CosyVoice repository (changes rarely)
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    git submodule update --init --recursive

# Install setuptools for pkg_resources (required by openai-whisper)
RUN pip install --no-cache-dir setuptools

# Install openai-whisper separately with --no-build-isolation to use system setuptools
# This fixes "No module named 'pkg_resources'" error
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

# Install CosyVoice specific dependencies
RUN cd /app/CosyVoice && \
    pip install --no-cache-dir -r requirements.txt

# Create cache directories
RUN mkdir -p /app/hf_cache /app/torch_cache /app/pretrained_models

# Download model from HuggingFace using new CLI (hf command)
# huggingface-cli is deprecated, use 'hf' instead
RUN hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir /app/pretrained_models/Fun-CosyVoice3-0.5B \
    && echo "Model downloaded successfully"

# Download ttsfrd for text normalization (optional but recommended)
RUN hf download FunAudioLLM/CosyVoice-ttsfrd \
    --local-dir /app/pretrained_models/CosyVoice-ttsfrd \
    || echo "ttsfrd download failed, will use wetext"

# Fix torch version conflict: CosyVoice installs 2.3.1, but torchvision needs 2.4.1
RUN pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Copy handler
COPY handler.py /app/handler.py

# Skip warmup during build - model loads on first request
# This avoids torch version conflicts and speeds up builds

# RunPod handler entrypoint
CMD ["python3", "-u", "/app/handler.py"]
