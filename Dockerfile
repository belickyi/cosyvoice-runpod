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

# HuggingFace settings for faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

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

# Install CosyVoice specific dependencies
RUN cd /app/CosyVoice && \
    pip install --no-cache-dir -r requirements.txt || true

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

# Copy handler and warmup script
COPY handler.py /app/handler.py
COPY warmup.py /app/warmup.py

# Warmup: Load model and run test inference to speed up cold start
RUN python3 /app/warmup.py

# RunPod handler entrypoint
CMD ["python3", "-u", "/app/handler.py"]
