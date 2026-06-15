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

# Install CosyVoice specific dependencies
RUN cd /app/CosyVoice && \
    pip install --no-cache-dir -r requirements.txt || true

# Create cache directories
RUN mkdir -p /app/hf_cache /app/torch_cache /app/pretrained_models

# Download model from HuggingFace using CLI (more memory efficient)
# Split into separate commands to reduce peak memory usage
RUN huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir /app/pretrained_models/Fun-CosyVoice3-0.5B \
    --local-dir-use-symlinks False \
    && echo "Model downloaded successfully"

# Download ttsfrd for text normalization (optional but recommended)
RUN huggingface-cli download FunAudioLLM/CosyVoice-ttsfrd \
    --local-dir /app/pretrained_models/CosyVoice-ttsfrd \
    --local-dir-use-symlinks False \
    || echo "ttsfrd download failed, will use wetext"

# Copy handler last (changes most often)
COPY handler.py /app/handler.py

# Warmup: Load model and run test inference to speed up cold start
RUN python3 -c "\
import sys; \
sys.path.insert(0, '/app/CosyVoice'); \
sys.path.insert(0, '/app/CosyVoice/third_party/Matcha-TTS'); \
from cosyvoice.cli.cosyvoice import CosyVoice3; \
print('Loading model for warmup...'); \
model = CosyVoice3('/app/pretrained_models/Fun-CosyVoice3-0.5B', load_jit=False, load_trt=False, fp16=True); \
print(f'Model loaded, sample rate: {model.sample_rate}'); \
# Quick inference to warm up all components
for output in model.inference_instruct2(tts_text='Test', instruct_text='You are a helpful assistant.<|endofprompt|>', prompt_wav=None, stream=False): \
    print(f'Warmup inference: {len(output[\"tts_speech\"])} samples'); \
    break; \
print('Warmup complete')"

# RunPod handler entrypoint
CMD ["python3", "-u", "/app/handler.py"]
