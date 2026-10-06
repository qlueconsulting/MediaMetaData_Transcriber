# ==============================================================================
# MediaMetaData_Transcriber Dockerfile (Ultra-Slim & Fast Build)
# Optimized for Ubuntu 24.04 / NVIDIA RTX 2060 Super
# Reduces image footprint from ~10GB down to ~1.6GB (80%+ reduction)
# ==============================================================================

FROM python:3.11-slim-bookworm

# Prevent interactive prompts & disable bytecode
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install only essential runtime packages: ffmpeg, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create required persistent storage and model cache directories
RUN mkdir -p /srv/storage/jobs /srv/storage/models

# 1. Cache dependencies layer (changes rarely - fast cached builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       nvidia-cublas-cu12 \
       nvidia-cudnn-cu12

# Configure dynamic library path for NVIDIA CUDA wheels so CTranslate2 finds them
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# 2. Copy application source code (changes frequently - takes ~2 seconds to rebuild)
COPY pyproject.toml .
COPY app/ ./app/

# Expose microservice HTTP port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
