# ==============================================================================
# MediaMetaData_Transcriber Dockerfile
# GPU-Accelerated Microservice for Ubuntu 24.04 / NVIDIA RTX 2060 Super
# ==============================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies: Python 3.11, ffmpeg, curl, git
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3.11-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set CUDA / cuDNN library path for CTranslate2
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

# Create required persistent storage and model cache directories
RUN mkdir -p /srv/storage/jobs /srv/storage/models

# Install Python requirements
COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt \
    && python3.11 -m pip install --no-cache-dir \
       nvidia-cublas-cu12 \
       nvidia-cudnn-cu12

# Copy application source code
COPY pyproject.toml .
COPY app/ ./app/

# Expose microservice HTTP port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
