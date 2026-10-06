#!/usr/bin/env bash
# ==============================================================================
# verify_host_env.sh: Verify host readiness on Ubuntu 24.04 LTS
# Target: Intel i7-7700K, 32GB RAM, NVIDIA RTX 2060 Super (8GB VRAM)
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " MediaMetaData_Transcriber Host Environment Verification"
echo "=========================================================="

# 1. OS & Kernel Check
echo "[1/6] Checking Operating System..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "  OS: $NAME $VERSION"
else
    echo "  OS: Unknown Linux distribution"
fi
uname -r

# 2. CPU & Memory Check
echo ""
echo "[2/6] Checking CPU & RAM..."
CPU_MODEL=$(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//')
RAM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
echo "  CPU: $CPU_MODEL"
echo "  Total RAM: $RAM_TOTAL (Requirement: 32GB recommended)"

# 3. NVIDIA Driver & RTX 2060 Super Check
echo ""
echo "[3/6] Checking NVIDIA GPU & Driver..."
if command -v nvidia-smi &> /dev/null; then
    echo "  nvidia-smi is available:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "  [ERROR] nvidia-smi not found! Install NVIDIA drivers: sudo apt install -y nvidia-driver-550"
    exit 1
fi

# 4. NVIDIA Container Toolkit Check
echo ""
echo "[4/6] Checking Docker & NVIDIA Container Toolkit..."
if command -v docker &> /dev/null; then
    docker --version
    if docker info 2>/dev/null | grep -i "Runtimes:.*nvidia" &> /dev/null; then
        echo "  [OK] NVIDIA Container Toolkit runtime detected in Docker."
    else
        echo "  [WARNING] NVIDIA Container Toolkit might not be configured as Docker default runtime."
        echo "  Run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
else
    echo "  [ERROR] Docker is not installed. Please install Docker Engine."
    exit 1
fi

# 5. Persistent Storage Mount Verification (/srv/storage/jobs)
echo ""
echo "[5/6] Verifying Persistent Storage (/srv/storage/jobs)..."
if [ ! -d "/srv/storage/jobs" ]; then
    echo "  Directory /srv/storage/jobs does not exist. Creating..."
    sudo mkdir -p /srv/storage/jobs /srv/storage/models
    sudo chmod -R 775 /srv/storage
    echo "  [OK] Created /srv/storage/jobs and /srv/storage/models"
else
    echo "  [OK] /srv/storage/jobs exists and is accessible."
fi

# 6. Test Docker GPU Access
echo ""
echo "[6/6] Testing Docker GPU passthrough..."
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "  [OK] Docker GPU passthrough succeeded!"
else
    echo "  [WARNING] Docker GPU container test failed. Verify nvidia-container-toolkit installation."
fi

echo ""
echo "=========================================================="
echo " Host verification complete. Ready to run docker compose up!"
echo "=========================================================="
