# MediaMetaData_Transcriber

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![faster-whisper](https://img.shields.io/badge/STT-faster--whisper%20large--v3-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![yt-dlp](https://img.shields.io/badge/Media-yt--dlp-red.svg)](https://github.com/yt-dlp/yt-dlp)

**MediaMetaData_Transcriber** is a production-ready, GPU-accelerated microservice designed to run on a Linux Ubuntu 24.04 server behind a residential IP and an OPNsense reverse proxy.

It resolves YouTube and Google Video CDN IP-locks (`HTTP 403`) and datacenter rate-limiting (`HTTP 429`) by executing video resolution, audio extraction, and AI speech-to-text locally on the residential network.

---

## 1. System Specifications & Architecture

- **Host Target:** Ubuntu 24.04 LTS (x86_64)
- **CPU:** Intel Core i7-7700K (4.20 GHz / 4.50 GHz Turbo)
- **RAM:** 32 GB DDR4
- **GPU:** NVIDIA GeForce RTX 2060 Super (8GB GDDR6 VRAM)
- **Framework:** Python 3.11+ with FastAPI & Uvicorn (ASGI)
- **Media Engine:** `yt-dlp` (latest build) + `ffmpeg`
- **Speech-to-Text Engine:** `faster-whisper` (`large-v3` model with CUDA acceleration, `float16` compute type)
- **Cloud Fallback:** Groq Whisper API (`whisper-large-v3-turbo`) if GPU is unavailable or on CUDA Out-Of-Memory
- **Containerization:** Docker & Docker Compose with NVIDIA Container Toolkit GPU device reservation
- **Networking:** Local residential IP host behind an OPNsense firewall and reverse proxy

---

## 2. Strict Operational Criteria & Guardrails

The service implements strict guardrails to prevent resource exhaustion and ensure high reliability:

### 1. Early 20-Minute Decision Tree
- Any video exceeding **1,200 seconds (20 minutes)** is flagged immediately in the `/api/v1/metadata` endpoint (`exceeds_duration_limit: true`, `allowed_for_transcription: false`).
- Any attempt to trigger audio extraction or transcription on a video exceeding 1,200 seconds is **rejected early with an HTTP 400 Bad Request error**, without downloading or processing the media stream.

### 2. Audio File Size Guard (<= 25 MB)
- Extracted audio is strictly converted to 16kHz mono MP3 using FFmpeg:
  ```bash
  -ac 1 -ar 16000 -b:a 64k
  ```
  *(64 kbps mono audio yields ~9.6 MB for 20 minutes, preserving optimal Whisper accuracy while minimizing footprint).*
- The microservice enforces a hard maximum file size limit of **25 MB (26,214,400 bytes)**.
- If an extracted audio file exceeds 25 MB, the artifact is purged from disk and the service fails gracefully with an **HTTP 400 Bad Request** error.

### 3. Common Persistent Storage Structure
All jobs persist artifacts to a structured volume directory mounted at `/srv/storage/jobs/`:
```text
/srv/storage/jobs/{job_id}/
├── meta.json         # Video title, creator, duration, platform, timestamps
├── audio.mp3         # Extracted 16kHz mono audio (guaranteed <= 25MB)
└── transcript.json   # Full transcript, language detected, and segment timestamps
```

---

## 3. Storage Artifact Schemas

### `meta.json`
```json
{
  "job_id": "8c01d904-7a55-46aa-8367-1bc59f3d9b43",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "creator": "Rick Astley",
  "uploader": "RickAstleyVEVO",
  "duration_seconds": 213.0,
  "duration_formatted": "03:33",
  "platform": "youtube",
  "extractor": "youtube",
  "upload_date": "20091025",
  "view_count": 1600000000,
  "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
  "description": "The official video for Never Gonna Give You Up...",
  "exceeds_duration_limit": false,
  "max_duration_seconds": 1200,
  "allowed_for_transcription": true,
  "warning": null,
  "created_at": "2026-10-06T05:20:00Z"
}
```

### `transcript.json`
```json
{
  "job_id": "8c01d904-7a55-46aa-8367-1bc59f3d9b43",
  "text": "We're no strangers to love. You know the rules and so do I...",
  "language": "en",
  "language_probability": 0.9982,
  "duration_seconds": 213.0,
  "engine": "faster-whisper-cuda",
  "model": "large-v3",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 18.5,
      "end": 22.8,
      "text": " We're no strangers to love.",
      "avg_logprob": -0.18,
      "no_speech_prob": 0.005
    }
  ],
  "completed_at": "2026-10-06T05:20:25Z"
}
```

---

## 4. REST API Reference

Interactive OpenAPI documentation is available at `/docs` (Swagger UI) and `/redoc`.

### Health & Hardware Diagnostics
```http
GET /api/v1/health
```
Returns system status, CUDA GPU availability, VRAM allocation, Groq API readiness, storage permissions, and operational guardrails.

```bash
curl -s http://localhost:8000/api/v1/health | jq .
```

---

### Extract Metadata (Early 20-Min Check)
```http
POST /api/v1/metadata
Content-Type: application/json
```
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```
*Does not download the video stream. Returns metadata and evaluates whether duration exceeds 1,200 seconds.*

---

### Transcribe Media (End-to-End Pipeline)
```http
POST /api/v1/transcribe
Content-Type: application/json
```
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "language": "en",
  "word_timestamps": false,
  "force_engine": null
}
```
**Pipeline Execution:**
1. Validates duration <= 1,200s (rejects with `400 Bad Request` if over).
2. Downloads audio via `yt-dlp` and converts to 16kHz mono MP3 (`-ac 1 -ar 16000 -b:a 64k`).
3. Validates audio size <= 25 MB (rejects with `400 Bad Request` if over).
4. Transcribes via `faster-whisper` (`large-v3`, CUDA `float16`) with automatic fallback to Groq (`whisper-large-v3-turbo`).
5. Persists `meta.json`, `audio.mp3`, and `transcript.json` to `/srv/storage/jobs/{job_id}/`.

---

### Extract Audio Only
```http
POST /api/v1/extract-audio
Content-Type: application/json
```
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```
*Performs 20-minute check, audio conversion, and 25MB check. Saves `meta.json` and `audio.mp3` without running transcription.*

---

### Retrieve Artifacts

#### List Recent Jobs
```bash
curl http://localhost:8000/api/v1/jobs
```

#### Get Job Summary & File Status
```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

#### Download Metadata (`meta.json`)
```bash
curl http://localhost:8000/api/v1/jobs/{job_id}/meta
```

#### Stream / Download Extracted Audio (`audio.mp3`)
```bash
curl -O http://localhost:8000/api/v1/jobs/{job_id}/audio
```

#### Retrieve Transcript in Multiple Formats
- **JSON format (default):**
  ```bash
  curl http://localhost:8000/api/v1/jobs/{job_id}/transcript?format=json
  ```
- **Plain Text (`.txt`):**
  ```bash
  curl http://localhost:8000/api/v1/jobs/{job_id}/transcript?format=text
  ```
- **SubRip Subtitles (`.srt`):**
  ```bash
  curl http://localhost:8000/api/v1/jobs/{job_id}/transcript?format=srt
  ```
- **WebVTT Subtitles (`.vtt`):**
  ```bash
  curl http://localhost:8000/api/v1/jobs/{job_id}/transcript?format=vtt
  ```

---

## 5. Host Setup & Deployment on Ubuntu 24.04

### Step 1: Install NVIDIA Drivers & Container Toolkit
On your Ubuntu 24.04 server with the NVIDIA RTX 2060 Super:
```bash
# 1. Install NVIDIA driver
sudo apt update && sudo apt install -y nvidia-driver-550

# 2. Install Docker
sudo apt install -y docker.io docker-compose-v2

# 3. Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit

# 4. Configure Docker daemon for GPU support
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 2: Create Storage Directories
```bash
sudo mkdir -p /srv/storage/jobs /srv/storage/models
sudo chown -R $USER:$USER /srv/storage
```

### Step 3: Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Optionally provide your `GROQ_API_KEY` in `.env` to enable cloud Whisper fallback.

### Step 4: Run Verification Script
```bash
chmod +x scripts/verify_host_env.sh
./scripts/verify_host_env.sh
```

### Step 5: Start the Service via Docker Compose
```bash
docker compose up -d --build
```

Check logs and GPU initialization:
```bash
docker compose logs -f transcriber
```

---

## 6. OPNsense Reverse Proxy Setup

To expose the service securely behind your residential IP and OPNsense firewall:

1. **OPNsense Nginx Plugin Configuration:**
   - In OPNsense WebGUI, navigate to **Services -> Nginx -> Upstream Server**.
   - Add upstream pointing to your internal server LAN IP (e.g., `192.168.1.50:8000`).
   - Navigate to **HTTP Server** and configure:
     - `Client Max Body Size`: `50M`
     - `Proxy Read Timeout`: `600s`
     - `Proxy Send Timeout`: `600s`
   - See [`deploy/nginx_opnsense_upstream.conf`](deploy/nginx_opnsense_upstream.conf) for reference.

2. **Why Residential IP execution matters:**
   - Datacenter IP ranges (AWS, GCP, DigitalOcean, Hetzner) are frequently blocked by YouTube CDN (`HTTP 429` / `HTTP 403`).
   - Running directly on your Ubuntu 24.04 host behind your residential ISP connection solves IP blocking organically without commercial proxies.

---

## 7. Running Tests Locally

Run the complete test suite:
```bash
# In virtual environment:
pytest -v
```

All guardrails (20-minute limit, 25MB file size ceiling, storage structure, fallback behavior) are covered by unit and integration tests.

---

## 8. License

MIT License.
