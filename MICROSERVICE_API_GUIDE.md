# MediaMetaData_Transcriber Microservice Integration Guide

## 1. Overview & Connection Architecture

`MediaMetaData_Transcriber` is a high-performance self-hosted GPU-accelerated microservice that resolves online media metadata, extracts and transcodes audio to 16kHz mono MP3, and performs high-speed speech-to-text using local multi-GPU Faster-Whisper (`large-v3-turbo`) with an adaptive `< 30s` SLA engine and optional Groq Cloud Turbo fallback.

### Network & URL Configuration
* **Base URL (Remote / Reverse Proxy):** `http://mediatranscript.mackley.online:443` *(Note: Uses plain HTTP on port 443 unless SSL termination is configured upstream)*
* **Base URL (Local Network / Direct):** `http://<SERVER_IP>:8000`
* **Route Prefixes:** Both `/api/...` and `/api/v1/...` are fully supported and alias the same endpoints.
* **Interactive Testing Apparatus (Web UI):** `http://<HOST>/ui` or `http://<HOST>/`

---

## 2. Authentication

Endpoints can be secured via the optional `API_KEY` setting. When configured on the server, all requests must provide the key using either header:

```http
X-API-Key: <YOUR_API_KEY>
```
or
```http
Authorization: Bearer <YOUR_API_KEY>
```

> **Public Paths:** `/`, `/ui`, `/health`, `/api/health`, `/api/v1/health`, and `/docs` are public for monitoring and testing.

---

## 3. API Endpoints

### 3.1. Resolve Media Metadata & Guardrails Check
Extracts video metadata without downloading the full media file, verifying whether it qualifies for transcription under operational duration guardrails.

* **Method:** `POST`
* **Endpoint:** `/api/metadata` (or `/api/v1/metadata`)
* **Headers:**
  * `Content-Type: application/json`
  * `X-API-Key: <YOUR_API_KEY>`

#### Request Body
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "bypass_cache": false,
  "max_duration_minutes": 20
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `url` | `string` | **Yes** | — | Target media URL (YouTube, shorts, Vimeo, direct MP4/MP3, etc.) |
| `bypass_cache` | `boolean` | No | `false` | Set `true` to force fresh extraction, bypassing cached metadata |
| `max_duration_minutes` | `integer` | No | `20` | Max duration threshold in minutes. Set `0` for unlimited |

#### Response (`200 OK`)
```json
{
  "job_id": null,
  "url": "https://www.youtube.com/shorts/IMsdJC7aThk",
  "title": "Does Every Dealership Play These Games?",
  "creator": "Delivrd",
  "uploader": "Delivrd",
  "duration_seconds": 157.0,
  "duration_formatted": "02:37",
  "platform": "Youtube",
  "extractor": "youtube",
  "upload_date": "20261004",
  "view_count": 89706,
  "thumbnail": "https://i.ytimg.com/vi_webp/IMsdJC7aThk/maxresdefault.webp",
  "description": "Video description text...",
  "exceeds_duration_limit": false,
  "max_duration_seconds": 1200,
  "allowed_for_transcription": true,
  "warning": null,
  "cached": false,
  "cached_job_id": null,
  "created_at": "2026-10-07T01:03:26.270892+00:00"
}
```

---

### 3.2. End-to-End Transcription Pipeline
Performs the complete pipeline: resolves metadata, checks guardrails, downloads lightweight audio stream, transcodes to 16kHz Mono MP3 with multi-threaded FFmpeg, and executes GPU Whisper inference with adaptive SLA batch scaling.

* **Method:** `POST`
* **Endpoint:** `/api/transcribe` (or `/api/v1/transcribe`)
* **Headers:**
  * `Content-Type: application/json`
  * `X-API-Key: <YOUR_API_KEY>`

#### Request Body
```json
{
  "url": "https://www.youtube.com/shorts/IMsdJC7aThk",
  "language": "en",
  "word_timestamps": false,
  "speed_profile": "adaptive",
  "bypass_cache": false,
  "max_duration_minutes": 20
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `url` | `string` | **Yes** | — | Target media URL |
| `language` | `string` | No | `null` | ISO language code (e.g. `"en"`, `"es"`). If omitted, auto-detected |
| `word_timestamps` | `boolean` | No | `false` | Set `true` to return precise start/end timestamps for every single word |
| `speed_profile` | `string` | No | `"adaptive"` | SLA Profile: `"adaptive"` (<30s target), `"turbo"` (Large-v3-Turbo), `"standard"` (Large-v3), or `"groq"` (Cloud Turbo) |
| `force_engine` | `string` | No | `null` | Force engine override: `"cuda"`, `"groq"`, or `"cpu"` |
| `prompt` | `string` | No | `null` | Optional context prompt or hotwords to guide Whisper decoding |
| `bypass_cache` | `boolean` | No | `false` | If `false`, returns cached transcript instantly if URL was previously transcribed |
| `max_duration_minutes` | `integer` | No | `20` | Max duration threshold in minutes (set `0` to override) |

#### Response (`200 OK`)
```json
{
  "job_id": "20261007_010415_a1b2c3d4",
  "url": "https://www.youtube.com/shorts/IMsdJC7aThk",
  "status": "completed",
  "elapsed_time": 4.82,
  "sla_met": true,
  "real_time_factor": 32.57,
  "speed_profile": "adaptive",
  "cached": false,
  "cached_job_id": null,
  "metadata": {
    "title": "Does Every Dealership Play These Games?",
    "duration_seconds": 157.0,
    "duration_formatted": "02:37"
  },
  "transcript": {
    "job_id": "20261007_010415_a1b2c3d4",
    "text": "Full transcribed paragraph here...",
    "language": "en",
    "language_probability": 0.9982,
    "duration_seconds": 157.0,
    "engine": "faster-whisper-cuda",
    "model": "large-v3-turbo",
    "segments": [
      {
        "id": 1,
        "start": 0.0,
        "end": 3.42,
        "text": " Does every dealership play these games?",
        "avg_logprob": -0.154,
        "no_speech_prob": 0.002,
        "words": null
      }
    ]
  },
  "audio_file": "/srv/storage/jobs/20261007_010415_a1b2c3d4/audio.mp3",
  "error": null
}
```

---

### 3.3. Retrieve Stored Job Details
Fetches previously saved job metadata and transcripts by `job_id`.

* **Method:** `GET`
* **Endpoint:** `/api/jobs/{job_id}` (or `/api/v1/jobs/{job_id}`)

#### Response (`200 OK`)
Returns the complete `JobResponse` object saved under `/srv/storage/jobs/{job_id}/`.

---

### 3.4. Hardware & System Health Diagnostics
Inspects server GPU readiness, multi-GPU cluster allocation, VRAM statistics, Groq fallback status, and operational guardrails.

* **Method:** `GET`
* **Endpoint:** `/api/health` (or `/api/v1/health`)

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "service": "MediaMetaData_Transcriber",
  "version": "1.0.0",
  "hardware": {
    "cuda_available": true,
    "device_count": 2,
    "device_name": "NVIDIA GeForce RTX 2060 SUPER",
    "total_cluster_vram_mb": 16384.0,
    "total_cluster_free_vram_mb": 15200.0,
    "devices": [
      { "index": 0, "name": "NVIDIA GeForce RTX 2060 SUPER", "vram_total_mb": 8192.0, "vram_free_mb": 7600.0, "vram_used_mb": 592.0 },
      { "index": 1, "name": "NVIDIA GeForce RTX 2060 SUPER", "vram_total_mb": 8192.0, "vram_free_mb": 7600.0, "vram_used_mb": 592.0 }
    ],
    "active_device_indices": [0, 1]
  },
  "whisper": {
    "model": "large-v3-turbo",
    "device": "cuda",
    "compute_type": "float16"
  },
  "groq_fallback": {
    "enabled": true,
    "configured": true,
    "model": "whisper-large-v3-turbo"
  },
  "guardrails": {
    "max_duration_seconds": 1200,
    "max_duration_minutes": 20,
    "max_audio_size_bytes": 26214400
  }
}
```

---

## 4. Client Integration Examples

### Python Integration (for AGY / FastAPI Apps)

```python
import httpx
from typing import Optional, Dict, Any

class MediaTranscriberClient:
    def __init__(self, base_url: str = "http://mediatranscript.mackley.online:443", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key

    async def get_metadata(self, url: str, bypass_cache: bool = False, max_duration_minutes: int = 20) -> Dict[str, Any]:
        """Resolve media metadata and verify operational guardrails."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/metadata",
                headers=self.headers,
                json={
                    "url": url,
                    "bypass_cache": bypass_cache,
                    "max_duration_minutes": max_duration_minutes,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def transcribe(
        self,
        url: str,
        language: Optional[str] = "en",
        word_timestamps: bool = False,
        speed_profile: str = "adaptive",
        bypass_cache: bool = false,
        max_duration_minutes: int = 20,
    ) -> Dict[str, Any]:
        """Download media, extract 16kHz MP3, and transcribe with Whisper."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/transcribe",
                headers=self.headers,
                json={
                    "url": url,
                    "language": language,
                    "word_timestamps": word_timestamps,
                    "speed_profile": speed_profile,
                    "bypass_cache": bypass_cache,
                    "max_duration_minutes": max_duration_minutes,
                },
            )
            resp.raise_for_status()
            return resp.json()

# Usage Example:
# client = MediaTranscriberClient(api_key="619b8e2c2672683b815eb58eddb95723cf7e8bbb573a353ba893fd8666361e33")
# result = await client.transcribe("https://www.youtube.com/shorts/IMsdJC7aThk")
# print(result["transcript"]["text"])
```

### TypeScript / Node.js Integration

```typescript
interface TranscribeRequest {
  url: string;
  language?: string;
  word_timestamps?: boolean;
  speed_profile?: 'adaptive' | 'turbo' | 'standard' | 'groq';
  bypass_cache?: boolean;
  max_duration_minutes?: number;
}

interface TranscribeResponse {
  job_id: string;
  status: string;
  elapsed_time: number;
  sla_met: boolean;
  transcript: {
    text: string;
    language: string;
    segments: Array<{ start: number; end: number; text: string }>;
  };
}

export async function transcribeMedia(
  req: TranscribeRequest,
  baseUrl = "http://mediatranscript.mackley.online:443",
  apiKey?: string
): Promise<TranscribeResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch(`${baseUrl}/api/transcribe`, {
    method: "POST",
    headers,
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(`Transcription failed (${res.status}): ${JSON.stringify(errorBody)}`);
  }

  return await res.json();
}
```

---

## 5. Error Codes & Troubleshooting

| HTTP Code | Error Scenario | Resolution |
|---|---|---|
| `400 Bad Request` | Video exceeds duration ceiling (e.g. > 20 mins) | Provide `"max_duration_minutes": 0` in request body to override the limiter |
| `400 Bad Request` | Invalid/unsupported media URL or private video | Verify URL is publicly accessible |
| `401 Unauthorized` | Missing or invalid API key | Supply valid `-H "X-API-Key: <key>"` |
| `404 Not Found` | Media track not found or job ID does not exist | Check job ID or media availability |
| `500 Server Error` | Unexpected pipeline failure | Automatic multi-stage self-healing cascade automatically falls back through `Batched` $\rightarrow$ `Sequential` $\rightarrow$ `int8_float16` $\rightarrow$ `CPU` |
