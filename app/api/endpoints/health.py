"""Health check and diagnostic endpoints."""

import os
from pathlib import Path
from fastapi import APIRouter
import yt_dlp

from app.config import settings
from app.services.whisper_local import local_whisper_service
from app.services.groq_client import groq_whisper_service

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health")
def health_check():
    """System health check and hardware readiness diagnostics."""
    gpu_info = local_whisper_service.get_gpu_info()
    storage_writable = False
    try:
        settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        test_file = settings.STORAGE_DIR / ".health_test"
        test_file.write_text("ok")
        test_file.unlink()
        storage_writable = True
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": "MediaMetaData_Transcriber",
        "version": "1.0.0",
        "hardware": {
            "cuda_available": gpu_info.get("cuda_available", False),
            "device_name": gpu_info.get("device_name"),
            "vram_total_mb": gpu_info.get("vram_total_mb"),
            "vram_free_mb": gpu_info.get("vram_free_mb"),
        },
        "whisper": {
            "model": settings.WHISPER_MODEL,
            "device": settings.WHISPER_DEVICE,
            "compute_type": settings.WHISPER_COMPUTE_TYPE,
        },
        "groq_fallback": {
            "enabled": settings.ENABLE_GROQ_FALLBACK,
            "configured": groq_whisper_service.is_configured,
            "model": settings.GROQ_MODEL,
        },
        "storage": {
            "jobs_directory": str(settings.STORAGE_DIR),
            "models_directory": str(settings.MODEL_DIR),
            "storage_writable": storage_writable,
        },
        "guardrails": {
            "max_duration_seconds": settings.MAX_DURATION_SECONDS,
            "max_duration_minutes": settings.max_duration_minutes,
            "max_audio_size_bytes": settings.MAX_AUDIO_SIZE_BYTES,
            "max_audio_size_mb": settings.max_audio_size_mb,
            "audio_encoding": {
                "sample_rate_hz": settings.AUDIO_SAMPLE_RATE,
                "channels": "mono (1)",
                "bitrate": settings.AUDIO_BITRATE,
            },
        },
        "yt_dlp_version": yt_dlp.version.__version__,
    }
