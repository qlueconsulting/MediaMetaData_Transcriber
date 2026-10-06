"""Transcription and audio extraction endpoints."""

import time
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

from app.models.job import (
    TranscribeRequest,
    JobResponse,
    JobStatus,
    ExtractAudioResponse,
)
from app.models.metadata import MediaMetadataRequest
from app.services.media import media_service
from app.services.transcription import transcription_orchestrator
from app.services.storage import storage_service
from app.utils.logger import log

router = APIRouter(tags=["Transcription"])


@router.post("/transcribe", response_model=JobResponse)
def transcribe_media(request: TranscribeRequest) -> JobResponse:
    """End-to-end media resolution, audio extraction, and transcription.

    Operational Guardrails Enforced:
    1. Early 20-Minute Rejection: Video > 1,200s (20 mins) returns HTTP 400 Bad Request.
    2. Audio Conversion: 16kHz Mono MP3 (-ac 1 -ar 16000 -b:a 64k).
    3. Audio File Size Guard: Audio > 25 MB returns HTTP 400 Bad Request.
    4. Common Storage: Saves meta.json, audio.mp3, and transcript.json under /srv/storage/jobs/{job_id}/.
    """
    job_id = str(uuid.uuid4())
    start_time = time.time()
    url = str(request.url).strip()

    log.info(f"Initiating transcription pipeline for job {job_id} on URL: {url}")

    # Step 1: Pre-extract metadata & Enforce Early 20-Minute Guardrail
    # extract_metadata raises HTTP 400 inside download_and_extract_audio if > 1200s
    metadata = media_service.extract_metadata(url=url, job_id=job_id)
    if metadata.exceeds_duration_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Video duration ({metadata.duration_seconds:.1f}s / {metadata.duration_formatted}) "
                f"exceeds the operational limit of {metadata.max_duration_seconds} seconds (20 minutes). "
                f"Audio extraction and transcription rejected."
            ),
        )

    # Step 2: Download & Extract Audio to 16kHz Mono MP3 (Enforces <= 25MB check)
    audio_path, saved_meta = media_service.download_and_extract_audio(
        url=url,
        job_id=job_id,
        pre_extracted_meta=metadata,
    )
    audio_size = storage_service.get_audio_size(job_id)

    # Step 3: Speech-to-Text with faster-whisper (CUDA float16) -> Groq Whisper fallback
    transcript_data = transcription_orchestrator.transcribe(
        audio_path=audio_path,
        job_id=job_id,
        language=request.language,
        prompt=request.prompt,
        word_timestamps=request.word_timestamps,
        force_engine=request.force_engine,
    )

    # Ensure transcript.json is saved in persistent storage
    if not storage_service.get_transcript_path(job_id).is_file():
        storage_service.save_transcript(job_id, transcript_data.model_dump())

    elapsed_time = round(time.time() - start_time, 2)
    log.info(f"Job {job_id} fully processed in {elapsed_time}s")

    return JobResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        url=url,
        created_at=saved_meta.created_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        execution_time_seconds=elapsed_time,
        meta=saved_meta,
        audio_size_bytes=audio_size,
        audio_file_available=True,
        transcript=transcript_data,
    )


@router.post("/extract-audio", response_model=ExtractAudioResponse)
def extract_audio_only(request: MediaMetadataRequest) -> ExtractAudioResponse:
    """Extract and encode audio without triggering speech-to-text.

    Enforces:
    1. Early 20-minute check (rejection if > 1,200s).
    2. Conversion to 16kHz mono MP3 (-ac 1 -ar 16000 -b:a 64k).
    3. Audio file size guard (rejection if > 25 MB).
    4. Saves meta.json and audio.mp3 to /srv/storage/jobs/{job_id}/.
    """
    job_id = str(uuid.uuid4())
    url = str(request.url).strip()

    metadata = media_service.extract_metadata(url=url, job_id=job_id)
    if metadata.exceeds_duration_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Video duration ({metadata.duration_seconds:.1f}s / {metadata.duration_formatted}) "
                f"exceeds the operational limit of {metadata.max_duration_seconds} seconds (20 minutes). "
                f"Audio extraction rejected."
            ),
        )

    audio_path, saved_meta = media_service.download_and_extract_audio(
        url=url,
        job_id=job_id,
        pre_extracted_meta=metadata,
    )
    audio_size = storage_service.get_audio_size(job_id) or 0

    return ExtractAudioResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        meta=saved_meta,
        audio_size_bytes=audio_size,
        audio_path=str(audio_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
