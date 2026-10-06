"""Transcription and audio extraction endpoints."""

import time
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

from app.models.job import (
    TranscribeRequest,
    JobTranscribeRequest,
    JobResponse,
    JobStatus,
    ExtractAudioResponse,
)
from app.config import settings
from app.models.metadata import MediaMetadataRequest, MediaMetadataResponse
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
    5. Cache lookup: If not bypass_cache and video was previously transcribed, returns cached result instantly.
    """
    url = str(request.url).strip()

    # Step 0: Check persistent storage cache if not bypassed
    if not request.bypass_cache:
        cached_id = None
        if request.job_id and storage_service.get_transcript_path(request.job_id).is_file():
            cached_id = request.job_id
        else:
            cached_id = storage_service.find_job_by_url(url, require_transcript=True)

        if cached_id:
            meta_data = storage_service.get_meta(cached_id)
            transcript_data = storage_service.get_transcript(cached_id)
            audio_size = storage_service.get_audio_size(cached_id)
            log.info(f"Serving fully cached job {cached_id} for URL: {url}")
            created_at_val = (meta_data.get("created_at") if meta_data else None) or datetime.now(timezone.utc).isoformat()
            return JobResponse(
                job_id=cached_id,
                status=JobStatus.COMPLETED,
                url=url,
                created_at=created_at_val,
                completed_at=datetime.now(timezone.utc).isoformat(),
                execution_time_seconds=0.01,
                cached=True,
                meta=MediaMetadataResponse(**meta_data, cached=True, cached_job_id=cached_id) if meta_data else None,
                audio_size_bytes=audio_size,
                audio_file_available=storage_service.has_audio(cached_id),
                transcript=transcript_data,
            )

    job_id = request.job_id or str(uuid.uuid4())
    start_time = time.time()

    log.info(f"Initiating live transcription pipeline for job {job_id} on URL: {url}")

    # Step 1: Pre-extract metadata & Enforce Duration Guardrail
    meta_kwargs = {"url": url, "job_id": job_id}
    if request.bypass_cache:
        meta_kwargs["bypass_cache"] = True
    if request.max_duration_minutes is not None:
        meta_kwargs["max_duration_minutes"] = request.max_duration_minutes
    metadata = media_service.extract_metadata(**meta_kwargs)

    if metadata.exceeds_duration_limit:
        limit_desc = (
            f"{metadata.max_duration_seconds} seconds ({metadata.max_duration_seconds // 60} minutes)"
            if metadata.max_duration_seconds and metadata.max_duration_seconds > 0
            else "duration limit"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Video duration ({metadata.duration_seconds:.1f}s / {metadata.duration_formatted}) "
                f"exceeds the operational limit of {limit_desc}. "
                f"Audio extraction and transcription rejected."
            ),
        )

    # Step 2: Download & Extract Audio to 16kHz Mono MP3 (Enforces <= 25MB check)
    dl_kwargs = {"url": url, "job_id": job_id, "pre_extracted_meta": metadata}
    if request.bypass_cache:
        dl_kwargs["bypass_cache"] = True
    if request.max_duration_minutes is not None:
        dl_kwargs["max_duration_minutes"] = request.max_duration_minutes
    audio_path, saved_meta = media_service.download_and_extract_audio(**dl_kwargs)
    audio_size = storage_service.get_audio_size(job_id)

    # Step 3: Speech-to-Text with faster-whisper (CUDA float16) -> Groq Whisper fallback
    transcript_data = transcription_orchestrator.transcribe(
        audio_path=audio_path,
        job_id=job_id,
        language=request.language,
        prompt=request.prompt,
        word_timestamps=request.word_timestamps,
        force_engine=request.force_engine,
        speed_profile=request.speed_profile,
        model_name=request.model,
        audio_duration=metadata.duration_seconds,
    )

    # Ensure transcript.json is saved in persistent storage
    if not storage_service.get_transcript_path(job_id).is_file():
        storage_service.save_transcript(job_id, transcript_data.model_dump())

    elapsed_time = round(time.time() - start_time, 2)
    log.info(f"Job {job_id} fully processed in {elapsed_time}s")

    sla_met = elapsed_time <= float(settings.WHISPER_SLA_TARGET_SECONDS)
    rtf = round(metadata.duration_seconds / elapsed_time, 2) if (metadata.duration_seconds and elapsed_time > 0) else None

    return JobResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        url=url,
        created_at=saved_meta.created_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        execution_time_seconds=elapsed_time,
        sla_met=sla_met,
        real_time_factor=rtf,
        speed_profile=request.speed_profile,
        cached=False,
        meta=saved_meta,
        audio_size_bytes=audio_size,
        audio_file_available=True,
        transcript=transcript_data,
    )


@router.post("/extract-audio", response_model=ExtractAudioResponse)
def extract_audio_only(request: MediaMetadataRequest) -> ExtractAudioResponse:
    """Extract and encode audio without triggering speech-to-text.

    Enforces:
    1. Duration limit check (rejection if exceeding limit).
    2. Conversion to 16kHz mono MP3 (-ac 1 -ar 16000 -b:a 64k).
    3. Audio file size guard (rejection if > 25 MB).
    4. Saves meta.json and audio.mp3 to /srv/storage/jobs/{job_id}/.
    5. Cache lookup: If not bypass_cache and audio is cached, returns existing audio instantly.
    """
    url = str(request.url).strip()
    job_id = str(uuid.uuid4())

    if not request.bypass_cache:
        cached_job_id = storage_service.find_job_by_url(url, require_audio=True)
        if cached_job_id:
            meta_dict = storage_service.get_meta(cached_job_id)
            audio_path = storage_service.get_audio_path(cached_job_id)
            audio_size = storage_service.get_audio_size(cached_job_id) or 0
            if meta_dict and audio_path.is_file():
                log.info(f"extract-audio: Serving cached audio for {url} from job {cached_job_id}")
                return ExtractAudioResponse(
                    job_id=cached_job_id,
                    status=JobStatus.COMPLETED,
                    meta=MediaMetadataResponse(**meta_dict, cached=True, cached_job_id=cached_job_id),
                    audio_size_bytes=audio_size,
                    audio_path=str(audio_path),
                    cached=True,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )

    meta_kwargs = {"url": url, "job_id": job_id}
    if request.bypass_cache:
        meta_kwargs["bypass_cache"] = True
    if request.max_duration_minutes is not None:
        meta_kwargs["max_duration_minutes"] = request.max_duration_minutes
    metadata = media_service.extract_metadata(**meta_kwargs)

    if metadata.exceeds_duration_limit:
        limit_desc = (
            f"{metadata.max_duration_seconds} seconds ({metadata.max_duration_seconds // 60} minutes)"
            if metadata.max_duration_seconds and metadata.max_duration_seconds > 0
            else "duration limit"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Video duration ({metadata.duration_seconds:.1f}s / {metadata.duration_formatted}) "
                f"exceeds the operational limit of {limit_desc}. "
                f"Audio extraction rejected."
            ),
        )

    dl_kwargs = {"url": url, "job_id": job_id, "pre_extracted_meta": metadata}
    if request.bypass_cache:
        dl_kwargs["bypass_cache"] = True
    if request.max_duration_minutes is not None:
        dl_kwargs["max_duration_minutes"] = request.max_duration_minutes
    audio_path, saved_meta = media_service.download_and_extract_audio(**dl_kwargs)
    audio_size = storage_service.get_audio_size(job_id) or 0

    return ExtractAudioResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        meta=saved_meta,
        audio_size_bytes=audio_size,
        audio_path=str(audio_path),
        cached=saved_meta.cached,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/jobs/{job_id}/transcribe", response_model=JobResponse)
def transcribe_job_audio(job_id: str, request: JobTranscribeRequest) -> JobResponse:
    """Transcribe audio for an already extracted job.

    Enforces:
    1. audio.mp3 must exist under /srv/storage/jobs/{job_id}/.
    2. If not bypass_cache and transcript.json exists, returns cached transcript instantly.
    3. Runs speech-to-text with faster-whisper and saves transcript.json.
    """
    if not storage_service.has_audio(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file audio.mp3 not found for job {job_id}. Extract audio first before transcribing.",
        )

    start_time = time.time()
    audio_path = storage_service.get_audio_path(job_id)
    audio_size = storage_service.get_audio_size(job_id) or 0
    saved_meta = storage_service.get_meta(job_id)
    meta_obj = MediaMetadataResponse(**saved_meta) if saved_meta else None
    url = meta_obj.url if meta_obj else ""

    created_at_val = (meta_obj.created_at if meta_obj else None) or datetime.now(timezone.utc).isoformat()

    if not request.bypass_cache and storage_service.get_transcript_path(job_id).is_file():
        cached_transcript = storage_service.get_transcript(job_id)
        if cached_transcript:
            log.info(f"Serving cached transcript for job {job_id}")
            return JobResponse(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                url=url,
                created_at=created_at_val,
                completed_at=datetime.now(timezone.utc).isoformat(),
                execution_time_seconds=0.01,
                cached=True,
                meta=meta_obj,
                audio_size_bytes=audio_size,
                audio_file_available=True,
                transcript=cached_transcript,
            )

    log.info(f"Running Whisper transcription for job {job_id} ({audio_path})")
    duration_sec = meta_obj.duration_seconds if meta_obj else None
    transcript_data = transcription_orchestrator.transcribe(
        audio_path=audio_path,
        job_id=job_id,
        language=request.language,
        prompt=request.prompt,
        word_timestamps=request.word_timestamps,
        force_engine=request.force_engine,
        speed_profile=request.speed_profile,
        model_name=request.model,
        audio_duration=duration_sec,
    )

    storage_service.save_transcript(job_id, transcript_data.model_dump())
    elapsed_time = round(time.time() - start_time, 2)
    log.info(f"Job {job_id} transcribed in {elapsed_time}s")

    sla_met = elapsed_time <= float(settings.WHISPER_SLA_TARGET_SECONDS)
    rtf = round(duration_sec / elapsed_time, 2) if (duration_sec and elapsed_time > 0) else None

    return JobResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        url=url,
        created_at=created_at_val,
        completed_at=datetime.now(timezone.utc).isoformat(),
        execution_time_seconds=elapsed_time,
        sla_met=sla_met,
        real_time_factor=rtf,
        speed_profile=request.speed_profile,
        cached=False,
        meta=meta_obj,
        audio_size_bytes=audio_size,
        audio_file_available=True,
        transcript=transcript_data,
    )

