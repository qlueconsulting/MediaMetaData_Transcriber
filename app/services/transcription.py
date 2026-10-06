"""Transcription orchestrator coordinating local faster-whisper and Groq Whisper API."""

from pathlib import Path
from typing import Optional
from fastapi import HTTPException, status

from app.config import settings
from app.utils.logger import log
from app.models.transcript import TranscriptData
from app.services.whisper_local import local_whisper_service
from app.services.groq_client import groq_whisper_service
from app.services.storage import storage_service


class TranscriptionOrchestrator:
    """Orchestrates speech-to-text inference with automatic GPU -> Cloud fallback."""

    def transcribe(
        self,
        audio_path: Path,
        job_id: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = False,
        force_engine: Optional[str] = None,
    ) -> TranscriptData:
        """Transcribe audio with resilient engine fallback.

        Preference order:
        1. Explicit override if requested (force_engine='groq' or 'cuda' or 'cpu').
        2. Force Groq flag if enabled in configuration.
        3. Local faster-whisper with CUDA acceleration.
        4. Groq Whisper Cloud API fallback if CUDA fails or is unavailable.
        5. CPU faster-whisper if Groq is unconfigured.
        """
        transcript: Optional[TranscriptData] = None
        error_history = []

        # 1. Force Groq if explicitly requested
        should_use_groq = (
            force_engine == "groq"
            or (settings.FORCE_GROQ_FALLBACK and groq_whisper_service.is_configured)
        )

        if should_use_groq:
            if not groq_whisper_service.is_configured:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Groq engine was requested, but GROQ_API_KEY is not configured.",
                )
            log.info(f"Using forced Groq Whisper API for job {job_id}")
            transcript = groq_whisper_service.transcribe(
                audio_path=audio_path,
                job_id=job_id,
                language=language,
                prompt=prompt,
                word_timestamps=word_timestamps,
            )
            # Save transcript.json
            storage_service.save_transcript(job_id, transcript.model_dump())
            return transcript

        # 2. Try Local Faster-Whisper (CUDA / GPU)
        try:
            log.info(f"Attempting local faster-whisper transcription for job {job_id}")
            transcript = local_whisper_service.transcribe(
                audio_path=audio_path,
                job_id=job_id,
                language=language,
                prompt=prompt,
                word_timestamps=word_timestamps,
            )
        except Exception as e:
            error_msg = f"Local faster-whisper failed: {str(e)}"
            log.warning(f"Job {job_id}: {error_msg}")
            error_history.append(error_msg)

            # Check if Groq fallback is viable
            if settings.ENABLE_GROQ_FALLBACK and groq_whisper_service.is_configured:
                log.info(f"Triggering Groq Whisper API fallback for job {job_id}")
                try:
                    transcript = groq_whisper_service.transcribe(
                        audio_path=audio_path,
                        job_id=job_id,
                        language=language,
                        prompt=prompt,
                        word_timestamps=word_timestamps,
                    )
                except Exception as groq_err:
                    groq_error_msg = f"Groq fallback failed: {str(groq_err)}"
                    log.error(f"Job {job_id}: {groq_error_msg}")
                    error_history.append(groq_error_msg)
            else:
                log.info(
                    "Groq fallback is not available (disabled or missing GROQ_API_KEY). "
                    "Cannot fallback to cloud API."
                )

        if transcript is None:
            combined_errors = " | ".join(error_history)
            log.error(f"All transcription engines failed for job {job_id}: {combined_errors}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed across all engines. Details: {combined_errors}",
            )

        # 3. Save transcript.json into /srv/storage/jobs/{job_id}/transcript.json
        storage_service.save_transcript(job_id, transcript.model_dump())
        return transcript


# Singleton instance
transcription_orchestrator = TranscriptionOrchestrator()
