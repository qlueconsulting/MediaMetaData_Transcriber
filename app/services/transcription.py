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
        speed_profile: Optional[str] = "adaptive",
        model_name: Optional[str] = None,
        audio_duration: Optional[float] = None,
    ) -> TranscriptData:
        """Transcribe audio with resilient engine fallback and adaptive SLA optimization.

        Preference order:
        1. Explicit override if requested (force_engine='groq' or 'cuda' or 'cpu').
        2. Force Groq flag or cloud profile if enabled.
        3. Adaptive SLA mode: If duration > 30m and Groq is available, route to Cloud Turbo;
           otherwise run local faster-whisper with large-v3-turbo and dynamic batch scaling.
        4. Local faster-whisper with CUDA acceleration.
        5. Groq Whisper Cloud API fallback if CUDA fails or is unavailable.
        6. CPU faster-whisper if Groq is unconfigured.
        """
        transcript: Optional[TranscriptData] = None
        error_history = []

        # 1. Force Groq if explicitly requested or cloud profile
        should_use_groq = (
            force_engine == "groq"
            or speed_profile == "groq"
            or (settings.FORCE_GROQ_FALLBACK and groq_whisper_service.is_configured)
            or (
                speed_profile == "adaptive"
                and audio_duration is not None
                and audio_duration > 1800
                and groq_whisper_service.is_configured
            )
        )

        if should_use_groq:
            if not groq_whisper_service.is_configured:
                if force_engine == "groq" or speed_profile == "groq":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Groq engine was requested, but GROQ_API_KEY is not configured.",
                    )
            else:
                log.info(f"Using Groq Whisper API for job {job_id} (profile={speed_profile})")
                transcript = groq_whisper_service.transcribe(
                    audio_path=audio_path,
                    job_id=job_id,
                    language=language,
                    prompt=prompt,
                    word_timestamps=word_timestamps,
                )
                storage_service.save_transcript(job_id, transcript.model_dump())
                return transcript

        # 2. Try Local Faster-Whisper (CUDA / GPU) with adaptive model selection
        chosen_model = model_name
        if not chosen_model:
            if speed_profile == "standard":
                chosen_model = "large-v3"
            elif speed_profile in ("turbo", "adaptive"):
                chosen_model = settings.WHISPER_MODEL  # defaults to large-v3-turbo

        try:
            log.info(
                f"Attempting local faster-whisper transcription for job {job_id} "
                f"(model={chosen_model}, profile={speed_profile}, duration={audio_duration}s)"
            )
            transcript = local_whisper_service.transcribe(
                audio_path=audio_path,
                job_id=job_id,
                language=language,
                prompt=prompt,
                word_timestamps=word_timestamps,
                model_name=chosen_model,
                audio_duration=audio_duration,
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
                    "Attempting emergency local CPU execution..."
                )
                try:
                    cpu_model = local_whisper_service.load_model(
                        force_reload=True,
                        model_name=chosen_model,
                        device="cpu",
                        compute_type="int8",
                    )
                    gen, info = cpu_model.transcribe(
                        str(audio_path),
                        beam_size=settings.WHISPER_BEAM_SIZE,
                        language=language,
                        initial_prompt=prompt,
                        word_timestamps=word_timestamps,
                        vad_filter=settings.WHISPER_VAD_FILTER,
                    )
                    text_parts = [seg.text for seg in gen]
                    transcript = TranscriptData(
                        job_id=job_id,
                        text="".join(text_parts).strip(),
                        language=getattr(info, "language", None),
                        engine="faster-whisper-cpu-emergency",
                        model=chosen_model or settings.WHISPER_MODEL,
                        segments=[],
                    )
                except Exception as cpu_err:
                    error_history.append(f"Emergency CPU fallback failed: {cpu_err}")

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
