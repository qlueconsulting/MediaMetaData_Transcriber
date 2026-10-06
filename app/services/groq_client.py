"""Groq Cloud Whisper API fallback client (whisper-large-v3-turbo)."""

from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, status
from groq import Groq

from app.config import settings
from app.utils.logger import log
from app.models.transcript import TranscriptSegment, TranscriptData, WordTimestamp


class GroqWhisperService:
    """Provides high-throughput cloud Whisper fallback using Groq's whisper-large-v3-turbo."""

    def __init__(self):
        self._client: Optional[Groq] = None

    @property
    def is_configured(self) -> bool:
        """Return True if GROQ_API_KEY is configured."""
        return bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())

    def _get_client(self) -> Groq:
        """Instantiate Groq client with API key."""
        if not self.is_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Groq Whisper fallback is not configured. GROQ_API_KEY is missing.",
            )
        if self._client is None:
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        return self._client

    def transcribe(
        self,
        audio_path: Path,
        job_id: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptData:
        """Send audio to Groq Whisper API and parse verbose JSON response."""
        client = self._get_client()
        model_name = settings.GROQ_MODEL

        log.info(f"Dispatching transcription for job {job_id} to Groq Whisper API ({model_name})")

        if not audio_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audio file not found at {audio_path}",
            )

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        if word_timestamps:
            kwargs["timestamp_granularities"] = ["segment", "word"]

        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    file=(audio_path.name, f),
                    **kwargs,
                )
        except Exception as e:
            log.error(f"Groq Whisper transcription failed for job {job_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Groq Whisper API transcription failed: {str(e)}",
            )

        # Parse response segments
        segments: List[TranscriptSegment] = []
        raw_segments = getattr(response, "segments", []) or []

        for idx, seg in enumerate(raw_segments):
            seg_dict = seg if isinstance(seg, dict) else seg.__dict__
            words = None
            if "words" in seg_dict and seg_dict["words"]:
                words = [
                    WordTimestamp(
                        word=w.get("word") if isinstance(w, dict) else getattr(w, "word", ""),
                        start=w.get("start") if isinstance(w, dict) else getattr(w, "start", 0.0),
                        end=w.get("end") if isinstance(w, dict) else getattr(w, "end", 0.0),
                    )
                    for w in seg_dict["words"]
                ]

            segments.append(
                TranscriptSegment(
                    id=seg_dict.get("id", idx),
                    seek=seg_dict.get("seek"),
                    start=float(seg_dict.get("start", 0.0)),
                    end=float(seg_dict.get("end", 0.0)),
                    text=str(seg_dict.get("text", "")),
                    avg_logprob=seg_dict.get("avg_logprob"),
                    no_speech_prob=seg_dict.get("no_speech_prob"),
                    words=words,
                )
            )

        full_text = getattr(response, "text", "")
        detected_language = getattr(response, "language", language or "en")
        duration = getattr(response, "duration", None)

        log.info(f"Groq transcription completed for job {job_id}: {len(segments)} segments")

        return TranscriptData(
            job_id=job_id,
            text=full_text,
            language=detected_language,
            language_probability=None,
            duration_seconds=float(duration) if duration is not None else None,
            engine="groq-whisper-cloud",
            model=model_name,
            segments=segments,
        )


# Singleton instance
groq_whisper_service = GroqWhisperService()
