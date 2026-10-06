"""Pydantic models for transcription segments and outputs."""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """Word-level alignment timestamp."""
    word: str
    start: float
    end: float
    probability: Optional[float] = None


class TranscriptSegment(BaseModel):
    """Individual transcribed audio segment with timing."""
    id: int = Field(..., description="Segment sequential index")
    seek: Optional[int] = Field(default=None, description="Seek position in milliseconds or samples")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Segment transcription text")
    avg_logprob: Optional[float] = Field(default=None, description="Average log probability score")
    no_speech_prob: Optional[float] = Field(default=None, description="No-speech probability score")
    words: Optional[List[WordTimestamp]] = Field(default=None, description="Optional word-level timestamps")


class TranscriptData(BaseModel):
    """Complete transcript output saved to /srv/storage/jobs/{job_id}/transcript.json."""
    job_id: str = Field(..., description="Unique job identifier")
    text: str = Field(..., description="Full concatenated transcript text")
    language: str = Field(..., description="Detected language code (e.g., 'en', 'es', 'fr')")
    language_probability: Optional[float] = Field(
        default=None,
        description="Confidence score for the detected language",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Audio duration in seconds",
    )
    engine: str = Field(
        ...,
        description="Inference engine utilized (e.g. 'faster-whisper-cuda', 'groq-whisper-large-v3-turbo')",
    )
    model: str = Field(
        ...,
        description="Whisper model name used (e.g. 'large-v3', 'whisper-large-v3-turbo')",
    )
    segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="List of transcribed segments with timestamps",
    )
    completed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of transcription completion",
    )
