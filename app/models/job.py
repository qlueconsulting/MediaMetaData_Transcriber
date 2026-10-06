"""Pydantic models for jobs and transcription requests."""

from enum import Enum
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field

from app.models.metadata import MediaMetadataResponse
from app.models.transcript import TranscriptData


class JobStatus(str, Enum):
    """Lifecycle statuses of a job."""
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTING_MEDIA = "extracting_media"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class TranscribeRequest(BaseModel):
    """Payload to trigger end-to-end media resolution, audio extraction, and transcription."""
    url: str = Field(
        ...,
        description="Public URL of the media",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )
    bypass_cache: bool = Field(
        default=False,
        description="If True, bypasses any cached audio and transcript, forcing a fresh download and GPU transcription",
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Optional pre-existing job ID to attach to or transcribe existing audio for",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language code ISO-639-1 (e.g. 'en', 'es'). Auto-detected if omitted.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt / vocabulary guidance for Whisper",
    )
    word_timestamps: bool = Field(
        default=False,
        description="Extract word-level timestamps in segments",
    )
    force_engine: Optional[str] = Field(
        default=None,
        description="Override engine: 'cuda', 'groq', or 'cpu'",
    )


class JobTranscribeRequest(BaseModel):
    """Payload to transcribe audio for an already extracted job."""
    language: Optional[str] = Field(
        default=None,
        description="Optional language code ISO-639-1 (e.g. 'en', 'es'). Auto-detected if omitted.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt / vocabulary guidance for Whisper",
    )
    word_timestamps: bool = Field(
        default=False,
        description="Extract word-level timestamps in segments",
    )
    force_engine: Optional[str] = Field(
        default=None,
        description="Override engine: 'cuda', 'groq', or 'cpu'",
    )
    bypass_cache: bool = Field(
        default=False,
        description="If True, bypasses any cached transcript.json and re-runs Whisper inference",
    )


class JobResponse(BaseModel):
    """Standardized response representing a transcription job."""
    job_id: str = Field(..., description="Unique UUID for this job")
    status: JobStatus = Field(..., description="Current status of the job")
    url: str = Field(..., description="Target media URL")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp in UTC",
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="Completion timestamp in UTC",
    )
    execution_time_seconds: Optional[float] = Field(
        default=None,
        description="Total processing time in seconds",
    )
    cached: bool = Field(
        default=False,
        description="True if this response was served from existing persistent storage cache",
    )
    meta: Optional[MediaMetadataResponse] = Field(
        default=None,
        description="Metadata extracted and saved to meta.json",
    )
    audio_size_bytes: Optional[int] = Field(
        default=None,
        description="Extracted audio file size in bytes (guaranteed <= 25MB)",
    )
    audio_file_available: bool = Field(
        default=False,
        description="True if audio.mp3 is successfully saved in persistent storage",
    )
    transcript: Optional[TranscriptData] = Field(
        default=None,
        description="Transcription result saved to transcript.json",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if the job failed or was rejected",
    )


class ExtractAudioResponse(BaseModel):
    """Response returned when extracting audio without immediate transcription."""
    job_id: str = Field(..., description="Unique job ID")
    status: JobStatus = Field(..., description="Job status")
    meta: MediaMetadataResponse = Field(..., description="Extracted video metadata")
    audio_size_bytes: int = Field(..., description="Size of generated audio.mp3 in bytes")
    audio_path: str = Field(..., description="Storage path to audio.mp3")
    cached: bool = Field(
        default=False,
        description="True if audio was served from existing persistent storage cache",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
